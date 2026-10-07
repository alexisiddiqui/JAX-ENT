"""Checkpoint 37: one pooled alpha per predictor/model across the ATLAS pilot.

The 24-system checkpoint-36 cohort and corrected feature cache are reused. Alpha is
pooled over replica-A pairs. To isolate this one change, local models retain the exact
system-specific k/shrinkage choices selected on replica B in checkpoint 36. Replica C
is evaluated only after the pooled fit is fixed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis import corrected_variance_recovery_checkpoint36 as cp36
from jaxent.examples.ATLAS_BV.analysis.common import HERE, load_config
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint37_global_alpha"
SOURCE = cp36.OUTPUT / "pilot"


def pooled_alphas(fits: pd.DataFrame, alpha_column: str) -> dict[tuple[str, str], float]:
    """Pool no-intercept least-squares scales from per-system sufficient statistics."""
    table = fits.copy()
    if "fit_pairs" not in table:
        table["fit_pairs"] = 1
    table["xx"] = table.fit_pairs * np.square(table.feature_rms)
    table["xy"] = table[alpha_column] * table.xx
    pooled = table.groupby(["metric", "model"], sort=False)[["xy", "xx"]].sum()
    return {key: max(0.0, float(row.xy / row.xx)) for key, row in pooled.iterrows()}


def evaluate_one(args):
    row, config, edges, alphas, local_parameters = args
    return row["system_id"], cp36.evaluate_system(
        row,
        config,
        edges,
        cp36.OUTPUT,
        alpha_overrides=alphas,
        local_parameter_overrides=local_parameters,
    )


def evaluate_cohort(rows, config, edges, alphas, local_parameters, workers):
    collected = {name: [] for name in ("results", "fits", "quantiles", "runtime")}
    payloads = [(row, config, edges, alphas, local_parameters) for row in rows]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_one, item): item[0]["system_id"] for item in payloads}
        for index, future in enumerate(as_completed(futures), 1):
            system, result = future.result()
            for name, records in zip(collected, result):
                records = records if isinstance(records, list) else [records]
                collected[name].extend(records)
            print(f"[{index}/{len(rows)}] {system}", flush=True)
    return {name: pd.DataFrame(records) for name, records in collected.items()}


def alpha_frame(alphas, fits):
    rows = []
    for (metric, model), block in fits.groupby(["metric", "model"], sort=False):
        alpha = alphas[(metric, model)]
        fit_norm = np.sqrt(np.sum(block.fit_pairs * np.square(block.feature_rms)))
        target_norm = np.sqrt(np.sum(block.fit_pairs * np.square(block.target_rms)))
        rows.append(
            {
                "metric": metric,
                "model": model,
                "global_alpha": alpha,
                "normalized_global_alpha": alpha * fit_norm / target_norm,
                "system_alpha_min": block.system_alpha.min(),
                "system_alpha_median": block.system_alpha.median(),
                "system_alpha_max": block.system_alpha.max(),
                "systems": block.system_id.nunique(),
                "fit_pairs": int(block.fit_pairs.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(results):
    rows = []
    for (metric, model, band), block in results.groupby(["metric", "model", "band"]):
        values = block.distribution_recovery.to_numpy()
        low, high = cp36.bootstrap_interval(values, cp36.SEED + sum(map(ord, metric + model + band)))
        rows.append(
            {
                "metric": metric,
                "model": model,
                "band": band,
                "recovery_mean": values.mean(),
                "recovery_median": np.median(values),
                "recovery_sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                "recovery_ci_low": low,
                "recovery_ci_high": high,
                "systems": block.system_id.nunique(),
                "pairs": int(block.pairs.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_alphas(table, destination):
    labels = table.metric + ":" + table.model.replace(
        {"variance_magnitude": "VM", "local_dispersion": "disp"}
    )
    colors = table.model.map(
        {"direct": "#4C78A8", "variance_magnitude": "#B279A2", "local_dispersion": "#F28E2B"}
    )
    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True, constrained_layout=True)
    axes[0].bar(x, table.global_alpha, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Global alpha")
    axes[0].set_title("Raw pooled alpha (replica A)")
    axes[1].bar(x, table.normalized_global_alpha, color=colors)
    axes[1].set_ylabel("Normalized global alpha")
    axes[1].set_title("Scale-normalized pooled alpha")
    axes[1].set_xticks(x, labels, rotation=62, ha="right", fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ATLAS BV checkpoint 37: one shared alpha per predictor/model")
    fig.savefig(destination / "global_alpha_parameters.png", dpi=180)
    plt.close(fig)


def plot_recovery(summary, old_summary, destination):
    families = {
        "Work": (*cp36.CLEAN_WORK, *cp36.DERIVED_WORK),
        "Protection factor": cp36.PF_METRICS,
        "Coordinates": cp36.COORDINATES,
        "Total energy": cp36.ENERGY_METRICS,
    }
    fig, axes = plt.subplots(2, 4, figsize=(25, 10), sharey=True)
    for column, (family, metrics) in enumerate(families.items()):
        for row_index, models in enumerate(
            (("direct",), ("variance_magnitude", "local_dispersion"))
        ):
            axis = axes[row_index, column]
            current = summary[
                summary.metric.isin(metrics) & summary.model.isin(models)
            ]
            previous = old_summary[
                old_summary.metric.isin(metrics) & old_summary.model.isin(models)
            ]
            for metric, points in current.groupby("metric"):
                points = points.assign(
                    order=points.band.str[1:].astype(int)
                ).sort_values("order")
                baseline = previous[previous.metric == metric].assign(
                    order=lambda frame: frame.band.str[1:].astype(int)
                ).sort_values("order")
                line = axis.plot(
                    points.order,
                    100 * points.recovery_mean,
                    marker="o",
                    label=metric,
                )[0]
                axis.plot(
                    baseline.order,
                    100 * baseline.recovery_mean,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.65,
                )
                axis.fill_between(
                    points.order,
                    100 * points.recovery_ci_low,
                    100 * points.recovery_ci_high,
                    color=line.get_color(),
                    alpha=0.10,
                )
            axis.set_title(
                f"{family}: {'direct' if row_index == 0 else 'local variance/dispersion'}"
            )
            axis.set_xticks(range(6), [f"q{i}" for i in range(6)])
            axis.set_xlabel("Global structural-W1 band")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Mean recovery (%)")
    axes[1, 0].set_ylabel("Mean recovery (%)")
    fig.suptitle(
        "ATLAS BV recovery: global alpha (solid) vs per-system alpha (dashed)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(destination / "recovery_vs_w1.png", dpi=180)
    fig.savefig(destination / "recovery_vs_w1.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    config = load_config()
    rows = cp36.selected_rows("pilot", cp36.SINGLE_SYSTEM)
    edge_path = HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
    edges = np.asarray(yaml.safe_load(edge_path.read_text())["edges_angstrom"])
    source_fits = pd.read_parquet(SOURCE / "corrected_variance_fits.parquet")
    source_fits["fit_pairs"] = 10_000
    source_fits["system_alpha"] = source_fits.alpha
    alphas = pooled_alphas(source_fits, "system_alpha")
    local = source_fits[source_fits.k.notna()]
    local_parameters = {
        (row.system_id, row.metric, row.model): (int(row.k), float(row.shrinkage))
        for row in local.itertuples()
    }
    print("global-alpha evaluation with checkpoint-36 local choices", flush=True)
    tables = evaluate_cohort(
        rows, config, edges, alphas, local_parameters, args.workers
    )
    for name, table in tables.items():
        atomic_parquet(table, destination / f"global_alpha_{name}.parquet")
    summary = summarize(tables["results"])
    atomic_parquet(summary, destination / "global_alpha_summary.parquet")
    alpha_table = alpha_frame(alphas, tables["fits"])
    alpha_table.to_csv(destination / "global_alpha_parameters.csv", index=False)
    pd.DataFrame(
        [{"alpha": "pooled on replica A", "local_parameters": "frozen from checkpoint 36 replica-B selection"}]
    ).to_csv(destination / "fit_protocol.csv", index=False)

    old_summary = pd.read_parquet(SOURCE / "corrected_variance_summary.parquet")
    comparison = summary.merge(
        old_summary[["metric", "model", "band", "recovery_mean"]],
        on=["metric", "model", "band"],
        suffixes=("_global", "_system"),
    )
    comparison["recovery_delta"] = (
        comparison.recovery_mean_global - comparison.recovery_mean_system
    )
    comparison.to_csv(destination / "recovery_vs_system_alpha.csv", index=False)
    plot_alphas(alpha_table, destination)
    plot_recovery(summary, old_summary, destination)

    report = {
        "checkpoint": 37,
        "systems": len(rows),
        "alpha_scope": "one pooled replica-A alpha per metric/model across all systems",
        "local_parameter_scope": "system-specific checkpoint-36 k/shrinkage held fixed",
        "evaluation": "replica C only",
        "global_alpha_parameters": alpha_table.to_dict(orient="records"),
    }
    (destination / "checkpoint37_report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"report: {destination / 'checkpoint37_report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
