"""Checkpoint 9 population report for the strict Gaussian likelihood baseline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


PRIMARY_METRICS = (
    "distribution_recovery", "distribution_l1", "distribution_l2",
    "distribution_sqrt_jsd", "distribution_kld_target_to_prediction",
    "distribution_cosine_distance", "distribution_correlation_distance",
    "coverage_90", "mean_interval_score", "median_interval_width", "effective_frames",
)


def system_population(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["system_id", "target", "model", "calibration", "band", "stratum"]
    system = summary.groupby(keys, as_index=False).agg(**{
        metric: (metric, "mean") for metric in PRIMARY_METRICS
    })
    population = system.groupby(keys[1:], as_index=False).median(numeric_only=True)
    return system, population


def paired_preprocessing(system: pd.DataFrame) -> pd.DataFrame:
    primary = system[
        (system.stratum == "common_support")
        & (((system.target == "w1") & (system.band == "q5"))
           | ((system.target == "rmsd") & (system.band == "global")))
    ]
    keys = ["system_id", "target", "calibration", "band", "stratum"]
    raw = primary[primary.model == "raw_logpf_ridge_gaussian"]
    zscore = primary[primary.model == "zscore_logpf_ridge_gaussian"]
    merged = zscore.merge(raw, on=keys, suffixes=("_zscore", "_raw"), validate="one_to_one")
    rows = []
    for (target, calibration, band), block in merged.groupby(["target", "calibration", "band"]):
        for metric, multiplier, direction in (
            ("distribution_recovery", 100.0, 1.0),
            ("coverage_90", 100.0, 1.0),
            ("mean_interval_score", 1.0, -1.0),
        ):
            delta = multiplier * direction * (
                block[f"{metric}_zscore"].to_numpy() - block[f"{metric}_raw"].to_numpy()
            )
            low, high = bootstrap_median_ci(delta, 10000, 900 + len(rows))
            rows.append({
                "target": target, "band": band, "calibration": calibration,
                "metric": metric, "positive_means_zscore_better": True,
                "systems": len(delta), "median_zscore_effect": float(np.median(delta)),
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "systems_zscore_better_percent": float(100 * np.mean(delta > 0)),
            })
    return pd.DataFrame(rows)


def primary_intervals(system: pd.DataFrame) -> list[dict]:
    selected = system[
        (system.target == "w1") & (system.band == "q5")
        & (system.stratum == "common_support")
    ]
    rows = []
    for (model, calibration), block in selected.groupby(["model", "calibration"]):
        row = {"model": model, "calibration": calibration, "systems": len(block)}
        for index, (metric, multiplier) in enumerate((
            ("distribution_recovery", 100.0), ("coverage_90", 100.0),
            ("mean_interval_score", 1.0),
        )):
            values = multiplier * block[metric].to_numpy()
            low, high = bootstrap_median_ci(values, 10000, 950 + 10 * len(rows) + index)
            row[f"median_{metric}"] = float(np.median(values))
            row[f"{metric}_ci95_low"] = low; row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def plot_fit(population: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, target in zip(axes, ("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for model, label, marker in (
            ("raw_logpf_ridge_gaussian", "Raw log-PF", "o"),
            ("zscore_logpf_ridge_gaussian", "A-only z-score", "s"),
        ):
            block = population[
                (population.target == target) & (population.model == model)
                & (population.calibration == "mondrian") & (population.stratum == "common_support")
            ].set_index("band")
            axis.plot(x, [100 * block.distribution_recovery.get(b, np.nan) for b in order],
                      marker=marker, label=label)
        axis.set_title(target.upper()); axis.set_xticks(x, order); axis.grid(alpha=.25)
    axes[0].set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].legend(); fig.suptitle("Strict conditional distribution fit (common support)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_calibration(population: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, target in zip(axes, ("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for model, label, marker in (
            ("raw_logpf_ridge_gaussian", "Raw log-PF", "o"),
            ("zscore_logpf_ridge_gaussian", "A-only z-score", "s"),
        ):
            for calibration, linestyle in (("marginal", "--"), ("mondrian", "-")):
                block = population[
                    (population.target == target) & (population.model == model)
                    & (population.calibration == calibration) & (population.stratum == "common_support")
                ].set_index("band")
                axis.plot(x, [100 * block.coverage_90.get(b, np.nan) for b in order], marker=marker,
                          linestyle=linestyle, label=f"{label} / {calibration}")
        axis.axhline(90, color="black", linestyle=":"); axis.set_title(target.upper())
        axis.set_xticks(x, order); axis.grid(alpha=.25)
    axes[0].set_ylabel("Empirical 90% interval coverage (%)")
    axes[1].legend(fontsize=7); fig.suptitle("Strict likelihood calibration (common support)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint9_strict_likelihood"
    summary = pd.read_parquet(output / "strict_likelihood_assignment_summary.parquet")
    system, population = system_population(summary)
    effects = paired_preprocessing(system)
    system.to_parquet(output / "strict_likelihood_system_summary.parquet", index=False)
    population.to_parquet(output / "strict_likelihood_population.parquet", index=False)
    effects.to_parquet(output / "strict_likelihood_paired_preprocessing.parquet", index=False)
    plot_fit(population, output / "strict_likelihood_distribution_fit.png")
    plot_calibration(population, output / "strict_likelihood_calibration.png")
    selected = population[
        (population.target == "w1") & (population.band == "q5")
        & (population.stratum == "common_support") & (population.calibration == "mondrian")
    ]
    report = {
        "checkpoint": "9", "status": "complete", "systems": int(system.system_id.nunique()),
        "decision": "strict_baseline_established_stage_2_requires_review",
        "primary_w1_q5_common_support": selected.to_dict(orient="records"),
        "primary_system_bootstrap": primary_intervals(system),
        "paired_zscore_minus_raw": effects.to_dict(orient="records"),
        "metric_contract": {
            "fit_error": "dimensionless probability-mass error",
            "recovery": "100 * (1 - sqrt(JSD))",
            "coverage": "separate empirical coverage of structural-target intervals",
        },
        "settings": load_config()["analysis"]["pairwise_geometry"]["strict_conformal"],
    }
    atomic_yaml(output / "checkpoint9_report.yaml", report)
    print(selected[["model", "distribution_recovery", "distribution_l1", "distribution_l2",
                    "distribution_sqrt_jsd", "coverage_90", "mean_interval_score",
                    "effective_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
