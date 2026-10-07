"""Compute selected-checkpoint ESS and plot the MSE-only ablation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CELL_ORDER = ["A0", "A1", "A2", "A3", "A4", "B1", "B2", "B3"]
SPLIT_SCOPES = ("pooled", "sequence_cluster", "spatial")


def _cell_code(value: str) -> str:
    return value.lstrip("_").split("_", 1)[0]


def _setting_label(row: pd.Series) -> str:
    reduction = "td" if row["frame_average_impl"] == "tensordot" else "legacy"
    return (
        f"{row['cell_code']}\nLR {row['learning_rate']:g} / "
        f"{row['lr_adjustment']} / {reduction}"
    )


def _experiment_label(row: pd.Series) -> str:
    return f"E{int(row['example'])} | {row['ensemble']} | MSE"


def _processed_root(row: pd.Series) -> Path:
    manifest_path = Path(row["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    explicit = manifest.get("output_paths", {}).get("processed")
    if explicit:
        return Path(explicit)
    results_root = manifest_path.parent
    return results_root.parent / f"_processed_{results_root.name}"


def _run_prefix(row: pd.Series) -> str:
    prefix = (
        f"{row['ensemble']}_{row['loss_function']}_{row['split_type']}_"
        f"split{int(row['split_idx']):03d}_maxent{float(row['maxent_value']):.1f}"
    )
    if int(row["example"]) == 3:
        prefix += (
            f"_bvreg{float(row['bv_reg_value']):.2f}"
            f"_bvregfn{row['bv_reg_function']}"
        )
    return prefix


def _selected_ess(row: pd.Series) -> tuple[float, int, str]:
    root = _processed_root(row)
    split_root = root / str(row["split_type"])
    prefix = _run_prefix(row)
    candidates = sorted(path for path in split_root.glob(f"{prefix}*") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one processed run for {prefix!r} in {split_root}, found {len(candidates)}"
        )
    run_dir = candidates[0]
    thresholds = np.loadtxt(run_dir / "convergence_thresholds.txt", ndmin=1)
    weights = np.load(run_dir / "frame_weights.npy")
    if len(thresholds) != len(weights):
        raise ValueError(f"Threshold/weight length mismatch in {run_dir}")
    matches = np.flatnonzero(
        np.isclose(
            thresholds,
            float(row["convergence_value"]),
            rtol=1e-5,
            atol=1e-12,
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one threshold match for {row['convergence_value']} in {run_dir}, "
            f"found {len(matches)}"
        )
    selected = np.asarray(weights[int(matches[0])], dtype=float)
    total = np.sum(selected)
    if not np.isfinite(selected).all() or not np.isfinite(total) or total <= 0:
        return np.nan, int(selected.size), str(run_dir)
    selected = selected / total
    ess = 1.0 / np.sum(selected**2)
    return float(ess), int(selected.size), str(run_dir)


def add_ess(selected: pd.DataFrame) -> pd.DataFrame:
    mse = selected[selected["loss_function"].eq("MSE")].copy()
    values = mse.apply(_selected_ess, axis=1)
    mse[["effective_sample_size", "num_frames", "processed_run_dir"]] = pd.DataFrame(
        values.tolist(), index=mse.index
    )
    mse["relative_ess_percent"] = (
        100.0 * mse["effective_sample_size"] / mse["num_frames"]
    )
    return mse


def summarise(selected: pd.DataFrame) -> pd.DataFrame:
    selected = selected[np.isfinite(selected["effective_sample_size"])].copy()
    selected["cell_code"] = selected["cell"].map(_cell_code)
    order = {cell: index for index, cell in enumerate(CELL_ORDER)}
    selected["cell_order"] = selected["cell_code"].map(order)
    selected["setting"] = selected.apply(_setting_label, axis=1)
    selected["experiment"] = selected.apply(_experiment_label, axis=1)

    outputs: list[pd.DataFrame] = []
    for scope in SPLIT_SCOPES:
        scoped = selected if scope == "pooled" else selected[selected["split_type"] == scope]
        summary = (
            scoped.groupby(
                ["cell", "cell_code", "cell_order", "setting", "example", "ensemble", "experiment"],
                as_index=False,
                sort=False,
            )["effective_sample_size"]
            .agg(ess_mean="mean", ess_sd="std", n="count")
        )
        summary["split_scope"] = scope
        summary["expected_n"] = 6 if scope == "pooled" else 3
        summary["complete"] = summary["n"] == summary["expected_n"]
        outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def _format_value(row: pd.Series) -> str:
    sd = "NA" if not np.isfinite(row["ess_sd"]) else f"{row['ess_sd']:.1f}"
    marker = "" if row["complete"] else "*"
    return f"{row['ess_mean']:.1f}\n± {sd}\nn={int(row['n'])}{marker}"


def _plot(summary: pd.DataFrame, scope: str, output_path: Path) -> None:
    means = summary.pivot(index="setting", columns="experiment", values="ess_mean")
    annotations = summary.assign(label=summary.apply(_format_value, axis=1)).pivot(
        index="setting", columns="experiment", values="label"
    )
    setting_order = (
        summary[["setting", "cell_order"]]
        .drop_duplicates()
        .sort_values("cell_order")["setting"]
        .tolist()
    )
    experiment_order = (
        summary[["experiment", "example", "ensemble"]]
        .drop_duplicates()
        .sort_values(["example", "ensemble"])["experiment"]
        .tolist()
    )
    means = means.reindex(index=setting_order, columns=experiment_order)
    annotations = annotations.reindex(index=setting_order, columns=experiment_order)

    fig, ax = plt.subplots(
        figsize=(max(12.0, 1.7 * len(means.columns) + 4.0), 9.0),
        constrained_layout=True,
    )
    sns.heatmap(
        means,
        annot=annotations.fillna("").to_numpy(),
        fmt="",
        cmap="mako",
        linewidths=0.5,
        linecolor="white",
        mask=means.isna(),
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Effective sample size"},
        ax=ax,
    )
    ax.set_title(f"Selected-model effective sample size ({scope.replace('_', ' ')})")
    ax.set_xlabel("Experiment / ensemble / loss")
    ax.set_ylabel("Optimizer setting")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.text(
        0.01,
        0.005,
        "Cells show mean ± sample SD and n; * marks incomplete split coverage.",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate(selected_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = add_ess(pd.read_csv(selected_path))
    selected.to_csv(output_dir / "mse_selected_models_with_ess.csv", index=False)
    summary = summarise(selected)
    summary.sort_values(
        ["split_scope", "cell_order", "example", "ensemble"]
    ).to_csv(output_dir / "ess_summary_long.csv", index=False)

    for scope in SPLIT_SCOPES:
        scoped = summary[summary["split_scope"].eq(scope)].copy()
        formatted = scoped.assign(value=scoped.apply(_format_value, axis=1))
        formatted.pivot(index="setting", columns="experiment", values="value").to_csv(
            output_dir / f"ess_by_setting_{scope}.csv"
        )
        _plot(scoped, scope, output_dir / f"ess_by_setting_{scope}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.selected_models, args.output_dir)


if __name__ == "__main__":
    main()
