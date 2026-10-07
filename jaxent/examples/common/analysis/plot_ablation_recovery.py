"""Create recovery tables and heatmaps from ablation selected-model rows."""

from __future__ import annotations

import argparse
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
    return f"E{int(row['example'])} | {row['ensemble']} | {row['loss_function']}"


def _summarise(selected: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    selected["recovery_percent"] = pd.to_numeric(
        selected["recovery_percent"], errors="coerce"
    )
    selected = selected[np.isfinite(selected["recovery_percent"])].copy()
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
                ["cell", "cell_code", "cell_order", "setting", "example", "ensemble", "loss_function", "experiment"],
                as_index=False,
                sort=False,
            )["recovery_percent"]
            .agg(recovery_mean="mean", recovery_sd="std", n="count")
        )
        summary["split_scope"] = scope
        summary["expected_n"] = 6 if scope == "pooled" else 3
        summary["complete"] = summary["n"] == summary["expected_n"]
        outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def _format_value(row: pd.Series) -> str:
    sd = "NA" if not np.isfinite(row["recovery_sd"]) else f"{row['recovery_sd']:.1f}"
    marker = "" if row["complete"] else "*"
    return f"{row['recovery_mean']:.1f}\n± {sd}\nn={int(row['n'])}{marker}"


def _plot_heatmap(
    summary: pd.DataFrame,
    *,
    index: str,
    columns: str,
    title: str,
    output_path: Path,
) -> None:
    means = summary.pivot(index=index, columns=columns, values="recovery_mean")
    annotations = summary.assign(label=summary.apply(_format_value, axis=1)).pivot(
        index=index, columns=columns, values="label"
    )

    if index == "setting":
        setting_order = (
            summary[["setting", "cell_order"]]
            .drop_duplicates()
            .sort_values("cell_order")["setting"]
            .tolist()
        )
        means = means.reindex(setting_order)
        annotations = annotations.reindex(setting_order)
    else:
        experiment_order = (
            summary[["experiment", "example", "ensemble", "loss_function"]]
            .drop_duplicates()
            .sort_values(["example", "ensemble", "loss_function"])["experiment"]
            .tolist()
        )
        means = means.reindex(experiment_order)
        annotations = annotations.reindex(experiment_order)

    fig_width = max(12.0, 1.25 * len(means.columns) + 4.0)
    fig_height = max(6.0, 0.72 * len(means.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(
        means,
        annot=annotations.fillna("").to_numpy(),
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Recovery (%)"},
        mask=means.isna(),
        annot_kws={"fontsize": 7.5},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(columns.replace("_", " ").title())
    ax.set_ylabel(index.replace("_", " ").title())
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


def generate(input_csv: Path, output_dir: Path) -> None:
    selected = pd.read_csv(input_csv)
    required = {
        "cell", "example", "ensemble", "loss_function", "split_type",
        "recovery_percent", "learning_rate", "lr_adjustment", "frame_average_impl",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected-model CSV is missing columns: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarise(selected)
    summary.sort_values(
        ["split_scope", "cell_order", "example", "ensemble", "loss_function"]
    ).to_csv(output_dir / "recovery_summary_long.csv", index=False)

    for scope in SPLIT_SCOPES:
        scoped = summary[summary["split_scope"] == scope].copy()
        formatted = scoped.assign(value=scoped.apply(_format_value, axis=1))

        by_experiment = formatted.pivot(
            index="experiment", columns="setting", values="value"
        )
        by_setting = formatted.pivot(
            index="setting", columns="experiment", values="value"
        )
        by_experiment.to_csv(output_dir / f"recovery_by_experiment_{scope}.csv")
        by_setting.to_csv(output_dir / f"recovery_by_setting_{scope}.csv")

        _plot_heatmap(
            scoped,
            index="experiment",
            columns="setting",
            title=f"Recovery by experiment ({scope.replace('_', ' ')})",
            output_path=output_dir / f"recovery_by_experiment_{scope}.png",
        )
        _plot_heatmap(
            scoped,
            index="setting",
            columns="experiment",
            title=f"Recovery by optimizer setting ({scope.replace('_', ' ')})",
            output_path=output_dir / f"recovery_by_setting_{scope}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.selected_models, args.output_dir)


if __name__ == "__main__":
    main()
