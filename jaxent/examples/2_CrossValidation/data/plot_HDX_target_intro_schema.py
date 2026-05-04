#!/usr/bin/env python3
"""Generate the CrossValidation intro state-population figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
ANALYSIS_DIR = EXP_DIR / "analysis"
PLOTS_DIR = SCRIPT_DIR / "plots"

STATE_RATIOS_PATH = ANALYSIS_DIR / "state_ratios.json"
CLUSTER_DIR = ANALYSIS_DIR / "_MoPrP_analysis_clusters_feature_spec_AF2_test" / "clusters"

TARGET_CLUSTER_FILE = CLUSTER_DIR / "AF2_Filtered" / "AF2_Filtered_frame_to_cluster.csv"
INTERMEDIATE_CLUSTER_FILE = CLUSTER_DIR / "AF2_MSAss" / "AF2_MSAss_frame_to_cluster.csv"

STATE_ORDER = ["Folded", "PUF1", "PUF2", "Unfolded"]
DISPLAY_NAMES = {
    "target": "Target",
    "ground_truth_only": "Ground truth-only",
    "ground_truth_intermediate": "Ground truth + intermediate",
}
COLOURS = {
    "target": "#F28E2B",
    "ground_truth_only": "#000000",
    "ground_truth_intermediate": "#7A7A7A",
}
STATE_LABELS = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 9,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_target_populations(path: Path) -> dict[str, float]:
    with path.open() as handle:
        data = json.load(handle)

    pops = data["fractional_populations"]
    return {
        "Folded": float(pops["folded"]["fraction"]) * 100.0,
        "PUF1": float(pops["PUF1"]["fraction"]) * 100.0,
        "PUF2": float(pops["PUF2"]["fraction"]) * 100.0,
        "Unfolded": float(pops.get("unfolded", {}).get("fraction", 0.0)) * 100.0,
    }


def load_cluster_populations(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "cluster_label" not in df.columns:
        raise ValueError(f"cluster_label column not found in {path}")

    labels = df["cluster_label"].to_numpy()
    if len(labels) == 0:
        raise ValueError(f"No cluster assignments found in {path}")

    counts = {state: 0 for state in STATE_ORDER}
    mapped = 0
    for label in labels:
        state = STATE_LABELS.get(int(label), "Unfolded")
        if state in counts:
            counts[state] += 1
            mapped += 1

    if mapped == 0:
        raise ValueError(f"No mapped cluster assignments found in {path}")

    return {state: (counts[state] / mapped) * 100.0 for state in STATE_ORDER}


def build_state_matrix() -> tuple[list[str], list[str], np.ndarray]:
    conditions = [
        ("target", load_target_populations(STATE_RATIOS_PATH)),
        ("ground_truth_only", load_cluster_populations(TARGET_CLUSTER_FILE)),
        ("ground_truth_intermediate", load_cluster_populations(INTERMEDIATE_CLUSTER_FILE)),
    ]

    matrix = np.array([[values[state] for state in STATE_ORDER] for _, values in conditions], dtype=float)
    condition_labels = [DISPLAY_NAMES[key] for key, _ in conditions]
    return condition_labels, STATE_ORDER, matrix


def annotate_bars(ax: plt.Axes, bars: list[plt.Rectangle]) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_grouped_bars() -> None:
    _, states, matrix = build_state_matrix()

    x = np.arange(len(states))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6, 4))

    bars_by_condition = []
    for idx, (condition_key, condition_label) in enumerate(DISPLAY_NAMES.items()):
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            matrix[idx],
            width=width,
            label=condition_label,
            color=COLOURS[condition_key],
            edgecolor="white",
            linewidth=0.8,
        )
        bars_by_condition.extend(bars)

    annotate_bars(ax, bars_by_condition)

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Population (%)")
    ax.set_ylim(0, 105)
    ax.set_title("MoPrP Cross-Validation State Ratios")
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, ncols=1)
    # ax.text(
    #     0.99,
    #     -0.18,
    #     "Cluster labels outside 0-2 are accumulated as Unfolded.",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=8,
    # )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = PLOTS_DIR / "HDX_target_intro_schema"
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_style()
    ensure_exists(STATE_RATIOS_PATH)
    ensure_exists(TARGET_CLUSTER_FILE)
    ensure_exists(INTERMEDIATE_CLUSTER_FILE)
    plot_grouped_bars()


if __name__ == "__main__":
    main()
