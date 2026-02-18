"""
Shared plotting functions for JAX-ENT example scripts.

All functions accept an optional ``PlotStyle`` dataclass instead of relying on
hardcoded colour/marker dicts.  The ``setup_publication_style()`` call replaces
the ``sns.set_style("ticks"); sns.set_context(...)`` block repeated in ~12 scripts.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PlotStyle


# ---------------------------------------------------------------------------
# Publication style setup
# ---------------------------------------------------------------------------

_DEFAULT_STYLE = PlotStyle(
    ensemble_colors={},
    loss_markers={"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"},
    split_type_colors={
        "r": "fuchsia",
        "s": "black",
        "R3": "green",
        "sequence_cluster": "green",
        "Sp": "grey",
    },
    split_name_mapping={
        "r": "Random",
        "s": "Sequence",
        "R3": "Non-Redundant",
        "sequence_cluster": "Non-Redundant",
        "Sp": "Spatial",
        "spatial": "Spatial",
    },
)


def setup_publication_style() -> None:
    """Apply publication-ready matplotlib/seaborn style.

    Replaces ~12 identical inline blocks across example scripts.
    """
    sns.set_style("ticks")
    sns.set_context(
        "paper",
        rc={
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 10,
        },
    )


def _get_style(style: PlotStyle | None) -> PlotStyle:
    return style if style is not None else _DEFAULT_STYLE


# ---------------------------------------------------------------------------
# Convergence / maxent heatmaps
# ---------------------------------------------------------------------------


def plot_convergence_maxent_heatmaps(
    df: pd.DataFrame,
    convergence_rates: List[float],
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot heatmaps of train/val error across convergence thresholds × maxent values."""
    style = _get_style(style)

    if "maxent_value" not in df.columns:
        print("No maxent_value column found in data")
        return

    df_maxent = df[df["maxent_value"] > 0].copy()
    if df_maxent.empty:
        print("No data with maxent values found")
        return

    split_types = df_maxent["split_type"].unique() if split_type is None else [split_type]

    for stype in split_types:
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)
        split_df = df_maxent[df_maxent["split_type"] == stype] if stype else df_maxent

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        for error_type in ["train_loss", "val_loss"]:
            error_label = "Training Error" if error_type == "train_loss" else "Validation Error"

            fig, axes = plt.subplots(
                len(ensembles),
                len(loss_functions),
                figsize=(8 * len(loss_functions), 6 * len(ensembles)),
                squeeze=False,
            )
            fig.suptitle(
                f"{error_label} Heatmap: Convergence vs MaxEnt{' - ' + stype if stype else ''}",
                fontsize=22,
                fontweight="bold",
            )

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    ax = axes[i, j]
                    combo_df = split_df[
                        (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                    ]

                    if len(combo_df) > 0:
                        pivot_data = combo_df.pivot_table(
                            values=error_type, index="maxent_value", columns="convergence_step", aggfunc="mean",
                        )
                        valid_steps = [s for s in pivot_data.columns if s <= len(convergence_rates)]
                        pivot_data = pivot_data[valid_steps]

                        if not pivot_data.empty:
                            pivot_data = pivot_data.sort_index(ascending=False)
                            col_labels = [
                                f"{convergence_rates[int(s) - 1]:.0e}" if s - 1 < len(convergence_rates) else f"Step {s}"
                                for s in pivot_data.columns
                            ]
                            sns.heatmap(
                                np.log10(pivot_data), annot=False, cmap="viridis",
                                cbar_kws={"label": f"log10({error_label})"}, ax=ax,
                            )
                            ax.set_title(f"{ensemble} - {loss_func}", fontweight="bold")
                            ax.set_xlabel("Convergence Threshold", fontweight="bold")
                            ax.set_ylabel("MaxEnt Value", fontweight="bold")
                            ax.set_xticklabels(col_labels, rotation=45, ha="right")
                            ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                            sns.despine(ax=ax)
                        else:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_title(f"{ensemble} - {loss_func}")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{ensemble} - {loss_func}")

            plt.tight_layout()
            filename = f"{error_type}_convergence_maxent_heatmap_{stype}.png" if stype else f"{error_type}_convergence_maxent_heatmap.png"
            plt.savefig(os.path.join(split_output_dir, filename), dpi=style.dpi, bbox_inches="tight")
            plt.close()


# ---------------------------------------------------------------------------
# Model score heatmaps
# ---------------------------------------------------------------------------


def plot_model_score_heatmaps(
    df: pd.DataFrame,
    convergence_rates: List[float],
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot heatmaps of model scores averaged over split replicates."""
    from .analysis import compute_model_scores

    style = _get_style(style)
    df_scored = compute_model_scores(df)
    df_maxent = df_scored[df_scored["maxent_value"] > 0].copy()

    if df_maxent.empty:
        print("No data with maxent values for scoring")
        return

    split_types = df_maxent["split_type"].unique() if split_type is None else [split_type]

    for stype in split_types:
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)
        split_df = df_maxent[df_maxent["split_type"] == stype] if stype else df_maxent

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles), len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )
        fig.suptitle(
            f"Model Scores: -log10(Val Error){' - ' + stype if stype else ''}",
            fontsize=22, fontweight="bold",
        )

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]
                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]
                if len(combo_df) > 0:
                    pivot_data = combo_df.pivot_table(
                        values="model_score", index="maxent_value",
                        columns="convergence_step", aggfunc="mean",
                    )
                    valid_steps = [s for s in pivot_data.columns if s <= len(convergence_rates)]
                    pivot_data = pivot_data[valid_steps]

                    if not pivot_data.empty:
                        pivot_data = pivot_data.sort_index(ascending=False)
                        col_labels = [
                            f"{convergence_rates[int(s) - 1]:.0e}" if s - 1 < len(convergence_rates) else f"Step {s}"
                            for s in pivot_data.columns
                        ]
                        sns.heatmap(pivot_data, annot=False, cmap="RdYlGn", cbar_kws={"label": "-log10(Val Error)"}, ax=ax)
                        ax.set_title(f"{ensemble} - {loss_func}", fontweight="bold")
                        ax.set_xlabel("Convergence Threshold", fontweight="bold")
                        ax.set_ylabel("MaxEnt Value", fontweight="bold")
                        ax.set_xticklabels(col_labels, rotation=45, ha="right")
                        ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                        sns.despine(ax=ax)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{ensemble} - {loss_func}")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{ensemble} - {loss_func}")

        plt.tight_layout()
        filename = f"model_score_heatmap_{stype}.png" if stype else "model_score_heatmap.png"
        plt.savefig(os.path.join(split_output_dir, filename), dpi=style.dpi, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Best model comparison bar charts
# ---------------------------------------------------------------------------


def plot_best_model_comparisons(
    df: pd.DataFrame,
    output_dir: str,
    style: PlotStyle | None = None,
) -> None:
    """Plot bar charts comparing best models across metrics (val_loss, KL, recovery)."""
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("val_loss", "Validation Loss", True),
        ("kl_divergence", "KL Divergence", True),
        ("recovery_percent", "Recovery %", False),
    ]

    available_metrics = [(col, label, log) for col, label, log in metrics if col in df.columns]
    if not available_metrics:
        print("No metrics available for comparison plots")
        return

    split_types = df["split_type"].unique() if "split_type" in df.columns else [None]
    colors = style.split_type_colors or _DEFAULT_STYLE.split_type_colors

    for col, label, use_log in available_metrics:
        fig, ax = plt.subplots(figsize=style.figsize_wide)
        plot_df = df.copy()
        if use_log:
            plot_df[col] = np.log10(plot_df[col].clip(lower=1e-300))
            label = f"log10({label})"

        sns.barplot(data=plot_df, x="ensemble", y=col, hue="split_type", ax=ax, palette=colors)
        ax.set_ylabel(label, fontweight="bold")
        ax.set_xlabel("Ensemble", fontweight="bold")
        ax.set_title(f"Best Models: {label}", fontweight="bold")
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"best_models_{col}.png"), dpi=style.dpi, bbox_inches="tight")
        plt.close()
