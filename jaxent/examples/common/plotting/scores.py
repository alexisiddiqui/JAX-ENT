from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PlotStyle
from .style import _get_style

def plot_model_score_heatmaps(
    df: pd.DataFrame,
    convergence_rates: List[float],
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot heatmaps of model scores averaged over split replicates."""
    from ..analysis.scoring import compute_model_scores

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


def create_violin_plots(
    scores_df: pd.DataFrame,
    output_dir: str,
    metric_columns: List[str] | None = None,
) -> None:
    """Create violin plots for score metrics, faceted by ensemble × loss function.

    Parameters
    ----------
    scores_df:
        DataFrame with columns including ``ensemble``, ``loss_function``,
        ``convergence_value``, ``split_type`` and the metric columns to plot.
    output_dir:
        Directory to save ``violin_<metric>.png`` files.
    metric_columns:
        Columns to plot.  When ``None``, a default set of common score columns
        is used, filtered to those present in *scores_df*.
    """
    os.makedirs(output_dir, exist_ok=True)

    if metric_columns is None:
        metric_columns = [
            "train_mse", "val_mse", "test_mse",
            "d_mse_train", "d_mse_val", "d_mse_test",
            "work_scale", "work_shape", "work_density", "work_fitting",
            "kl_divergence", "recovery_percent", "val_loss",
        ]

    available_scores = [col for col in metric_columns if col in scores_df.columns]

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)

    for score_col in available_scores:
        if scores_df[score_col].isna().all():
            print(f"  Skipping {score_col} (all NaN)")
            continue

        ensembles = sorted(scores_df["ensemble"].unique())
        loss_functions = sorted(scores_df["loss_function"].unique())
        n_ensembles = len(ensembles)
        n_losses = len(loss_functions)

        fig, axes = plt.subplots(n_ensembles, n_losses, figsize=(5 * n_losses, 5 * n_ensembles))

        if n_ensembles == 1 and n_losses == 1:
            axes = np.array([[axes]])
        elif n_ensembles == 1:
            axes = axes.reshape(1, -1)
        elif n_losses == 1:
            axes = axes.reshape(-1, 1)

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]
                subset = scores_df[
                    (scores_df["ensemble"] == ensemble)
                    & (scores_df["loss_function"] == loss_func)
                ].copy()

                if subset.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                if "convergence_value" in subset.columns:
                    subset["convergence_rounded"] = subset["convergence_value"].round(4)
                    x_col = "convergence_rounded"
                else:
                    x_col = subset.columns[0]

                sns.violinplot(
                    data=subset,
                    x=x_col,
                    y=score_col,
                    hue="split_type" if "split_type" in subset.columns else None,
                    ax=ax,
                    palette="Set2",
                )

                ax.set_title(f"{ensemble} - {loss_func}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Convergence Value", fontsize=10)
                ax.set_ylabel(score_col, fontsize=10)
                ax.tick_params(axis="x", rotation=45)

                if "split_type" in subset.columns and len(subset["split_type"].unique()) > 1:
                    ax.legend(title="Split Type", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                elif ax.get_legend():
                    ax.get_legend().remove()

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"violin_{score_col}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close()
