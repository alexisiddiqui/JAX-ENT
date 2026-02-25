from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PlotStyle
from .style import _get_style

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


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    convergence_rates: List[float],
    output_dir: str,
    *,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    title_prefix: str = "",
    index_col: str = "maxent_value",
    columns_col: str = "convergence_step",
    style: PlotStyle | None = None,
) -> None:
    """Plot per-ensemble/loss heatmap of *metric* vs convergence step × maxent value.

    Generalises ``plot_recovery_heatmap``, ``plot_ess_heatmaps``,
    ``plot_kld_uniform_heatmaps``, ``plot_kld_between_splits_heatmap`` and
    similar functions that share the maxent × convergence_step pivot structure.

    Parameters
    ----------
    df:
        DataFrame containing at minimum ``ensemble``, ``loss_function``,
        ``split_type``, *index_col*, *columns_col*, and *metric* columns.
    metric:
        Column name of the metric to plot.
    convergence_rates:
        List of convergence rate values (used to label x-axis ticks).
    output_dir:
        Base directory; per-split-type subdirectories are created automatically.
    cmap, vmin, vmax:
        Colormap and colour range overrides.
    title_prefix:
        Optional prefix prepended to the figure suptitle.
    index_col, columns_col:
        Column names used for the pivot table index / columns.
    """
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    if metric not in df.columns:
        print(f"Metric '{metric}' not found in DataFrame columns")
        return

    split_types: list = (
        df["split_type"].unique().tolist() if "split_type" in df.columns else [None]
    )

    for stype in split_types:
        stype_safe = stype.replace("/", "_") if stype else stype
        split_output_dir = os.path.join(output_dir, stype_safe) if stype_safe else output_dir
        os.makedirs(split_output_dir, exist_ok=True)
        split_df = df[df["split_type"] == stype].copy() if stype else df.copy()

        ensembles: list = (
            sorted(split_df["ensemble"].unique()) if "ensemble" in split_df.columns else [None]
        )
        loss_functions: list = (
            sorted(split_df["loss_function"].unique())
            if "loss_function" in split_df.columns
            else [None]
        )

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )
        title = f"{title_prefix}{metric}" + (f" — {stype}" if stype else "")
        fig.suptitle(title, fontsize=22, fontweight="bold")

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]
                combo_df = split_df.copy()
                if ensemble is not None:
                    combo_df = combo_df[combo_df["ensemble"] == ensemble]
                if loss_func is not None:
                    combo_df = combo_df[combo_df["loss_function"] == loss_func]

                if (
                    len(combo_df) > 0
                    and index_col in combo_df.columns
                    and columns_col in combo_df.columns
                ):
                    pivot_data = combo_df.pivot_table(
                        values=metric,
                        index=index_col,
                        columns=columns_col,
                        aggfunc="mean",
                    )
                    if not pivot_data.empty:
                        pivot_data = pivot_data.sort_index(ascending=False)
                        col_labels = []
                        for s in pivot_data.columns:
                            try:
                                col_labels.append(f"{convergence_rates[int(s) - 1]:.0e}")
                            except (IndexError, ValueError, TypeError):
                                col_labels.append(str(s))

                        _vmin = vmin if vmin is not None else pivot_data.min().min()
                        _vmax = vmax if vmax is not None else pivot_data.max().max()

                        sns.heatmap(
                            pivot_data,
                            annot=False,
                            cmap=cmap,
                            vmin=_vmin,
                            vmax=_vmax,
                            cbar_kws={"label": metric},
                            ax=ax,
                        )
                        label_parts = [p for p in [ensemble, loss_func] if p]
                        ax.set_title(" — ".join(label_parts), fontweight="bold")
                        ax.set_xlabel(columns_col.replace("_", " ").title(), fontweight="bold")
                        ax.set_ylabel(index_col.replace("_", " ").title(), fontweight="bold")
                        ax.set_xticklabels(col_labels, rotation=45, ha="right")
                        sns.despine(ax=ax)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{ensemble or ''} — {loss_func or ''}")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{ensemble or ''} — {loss_func or ''}")

        plt.tight_layout()
        filename = f"{metric}_heatmap{'_' + stype_safe if stype_safe else ''}.png"
        plt.savefig(os.path.join(split_output_dir, filename), dpi=style.dpi, bbox_inches="tight")
        plt.close()
