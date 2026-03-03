from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

def plot_2d_heatmaps_grid(
    df: pd.DataFrame,
    output_dir: str,
    metric: str = "effective_sample_size",
    metric_label: str = "Effective Sample Size",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot 2D heatmaps of a weight metric across the (maxent, bv_reg) sweep.

    Grid layout: rows = BV regularisation functions, columns = loss functions.
    One figure per ensemble per split type is produced.

    Parameters
    ----------
    df:
        DataFrame produced by ``analysis.extract_final_weights_2d`` or
        ``compute_pairwise_kld_between_splits_2d``.  Must contain columns
        ``split_type``, ``ensemble``, ``loss_function``, ``bv_reg_function``,
        ``bv_reg_value``, ``maxent_value``, and *metric*.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    if df.empty:
        print("No data available for 2D heatmaps")
        return

    for split_type in sorted(df["split_type"].unique()):
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        split_df = df[df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )
            fig.suptitle(
                f"2D Analysis — {ensemble} — {split_type}\n{metric_label}",
                fontsize=16, fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_fn in enumerate(loss_functions):
                    ax = axes[i, j]
                    combo_df = ensemble_df[
                        (ensemble_df["loss_function"] == loss_fn)
                        & (ensemble_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        pivot_data = combo_df.pivot_table(
                            values=metric,
                            index="bv_reg_value",
                            columns="maxent_value",
                            aggfunc="mean",
                        )
                        if not pivot_data.empty:
                            pivot_data = pivot_data.sort_index(ascending=False).sort_index(axis=1)

                            if cmap is not None:
                                _cmap = cmap
                            elif metric == "effective_sample_size":
                                _cmap = "viridis"
                            elif metric == "kl_divergence":
                                _cmap = "YlOrRd"
                            else:
                                _cmap = "viridis"

                            _vmin = vmin if vmin is not None else 0
                            _vmax = vmax if vmax is not None else combo_df[metric].max()

                            sns.heatmap(
                                pivot_data, annot=True, fmt=".2f",
                                cmap=_cmap, vmin=_vmin, vmax=_vmax,
                                cbar_kws={"label": metric_label}, ax=ax,
                                cbar=(j == len(loss_functions) - 1),
                            )
                            ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                            ax.set_xlabel("MaxEnt Value")
                            ax.set_ylabel("BV Reg Value")
                            try:
                                xlabels = [f"{float(t.get_text()):.0f}" for t in ax.get_xticklabels()]
                                ax.set_xticklabels(xlabels, rotation=45, ha="right")
                            except (ValueError, AttributeError):
                                pass
                        else:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, f"2d_heatmap_{ensemble}_{metric}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)


def plot_1d_slices_2d_sweep(
    df: pd.DataFrame,
    output_dir: str,
    metric: str = "effective_sample_size",
    metric_label: str = "Effective Sample Size",
) -> None:
    """Plot metric vs maxent (for fixed bv_reg values) as 1D line plots.

    One figure per ensemble per split type is produced, with a grid of
    subplots matching (bv_reg_function rows) × (loss_function columns).

    Parameters
    ----------
    df:
        Same DataFrame structure as accepted by :func:`plot_2d_heatmaps_grid`.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    if df.empty:
        print("No data available for 1D slices")
        return

    for split_type in sorted(df["split_type"].unique()):
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        split_df = df[df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )
            fig.suptitle(
                f"1D Slices: {metric_label} vs MaxEnt — {ensemble} — {split_type}",
                fontsize=16, fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_fn in enumerate(loss_functions):
                    ax = axes[i, j]
                    combo_df = ensemble_df[
                        (ensemble_df["loss_function"] == loss_fn)
                        & (ensemble_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        bv_reg_values = sorted(combo_df["bv_reg_value"].unique())
                        colors = plt.cm.viridis(np.linspace(0, 1, len(bv_reg_values)))

                        for k, bv_reg_val in enumerate(bv_reg_values):
                            bv_subset = combo_df[combo_df["bv_reg_value"] == bv_reg_val]
                            avg_data = bv_subset.groupby("maxent_value")[metric].agg(["mean", "std"]).reset_index()

                            ax.errorbar(
                                avg_data["maxent_value"],
                                avg_data["mean"],
                                yerr=avg_data["std"],
                                marker="o",
                                label=f"BV Reg={bv_reg_val:.2f}",
                                color=colors[k],
                                capsize=5,
                                capthick=2,
                                linewidth=2,
                                markersize=6,
                            )

                        ax.set_xscale("log")
                        ax.set_xlabel("MaxEnt Value (log scale)")
                        ax.set_ylabel(metric_label)
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=8, loc="best")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, f"1d_slice_maxent_{ensemble}_{metric}.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)



def plot_best_hyperparameters(
    recovery_df: pd.DataFrame,
    output_dir: str,
    metric: str = "recovery_percent",
):
    """Plot a summary showing the best (maxent, bv_reg) combination for each loss-reg pairing."""
    plt.style.use("seaborn-v0_8-whitegrid")

    df = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(df) == 0:
        print("No data available for best hyperparameters plot")
        return

    split_types = sorted(df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating best hyperparameters summary for {split_type}...")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df[df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            loss_functions = sorted(ensemble_df["loss_function"].unique())
            bv_reg_functions = sorted(ensemble_df["bv_reg_function"].unique())

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"Best Hyperparameters by {metric} - {ensemble} - {split_type}",
                fontsize=16,
                fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_fn in enumerate(loss_functions):
                    ax = axes[i, j]

                    combo_df = ensemble_df[
                        (ensemble_df["loss_function"] == loss_fn)
                        & (ensemble_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        best_idx = combo_df[metric].idxmax()
                        best_row = combo_df.loc[best_idx]

                        maxent_vals = combo_df["maxent_value"].unique()
                        bvreg_vals = combo_df["bv_reg_value"].unique()

                        for maxent_val in maxent_vals:
                            for bvreg_val in bvreg_vals:
                                subset = combo_df[
                                    (combo_df["maxent_value"] == maxent_val)
                                    & (combo_df["bv_reg_value"] == bvreg_val)
                                ]
                                if len(subset) > 0:
                                    avg_metric = subset[metric].mean()
                                    color = "red" if (
                                        maxent_val == best_row["maxent_value"]
                                        and bvreg_val == best_row["bv_reg_value"]
                                    ) else "blue"
                                    size = 200 if color == "red" else 100
                                    marker = "*" if color == "red" else "o"

                                    ax.scatter(
                                        maxent_val,
                                        bvreg_val,
                                        s=size,
                                        c=color,
                                        marker=marker,
                                        alpha=0.7,
                                        edgecolors="black",
                                        linewidth=1,
                                    )

                                    ax.text(
                                        maxent_val,
                                        bvreg_val,
                                        f"{avg_metric:.1f}%",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                    )

                        ax.set_xscale("log")
                        ax.set_xlabel("MaxEnt Value (log scale)")
                        ax.set_ylabel("BV Reg Value")
                        ax.set_title(
                            f"{loss_fn} + {bv_reg_fn}\nBest: MaxEnt={best_row['maxent_value']:.1f}, "
                            f"BVReg={best_row['bv_reg_value']:.2f}"
                        )
                        ax.grid(True, alpha=0.3)

                        red_patch = mpatches.Patch(color="red", label="Best")
                        blue_patch = mpatches.Patch(color="blue", label="Other")
                        ax.legend(handles=[red_patch, blue_patch], loc="best")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, f"best_hyperparameters_{ensemble}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
