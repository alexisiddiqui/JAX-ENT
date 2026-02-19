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
from matplotlib.colors import ListedColormap
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


# ---------------------------------------------------------------------------
# Score violin plots
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Generalised metric heatmap
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Weight distribution line plots
# ---------------------------------------------------------------------------


def plot_weight_distribution_lines(
    weights_data,
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Plot frame-weight distributions as line plots with maxent as hue.

    Parameters
    ----------
    weights_data:
        DataFrame (or list of dicts) produced by
        ``analysis.extract_frame_weights_kl``.  Must contain columns
        ``ensemble``, ``loss_function``, ``split_type``, ``maxent_value``,
        ``weights``.
    output_dir:
        Directory to save the output PNG files.
    split_name_mapping:
        Optional ``{split_type_key: display_name}`` dict for axis titles.
        Defaults to identity (raw key used as title).
    """
    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    weights_df = pd.DataFrame(weights_data) if not isinstance(weights_data, pd.DataFrame) else weights_data.copy()

    if weights_df.empty:
        print("  No weights data available for plotting")
        return

    weights_df["ensemble_loss"] = weights_df["ensemble"] + "|" + weights_df["loss_function"]

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        split_types = sorted(ensemble_loss_data["split_type"].unique())
        if not split_types:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, split_type in enumerate(split_types[:4]):
            ax = axes[idx]
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            if split_data.empty:
                ax.set_visible(False)
                continue

            maxent_groups: Dict = {}
            for _, row in split_data.iterrows():
                maxent = row["maxent_value"]
                if maxent not in maxent_groups:
                    maxent_groups[maxent] = []
                maxent_groups[maxent].append(row["weights"])

            maxent_values = sorted(maxent_groups.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

            weight_bins = np.logspace(-50, 0, 50)
            bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

            for maxent_val, color in zip(maxent_values, colors):
                weights_list = maxent_groups[maxent_val]
                hist_counts = []
                for w in weights_list:
                    if len(w) > 0 and np.sum(w) > 0:
                        counts, _ = np.histogram(w, bins=weight_bins, density=True)
                        hist_counts.append(counts)

                if hist_counts:
                    mean_counts = np.mean(hist_counts, axis=0)
                    std_counts = (
                        np.std(hist_counts, axis=0) if len(hist_counts) > 1 else np.zeros_like(mean_counts)
                    )
                    ax.plot(bin_centers, mean_counts, color=color, alpha=0.8,
                            label=f"MaxEnt={maxent_val:.0e}", linewidth=2)
                    if len(hist_counts) > 1:
                        ax.fill_between(bin_centers, mean_counts - std_counts,
                                        mean_counts + std_counts, color=color, alpha=0.2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")
            ax.set_title(split_name_mapping.get(split_type, split_type))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(split_types), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Weight Distributions — {ensemble_loss}", fontsize=16, y=0.98)
        plt.tight_layout()
        filename = f"weight_distributions_lines_{ensemble_loss.replace('|', '_').replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# 2D sweep heatmaps (Exp3)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Split analysis plots (from analyse_split_ISO_TRI_BI.py)
# ---------------------------------------------------------------------------


def plot_split_distributions(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Plot peptide distribution across splits as a seaborn countplot."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=all_splits_df, x="peptide_index", hue="split_type")
    plt.title("Peptide Distribution Across All Splits")
    plt.xlabel("Peptide Index")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peptide_split_distribution.png"))
    plt.close()


def plot_enhanced_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Three-panel heatmap: training set, validation set, and combined train/val/gap."""
    os.makedirs(output_dir, exist_ok=True)

    all_peptides = sorted(all_splits_df["peptide_index"].unique())
    all_splits = sorted(all_splits_df["split_dir"].unique())

    train_matrix = np.zeros((len(all_peptides), len(all_splits)))
    val_matrix = np.zeros((len(all_peptides), len(all_splits)))

    for i, peptide in enumerate(all_peptides):
        for j, split_dir in enumerate(all_splits):
            train_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "train")
            )
            if train_mask.any():
                train_matrix[i, j] = 1

            val_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "validation")
            )
            if val_mask.any():
                val_matrix[i, j] = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    im1 = ax1.imshow(train_matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    ax1.set_title("Training Set Distribution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split Directory")
    ax1.set_ylabel("Peptide Index")
    ax1.set_xticks(range(len(all_splits)))
    ax1.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax1.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax1.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax1.grid(True, alpha=0.3)

    im2 = ax2.imshow(val_matrix, cmap="Reds", aspect="auto", interpolation="nearest")
    ax2.set_title("Validation Set Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split Directory")
    ax2.set_ylabel("Peptide Index")
    ax2.set_xticks(range(len(all_splits)))
    ax2.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax2.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax2.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax2.grid(True, alpha=0.3)

    combined_matrix = np.zeros((len(all_peptides), len(all_splits)))
    combined_matrix[train_matrix == 1] = 1
    combined_matrix[val_matrix == 1] = 2

    colors_list = ["white", "#2E86AB", "#A23B72"]
    cmap_combined = ListedColormap(colors_list)

    im3 = ax3.imshow(
        combined_matrix, cmap=cmap_combined, aspect="auto", interpolation="nearest", vmin=0, vmax=2
    )
    ax3.set_title(
        "Combined Train/Val Distribution\n(Blue=Train, Red=Val, White=Gap)",
        fontsize=14, fontweight="bold",
    )
    ax3.set_xlabel("Split Directory")
    ax3.set_ylabel("Peptide Index")
    ax3.set_xticks(range(len(all_splits)))
    ax3.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax3.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax3.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax3.grid(True, alpha=0.3)

    plt.colorbar(im1, ax=ax1, label="In Training")
    plt.colorbar(im2, ax=ax2, label="In Validation")
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2])
    cbar3.set_ticklabels(["Gap", "Train", "Val"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "enhanced_peptide_split_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Pivot-table heatmap showing which peptides appear in which split."""
    os.makedirs(output_dir, exist_ok=True)
    pivot_table = all_splits_df.pivot_table(
        index="peptide_index", columns="split_dir", values="present", aggfunc="first"
    ).fillna(0)
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap="viridis", cbar=False)
    plt.title("Peptide Presence in Splits")
    plt.xlabel("Split Directory")
    plt.ylabel("Peptide Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peptide_split_heatmap.png"))
    plt.close()


def plot_uptake_heatmap(
    peptides: list,
    dfracs: np.ndarray,
    title: str,
    filename: str,
    split_type: str = "train",
) -> None:
    """Heatmap of deuterium uptake curves with colormap based on split type.

    Parameters
    ----------
    peptides:
        List of topology objects with ``residue_start`` and ``residue_end``
        attributes (e.g. ``Partial_Topology``).
    dfracs:
        Uptake array of shape ``(n_peptides, n_timepoints)`` or
        ``(n_peptides, n_timepoints, 1)``; leading singleton axes are squeezed.
    title:
        Figure title.
    filename:
        Full output path (including directory and ``.png`` extension).
    split_type:
        ``"train"`` → Blues colormap, ``"validation"`` → Reds, else viridis.
    """
    peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in peptides]
    colormap = {"train": "Blues", "validation": "Reds"}.get(split_type, "viridis")

    # Squeeze trailing singleton dimensions so heatmap receives a 2-D array.
    plot_dfracs = np.squeeze(dfracs)
    if plot_dfracs.ndim == 1:
        plot_dfracs = plot_dfracs.reshape(-1, 1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        plot_dfracs,
        xticklabels=False,
        yticklabels=peptide_names,
        cmap=colormap,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Peptide")
    plt.xlabel("Timepoint")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_uptake_comparison(
    train_peptides: list,
    train_dfracs: np.ndarray,
    val_peptides: list,
    val_dfracs: np.ndarray,
    split_name: str,
    output_dir: str,
) -> None:
    """Side-by-side heatmaps comparing training and validation uptake curves."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    train_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in train_peptides]
    sns.heatmap(
        np.squeeze(train_dfracs),
        xticklabels=False,
        yticklabels=train_peptide_names,
        cmap="Blues",
        ax=ax1,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax1.set_title(f"Training Set Uptake — {split_name}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Peptide")
    ax1.set_xlabel("Timepoint")

    val_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in val_peptides]
    sns.heatmap(
        np.squeeze(val_dfracs),
        xticklabels=False,
        yticklabels=val_peptide_names,
        cmap="Reds",
        ax=ax2,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax2.set_title(f"Validation Set Uptake — {split_name}", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Peptide")
    ax2.set_xlabel("Timepoint")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"combined_uptake_{split_name}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_gap_analysis(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Two-panel bar chart: missing peptide counts and coverage % per split."""
    os.makedirs(output_dir, exist_ok=True)

    all_peptides = set(all_splits_df["peptide_index"].unique())
    splits = all_splits_df["split_dir"].unique()

    gap_data = []
    for split_dir in splits:
        split_data = all_splits_df[all_splits_df["split_dir"] == split_dir]
        present_peptides = set(split_data["peptide_index"].unique())
        missing_peptides = all_peptides - present_peptides
        gap_data.append(
            {
                "split": split_dir.replace("split_", ""),
                "missing_count": len(missing_peptides),
                "present_count": len(present_peptides),
                "total_possible": len(all_peptides),
                "coverage_pct": (len(present_peptides) / len(all_peptides)) * 100,
            }
        )

    gap_df = pd.DataFrame(gap_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    bars1 = ax1.bar(gap_df["split"], gap_df["missing_count"], color="lightcoral", alpha=0.7)
    ax1.set_title("Missing Peptides per Split", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("Number of Missing Peptides")
    ax1.tick_params(axis="x", rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    bars2 = ax2.bar(gap_df["split"], gap_df["coverage_pct"], color="lightblue", alpha=0.7)
    ax2.set_title("Peptide Coverage per Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
