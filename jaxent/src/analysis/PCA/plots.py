"""
jaxent.src.analysis.PCA.plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Publication-quality PCA plot functions for single- and multi-trajectory workflows.

All figures follow the scientific-visualization skill guidance:
- Arial/Helvetica fonts, sans-serif
- Minimum 7 pt tick labels, 8 pt axis labels at final print size
- Colorblind-safe palettes (Okabe-Ito for categorical, viridis for continuous)
- 300 DPI PNG export; spine cleanup via seaborn.despine()
- No 3-D effects, no jet/rainbow colormaps, no chart junk
- Explained variance always shown in axis labels

Public API
----------
create_publication_plots  — kCluster single-trajectory figure (relocated, unchanged behaviour)
plot_combined_scatter     — iPCA cross-condition scatter (PC1 vs PC2)
plot_combined_density     — iPCA cross-condition KDE density
plot_condition_replicates — iPCA per-condition replicate grid
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import IncrementalPCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication-style constants
# ---------------------------------------------------------------------------

_OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # rose
    "#000000",  # black
]

# Condition colormap families (cycling)
_CONDITION_CMAPS = ["Blues", "Oranges", "Greens", "Purples", "Reds"]

# Shade range sampled from each colormap (avoids near-white and near-black extremes)
_SHADE_RANGE = (0.3, 0.9)

_FIGURE_DPI = 300
_SINGLE_COL_W = 3.54  # inches — 89 mm (Nature single column)
_DOUBLE_COL_W = 7.20  # inches — 183 mm (Nature double column)


def _apply_publication_style() -> None:
    """Apply baseline publication rcParams (idempotent)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.labelweight": "normal",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 100,  # screen; export uses savefig dpi
            "savefig.dpi": _FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,  # embeds True Type fonts in PDF
            "ps.fonttype": 42,
        }
    )


def _pc_label(component: int, explained_variance: np.ndarray) -> str:
    """Return an axis label like 'PC1 (34.2% variance)'."""
    if explained_variance is not None and len(explained_variance) > component - 1:
        pct = explained_variance[component - 1] * 100
        return f"PC{component} ({pct:.1f}% variance)"
    return f"PC{component}"


def _condition_color(
    condition: str, all_conditions: Sequence[str], cmap_map: dict
) -> str:
    """Return the mid-shade colour for a condition (used in density/legend)."""
    cm_name = cmap_map.get(condition, "Blues")
    cm = plt.get_cmap(cm_name)
    return cm(0.6)


def _build_condition_cmap_map(conditions: Sequence[str]) -> dict[str, str]:
    """Map each unique condition to a distinct sequential colormap family."""
    unique = list(dict.fromkeys(conditions))  # stable de-dup
    return {
        c: _CONDITION_CMAPS[i % len(_CONDITION_CMAPS)] for i, c in enumerate(unique)
    }


def _shade_for_replicate(
    replicate: int, all_replicates_in_condition: list[int]
) -> float:
    """Sample a shade in [0.3, 0.9] for *replicate* within its condition."""
    sorted_reps = sorted(set(all_replicates_in_condition))
    n = len(sorted_reps)
    idx = sorted_reps.index(replicate)
    if n == 1:
        return 0.6
    lo, hi = _SHADE_RANGE
    return lo + (hi - lo) * idx / (n - 1)


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# kCluster: relocated create_publication_plots (unchanged behaviour, upgraded style)
# ---------------------------------------------------------------------------


def create_publication_plots(
    pca_coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    pca: IncrementalPCA,
    output_dir: str,
) -> dict[str, str | None]:
    """Publication-quality cluster visualisations for the kCluster workflow.

    Produces:
    - ``plots/pca_clusters.png``  — main 2-D scatter + KDE + marginals + scree
    - ``plots/pca_3d.png``        — 3-D scatter (only when n_components ≥ 3)
    - ``plots/cluster_distribution.png`` — cluster-size histogram

    Parameters
    ----------
    pca_coords:
        Shape ``(n_frames, n_components)``.
    cluster_labels:
        Integer cluster label per frame.
    cluster_centers:
        K-means centres in PCA space.
    pca:
        Fitted IncrementalPCA estimator (carries ``explained_variance_ratio_``).
    output_dir:
        Root output directory; figures go into ``{output_dir}/plots/``.

    Returns
    -------
    dict with keys ``"pca_plot"``, ``"pca_3d_plot"`` (may be None), ``"cluster_dist"``.
    """
    _apply_publication_style()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ev = pca.explained_variance_ratio_

    # ------------------------------------------------------------------ #
    # Figure 1 — main PCA scatter + KDE + marginals + scree              #
    # ------------------------------------------------------------------ #
    logger.info("create_publication_plots: main PCA figure")

    fig = plt.figure(figsize=(_DOUBLE_COL_W, _DOUBLE_COL_W * 0.85))
    gs = GridSpec(
        3, 3, figure=fig, height_ratios=[0.25, 1, 0.4], hspace=0.45, wspace=0.35
    )

    x, y = pca_coords[:, 0], pca_coords[:, 1]

    # --- Main scatter ---
    ax_main = fig.add_subplot(gs[1, :2])
    sns.kdeplot(
        x=x,
        y=y,
        ax=ax_main,
        levels=10,
        cmap="Blues",
        fill=True,
        alpha=0.35,
        zorder=0,
    )
    sc = ax_main.scatter(
        x,
        y,
        c=cluster_labels,
        cmap="viridis",
        s=6,
        alpha=0.6,
        linewidths=0,
        zorder=5,
    )
    ax_main.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="white",
        s=40,
        marker="X",
        edgecolors="#D55E00",
        linewidths=0.8,
        zorder=10,
        label="Cluster centres",
    )
    ax_main.set_xlabel(_pc_label(1, ev))
    ax_main.set_ylabel(_pc_label(2, ev))
    ax_main.legend(loc="upper right", markerscale=1.2)
    sns.despine(ax=ax_main)

    # --- PC1 marginal ---
    ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
    ax_top.hist(x, bins=50, color="#0072B2", alpha=0.7, linewidth=0)
    ax_top.set_ylabel("Count")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(
        "PCA projection with k-means clustering", fontsize=9, fontweight="bold", pad=4
    )
    sns.despine(ax=ax_top)

    # --- PC2 marginal ---
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_right.hist(
        y, bins=50, orientation="horizontal", color="#0072B2", alpha=0.7, linewidth=0
    )
    ax_right.set_xlabel("Count")
    ax_right.tick_params(labelleft=False)
    sns.despine(ax=ax_right)

    # Colourbar
    cbar = fig.colorbar(sc, ax=ax_right, shrink=0.8, pad=0.02)
    cbar.set_label("Cluster", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # --- Scree plot ---
    ax_scree = fig.add_subplot(gs[2, :])
    comps = np.arange(1, len(ev) + 1)
    cum_ev = np.cumsum(ev)
    ax_scree.bar(comps, ev, color="#0072B2", alpha=0.75, width=0.6, label="Individual")
    ax_twin = ax_scree.twinx()
    ax_twin.plot(
        comps,
        cum_ev,
        "o-",
        color="#D55E00",
        linewidth=1.2,
        markersize=3,
        label="Cumulative",
    )
    ax_twin.set_ylim(0, 1.05)
    ax_twin.set_ylabel("Cumulative EVR", fontsize=7)
    ax_twin.tick_params(labelsize=6)
    ax_scree.set_xlabel("Principal component")
    ax_scree.set_ylabel("Explained variance ratio")
    ax_scree.set_xticks(comps)
    # Merge legends
    h1, l1 = ax_scree.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax_scree.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2)
    sns.despine(ax=ax_scree)

    pca_plot_path = os.path.join(plots_dir, "pca_clusters.png")
    _save_fig(fig, pca_plot_path)

    # ------------------------------------------------------------------ #
    # Figure 2 — 3-D PCA (optional)                                      #
    # ------------------------------------------------------------------ #
    pca_3d_path: str | None = None
    if pca_coords.shape[1] >= 3:
        logger.info("create_publication_plots: 3-D PCA figure")
        fig3d = plt.figure(figsize=(_SINGLE_COL_W * 1.4, _SINGLE_COL_W * 1.2))
        ax3d = fig3d.add_subplot(111, projection="3d")
        sc3d = ax3d.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            pca_coords[:, 2],
            c=cluster_labels,
            cmap="viridis",
            s=5,
            alpha=0.6,
        )
        ax3d.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            cluster_centers[:, 2],
            c="white",
            s=30,
            marker="X",
            edgecolors="#D55E00",
            linewidths=0.8,
        )
        ax3d.set_xlabel(_pc_label(1, ev), labelpad=4)
        ax3d.set_ylabel(_pc_label(2, ev), labelpad=4)
        ax3d.set_zlabel(_pc_label(3, ev), labelpad=4)
        ax3d.set_title("3-D PCA projection", fontsize=9, fontweight="bold")
        fig3d.colorbar(sc3d, ax=ax3d, shrink=0.6, pad=0.1, label="Cluster")
        pca_3d_path = os.path.join(plots_dir, "pca_3d.png")
        _save_fig(fig3d, pca_3d_path)

    # ------------------------------------------------------------------ #
    # Figure 3 — cluster-size distribution                               #
    # ------------------------------------------------------------------ #
    logger.info("create_publication_plots: cluster size distribution")
    _unique, counts = np.unique(cluster_labels, return_counts=True)
    fig_dist, ax_dist = plt.subplots(figsize=(_SINGLE_COL_W, _SINGLE_COL_W * 0.75))
    ax_dist.hist(counts, bins=30, color="#0072B2", alpha=0.75, linewidth=0)
    ax_dist.axvline(
        np.mean(counts),
        color="#D55E00",
        linewidth=1.0,
        linestyle="--",
        label=f"Mean {np.mean(counts):.0f}",
    )
    ax_dist.axvline(
        np.median(counts),
        color="#009E73",
        linewidth=1.0,
        linestyle=":",
        label=f"Median {np.median(counts):.0f}",
    )
    ax_dist.set_xlabel("Frames per cluster")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Cluster-size distribution", fontsize=9, fontweight="bold")
    ax_dist.legend()
    sns.despine(ax=ax_dist)

    cluster_dist_path = os.path.join(plots_dir, "cluster_distribution.png")
    _save_fig(fig_dist, cluster_dist_path)

    return {
        "pca_plot": pca_plot_path,
        "pca_3d_plot": pca_3d_path,
        "cluster_dist": cluster_dist_path,
    }


# ---------------------------------------------------------------------------
# iPCA: Figure 1 — combined scatter
# ---------------------------------------------------------------------------


def plot_combined_scatter(
    pca_coords: np.ndarray,
    metadata: dict,
    condition_cmap_map: dict[str, str],
    output_path: str,
    explained_variance: np.ndarray | None = None,
) -> None:
    """PC1 vs PC2 scatter for all frames, coloured by condition × replicate.

    Parameters
    ----------
    pca_coords:
        Shape ``(total_frames, n_components)``.
    metadata:
        Must contain keys ``"conditions"`` and ``"replicates"`` (parallel arrays).
    condition_cmap_map:
        Maps each unique condition string to a matplotlib colormap name.
    output_path:
        Full path to write the PNG figure.
    explained_variance:
        Shape ``(n_components,)`` — used to annotate axis labels.
    """
    _apply_publication_style()

    conditions: np.ndarray = np.asarray(metadata["conditions"])
    replicates: np.ndarray = np.asarray(metadata["replicates"])

    unique_conditions = list(dict.fromkeys(conditions))
    fig, ax = plt.subplots(figsize=(_SINGLE_COL_W, _SINGLE_COL_W))

    for cond in unique_conditions:
        cond_mask = conditions == cond
        cmap_name = condition_cmap_map.get(cond, "Blues")
        cmap = plt.get_cmap(cmap_name)
        reps_in_cond = sorted(set(int(r) for r in replicates[cond_mask]))

        for rep in reps_in_cond:
            rep_mask = cond_mask & (replicates == rep)
            shade = _shade_for_replicate(rep, reps_in_cond)
            color = cmap(shade)
            ax.scatter(
                pca_coords[rep_mask, 0],
                pca_coords[rep_mask, 1],
                s=4,
                alpha=0.5,
                linewidths=0,
                color=color,
                label=f"{cond} rep{rep}",
            )

    ax.set_xlabel(
        _pc_label(
            1, explained_variance if explained_variance is not None else np.array([])
        )
    )
    ax.set_ylabel(
        _pc_label(
            2, explained_variance if explained_variance is not None else np.array([])
        )
    )
    ax.set_title("Combined PCA scatter", fontsize=9, fontweight="bold")
    # Place legend outside axes to avoid overplotting
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        handlelength=0.8,
    )
    sns.despine(ax=ax)

    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# iPCA: Figure 2 — combined density
# ---------------------------------------------------------------------------


def plot_combined_density(
    pca_coords: np.ndarray,
    metadata: dict,
    condition_cmap_map: dict[str, str],
    output_path: str,
    explained_variance: np.ndarray | None = None,
) -> None:
    """Filled KDE contours per condition overlaid in a single panel.

    Parameters
    ----------
    pca_coords:
        Shape ``(total_frames, n_components)``.
    metadata:
        Must contain key ``"conditions"`` (parallel array).
    condition_cmap_map:
        Maps each unique condition to a matplotlib colormap name.
    output_path:
        Full path to write the PNG figure.
    explained_variance:
        Shape ``(n_components,)`` — used to annotate axis labels.
    """
    _apply_publication_style()

    conditions: np.ndarray = np.asarray(metadata["conditions"])
    unique_conditions = list(dict.fromkeys(conditions))

    fig, ax = plt.subplots(figsize=(_SINGLE_COL_W, _SINGLE_COL_W))

    for i, cond in enumerate(unique_conditions):
        mask = conditions == cond
        cmap_name = condition_cmap_map.get(cond, "Blues")
        sns.kdeplot(
            x=pca_coords[mask, 0],
            y=pca_coords[mask, 1],
            ax=ax,
            fill=True,
            levels=8,
            cmap=cmap_name,
            alpha=0.45,
            label=cond,
        )

    ax.set_xlabel(
        _pc_label(
            1, explained_variance if explained_variance is not None else np.array([])
        )
    )
    ax.set_ylabel(
        _pc_label(
            2, explained_variance if explained_variance is not None else np.array([])
        )
    )
    ax.set_title("Combined PCA density", fontsize=9, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    sns.despine(ax=ax)

    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# iPCA: Figure 3 — per-condition replicate panels
# ---------------------------------------------------------------------------


def plot_condition_replicates(
    pca_coords: np.ndarray,
    metadata: dict,
    condition: str,
    output_path: str,
    explained_variance: np.ndarray | None = None,
) -> None:
    """2-row grid (scatter + KDE) with one column per replicate for a single condition.

    Row 0 — scatter coloured by frame index (viridis, shows temporal progression).
    Row 1 — KDE density coloured by density.

    All panels share the same PC1/PC2 axis limits for direct comparison.

    Parameters
    ----------
    pca_coords:
        Shape ``(total_frames, n_components)``.
    metadata:
        Must contain ``"conditions"`` and ``"replicates"`` (parallel arrays).
    condition:
        Which condition to plot.
    output_path:
        Full path to write the PNG figure.
    explained_variance:
        Shape ``(n_components,)`` — used to annotate axis labels.
    """
    _apply_publication_style()

    conditions: np.ndarray = np.asarray(metadata["conditions"])
    replicates: np.ndarray = np.asarray(metadata["replicates"])

    cond_mask = conditions == condition
    reps_in_cond = sorted(set(int(r) for r in replicates[cond_mask]))
    n_reps = len(reps_in_cond)

    if n_reps == 0:
        logger.warning(
            "plot_condition_replicates: no replicates found for condition '%s'",
            condition,
        )
        return

    # Determine global PC1/PC2 limits across all replicates of this condition
    x_all = pca_coords[cond_mask, 0]
    y_all = pca_coords[cond_mask, 1]
    pad_x = (x_all.max() - x_all.min()) * 0.05 or 0.1
    pad_y = (y_all.max() - y_all.min()) * 0.05 or 0.1
    xlim = (x_all.min() - pad_x, x_all.max() + pad_x)
    ylim = (y_all.min() - pad_y, y_all.max() + pad_y)

    col_width = _SINGLE_COL_W if n_reps == 1 else _DOUBLE_COL_W / max(n_reps, 2)
    fig_w = col_width * n_reps
    fig_h = col_width * 2 * 0.85

    fig, axes = plt.subplots(
        2,
        n_reps,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    xlab = _pc_label(
        1, explained_variance if explained_variance is not None else np.array([])
    )
    ylab = _pc_label(
        2, explained_variance if explained_variance is not None else np.array([])
    )

    for col_i, rep in enumerate(reps_in_cond):
        rep_mask = cond_mask & (replicates == rep)
        x = pca_coords[rep_mask, 0]
        y = pca_coords[rep_mask, 1]
        n_frames_rep = rep_mask.sum()

        # --- Row 0: scatter by frame index ---
        ax_sc = axes[0, col_i]
        sc = ax_sc.scatter(
            x,
            y,
            c=np.arange(n_frames_rep),
            cmap="viridis",
            s=4,
            alpha=0.6,
            linewidths=0,
        )
        ax_sc.set_xlim(xlim)
        ax_sc.set_ylim(ylim)
        ax_sc.set_title(f"{condition} rep{rep}", fontsize=8, fontweight="bold")
        if col_i == 0:
            ax_sc.set_ylabel(ylab)
        # Colourbar on last column only
        if col_i == n_reps - 1:
            cb = fig.colorbar(sc, ax=ax_sc, shrink=0.8, pad=0.02)
            cb.set_label("Frame index", fontsize=6)
            cb.ax.tick_params(labelsize=5)
        sns.despine(ax=ax_sc)

        # --- Row 1: KDE density ---
        ax_kd = axes[1, col_i]
        try:
            sns.kdeplot(
                x=x,
                y=y,
                ax=ax_kd,
                fill=True,
                cmap="viridis",
                levels=8,
                alpha=0.8,
            )
        except Exception:
            # Fall back gracefully if too few unique frames for KDE
            ax_kd.scatter(x, y, s=4, alpha=0.5, color="#0072B2", linewidths=0)

        ax_kd.set_xlim(xlim)
        ax_kd.set_ylim(ylim)
        ax_kd.set_xlabel(xlab)
        if col_i == 0:
            ax_kd.set_ylabel(ylab)
        sns.despine(ax=ax_kd)

    fig.suptitle(
        f"Per-replicate PCA — {condition}",
        fontsize=9,
        fontweight="bold",
        y=1.02,
    )

    _save_fig(fig, output_path)
