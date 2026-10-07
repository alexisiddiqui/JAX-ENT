"""Publication exports for MSE-fitted ISO sidecars (two checkpoint selectors)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SPLITS = ("sequence_cluster", "spatial")
SPLIT_LABELS = {"sequence_cluster": "Sequence-cluster", "spatial": "Spatial"}
SELECTORS = ("val_mse", "val_closed_sigma_mse")
SELECTOR_LABELS = {"val_mse": "MSE selection", "val_closed_sigma_mse": "Closed-Sigma selection"}
STYLES = {"val_mse": "-", "val_closed_sigma_mse": "--"}
CONSTRUCTIONS = ("work_scale", "rmsd", "pyrosetta")
LABELS = {"work_scale": "Work Scale", "rmsd": "Cα RMSD", "pyrosetta": "PyRosetta ref2015"}
METRICS = {"recovery_percent": "Recovery (%)", "ess_percent": "ESS (%)"}


def decade_label(value):
    exponent = np.log10(value)
    return rf"$10^{{{round(exponent)}}}$" if np.isclose(exponent, round(exponent)) else f"{value:g}"


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def trace(axis, rows, metric, color, style, label=None, bandwidth=False):
    if rows.empty:
        return
    stats = rows.groupby("data_strength")[metric].agg(["mean", "std", "count"]).sort_index()
    # Match the existing sidecar: plot each replicate and the available mean.
    for _, replicate in rows.groupby("split_idx"):
        replicate = replicate.sort_values("data_strength")
        axis.plot(replicate.data_strength, replicate[metric], color=color,
                  linestyle=style, alpha=.18, lw=.7)
    y = stats["mean"]
    axis.plot(stats.index, y, linestyle=style, marker="o", ms=3,
              color=color, label=label, lw=1.6)
    axis.fill_between(stats.index, y-stats["std"], y+stats["std"],
                      color=color, alpha=.045 if bandwidth else .13)


def axes_style(axis, xlabel):
    axis.set(xscale="log", xlabel=xlabel, ylim=(0, 102))
    axis.grid(alpha=.18)


def plot_maxent(selected, metric, output):
    rows = selected[selected.method == "maxent"]
    colors = {"sequence_cluster": "#0072B2", "spatial": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, ensemble in zip(axes, ("ISO_BI", "ISO_TRI"), strict=True):
        for split in SPLITS:
            for selector in SELECTORS:
                subset = rows[(rows.ensemble == ensemble) & (rows.split_type == split)
                              & (rows.selection_metric == selector)]
                trace(ax, subset, metric, colors[split], STYLES[selector])
        ax.set_title(ensemble.replace("_", " "))
        axes_style(ax, "MaxEnt scaling M (KL coefficient = 1/M)")
    axes[0].set_ylabel(METRICS[metric])
    handles = [Line2D([], [], color=c, label=SPLIT_LABELS[s]) for s, c in colors.items()]
    handles += [Line2D([], [], color=".25", linestyle=STYLES[s], label=SELECTOR_LABELS[s])
                for s in SELECTORS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("ISO · full uptake · MSE fitting · mean ± SD (3 replicates)")
    fig.subplots_adjust(bottom=.22, top=.82, wspace=.1)
    save(fig, output)


def plot_omc_curves(selected, metric, output):
    rows = selected[selected.ensemble == "ISO_TRI"]
    bandwidths = sorted(rows.loc[rows.method == "omc", "bandwidth"].unique())
    strengths = rows.loc[rows.method == "omc", "data_strength"].unique()
    colors = dict(zip(bandwidths, plt.get_cmap("viridis")(np.linspace(.08, .9, len(bandwidths))), strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for i, split in enumerate(SPLITS):
        for j, construction in enumerate(CONSTRUCTIONS):
            ax = axes[i, j]
            for selector in SELECTORS:
                for h, color in colors.items():
                    subset = rows[(rows.method == "omc") & (rows.graph_metric == construction)
                                  & (rows.split_type == split) & (rows.bandwidth == h)
                                  & (rows.selection_metric == selector)]
                    trace(ax, subset, metric, color, STYLES[selector], bandwidth=True)
                baseline = rows[(rows.method == "maxent") & (rows.split_type == split)
                                & (rows.selection_metric == selector)]
                baseline = baseline[baseline.data_strength.isin(strengths)]
                trace(ax, baseline, metric, "black", STYLES[selector])
            ax.set_title(f"{LABELS[construction]} · {SPLIT_LABELS[split]}")
            axes_style(ax, "Strength S (regularizer coefficient = 1/S)" if i else "")
            ax.set_xlim(min(strengths)/1.3, max(strengths)*1.3)
        axes[i, 0].set_ylabel(METRICS[metric])
    handles = [Line2D([], [], color="black", label="MaxEnt")]
    handles += [Line2D([], [], color=c, label=f"h={h:g}") for h, c in colors.items()]
    handles += [Line2D([], [], color=".25", linestyle=STYLES[s], label=SELECTOR_LABELS[s]) for s in SELECTORS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("ISO TRI · full uptake · MSE fitting · all-pairs OMC\nBandwidth h in median-distance units · mean ± SD (3 replicates)")
    fig.subplots_adjust(bottom=.17, top=.88, hspace=.3, wspace=.12)
    save(fig, output)


def plot_omc_heatmaps(selected, metric, output):
    rows = selected[selected.ensemble == "ISO_TRI"]
    omc = rows[rows.method == "omc"]
    strengths = sorted(omc.data_strength.unique())
    bandwidths = sorted(omc.bandwidth.unique())
    fig = plt.figure(figsize=(14, 17))
    grid = fig.add_gridspec(4, 3, hspace=.42, wspace=.25)
    image = None
    for r, (split, selector) in enumerate((s, v) for s in SPLITS for v in SELECTORS):
        for c, construction in enumerate(CONSTRUCTIONS):
            pair = grid[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=.18)
            ax = fig.add_subplot(pair[0])
            subset = omc[(omc.split_type == split) & (omc.selection_metric == selector)
                         & (omc.graph_metric == construction)]
            stats = subset.groupby(["bandwidth", "data_strength"])[metric].agg(["mean", "count"])
            values = stats["mean"].unstack("data_strength").reindex(index=bandwidths, columns=strengths)
            image = ax.imshow(values, origin="lower", aspect="auto", vmin=0, vmax=100, cmap="viridis")
            ax.set(xticks=np.arange(len(strengths)), xticklabels=[decade_label(s) for s in strengths],
                   yticks=np.arange(len(bandwidths)), yticklabels=[decade_label(h) for h in bandwidths])
            ax.tick_params(axis="x", labelbottom=False)
            if c == 0:
                ax.set_ylabel("Bandwidth h")
            ax.set_title(f"{LABELS[construction]} · {SPLIT_LABELS[split]}\n{SELECTOR_LABELS[selector]}", fontsize=10)
            baseline_ax = fig.add_subplot(pair[1])
            baseline = rows[(rows.method == "maxent") & (rows.split_type == split)
                            & (rows.selection_metric == selector) & rows.data_strength.isin(strengths)]
            # Categorical strength coordinates align exactly with heatmap columns.
            summary = baseline.groupby("data_strength")[metric].agg(["mean", "std", "count"]).reindex(strengths)
            x = np.arange(len(strengths))
            baseline_ax.plot(x, summary["mean"], color="black", linestyle=STYLES[selector], marker="o", ms=3)
            baseline_ax.fill_between(x, summary["mean"]-summary["std"], summary["mean"]+summary["std"], color="black", alpha=.12)
            baseline_ax.set(xlim=(-.5, len(strengths)-.5), ylim=(0, 102), xticks=x,
                            xticklabels=[decade_label(s) for s in strengths],
                            xlabel="Strength S" if r == 3 else "", ylabel="MaxEnt (%)")
            baseline_ax.grid(alpha=.15)
    fig.subplots_adjust(right=.9, bottom=.06, top=.91)
    colorbar_ax = fig.add_axes([.93, .15, .015, .7])
    fig.colorbar(image, cax=colorbar_ax, label=METRICS[metric])
    fig.suptitle("ISO TRI · full uptake · MSE fitting · all-pairs OMC\nBandwidth in median-distance units · mean of 3 replicates · black: MaxEnt mean ± SD", y=.98)
    fig.text(.5, .015, "Strength columns are aligned with the MaxEnt references.",
             ha="center", fontsize=10)
    save(fig, output)


def export_figures(selected, directory):
    for metric in METRICS:
        for name, plot in (("maxent", plot_maxent), ("omc_curves", plot_omc_curves),
                           ("omc_heatmaps", plot_omc_heatmaps)):
            plot(selected, metric, directory / f"{name}_{metric}")
