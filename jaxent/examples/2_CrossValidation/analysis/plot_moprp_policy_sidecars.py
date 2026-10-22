"""Three MoPrP sidecars, using raw file-weighted validation selection only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT/"fitting/jaxENT"))
from moprp_sidecar_support import ISO
sys.path.insert(0, str(ISO))
from plot_iso_policy_sidecars import plt, save, trace
from matplotlib.lines import Line2D

SPLITS = ("sequence_cluster", "spatial")
SPLIT_LABELS = {"sequence_cluster": "Non-Redundant", "spatial": "Spatial"}
ENSEMBLES = ("AF2_MSAss", "AF2_filtered")
LABELS = {"AF2_MSAss": "AF2-MSAss", "AF2_filtered": "AF2-Filtered",
    "work_scale": "Work Scale", "dihedral_rmsd": "Dihedral RMSD"}
METRICS = {"recovery_percent": "Recovery (%)", "ess_percent": "ESS (%)"}
VALIDATION_METRICS = {"val_weighted_mse": "Weighted validation MSE", "val_mse": "Validation MSE"}
METRIC_LABELS = METRICS | VALIDATION_METRICS | {"val_coordinate_sigma_mse": "Coordinate Sigma-MSE (validation)"}
ERROR_METRICS = set(VALIDATION_METRICS) | {"val_coordinate_sigma_mse"}


def uptake_label(selected):
    modes = set(selected.forward_model) if "forward_model" in selected else {"frame_uptake"}
    if modes == {"frame_uptake"}:
        return "full uptake"
    if modes == {"rate"}:
        return "rate-averaged uptake"
    if modes == {"linear_uptake"}:
        return "additive-linear-BV uptake"
    raise ValueError(f"Cannot combine uptake modes in a figure: {modes}")


def sweep_plot_rows(selected, smoke=False):
    """Production display range only; retain the full fitting/selection grid."""
    if smoke:
        return selected
    return selected[(selected.strength >= 1.) & (selected.strength <= 1000.)]


def style_axis(axis, rows, xlabel, metric="recovery_percent"):
    strengths = rows.strength.unique()
    axis.set(xscale="log", xlabel=xlabel,
        xlim=(min(strengths)/1.3, max(strengths)*1.3))
    if metric not in ERROR_METRICS:
        axis.set_ylim(0, 102)
    axis.grid(alpha=.2)


def plot_maxent(selected, metric, output, smoke=False, bv_fitted=False):
    selected = sweep_plot_rows(selected, smoke)
    rows = selected[selected.method == "maxent"]
    colors = yaml.safe_load((EXPERIMENT/"config.yaml").read_text())["style"]["split_type_colors"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for axis, ensemble in zip(axes, ENSEMBLES, strict=True):
        subset = rows[rows.ensemble == ensemble]
        for split in SPLITS:
            trace(axis, subset[subset.split_type == split].rename(columns={"strength": "data_strength"}),
                metric, colors[split], "-", label=SPLIT_LABELS[split])
        axis.set_title(LABELS[ensemble])
        style_axis(axis, subset, "MaxEnt scaling S (KL coefficient = 1/S)", metric)
    if metric in ERROR_METRICS:
        axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(METRIC_LABELS[metric])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    prefix = "SMOKE TEST · " if smoke else ""
    fitting = "MSE fitting · BV fitted" if bv_fitted else "MSE fitting"
    penalty = " · L1 penalty selected per replicate" if bv_fitted else ""
    fig.suptitle(prefix+f"MoPrP · {uptake_label(selected)} · {fitting}\nSelection: raw peptide × timepoint MoPrP.weights validation MSE{penalty}")
    fig.subplots_adjust(bottom=.18, top=.77, wspace=.12)
    save(fig, output)


def plot_omc(selected, metric, output, graph_k, smoke=False, bv_fitted=False):
    selected = sweep_plot_rows(selected, smoke)
    rows = selected[selected.ensemble == "AF2_MSAss"]
    omc = rows[(rows.method == "omc") & (rows.graph_k == graph_k)]
    bandwidths = sorted(omc.bandwidth.unique())
    colors = dict(zip(bandwidths, plt.get_cmap("viridis")(np.linspace(.08, .9, len(bandwidths))), strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for i, split in enumerate(SPLITS):
        for j, edge_type in enumerate(("work_scale", "dihedral_rmsd")):
            axis = axes[i, j]
            subset = omc[(omc.split_type == split) & (omc.edge_type == edge_type)]
            for bandwidth, color in colors.items():
                trace(axis, subset[subset.bandwidth == bandwidth].rename(columns={"strength": "data_strength"}),
                    metric, color, "-", bandwidth=True)
            baseline = rows[(rows.method == "maxent") & (rows.split_type == split)]
            trace(axis, baseline.rename(columns={"strength": "data_strength"}), metric, "black", "-")
            axis.set_title(f"{LABELS[edge_type]} · {SPLIT_LABELS[split]}")
            style_axis(axis, subset, "Strength S (regularizer coefficient = 1/S)" if i else "", metric)
        axes[i, 0].set_ylabel(METRIC_LABELS[metric])
    if metric in ERROR_METRICS:
        axes[0, 0].set_ylim(bottom=0)
    handles = [Line2D([], [], color="black", label="MaxEnt")]
    handles += [Line2D([], [], color=color, label=f"h={bandwidth:g}") for bandwidth, color in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    graph = "L2-logPF neighbours, k=20" if graph_k else "all pairs"
    prefix = "SMOKE TEST · " if smoke else ""
    fitting = "MSE fitting · BV fitted" if bv_fitted else "MSE fitting"
    penalty = " · L1 penalty selected per replicate" if bv_fitted else ""
    fig.suptitle(f"{prefix}MoPrP AF2-MSAss · {uptake_label(selected)} · {fitting} · {graph}\nRaw MoPrP.weights validation selection · h in median-distance units{penalty}")
    fig.subplots_adjust(bottom=.12, top=.85, hspace=.3, wspace=.12)
    save(fig, output)


def export_figures(selected, directory, smoke=False, bv_fitted=False, metrics=None):
    for metric in METRICS if metrics is None else metrics:
        plot_maxent(selected, metric, directory/f"maxent_{metric}", smoke, bv_fitted)
        graphs = set(selected.loc[selected.method == "omc", "graph_k"])
        if 0 in graphs:
            plot_omc(selected, metric, directory/f"omc_all_pairs_{metric}", 0, smoke, bv_fitted)
        if 20 in graphs:
            plot_omc(selected, metric, directory/f"omc_knn20_{metric}", 20, smoke, bv_fitted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=EXPERIMENT/"analysis/_moprp_policy_sidecars")
    parser.add_argument("--metrics", nargs="+", choices=list(METRIC_LABELS), default=list(METRICS))
    args = parser.parse_args()
    audit = args.analysis_dir/"audit.json"
    smoke = json.loads(audit.read_text()).get("smoke", False) if audit.exists() else False
    export_figures(pd.read_csv(args.analysis_dir/"selected.csv"), args.analysis_dir/"figures", smoke,
        metrics=args.metrics)
