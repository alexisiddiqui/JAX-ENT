"""MoPrP bars after weighted validation selection over states and parameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_moprp_policy_sidecars import EXPERIMENT, METRICS, plt, save, uptake_label

PANELS = {0: "All pairs", 20: "k=20 · L2-logPF neighbours"}
MODELS = ("MaxEnt", "OMC Work Scale", "OMC Dihedral RMSD")
MODEL_LABELS = {"work_scale": MODELS[1], "dihedral_rmsd": MODELS[2]}
COLORS = ("black", "#0072B2", "#D55E00")


def select_across_parameters(candidates):
    """Select independently within each split replicate, before pooling splits."""
    candidates = candidates[candidates.ensemble == "AF2_MSAss"].copy()
    candidates["model"] = candidates.edge_type.map(MODEL_LABELS)
    candidates.loc[candidates.method == "maxent", "model"] = "MaxEnt"
    if candidates.model.isna().any():
        raise ValueError("Unknown OMC edge-weight construction")
    if not np.isfinite(candidates.val_weighted_mse).all():
        raise ValueError("Nonfinite weighted validation MSE")
    selections = []
    for graph_k, panel in PANELS.items():
        rows = candidates[(candidates.method == "maxent") |
            ((candidates.method == "omc") & (candidates.graph_k == graph_k))]
        if not (rows.method == "omc").any():
            continue
        order = ["val_weighted_mse", "strength", "bandwidth"]
        if "bv_reg_scaling" in rows:
            order += ["bv_reg_scaling"]
        best = rows.sort_values(
            order+["step", "candidate_index"],
            kind="stable").groupby(["model", "split_type", "split_idx"], sort=False).head(1).copy()
        best["panel"], best["panel_graph_k"] = panel, graph_k
        selections.append(best)
    return pd.concat(selections, ignore_index=True)


def summarize(selected):
    return selected.groupby(["panel_graph_k", "panel", "model"], sort=False).agg(
        n=("run_id", "size"), recovery_mean=("recovery_percent", "mean"),
        recovery_sd=("recovery_percent", "std"), ess_mean=("ess_percent", "mean"),
        ess_sd=("ess_percent", "std")).reset_index()


def plot_bars(selected, summary, metric, output, bandwidths, bv_fitted=False, smoke=False):
    field = "recovery" if metric == "recovery_percent" else "ess"
    mean, sd = f"{field}_mean", f"{field}_sd"
    panels = [key for key in PANELS if key in set(summary.panel_graph_k)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6*len(panels), 5.5), sharey=True, squeeze=False)
    limit = max(100. if field == "recovery" else 10.,
        float((summary[mean]+summary[sd].fillna(0)).max())*1.25,
        float(selected[metric].max())*1.1)
    for axis, graph_k in zip(axes[0], panels, strict=True):
        rows = summary[summary.panel_graph_k == graph_k].set_index("model").loc[list(MODELS)]
        x = np.arange(len(MODELS))
        axis.bar(x, rows[mean], .65, yerr=rows[sd].fillna(0), capsize=4, color=COLORS)
        for position, model in enumerate(MODELS):
            replicates = selected[(selected.panel_graph_k == graph_k) & (selected.model == model)]
            axis.scatter(position+np.linspace(-.16, .16, len(replicates)), replicates[metric],
                s=18, color=".25", edgecolor="white", linewidth=.5, zorder=3)
            value, error = rows.loc[model, [mean, sd]]
            axis.text(position, max(value+(error if np.isfinite(error) else 0),
                replicates[metric].max())+limit*.025, f"{value:.1f}", ha="center", fontsize=10)
        axis.set(title=PANELS[graph_k], xticks=x,
            xticklabels=["MaxEnt", "OMC\nWork Scale", "OMC\nDihedral RMSD"], ylim=(0, limit))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0, 0].set_ylabel(METRICS[metric])
    grid = ", ".join(f"{value:g}" for value in bandwidths)
    fitting = "MSE fitting · BV fitted" if bv_fitted else "MSE fitting"
    penalty = ", BV penalty" if bv_fitted else ""
    prefix = "SMOKE TEST · " if smoke else ""
    fig.suptitle(prefix+f"MoPrP AF2-MSAss · {uptake_label(selected)} · {fitting}\n"
        f"Final selection: raw MoPrP.weights validation MSE over saved states, strength{penalty} and bandwidth")
    fig.text(.5, .07, "Both split types pooled · bars: mean ± sample SD · dots: individual split replicates",
        ha="center", fontsize=9)
    fig.text(.5, .025, f"OMC bandwidths h = {grid}", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.22, top=.8, wspace=.12)
    save(fig, output)


def export_figures(candidates, directory):
    selected = select_across_parameters(candidates)
    summary = summarize(selected)
    prefix = "all_pairs_knn20_parameter_selected"
    selected.to_csv(directory/f"{prefix}.csv", index=False)
    summary.to_csv(directory/f"{prefix}_summary.csv", index=False)
    bandwidths = sorted(candidates.loc[candidates.method == "omc", "bandwidth"].unique())
    for metric in METRICS:
        plot_bars(selected, summary, metric, directory/"figures"/f"{prefix}_{metric}", bandwidths)
    (directory/f"{prefix}_policy.json").write_text(json.dumps(dict(
        ensemble="AF2_MSAss", bandwidths=bandwidths,
        candidate_policy="existing saved trajectory/convergence/running-best states",
        selection="minimum raw weighted validation MSE over saved states, strength and bandwidth within each split replicate",
        weighting="mean(moprp.weights * residual squared); direct file weights, no normalization",
        aggregation="pool both split types; mean and sample SD over split-replicate selections",
        source=str((directory/"candidates.csv").resolve())), indent=2))
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
        default=EXPERIMENT/"analysis/_moprp_policy_sidecars")
    args = parser.parse_args()
    export_figures(pd.read_csv(args.analysis_dir/"candidates.csv"), args.analysis_dir)


if __name__ == "__main__":
    main()
