"""ISO TRI bars after validation selection over saved states, strength and bandwidth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_iso_policy_sidecars import METRICS, SELECTORS, SELECTOR_LABELS, plt, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PANELS = ("All pairs", "k=20")
MODELS = ("MaxEnt", "OMC Work Scale", "OMC RMSD", "OMC PyRosetta")
COLORS = {"val_mse": "#0072B2", "val_closed_sigma_mse": "#D55E00"}
MODEL_LABELS = {"work_scale": "OMC Work Scale", "rmsd": "OMC RMSD", "pyrosetta": "OMC PyRosetta"}


def select_across_parameters(frames, max_bandwidth=1.):
    selected = []
    for panel, candidates in frames.items():
        candidates = candidates[(candidates.ensemble == "ISO_TRI") &
            ((candidates.method == "maxent") | ((candidates.method == "omc") &
                (candidates.bandwidth <= max_bandwidth)))].copy()
        candidates["model"] = candidates.graph_metric.map(MODEL_LABELS)
        candidates.loc[candidates.method == "maxent", "model"] = "MaxEnt"
        if candidates.model.isna().any():
            raise ValueError("Unknown OMC edge-weight construction")
        for selector in SELECTORS:
            if not np.isfinite(candidates[selector]).all():
                raise ValueError(f"Nonfinite {selector} candidates in {panel}")
            best = candidates.sort_values(
                [selector, "data_strength", "bandwidth", "step", "candidate_index"],
                kind="stable").groupby(["model", "split_type", "split_idx"], sort=False).head(1)
            best = best.copy()
            best["panel"], best["selection_metric"] = panel, selector
            selected.append(best)
    return pd.concat(selected, ignore_index=True)


def summarize(selected):
    keys = ["panel", "model", "selection_metric"]
    summary = selected.groupby(keys, sort=False).agg(
        n=("run_id", "size"), recovery_mean=("recovery_percent", "mean"),
        recovery_sd=("recovery_percent", "std"), ess_mean=("ess_percent", "mean"),
        ess_sd=("ess_percent", "std")).reset_index()
    for panel in PANELS:
        for model in MODELS:
            for selector in SELECTORS:
                rows = selected[(selected.panel == panel) & (selected.model == model) &
                    (selected.selection_metric == selector)]
                if set(zip(rows.split_type, rows.split_idx)) != {
                        (split, index) for split in ("sequence_cluster", "spatial") for index in range(3)} or len(rows) != 6:
                    raise ValueError(f"Expected six split-replicate selections: {panel}/{model}/{selector}")
    return summary


def plot_bars(summary, metric, output, max_bandwidth):
    field = "recovery" if metric == "recovery_percent" else "ess"
    mean, sd = f"{field}_mean", f"{field}_sd"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    width = .34
    limit = 100 if field == "recovery" else max(10., float((summary[mean]+summary[sd]).max())*1.3)
    for ax, panel in zip(axes, PANELS, strict=True):
        for i, selector in enumerate(SELECTORS):
            rows = summary[(summary.panel == panel) & (summary.selection_metric == selector)].set_index("model").loc[list(MODELS)]
            x = np.arange(len(MODELS)) + (i-.5)*width
            ax.bar(x, rows[mean], width, yerr=rows[sd], capsize=3,
                color=COLORS[selector], label=SELECTOR_LABELS[selector])
            for position, value, error in zip(x, rows[mean], rows[sd], strict=True):
                ax.text(position, value+error+limit*.018, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set(title=panel, xticks=np.arange(len(MODELS)),
            xticklabels=["MaxEnt", "OMC\nWork Scale", "OMC\nCα RMSD", "OMC\nPyRosetta"], ylim=(0, limit))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(METRICS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .045))
    fig.suptitle(f"ISO TRI · MSE fitting · validation selection over all saved states and parameters\nOMC bandwidth h ≤ {max_bandwidth:g}")
    fig.text(.5, .015, "Mean ± sample SD over six selections: three replicates for each of two split types.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.22, top=.8, wspace=.12)
    save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-pairs-dir", type=Path, default=REPO/"artifacts/iso_policy_sidecars")
    parser.add_argument("--knn-dir", type=Path, default=REPO/"artifacts/iso_policy_sidecars_knn20")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-bandwidth", type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.max_bandwidth) or args.max_bandwidth <= 0:
        parser.error("Maximum bandwidth must be positive and finite")
    output = args.output_dir or args.all_pairs_dir
    frames = {panel: pd.read_csv(directory/"candidates.csv")
        for panel, directory in zip(PANELS, (args.all_pairs_dir, args.knn_dir), strict=True)}
    selected = select_across_parameters(frames, args.max_bandwidth)
    summary = summarize(selected)
    prefix = "all_pairs_knn20_parameter_selected"
    selected.to_csv(output/f"{prefix}.csv", index=False)
    summary.to_csv(output/f"{prefix}_summary.csv", index=False)
    for metric in METRICS:
        plot_bars(summary, metric, output/"figures"/f"{prefix}_{metric}", args.max_bandwidth)
    (output/f"{prefix}_policy.json").write_text(json.dumps(dict(
        ensemble="ISO_TRI", maximum_omc_bandwidth=args.max_bandwidth,
        candidate_policy="existing saved trajectory/convergence/running-best states",
        selection="minimum corresponding validation metric over states, strength and bandwidth within each split replicate",
        aggregation="six selections per bar; mean and sample SD; equal weight for both split types",
        all_pairs_candidates=str(args.all_pairs_dir/"candidates.csv"),
        knn_candidates=str(args.knn_dir/"candidates.csv")), indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
