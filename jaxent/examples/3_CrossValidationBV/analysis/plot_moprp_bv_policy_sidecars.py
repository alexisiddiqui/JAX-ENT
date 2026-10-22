"""Existing MoPrP plot layouts, selecting BV penalty within each replicate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT.parent/"2_CrossValidation/analysis"))
from plot_moprp_policy_sidecars import VALIDATION_METRICS, export_figures as export_sweeps
from plot_moprp_policy_model_selection_bars import METRICS, plot_bars, select_across_parameters, summarize


def profile_bv_penalties(selected):
    keys = ["ensemble", "method", "edge_type", "graph_k", "split_type", "split_idx", "strength", "bandwidth"]
    return selected.sort_values(["val_weighted_mse", "bv_reg_scaling", "step", "candidate_index"],
        kind="stable").groupby(keys, sort=False).head(1).copy()


def export_figures(candidates, selected, directory, suffix="", smoke=False):
    profiled = profile_bv_penalties(selected)
    profiled.to_csv(directory/f"selected_profiled{suffix}.csv", index=False)
    export_sweeps(profiled, directory/"figures", smoke, bv_fitted=True,
        metrics=list(METRICS)+list(VALIDATION_METRICS))
    layouts = ["maxent"]
    graphs = set(selected.loc[selected.method == "omc", "graph_k"])
    layouts += [label for key, label in ((0, "omc_all_pairs"), (20, "omc_knn20")) if key in graphs]
    if not graphs:
        return layouts
    final = select_across_parameters(candidates)
    summary = summarize(final)
    prefix = "all_pairs_knn20_parameter_selected"
    final.to_csv(directory/f"{prefix}{suffix}.csv", index=False)
    summary.to_csv(directory/f"{prefix}{suffix}_summary.csv", index=False)
    bandwidths = sorted(candidates.loc[candidates.method == "omc", "bandwidth"].unique())
    figure_prefix = f"{prefix}{suffix}"
    for metric in METRICS:
        plot_bars(final, summary, metric, directory/"figures"/f"{figure_prefix}_{metric}", bandwidths,
            bv_fitted=True, smoke=smoke)
    (directory/f"{prefix}{suffix}_policy.json").write_text(json.dumps(dict(
        forward_model=str(candidates.forward_model.iloc[0]), bv_fitted=True, training_loss="MSE", bandwidths=bandwidths,
        bv_penalties=sorted(candidates.bv_reg_scaling.unique()),
        selection="minimum mean(moprp.weights * residual squared) over saved states, strength, bandwidth and BV penalty within each split replicate",
        weight_normalization=False, aggregation="pool both split types; mean and sample SD; BV penalties are hyperparameters",
        candidate_policy="deduplicate step, weights and fitted model parameters", smoke=smoke), indent=2))
    return layouts+[figure_prefix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=EXPERIMENT/"analysis/_moprp_policy_sidecars")
    parser.add_argument("--validation-only", action="store_true",
        help="Plot weighted and ordinary validation MSE using the existing selected sweep points")
    parser.add_argument("--sweeps-only", action="store_true",
        help="Regenerate sweep plots from existing tables without reselection or table updates")
    args = parser.parse_args()
    smoke = json.loads((args.analysis_dir/"audit.json").read_text())["smoke"]
    if args.sweeps_only:
        export_sweeps(pd.read_csv(args.analysis_dir/"selected_profiled.csv"),
            args.analysis_dir/"figures", smoke, bv_fitted=True, metrics=list(METRICS)+list(VALIDATION_METRICS))
        sigma = args.analysis_dir/"selected_profiled_coordinate_sigma.csv"
        if sigma.exists():
            export_sweeps(pd.read_csv(sigma), args.analysis_dir/"figures", smoke,
                bv_fitted=True, metrics=["val_coordinate_sigma_mse"])
        return
    if args.validation_only:
        export_sweeps(pd.read_csv(args.analysis_dir/"selected_profiled.csv"),
            args.analysis_dir/"figures", smoke, bv_fitted=True, metrics=list(VALIDATION_METRICS))
        return
    export_figures(pd.read_csv(args.analysis_dir/"candidates.csv"),
        pd.read_csv(args.analysis_dir/"selected.csv"), args.analysis_dir, smoke=smoke)


if __name__ == "__main__":
    main()
