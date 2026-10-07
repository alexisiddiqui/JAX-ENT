#!/usr/bin/env python3
"""Re-select existing convergence checkpoints using audited closed-Sigma scores.

Reads the all-state selector analysis; never changes fits or source outputs.
The candidate set remains the original sidecar convergence checkpoints.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import run_maxent_weighted_mse_sidecar as sweep
from sidecar_selection import select_best_rows, SELECTION_POLICY


def attach_scores(frame, candidates):
    if candidates.duplicated(["run_id", "step"]).any():
        raise ValueError("Ambiguous candidate step; cannot safely match saved checkpoints")
    columns = candidates[["run_id", "step", "closed_coordinate", "mse"]].rename(
        columns={"closed_coordinate": "val_closed_sigma_mse", "mse": "audited_val_mse"})
    result = frame.drop(columns=["val_closed_sigma_mse"], errors="ignore").merge(
        columns, on=["run_id", "step"], how="left", validate="many_to_one")
    if result.val_closed_sigma_mse.isna().any():
        raise ValueError("Missing audited closed-Sigma scores")
    np.testing.assert_allclose(result.val_mse, result.audited_val_mse, rtol=3e-3, atol=2e-7)
    return result.drop(columns="audited_val_mse")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=sweep.HERE / "_maxent_weighted_mse_per_peptide_tri1e7_jobs8")
    parser.add_argument("--candidate-scores", type=Path,
                        default=sweep.HERE / "_within_trajectory_selection/candidate_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=sweep.HERE / "_maxent_sidecar_closed_sigma_selection")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.campaign.resolve():
        raise ValueError("Use a separate output directory to preserve the original selection")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidate_scores)
    convergence = attach_scores(pd.read_csv(args.campaign / "convergence_scores.csv"), candidates)
    final = attach_scores(pd.read_csv(args.campaign / "final_step_results.csv"), candidates)
    selected = select_best_rows(convergence)
    convergence.to_csv(args.output_dir / "convergence_scores.csv", index=False)
    for policy, frame in (("selected", selected), ("final_step", final)):
        frame.to_csv(args.output_dir / f"{policy}_results.csv", index=False)
        frame.groupby(["ensemble", "uptake_mode", "sigma_source", "maxent"])[
            ["recovery_percent", "ess_percent", "val_mse", "val_closed_sigma_mse"]
        ].agg(["mean", "std", "count"]).to_csv(args.output_dir / f"{policy}_summary.csv")
        for metric in ("recovery_percent", "ess_percent"):
            sweep.plot_metric(frame, metric, args.output_dir / "plots" / f"{metric}_vs_maxent_{policy}")
    for name in ("incomplete_runs", "missing_selection"):
        pd.read_csv(args.campaign / f"{name}.csv").to_csv(args.output_dir / f"{name}.csv", index=False)
    # All checkpoint weights are unchanged by selection; retain them so this
    # reanalysis is a complete, extendable result bundle.
    shutil.copy2(args.campaign / "frame_weights.npz", args.output_dir / "frame_weights.npz")
    original = pd.read_csv(args.campaign / "selected_results.csv")
    changes = original[["run_id", "step", "recovery_percent", "ess_percent"]].merge(
        selected[["run_id", "step", "recovery_percent", "ess_percent"]], on="run_id", suffixes=("_mse", "_closed"))
    changes["changed_checkpoint"] = changes.step_mse != changes.step_closed
    changes.to_csv(args.output_dir / "selection_changes.csv", index=False)
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    manifest.update(selection_metric=SELECTION_POLICY, selection_reanalysis=True,
        original_campaign=str(args.campaign.resolve()), candidate_score_source=str(args.candidate_scores.resolve()),
        selected_count=len(selected), final_count=len(final),
        changed_checkpoints=int(changes.changed_checkpoint.sum()), fitting_unchanged=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Selected {len(selected)} trajectories; {changes.changed_checkpoint.sum()} checkpoints changed; "
          f"{len(final)} final states unchanged")


if __name__ == "__main__":
    main()
