#!/usr/bin/env python3
"""Merge completed fake-peptide investigation shards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pilot-root", type=Path)
    args = parser.parse_args()
    shards = sorted(path for path in args.shard_root.iterdir() if path.is_dir())
    if len(shards) != 9:
        raise ValueError(f"Expected 9 panel/split shards, found {len(shards)}")

    raw = pd.concat([pd.read_csv(path / "raw_results.csv") for path in shards], ignore_index=True)
    selected = pd.concat(
        [pd.read_csv(path / "selected_results.csv") for path in shards], ignore_index=True
    )
    comparisons = pd.concat(
        [pd.read_csv(path / "comparisons.csv") for path in shards], ignore_index=True
    )
    moment_validation = pd.concat(
        [pd.read_csv(path / "target_moment_validation.csv") for path in shards],
        ignore_index=True,
    ).drop_duplicates(["panel", "target_scale"])
    if raw.run_id.duplicated().any() or selected.run_id.duplicated().any():
        raise ValueError("Shard outputs contain duplicate run IDs")
    weights: dict[str, np.ndarray] = {}
    panel_records = []
    split_metadata = {}
    first_manifest = json.loads((shards[0] / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        with np.load(shard / "selected_weights.npz") as archive:
            weights.update({key: np.asarray(archive[key]) for key in archive.files})
        panel_records.extend(json.loads((shard / "peptide_panels.json").read_text()))
        manifest = json.loads((shard / "manifest.json").read_text())
        if manifest["target_scales"] != first_manifest["target_scales"]:
            raise ValueError("Shard target scales differ")
        split_metadata.update(manifest["split_metadata"])
        for topology_path in shard.glob("panel_*_topology.json"):
            destination = args.output_dir / topology_path.name
            if not destination.exists():
                shutil.copy2(topology_path, destination)
        for mapping_path in shard.glob("panel_*_mapping.npz"):
            destination = args.output_dir / mapping_path.name
            if not destination.exists():
                shutil.copy2(mapping_path, destination)

    panels = pd.DataFrame(panel_records).drop_duplicates(["panel", "fragment_index"])
    expected_panels = 15 * panels.panel.nunique()
    if len(panels) != expected_panels or set(weights) != set(selected.run_id):
        raise ValueError("Merged panel metadata or selected weights are incomplete")
    target_scale_count = len(first_manifest["target_scales"])
    shard_config = first_manifest["config"]
    fits_per_condition = int(shard_config["starts"]) * (
        len(shard_config["maxent_values"])
        + 2
        * len(shard_config["gammas"])
        * len(shard_config["alphas"])
        * len(shard_config["maxent_values"])
    )
    expected_raw = len(shards) * target_scale_count * fits_per_condition
    expected_selected = len(shards) * target_scale_count * 3
    expected_comparisons = len(shards) * target_scale_count * 2
    if (
        len(raw) != expected_raw
        or len(selected) != expected_selected
        or len(comparisons) != expected_comparisons
    ):
        raise ValueError(
            "Incomplete merge: "
            f"raw={len(raw)}/{expected_raw}, "
            f"selected={len(selected)}/{expected_selected}, "
            f"comparisons={len(comparisons)}/{expected_comparisons}"
        )
    first_manifest["config"].update(
        {
            "panels": ["equal", "random_fixed", "random_variable"],
            "split_indices": [0, 1, 2],
            "starts": int(first_manifest["config"]["starts"]),
            "pilot": False,
            "smoke": False,
        }
    )
    first_manifest["split_metadata"] = split_metadata
    first_manifest["merged_shards"] = [str(path.resolve()) for path in shards]
    if args.pilot_root is not None:
        combined_pilot = args.pilot_root / "pilot_decision.json"
        if combined_pilot.exists():
            pilot_decision = json.loads(combined_pilot.read_text())
            first_manifest["pilot_decision"] = pilot_decision
            first_manifest["adaptive_start_decision"] = (
                "one_start"
                if pilot_decision["use_one_start_for_full_grid"]
                else "three_starts"
            )
        else:
            pilot_decisions = {
                panel: json.loads(
                    (args.pilot_root / panel / "pilot_decision.json").read_text()
                )
                for panel in ("equal", "random_fixed", "random_variable")
            }
            first_manifest["pilot_decisions"] = pilot_decisions
            first_manifest["adaptive_start_decision"] = "three_starts"

    support = []
    for (panel, target_scale, method), group in comparisons.groupby(
        ["panel", "target_scale", "method"]
    ):
        primary = bool(np.isclose(target_scale, 1.0))
        supported = bool(
            (group.recovery_gain_pp > 0).sum() >= 2
            and group.val_curve_mse_ratio.median() <= 1.05
            and (
                (group.baseline_gradient_cosine > 0)
                & (group.baseline_gradient_jsd_step_change < 0)
            ).sum()
            >= 2
        ) if primary else None
        support.append(
            {
                "panel": panel,
                "target_scale": float(target_scale),
                "method": method,
                "interpretation": "primary_recovery_gate" if primary else "tension_diagnostic",
                "supported": supported,
                "recovery_wins": int((group.recovery_gain_pp > 0).sum()),
                "required_wins": 2 if primary else None,
                "median_mean_preserved": bool(group.val_curve_mse_ratio.median() <= 1.05),
                "gradient_aligned": int(
                    (
                        (group.baseline_gradient_cosine > 0)
                        & (group.baseline_gradient_jsd_step_change < 0)
                    ).sum()
                ),
                "median_recovery_gain_pp": float(group.recovery_gain_pp.median()),
                "median_val_curve_mse_ratio": float(group.val_curve_mse_ratio.median()),
                "median_open_ess": float(group.open_ess.median()),
            }
        )
    support_frame = pd.DataFrame(support)
    paired = comparisons.pivot(
        index=["panel", "split_index", "target_scale"],
        columns="method",
        values="candidate_recovery_pct",
    ).reset_index()
    paired["conditional_minus_marginal_recovery_pp"] = paired.conditional - paired.marginal
    paired["winner"] = np.where(
        paired.conditional > paired.marginal, "conditional", "marginal"
    )
    summary = {
        "status": "evaluated",
        "comparison_count": len(comparisons),
        "target_moment_validation_passed": bool(
            (moment_validation.mapping_relative_error < 2e-6).all()
            and (moment_validation.scaled_covariance_relative_error < 1e-12).all()
            and (
                moment_validation.loc[
                    moment_validation.target_scale == 0.0, "covariance_trace"
                ] == 0.0
            ).all()
        ),
        "whole_ensemble_ess_used": False,
        "open_ess_used_as_gate": False,
        "support_rule": {
            "s1_recovery_and_gradient_wins": "at least 2 of 3 splits",
            "s0.1_policy": "diagnostic_only_nonrepresentable_by_40:60_unscaled_TRI_truth",
            "maximum_median_val_curve_mse_ratio": 1.05,
        },
        "groups": support,
        "method_supported_in_all_panels": {
            method: bool(
                support_frame[
                    (support_frame.method == method) & (support_frame.target_scale == 1.0)
                ].supported.all()
            )
            for method in ("marginal", "conditional")
        },
        "conditional_recovery_wins": int((paired.winner == "conditional").sum()),
        "marginal_recovery_wins": int((paired.winner == "marginal").sum()),
        "split_leakage": {
            key: value["duplicate_fragment_ids"]
            for key, value in split_metadata.items()
            if value["duplicate_fragment_ids"]
        },
    }

    raw.to_csv(args.output_dir / "raw_results.csv", index=False)
    selected.to_csv(args.output_dir / "selected_results.csv", index=False)
    comparisons.to_csv(args.output_dir / "comparisons.csv", index=False)
    moment_validation.to_csv(args.output_dir / "target_moment_validation.csv", index=False)
    paired.to_csv(args.output_dir / "paired_profile_results.csv", index=False)
    support_frame.to_csv(args.output_dir / "support_summary.csv", index=False)
    panels.to_json(args.output_dir / "peptide_panels.json", orient="records", indent=2)
    np.savez_compressed(args.output_dir / "selected_weights.npz", **weights)
    (args.output_dir / "manifest.json").write_text(json.dumps(first_manifest, indent=2))
    (args.output_dir / "decision.json").write_text(json.dumps(summary, indent=2))

    method_order = ["covariance_mse", "marginal", "conditional"]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, panel in zip(axes, ["equal", "random_fixed", "random_variable"], strict=True):
        group = selected[selected.panel == panel]
        values = [group[group.method == method].recovery_pct.to_numpy() for method in method_order]
        axis.boxplot(values, tick_labels=["Cov-MSE", "Marginal", "Conditional"])
        axis.set_title(panel.replace("_", " ").title())
        axis.set_ylabel("Cluster recovery (%)")
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(args.output_dir / "recovery_comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
