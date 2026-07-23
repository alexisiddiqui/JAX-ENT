#!/usr/bin/env python3
"""Reinterpret selected PF-variance fits using cluster-population recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.src.analysis.pf_variance import (
    conditional_subset_effective_sample_size,
    conditional_variance_log_ratio_loss,
    conditional_variance_profile,
    jensen_shannon_divergence,
    jensen_shannon_recovery_percent,
    marginal_variance_profile,
    weights_from_logits,
)


def _cluster_probabilities(weights: jax.Array, assignments: np.ndarray) -> jax.Array:
    labels = (0, 1, -1) if np.any(assignments == -1) else (0, 1)
    return jnp.stack([jnp.sum(weights[jnp.asarray(assignments == label)]) for label in labels])


def _target_probabilities(assignments: np.ndarray) -> jax.Array:
    return jnp.asarray([0.4, 0.6, 0.0] if np.any(assignments == -1) else [0.4, 0.6])


def _load_log_pf(path: Path) -> np.ndarray:
    with np.load(path) as features:
        return 0.35 * features["heavy_contacts"] + 2.0 * features["acceptor_contacts"]


def _load_train_indices(path: Path, dimension: int) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(
        [
            item["fragment_index"]
            for item in payload["topologies"]
            if 0 <= item["fragment_index"] < dimension
        ],
        dtype=int,
    )


def analyze(output_dir: Path, repo_root: Path, minimum_open_ess_fraction: float) -> None:
    example_root = repo_root / "jaxent/examples/1_IsoValidation_OMass"
    fitting_root = example_root / "fitting/jaxENT"
    selected = pd.read_csv(output_dir / "selected_results.csv")
    weight_archive = np.load(output_dir / "selected_weights.npz")
    log_pf = {
        name: jnp.asarray(_load_log_pf(fitting_root / f"_featurise/features_iso_{name.lower()}.npz"))
        for name in ("BI", "TRI")
    }
    assignments = {
        name: pd.read_csv(
            example_root / f"data/_clustering_results/cluster_assignments_ISO_{name}.csv"
        )["cluster_assignment"].to_numpy(dtype=int)
        for name in ("BI", "TRI")
    }
    target_weights = np.zeros(assignments["BI"].size, dtype=np.float32)
    target_weights[assignments["BI"] == 0] = 0.4 / np.sum(assignments["BI"] == 0)
    target_weights[assignments["BI"] == 1] = 0.6 / np.sum(assignments["BI"] == 1)
    target_weights_jax = jnp.asarray(target_weights)

    rows: list[dict[str, Any]] = []
    profile_name = (
        str(selected.variance_profile.iloc[0])
        if "variance_profile" in selected.columns
        else "conditional"
    )
    profile_function = (
        conditional_variance_profile if profile_name == "conditional" else marginal_variance_profile
    )
    grouping = ["ensemble", "target_mode", "split_type", "split_index"]
    for keys, group in selected.groupby(grouping):
        baseline = group[group.method == "baseline"].iloc[0]
        candidate = group[group.method == "variance_match"].iloc[0]
        ensemble, target_mode, split_type, split_index = keys
        assignment = assignments[ensemble]
        target_populations = _target_probabilities(assignment)
        baseline_weights = np.asarray(weight_archive[baseline.run_id])
        candidate_weights = np.asarray(weight_archive[candidate.run_id])
        train_indices = _load_train_indices(
            fitting_root
            / f"_datasplits/{split_type}/split_{int(split_index):03d}/train_topology.json",
            log_pf["BI"].shape[0],
        )
        alpha = float(candidate.alpha)
        target_profile = jax.lax.stop_gradient(
            profile_function(
                log_pf["BI"][train_indices], target_weights_jax, alpha=alpha
            )
        )

        def variance_loss(logits: jax.Array) -> jax.Array:
            profile = profile_function(
                log_pf[ensemble][train_indices], weights_from_logits(logits), alpha=alpha
            )
            return conditional_variance_log_ratio_loss(profile, target_profile)

        def population_jsd(logits: jax.Array) -> jax.Array:
            populations = _cluster_probabilities(weights_from_logits(logits), assignment)
            return jensen_shannon_divergence(populations, target_populations)

        def gradient_diagnostics(weights: np.ndarray) -> tuple[float, float]:
            logits = jnp.log(jnp.asarray(weights))
            variance_gradient = jax.grad(variance_loss)(logits)
            jsd_gradient = jax.grad(population_jsd)(logits)
            cosine = jnp.vdot(variance_gradient, jsd_gradient) / (
                jnp.linalg.norm(variance_gradient) * jnp.linalg.norm(jsd_gradient) + 1e-30
            )
            stepped_logits = logits - 0.01 * variance_gradient / (
                jnp.linalg.norm(variance_gradient) + 1e-30
            )
            step_change = population_jsd(stepped_logits) - population_jsd(logits)
            return float(cosine), float(step_change)

        baseline_populations = _cluster_probabilities(jnp.asarray(baseline_weights), assignment)
        candidate_populations = _cluster_probabilities(jnp.asarray(candidate_weights), assignment)
        uniform_weights = np.full(assignment.size, 1.0 / assignment.size, dtype=np.float32)
        uniform_cosine, uniform_step_change = gradient_diagnostics(uniform_weights)
        baseline_cosine, baseline_step_change = gradient_diagnostics(baseline_weights)
        baseline_recovery = float(
            jensen_shannon_recovery_percent(baseline_populations, target_populations)
        )
        candidate_recovery = float(
            jensen_shannon_recovery_percent(candidate_populations, target_populations)
        )
        open_mask = assignment == 0
        baseline_open_ess = float(
            conditional_subset_effective_sample_size(baseline_weights, open_mask)
        )
        candidate_open_ess = float(
            conditional_subset_effective_sample_size(candidate_weights, open_mask)
        )
        open_count = int(np.sum(open_mask))
        rows.append(
            {
                "ensemble": ensemble,
                "variance_profile": profile_name,
                "target_mode": target_mode,
                "split_type": split_type,
                "split_index": int(split_index),
                "baseline_recovery_pct": baseline_recovery,
                "candidate_recovery_pct": candidate_recovery,
                "recovery_gain_pp": candidate_recovery - baseline_recovery,
                "baseline_open_ess": baseline_open_ess,
                "candidate_open_ess": candidate_open_ess,
                "candidate_open_ess_fraction": candidate_open_ess / open_count,
                "uniform_variance_jsd_gradient_cosine": uniform_cosine,
                "uniform_variance_step_jsd_change": uniform_step_change,
                "baseline_variance_jsd_gradient_cosine": baseline_cosine,
                "baseline_variance_step_jsd_change": baseline_step_change,
                "mean_preserved": bool(candidate.val_curve_mse <= 1.05 * baseline.val_curve_mse),
            }
        )

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output_dir / "cluster_recovery_diagnostics.csv", index=False)
    comparisons = []
    for row in diagnostics.to_dict(orient="records"):
        comparisons.append(
            {
                **row,
                "recovery_improved": row["recovery_gain_pp"] > 0.0,
                "baseline_gradient_aligned": row["baseline_variance_jsd_gradient_cosine"] > 0.0
                and row["baseline_variance_step_jsd_change"] < 0.0,
                "open_ess_preserved": row["candidate_open_ess_fraction"]
                >= minimum_open_ess_fraction,
            }
        )
    complete = len(comparisons) == 24
    passed = complete and all(
        row["recovery_improved"]
        and row["baseline_gradient_aligned"]
        and row["open_ess_preserved"]
        and row["mean_preserved"]
        for row in comparisons
    )
    decision = {
        "status": "go" if passed else ("no_go" if complete else "not_evaluated"),
        "interpretation": "cluster_population_recovery",
        "variance_profile": profile_name,
        "recovery_definition": "100 * (1 - sqrt(base-2 JSD(predicted, target)))",
        "open_ess_definition": "ESS conditional on the open-cluster frame weights",
        "minimum_open_ess_fraction": minimum_open_ess_fraction,
        "whole_ensemble_ess_used": False,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-open-ess-fraction", type=float, default=0.8)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[5]
    analyze(args.output_dir.resolve(), repo_root, args.minimum_open_ess_fraction)


if __name__ == "__main__":
    main()
