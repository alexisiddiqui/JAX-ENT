#!/usr/bin/env python3
"""Fit-free MoPrP pivot litmus against the ensemble-independent exPfact PF solution.

Frame weights are fixed at ``w_NMR`` throughout.  The only search is an explicit
two-dimensional grid over the Best--Vendruscolo coefficients.  ``slow-N`` is
compared only in uptake space because a mixture of exponentials has no scalar PF.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

import _moprp_recovery_common as common
from jaxent.examples.common.analysis import frame_averaging
from jaxent.src.analysis import hdx_target_variance
from jaxent.src.analysis.hdx_ex2 import (
    fit_ex2_solution_set,
    load_expfact_dataset,
    predict_ex2_uptake,
    predict_trajectory_ex2,
)

JENSEN_GUARD_TOL = 1e-12
DEFAULT_GRID_SIZE = 41


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _scalar_log_pf(log_pf: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    legacy = log_pf @ weights
    fast = -logsumexp(-log_pf, axis=1, b=weights[None, :])
    return legacy, fast


def pivot_observable(log_pf, k_ints, timepoints, mapping, weights, pivot: str):
    """Differentiable shared pivot forward model, returning peptide-by-time uptake.

    This is the population-fit injection point as well as the canonical algebra used by
    the step-2 runner.  ``slow-N`` averages frame-resolved uptake curves and therefore has
    no associated scalar effective protection factor.
    """

    log_pf = jnp.asarray(log_pf)
    weights = jnp.asarray(weights)
    k_ints = jnp.asarray(k_ints)
    timepoints = jnp.asarray(timepoints)
    mapping = jnp.asarray(mapping)
    if pivot == "legacy":
        effective = log_pf @ weights
        pf = jnp.exp(effective)
        residue = 1.0 - jnp.exp(
            -timepoints[:, None] * k_ints[None, :] / pf[None, :]
        )
    elif pivot == "fast":
        rates = k_ints[:, None] * jnp.exp(-log_pf)
        effective_rates = rates @ weights
        residue = 1.0 - jnp.exp(-timepoints[:, None] * effective_rates[None, :])
    elif pivot == "slow-N":
        rates = k_ints[:, None] * jnp.exp(-log_pf)
        frame_uptake = 1.0 - jnp.exp(-timepoints[:, None, None] * rates[None, :, :])
        residue = jnp.einsum("trf,f->tr", frame_uptake, weights)
    else:
        raise ValueError(f"unknown pivot {pivot!r}")
    return (residue @ mapping.T).T


def pivot_effective_log_pf(log_pf, weights, pivot: str):
    """Differentiable scalar effective log-PF for the legacy/fast Jensen pair."""

    log_pf = jnp.asarray(log_pf)
    weights = jnp.asarray(weights)
    if pivot == "legacy":
        return log_pf @ weights
    if pivot == "fast":
        return -jax.scipy.special.logsumexp(-log_pf, axis=1, b=weights[None, :])
    raise ValueError("slow-N has no exact scalar effective protection factor")


def _curves(inputs, peptide_map, log_pf: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    legacy, slow_n = predict_trajectory_ex2(
        log_pf,
        inputs.k_ints,
        inputs.timepoints,
        peptide_map,
        frame_weights=weights,
    )
    _, fast_pf = _scalar_log_pf(log_pf, weights)
    fast = predict_ex2_uptake(fast_pf, inputs.k_ints, inputs.timepoints, peptide_map)
    return {"legacy": legacy, "fast": fast, "slow-N": slow_n}


def _peptide_map(inputs):
    dataset = load_expfact_dataset(common.MOPRP)
    return dataset.peptide_map.aligned_to(inputs.feature_residue_ids)


def _metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(predicted) - np.asarray(reference)
    return {
        "mean_signed_difference": float(np.mean(difference)),
        "rmse": _rmse(predicted, reference),
        "spearman": float(spearmanr(predicted, reference).statistic),
    }


def _load_expfact_reference(inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(common.MOPRP / "_output/MoPrP_pfactors.dat", dtype=float)
    ids = values[:, 0].astype(int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate residue IDs in MoPrP_pfactors.dat")
    lookup = {int(residue): float(value) for residue, value in values}
    overlap_mask = np.asarray([int(residue) in lookup for residue in inputs.feature_residue_ids])
    reference = np.asarray([lookup[int(r)] for r in inputs.feature_residue_ids[overlap_mask]])
    return ids, overlap_mask, reference


def _scan(inputs, peptide_map, weights, overlap_mask, reference_pf, reference_curves, grid_size):
    bc_grid = np.linspace(0.0, 1.0, grid_size)
    bh_grid = np.linspace(0.0, 4.0, grid_size)
    best = {
        pivot: {
            target: {"rmse": np.inf, "bc": None, "bh": None}
            for target in (("expfact_pf",) if pivot != "slow-N" else ()) +
            ("expfact_reconstruction", "observed")
        }
        for pivot in ("legacy", "fast", "slow-N")
    }
    rows = []
    for bc in bc_grid:
        for bh in bh_grid:
            log_pf = inputs.log_pf_by_frame(float(bc), float(bh))
            legacy_pf, fast_pf = _scalar_log_pf(log_pf, weights)
            curves = _curves(inputs, peptide_map, log_pf, weights)
            pf_by_pivot = {"legacy": legacy_pf, "fast": fast_pf}
            row = {"bc": float(bc), "bh": float(bh)}
            for pivot in ("legacy", "fast", "slow-N"):
                scores = {
                    "expfact_reconstruction": _rmse(curves[pivot], reference_curves),
                    "observed": _rmse(curves[pivot], inputs.observed_uptake),
                }
                if pivot in pf_by_pivot:
                    scores["expfact_pf"] = _rmse(pf_by_pivot[pivot][overlap_mask], reference_pf)
                for target, score in scores.items():
                    row[f"{pivot}_{target}_rmse"] = score
                    if score < best[pivot][target]["rmse"]:
                        best[pivot][target] = {
                            "rmse": score,
                            "bc": float(bc),
                            "bh": float(bh),
                        }
            rows.append(row)
    return best, rows, bc_grid, bh_grid


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = common.load_blinded_ensemble_inputs("AF2_MSAss", args.rate_source)
    _, _, _, weights = common.reveal_nmr_reference(
        "AF2_MSAss", expected_frames=inputs.n_frames
    )
    peptide_map = _peptide_map(inputs)
    expfact_ids, overlap_mask, reference_pf = _load_expfact_reference(inputs)
    overlap_ids = inputs.feature_residue_ids[overlap_mask]
    if not np.array_equal(np.sort(overlap_ids), np.sort(expfact_ids)):
        missing = sorted(set(expfact_ids) - set(inputs.feature_residue_ids))
        raise AssertionError(f"exPfact/feature overlap is incomplete; missing {missing}")

    # The exported 49-row file omits exPfact's -1 sentinel entries.  Its own
    # reconstruction requires the complete fitted vector from median.pfact.
    median = np.loadtxt(common.MOPRP / "median.pfact", dtype=float)
    median_lookup = {int(residue): float(value) for residue, value in median}
    full_reference_pf = np.asarray(
        [median_lookup[int(residue)] for residue in inputs.feature_residue_ids], dtype=float
    )
    if not np.array_equal(full_reference_pf[overlap_mask], reference_pf):
        raise AssertionError("exported exPfact PFs disagree with median.pfact")
    reference_curves = predict_ex2_uptake(
        full_reference_pf, inputs.k_ints, inputs.timepoints, peptide_map
    )

    published_log_pf = inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)
    rates_a = frame_averaging.effective_rates(published_log_pf, inputs.k_ints)
    rates_b = hdx_target_variance.effective_rates(published_log_pf, inputs.k_ints)
    rate_max_abs = float(np.max(np.abs(rates_a - rates_b)))
    if not np.array_equal(rates_a, rates_b):
        raise AssertionError("effective-rate implementations disagree")

    legacy_pf, fast_pf = _scalar_log_pf(published_log_pf, weights)
    gap = legacy_pf - fast_pf
    if np.any(fast_pf > legacy_pf + JENSEN_GUARD_TOL):
        raise AssertionError("Jensen guard failed")
    centered = published_log_pf - legacy_pf[:, None]
    half_variance = 0.5 * np.sum(weights[None, :] * centered**2, axis=1)
    second_order = _metrics(half_variance[overlap_mask], gap[overlap_mask])

    published_curves = _curves(inputs, peptide_map, published_log_pf, weights)
    baseline, _ = predict_trajectory_ex2(
        published_log_pf,
        inputs.k_ints,
        inputs.timepoints,
        peptide_map,
        frame_weights=weights,
    )
    baseline_max_abs = float(np.max(np.abs(baseline - published_curves["legacy"])))
    if baseline_max_abs != 0.0:
        raise AssertionError("legacy baseline reproduction failed")

    one_frame_weights = np.zeros(inputs.n_frames)
    one_frame_weights[0] = 1.0
    degenerate = _curves(inputs, peptide_map, published_log_pf, one_frame_weights)
    degenerate_max_abs = max(
        float(np.max(np.abs(degenerate[pivot] - degenerate["legacy"])))
        for pivot in ("fast", "slow-N")
    )
    if degenerate_max_abs > 2e-15:
        raise AssertionError("degenerate-weight pivot equality failed")

    solution_set = fit_ex2_solution_set(
        inputs.observed_uptake,
        inputs.k_ints,
        inputs.timepoints,
        peptide_map,
        starts=args.starts,
        initial_log_pf_vectors=[full_reference_pf],
        maxiter=args.maxiter,
    )
    lower, upper = solution_set.solution_range
    solution_spread = upper - lower
    finite_gate = overlap_mask & np.isfinite(solution_spread)
    exceeds = gap[finite_gate] > solution_spread[finite_gate]
    fraction_exceeding = float(np.mean(exceeds))
    gate_passed = bool(fraction_exceeding > 0.5)

    best, scan_rows, bc_grid, bh_grid = _scan(
        inputs,
        peptide_map,
        weights,
        overlap_mask,
        reference_pf,
        reference_curves,
        args.grid_size,
    )
    covered_peptides = inputs.peptide_ids[np.any(inputs.mapping[:, overlap_mask] > 0, axis=1)]

    residue_rows = []
    for index in np.flatnonzero(overlap_mask):
        residue_rows.append({
            "residue_id": int(inputs.feature_residue_ids[index]),
            "expfact_log_pf": float(full_reference_pf[index]),
            "legacy_log_pf": float(legacy_pf[index]),
            "fast_log_pf": float(fast_pf[index]),
            "legacy_minus_expfact": float(legacy_pf[index] - full_reference_pf[index]),
            "fast_minus_expfact": float(fast_pf[index] - full_reference_pf[index]),
            "legacy_minus_fast": float(gap[index]),
            "half_weighted_variance": float(half_variance[index]),
            "expfact_multistart_range": float(solution_spread[index]),
            "gap_exceeds_multistart_range": bool(gap[index] > solution_spread[index]),
        })

    frozen = {
        pivot: {
            "versus_observed_rmse": _rmse(curve, inputs.observed_uptake),
            "versus_expfact_reconstruction_rmse": _rmse(curve, reference_curves),
        }
        for pivot, curve in published_curves.items()
    }
    frozen["legacy"]["versus_expfact_pf"] = _metrics(legacy_pf[overlap_mask], reference_pf)
    frozen["fast"]["versus_expfact_pf"] = _metrics(fast_pf[overlap_mask], reference_pf)
    payload = {
        "description": "fit-free full-ensemble MoPrP pivot litmus at fixed w_NMR",
        "rate_provenance": common.rate_source_provenance(args.rate_source),
        "ensemble": "AF2_MSAss",
        "n_frames": inputs.n_frames,
        "overlap": {
            "count": int(overlap_mask.sum()),
            "residue_ids": overlap_ids.tolist(),
            "covered_peptide_ids": covered_peptides.tolist(),
        },
        "published_coefficients": {"bc": common.PUBLISHED_BC, "bh": common.PUBLISHED_BH},
        "frozen_results": frozen,
        "second_order_gap_check": second_order,
        "scan": {
            "bc_range": [float(bc_grid[0]), float(bc_grid[-1])],
            "bh_range": [float(bh_grid[0]), float(bh_grid[-1])],
            "grid_size_per_axis": args.grid_size,
            "best": best,
        },
        "resolvability_gate": {
            "starts_requested": args.starts,
            "finite_solutions": len(solution_set.solutions),
            "criterion": "legacy-fast gap exceeds solution range on >50% of overlap residues",
            "fraction_exceeding": fraction_exceeding,
            "count_exceeding": int(np.sum(exceeds)),
            "count_tested": int(np.sum(finite_gate)),
            "passed": gate_passed,
            "verdict": "resolved" if gate_passed else "inconclusive (underpowered)",
        },
        "verification": {
            "rate_convention_max_abs": rate_max_abs,
            "jensen_violations": int(np.sum(fast_pf > legacy_pf + JENSEN_GUARD_TOL)),
            "degenerate_weight_curve_max_abs": degenerate_max_abs,
            "legacy_baseline_max_abs": baseline_max_abs,
        },
        "caveats": [
            "exPfact PFs are fitted from the same uptake data, but independently of ensemble and pivot",
            "exPfact assumes EX2 and one PF per residue; slow-N therefore has no Panel-A scalar",
            "the overlap is a peptide-map-constrained, non-random subset",
            "w_NMR is pseudo-truth spread uniformly within state",
        ],
    }
    (args.output_dir / "moprp_pivot_litmus.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "published_uptake_curves.npz",
        observed=inputs.observed_uptake,
        expfact_reconstruction=reference_curves,
        legacy=published_curves["legacy"],
        fast=published_curves["fast"],
        slow_n=published_curves["slow-N"],
        peptide_ids=inputs.peptide_ids,
        timepoints_min=inputs.timepoints,
    )
    for filename, rows in (("residue_results.csv", residue_rows), ("coefficient_scan.csv", scan_rows)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_pivot_litmus",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    args = parser.parse_args()
    if args.grid_size < 2 or args.starts < 1 or args.maxiter < 1:
        parser.error("grid-size must be >=2 and starts/maxiter must be positive")
    run(args)


if __name__ == "__main__":
    main()
