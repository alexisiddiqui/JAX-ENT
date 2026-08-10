#!/usr/bin/env python3
"""Lock shared MoPrP Best--Vendruscolo coefficients separately for each HDX pivot.

The NMR reference weights remain fixed.  Coefficients are calibrated jointly across the
AF2-MSAss and AF2-Filtered ensembles on peptides 2--14; no frame weight is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import _moprp_recovery_common as common
from moprp_pivot_litmus import JENSEN_GUARD_TOL, _curves, _peptide_map, _scalar_log_pf

PEPTIDE1_INDEX = common.PEPTIDE1_INDEX
PIVOTS = ("legacy", "fast", "slow-N")
STARTS = ((0.35, 2.0), (0.5, 0.5), (0.1, 0.1), (1.0, 0.0))
SCALE_BOUNDS = (1e-3, 3.0)


def _predicted_peptide_uptake(inputs, peptide_map, bc: float, bh: float, pivot: str) -> np.ndarray:
    """Return peptide-by-time uptake for one of the shared litmus pivot definitions."""

    if pivot not in PIVOTS:
        raise ValueError(f"unknown pivot {pivot!r}; expected one of {PIVOTS}")
    log_pf = inputs.log_pf_by_frame(bc, bh)
    if pivot == "slow-N":
        return _curves(inputs, peptide_map, log_pf, inputs.reference_weights)[pivot]

    legacy_log_pf, fast_log_pf = _scalar_log_pf(log_pf, inputs.reference_weights)
    effective_log_pf = legacy_log_pf if pivot == "legacy" else fast_log_pf
    # Preserve the original lock's division arithmetic so legacy is bit-for-bit.
    residue_uptake = 1.0 - np.exp(
        -inputs.timepoints[:, None]
        * inputs.k_ints[None, :]
        / np.exp(effective_log_pf)[None, :]
    )
    return (residue_uptake @ inputs.mapping.T).T


def _legacy_before_refactor(inputs, bc: float, bh: float) -> np.ndarray:
    """Frozen pre-refactor expression, retained only for the bit-for-bit regression guard."""

    mean_log_pf = inputs.log_pf_by_frame(bc, bh) @ inputs.reference_weights
    pf = np.exp(mean_log_pf)
    residue_uptake = 1.0 - np.exp(
        -inputs.timepoints[:, None] * inputs.k_ints[None, :] / pf[None, :]
    )
    return (residue_uptake @ inputs.mapping.T).T


def _calibration_mse(
    bc: float,
    bh: float,
    ensembles: list[tuple[object, object]],
    pivot: str,
    *,
    exclude_peptide1: bool = True,
) -> float:
    """Combined uptake MSE, normally over peptides 2--14 and all timepoints."""

    total = 0.0
    count = 0
    for inputs, peptide_map in ensembles:
        predicted = _predicted_peptide_uptake(inputs, peptide_map, bc, bh, pivot)
        keep = np.ones(predicted.shape[0], dtype=bool)
        if exclude_peptide1:
            keep[PEPTIDE1_INDEX] = False
        residual = predicted[keep] - inputs.observed_uptake[keep]
        total += float(np.sum(residual**2))
        count += residual.size
    return total / count


def _fit_shared_coefficients(ensembles, starts, pivot: str) -> dict:
    best = None
    for start in starts:
        result = minimize(
            lambda theta: _calibration_mse(theta[0], theta[1], ensembles, pivot),
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(0.0, None), (0.0, None)],
        )
        candidate = {
            "bc": float(result.x[0]),
            "bh": float(result.x[1]),
            "mse": float(result.fun),
            "success": bool(result.success),
            "start": [float(start[0]), float(start[1])],
        }
        if best is None or candidate["mse"] < best["mse"]:
            best = candidate
    return best


def _fit_scaled_published(ensembles, pivot: str) -> dict:
    result = minimize_scalar(
        lambda scale: _calibration_mse(
            common.PUBLISHED_BC * scale, common.PUBLISHED_BH * scale, ensembles, pivot
        ),
        bounds=SCALE_BOUNDS,
        method="bounded",
    )
    scale = float(result.x)
    if not SCALE_BOUNDS[0] < scale < SCALE_BOUNDS[1]:
        raise AssertionError(f"{pivot}: scaled-published optimum is not interior")
    return {
        "scale": scale,
        "bc": common.PUBLISHED_BC * scale,
        "bh": common.PUBLISHED_BH * scale,
        "mse": float(result.fun),
    }


def _coefficient_profile(ensembles, bc_grid, bh_grid) -> list[dict]:
    rows = []
    for bc in bc_grid:
        for bh in bh_grid:
            row = {"bc": float(bc), "bh": float(bh)}
            for pivot in PIVOTS:
                row[f"{pivot}_combined_mse"] = _calibration_mse(
                    float(bc), float(bh), ensembles, pivot
                )
            rows.append(row)
    return rows


def _angle_degrees(left: dict, right: dict) -> float:
    a = np.asarray([left["bc"], left["bh"]])
    b = np.asarray([right["bc"], right["bh"]])
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _absorption_readout(optima: dict, scaled: dict, ensembles) -> dict:
    pairs = {}
    for left, right in combinations(PIVOTS, 2):
        left_norm = float(np.hypot(optima[left]["bc"], optima[left]["bh"]))
        right_norm = float(np.hypot(optima[right]["bc"], optima[right]["bh"]))
        pairs[f"{left}_to_{right}"] = {
            "angle_degrees": _angle_degrees(optima[left], optima[right]),
            "optimum_norm_ratio_right_over_left": right_norm / left_norm,
        }

    multipliers = {}
    for pivot, optimum in optima.items():
        values = []
        for inputs, _ in ensembles:
            log_pf = inputs.log_pf_by_frame(optimum["bc"], optimum["bh"])
            mean = log_pf @ inputs.reference_weights
            half_variance = 0.5 * np.sum(
                inputs.reference_weights[None, :] * (log_pf - mean[:, None]) ** 2, axis=1
            )
            values.extend(np.exp(half_variance).tolist())
        multipliers[pivot] = {
            "mean_exp_half_variance": float(np.mean(values)),
            "median_exp_half_variance": float(np.median(values)),
        }

    return {
        "pairwise_optimum_geometry": pairs,
        "predicted_rate_multiplier": multipliers,
        "direction_vs_magnitude": {
            pivot: {
                "constrained_mse": optima[pivot]["mse"],
                "scaled_published_mse": scaled[pivot]["mse"],
                "absolute_mse_gap": scaled[pivot]["mse"] - optima[pivot]["mse"],
                "ratio": scaled[pivot]["mse"] / optima[pivot]["mse"],
            }
            for pivot in PIVOTS
        },
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_list = [common.load_ensemble_inputs(name) for name in common.ENSEMBLES]
    ensembles = [(inputs, _peptide_map(inputs)) for inputs in inputs_list]

    legacy_regression = {}
    for inputs, peptide_map in ensembles:
        old = _legacy_before_refactor(inputs, common.PUBLISHED_BC, common.PUBLISHED_BH)
        new = _predicted_peptide_uptake(
            inputs, peptide_map, common.PUBLISHED_BC, common.PUBLISHED_BH, "legacy"
        )
        if not np.array_equal(old, new):
            raise AssertionError(f"{inputs.ensemble}: legacy refactor is not bit-for-bit")
        legacy_regression[inputs.ensemble] = float(np.max(np.abs(old - new)))

    msass = [pair for pair in ensembles if pair[0].ensemble == "AF2_MSAss"]
    litmus_mse = _calibration_mse(
        common.PUBLISHED_BC,
        common.PUBLISHED_BH,
        msass,
        "legacy",
        exclude_peptide1=False,
    )
    expected_litmus_mse = 0.3701202**2
    if not np.isclose(litmus_mse, expected_litmus_mse, rtol=2e-6, atol=1e-10):
        raise AssertionError(f"litmus cross-check failed: {litmus_mse} != {expected_litmus_mse}")

    jensen_violations = 0
    degenerate_max_abs = 0.0
    for inputs, peptide_map in ensembles:
        log_pf = inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)
        legacy_pf, fast_pf = _scalar_log_pf(log_pf, inputs.reference_weights)
        jensen_violations += int(np.sum(fast_pf > legacy_pf + JENSEN_GUARD_TOL))
        one_frame = np.zeros(inputs.n_frames)
        one_frame[0] = 1.0
        curves = _curves(inputs, peptide_map, log_pf, one_frame)
        degenerate_max_abs = max(
            degenerate_max_abs,
            *(float(np.max(np.abs(curves[pivot] - curves["legacy"]))) for pivot in PIVOTS[1:]),
        )
    if jensen_violations:
        raise AssertionError("Jensen ordering failed")
    if degenerate_max_abs > 2e-15:
        raise AssertionError("degenerate-frame pivot equality failed")

    optima = {pivot: _fit_shared_coefficients(ensembles, STARTS, pivot) for pivot in PIVOTS}
    scaled = {pivot: _fit_scaled_published(ensembles, pivot) for pivot in PIVOTS}
    published_mse = {
        pivot: _calibration_mse(common.PUBLISHED_BC, common.PUBLISHED_BH, ensembles, pivot)
        for pivot in PIVOTS
    }
    per_ensemble = {
        inputs.ensemble: {
            pivot: _fit_shared_coefficients([(inputs, peptide_map)], STARTS, pivot)
            for pivot in PIVOTS
        }
        for inputs, peptide_map in ensembles
    }

    frozen_by_pivot = {
        pivot: {
            "published": {"bc": common.PUBLISHED_BC, "bh": common.PUBLISHED_BH},
            "constrained_optimum": {"bc": optima[pivot]["bc"], "bh": optima[pivot]["bh"]},
            "scaled_published": {"bc": scaled[pivot]["bc"], "bh": scaled[pivot]["bh"]},
        }
        for pivot in PIVOTS
    }
    boundaries = {
        pivot: optima[pivot]["bc"] <= 1e-6 or optima[pivot]["bh"] <= 1e-6 for pivot in PIVOTS
    }

    if args.smoke:
        bc_grid, bh_grid = np.linspace(0.0, 1.0, 11), np.linspace(0.0, 3.0, 7)
    else:
        bc_grid, bh_grid = np.linspace(0.0, 1.0, 41), np.linspace(0.0, 4.0, 41)
    profile = _coefficient_profile(ensembles, bc_grid, bh_grid)

    payload = {
        "description": "per-pivot shared non-negative BV coefficient lock at w_NMR, peptide 1 excluded",
        "semantics": "per_pivot: legacy=average_first, fast=rate_average, slow-N=frame_mixture",
        "frozen_settings": frozen_by_pivot["legacy"],
        "frozen_settings_by_pivot": frozen_by_pivot,
        "constrained_optimum_fit": optima["legacy"],
        "scaled_published_fit": scaled["legacy"],
        "published_calibration_mse": published_mse["legacy"],
        "fits_by_pivot": {
            pivot: {
                "constrained_optimum": optima[pivot],
                "scaled_published": scaled[pivot],
                "published_calibration_mse": published_mse[pivot],
            }
            for pivot in PIVOTS
        },
        "per_ensemble_optima": per_ensemble,
        "absorption_readout": _absorption_readout(optima, scaled, ensembles),
        "boundary_solution": boundaries["legacy"],
        "boundary_solution_by_pivot": boundaries,
        "boundary_note": (
            "Boundary optima reflect the AF2 hydrogen-bond geometry limitation documented in "
            "hdx_moprp_pivot_calibration.md §13.1: predicted structures provide little trustworthy "
            "acceptor-channel signal; they are not evidence of general BV model inadequacy."
        ),
        "verification": {
            "legacy_bitwise_max_abs_by_ensemble": legacy_regression,
            "litmus_legacy_all14_mse": litmus_mse,
            "litmus_expected_rmse": 0.3701202,
            "jensen_violations": jensen_violations,
            "degenerate_weight_curve_max_abs": degenerate_max_abs,
            "scaled_published_bounds": list(SCALE_BOUNDS),
            "all_scaled_published_interior": True,
        },
        "n_frames": {inputs.ensemble: inputs.n_frames for inputs in inputs_list},
        "input_hashes": common.input_hashes(),
    }
    (args.output_dir / "coefficient_lock.json").write_text(json.dumps(payload, indent=2) + "\n")

    fieldnames = ["bc", "bh", *(f"{pivot}_combined_mse" for pivot in PIVOTS)]
    with (args.output_dir / "coefficient_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(profile)

    for pivot in PIVOTS:
        optimum, scale = optima[pivot], scaled[pivot]
        print(
            f"{pivot}: optimum (bc={optimum['bc']:.6g}, bh={optimum['bh']:.6g}) "
            f"MSE={optimum['mse']:.8g}; scaled published lambda={scale['scale']:.6g} "
            f"MSE={scale['mse']:.8g}"
        )
    print(f"wrote {args.output_dir / 'coefficient_lock.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_recovery_coefficient_lock",
    )
    parser.add_argument("--smoke", action="store_true", help="coarse coefficient-profile grid")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
