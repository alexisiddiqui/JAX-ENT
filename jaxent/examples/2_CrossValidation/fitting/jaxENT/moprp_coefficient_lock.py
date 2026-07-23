#!/usr/bin/env python3
"""Stage A3: lock a shared Best-Vendruscolo coefficient pair for the recovery experiment.

Fit one shared, non-negative ``(Bc, Bh)`` across *both* MoPrP ensembles at the fixed NMR
reference weights ``w_NMR``, using average-first semantics, all 14 peptides but excluding
peptide 1 from calibration, the trim-one exPfact map, canonical rates, and hard-count
features.  Two coefficient settings are frozen for the downstream oracle and reweighting:

* ``published`` = (0.35, 2.0);
* ``constrained_optimum`` = the shared non-negative optimum found here.

A boundary optimum (e.g. Bh -> 0) is reported as *model inadequacy*, not a physical estimate.
No intercept, no timepoint-specific coefficients, no linear forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import _moprp_recovery_common as common

PEPTIDE1_INDEX = common.PEPTIDE1_INDEX


def _predicted_peptide_uptake(inputs, bc: float, bh: float) -> np.ndarray:
    """Average-first peptide mean uptake at w_NMR -> (P, T)."""

    log_pf = inputs.log_pf_by_frame(bc, bh)  # (R, F)
    mean_log_pf = log_pf @ inputs.reference_weights  # (R,)
    pf = np.exp(mean_log_pf)
    # residue uptake (T, R): 1 - exp(-t * k / pf)
    residue_uptake = 1.0 - np.exp(
        -inputs.timepoints[:, None] * inputs.k_ints[None, :] / pf[None, :]
    )
    peptide_uptake = residue_uptake @ inputs.mapping.T  # (T, P)
    return peptide_uptake.T  # (P, T)


def _calibration_mse(bc: float, bh: float, ensembles: list) -> float:
    """Combined mean-uptake MSE over both ensembles, peptides 2..14, all timepoints."""

    total = 0.0
    count = 0
    for inputs in ensembles:
        predicted = _predicted_peptide_uptake(inputs, bc, bh)  # (P, T)
        keep = np.ones(predicted.shape[0], dtype=bool)
        keep[PEPTIDE1_INDEX] = False
        residual = predicted[keep] - inputs.observed_uptake[keep]
        total += float(np.sum(residual**2))
        count += residual.size
    return total / count


def _fit_shared_coefficients(ensembles: list, starts: list[tuple[float, float]]) -> dict:
    best = None
    for start in starts:
        result = minimize(
            lambda theta: _calibration_mse(theta[0], theta[1], ensembles),
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


def _fit_scaled_published(ensembles: list) -> dict:
    """Scale the published (0.35, 2.0) direction by one scalar to match the target mean scale.

    The published coefficients are calibrated for a standard condition; scaling the whole
    direction by ``s`` preserves the published Bc:Bh ratio (and keeps Bh) while adjusting the
    overall protection magnitude to the experimental condition.
    """

    result = minimize_scalar(
        lambda s: _calibration_mse(common.PUBLISHED_BC * s, common.PUBLISHED_BH * s, ensembles),
        bounds=(1e-3, 3.0),
        method="bounded",
    )
    scale = float(result.x)
    return {
        "scale": scale,
        "bc": common.PUBLISHED_BC * scale,
        "bh": common.PUBLISHED_BH * scale,
        "mse": float(result.fun),
    }


def _coefficient_profile(ensembles: list, bc_grid: np.ndarray, bh_grid: np.ndarray) -> list[dict]:
    rows = []
    for bc in bc_grid:
        for bh in bh_grid:
            rows.append(
                {"bc": float(bc), "bh": float(bh), "mse": _calibration_mse(float(bc), float(bh), ensembles)}
            )
    return rows


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensembles = [common.load_ensemble_inputs(name) for name in common.ENSEMBLES]

    starts = [(0.35, 2.0), (0.5, 0.5), (0.1, 0.1), (1.0, 0.0)]
    optimum = _fit_shared_coefficients(ensembles, starts)
    scaled = _fit_scaled_published(ensembles)

    if args.smoke:
        bc_grid = np.linspace(0.0, 1.0, 11)
        bh_grid = np.linspace(0.0, 3.0, 7)
    else:
        bc_grid = np.linspace(0.0, 1.0, 41)
        bh_grid = np.linspace(0.0, 4.0, 41)
    profile = _coefficient_profile(ensembles, bc_grid, bh_grid)

    boundary = optimum["bc"] <= 1e-6 or optimum["bh"] <= 1e-6
    settings = {
        "published": {"bc": common.PUBLISHED_BC, "bh": common.PUBLISHED_BH},
        "constrained_optimum": {"bc": optimum["bc"], "bh": optimum["bh"]},
        "scaled_published": {"bc": scaled["bc"], "bh": scaled["bh"]},
    }
    published_mse = _calibration_mse(common.PUBLISHED_BC, common.PUBLISHED_BH, ensembles)

    payload = {
        "description": "shared non-negative BV coefficient lock at w_NMR, peptide 1 excluded",
        "semantics": "average_first",
        "frozen_settings": settings,
        "constrained_optimum_fit": optimum,
        "scaled_published_fit": scaled,
        "published_calibration_mse": published_mse,
        "boundary_solution": boundary,
        "boundary_note": (
            "constrained optimum sits on a coefficient boundary; reported as model inadequacy, "
            "not a physical coefficient estimate"
        )
        if boundary
        else "interior optimum",
        "n_frames": {inputs.ensemble: inputs.n_frames for inputs in ensembles},
        "input_hashes": common.input_hashes(),
    }
    (args.output_dir / "coefficient_lock.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (args.output_dir / "coefficient_profile.csv").open("w") as handle:
        handle.write("bc,bh,combined_mse\n")
        for row in profile:
            handle.write(f"{row['bc']:.6g},{row['bh']:.6g},{row['mse']:.8g}\n")

    print(f"published (0.35, 2.0) MSE = {published_mse:.6g}")
    print(
        f"constrained optimum (bc={optimum['bc']:.4g}, bh={optimum['bh']:.4g}) "
        f"MSE = {optimum['mse']:.6g}"
        + ("  [BOUNDARY -> model inadequacy]" if boundary else "")
    )
    print(
        f"scaled published  (s={scaled['scale']:.4g}: bc={scaled['bc']:.4g}, bh={scaled['bh']:.4g}) "
        f"MSE = {scaled['mse']:.6g}"
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
