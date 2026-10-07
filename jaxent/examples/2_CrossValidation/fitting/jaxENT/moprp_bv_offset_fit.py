#!/usr/bin/env python3
"""Peptide-level BV coefficient fit with a free global ln(Pf) offset.

The plain coefficient lock fits ``ln Pf = bc*Nc + bh*Nh``.  Because BV over-protects MoPrP by a
roughly constant amount, bh is driven to 0 to shed protection.  Here ``ln Pf = bc*Nc + bh*Nh + c``
(c free, shared across ensembles) so the global scale is absorbed by c and bh is only judged on
shape.  Legacy (average-first) pivot, peptides 2--14, w_NMR, shared across both ensembles.

One run per (MOPRP_STRUCTURE, MOPRP_FEATURES_SUFFIX) feature set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import _moprp_recovery_common as common
from moprp_coefficient_lock import _calibration_mse
from moprp_pivot_litmus import _peptide_map

PEPTIDE1_INDEX = common.PEPTIDE1_INDEX
STARTS = ((0.2, 0.5, -2.0), (0.1, 0.1, 0.0), (0.35, 2.0, -5.0), (0.05, 1.0, -1.0))
BH_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def _mse(bc, bh, c, ensembles) -> float:
    total, count = 0.0, 0
    for inputs, peptide_map in ensembles:
        mean_log_pf = inputs.log_pf_by_frame(bc, bh) @ inputs.reference_weights + c
        uptake = 1.0 - np.exp(
            -inputs.timepoints[:, None] * inputs.k_ints[None, :] / np.exp(mean_log_pf)[None, :]
        )
        predicted = (uptake @ inputs.mapping.T).T
        keep = np.ones(predicted.shape[0], dtype=bool)
        keep[PEPTIDE1_INDEX] = False
        residual = predicted[keep] - inputs.observed_uptake[keep]
        total += float(np.sum(residual**2))
        count += residual.size
    return total / count


def _fit(ensembles, *, fix_bh: float | None = None, fix_c: float | None = None) -> dict:
    """Minimise over (bc, bh, c); optionally fix bh or c.  Multi-start L-BFGS-B."""

    def unpack(theta):
        it = iter(theta)
        bc = next(it)
        bh = fix_bh if fix_bh is not None else next(it)
        c = fix_c if fix_c is not None else next(it)
        return bc, bh, c

    bounds = [(0.0, None)]
    if fix_bh is None:
        bounds.append((0.0, None))
    if fix_c is None:
        bounds.append((-15.0, 15.0))
    best = None
    for start in STARTS:
        x0 = [start[0]] + ([start[1]] if fix_bh is None else []) + ([start[2]] if fix_c is None else [])
        res = minimize(lambda t: _mse(*unpack(t), ensembles), x0=x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best["mse"]:
            bc, bh, c = unpack(res.x)
            best = {"bc": float(bc), "bh": float(bh), "c": float(c), "mse": float(res.fun), "success": bool(res.success)}
    return best


def run(args) -> None:
    ensembles = []
    for name in common.ENSEMBLES:
        inputs = common.load_ensemble_inputs(name, args.rate_source)
        ensembles.append((inputs, _peptide_map(inputs)))

    # c = 0 must reproduce the plain lock objective
    ref = _calibration_mse(0.2, 0.7, ensembles, "legacy")
    if not np.isclose(ref, _mse(0.2, 0.7, 0.0, ensembles), rtol=1e-10):
        raise AssertionError("offset objective does not reduce to the coefficient-lock objective at c=0")

    full = _fit(ensembles)
    no_offset = _fit(ensembles, fix_c=0.0)
    bh0_offset = _fit(ensembles, fix_bh=0.0)
    profile = [{"bh": bh, **_fit(ensembles, fix_bh=bh)} for bh in BH_GRID]
    per_ensemble = {
        inputs.ensemble: {"free": _fit([(inputs, pm)]), "bh0": _fit([(inputs, pm)], fix_bh=0.0)}
        for inputs, pm in ensembles
    }
    payload = {
        "structure": common.STRUCTURE,
        "features_dir": str(common.FEATURES_V2),
        "rate_source": args.rate_source,
        "pivot": "legacy",
        "model": "ln Pf = bc*Nc + bh*Nh + c",
        "no_offset_c0": no_offset,
        "free_bc_bh_c": full,
        "bh0_with_offset": bh0_offset,
        "delta_mse_bh_free_vs_bh0_with_offset": bh0_offset["mse"] - full["mse"],
        "bh_profile_with_offset": profile,
        "per_ensemble": per_ensemble,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    run(p.parse_args())
