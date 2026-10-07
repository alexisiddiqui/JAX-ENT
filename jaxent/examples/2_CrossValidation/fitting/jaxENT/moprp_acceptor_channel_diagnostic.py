#!/usr/bin/env python3
"""Why does the BV acceptor (bh) channel carry no signal on MoPrP?

One run per (MOPRP_STRUCTURE, MOPRP_FEATURES_SUFFIX) feature set.  Reports

* contact statistics of the amide-H -> O channel (heavy channel for scale);
* peptide-level MSE profile over bh (bc re-optimised at each bh) and over bc at fixed bh;
* residue-level regression of median.pfact ln(Pf) on the w_NMR-mean contact counts
  (heavy only / acceptor only / both, non-negative, with and without intercept).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, nnls
from scipy.stats import pearsonr, spearmanr

import _moprp_recovery_common as common
from moprp_coefficient_lock import _calibration_mse
from moprp_pivot_litmus import _peptide_map

PFACT = common.MOPRP / "median.pfact"  # one-based moprp.seq position, ln(Pf), -1 = absent
BH_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
BC_AT_BH0 = None


def _contact_stats(inputs) -> dict:
    w = inputs.reference_weights
    out = {}
    for name, arr in (("heavy", inputs.heavy_contacts), ("acceptor", inputs.acceptor_contacts)):
        mean = arr @ w
        var = (w[None, :] * (arr - mean[:, None]) ** 2).sum(axis=1)
        out[name] = {
            "mean_count_per_residue_frame": float(arr.mean()),
            "frac_residue_frames_nonzero": float((arr > 0).mean()),
            "frac_residues_any_contact": float((arr.max(axis=1) > 0).mean()),
            "frac_residues_wmean_ge_0.5": float((mean >= 0.5).mean()),
            "mean_wnmr_count": float(mean.mean()),
            "mean_wnmr_std_across_frames": float(np.sqrt(var).mean()),
            "frac_residues_zero_variance": float((var < 1e-12).mean()),
        }
    mean_h, mean_a = inputs.heavy_contacts @ w, inputs.acceptor_contacts @ w
    out["residue_corr_heavy_vs_acceptor"] = float(pearsonr(mean_h, mean_a).statistic)
    return out


def _bc_opt(ensembles, bh: float, pivot: str = "legacy") -> tuple[float, float]:
    res = minimize_scalar(
        lambda bc: _calibration_mse(bc, bh, ensembles, pivot), bounds=(0.0, 2.0), method="bounded"
    )
    return float(res.x), float(res.fun)


def _residue_regression(inputs) -> dict:
    pf = {int(p): float(v) for p, v in (line.split() for line in PFACT.read_text().splitlines())}
    ids = inputs.feature_residue_ids
    y = np.array([pf.get(int(r), -1.0) for r in ids])
    ok = y > 0
    w = inputs.reference_weights
    h = (inputs.heavy_contacts @ w)[ok]
    a = (inputs.acceptor_contacts @ w)[ok]
    y = y[ok]

    def fit(cols, intercept):
        X = np.column_stack(cols + ([np.ones_like(y)] if intercept else []))
        coef, _ = nnls(X, y)
        pred = X @ coef
        ss = float(np.sum((y - pred) ** 2))
        tot = float(np.sum((y - (y.mean() if intercept else 0.0)) ** 2))
        return {"coef": coef.tolist(), "rmse": float(np.sqrt(ss / y.size)), "r2": 1.0 - ss / tot}

    return {
        "n_residues": int(y.size),
        "spearman_heavy_vs_lnPf": float(spearmanr(h, y).statistic),
        "spearman_acceptor_vs_lnPf": float(spearmanr(a, y).statistic),
        "no_intercept": {
            "heavy": fit([h], False),
            "acceptor": fit([a], False),
            "both": fit([h, a], False),
        },
        "with_intercept": {
            "heavy": fit([h], True),
            "acceptor": fit([a], True),
            "both": fit([h, a], True),
        },
    }


def run(args) -> None:
    ens = []
    stats, regression = {}, {}
    for name in common.ENSEMBLES:
        inputs = common.load_ensemble_inputs(name, args.rate_source)
        ens.append((inputs, _peptide_map(inputs)))
        stats[name] = _contact_stats(inputs)
        regression[name] = _residue_regression(inputs)

    bh_profile = []
    for bh in BH_GRID:
        bc, mse = _bc_opt(ens, bh)
        bh_profile.append({"bh": bh, "bc_opt": bc, "mse": mse})
    bc0 = bh_profile[0]["bc_opt"]
    bh_at_fixed_bc = [
        {"bh": bh, "mse": _calibration_mse(bc0, bh, ens, "legacy")} for bh in BH_GRID
    ]
    payload = {
        "structure": common.STRUCTURE,
        "features_dir": str(common.FEATURES_V2),
        "rate_source": args.rate_source,
        "contact_stats": stats,
        "bh_profile_bc_reoptimised": bh_profile,
        "bh_profile_bc_fixed_at_bh0_opt": bh_at_fixed_bc,
        "residue_regression_median_pfact": regression,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    run(p.parse_args())
