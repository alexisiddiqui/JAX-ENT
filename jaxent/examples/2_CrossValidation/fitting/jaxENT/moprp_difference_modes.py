#!/usr/bin/env python3
"""Stage G: the difference modes as a linear model (residual = target - unweighted shape).

The discriminative signal is the population shift `corr(C_target) - corr(C_unweighted)` — the
"difference modes".  This runner (1) characterises how concentrated that difference is, and (2)
tests whether it is predictable from population-free features (ANM cross-correlation, sequence
separation, spatial distance, unweighted correlation) with leave-one-ensemble-out transfer.

If the difference residual is predictable non-circularly, projecting data onto it is the
population-sensitive operation.  Predicting the target correlation directly (common shape) is the
fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import jax.numpy as jnp
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _moprp_recovery_common as common
import moprp_noncircular_recovery as ND
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis.state_population import peptide_logpf_covariance
from jaxent.src.analysis.pf_variance import weighted_population_covariance

STRUCTURE = common.BASE / "data/_MoPrP/MoPrP_max_plddt_4334.pdb"
COEFFICIENT = "scaled_published"
ALPHAS = np.logspace(-4, 2, 13)


def _structure_coords_anmcorr(feature_ids):
    u = mda.Universe(str(STRUCTURE))
    ca = u.select_atoms("name CA")
    r2i = {int(r): i for i, r in enumerate(ca.resids)}
    idx = np.array([r2i[int(r)] for r in feature_ids])
    from jaxent.src.analysis import elastic_network as en
    anm = en.anm_covariance(ca.positions.astype(float), cutoff=24.0)[np.ix_(idx, idx)]
    return ca.positions.astype(float)[idx], cc.to_correlation(anm)


def _pair_frame(residue_ids, coords, anm_corr, unw_corr, target_corr):
    n = len(residue_ids)
    iu = np.triu_indices(n, 1)
    import pandas as pd
    seqsep = np.abs(residue_ids[iu[0]] - residue_ids[iu[1]]).astype(float)
    dist = np.linalg.norm(coords[iu[0]] - coords[iu[1]], axis=1)
    X = pd.DataFrame({
        "anm_xcorr": anm_corr[iu],
        "unw_xcorr": unw_corr[iu],
        "log_seqsep": np.log(seqsep),
        "spatial_dist": dist,
    })
    y_diff = (target_corr - unw_corr)[iu]   # the difference / discriminative residual
    y_common = target_corr[iu]              # fallback: the target shape itself
    return X, y_diff, y_common, iu, n


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    bc, bh = lock["frozen_settings"][COEFFICIENT]["bc"], lock["frozen_settings"][COEFFICIENT]["bh"]

    data = {}
    for ens in common.ENSEMBLES:
        e = common.load_ensemble_inputs(ens)
        lp = jnp.asarray(e.log_pf_by_frame(bc, bh))
        uniform = jnp.full(e.n_frames, 1.0 / e.n_frames)
        C_unw = np.asarray(weighted_population_covariance(lp, uniform))
        C_true = np.asarray(weighted_population_covariance(lp, jnp.asarray(e.reference_weights)))
        coords, anm_corr = _structure_coords_anmcorr(e.feature_residue_ids)
        X, y_diff, y_common, iu, n = _pair_frame(
            e.feature_residue_ids, coords, anm_corr, cc.to_correlation(C_unw), cc.to_correlation(C_true))
        # difference-mode concentration (peptide level, more interpretable)
        keep = np.ones(e.mapping.shape[0], bool); keep[0] = False
        Mk = jnp.asarray(e.mapping[keep])
        Rt = cc.to_correlation(np.asarray(peptide_logpf_covariance(lp, Mk, jnp.asarray(e.reference_weights))))
        Ru = cc.to_correlation(np.asarray(peptide_logpf_covariance(lp, Mk, uniform)))
        diff_pep = Rt - Ru
        evals = np.abs(np.linalg.eigvalsh(diff_pep))[::-1]
        top_frac = float(evals[0] / evals.sum())
        data[ens] = dict(X=X, y_diff=y_diff, y_common=y_common,
                         diff_norm=float(np.linalg.norm(diff_pep)), top_mode_frac=top_frac)

    ea, eb = list(common.ENSEMBLES)
    A, B = data[ea], data[eb]
    rows = []
    for target_name in ("difference", "common"):
        key = "y_diff" if target_name == "difference" else "y_common"
        for direction, (src, dst) in [(f"{ea}->{eb}", (A, B)), (f"{eb}->{ea}", (B, A))]:
            m = _ridge().fit(src["X"].values, src[key])
            pred = m.predict(dst["X"].values)
            r2 = float(r2_score(dst[key], pred))
            # in-place self-fit R2 (fit quality, not transfer)
            self_r2 = float(r2_score(src[key], m.predict(src["X"].values)))
            coefs = dict(zip(src["X"].columns, _ridge().fit(src["X"].values, src[key]).steps[-1][1].coef_))
            rows.append(dict(target=target_name, direction=direction, transfer_r2=r2, self_r2=self_r2,
                             **{f"coef_{k}": v for k, v in coefs.items()}))

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "difference_modes.csv", index=False)
    print("difference-mode concentration (peptide corr, top eigenmode fraction of |Δ|):")
    for ens in common.ENSEMBLES:
        print("   {:12s} ||Δcorr||={:.2f}  top-mode={:.0%}".format(ens, data[ens]["diff_norm"], data[ens]["top_mode_frac"]))
    print("\nlinear-model transfer R² (population-free features -> shape):")
    print(frame[["target", "direction", "self_r2", "transfer_r2"]].to_string(index=False))
    print("\ncoefficients (difference target):")
    print(frame[frame.target == "difference"][[c for c in frame.columns if c.startswith("coef_") or c == "direction"]].to_string(index=False))
    print(f"\nwrote {args.output_dir / 'difference_modes.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_difference_modes")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
