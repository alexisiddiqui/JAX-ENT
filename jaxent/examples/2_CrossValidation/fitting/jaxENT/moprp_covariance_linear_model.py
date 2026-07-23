#!/usr/bin/env python3
"""Stage C, Part B: can a non-circular linear model rebuild the target covariance?

Predicts the target covariance information from **population-free** features only (unweighted
trajectory covariance, ANM/GNM elastic-network priors, intrinsic rates, contacts, amino-acid
identity, sequence position/separation, spatial distance) — never the target weights.

* Model 1 (marginal variance): predict the target log-variance profile (residue + peptide),
  separating population-free *pattern* from the *scale* the population sets.
* Model 2 (off-diagonal structure): predict the target correlation entries, scored by Mantel of
  the predicted-vs-true correlation matrix.

The real test is **leave-one-ensemble-out transfer** (train MSAss -> predict Filtered and vice
versa): if features that never see the target predict it across ensembles, the covariance target
is reconstructable without knowing the population — breaking the circularity.
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
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis import elastic_network as en
from jaxent.src.analysis.state_population import peptide_logpf_covariance
from jaxent.src.analysis.pf_variance import weighted_population_covariance, peptide_overlap_similarity

STRUCTURE = common.BASE / "data/_MoPrP/MoPrP_max_plddt_4334.pdb"
ANM_CUTOFF = 24.0   # best structural prior from Part A
GNM_CUTOFF = 8.0
COEFFICIENT = "scaled_published"
ALPHAS = np.logspace(-3, 3, 13)
AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]


def _structure(feature_ids):
    u = mda.Universe(str(STRUCTURE))
    ca = u.select_atoms("name CA")
    r2i = {int(r): i for i, r in enumerate(ca.resids)}
    r2name = {int(r): n for r, n in zip(ca.resids, ca.resnames)}
    idx = np.array([r2i[int(r)] for r in feature_ids])
    coords = ca.positions.astype(float)
    anm = en.anm_covariance(coords, cutoff=ANM_CUTOFF)[np.ix_(idx, idx)]
    gnm = en.gnm_covariance(coords, cutoff=GNM_CUTOFF)[np.ix_(idx, idx)]
    resnames = [r2name[int(r)] for r in feature_ids]
    return coords[idx], anm, gnm, resnames


def _residue_features(inputs, anm, gnm, resnames):
    lp = jnp.asarray(inputs.log_pf_by_frame(*(_coeff())))
    uniform = jnp.full(inputs.n_frames, 1.0 / inputs.n_frames)
    unw_res = np.asarray(weighted_population_covariance(lp, uniform))
    feats = {
        "unweighted_logvar": np.log(np.clip(np.diag(unw_res), 1e-12, None)),
        "anm_msf": np.log(np.clip(np.diag(anm), 1e-12, None)),
        "gnm_msf": np.log(np.clip(np.diag(gnm), 1e-12, None)),
        "log_kint": np.log(inputs.k_ints),
        "mean_heavy": inputs.heavy_contacts.mean(axis=1),
        "mean_acceptor": inputs.acceptor_contacts.mean(axis=1),
        "mean_logpf": np.asarray(inputs.log_pf_by_frame(*_coeff())).mean(axis=1),
        "position": (inputs.feature_residue_ids - inputs.feature_residue_ids.min())
        / np.ptp(inputs.feature_residue_ids),
    }
    for aa in AA:
        feats[f"aa_{aa}"] = np.array([1.0 if n == aa else 0.0 for n in resnames])
    import pandas as pd
    return pd.DataFrame(feats), unw_res


_LOCK = None
def _coeff():
    return _LOCK["frozen_settings"][COEFFICIENT]["bc"], _LOCK["frozen_settings"][COEFFICIENT]["bh"]


def _target_residue_logvar(inputs):
    lp = jnp.asarray(inputs.log_pf_by_frame(*_coeff()))
    wT = jnp.asarray(inputs.reference_weights)
    C = np.asarray(weighted_population_covariance(lp, wT))
    return np.log(np.clip(np.diag(C), 1e-12, None)), C


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


FEATURE_GROUPS = {
    "unweighted": ["unweighted_logvar"],
    "elastic_network": ["anm_msf", "gnm_msf"],
    "sequence": ["log_kint", "mean_heavy", "mean_acceptor", "mean_logpf", "position"] + [f"aa_{a}" for a in AA],
}


def _fit_transfer(Xa, ya, Xb, yb, cols):
    """Fit on ensemble A, report transfer R2 predicting ensemble B (and A->A CV proxy)."""
    m = _ridge().fit(Xa[cols].values, ya)
    return float(r2_score(yb, m.predict(Xb[cols].values)))


def run(args: argparse.Namespace) -> None:
    global _LOCK
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _LOCK = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())

    data = {}
    for ens in common.ENSEMBLES:
        inp = common.load_ensemble_inputs(ens)
        coords, anm, gnm, resnames = _structure(inp.feature_residue_ids)
        X, unw_res = _residue_features(inp, anm, gnm, resnames)
        y, Ctar = _target_residue_logvar(inp)
        data[ens] = dict(inp=inp, X=X, y=y, anm=anm, gnm=gnm, coords=coords,
                         unw_res=unw_res, Ctar=Ctar)

    ens_a, ens_b = list(common.ENSEMBLES)
    A, B = data[ens_a], data[ens_b]
    all_cols = list(A["X"].columns)

    # ---- Model 1: marginal variance, transfer + attribution + pattern/scale ----
    rows = []
    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    for direction, (src, dst) in [(f"{ens_a}->{ens_b}", (A, B)), (f"{ens_b}->{ens_a}", (B, A))]:
        for gname, cols in {"full": all_cols, **FEATURE_GROUPS}.items():
            raw = _fit_transfer(src["X"], src["y"], dst["X"], dst["y"], cols)
            pattern = _fit_transfer(src["X"], z(src["y"]), dst["X"], z(dst["y"]), cols)
            rows.append({"model": "marginal", "direction": direction, "features": gname,
                         "transfer_r2_raw": raw, "transfer_r2_pattern": pattern})
        # shuffled-target null
        rng = np.random.default_rng(0)
        null = _fit_transfer(src["X"], rng.permutation(src["y"]), dst["X"], dst["y"], all_cols)
        rows.append({"model": "marginal", "direction": direction, "features": "shuffled_null",
                     "transfer_r2_raw": null, "transfer_r2_pattern": float("nan")})

    # ---- Model 2: off-diagonal correlation structure, transfer + Mantel ----
    def pair_frame(d):
        n = d["anm"].shape[0]
        iu = np.triu_indices(n, 1)
        seqsep = np.abs(d["inp"].feature_residue_ids[iu[0]] - d["inp"].feature_residue_ids[iu[1]])
        dist = np.linalg.norm(d["coords"][iu[0]] - d["coords"][iu[1]], axis=1)
        import pandas as pd
        X = pd.DataFrame({
            "anm_xcorr": cc.to_correlation(d["anm"])[iu],
            "gnm_xcorr": cc.to_correlation(d["gnm"])[iu],
            "unweighted_corr": cc.to_correlation(d["unw_res"])[iu],
            "seqsep": np.log(seqsep.astype(float)),
            "spatial_dist": dist,
        })
        y = cc.to_correlation(d["Ctar"])[iu]
        return X, y, iu, n
    Xa2, ya2, iu, n = pair_frame(A)
    Xb2, yb2, _, _ = pair_frame(B)
    for direction, (sx, sy, dx, dy, dd) in [
        (f"{ens_a}->{ens_b}", (Xa2, ya2, Xb2, yb2, B)),
        (f"{ens_b}->{ens_a}", (Xb2, yb2, Xa2, ya2, A)),
    ]:
        model = _ridge().fit(sx.values, sy)
        pred = model.predict(dx.values)
        r2 = float(r2_score(dy, pred))
        pred_mat = np.eye(n)
        pred_mat[iu] = pred; pred_mat[(iu[1], iu[0])] = pred
        mantel_r, mantel_p = cc.mantel_test(pred_mat, cc.to_correlation(dd["Ctar"]), permutations=args.permutations)
        rows.append({"model": "offdiag", "direction": direction, "features": "full",
                     "transfer_r2_raw": r2, "predicted_vs_true_mantel_r": mantel_r,
                     "predicted_vs_true_mantel_p": mantel_p})

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "linear_model_transfer.csv", index=False)
    print(frame.to_string(index=False))
    print(f"\nwrote {args.output_dir / 'linear_model_transfer.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_covariance_linear_model")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    parser.add_argument("--permutations", type=int, default=999)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
