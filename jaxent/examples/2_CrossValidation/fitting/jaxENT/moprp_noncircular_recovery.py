#!/usr/bin/env python3
"""Stage D: does a population-free covariance target still recover the known population?

Builds `C_approx = D^{1/2} R D^{1/2}` from population-free parts — correlation *structure* R from
the unweighted trajectory or a single-structure ANM (Stage C), and *scale* D from the mean-only
fit (`w_baseline`, which uses only the target uptake, no population) or a swept global scalar —
then reweights against `C_approx` with the Stage B symmetric projected loss and measures recovery.

Compared against (i) the mean-only baseline and (ii) the true target `C(w_NMR)`.  If the
non-circular target recovers the population (robustly across scale), the circularity is broken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import MDAnalysis as mda
import jax
import jax.numpy as jnp

import _moprp_recovery_common as common
import moprp_covariance_recovery as R
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis import elastic_network as en
from jaxent.src.analysis.state_population import peptide_logpf_covariance, state_populations, strict_recovery_percent, FULL_STATE_SUPPORT
from jaxent.src.analysis.pf_variance import weighted_population_covariance, overlap_projection, kl_to_uniform

STRUCTURE = common.BASE / "data/_MoPrP/MoPrP_max_plddt_4334.pdb"
ANM_CUTOFF = 24.0
COEFFICIENT = "scaled_published"
COORDINATE = "logpf_projected"
GAMMA, ETA = 1.0, 0.01
STEPS, LR, N_START = 2000, 0.03, 3
SCALE_SWEEP = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _anm_peptide_structure(inputs, mapping_keep):
    u = mda.Universe(str(STRUCTURE))
    ca = u.select_atoms("name CA")
    r2i = {int(r): i for i, r in enumerate(ca.resids)}
    idx = np.array([r2i[int(r)] for r in inputs.feature_residue_ids])
    C_res = en.anm_covariance(ca.positions.astype(float), cutoff=ANM_CUTOFF)[np.ix_(idx, idx)]
    return mapping_keep @ C_res @ mapping_keep.T


def _optimize_recovery(log_pf, k, timepoints, mapping_keep, projection, obs_keep, mse_uniform,
                       inputs, target_cov, gamma):
    def loss(logits):
        w = jax.nn.softmax(logits)
        mean = R._mean_mse(R._predict_uptake(log_pf, k, timepoints, mapping_keep, w).T, obs_keep) / mse_uniform
        if gamma == 0.0:
            cov = 0.0
        else:
            cov = R._coordinate_distance(COORDINATE, log_pf, k, timepoints, mapping_keep, projection, w, target_cov)
        return mean + gamma * cov + ETA * kl_to_uniform(w)

    best = None
    for s in range(N_START):
        start = np.zeros(inputs.n_frames) if s == 0 else np.random.default_rng(s).normal(scale=0.01, size=inputs.n_frames)
        logits = R._optimize(loss, start, STEPS, LR)
        obj = float(loss(jnp.asarray(logits)))
        if best is None or obj < best[0]:
            best = (obj, np.asarray(jax.nn.softmax(jnp.asarray(logits))))
    w = best[1]
    rec = float(strict_recovery_percent(w, inputs.states, inputs.support, inputs.targets))
    pops = np.asarray(state_populations(w, inputs.states, inputs.support))
    decoy = float(sum(pops[FULL_STATE_SUPPORT.index(st)] for st in inputs.support if st not in TARGET_STATES))
    val_mse = float(R._mean_mse(R._predict_uptake(log_pf, k, timepoints, mapping_keep, jnp.asarray(w)).T, obs_keep))
    return rec, decoy, val_mse, w


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    bc, bh = lock["frozen_settings"][COEFFICIENT]["bc"], lock["frozen_settings"][COEFFICIENT]["bh"]

    rows, recon_rows = [], []
    for ensemble in common.ENSEMBLES:
        inp = common.load_ensemble_inputs(ensemble)
        log_pf = jnp.asarray(inp.log_pf_by_frame(bc, bh))
        k = jnp.asarray(inp.k_ints)
        timepoints = jnp.asarray(inp.timepoints)
        keep = np.ones(inp.mapping.shape[0], dtype=bool); keep[common.PEPTIDE1_INDEX] = False
        M_keep = jnp.asarray(inp.mapping[keep])
        obs_keep = jnp.asarray(inp.observed_uptake[keep])
        projection = overlap_projection(M_keep)
        uniform = jnp.full(inp.n_frames, 1.0 / inp.n_frames)
        w_nmr = jnp.asarray(inp.reference_weights)
        mse_uniform = float(R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, uniform).T, obs_keep))

        # true target + population-free structure sources (peptide level, kept peptides)
        C_true = np.asarray(peptide_logpf_covariance(log_pf, M_keep, w_nmr))
        C_unw = np.asarray(peptide_logpf_covariance(log_pf, M_keep, uniform))
        C_anm = np.asarray(_anm_peptide_structure(inp, np.asarray(M_keep)))

        # mean-only fit -> w_baseline -> mean-derived scale
        base_rec, base_decoy, base_mse, w_base = _optimize_recovery(
            log_pf, k, timepoints, M_keep, projection, obs_keep, mse_uniform, inp, None, 0.0)
        C_base = np.asarray(peptide_logpf_covariance(log_pf, M_keep, jnp.asarray(w_base)))
        mean_scale = np.clip(np.diag(C_base), 1e-12, None)

        rows.append(dict(ensemble=ensemble, target="baseline_mean_only", structure="-", scale="-",
                         recovery=base_rec, decoy=base_decoy, val_mse=base_mse))
        # true-target ceiling
        tr, td, tm, _ = _optimize_recovery(log_pf, k, timepoints, M_keep, projection, obs_keep, mse_uniform, inp, C_true, GAMMA)
        rows.append(dict(ensemble=ensemble, target="true_C_wNMR", structure="true", scale="true",
                         recovery=tr, decoy=td, val_mse=tm))

        structures = {"unweighted": C_unw, "anm": C_anm}
        for sname, S in structures.items():
            recon_rows.append(dict(ensemble=ensemble, structure=sname,
                                   **{f"struct_{k2}": v for k2, v in cc.covariance_metrics(S, C_true, permutations=199).items()}))
            # mean-derived scale
            C_mean = cc.rebuild_covariance(S, mean_scale)
            recon_rows.append(dict(ensemble=ensemble, structure=f"{sname}+mean_scale",
                                   **{f"struct_{k2}": v for k2, v in cc.covariance_metrics(C_mean, C_true, permutations=199).items()}))
            r, d, m, _ = _optimize_recovery(log_pf, k, timepoints, M_keep, projection, obs_keep, mse_uniform, inp, jnp.asarray(C_mean), GAMMA)
            rows.append(dict(ensemble=ensemble, target=f"{sname}+mean_scale", structure=sname, scale="mean_fit",
                             recovery=r, decoy=d, val_mse=m))
            # global-scalar scale sweep (variances = s * structure's own diagonal)
            own = np.clip(np.diag(S), 1e-12, None)
            for s in SCALE_SWEEP:
                C_s = cc.rebuild_covariance(S, s * own)
                r, d, m, _ = _optimize_recovery(log_pf, k, timepoints, M_keep, projection, obs_keep, mse_uniform, inp, jnp.asarray(C_s), GAMMA)
                rows.append(dict(ensemble=ensemble, target=f"{sname}+scale{s}", structure=sname, scale=f"x{s}",
                                 recovery=r, decoy=d, val_mse=m))

    import pandas as pd
    pd.DataFrame(rows).to_csv(args.output_dir / "noncircular_recovery.csv", index=False)
    pd.DataFrame(recon_rows).to_csv(args.output_dir / "noncircular_reconstruction.csv", index=False)
    frame = pd.DataFrame(rows)
    for ensemble in common.ENSEMBLES:
        print(f"== {ensemble}")
        for _, r in frame[frame.ensemble == ensemble].iterrows():
            print("   {:22s} recovery={:6.1f}%  decoy={:.3f}  val_mse={:.4f}".format(r.target, r.recovery, r.decoy, r.val_mse))
    print(f"wrote {args.output_dir / 'noncircular_recovery.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_noncircular_recovery")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
