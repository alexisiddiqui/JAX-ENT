#!/usr/bin/env python3
"""Stage E: automatic, population-free scale calibration from the mean-only fit.

Stage D showed a population-free covariance target recovers the known population if the scale is
native-consistent (small). This runner tests whether an *automatic* scale rule derived from the
mean-only fit `w_baseline` (which uses only the target uptake, no population) lands that window for
both ensembles with no per-ensemble tuning:

  R1  mean-fit diagonal:            D = diag(C(w_baseline))
  R2  trace-match (primary):        structure scaled so trace = trace(C(w_baseline))
  R3  trace-match, MaxEnt baseline: R2 with a stronger-eta (broader) mean fit
  R4  iterative self-consistent:    s <- trace(C(w_fit)), iterated (collapse-guarded)

Success = one rule puts *both* ensembles clearly above the mean-only baseline and within/near the
1.05x mean gate, automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import _moprp_recovery_common as common
import moprp_covariance_recovery as R
import moprp_noncircular_recovery as ND
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis.state_population import peptide_logpf_covariance
from jaxent.src.analysis.pf_variance import weighted_population_covariance, overlap_projection, kl_to_uniform

COEFFICIENT = "scaled_published"


def _mean_fit(log_pf, k, timepoints, M_keep, obs_keep, mse_uniform, n_frames, eta):
    def loss(logits):
        w = jax.nn.softmax(logits)
        return R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, w).T, obs_keep) / mse_uniform + eta * kl_to_uniform(w)

    best = None
    for s in range(2):
        start = np.zeros(n_frames) if s == 0 else np.random.default_rng(s).normal(scale=0.01, size=n_frames)
        logits = R._optimize(loss, start, ND.STEPS, ND.LR)
        obj = float(loss(jnp.asarray(logits)))
        if best is None or obj < best[0]:
            best = (obj, np.asarray(jax.nn.softmax(jnp.asarray(logits))))
    return best[1]


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    bc, bh = lock["frozen_settings"][COEFFICIENT]["bc"], lock["frozen_settings"][COEFFICIENT]["bh"]

    rows = []
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

        def cov(w):
            return np.asarray(peptide_logpf_covariance(log_pf, M_keep, jnp.asarray(w)))

        def recover(target_cov):
            rec, decoy, val_mse, w = ND._optimize_recovery(
                log_pf, k, timepoints, M_keep, projection, obs_keep, mse_uniform, inp,
                jnp.asarray(target_cov), ND.GAMMA)
            return rec, decoy, val_mse, w

        C_true = cov(w_nmr)
        structures = {"unweighted": cov(uniform), "anm": np.asarray(ND._anm_peptide_structure(inp, np.asarray(M_keep)))}

        # mean-only baseline via direct mean fit (population-free)
        from jaxent.src.analysis.state_population import strict_recovery_percent, state_populations, FULL_STATE_SUPPORT
        w_base = _mean_fit(log_pf, k, timepoints, M_keep, obs_keep, mse_uniform, inp.n_frames, ND.ETA)
        base_rec = float(strict_recovery_percent(w_base, inp.states, inp.support, inp.targets))
        base_pops = np.asarray(state_populations(w_base, inp.states, inp.support))
        base_decoy = float(sum(base_pops[FULL_STATE_SUPPORT.index(s)] for s in inp.support if s not in ("Folded", "PUF1", "PUF2")))
        base_mse = float(R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, jnp.asarray(w_base)).T, obs_keep))
        gate = 1.05 * base_mse

        C_base = cov(w_base); trace_base = float(np.trace(C_base))
        w_base_hi = _mean_fit(log_pf, k, timepoints, M_keep, obs_keep, mse_uniform, inp.n_frames, 1.0)
        trace_base_hi = float(np.trace(cov(w_base_hi)))

        rows.append(dict(ensemble=ensemble, rule="baseline_mean_only", structure="-",
                         recovery=base_rec, decoy=base_decoy, val_mse=base_mse, scale_ratio=np.nan, gate=base_mse <= gate))
        tr, td, tm, _ = recover(C_true)
        rows.append(dict(ensemble=ensemble, rule="true_C_wNMR", structure="true",
                         recovery=tr, decoy=td, val_mse=tm, scale_ratio=np.nan, gate=tm <= gate))

        for sname, S in structures.items():
            trace_S = float(np.trace(S))
            # R1 mean-fit diagonal
            C1 = cc.rebuild_covariance(S, np.clip(np.diag(C_base), 1e-12, None))
            # R2 trace-match to mean-fit baseline
            C2 = cc.rebuild_covariance(S, cc.trace_match_scale(S, trace_base))
            # R3 trace-match to MaxEnt(eta=1) baseline
            C3 = cc.rebuild_covariance(S, cc.trace_match_scale(S, trace_base_hi))
            for rule, Cx, ratio in [("R1_meanfit_diag", C1, np.nan),
                                    ("R2_tracematch", C2, trace_base / trace_S),
                                    ("R3_tracematch_maxent", C3, trace_base_hi / trace_S)]:
                rec, decoy, val_mse, _ = recover(Cx)
                rows.append(dict(ensemble=ensemble, rule=rule, structure=sname,
                                 recovery=rec, decoy=decoy, val_mse=val_mse, scale_ratio=ratio, gate=val_mse <= gate))
            # R4 iterative self-consistent (collapse-guarded)
            t = trace_base
            traj = []
            for _ in range(5):
                Cx = cc.rebuild_covariance(S, cc.trace_match_scale(S, t))
                rec, decoy, val_mse, w = recover(Cx)
                t_new = float(np.trace(cov(w)))
                traj.append((t / trace_S, rec))
                if t_new < 1e-4 * trace_S:  # collapse guard
                    break
                t = t_new
            rows.append(dict(ensemble=ensemble, rule="R4_iterative", structure=sname,
                             recovery=rec, decoy=decoy, val_mse=val_mse, scale_ratio=t / trace_S, gate=val_mse <= gate,
                             note=f"traj={[ (round(a,3), round(b,1)) for a,b in traj ]}"))

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "scale_calibration.csv", index=False)
    for ensemble in common.ENSEMBLES:
        print(f"== {ensemble}")
        for _, r in frame[frame.ensemble == ensemble].iterrows():
            print("   {:22s} {:11s} rec={:6.1f}% decoy={:.3f} val_mse={:.4f} scale={:>7} gate={}".format(
                r.rule, str(r.structure), r.recovery, r.decoy, r.val_mse,
                ("%.3f" % r.scale_ratio) if np.isfinite(r.scale_ratio) else "-", "Y" if r.gate else "n"))
    print(f"wrote {args.output_dir / 'scale_calibration.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_scale_calibration")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
