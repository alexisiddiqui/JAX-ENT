#!/usr/bin/env python3
"""Stage F: which shape-prior mechanism is sensitive to unweighted-vs-target shape?

A population-free covariance *shape* prior only regularises usefully if the mechanism is sensitive
to the difference between the unweighted correlation shape and the weighted-target shape — if
swapping the two prior sources changes nothing, the prior is inert.  This runner projects the
reweighting through each mechanism with the prior sourced from (a) the unweighted correlation and
(b) the target correlation, and reports the sensitivity (difference in recovery / behaviour).

Mechanisms (all scale-free; strength γ is a hyperparameter):
  M1  mode-subspace projection  — penalise solution covariance outside the prior's top-k modes
  M2  soft shape regulariser    — projected log-Euclidean distance of corr(C(w)) to the prior corr
  M3  residual whitening (GLS)  — fit the peptide mean residual in the prior-shape metric
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
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis.state_population import (
    peptide_logpf_covariance, state_populations, strict_recovery_percent, FULL_STATE_SUPPORT,
    shrunk_trace_normalized_precision,
)
from jaxent.src.analysis.pf_variance import (
    overlap_projection, projected_log_euclidean_covariance_loss, kl_to_uniform,
)

COEFFICIENT = "scaled_published"
K_MODES = 5
ETA, GAMMA = 0.01, 1.0
STEPS, LR, N_START = 2000, 0.03, 3
ALPHA = 0.05
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _to_corr(C):
    d = jnp.sqrt(jnp.clip(jnp.diag(C), 1e-12, None))
    return C / jnp.outer(d, d)


def _topk_projector(correlation, k):
    _, V = np.linalg.eigh(correlation)
    Vk = V[:, ::-1][:, :k]
    return Vk @ Vk.T


def _optimize(loss, n_frames):
    best = None
    for s in range(N_START):
        start = np.zeros(n_frames) if s == 0 else np.random.default_rng(s).normal(scale=0.01, size=n_frames)
        logits = R._optimize(loss, start, STEPS, LR)
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
        keep = np.ones(inp.mapping.shape[0], bool); keep[common.PEPTIDE1_INDEX] = False
        M_keep = jnp.asarray(inp.mapping[keep])
        obs_keep = jnp.asarray(inp.observed_uptake[keep])
        projection = overlap_projection(M_keep)
        uniform = jnp.full(inp.n_frames, 1.0 / inp.n_frames)
        w_nmr = jnp.asarray(inp.reference_weights)
        mse_uniform = float(R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, uniform).T, obs_keep))

        def cov(w):
            return peptide_logpf_covariance(log_pf, M_keep, jnp.asarray(w))

        def mean_term(w):
            return R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, w).T, obs_keep) / mse_uniform

        def recover(w):
            rec = float(strict_recovery_percent(w, inp.states, inp.support, inp.targets))
            pops = np.asarray(state_populations(w, inp.states, inp.support))
            decoy = float(sum(pops[FULL_STATE_SUPPORT.index(s)] for s in inp.support if s not in TARGET_STATES))
            val_mse = float(R._mean_mse(R._predict_uptake(log_pf, k, timepoints, M_keep, jnp.asarray(w)).T, obs_keep))
            return rec, decoy, val_mse

        # baseline (mean-only)
        w_base = _optimize(lambda lg: mean_term(jax.nn.softmax(lg)) + ETA * kl_to_uniform(jax.nn.softmax(lg)), inp.n_frames)
        _, _, base_mse = recover(w_base)
        rb, db, mb = recover(w_base)
        rows.append(dict(ensemble=ensemble, rule="baseline", prior="-", recovery=rb, decoy=db, val_mse=mb, gate=mb <= 1.05 * base_mse))

        C_true = np.asarray(cov(w_nmr))
        # sensitivity control: how different are the two prior shapes?
        shape_gap = cc.covariance_metrics(np.asarray(_to_corr(cov(uniform))), cc.to_correlation(C_true), permutations=199)
        prior_sources = {"unweighted": np.asarray(_to_corr(cov(uniform))), "target": cc.to_correlation(C_true)}

        for psource, R_prior in prior_sources.items():
            Pk = jnp.asarray(_topk_projector(R_prior, K_MODES))
            Ident = jnp.eye(R_prior.shape[0])
            Wj = jnp.asarray(shrunk_trace_normalized_precision(jnp.asarray(R_prior), ALPHA))
            Rp = jnp.asarray(R_prior)

            def m1(lg):
                w = jax.nn.softmax(lg); Rw = _to_corr(cov(w))
                resid = (Ident - Pk) @ Rw @ (Ident - Pk)
                return mean_term(w) + GAMMA * (jnp.sum(resid**2) / (jnp.sum(Rw**2) + 1e-12)) + ETA * kl_to_uniform(w)

            def m2(lg):
                w = jax.nn.softmax(lg)
                pen = projected_log_euclidean_covariance_loss(_to_corr(cov(w)), Rp, projection, ALPHA)
                return mean_term(w) + GAMMA * pen + ETA * kl_to_uniform(w)

            def m3(lg):
                w = jax.nn.softmax(lg)
                resid = (R._predict_uptake(log_pf, k, timepoints, M_keep, w).T - obs_keep)
                quad = jnp.mean(jax.vmap(lambda r: r @ Wj @ r)(resid.T)) / resid.shape[0]
                return mean_term(w) + GAMMA * quad + ETA * kl_to_uniform(w)

            for rule, loss in [("M1_mode_projection", m1), ("M2_shape_match", m2), ("M3_gls_whitening", m3)]:
                rec, decoy, val_mse = recover(_optimize(loss, inp.n_frames))
                rows.append(dict(ensemble=ensemble, rule=rule, prior=psource, recovery=rec, decoy=decoy,
                                 val_mse=val_mse, gate=val_mse <= 1.05 * base_mse))

        rows.append(dict(ensemble=ensemble, rule="__shape_gap_unw_vs_target", prior="-",
                         recovery=shape_gap["mantel_r"], decoy=shape_gap["norm_distance"], val_mse=np.nan, gate=None))

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "shape_prior_sensitivity.csv", index=False)
    for ensemble in common.ENSEMBLES:
        sub = frame[frame.ensemble == ensemble]
        gap = sub[sub.rule == "__shape_gap_unw_vs_target"].iloc[0]
        base = sub[sub.rule == "baseline"].iloc[0]
        print(f"== {ensemble}  (unweighted-vs-target shape: mantel={gap.recovery:.2f} normdist={gap.decoy:.2f}; baseline rec={base.recovery:.1f}%)")
        for rule in ["M1_mode_projection", "M2_shape_match", "M3_gls_whitening"]:
            u = sub[(sub.rule == rule) & (sub.prior == "unweighted")].iloc[0]
            t = sub[(sub.rule == rule) & (sub.prior == "target")].iloc[0]
            print("   {:20s} unweighted-prior rec={:6.1f}% (gate {})  target-prior rec={:6.1f}% (gate {})  |sensitivity|={:5.1f}pp".format(
                rule, u.recovery, "Y" if u.gate else "n", t.recovery, "Y" if t.gate else "n", abs(t.recovery - u.recovery)))
    print(f"wrote {args.output_dir / 'shape_prior_sensitivity.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_shape_prior")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
