#!/usr/bin/env python3
"""Stage H: reweight MoPrP through the population-free covariance-*shape* prior (M2).

The effective non-circular tool (Stages F/G) is the unweighted covariance *shape* used as a soft
regulariser:  L = mean_MSE/mean_uniform + γ·correlation_shape_loss(C(w), prior_corr) + η·KL.

The prior correlation is the population-free unweighted peptide shape.  Cross-validation uses five
blocked three-timepoint folds on peptides 2–14 (peptide 1 held out).  Because there is no known
population in deployment, γ is selected population-free: the **largest γ whose validation mean MSE
stays within 1.05× the mean-only baseline** (apply the prior as hard as the mean fit tolerates).
Recovery is reported as validation only — never used for selection.  The true-target shape is run
as a reference ceiling.
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
from jaxent.src.analysis.state_population import (
    correlation_of, correlation_shape_loss, peptide_logpf_covariance,
    state_populations, strict_recovery_percent, FULL_STATE_SUPPORT,
)
from jaxent.src.analysis.pf_variance import overlap_projection, weighted_population_covariance, kl_to_uniform

COEFFICIENTS = ("scaled_published", "constrained_optimum")
GAMMAS = (0.1, 1.0, 10.0, 30.0)
ETAS = (0.0, 0.01, 0.1)
ALPHA = 0.05
STEPS, LR, N_START = 2000, 0.03, 2
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _select_largest_gamma(agg, baseline_mse):
    """Largest γ whose CV mean MSE ≤ 1.05× baseline; tie-break η by lowest val MSE."""

    gate = 1.05 * baseline_mse
    eligible = [r for r in agg if r["val_mse"] <= gate]
    pool = eligible if eligible else agg
    return sorted(pool, key=lambda r: (-r["gamma"], r["val_mse"]))[0]


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

    raw_rows, selected_rows = [], []
    for ensemble in common.ENSEMBLES:
        inp = common.load_ensemble_inputs(ensemble)
        for coeff_name in COEFFICIENTS:
            bc, bh = lock["frozen_settings"][coeff_name]["bc"], lock["frozen_settings"][coeff_name]["bh"]
            log_pf = jnp.asarray(inp.log_pf_by_frame(bc, bh))
            k = jnp.asarray(inp.k_ints)
            timepoints = jnp.asarray(inp.timepoints)
            keep = np.ones(inp.mapping.shape[0], bool); keep[common.PEPTIDE1_INDEX] = False
            M_keep = jnp.asarray(inp.mapping[keep])
            obs = jnp.asarray(inp.observed_uptake[keep])
            projection = overlap_projection(M_keep)
            uniform = jnp.full(inp.n_frames, 1.0 / inp.n_frames)
            w_nmr = jnp.asarray(inp.reference_weights)

            def cov(w):
                return peptide_logpf_covariance(log_pf, M_keep, jnp.asarray(w))

            priors = {
                "unweighted": correlation_of(cov(uniform)),                # population-free (deployment)
                "true_shape": correlation_of(cov(w_nmr)),                  # reference ceiling only
            }

            def recovery_of(w):
                rec = float(strict_recovery_percent(w, inp.states, inp.support, inp.targets))
                pops = np.asarray(state_populations(w, inp.states, inp.support))
                decoy = float(sum(pops[FULL_STATE_SUPPORT.index(s)] for s in inp.support if s not in TARGET_STATES))
                return rec, decoy

            folds = R._time_folds(inp.timepoints.size)
            # accumulate per (prior, gamma, eta) across folds
            acc = {}
            base_acc = []
            for val_idx in folds:
                train_idx = np.setdiff1d(np.arange(inp.timepoints.size), val_idx)
                t_tr, t_val = timepoints[train_idx], timepoints[val_idx]
                obs_tr, obs_val = obs[:, train_idx], obs[:, val_idx]
                mse_u = float(R._mean_mse(R._predict_uptake(log_pf, k, t_tr, M_keep, uniform).T, obs_tr))

                def val_mse(w):
                    return float(R._mean_mse(R._predict_uptake(log_pf, k, t_val, M_keep, jnp.asarray(w)).T, obs_val))

                # baseline
                def base_loss(lg):
                    w = jax.nn.softmax(lg)
                    return R._mean_mse(R._predict_uptake(log_pf, k, t_tr, M_keep, w).T, obs_tr) / mse_u + 0.01 * kl_to_uniform(w)
                wb = _optimize(base_loss, inp.n_frames)
                base_acc.append(val_mse(wb))

                for pname, Pc in priors.items():
                    for gamma in GAMMAS:
                        for eta in ETAS:
                            def loss(lg, Pc=Pc, gamma=gamma, eta=eta):
                                w = jax.nn.softmax(lg)
                                mean = R._mean_mse(R._predict_uptake(log_pf, k, t_tr, M_keep, w).T, obs_tr) / mse_u
                                shape = correlation_shape_loss(cov(w), Pc, projection, ALPHA)
                                return mean + gamma * shape + eta * kl_to_uniform(w)
                            w = _optimize(loss, inp.n_frames)
                            rec, decoy = recovery_of(w)
                            acc.setdefault((pname, gamma, eta), []).append((val_mse(w), rec, decoy))

            base_mse = float(np.mean(base_acc))
            base_w = wb  # last fold's baseline weights, for reference recovery
            base_rec, base_decoy = recovery_of(base_w)
            raw_rows.append(dict(ensemble=ensemble, coefficient=coeff_name, prior="baseline",
                                 gamma=0.0, eta=0.0, val_mse=base_mse, recovery=base_rec, decoy=base_decoy))
            for (pname, gamma, eta), vals in acc.items():
                v = np.array(vals)
                raw_rows.append(dict(ensemble=ensemble, coefficient=coeff_name, prior=pname, gamma=gamma, eta=eta,
                                     val_mse=float(v[:, 0].mean()), recovery=float(v[:, 1].mean()), decoy=float(v[:, 2].mean())))

            for pname in priors:
                agg = [r for r in raw_rows if r["ensemble"] == ensemble and r["coefficient"] == coeff_name and r["prior"] == pname]
                sel = _select_largest_gamma(agg, base_mse)
                selected_rows.append(dict(ensemble=ensemble, coefficient=coeff_name, prior=pname,
                                          selected_gamma=sel["gamma"], selected_eta=sel["eta"], val_mse=sel["val_mse"],
                                          recovery=sel["recovery"], decoy=sel["decoy"],
                                          baseline_recovery=base_rec, baseline_val_mse=base_mse,
                                          mean_gate_passed=sel["val_mse"] <= 1.05 * base_mse,
                                          recovery_gain_pp=sel["recovery"] - base_rec))

    import pandas as pd
    pd.DataFrame(raw_rows).to_csv(args.output_dir / "shape_prior_reweighting_raw.csv", index=False)
    sel = pd.DataFrame(selected_rows)
    sel.to_csv(args.output_dir / "shape_prior_reweighting_selected.csv", index=False)
    for _, r in sel.iterrows():
        print("{:12s}/{:16s} {:11s} γ={:>4} rec={:6.1f}% (base {:5.1f}, +{:5.1f}pp) decoy={:.3f} val_mse={:.4f} gate={}".format(
            r.ensemble, r.coefficient, r.prior, r.selected_gamma, r.recovery, r.baseline_recovery,
            r.recovery_gain_pp, r.decoy, r.val_mse, "Y" if r.mean_gate_passed else "n"))
    print(f"wrote {args.output_dir / 'shape_prior_reweighting_selected.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_shape_prior_reweighting")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
