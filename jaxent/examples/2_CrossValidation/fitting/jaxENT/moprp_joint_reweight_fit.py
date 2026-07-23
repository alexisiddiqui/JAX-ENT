#!/usr/bin/env python3
"""Joint reweighting + BV-coefficient fitting on MoPrP, regularised by the shape prior.

Rather than pre-fitting coefficients to the mean (which gave the degenerate Bh=0) and reweighting
with a swept γ, this optimises **per-ensemble frame weights** and **shared non-negative (Bc, Bh)**
*together*, with γ fixed = 1 and 5 starts.

    L = Σ_e [ mean_MSE_e/mean_ref_e + γ·shape_loss(C_e(w_e,Bc,Bh), prior_e) ] + η·Σ_e KL(w_e)

The prior is the population-free unweighted covariance *shape* frozen at the published coefficient
direction.  Questions: does the covariance prior break the mean-only Bh=0 degeneracy, and does one
joint fit recover the known population without a γ sweep?  The γ=0 ablation isolates whether it is
the *covariance* (not the mean) that identifies Bh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

import _moprp_recovery_common as common
import moprp_covariance_recovery as R
from jaxent.src.analysis.state_population import (
    correlation_of, correlation_shape_loss, peptide_logpf_covariance,
    state_populations, strict_recovery_percent, FULL_STATE_SUPPORT,
)
from jaxent.src.analysis.pf_variance import overlap_projection, kl_to_uniform

REF_BC, REF_BH = 0.35, 2.0        # published direction: frozen prior shape + mean normaliser
ETA = 0.01
STEPS, LR, N_START = 2000, 0.03, 5
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _coeffs_from_theta(theta):
    """Softplus -> strictly non-negative (Bc, Bh)."""
    return jax.nn.softplus(theta[0]), jax.nn.softplus(theta[1])


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


def _load_cell(ensemble):
    inp = common.load_ensemble_inputs(ensemble)
    keep = np.ones(inp.mapping.shape[0], bool); keep[common.PEPTIDE1_INDEX] = False
    heavy = jnp.asarray(inp.heavy_contacts)
    acceptor = jnp.asarray(inp.acceptor_contacts)
    M_keep = jnp.asarray(inp.mapping[keep])
    obs = jnp.asarray(inp.observed_uptake[keep])
    k = jnp.asarray(inp.k_ints)
    tp = jnp.asarray(inp.timepoints)
    projection = overlap_projection(M_keep)
    uniform = jnp.full(inp.n_frames, 1.0 / inp.n_frames)
    ref_logpf = REF_BC * heavy + REF_BH * acceptor
    prior = correlation_of(peptide_logpf_covariance(ref_logpf, M_keep, uniform))
    mean_ref = float(R._mean_mse(R._predict_uptake(ref_logpf, k, tp, M_keep, uniform).T, obs))
    return dict(inp=inp, heavy=heavy, acceptor=acceptor, M_keep=M_keep, obs=obs, k=k, tp=tp,
                projection=projection, prior=prior, mean_ref=mean_ref, n=inp.n_frames)


def _run(cells, gamma, steps, lr, n_start):
    names = list(cells)
    theta0 = jnp.asarray([_inv_softplus(REF_BC), _inv_softplus(REF_BH)])

    def loss_fn(params):
        bc, bh = _coeffs_from_theta(params["theta"])
        total = 0.0
        for name in names:
            c = cells[name]
            w = jax.nn.softmax(params["logits"][name])
            log_pf = bc * c["heavy"] + bh * c["acceptor"]
            pred = R._predict_uptake(log_pf, c["k"], c["tp"], c["M_keep"], w).T
            mean = R._mean_mse(pred, c["obs"]) / c["mean_ref"]
            cov = peptide_logpf_covariance(log_pf, c["M_keep"], w)
            shape = correlation_shape_loss(cov, c["prior"], c["projection"], 0.05)
            total = total + mean + gamma * shape + ETA * kl_to_uniform(w)
        return total

    grad_fn = jax.jit(jax.grad(loss_fn))
    best = None
    for s in range(n_start):
        rng = np.random.default_rng(100 + s)
        params = {"theta": theta0 + (0.0 if s == 0 else jnp.asarray(rng.normal(scale=0.1, size=2))),
                  "logits": {name: jnp.asarray(np.zeros(cells[name]["n"]) if s == 0 else rng.normal(scale=0.01, size=cells[name]["n"]))
                             for name in names}}
        opt = optax.adam(lr); state = opt.init(params)
        for _ in range(steps):
            g = grad_fn(params)
            updates, state = opt.update(g, state)
            params = optax.apply_updates(params, updates)
        obj = float(loss_fn(params))
        if best is None or obj < best[0]:
            best = (obj, params)
    return best[1]


def _report(cells, params, gamma):
    bc, bh = (float(x) for x in _coeffs_from_theta(params["theta"]))
    rows = []
    for name, c in cells.items():
        w = np.asarray(jax.nn.softmax(params["logits"][name]))
        inp = c["inp"]
        rec = float(strict_recovery_percent(w, inp.states, inp.support, inp.targets))
        pops = np.asarray(state_populations(w, inp.states, inp.support))
        decoy = float(sum(pops[FULL_STATE_SUPPORT.index(s)] for s in inp.support if s not in TARGET_STATES))
        log_pf = bc * c["heavy"] + bh * c["acceptor"]
        val_mse = float(R._mean_mse(R._predict_uptake(log_pf, c["k"], c["tp"], c["M_keep"], jnp.asarray(w)).T, c["obs"]))
        rows.append(dict(ensemble=name, gamma=gamma, bc=bc, bh=bh, recovery=rec, decoy=decoy, val_mse=val_mse))
    return rows


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = {name: _load_cell(name) for name in common.ENSEMBLES}

    all_rows = []
    for gamma in (1.0, 0.0):  # γ=1 joint fit; γ=0 ablation (mean-only, coefficients free)
        params = _run(cells, gamma, STEPS, LR, N_START)
        all_rows.extend(_report(cells, params, gamma))

    import pandas as pd
    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_dir / "joint_reweight_fit.csv", index=False)
    for gamma in (1.0, 0.0):
        sub = frame[frame.gamma == gamma]
        bc, bh = sub.bc.iloc[0], sub.bh.iloc[0]
        tag = "joint (shape prior)" if gamma == 1.0 else "ablation γ=0 (mean-only)"
        print(f"== γ={gamma:.0f}  {tag}   fitted (Bc={bc:.4f}, Bh={bh:.4f})")
        for _, r in sub.iterrows():
            print("   {:12s} rec={:6.1f}%  decoy={:.3f}  val_mse={:.4f}".format(r.ensemble, r.recovery, r.decoy, r.val_mse))
    print(f"wrote {args.output_dir / 'joint_reweight_fit.csv'}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_joint_reweight_fit")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
