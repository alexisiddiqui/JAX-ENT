#!/usr/bin/env python3
"""Synthetic gradient study: Sigma-MSE vs eye-MSE on the state-population oracle geometry.

Pre-Phase-4 diagnostic. Before substituting the frozen Phase 3 covariance into the real-uptake
population fit, this asks the cheap synthetic question on the oracle's 5-structure
(state-population) parameterisation: with uptake generated at the known NMR populations and
noise drawn from the frozen Sigma itself, are the gradients of the Sigma-weighted losses
(a) correct against finite differences, (b) comparably strong and well-aligned with the descent
direction toward the truth, and (c) no noisier than the eye-MSE control at the true optimum?

Losses compared, all on the same residual r = predicted - observed, (P, T):

* ``mse``        - mean(r^2); the eye-MSE control.
* ``sigma_mse``  - per-timepoint 0.5 r^T W r averaged over cells, with W the trace-normalised
                   collapsed precision from the Phase 3 compatibility export. This reproduces
                   the exact math of ``hdx_uptake_sigma_MSE_loss`` (losses.py:1517) without
                   constructing a Simulation/Dataloader.
* ``sigma_mse_shipped`` - the same per-timepoint math with the *previous* shipped precision
                   (``data/_MoPrP_covariance_matrices/Sigma.npz`` ``Sigma_inv``, the broken
                   np.cov-era matrix with the 90.6% leading-direction pathology), trace-normalised
                   to trace(W) == P exactly as the dataloader pathway would.
* ``joint``      - 0.5 vec(r)^T Sigma^{-1} vec(r) / N with the full frozen 210x210 covariance,
                   solved through its stored Cholesky factor (time-major vec, index = j*P + p).

Diagnostics per loss: analytic grad vs central finite differences; gradient norms and cosines
at the truth, at uniform logits, and at a decoy-saturated corner; gradient bias/SD over noise
realisations at the truth (SNR); Gauss-Newton spectra J^T W J at the truth (identifiability of
the population directions under each metric); short Adam recovery runs as an end-to-end control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import _moprp_recovery_common as common
from moprp_pivot_litmus import pivot_observable
from moprp_population_oracle import (
    _adversarial_starts,
    _full_support_vector,
    _neutral_starts,
    _present_support,
    _target_populations,
    _uniform_within_state_matrix,
)
from jaxent.src.analysis.pf_variance import jensen_shannon_recovery_percent
from jaxent.src.analysis.state_population import FULL_STATE_SUPPORT

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
NOISE_MODEL_DIR = HERE / "_moprp_sigma_noise_model"
SHIPPED_SIGMA = HERE.parents[1] / "data/_MoPrP_covariance_matrices/Sigma.npz"
ENSEMBLE = "AF2_MSAss"  # the ensemble the Phase 3 covariance was fitted on
PIVOT = "fast"  # default forward path per the pivot record
TARGET_STATE_NAMES = ("Folded", "PUF1", "PUF2")
RECOVERY_THRESHOLD = 99.0
DECOY_MASS_THRESHOLD = 0.02


# --------------------------------------------------------------------------------------
# Frozen Sigma artifacts
# --------------------------------------------------------------------------------------
def load_frozen_sigma() -> dict:
    with np.load(NOISE_MODEL_DIR / "target_modes.npz") as z:
        cov = np.asarray(z["covariance"], dtype=np.float64)
        chol = np.asarray(z["cholesky"], dtype=np.float64)
        vector_order = str(z["vector_order"])
        params = dict(zip([str(n) for n in z["parameter_names"]], np.asarray(z["parameters"])))
    if "time-major" not in vector_order:
        raise ValueError(f"unexpected vector order: {vector_order!r}")
    if not np.allclose(chol @ chol.T, cov, atol=1e-10):
        raise ValueError("stored cholesky is not a lower factor of the stored covariance")
    with np.load(NOISE_MODEL_DIR / "compatibility_covariances.npz") as c:
        collapsed_precision = np.asarray(c["trace_normalized_collapsed_precision"], np.float64)
    with np.load(SHIPPED_SIGMA) as s:
        shipped = np.asarray(s["Sigma_inv"], dtype=np.float64)
    shipped = 0.5 * (shipped + shipped.T)
    shipped = shipped * (shipped.shape[0] / np.trace(shipped))  # loader trace-normalisation
    return {
        "shipped_precision": shipped,
        "covariance": cov,
        "cholesky": chol,
        "collapsed_precision": collapsed_precision,
        "vector_order": vector_order,
        "parameters": {k: float(v) for k, v in params.items()},
    }


def vec_time_major(residual_pt: jnp.ndarray) -> jnp.ndarray:
    """(P, T) -> (T*P,) with index = j * P + p, matching the frozen vector order."""

    return residual_pt.T.reshape(-1)


# --------------------------------------------------------------------------------------
# Losses (shared residual algebra)
# --------------------------------------------------------------------------------------
def make_losses(sigma: dict, n_peptides: int, n_times: int) -> dict:
    chol = jnp.asarray(sigma["cholesky"])
    n_cells = n_peptides * n_times

    def mse(residual):
        return jnp.mean(residual**2)

    def per_time_precision_loss(w):
        # exact math of hdx_uptake_sigma_MSE_loss: per-timepoint 0.5 r^T W r, / (T * P)
        w = jnp.asarray(w)

        def fn(residual):
            per_time = jax.vmap(lambda r: 0.5 * r @ (w @ r))(residual.T)
            return jnp.sum(per_time) / n_cells

        return fn

    def joint(residual):
        white = jax.scipy.linalg.solve_triangular(chol, vec_time_major(residual), lower=True)
        return 0.5 * jnp.sum(white**2) / n_cells

    return {
        "mse": mse,
        "sigma_mse": per_time_precision_loss(sigma["collapsed_precision"]),
        "sigma_mse_shipped": per_time_precision_loss(sigma["shipped_precision"]),
        "joint": joint,
    }


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------
def finite_difference_grad(fn, theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[i] += step
        down[i] -= step
        grad[i] = (float(fn(jnp.asarray(up))) - float(fn(jnp.asarray(down)))) / (2 * step)
    return grad


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def gauss_newton_spectra(jac_vec: np.ndarray, sigma: dict, n_peptides: int) -> dict:
    """Singular spectra of the whitened Jacobian under each metric.

    ``jac_vec`` is d vec(pred) / d theta in time-major order, shape (T*P, S).
    """

    out = {}
    out["mse"] = np.linalg.svd(jac_vec, compute_uv=False)

    def per_time_whitened_svd(w):
        # precision applied per timepoint: block-diagonal W over time-major blocks;
        # eigh square root so a near-singular shipped precision cannot break Cholesky
        vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
        root = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T
        blocks = jac_vec.reshape(-1, n_peptides, jac_vec.shape[1])  # (T, P, S)
        whitened = np.einsum("qp,tps->tqs", root, blocks).reshape(-1, jac_vec.shape[1])
        return np.linalg.svd(whitened, compute_uv=False)

    out["sigma_mse"] = per_time_whitened_svd(sigma["collapsed_precision"])
    out["sigma_mse_shipped"] = per_time_whitened_svd(sigma["shipped_precision"])
    solved = np.linalg.solve(sigma["cholesky"], jac_vec)
    out["joint"] = np.linalg.svd(solved, compute_uv=False)
    return {k: [float(s) for s in v] for k, v in out.items()}


def optimize(loss_theta, theta0: np.ndarray, steps: int, lr: float) -> np.ndarray:
    optimizer = optax.adam(lr)
    theta = jnp.asarray(theta0)
    state = optimizer.init(theta)
    grad_fn = jax.jit(jax.grad(loss_theta))
    for _ in range(steps):
        grads = grad_fn(theta)
        updates, state = optimizer.update(grads, state)
        theta = optax.apply_updates(theta, updates)
    return np.asarray(theta)


# --------------------------------------------------------------------------------------
# Main study
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sigma = load_frozen_sigma()
    inputs = common.load_ensemble_inputs(ENSEMBLE)

    present = _present_support(inputs.states)
    weight_map = jnp.asarray(_uniform_within_state_matrix(inputs.states, present))
    target_pop = _target_populations(present, inputs.support, inputs.targets)
    theta_true = np.log(np.maximum(target_pop, 1e-6))
    log_pf = jnp.asarray(inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH))
    k_ints = jnp.asarray(inputs.k_ints)
    timepoints = jnp.asarray(inputs.timepoints)
    mapping = jnp.asarray(inputs.mapping)
    n_peptides, n_times = mapping.shape[0], timepoints.shape[0]
    if sigma["covariance"].shape[0] != n_peptides * n_times:
        raise ValueError("frozen Sigma dimension does not match ensemble geometry")

    def predict(theta):
        weights = weight_map @ jax.nn.softmax(theta)
        return pivot_observable(log_pf, k_ints, timepoints, mapping, weights, PIVOT)

    clean_obs = np.asarray(predict(jnp.asarray(theta_true)))  # (P, T) at the truth

    rng = np.random.default_rng(args.seed)

    def noise_draw() -> np.ndarray:
        eps = sigma["cholesky"] @ rng.standard_normal(n_peptides * n_times)
        return eps.reshape(n_times, n_peptides).T  # time-major vec -> (P, T)

    losses = make_losses(sigma, n_peptides, n_times)

    # fixed primary noisy realisation shared by every loss
    primary_obs = clean_obs + noise_draw()

    def loss_theta_fn(name, observed):
        observed = jnp.asarray(observed)

        def fn(theta):
            return losses[name](predict(theta) - observed)

        return fn

    probe_points = {
        "theta_true": theta_true,
        "uniform": np.zeros(len(present)),
    }
    for family, name, theta0 in [("adversarial", n, t) for n, t in _adversarial_starts(present)][:1]:
        probe_points["adversarial"] = theta0

    n_draws = 8 if args.smoke else 64
    steps = 200 if args.smoke else 1500
    lr = 0.05

    results: dict = {
        "ensemble": ENSEMBLE,
        "pivot": PIVOT,
        "coefficients": {"bc": common.PUBLISHED_BC, "bh": common.PUBLISHED_BH},
        "present_states": list(present),
        "frozen_sigma_parameters": sigma["parameters"],
        "seed": args.seed,
        "noise_draws": n_draws,
        "input_hashes": common.input_hashes(),
    }

    # 1) gradient correctness + strength at probe points (fixed noisy realisation)
    grads_at: dict = {}
    correctness: dict = {}
    for name in losses:
        fn = loss_theta_fn(name, primary_obs)
        grad_fn = jax.grad(fn)
        correctness[name] = {}
        grads_at[name] = {}
        for point, theta in probe_points.items():
            analytic = np.asarray(grad_fn(jnp.asarray(theta)))
            numeric = finite_difference_grad(fn, np.asarray(theta, dtype=np.float64))
            denom = max(np.linalg.norm(numeric), 1e-30)
            correctness[name][point] = {
                "relative_error": float(np.linalg.norm(analytic - numeric) / denom),
                "grad_norm": float(np.linalg.norm(analytic)),
            }
            grads_at[name][point] = analytic
    results["gradient_correctness"] = correctness

    # 2) alignment: cosine vs eye-MSE gradient and vs the descent direction toward truth
    alignment: dict = {}
    for name in losses:
        alignment[name] = {}
        for point, theta in probe_points.items():
            grad = grads_at[name][point]
            to_truth = theta_true - np.asarray(theta)
            alignment[name][point] = {
                "cosine_vs_mse": cosine(grad, grads_at["mse"][point]),
                "cosine_descent_vs_truth_direction": cosine(-grad, to_truth)
                if point != "theta_true"
                else None,
            }
    results["gradient_alignment"] = alignment

    # 3) gradient SNR at the truth over noise realisations
    snr: dict = {}
    grad_fns = {name: jax.jit(jax.grad(loss_theta_fn(name, clean_obs))) for name in losses}
    theta_true_j = jnp.asarray(theta_true)
    draws = {name: [] for name in losses}
    for _ in range(n_draws):
        observed = clean_obs + noise_draw()
        for name in losses:
            fn = jax.grad(loss_theta_fn(name, observed))
            draws[name].append(np.asarray(fn(theta_true_j)))
    for name in losses:
        arr = np.stack(draws[name])  # (K, S)
        clean_grad = np.asarray(grad_fns[name](theta_true_j))
        mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        snr[name] = {
            "clean_grad_norm_at_truth": float(np.linalg.norm(clean_grad)),
            "noise_mean_grad_norm": float(np.linalg.norm(mean)),
            "noise_sd_norm": float(np.linalg.norm(sd)),
            "per_state_mean": [float(v) for v in mean],
            "per_state_sd": [float(v) for v in sd],
        }
    results["gradient_noise_response"] = snr

    # 4) Gauss-Newton identifiability spectra at the truth
    def predict_vec(theta):
        return vec_time_major(predict(theta))

    jac_vec = np.asarray(jax.jacrev(predict_vec)(theta_true_j))  # (T*P, S)
    results["gauss_newton_spectra"] = gauss_newton_spectra(jac_vec, sigma, n_peptides)

    # 5) recovery control: short fits from neutral + adversarial starts
    full_targets = jnp.asarray(inputs.targets)
    starts = [("neutral", n, t) for n, t in _neutral_starts(len(present), 2 if args.smoke else 3)]
    starts += [("adversarial", n, t) for n, t in _adversarial_starts(present)]
    recovery: dict = {}
    for regime, observed in (("noiseless", clean_obs), ("noisy", primary_obs)):
        recovery[regime] = {}
        for name in losses:
            fn = loss_theta_fn(name, observed)
            runs = []
            for family, start_name, theta0 in starts:
                theta_star = optimize(fn, np.asarray(theta0, dtype=np.float64), steps, lr)
                populations = np.asarray(jax.nn.softmax(jnp.asarray(theta_star)))
                full_pop = _full_support_vector(present, populations)
                rec = float(
                    jensen_shannon_recovery_percent(jnp.asarray(full_pop), full_targets)
                )
                decoy = float(
                    sum(
                        full_pop[FULL_STATE_SUPPORT.index(s)]
                        for s in present
                        if s not in TARGET_STATE_NAMES
                    )
                )
                runs.append(
                    {
                        "start_family": family,
                        "start_name": start_name,
                        "recovery_percent": rec,
                        "decoy_mass": decoy,
                        "final_loss": float(fn(jnp.asarray(theta_star))),
                        "recovered": rec >= RECOVERY_THRESHOLD and decoy <= DECOY_MASS_THRESHOLD,
                    }
                )
            neutral = [r for r in runs if r["start_family"] == "neutral"]
            recovery[regime][name] = {
                "runs": runs,
                "all_neutral_recovered": all(r["recovered"] for r in neutral),
                "worst_neutral_recovery_percent": min(r["recovery_percent"] for r in neutral),
            }
    results["recovery_control"] = recovery
    results["smoke"] = bool(args.smoke)

    out_path = args.output_dir / "gradient_synthetic_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print(f"ensemble={ENSEMBLE} pivot={PIVOT} states={list(present)} smoke={args.smoke}")
    print("gradient correctness (max relative error vs central FD across probe points):")
    for name in losses:
        worst = max(v["relative_error"] for v in correctness[name].values())
        print(f"  {name:10s} {worst:.3e}")
    print("gradient norms at uniform start (strength) and cosine vs MSE:")
    for name in losses:
        g = correctness[name]["uniform"]["grad_norm"]
        c = alignment[name]["uniform"]["cosine_vs_mse"]
        d = alignment[name]["uniform"]["cosine_descent_vs_truth_direction"]
        print(f"  {name:10s} |grad|={g:.4e}  cos_vs_mse={c:+.4f}  cos_descent_to_truth={d:+.4f}")
    print("gradient SNR at truth (clean-grad norm should be ~0; sd norm = noise floor):")
    for name in losses:
        s = snr[name]
        print(
            f"  {name:10s} clean={s['clean_grad_norm_at_truth']:.2e}  "
            f"noise-mean={s['noise_mean_grad_norm']:.2e}  noise-sd={s['noise_sd_norm']:.2e}"
        )
    print("Gauss-Newton singular spectra at truth (metric-whitened Jacobian):")
    for name, sv in results["gauss_newton_spectra"].items():
        cond = sv[0] / max(sv[-1], 1e-30)
        print(f"  {name:10s} sv={['%.3e' % s for s in sv]}  cond={cond:.2e}")
    print("recovery control (worst neutral recovery %):")
    for regime in recovery:
        row = "  ".join(
            f"{name}={recovery[regime][name]['worst_neutral_recovery_percent']:.1f}"
            for name in losses
        )
        print(f"  {regime:9s} {row}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "_moprp_sigma_gradient_synthetic",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
