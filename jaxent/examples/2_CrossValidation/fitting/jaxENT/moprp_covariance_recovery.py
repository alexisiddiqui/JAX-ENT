#!/usr/bin/env python3
"""Stage B: 500-frame reweighting for the known-population MoPrP recovery experiment.

Optimize all 500 frame weights (raw logits, softmax exactly once) separately per ensemble,
coefficient setting, and covariance coordinate, under four loss regimes:

1. ``baseline``        - unweighted mean-uptake MSE + MaxEnt (KL to uniform).
2. ``symmetric``       - mean MSE / uniform + gamma * d(C(w), C(w_NMR)) + eta * KL.  The
                         covariance distance is matched in one of the four coordinates.
3. ``dynamic``         - mean_t [ r_t^T W_pred(w, t) r_t ] + eta * KL, with predicted-uptake
                         precision (shrunk, trace-normalized), differentiated through w.
4. ``fixed_reference`` - same residual geometry but with the fixed C(w_NMR) precision.

Peptide 1 is held out entirely.  Cross-validation uses five blocked three-timepoint folds on
peptides 2..14.  Selection never uses recovery percent: symmetric/fixed require validation
mean MSE <= 1.05x the mean-only baseline, then lowest validation covariance loss; dynamic and
baseline select on ordinary validation MSE.

This is model geometry driven by a known population, not experimental covariance inference.
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
from jaxent.src.analysis.pf_variance import (
    kl_to_uniform,
    overlap_projection,
    projected_log_euclidean_covariance_loss,
    uptake_from_log_pf,
)
from jaxent.src.analysis.state_population import (
    FULL_STATE_SUPPORT,
    log_ratio_profile_loss,
    marginal_profile,
    peptide_logpf_covariance,
    peptide_uptake_covariances,
    shrunk_trace_normalized_precision,
    state_populations,
)
from jaxent.src.analysis.pf_variance import jensen_shannon_recovery_percent

ALPHA = 0.05
COORDINATES = ("logpf_marginal", "logpf_projected", "uptake_marginal", "uptake_projected")
TARGET_STATE_NAMES = ("Folded", "PUF1", "PUF2")
PEPTIDE1_INDEX = common.PEPTIDE1_INDEX


def _time_folds(n_timepoints: int, block: int = 3) -> list[np.ndarray]:
    """Contiguous blocked timepoint folds (validation blocks)."""

    return [np.arange(start, min(start + block, n_timepoints)) for start in range(0, n_timepoints, block)]


def _predict_uptake(log_pf, k_ints, timepoints, mapping, weights):
    """Average-first peptide mean uptake -> (T, P)."""

    mean_log_pf = jnp.asarray(log_pf) @ weights
    residue_uptake = uptake_from_log_pf(mean_log_pf, k_ints, timepoints)  # (T, R)
    return residue_uptake @ jnp.asarray(mapping).T


def _mean_mse(predicted, observed):
    residual = predicted - observed
    return jnp.mean(residual**2)


def _coordinate_distance(coordinate, log_pf, k_ints, timepoints, mapping, projection, weights, target):
    if coordinate == "logpf_marginal":
        cov = peptide_logpf_covariance(log_pf, mapping, weights)
        return log_ratio_profile_loss(marginal_profile(cov, ALPHA), target)
    if coordinate == "logpf_projected":
        cov = peptide_logpf_covariance(log_pf, mapping, weights)
        return projected_log_euclidean_covariance_loss(cov, target, projection, ALPHA)
    covs = peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)
    if coordinate == "uptake_marginal":
        profiles = jax.vmap(lambda c: marginal_profile(c, ALPHA))(covs)
        return jnp.mean(jax.vmap(log_ratio_profile_loss)(profiles, target))
    return jnp.mean(
        jax.vmap(lambda p, t: projected_log_euclidean_covariance_loss(p, t, projection, ALPHA))(
            covs, target
        )
    )


def _coordinate_target(coordinate, log_pf, k_ints, timepoints, mapping, weights):
    if coordinate == "logpf_marginal":
        return marginal_profile(peptide_logpf_covariance(log_pf, mapping, weights), ALPHA)
    if coordinate == "logpf_projected":
        return peptide_logpf_covariance(log_pf, mapping, weights)
    covs = peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)
    if coordinate == "uptake_marginal":
        return jax.vmap(lambda c: marginal_profile(c, ALPHA))(covs)
    return covs


def _optimize(loss_fn, logits0, steps, lr):
    optimizer = optax.adam(lr)
    logits = jnp.asarray(logits0)
    state = optimizer.init(logits)
    grad_fn = jax.jit(jax.grad(loss_fn))
    for _ in range(steps):
        grads = grad_fn(logits)
        updates, state = optimizer.update(grads, state)
        logits = optax.apply_updates(logits, updates)
    return np.asarray(logits)


def _starts(n_frames, n_random):
    starts = [np.zeros(n_frames)]
    for seed in range(n_random):
        starts.append(np.random.default_rng(3000 + seed).normal(scale=0.01, size=n_frames))
    return starts


def _recovery_record(weights, inputs):
    populations = np.asarray(state_populations(weights, inputs.states, inputs.support))
    recovery = float(jensen_shannon_recovery_percent(jnp.asarray(populations), jnp.asarray(inputs.targets)))
    decoy_mass = float(
        sum(populations[FULL_STATE_SUPPORT.index(s)] for s in inputs.support if s not in TARGET_STATE_NAMES)
    )
    return recovery, decoy_mass, {s: float(m) for s, m in zip(inputs.support, populations)}


def _run_cell(inputs, coeff, grid, steps, lr, n_random):
    """Run every method/hyperparameter/fold/start for one ensemble x coefficient cell."""

    log_pf = jnp.asarray(inputs.log_pf_by_frame(coeff["bc"], coeff["bh"]))
    k_ints = jnp.asarray(inputs.k_ints)
    timepoints = jnp.asarray(inputs.timepoints)
    w_nmr = jnp.asarray(inputs.reference_weights)

    # Hold out peptide 1; covariance/mean over peptides 2..14.
    keep = np.ones(inputs.mapping.shape[0], dtype=bool)
    keep[PEPTIDE1_INDEX] = False
    mapping = jnp.asarray(inputs.mapping[keep])  # (P-1, R)
    observed = jnp.asarray(inputs.observed_uptake[keep])  # (P-1, T)
    projection = overlap_projection(mapping)

    folds = _time_folds(inputs.timepoints.size)
    uniform = jnp.full(inputs.n_frames, 1.0 / inputs.n_frames)
    rows = []
    weights_out = {}

    for fold_index, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(inputs.timepoints.size), val_idx)
        t_train = timepoints[train_idx]
        t_val = timepoints[val_idx]
        obs_train = observed[:, train_idx]
        obs_val = observed[:, val_idx]

        # baselines and fixed references at uniform / w_NMR
        pred_uniform_train = _predict_uptake(log_pf, k_ints, t_train, mapping, uniform).T
        mse_uniform = float(_mean_mse(pred_uniform_train, obs_train))

        def mean_loss(logits, t_idx, obs):
            weights = jax.nn.softmax(logits)
            predicted = _predict_uptake(log_pf, k_ints, timepoints[t_idx], mapping, weights).T
            return _mean_mse(predicted, obs)

        def val_mse(logits):
            weights = jax.nn.softmax(logits)
            predicted = _predict_uptake(log_pf, k_ints, t_val, mapping, weights).T
            return float(_mean_mse(predicted, obs_val))

        # coordinate targets at w_NMR (train and val timepoints)
        targets_train = {
            c: jax.lax.stop_gradient(
                _coordinate_target(c, log_pf, k_ints, t_train, mapping, w_nmr)
            )
            for c in COORDINATES
        }
        targets_val = {
            c: jax.lax.stop_gradient(
                _coordinate_target(c, log_pf, k_ints, t_val, mapping, w_nmr)
            )
            for c in COORDINATES
        }

        def val_cov_loss(logits, coordinate):
            weights = jax.nn.softmax(logits)
            return float(
                _coordinate_distance(
                    coordinate, log_pf, k_ints, t_val, mapping, projection, weights,
                    targets_val[coordinate],
                )
            )

        def register(method, coordinate, gamma, eta, loss_fn):
            best = None
            for start_index, logits0 in enumerate(_starts(inputs.n_frames, n_random)):
                logits_star = _optimize(loss_fn, logits0, steps, lr)
                objective = float(loss_fn(jnp.asarray(logits_star)))
                if best is None or objective < best["objective"]:
                    best = {"objective": objective, "logits": logits_star, "start_index": start_index}
            weights_star = np.asarray(jax.nn.softmax(jnp.asarray(best["logits"])))
            recovery, decoy_mass, populations = _recovery_record(weights_star, inputs)
            run_id = f"{inputs.ensemble}|{coeff['name']}|{method}|{coordinate}|g{gamma}|e{eta}|f{fold_index}"
            weights_out[run_id] = weights_star
            rows.append(
                {
                    "run_id": run_id,
                    "ensemble": inputs.ensemble,
                    "coefficient": coeff["name"],
                    "method": method,
                    "coordinate": coordinate,
                    "gamma": gamma,
                    "eta": eta,
                    "fold": fold_index,
                    "train_objective": best["objective"],
                    "val_mse": val_mse(jnp.asarray(best["logits"])),
                    "val_mse_uniform": mse_uniform,
                    "val_cov_loss": (
                        val_cov_loss(jnp.asarray(best["logits"]), coordinate)
                        if coordinate != "none"
                        else float("nan")
                    ),
                    "recovery_percent": recovery,
                    "decoy_mass": decoy_mass,
                    **{f"pop_{s}": populations[s] for s in inputs.support},
                }
            )

        # --- Regime 1: baseline (mean MSE + MaxEnt) ---
        for eta in grid["eta"]:
            def loss_fn(logits, eta=eta):
                return mean_loss(logits, train_idx, obs_train) / mse_uniform + eta * kl_to_uniform(
                    jax.nn.softmax(logits)
                )

            register("baseline", "none", 0.0, eta, loss_fn)

        # --- Regime 2: symmetric covariance matching (per coordinate) ---
        for coordinate in grid["coordinates"]:
            for gamma in grid["gamma"]:
                for eta in grid["eta"]:
                    def loss_fn(logits, coordinate=coordinate, gamma=gamma, eta=eta):
                        weights = jax.nn.softmax(logits)
                        mean_term = mean_loss(logits, train_idx, obs_train) / mse_uniform
                        cov_term = _coordinate_distance(
                            coordinate, log_pf, k_ints, t_train, mapping, projection, weights,
                            targets_train[coordinate],
                        )
                        return mean_term + gamma * cov_term + eta * kl_to_uniform(weights)

                    register("symmetric", coordinate, gamma, eta, loss_fn)

        # --- Regimes 3 & 4: dynamic and fixed-reference uptake residual geometry ---
        reference_precisions = jax.lax.stop_gradient(
            jax.vmap(lambda c: shrunk_trace_normalized_precision(c, ALPHA))(
                peptide_uptake_covariances(log_pf, k_ints, t_train, mapping, w_nmr)
            )
        )
        for eta in grid["eta"]:
            def dynamic_loss(logits, eta=eta):
                weights = jax.nn.softmax(logits)
                predicted = _predict_uptake(log_pf, k_ints, t_train, mapping, weights).T  # (P, T)
                residual = predicted - obs_train
                covs = peptide_uptake_covariances(log_pf, k_ints, t_train, mapping, weights)
                precisions = jax.vmap(lambda c: shrunk_trace_normalized_precision(c, ALPHA))(covs)
                quad = jax.vmap(lambda W, r: r @ W @ r)(precisions, residual.T)
                return jnp.mean(quad) + eta * kl_to_uniform(weights)

            register("dynamic", "uptake_projected", 0.0, eta, dynamic_loss)

            def fixed_loss(logits, eta=eta):
                weights = jax.nn.softmax(logits)
                predicted = _predict_uptake(log_pf, k_ints, t_train, mapping, weights).T
                residual = predicted - obs_train
                quad = jax.vmap(lambda W, r: r @ W @ r)(reference_precisions, residual.T)
                return jnp.mean(quad) + eta * kl_to_uniform(weights)

            register("fixed_reference", "uptake_projected", 0.0, eta, fixed_loss)

    return rows, weights_out


def _select(rows: list[dict]) -> list[dict]:
    """Per (ensemble, coefficient, method, coordinate), select hyperparameters via CV.

    Selection gates against the **mean-only baseline** (regime 1), not uniform weights:
    symmetric/fixed require CV mean MSE within 1.05x the baseline method's mean MSE, then
    pick the lowest CV covariance loss; baseline/dynamic pick the lowest CV mean MSE.
    Recovery percent is never used for selection; it is only reported.
    """

    import pandas as pd

    frame = pd.DataFrame(rows)
    selected = []
    for (ensemble, coefficient), cell in frame.groupby(["ensemble", "coefficient"]):
        # mean-only baseline: regime 1, selected on lowest CV mean MSE.
        base = (
            cell[cell.method == "baseline"]
            .groupby(["gamma", "eta"])["val_mse"].mean().reset_index()
        )
        baseline_mse = float(base["val_mse"].min())
        baseline_recovery = None

        for (method, coordinate), group in cell.groupby(["method", "coordinate"]):
            agg = group.groupby(["gamma", "eta"]).agg(
                val_mse=("val_mse", "mean"),
                val_cov_loss=("val_cov_loss", "mean"),
                recovery_percent=("recovery_percent", "mean"),
                decoy_mass=("decoy_mass", "mean"),
            ).reset_index()
            if method in ("symmetric", "fixed_reference"):
                eligible = agg[agg["val_mse"] <= 1.05 * baseline_mse]
                pool = eligible if not eligible.empty else agg
                choice = pool.sort_values("val_cov_loss").iloc[0]
            else:  # baseline, dynamic -> ordinary validation MSE
                choice = agg.sort_values("val_mse").iloc[0]
            if method == "baseline":
                baseline_recovery = float(choice["recovery_percent"])
            selected.append(
                {
                    "ensemble": ensemble,
                    "coefficient": coefficient,
                    "method": method,
                    "coordinate": coordinate,
                    "selected_gamma": float(choice["gamma"]),
                    "selected_eta": float(choice["eta"]),
                    "val_mse": float(choice["val_mse"]),
                    "baseline_val_mse": baseline_mse,
                    "mean_gate_passed": bool(choice["val_mse"] <= 1.05 * baseline_mse),
                    "val_cov_loss": float(choice["val_cov_loss"]),
                    "recovery_percent": float(choice["recovery_percent"]),
                    "decoy_mass": float(choice["decoy_mass"]),
                }
            )

        # annotate recovery gain vs the mean-only baseline for this cell
        for row in selected:
            if row["ensemble"] == ensemble and row["coefficient"] == coefficient:
                row["baseline_recovery_percent"] = baseline_recovery
                row["recovery_gain_pp"] = row["recovery_percent"] - baseline_recovery
                row["promotable"] = bool(
                    row["method"] != "baseline"
                    and row["mean_gate_passed"]
                    and row["recovery_gain_pp"] > 0.0
                )
    return selected


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())

    if args.smoke:
        grid = {"coordinates": ["logpf_projected"], "gamma": [0.1], "eta": [0.01]}
        steps, n_random, folds_limit = 150, 0, 1
        coeffs = [dict(name="published", **lock["frozen_settings"]["published"])]
        ensembles = ["AF2_filtered"]
    else:
        grid = {
            "coordinates": list(args.coordinates),
            "gamma": [0.01, 0.1, 1.0, 10.0],
            "eta": [0.0, 0.01, 0.1, 1.0],
        }
        steps, n_random, folds_limit = 2000, 2, None
        coeffs = [
            dict(name=name, **lock["frozen_settings"][name])
            for name in args.coefficient_settings
        ]
        ensembles = list(common.ENSEMBLES)

    all_rows, all_weights = [], {}
    for ensemble in ensembles:
        inputs = common.load_ensemble_inputs(ensemble)
        for coeff in coeffs:
            rows, weights = _run_cell(inputs, coeff, grid, steps, args.lr, n_random)
            if folds_limit is not None:
                rows = [r for r in rows if r["fold"] < folds_limit]
            all_rows.extend(rows)
            all_weights.update(weights)

    import pandas as pd

    pd.DataFrame(all_rows).to_csv(args.output_dir / "raw_results.csv", index=False)
    selected = _select(all_rows)
    pd.DataFrame(selected).to_csv(args.output_dir / "selected_results.csv", index=False)
    np.savez(args.output_dir / "selected_weights.npz", **all_weights)

    status = "not_evaluated" if args.smoke else "evaluated"
    decision = {
        "status": status,
        "interpretation": "known_population_frame_reweighting",
        "selection_rule": "symmetric/fixed: val MSE <= 1.05x mean-only, then min val covariance loss; "
        "baseline/dynamic: min val MSE; recovery percent never used for selection",
        "held_out_peptide": 1,
        "time_folds": "five blocked three-timepoint folds on peptides 2..14",
        "note": "smoke run: not scientific evidence" if args.smoke else
        "compare selected covariance methods against the mean-only baseline per ensemble and "
        "coefficient setting; report recovery separately and never pool coefficient/ensemble effects",
        "input_hashes": common.input_hashes(),
        "coefficient_settings": lock["frozen_settings"],
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(f"status={status}  rows={len(all_rows)}  selected={len(selected)}")
    print(f"wrote {args.output_dir / 'selected_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_recovery_reweighting",
    )
    parser.add_argument(
        "--coefficient-lock",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_recovery_coefficient_lock",
    )
    parser.add_argument("--coordinates", nargs="+", default=list(COORDINATES))
    parser.add_argument(
        "--coefficient-settings",
        nargs="+",
        default=["constrained_optimum", "scaled_published"],
        help="frozen coefficient settings to run (published is dropped by default)",
    )
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
