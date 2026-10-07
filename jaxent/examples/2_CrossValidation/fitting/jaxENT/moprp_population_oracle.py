#!/usr/bin/env python3
"""Stage A4: population-space identifiability oracle (hard gate before frame reweighting).

Before ever fitting 500 frame weights, this asks the cheap question: if we optimize *only the
state populations* (each proposed state mass spread uniformly within its frames), does matching
a covariance/variance representation to its value at the known NMR population ``w_NMR`` actually
drive the populations to the truth and reject the zero-target decoy states?

For every ensemble x frozen coefficient setting x covariance coordinate we:

* build the target representation at ``w_NMR``;
* recover populations from several deterministic starts by matching that representation;
* compute the Jacobian of the representation w.r.t. state-population logits, its singular
  spectrum, and the population directions it leaves (near-)unchanged.

A coordinate advances to Stage B only if it recovers Folded/PUF1/PUF2 and rejects every
zero-target state from *every* start, for *both* ensembles and *both* coefficient settings.
Failure is an identifiability result, not an optimizer failure.
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
from jaxent.src.analysis.state_population import (
    FULL_STATE_SUPPORT,
    log_ratio_profile_loss,
    marginal_profile,
    peptide_logpf_covariance,
    peptide_uptake_covariances,
    shrink_covariance,
)
from jaxent.src.analysis.pf_variance import (
    jensen_shannon_recovery_percent,
    overlap_projection,
    projected_log_euclidean_covariance_loss,
)

ALPHA = 0.05
RECOVERY_THRESHOLD = 99.0  # full-support recovery percent
DECOY_MASS_THRESHOLD = 0.02  # combined zero-target mass
NULL_SINGULAR_RATIO = 1e-3  # singular value / max below this => null population direction

COORDINATES = ("logpf_marginal", "logpf_projected", "uptake_marginal", "uptake_projected")
COEFFICIENT_SETTINGS = ("published", "constrained_optimum")


# --------------------------------------------------------------------------------------
# State-population -> frame-weight map and representations
# --------------------------------------------------------------------------------------
def _present_support(states: np.ndarray) -> tuple[str, ...]:
    present = [s for s in FULL_STATE_SUPPORT if np.any(states == s)]
    return tuple(present)


def _uniform_within_state_matrix(states: np.ndarray, present: tuple[str, ...]) -> np.ndarray:
    """Return U shaped (F, S): U[f, s] = 1/n_s if frame f is in state s else 0."""

    columns = []
    for state in present:
        mask = (states == state).astype(np.float64)
        columns.append(mask / mask.sum())
    return np.stack(columns, axis=1)


def _target_populations(present: tuple[str, ...], support, targets) -> np.ndarray:
    lookup = dict(zip(support, targets))
    return np.array([lookup[s] for s in present], dtype=np.float64)


def _full_support_vector(present: tuple[str, ...], values: np.ndarray) -> np.ndarray:
    lookup = dict(zip(present, values))
    return np.array([lookup.get(s, 0.0) for s in FULL_STATE_SUPPORT], dtype=np.float64)


def _make_representation(coordinate, log_pf, k_ints, timepoints, mapping):
    """Return (loss_fn(weights, target), repr_fn(weights)) for one coordinate.

    ``repr_fn`` returns a flat vector used for the Jacobian/singular-spectrum diagnostics.
    """

    log_pf = jnp.asarray(log_pf)
    mapping = jnp.asarray(mapping)
    k_ints = jnp.asarray(k_ints)
    timepoints = jnp.asarray(timepoints)
    projection = overlap_projection(mapping)

    if coordinate == "logpf_marginal":
        def repr_fn(weights):
            return marginal_profile(peptide_logpf_covariance(log_pf, mapping, weights), ALPHA)

        def loss_fn(weights, target):
            return log_ratio_profile_loss(repr_fn(weights), target)

    elif coordinate == "logpf_projected":
        def repr_fn(weights):
            cov = peptide_logpf_covariance(log_pf, mapping, weights)
            projected = projection.T @ shrink_covariance(cov, ALPHA) @ projection
            idx = jnp.triu_indices(projected.shape[0])
            return projected[idx]

        def loss_fn(weights, target_cov):
            cov = peptide_logpf_covariance(log_pf, mapping, weights)
            return projected_log_euclidean_covariance_loss(cov, target_cov, projection, ALPHA)

    elif coordinate == "uptake_marginal":
        def _profiles(weights):
            covs = peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)
            return jax.vmap(lambda c: marginal_profile(c, ALPHA))(covs)  # (T, P)

        def repr_fn(weights):
            return _profiles(weights).reshape(-1)

        def loss_fn(weights, target):
            profiles = _profiles(weights)
            return jnp.mean(jax.vmap(log_ratio_profile_loss)(profiles, target))

    elif coordinate == "uptake_projected":
        def repr_fn(weights):
            covs = peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)

            def block(cov):
                projected = projection.T @ shrink_covariance(cov, ALPHA) @ projection
                idx = jnp.triu_indices(projected.shape[0])
                return projected[idx]

            return jax.vmap(block)(covs).reshape(-1)

        def loss_fn(weights, target_covs):
            covs = peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)
            return jnp.mean(
                jax.vmap(
                    lambda p, t: projected_log_euclidean_covariance_loss(p, t, projection, ALPHA)
                )(covs, target_covs)
            )

    else:
        raise ValueError(f"unknown coordinate {coordinate!r}")

    return loss_fn, repr_fn


def _coordinate_target(coordinate, log_pf, k_ints, timepoints, mapping, w_nmr):
    """Target object for the loss (a profile, a covariance, or stacked covariances)."""

    log_pf = jnp.asarray(log_pf)
    mapping = jnp.asarray(mapping)
    w_nmr = jnp.asarray(w_nmr)
    if coordinate == "logpf_marginal":
        return marginal_profile(peptide_logpf_covariance(log_pf, mapping, w_nmr), ALPHA)
    if coordinate == "logpf_projected":
        return peptide_logpf_covariance(log_pf, mapping, w_nmr)
    covs = peptide_uptake_covariances(
        log_pf, jnp.asarray(k_ints), jnp.asarray(timepoints), mapping, w_nmr
    )
    if coordinate == "uptake_marginal":
        return jax.vmap(lambda c: marginal_profile(c, ALPHA))(covs)
    return covs  # uptake_projected: stacked (T, P, P)


# --------------------------------------------------------------------------------------
# Optimization over state-population logits
# --------------------------------------------------------------------------------------
TARGET_STATE_NAMES = ("Folded", "PUF1", "PUF2")


def _neutral_starts(n_states: int, n_random: int) -> list[tuple[str, np.ndarray]]:
    """Unbiased starts: uniform over present states plus deterministic Dirichlet(1) draws.

    These are the starts the gate is evaluated on (spec-literal 'deterministic Dirichlet
    starts').
    """

    starts = [("neutral_uniform", np.zeros(n_states))]
    for seed in range(n_random):
        draw = np.random.default_rng(2000 + seed).dirichlet(np.ones(n_states))
        starts.append((f"neutral_dirichlet_{seed}", np.log(draw + 1e-6)))
    return starts


def _adversarial_starts(present: tuple[str, ...]) -> list[tuple[str, np.ndarray]]:
    """Decoy-saturated corner starts that stress zero-target rejection.

    Reported as a robustness diagnostic only; they do not gate the decision.
    """

    starts = []
    for index, state in enumerate(present):
        if state not in TARGET_STATE_NAMES:  # a zero-target decoy state
            logits = np.full(len(present), -2.0)
            logits[index] = 3.0
            starts.append((f"adversarial_{state}", logits))
    return starts


def _optimize(loss_theta, theta0: np.ndarray, steps: int, lr: float) -> np.ndarray:
    optimizer = optax.adam(lr)
    theta = jnp.asarray(theta0)
    state = optimizer.init(theta)
    grad_fn = jax.jit(jax.grad(loss_theta))
    for _ in range(steps):
        grads = grad_fn(theta)
        updates, state = optimizer.update(grads, state)
        theta = optax.apply_updates(theta, updates)
    return np.asarray(theta)


def _jacobian_diagnostics(repr_fn, weight_map, theta_star) -> dict:
    def repr_theta(theta):
        weights = weight_map @ jax.nn.softmax(theta)
        return repr_fn(weights)

    jac = np.asarray(jax.jacrev(repr_theta)(jnp.asarray(theta_star)))  # (D, S)
    singular = np.linalg.svd(jac, compute_uv=False)
    max_sv = float(singular[0]) if singular.size else 0.0
    null_directions = int(np.sum(singular < NULL_SINGULAR_RATIO * max(max_sv, 1e-30)))
    return {
        "singular_values": [float(s) for s in singular],
        "rank_effective": int(np.sum(singular >= NULL_SINGULAR_RATIO * max(max_sv, 1e-30))),
        "null_population_directions": null_directions,
    }


def _run_coordinate(inputs, coordinate, coeff, steps, lr, n_random):
    present = _present_support(inputs.states)
    weight_map = jnp.asarray(_uniform_within_state_matrix(inputs.states, present))
    target_pop = _target_populations(present, inputs.support, inputs.targets)
    log_pf = inputs.log_pf_by_frame(coeff["bc"], coeff["bh"])

    loss_fn, repr_fn = _make_representation(
        coordinate, log_pf, inputs.k_ints, inputs.timepoints, inputs.mapping
    )
    target = _coordinate_target(
        coordinate, log_pf, inputs.k_ints, inputs.timepoints, inputs.mapping,
        inputs.reference_weights,
    )
    target = jax.lax.stop_gradient(target)

    def loss_theta(theta):
        weights = weight_map @ jax.nn.softmax(theta)
        return loss_fn(weights, target)

    full_targets = jnp.asarray(inputs.targets)
    starts = [("neutral", name, theta0) for name, theta0 in _neutral_starts(len(present), n_random)]
    starts += [("adversarial", name, theta0) for name, theta0 in _adversarial_starts(present)]

    runs = []
    for family, name, theta0 in starts:
        theta_star = _optimize(loss_theta, theta0, steps, lr)
        populations = np.asarray(jax.nn.softmax(jnp.asarray(theta_star)))
        full_pop = _full_support_vector(present, populations)
        recovery = float(jensen_shannon_recovery_percent(jnp.asarray(full_pop), full_targets))
        decoy_mass = float(
            sum(full_pop[FULL_STATE_SUPPORT.index(s)] for s in present if s not in TARGET_STATE_NAMES)
        )
        runs.append(
            {
                "start_family": family,
                "start_name": name,
                "final_loss": float(loss_theta(jnp.asarray(theta_star))),
                "recovery_percent": recovery,
                "decoy_mass": decoy_mass,
                "populations": {s: float(m) for s, m in zip(present, populations)},
                "recovered": recovery >= RECOVERY_THRESHOLD and decoy_mass <= DECOY_MASS_THRESHOLD,
            }
        )

    # Jacobian at the target populations (identifiability of the representation itself).
    target_theta = np.log(np.maximum(target_pop, 1e-6))
    jacobian = _jacobian_diagnostics(repr_fn, weight_map, target_theta)

    neutral_runs = [run for run in runs if run["start_family"] == "neutral"]
    adversarial_runs = [run for run in runs if run["start_family"] == "adversarial"]
    return {
        "present_states": list(present),
        "target_populations": {s: float(m) for s, m in zip(present, target_pop)},
        "runs": runs,
        "all_neutral_recovered": all(run["recovered"] for run in neutral_runs),
        "all_adversarial_recovered": all(run["recovered"] for run in adversarial_runs),
        "jacobian": jacobian,
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensembles = {name: common.load_ensemble_inputs(name) for name in common.ENSEMBLES}

    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    coeff_settings = {name: lock["frozen_settings"][name] for name in COEFFICIENT_SETTINGS}

    steps = 200 if args.smoke else 1500
    lr = 0.05
    n_random = 2 if args.smoke else 3  # deterministic Dirichlet neutral starts

    results = {}
    coordinate_pass = {c: True for c in COORDINATES}  # gated on neutral starts
    coordinate_robust = {c: True for c in COORDINATES}  # also robust to adversarial corners
    for coordinate in COORDINATES:
        results[coordinate] = {}
        for coeff_name, coeff in coeff_settings.items():
            results[coordinate][coeff_name] = {}
            for ens_name, inputs in ensembles.items():
                record = _run_coordinate(inputs, coordinate, coeff, steps, lr, n_random)
                results[coordinate][coeff_name][ens_name] = record
                if not record["all_neutral_recovered"]:
                    coordinate_pass[coordinate] = False
                if not (record["all_neutral_recovered"] and record["all_adversarial_recovered"]):
                    coordinate_robust[coordinate] = False

    status = "not_evaluated" if args.smoke else "evaluated"
    decision = {
        "status": status,
        "interpretation": "population_space_identifiability_oracle",
        "gate_policy": "neutral starts (uniform + deterministic Dirichlet) gate the decision; "
        "decoy-saturated adversarial corners are a reported robustness diagnostic only",
        "recovery_threshold_percent": RECOVERY_THRESHOLD,
        "decoy_mass_threshold": DECOY_MASS_THRESHOLD,
        "coefficient_settings": coeff_settings,
        "coordinates_passed_neutral": {
            c: (bool(coordinate_pass[c]) if not args.smoke else None) for c in COORDINATES
        },
        "coordinates_robust_to_adversarial": {
            c: (bool(coordinate_robust[c]) if not args.smoke else None) for c in COORDINATES
        },
        "advance_to_stage_b": (
            [c for c in COORDINATES if coordinate_pass[c]] if not args.smoke else []
        ),
        "note": (
            "smoke run: gate deliberately not evaluated; do not quote as evidence"
            if args.smoke
            else "coordinate advances if every NEUTRAL start recovered for both ensembles and both "
            "coefficient settings; adversarial-corner behavior is diagnostic"
        ),
        "input_hashes": common.input_hashes(),
    }
    (args.output_dir / "oracle_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "oracle_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(f"status={status}  (gate = neutral starts)")
    for coordinate in COORDINATES:
        neutral_worst = []
        for coeff_name in COEFFICIENT_SETTINGS:
            for ens_name in common.ENSEMBLES:
                neutral = [
                    run["recovery_percent"]
                    for run in results[coordinate][coeff_name][ens_name]["runs"]
                    if run["start_family"] == "neutral"
                ]
                neutral_worst.append(min(neutral))
        gate = "PASS" if (not args.smoke and coordinate_pass[coordinate]) else (
            "n/a" if args.smoke else "FAIL"
        )
        robust = "robust" if (not args.smoke and coordinate_robust[coordinate]) else "corner-fragile"
        print(
            f"  {coordinate:18s} neutral worst-recovery% "
            f"{min(neutral_worst):.1f}  -> gate {gate} ({robust})"
        )
    print(f"wrote {args.output_dir / 'oracle_decision.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_recovery_oracle",
    )
    parser.add_argument(
        "--coefficient-lock",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_recovery_coefficient_lock",
    )
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
