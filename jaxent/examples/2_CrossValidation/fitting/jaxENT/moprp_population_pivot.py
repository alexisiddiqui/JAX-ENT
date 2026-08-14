#!/usr/bin/env python3
"""Step 2: MoPrP population recovery and population Jacobians by HDX pivot.

The primary arm fits the real measured uptake.  A protection-factor mirror uses the
ensemble-independent exPfact solution, and a separately labelled synthetic sweep calibrates
the population-resolution floor.  Frame weights are always uniform within each state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import _moprp_recovery_common as common
import moprp_covariance_recovery as covariance_recovery
import moprp_population_oracle as oracle
from jaxent.src.analysis.pf_variance import jensen_shannon_recovery_percent, kl_to_uniform
from jaxent.src.analysis.state_population import FULL_STATE_SUPPORT
from moprp_pivot_litmus import (
    JENSEN_GUARD_TOL,
    _load_expfact_reference,
    pivot_effective_log_pf,
    pivot_observable,
)

PIVOTS = ("legacy", "fast", "slow-N")
PF_PIVOTS = ("legacy", "fast")
COEFFICIENT_SETTINGS = ("published", "constrained_optimum")
MINORITY_GRID = (0.015, 0.0216, 0.02881, 0.0448, 0.0645, 0.0929, 0.1338, 0.1927, 0.2776, 0.4)
REDUCED_NEW_FAMILY_GRID = tuple(MINORITY_GRID[i] for i in (0, 2, 4, 6, 8, 9))
POPULATION_FAMILIES = ("folded_dominant", "puf1_dominant", "balanced")
ETA = 0.01
TARGET_STATES = ("Folded", "PUF1", "PUF2")


def _expfact_data(inputs, litmus_dir: Path):
    _, overlap, reference = _load_expfact_reference(inputs)
    rows = list(csv.DictReader((litmus_dir / "residue_results.csv").open()))
    ranges = {int(r["residue_id"]): float(r["expfact_multistart_range"]) for r in rows}
    spread = np.asarray([ranges[int(r)] for r in inputs.feature_residue_ids[overlap]])
    finite = np.isfinite(spread)
    return overlap, reference, spread, finite


def _coefficient_settings(path: Path, rate_source: str) -> dict:
    payload = json.loads((path / "coefficient_lock.json").read_text())
    recorded = payload.get("rate_provenance", {}).get("rate_source", common.DEFAULT_RATE_SOURCE)
    if recorded != rate_source:
        raise ValueError(
            f"coefficient lock rate source {recorded!r} does not match requested {rate_source!r}"
        )
    by_pivot = payload["frozen_settings_by_pivot"]
    return {p: {s: by_pivot[p][s] for s in COEFFICIENT_SETTINGS} for p in PIVOTS}


def _population_map(inputs, frame_indices=None):
    present = oracle._present_support(inputs.states)
    if frame_indices is None:
        matrix = oracle._uniform_within_state_matrix(inputs.states, present)
        states = inputs.states
    else:
        states = inputs.states[np.asarray(frame_indices)]
        matrix = oracle._uniform_within_state_matrix(states, present)
    return present, jnp.asarray(matrix)


def _medoids(inputs):
    """One fixed medoid/state in published-coefficient BV feature space."""

    z = inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)
    present = oracle._present_support(inputs.states)
    indices, diagnostics = [], {}
    for state in present:
        members = np.flatnonzero(inputs.states == state)
        block = z[:, members]
        center = np.mean(block, axis=1)
        distances = np.linalg.norm(block - center[:, None], axis=0)
        local = int(np.argmin(distances))
        chosen = int(members[local])
        indices.append(chosen)
        scale = max(float(np.linalg.norm(center)), 1e-30)
        diagnostics[state] = {
            "n_frames": int(members.size),
            "frame_index": chosen,
            "distance_to_cluster_mean": float(distances[local]),
            "relative_distance_to_cluster_mean": float(distances[local] / scale),
            "cluster_rms_radius": float(np.sqrt(np.mean(distances**2))),
        }
    return np.asarray(indices), diagnostics


def _full_vector(present, population):
    lookup = dict(zip(present, np.asarray(population)))
    return np.asarray([lookup.get(s, 0.0) for s in FULL_STATE_SUPPORT])


def _scores(present, population, truth):
    full = _full_vector(present, population)
    recovery = float(jensen_shannon_recovery_percent(jnp.asarray(full), jnp.asarray(truth)))
    decoy = float(sum(full[FULL_STATE_SUPPORT.index(s)] for s in present if s not in TARGET_STATES))
    tvd = float(0.5 * np.sum(np.abs(full - truth)))
    signed_error = {
        state: float(recovered - expected)
        for state, recovered, expected in zip(FULL_STATE_SUPPORT, full, truth)
    }
    positive = full[full > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    folded_mass = float(full[FULL_STATE_SUPPORT.index("Folded")])
    return {
        "recovery_percent": recovery,
        "decoy_mass": decoy,
        "tvd": tvd,
        "signed_error": signed_error,
        "shannon_entropy": entropy,
        "folded_mass": folded_mass,
        "populations": {s: float(v) for s, v in zip(present, population)},
    }


def _jacobian(repr_weights, weight_map, theta, present):
    def fn(x):
        return repr_weights(weight_map @ jax.nn.softmax(x))

    jac = np.asarray(jax.jacrev(fn)(jnp.asarray(theta)))
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    maximum = float(singular[0]) if singular.size else 0.0
    cutoff = oracle.NULL_SINGULAR_RATIO * max(maximum, 1e-30)
    finite_difference = np.empty_like(jac)
    eps = 1e-4
    for column in range(theta.size):
        delta = np.zeros_like(theta)
        delta[column] = eps
        finite_difference[:, column] = (
            np.asarray(fn(theta + delta)) - np.asarray(fn(theta - delta))
        ) / (2 * eps)
    fd_error = float(np.max(np.abs(jac - finite_difference)))
    null = []
    for row in np.flatnonzero(singular < cutoff):
        null.append({s: float(v) for s, v in zip(present, vh[row])})
    return {
        "singular_values": singular.tolist(),
        "effective_rank": int(np.sum(singular >= cutoff)),
        "null_population_directions": null,
        "finite_difference_max_abs": fd_error,
    }


def _fit(loss_theta, present, truth, steps, lr, n_random, *, all_starts=True):
    neutral = oracle._neutral_starts(len(present), n_random)
    starts = [("neutral", n, t) for n, t in (neutral if all_starts else neutral[:1])]
    if all_starts:
        starts += [("adversarial", n, t) for n, t in oracle._adversarial_starts(present)]
    if all_starts:
        optimizer = oracle.optax.adam(lr)
        theta_batch = jnp.asarray(np.stack([theta0 for _, _, theta0 in starts]))
        state = optimizer.init(theta_batch)
        grad_fn = jax.jit(jax.vmap(jax.grad(loss_theta)))
        for _ in range(steps):
            grads = grad_fn(theta_batch)
            updates, state = optimizer.update(grads, state)
            theta_batch = oracle.optax.apply_updates(theta_batch, updates)
        optimized = np.asarray(theta_batch)
    else:
        optimized = np.stack([
            oracle._optimize(loss_theta, theta0, steps, lr) for _, _, theta0 in starts
        ])
    runs = []
    for (family, name, _), theta in zip(starts, optimized):
        pop = np.asarray(jax.nn.softmax(theta))
        scores = _scores(present, pop, truth)
        runs.append({
            "start_family": family,
            "start_name": name,
            "objective": float(loss_theta(jnp.asarray(theta))),
            **scores,
            "theta": theta.tolist(),
        })
    return min(runs, key=lambda r: r["objective"]), runs


def _target_population(inputs, minority, family="folded_dominant"):
    puf_ratio = inputs.targets[FULL_STATE_SUPPORT.index("PUF1")] / (
        inputs.targets[FULL_STATE_SUPPORT.index("PUF1")]
        + inputs.targets[FULL_STATE_SUPPORT.index("PUF2")]
    )
    folded_ratio = inputs.targets[FULL_STATE_SUPPORT.index("Folded")] / (
        inputs.targets[FULL_STATE_SUPPORT.index("Folded")]
        + inputs.targets[FULL_STATE_SUPPORT.index("PUF2")]
    )
    full = np.zeros(len(FULL_STATE_SUPPORT), dtype=float)
    folded = FULL_STATE_SUPPORT.index("Folded")
    puf1 = FULL_STATE_SUPPORT.index("PUF1")
    puf2 = FULL_STATE_SUPPORT.index("PUF2")
    if family == "folded_dominant":
        full[folded] = 1.0 - minority
        full[puf1] = minority * puf_ratio
        full[puf2] = minority * (1.0 - puf_ratio)
    elif family == "puf1_dominant":
        full[puf1] = 1.0 - minority
        full[folded] = minority * folded_ratio
        full[puf2] = minority * (1.0 - folded_ratio)
    elif family == "balanced":
        full[folded] = 1.0 / 3.0 + minority
        full[puf1] = full[puf2] = 1.0 / 3.0 - minority / 2.0
    else:
        raise ValueError(f"unknown population family: {family}")
    return full


def _null_population(inputs):
    """Population induced by uniform weights over all frames."""

    return np.asarray([
        np.mean(inputs.states == state) for state in FULL_STATE_SUPPORT
    ])


def _run_fit(inputs, log_pf, present, weight_map, pivot, target, target_space, truth, steps, lr,
             n_random, overlap, reference_pf, spread_finite, eta=ETA, all_starts=True,
             report_observed_uptake_mse=False, metric=None):
    keep = np.ones(inputs.mapping.shape[0], dtype=bool)
    keep[common.PEPTIDE1_INDEX] = False
    mapping = inputs.mapping[keep]
    observed = inputs.observed_uptake[keep]
    uniform_pop = jnp.full(len(present), 1.0 / len(present))
    uniform_weights = weight_map @ uniform_pop

    def uptake_repr(weights):
        return pivot_observable(log_pf, inputs.k_ints, inputs.timepoints, mapping, weights, pivot)

    if target_space == "uptake":
        target_array = jnp.asarray(target)
        if metric is None:
            baseline = jnp.mean((uptake_repr(uniform_weights) - target_array) ** 2)
            baseline = jnp.maximum(baseline, 1e-12)

            def repr_weights(weights):
                return uptake_repr(weights).reshape(-1)

            def loss(theta):
                population = jax.nn.softmax(theta)
                prediction = uptake_repr(weight_map @ population)
                return jnp.mean((prediction - target_array) ** 2) / baseline + eta * kl_to_uniform(population)
        else:
            baseline = metric(uptake_repr(uniform_weights) - target_array)
            baseline = jnp.maximum(baseline, 1e-12)

            def repr_weights(weights):
                return uptake_repr(weights).reshape(-1)

            def loss(theta):
                population = jax.nn.softmax(theta)
                prediction = uptake_repr(weight_map @ population)
                return metric(prediction - target_array) / baseline + eta * kl_to_uniform(population)
    else:
        target_array = jnp.asarray(target)
        overlap_j = jnp.asarray(overlap)

        def pf_repr(weights):
            return pivot_effective_log_pf(log_pf, weights, pivot)[overlap_j]

        baseline = jnp.maximum(jnp.mean((pf_repr(uniform_weights) - target_array) ** 2), 1e-12)

        def repr_weights(weights):
            return pf_repr(weights)

        def loss(theta):
            population = jax.nn.softmax(theta)
            prediction = pf_repr(weight_map @ population)
            return jnp.mean((prediction - target_array) ** 2) / baseline + eta * kl_to_uniform(population)

    best, runs = _fit(loss, present, truth, steps, lr, n_random, all_starts=all_starts)
    population = np.asarray([best["populations"][s] for s in present])
    weights = np.asarray(weight_map @ jnp.asarray(population))
    uptake = np.asarray(uptake_repr(weights))
    uptake_reference = (
        observed if report_observed_uptake_mse or target_space != "uptake" else np.asarray(target)
    )
    uptake_mse = float(np.mean((uptake - uptake_reference) ** 2))
    pf_rmse = None
    pf_band = None
    if pivot in PF_PIVOTS:
        predicted_pf = np.asarray(pivot_effective_log_pf(log_pf, weights, pivot))[overlap]
        pf_rmse = float(np.sqrt(np.mean((predicted_pf - reference_pf) ** 2)))
        finite_spread = spread_finite[np.isfinite(spread_finite)]
        pf_band = {
            "median_solution_range": float(np.median(finite_spread)),
            "mean_solution_range": float(np.mean(finite_spread)),
        }
    theta = np.asarray(best["theta"])
    return {
        "best": {k: v for k, v in best.items() if k != "theta"},
        "runs": [{k: v for k, v in r.items() if k != "theta"} for r in runs],
        "uptake_mse": uptake_mse,
        "pf_rmse": pf_rmse,
        "pf_reference_band": pf_band,
        "jacobian": _jacobian(repr_weights, weight_map, theta, present),
    }


def run(args):
    if args.legacy_reproduction:
        args.families = ("folded_dominant",)
        args.sweep_starts = "single"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = common.load_ensemble_inputs("AF2_MSAss", args.rate_source)
    coeffs = _coefficient_settings(args.coefficient_lock, args.rate_source)
    overlap, reference_pf, spread, finite_spread = _expfact_data(inputs, args.litmus_dir)
    medoid_indices, medoid_diagnostics = _medoids(inputs)
    modes = {
        "full_500": (np.arange(inputs.n_frames),),
        "five_medoids": (medoid_indices,),
    }
    steps = 30 if args.smoke else args.steps
    n_random = 0 if args.smoke else 3
    primary, sanity = {}, {}

    for mode, (indices,) in modes.items():
        present, weight_map = _population_map(inputs, None if mode == "full_500" else indices)
        log_pf_source = inputs.log_pf_by_frame
        primary[mode], sanity[mode] = {}, {}
        for pivot in PIVOTS:
            primary[mode][pivot], sanity[mode][pivot] = {}, {}
            for setting in COEFFICIENT_SETTINGS:
                coeff = coeffs[pivot][setting]
                log_pf = log_pf_source(coeff["bc"], coeff["bh"])
                if mode == "five_medoids":
                    log_pf = log_pf[:, indices]
                primary[mode][pivot][setting] = _run_fit(
                    inputs, log_pf, present, weight_map, pivot, inputs.observed_uptake[1:],
                    "uptake", inputs.targets, steps, args.lr, n_random, overlap, reference_pf,
                    spread, ETA,
                )
                target_weights = weight_map @ jnp.asarray(
                    oracle._target_populations(present, inputs.support, inputs.targets)
                )
                synthetic = np.asarray(pivot_observable(
                    log_pf, inputs.k_ints, inputs.timepoints, inputs.mapping[1:], target_weights, pivot
                ))
                sanity[mode][pivot][setting] = _run_fit(
                    inputs, log_pf, present, weight_map, pivot, synthetic, "uptake", inputs.targets,
                    steps, args.lr, n_random, overlap, reference_pf, spread, 0.0,
                )

    # PF mirror is a full-ensemble observable only; slow-N has no scalar PF.
    pf_mirror = {}
    present, weight_map = _population_map(inputs)
    for pivot in PF_PIVOTS:
        pf_mirror[pivot] = {}
        for setting in COEFFICIENT_SETTINGS:
            coeff = coeffs[pivot][setting]
            log_pf = inputs.log_pf_by_frame(coeff["bc"], coeff["bh"])
            pf_mirror[pivot][setting] = _run_fit(
                inputs, log_pf, present, weight_map, pivot, reference_pf, "pf", inputs.targets,
                steps, args.lr, n_random, overlap, reference_pf, spread, ETA,
            )

    # Synthetic 3x3 instrument calibration, AF2-MSAss full ensemble.
    null_population = _null_population(inputs)
    sweep, sweep_rows = {}, []
    for family in args.families:
        sweep[family] = {}
        family_grid = (
            REDUCED_NEW_FAMILY_GRID
            if args.reduced_new_family_grid and family != "folded_dominant"
            else MINORITY_GRID
        )
        for setting in COEFFICIENT_SETTINGS:
            sweep[family][setting] = {}
            for target_pivot in PIVOTS:
                sweep[family][setting][target_pivot] = {}
                target_coeff = coeffs[target_pivot][setting]
                target_log_pf = inputs.log_pf_by_frame(target_coeff["bc"], target_coeff["bh"])
                for fitter_pivot in PIVOTS:
                    cell = []
                    fitter_coeff = coeffs[fitter_pivot][setting]
                    fitter_log_pf = inputs.log_pf_by_frame(fitter_coeff["bc"], fitter_coeff["bh"])
                    for minority in family_grid:
                        truth = _target_population(inputs, minority, family)
                        null_scores = _scores(FULL_STATE_SUPPORT, null_population, truth)
                        target_pop = oracle._target_populations(present, FULL_STATE_SUPPORT, truth)
                        target_weights = weight_map @ jnp.asarray(target_pop)
                        target = np.asarray(pivot_observable(
                            target_log_pf, inputs.k_ints, inputs.timepoints, inputs.mapping[1:],
                            target_weights, target_pivot,
                        ))
                        record = _run_fit(
                            inputs, fitter_log_pf, present, weight_map, fitter_pivot, target, "uptake",
                            truth, steps, args.lr, n_random, overlap, reference_pf, spread, 0.0,
                            all_starts=args.sweep_starts == "all",
                            report_observed_uptake_mse=args.legacy_reproduction,
                        )
                        best = record["best"]
                        compact = {
                            "minority_mass": minority,
                            "recovery_percent": best["recovery_percent"],
                            "null_recovery_percent": null_scores["recovery_percent"],
                            "recovery_gain_over_null": (
                                best["recovery_percent"] - null_scores["recovery_percent"]
                            ),
                            "tvd": best["tvd"],
                            "null_tvd": null_scores["tvd"],
                            "tvd_gain_over_null": null_scores["tvd"] - best["tvd"],
                            "decoy_mass": best["decoy_mass"],
                            "shannon_entropy": best["shannon_entropy"],
                            "folded_mass": best["folded_mass"],
                            "signed_error": best["signed_error"],
                            "uptake_mse": record["uptake_mse"],
                            "selected_start_family": best["start_family"],
                            "selected_start_name": best["start_name"],
                            "n_starts": len(record["runs"]),
                        }
                        cell.append(compact)
                        if args.legacy_reproduction:
                            sweep_rows.append({
                                "coefficient_setting": setting,
                                "target_pivot": target_pivot,
                                "fitter_pivot": fitter_pivot,
                                "minority_mass": minority,
                                "recovery_percent": best["recovery_percent"],
                                "decoy_mass": best["decoy_mass"],
                                "uptake_mse": record["uptake_mse"],
                            })
                        else:
                            csv_compact = {
                                k: v for k, v in compact.items() if k != "signed_error"
                            }
                            csv_compact.update({
                                f"signed_error_{state}": compact["signed_error"][state]
                                for state in FULL_STATE_SUPPORT
                            })
                            sweep_rows.append({
                                "population_family": family,
                                "coefficient_setting": setting,
                                "target_pivot": target_pivot,
                                "fitter_pivot": fitter_pivot,
                                **csv_compact,
                            })
                    qualified = [
                        r["minority_mass"] for r in cell
                        if r["recovery_gain_over_null"] > 0.0 and r["tvd_gain_over_null"] > 0.0
                    ]
                    sweep[family][setting][target_pivot][fitter_pivot] = {
                        "positive_gain_floor": min(qualified) if qualified else None,
                        "runs": cell,
                    }

    # Global algebra checks.
    jensen_violations = 0
    for pivot in PF_PIVOTS:
        for setting in COEFFICIENT_SETTINGS:
            coeff = coeffs[pivot][setting]
            z = inputs.log_pf_by_frame(coeff["bc"], coeff["bh"])
            for record in primary["full_500"][pivot][setting]["runs"]:
                pop = np.asarray([record["populations"][s] for s in present])
                w = np.asarray(weight_map @ jnp.asarray(pop))
                legacy = np.asarray(pivot_effective_log_pf(z, w, "legacy"))
                fast = np.asarray(pivot_effective_log_pf(z, w, "fast"))
                jensen_violations += int(np.sum(fast > legacy + JENSEN_GUARD_TOL))
    z = inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)
    one = np.zeros(inputs.n_frames)
    one[0] = 1.0
    curves = [np.asarray(pivot_observable(z, inputs.k_ints, inputs.timepoints, inputs.mapping, one, p)) for p in PIVOTS]
    degenerate = max(float(np.max(np.abs(c - curves[0]))) for c in curves[1:])
    regression_old = np.asarray(covariance_recovery._predict_uptake(
        z, inputs.k_ints, inputs.timepoints, inputs.mapping, inputs.reference_weights
    )).T
    regression_new = np.asarray(pivot_observable(
        z, inputs.k_ints, inputs.timepoints, inputs.mapping, inputs.reference_weights, "legacy"
    ))
    legacy_regression = float(np.max(np.abs(regression_old - regression_new)))
    litmus = json.loads((args.litmus_dir / "moprp_pivot_litmus.json").read_text())
    litmus_rate_source = litmus.get("rate_provenance", {}).get(
        "rate_source", common.DEFAULT_RATE_SOURCE
    )
    if litmus_rate_source != args.rate_source:
        raise ValueError(
            f"litmus rate source {litmus_rate_source!r} does not match requested {args.rate_source!r}"
        )
    legacy_pf = np.asarray(pivot_effective_log_pf(z, inputs.reference_weights, "legacy"))[overlap]
    fast_pf = np.asarray(pivot_effective_log_pf(z, inputs.reference_weights, "fast"))[overlap]
    mirror_consistency = {
        "legacy_rmse": float(np.max(np.abs(legacy_pf - np.asarray([float(r["legacy_log_pf"]) for r in csv.DictReader((args.litmus_dir / "residue_results.csv").open())])))),
        "fast_rmse": float(np.max(np.abs(fast_pf - np.asarray([float(r["fast_log_pf"]) for r in csv.DictReader((args.litmus_dir / "residue_results.csv").open())])))),
    }
    payload = {
        "description": "MoPrP step-2 population recovery by pivot; real primary plus synthetic calibration",
        "rate_provenance": common.rate_source_provenance(args.rate_source),
        "ensemble": "AF2_MSAss",
        "execution_mode": "smoke" if args.smoke else "full",
        "optimization_steps": steps,
        "coefficient_settings": coeffs,
        "eta_primary": ETA,
        "primary_real_uptake": primary,
        "pf_mirror": pf_mirror,
        "synthetic_resolution_sweep": sweep,
        "synthetic_sweep_configuration": {
            "population_families": list(args.families),
            "minority_grid_by_family": {
                family: list(
                    REDUCED_NEW_FAMILY_GRID
                    if args.reduced_new_family_grid and family != "folded_dominant"
                    else MINORITY_GRID
                )
                for family in args.families
            },
            "starts": args.sweep_starts,
            "null_population": {
                state: float(value) for state, value in zip(FULL_STATE_SUPPORT, null_population)
            },
            "qualification": "recovery_gain_over_null > 0 and tvd_gain_over_null > 0",
            "balanced_parameterization": "Folded=1/3+minority; PUF1=PUF2=1/3-minority/2",
        },
        "medoids": medoid_diagnostics,
        "verification": {
            "sanity_self_consistent": sanity,
            "jensen_violations": jensen_violations,
            "degenerate_curve_max_abs": degenerate,
            "legacy_covariance_regime1_max_abs": legacy_regression,
            "pf_mirror_litmus_max_abs": mirror_consistency,
            "pf_overlap_count": int(overlap.sum()),
            "pf_spread_testable_count": int(finite_spread.sum()),
            "litmus_gate": litmus["resolvability_gate"],
        },
        "caveats": [
            "The real target has no target-pivot semantics; target-pivot exists only in the synthetic sweep.",
            "slow-N is uptake-only because a curve mixture has no exact scalar effective PF.",
            "The 49-residue PF subset is non-random and constrained by the peptide map.",
            "exPfact is ensemble/pivot-independent but fitted from the same uptake and is smoothed and degenerate.",
            "w_NMR is NMR-derived pseudo-truth with assumed uniform within-state frame weights.",
        ],
        "input_hashes": common.input_hashes(args.rate_source),
    }
    (args.output_dir / "population_pivot_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "synthetic_resolution_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"wrote {args.output_dir / 'population_pivot_results.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "_moprp_population_oracle_pivot")
    parser.add_argument("--coefficient-lock", type=Path, default=Path(__file__).parent / "_moprp_recovery_coefficient_lock")
    parser.add_argument("--litmus-dir", type=Path, default=Path(__file__).parent / "_moprp_pivot_litmus")
    parser.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sweep-starts", choices=("single", "all"), default="all")
    parser.add_argument(
        "--families", nargs="+", choices=POPULATION_FAMILIES, default=POPULATION_FAMILIES,
    )
    parser.add_argument(
        "--legacy-reproduction", action="store_true",
        help="emit the original folded/single-start 180-row CSV, including its legacy MSE field",
    )
    parser.add_argument(
        "--reduced-new-family-grid", action="store_true",
        help="use six preselected grid points for PUF1-dominant and balanced families",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
