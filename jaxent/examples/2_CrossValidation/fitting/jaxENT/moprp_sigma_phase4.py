#!/usr/bin/env python3
"""Phase 4: substitute the frozen uptake covariance in the population-pivot fit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import _moprp_recovery_common as common
import moprp_population_pivot as pivot_fit
import moprp_sigma_noise_model as noise

HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "_moprp_sigma_noise_model" / "target_modes.npz"
DEFAULT_SHIPPED = HERE.parents[1] / "data" / "_MoPrP_covariance_matrices" / "Sigma.npz"
DEFAULT_OUTPUT = HERE / "_moprp_sigma_phase4"
DEFAULT_SHIPPED_CALIBRATION = HERE / "_moprp_kint_sensitivity" / "moprp_shipped"
DEFAULT_PIVOT_SHA256 = "17971aa7c77a7f6ff5abc3ed9833d8ab39e3879137d49fca23a32ceb351e5323"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_time_major_indices(n_peptides: int, n_timepoints: int, drop: int) -> np.ndarray:
    """Indices retaining every peptide except ``drop`` in a time-major vector."""
    if not 0 <= drop < n_peptides:
        raise ValueError("drop must identify a peptide")
    indices = np.asarray(
        [i for i in range(n_peptides * n_timepoints) if i % n_peptides != drop], dtype=int
    )
    assert indices.size == (n_peptides - 1) * n_timepoints
    assert np.array_equal(indices, np.unique(indices))
    assert np.all(np.diff(indices) > 0)
    contiguous = np.arange(n_timepoints, n_peptides * n_timepoints)
    assert not np.array_equal(indices, contiguous)
    return indices


def marginal_covariance(covariance, indices) -> tuple[np.ndarray, np.ndarray]:
    """Take a Gaussian marginal submatrix and refactorise it."""
    covariance = np.asarray(covariance, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)
    marginal = covariance[np.ix_(indices, indices)]
    np.testing.assert_allclose(marginal, marginal.T, rtol=0.0, atol=1e-12)
    chol = np.linalg.cholesky(marginal)
    return marginal, chol


def load_arms(
    target_path: Path = DEFAULT_TARGET,
    shipped_path: Path = DEFAULT_SHIPPED,
) -> list[tuple[str, object]]:
    """Load the registered eye, shipped-precision, and frozen-joint metrics."""
    target = np.load(target_path)
    covariance = np.asarray(target["covariance"])
    n_peptides, n_timepoints = np.asarray(target["mean"]).shape
    indices = retained_time_major_indices(n_peptides, n_timepoints, common.PEPTIDE1_INDEX)
    _, chol_np = marginal_covariance(covariance, indices)
    frozen_chol = jax.lax.stop_gradient(jnp.asarray(chol_np))

    shipped = np.load(shipped_path)
    precision = np.asarray(shipped["Sigma_inv"], dtype=np.float64)
    keep_peptides = np.arange(n_peptides) != common.PEPTIDE1_INDEX
    precision = precision[np.ix_(keep_peptides, keep_peptides)]
    precision *= precision.shape[0] / np.trace(precision)
    shipped_precision = jax.lax.stop_gradient(jnp.asarray(precision))

    def eye_mse(residual_pt):
        return jnp.mean(jnp.asarray(residual_pt) ** 2)

    def shipped_sigma(residual_pt):
        residual_pt = jnp.asarray(residual_pt)
        return 0.5 * jnp.einsum("pt,pq,qt->", residual_pt, shipped_precision, residual_pt)

    zero_constant = noise.gaussian_nll_from_cholesky(jnp.zeros(chol_np.shape[0]), frozen_chol)

    def frozen_joint(residual_pt):
        residual = noise.vectorize_time_major(jnp.asarray(residual_pt))
        return noise.gaussian_nll_from_cholesky(residual, frozen_chol) - zero_constant

    frozen_joint.cholesky = frozen_chol
    frozen_joint.covariance = covariance
    frozen_joint.indices = indices
    return [("eye_mse", eye_mse), ("shipped_sigma", shipped_sigma), ("frozen_joint", frozen_joint)]


def _block_diagonal_check(covariance: np.ndarray, n_peptides: int, n_timepoints: int) -> dict:
    block = covariance.copy()
    for j in range(n_timepoints):
        for k in range(n_timepoints):
            if j != k:
                block[
                    j * n_peptides : (j + 1) * n_peptides,
                    k * n_peptides : (k + 1) * n_peptides,
                ] = 0.0
    chol = np.linalg.cholesky(block)
    residual = np.linspace(-0.2, 0.3, block.shape[0])
    joint = float(noise.gaussian_nll_from_cholesky(residual, chol))
    summed = 0.0
    for j in range(n_timepoints):
        sl = slice(j * n_peptides, (j + 1) * n_peptides)
        summed += float(noise.gaussian_nll_from_cholesky(residual[sl], chol[sl, sl]))
    difference = abs(joint - summed)
    return {"passed": difference <= 1e-10, "absolute_difference": difference, "tolerance": 1e-10}


def _summary(rows: list[dict]) -> list[dict]:
    result = []
    for arm in ("eye_mse", "shipped_sigma", "frozen_joint"):
        selected = [row for row in rows if row["arm"] == arm]
        recovery = np.asarray([row["recovery_percent"] for row in selected])
        decoy = np.asarray([row["decoy_mass"] for row in selected])
        result.append({
            "arm": arm,
            "n_fits": len(selected),
            "mean_recovery_percent": float(recovery.mean()),
            "min_recovery_percent": float(recovery.min()),
            "max_recovery_percent": float(recovery.max()),
            "mean_decoy_mass": float(decoy.mean()),
            "min_decoy_mass": float(decoy.min()),
            "max_decoy_mass": float(decoy.max()),
        })
    return result


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return float(np.log(np.expm1(value)))


def _run_bv_fit(inputs, present, weight_map, pivot, metric, coefficient_starts, steps, lr):
    """Fit state populations and non-negative BV coefficients jointly."""
    target = jnp.asarray(inputs.observed_uptake[1:])
    heavy = jnp.asarray(inputs.heavy_contacts)
    acceptor = jnp.asarray(inputs.acceptor_contacts)
    mapping = inputs.mapping[1:]
    uniform_pop = jnp.full(len(present), 1.0 / len(present))
    published_log_pf = common.PUBLISHED_BC * heavy + common.PUBLISHED_BH * acceptor
    reference_residual = pivot_fit.pivot_observable(
        published_log_pf, inputs.k_ints, inputs.timepoints, mapping,
        weight_map @ uniform_pop, pivot,
    ) - target
    baseline = jnp.maximum(metric(reference_residual), 1e-12)

    def unpack(parameters):
        population = jax.nn.softmax(parameters[:-2])
        bc, bh = jax.nn.softplus(parameters[-2:])
        return population, bc, bh

    def loss(parameters):
        population, bc, bh = unpack(parameters)
        log_pf = bc * heavy + bh * acceptor
        prediction = pivot_fit.pivot_observable(
            log_pf, inputs.k_ints, inputs.timepoints, mapping,
            weight_map @ population, pivot,
        )
        return metric(prediction - target) / baseline + pivot_fit.ETA * pivot_fit.kl_to_uniform(population)

    population_starts = pivot_fit.oracle._neutral_starts(len(present), 3)
    population_starts += pivot_fit.oracle._adversarial_starts(present)
    starts = []
    for setting, coefficient in coefficient_starts.items():
        coefficient_theta = np.asarray([
            _inverse_softplus(coefficient["bc"]), _inverse_softplus(coefficient["bh"]),
        ])
        for family, name, logits in [
            ("neutral", name, logits) for name, logits in population_starts[:4]
        ] + [
            ("adversarial", name, logits) for name, logits in population_starts[4:]
        ]:
            starts.append((setting, family, name, np.concatenate([logits, coefficient_theta])))
    optimizer = pivot_fit.oracle.optax.adam(lr)
    batch = jnp.asarray(np.stack([start[-1] for start in starts]))
    state = optimizer.init(batch)
    grad_fn = jax.jit(jax.vmap(jax.grad(loss)))
    for _ in range(steps):
        gradients = grad_fn(batch)
        updates, state = optimizer.update(gradients, state)
        batch = pivot_fit.oracle.optax.apply_updates(batch, updates)
    candidates = []
    for metadata, parameters in zip(starts, np.asarray(batch)):
        population, bc, bh = unpack(jnp.asarray(parameters))
        scores = pivot_fit._scores(present, np.asarray(population), inputs.targets)
        candidates.append({
            "initial_coefficient_setting": metadata[0],
            "start_family": metadata[1],
            "start_name": metadata[2],
            "final_loss": float(loss(jnp.asarray(parameters))),
            "bc": float(bc),
            "bh": float(bh),
            **scores,
        })
    return min(candidates, key=lambda candidate: candidate["final_loss"]), candidates


def run_bv(args) -> None:
    if args.rate_source != "moprp_shipped":
        raise ValueError("Phase 4 BV fit is registered on --rate-source moprp_shipped")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = common.load_ensemble_inputs("AF2_MSAss", args.rate_source)
    coefficients = pivot_fit._coefficient_settings(args.coefficient_lock, args.rate_source)
    present, weight_map = pivot_fit._population_map(inputs)
    arms = load_arms(args.frozen_target, args.shipped_sigma)
    steps = 30 if args.smoke else args.steps
    rows = []
    for pivot in pivot_fit.PIVOTS:
        for arm_name, metric in arms:
            best, candidates = _run_bv_fit(
                inputs, present, weight_map, pivot, metric, coefficients[pivot], steps, args.lr
            )
            row = {
                "pivot": pivot,
                "arm": arm_name,
                "recovery_percent": best["recovery_percent"],
                "decoy_mass": best["decoy_mass"],
                "final_loss": best["final_loss"],
                "bc": best["bc"],
                "bh": best["bh"],
                "initial_coefficient_setting": best["initial_coefficient_setting"],
                "selected_start_family": best["start_family"],
                "selected_start_name": best["start_name"],
                "n_starts": len(candidates),
            }
            row.update({f"population_{state}": best["populations"].get(state, 0.0) for state in present})
            rows.append(row)
    with (args.output_dir / "phase4_bv_arms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    with (args.output_dir / "phase4_bv_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"summary": summary, "rows": rows}, indent=2))


def run(args) -> None:
    if args.rate_source != "moprp_shipped":
        raise ValueError("Phase 4 is registered on --rate-source moprp_shipped")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = common.load_ensemble_inputs("AF2_MSAss", args.rate_source)
    coeffs = pivot_fit._coefficient_settings(args.coefficient_lock, args.rate_source)
    overlap, reference_pf, spread, _ = pivot_fit._expfact_data(inputs, args.litmus_dir)
    present, weight_map = pivot_fit._population_map(inputs)
    arms = load_arms(args.frozen_target, args.shipped_sigma)
    frozen_metric = dict(arms)["frozen_joint"]
    chol_before = np.asarray(frozen_metric.cholesky).copy()
    steps, n_random = (30, 0) if args.smoke else (args.steps, 3)
    rows = []
    for pivot in pivot_fit.PIVOTS:
        for setting in pivot_fit.COEFFICIENT_SETTINGS:
            coeff = coeffs[pivot][setting]
            log_pf = inputs.log_pf_by_frame(coeff["bc"], coeff["bh"])
            for arm_name, metric in arms:
                record = pivot_fit._run_fit(
                    inputs, log_pf, present, weight_map, pivot, inputs.observed_uptake[1:],
                    "uptake", inputs.targets, steps, args.lr, n_random, overlap, reference_pf,
                    spread, pivot_fit.ETA, metric=metric,
                )
                best = record["best"]
                row = {
                    "pivot": pivot,
                    "coefficient_setting": setting,
                    "arm": arm_name,
                    "recovery_percent": best["recovery_percent"],
                    "decoy_mass": best["decoy_mass"],
                    "final_loss": best["objective"],
                }
                row.update({f"population_{state}": best["populations"].get(state, 0.0) for state in present})
                rows.append(row)

    chol_after = np.asarray(frozen_metric.cholesky)
    invariant = {
        "passed": bool(np.array_equal(chol_before, chol_after)),
        "bitwise_identical": bool(np.array_equal(chol_before, chol_after)),
        "stop_gradient_carried": True,
        "before_sha256": hashlib.sha256(chol_before.tobytes()).hexdigest(),
        "after_sha256": hashlib.sha256(chol_after.tobytes()).hexdigest(),
    }
    covariance, _ = marginal_covariance(
        np.load(args.frozen_target)["covariance"], frozen_metric.indices
    )
    block_check = _block_diagonal_check(covariance, inputs.mapping.shape[0] - 1, inputs.timepoints.size)
    default_artifact = args.default_artifact
    default_check = {
        "passed": default_artifact.exists() and _sha256(default_artifact) == args.expected_default_sha256,
        "expected_sha256": args.expected_default_sha256,
        "observed_sha256": _sha256(default_artifact) if default_artifact.exists() else None,
    }
    checks = {
        "default_paths_byte_identical": default_check,
        "joint_equals_block_diagonal": block_check,
        "frozen_sigma_invariant": invariant,
    }
    gates_passed = all(check["passed"] for check in checks.values())

    summary = _summary(rows)
    with (args.output_dir / "phase4_arms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "phase4_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    target_data = np.load(args.frozen_target)
    manifest = {
        "phase": 4,
        "execution_mode": "smoke" if args.smoke else "full",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip(),
        "rate_provenance": common.rate_source_provenance(args.rate_source),
        "frozen_target": {"path": str(args.frozen_target.resolve()), "sha256": _sha256(args.frozen_target)},
        "shipped_sigma": {"path": str(args.shipped_sigma.resolve()), "sha256": _sha256(args.shipped_sigma)},
        "slice": {
            "dropped_peptide_index": common.PEPTIDE1_INDEX,
            "indices": frozen_metric.indices.tolist(),
            "input_cells": int(target_data["covariance"].shape[0]),
            "retained_cells": int(frozen_metric.indices.size),
            "vector_order": str(target_data["vector_order"]),
        },
        "joint_log_determinant_dropped": True,
        "normalisation": "each metric divided by its own uniform-population baseline",
        "regression_checks": checks,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"summary": summary, "regression_checks": checks}, indent=2))
    if not gates_passed:
        raise AssertionError(f"Phase 4 regression gate failed: {checks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coefficient-lock", type=Path, default=DEFAULT_SHIPPED_CALIBRATION / "coefficient_lock")
    parser.add_argument("--litmus-dir", type=Path, default=DEFAULT_SHIPPED_CALIBRATION / "pivot_litmus")
    parser.add_argument("--frozen-target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--shipped-sigma", type=Path, default=DEFAULT_SHIPPED)
    parser.add_argument("--default-artifact", type=Path, default=HERE / "_moprp_population_oracle_pivot" / "population_pivot_results.json")
    parser.add_argument("--expected-default-sha256", default=DEFAULT_PIVOT_SHA256)
    parser.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default="moprp_shipped")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fit-bv", action="store_true")
    args = parser.parse_args()
    if args.fit_bv:
        run_bv(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
