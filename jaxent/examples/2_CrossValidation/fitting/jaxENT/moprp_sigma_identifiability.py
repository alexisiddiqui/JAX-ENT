#!/usr/bin/env python3
"""Phase-2 identifiability simulations for the MoPrP joint HDX noise model.

Synthetic observations are generated at the real 14-peptide x 15-timepoint geometry.
The full study is intentionally explicit and resumable: each question writes its own CSV,
and ``--profile smoke`` provides a fast pipeline check without pretending to be a scientific
result.  Use ``--profile full`` for the pre-registered replicate grids.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import MDAnalysis as mda
import jax
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import _moprp_recovery_common as common
import moprp_sigma_noise_model as noise
from jaxent.src.analysis.elastic_network import anm_covariance

STRUCTURE = common.BASE / "data/MoPrP_max_plddt_4334.pdb"
STRUCTURE_SHA256 = "1fd3090c54142ff464bab9f46facedea5bbad14621f8abadf46079958ff60453"
ANM_CUTOFF = 24.0
PARAMETER_NAMES = ("sigma_exp", "tau_z", "tau_peptide", "tau_time", "kappa")
GAMMA_LADDER = (1e4, 1e2, 10.0, 3.0, 1.0, 0.3)
# K1 is deliberately separate: GAMMA_LADDER[0] still carries O(k_int/gamma) physics.
K1_FAST_LIMIT_C = 1e6
K1_SCALING_C = (1e4, 1e5, 1e6)
K1_TOLERANCE = 1e-6
# Every *_full manifest must carry an identical key set (asserted by the unit tests), so these
# gamma-only fields are written as None by the other questions. backfill_manifests uses this list
# to lift manifests written before a field existed up to the current schema.
GAMMA_MANIFEST_KEYS = (
    "median_k_int_per_min", "gamma_ladder_units",
    "k1_fast_limit_max_abs", "k1_fast_limit_c", "k1_fast_limit_tolerance",
    "k1_scaling_check", "k1_control_rationale",
    "ll_logpf_clip_count", "ll_covariance_sensitivity_approximation",
    "truth_parameter_provenance", "post_hoc_provenance_additions",
)
K1_SCALING_RTOL = 0.02
TARGET_MODES = Path(__file__).with_name("_moprp_sigma_noise_model") / "target_modes.npz"
NUMERICAL_FLOOR = 1e-10
UPTAKE_NORMALISATION = (
    "MoPrP uptake is peptide-wise maxD normalised using a completely deuterated control subjected to\n"
    "the same quench, digestion, LC and MS processing. This normalisation implicitly compensates\n"
    "peptide-dependent mean back-exchange in centroid uptake but does not reconstruct absolute\n"
    "pre-quench deuterium occupancy or explicitly model residue-specific / time-dependent\n"
    "back-exchange. Labelling experiments were conducted at approximately 95% D₂O."
)
SEED_REGISTRY = {
    "anm_surface": "1000 + 101 * replicate + int(100 * true_lambda)",
    "anm_shuffle": 991,
    "sign_surface": "2000 + replicate; paired between null and signed truth",
    "scale_surface": "3000 + replicate + int(100 * ratio); paired across varied components",
    "delta_monte_carlo": 4000,
}
GAMMA_SEED = "5000 + 101 * replicate + rung_index"


@dataclass(frozen=True)
class SimulationParameters:
    sigma_exp: float = 0.025
    tau_z: float = 0.25
    tau_peptide: float = 0.025
    tau_time: float = 0.02
    kappa: float = 0.5
    anm_lambda: float = 0.0
    log_gamma: float | None = None


@dataclass(frozen=True)
class Geometry:
    mean: np.ndarray
    mapping: np.ndarray
    log_pf: np.ndarray
    k_int: np.ndarray
    times: np.ndarray
    sensitivity: np.ndarray
    propagation: np.ndarray
    logpf_scale: np.ndarray
    anm_correlation: np.ndarray
    peptide_loading: np.ndarray
    time_loading: np.ndarray
    n_peptides: int
    n_timepoints: int


def correlation_of(covariance: np.ndarray) -> np.ndarray:
    """Convert a covariance to a finite unit-diagonal correlation matrix."""
    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-15, None))
    correlation = covariance / scale[:, None] / scale[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _structure_anm(residue_ids: np.ndarray) -> np.ndarray:
    universe = mda.Universe(str(STRUCTURE))
    ca = universe.select_atoms("name CA")
    lookup = {int(residue): index for index, residue in enumerate(ca.resids)}
    missing = [int(residue) for residue in residue_ids if int(residue) not in lookup]
    if missing:
        raise ValueError(f"structure lacks feature residues: {missing}")
    indices = np.asarray([lookup[int(residue)] for residue in residue_ids])
    covariance = anm_covariance(ca.positions.astype(float), cutoff=ANM_CUTOFF)
    return correlation_of(covariance[np.ix_(indices, indices)])


def build_geometry(inputs, *, anm_correlation: np.ndarray | None = None) -> Geometry:
    """Construct the 76-residue PF-conditioned geometry, never sentinel-filled columns."""
    active = np.any(inputs.mapping != 0.0, axis=0)
    if not np.all(inputs.mapping[:, ~active] == 0.0):
        raise ValueError("excluded residue columns must be identically zero in the peptide map")
    mapping = np.asarray(inputs.mapping[:, active], dtype=float)
    residue_ids = np.asarray(inputs.feature_residue_ids[active], dtype=int)
    if residue_ids.size != 76:
        raise ValueError(f"expected 76 peptide-covered residues, found {residue_ids.size}")
    logpf_frames = np.asarray(
        inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)[active], dtype=float
    )
    log_pf = np.mean(logpf_frames, axis=1)
    if not np.all(np.isfinite(log_pf)):
        raise ValueError("non-finite/sentinel value reached the PF-conditioned residue set")
    k_int = np.asarray(inputs.k_ints[active], dtype=float)
    times = np.asarray(inputs.timepoints, dtype=float)
    backend = noise.EX2Backend()
    residue_uptake = np.asarray(backend.residue_uptake(log_pf, k_int, times))
    sensitivity = np.asarray(backend.logpf_sensitivity(log_pf, k_int, times))
    mean = np.asarray(noise.peptide_uptake(residue_uptake, mapping))
    propagation = np.asarray(noise.stack_propagation_matrix(sensitivity, mapping))

    variance = np.var(logpf_frames, axis=1, ddof=0)
    logpf_scale = np.sqrt(np.clip(variance / np.mean(variance), 1e-8, None))
    r_anm = _structure_anm(residue_ids) if anm_correlation is None else anm_correlation
    if r_anm.shape != (residue_ids.size, residue_ids.size):
        raise ValueError("ANM correlation must be constructed on the active 76-residue set")

    p, t = mean.shape
    mu_flat = np.asarray(noise.vectorize_time_major(mean))
    peptide_loading = np.zeros((p * t, p))
    time_loading = np.zeros((p * t, t))
    slope = -np.asarray(noise.peptide_uptake(sensitivity, mapping))
    for j in range(t):
        for peptide in range(p):
            index = j * p + peptide
            peptide_loading[index, peptide] = mu_flat[index]
            time_loading[index, j] = slope[peptide, j]
    return Geometry(mean, mapping, log_pf, k_int, times, sensitivity, propagation, logpf_scale, r_anm,
                    peptide_loading, time_loading, p, t)


def covariance_from_parameters(
    geometry: Geometry,
    parameters: SimulationParameters,
    *,
    correlation: np.ndarray | None = None,
    acquisition_diagonal: np.ndarray | None = None,
) -> np.ndarray:
    """Build the frozen unnormalised joint covariance at known simulation parameters."""
    r_anm = geometry.anm_correlation if correlation is None else np.asarray(correlation)
    base = (1.0 - parameters.anm_lambda) * np.eye(r_anm.shape[0]) + parameters.anm_lambda * r_anm
    dz = geometry.logpf_scale
    c_z = parameters.tau_z**2 * dz[:, None] * base * dz[None, :]
    diagonal = (noise.heteroscedastic_diagonal(geometry.mean, parameters.kappa)
                if acquisition_diagonal is None else acquisition_diagonal)
    covariance = noise.build_joint_covariance(
        geometry.propagation,
        c_z,
        peptide_loading=geometry.peptide_loading,
        time_loading=geometry.time_loading,
        acquisition_diagonal=diagonal,
        tau_peptide=parameters.tau_peptide,
        tau_time=parameters.tau_time,
        sigma_exp=parameters.sigma_exp,
        numerical_floor=NUMERICAL_FLOOR,
    )
    return np.asarray(covariance)


def mean_for_kinetics(geometry: Geometry, log_gamma: float | None) -> np.ndarray:
    """Return the peptide mean under strict EX2 or finite-gating LL kinetics."""
    if log_gamma is None:
        return np.asarray(geometry.mean)
    residue_uptake = noise.LLBackend().residue_uptake(
        np.clip(geometry.log_pf, 0.0, None), geometry.k_int, geometry.times,
        kinetics=log_gamma,
    )
    return np.asarray(noise.peptide_uptake(residue_uptake, geometry.mapping))


def _k1_scaling_diagnostics(geometry: Geometry, median_rate: float) -> dict[str, object]:
    """Assert and return the asymptotic inverse-gamma LL-to-EX2 scaling control."""
    differences = {
        str(c): float(np.max(np.abs(
            mean_for_kinetics(geometry, np.log(c * median_rate)) - geometry.mean
        )))
        for c in K1_SCALING_C
    }
    ratios = []
    for lower_c, upper_c in zip(K1_SCALING_C[:-1], K1_SCALING_C[1:]):
        ratio = differences[str(lower_c)] / differences[str(upper_c)]
        if not np.isclose(ratio, 10.0, rtol=K1_SCALING_RTOL, atol=0.0):
            raise AssertionError(
                f"K1 inverse-gamma scaling failed for c={lower_c:g} to c={upper_c:g}: "
                f"ratio={ratio:.8g}"
            )
        ratios.append({"lower_c": lower_c, "upper_c": upper_c, "ratio": ratio})
    return {
        "max_abs_by_c": differences,
        "consecutive_decade_ratios": ratios,
        "ratio_target": 10.0,
        "ratio_relative_tolerance": K1_SCALING_RTOL,
    }


def simulate_surface(
    geometry: Geometry,
    parameters: SimulationParameters,
    seed: int,
    *,
    correlation: np.ndarray | None = None,
    acquisition_diagonal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one ``(P,T)`` surface and return it with its generating covariance."""
    covariance = covariance_from_parameters(geometry, parameters, correlation=correlation)
    rng = np.random.default_rng(seed)
    residual = np.linalg.cholesky(covariance) @ rng.standard_normal(covariance.shape[0])
    surface = np.asarray(geometry.mean) + np.asarray(
        noise.unvectorize_time_major(residual, geometry.n_peptides, geometry.n_timepoints)
    )
    return surface, covariance


def time_folds(n_peptides: int, n_timepoints: int, block: int = 3):
    """Yield train/test flat indices for contiguous held-out time blocks."""
    all_indices = np.arange(n_peptides * n_timepoints)
    for start in range(0, n_timepoints, block):
        held_times = np.arange(start, min(start + block, n_timepoints))
        test = np.concatenate([np.arange(j*n_peptides, (j+1)*n_peptides) for j in held_times])
        yield np.setdiff1d(all_indices, test), test


def conditional_gaussian_nll(residual, covariance, train_indices, test_indices) -> float:
    """Held-out Gaussian NLL conditional on the observed training residual."""
    residual, covariance = np.asarray(residual), np.asarray(covariance)
    tt = covariance[np.ix_(train_indices, train_indices)]
    ht = covariance[np.ix_(test_indices, train_indices)]
    hh = covariance[np.ix_(test_indices, test_indices)]
    alpha = np.linalg.solve(tt, residual[train_indices])
    conditional_mean = ht @ alpha
    conditional_cov = hh - ht @ np.linalg.solve(tt, ht.T)
    conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
    chol = np.linalg.cholesky(conditional_cov)
    return float(noise.gaussian_nll_from_cholesky(
        residual[test_indices] - conditional_mean, chol
    ))


def _decode(raw: np.ndarray, template: SimulationParameters, free: tuple[str, ...]):
    values = asdict(template)
    for name, value in zip(free, raw):
        if name == "log_gamma":
            values[name] = float(value)
        elif name == "kappa" or name == "anm_lambda":
            values[name] = float(expit(value))
        else:
            values[name] = float(np.exp(value))
    return SimulationParameters(**values)


def _encode(parameters: SimulationParameters, free: tuple[str, ...]):
    values = []
    for name in free:
        value = getattr(parameters, name)
        if name == "log_gamma":
            if value is None:
                raise ValueError("free log_gamma requires a finite initial value")
            values.append(float(value))
        elif name == "kappa" or name == "anm_lambda":
            value = np.clip(value, 1e-6, 1 - 1e-6)
            values.append(np.log(value / (1 - value)))
        else:
            values.append(np.log(max(value, 1e-8)))
    return np.asarray(values)


def fit_kinetic_parameters(
    surface: np.ndarray,
    geometry: Geometry,
    initial: SimulationParameters,
    *,
    free: tuple[str, ...],
    train_indices: np.ndarray | None = None,
    maxiter: int = 250,
) -> tuple[SimulationParameters, float, bool]:
    """Fit kinetic and frozen-architecture covariance parameters."""
    n = geometry.n_peptides * geometry.n_timepoints
    indices = np.arange(n) if train_indices is None else np.asarray(train_indices)

    def objective(raw):
        if not np.all(np.isfinite(raw)) or np.any(np.abs(raw) > 100.0):
            return 1e30
        try:
            parameters = _decode(raw, initial, free)
            residual = np.asarray(noise.vectorize_time_major(
                surface - mean_for_kinetics(geometry, parameters.log_gamma)
            ))
            covariance = covariance_from_parameters(geometry, parameters)
        except (FloatingPointError, OverflowError, ValueError):
            return 1e30
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(covariance)):
            return 1e30
        selected = covariance[np.ix_(indices, indices)]
        try:
            chol = np.linalg.cholesky(selected)
        except np.linalg.LinAlgError:
            return 1e30
        return float(noise.gaussian_nll_from_cholesky(residual[indices], chol))

    result = minimize(objective, _encode(initial, free), method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-9})
    return _decode(result.x, initial, free), float(result.fun), bool(result.success)


def kinetic_cross_fitted_score(surface, geometry, initial, *, free, maxiter=250):
    """Return conditional held-out NLL with kinetic parameters refit per time fold."""
    total = 0.0
    estimates = []
    for train, test in time_folds(geometry.n_peptides, geometry.n_timepoints):
        fitted, _, success = fit_kinetic_parameters(
            surface, geometry, initial, free=free, train_indices=train, maxiter=maxiter
        )
        residual = np.asarray(noise.vectorize_time_major(
            surface - mean_for_kinetics(geometry, fitted.log_gamma)
        ))
        covariance = covariance_from_parameters(geometry, fitted)
        total += conditional_gaussian_nll(residual, covariance, train, test)
        estimates.append({**asdict(fitted), "success": success})
    return total, estimates


def fit_parameters(
    surface: np.ndarray,
    geometry: Geometry,
    initial: SimulationParameters,
    *,
    free: tuple[str, ...] = PARAMETER_NAMES,
    correlation: np.ndarray | None = None,
    acquisition_diagonal: np.ndarray | None = None,
    train_indices: np.ndarray | None = None,
    maxiter: int = 250,
) -> tuple[SimulationParameters, float, bool]:
    """Fit selected covariance parameters by full Gaussian likelihood."""
    residual = np.asarray(noise.vectorize_time_major(surface - geometry.mean))
    indices = np.arange(residual.size) if train_indices is None else np.asarray(train_indices)

    def objective(raw):
        parameters = _decode(raw, initial, free)
        covariance = covariance_from_parameters(
            geometry, parameters, correlation=correlation,
            acquisition_diagonal=acquisition_diagonal,
        )
        selected = covariance[np.ix_(indices, indices)]
        try:
            chol = np.linalg.cholesky(selected)
        except np.linalg.LinAlgError:
            return 1e30
        return float(noise.gaussian_nll_from_cholesky(residual[indices], chol))

    result = minimize(objective, _encode(initial, free), method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-9})
    return _decode(result.x, initial, free), float(result.fun), bool(result.success)


def cross_fitted_score(surface, geometry, initial, *, free, correlation=None,
                       acquisition_diagonal=None, maxiter=250):
    """Return summed held-out conditional NLL with parameters refit in every fold."""
    residual = np.asarray(noise.vectorize_time_major(surface - geometry.mean))
    total = 0.0
    estimates = []
    for train, test in time_folds(geometry.n_peptides, geometry.n_timepoints):
        fitted, _, success = fit_parameters(surface, geometry, initial, free=free,
                                             correlation=correlation, train_indices=train,
                                             acquisition_diagonal=acquisition_diagonal,
                                             maxiter=maxiter)
        covariance = covariance_from_parameters(
            geometry, fitted, correlation=correlation,
            acquisition_diagonal=acquisition_diagonal,
        )
        total += conditional_gaussian_nll(residual, covariance, train, test)
        estimates.append({**asdict(fitted), "success": success})
    return total, estimates


def shuffled_correlation(correlation: np.ndarray, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(correlation.shape[0])
    return correlation[np.ix_(permutation, permutation)]


def run_anm(geometry, base, lambdas, replicates, maxiter):
    rows = []
    shuffled = shuffled_correlation(geometry.anm_correlation, 991)
    for lam in lambdas:
        truth = replace(base, anm_lambda=lam)
        for replicate in range(replicates):
            surface, _ = simulate_surface(geometry, truth, 1000 + 101*replicate + int(100*lam))
            for arm, corr in (("anm", geometry.anm_correlation), ("shuffled", shuffled)):
                initial = replace(base, anm_lambda=0.2)
                fitted, objective, success = fit_parameters(
                    surface, geometry, initial, free=("anm_lambda",), correlation=corr,
                    maxiter=maxiter
                )
                score, _ = cross_fitted_score(surface, geometry, fitted,
                                               free=("anm_lambda",), correlation=corr,
                                               maxiter=maxiter)
                rows.append({"question": "anm", "true_lambda": lam, "replicate": replicate,
                             "arm": arm, "fitted_lambda": fitted.anm_lambda,
                             "objective": objective, "heldout_nll": score, "success": success})
    return pd.DataFrame(rows)


def summarize_anm(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibrate ANM detection against lambda=0 and shuffled-geometry nulls."""
    pivot = frame.pivot(index=["true_lambda", "replicate"], columns="arm",
                        values=["heldout_nll", "fitted_lambda"]).reset_index()
    rows = []
    null_group = pivot[pivot[("true_lambda", "")] == 0.0]
    null_advantage = np.asarray(null_group[("heldout_nll", "shuffled")]
                                - null_group[("heldout_nll", "anm")])
    null_fitted = np.asarray(null_group[("fitted_lambda", "anm")])
    null_advantage_upper = float(np.quantile(null_advantage, 0.95))
    null_fitted_upper = float(np.quantile(null_fitted, 0.95))
    for true_lambda, group in pivot.groupby("true_lambda"):
        advantage = np.asarray(group[("heldout_nll", "shuffled")]
                               - group[("heldout_nll", "anm")])
        fitted = np.asarray(group[("fitted_lambda", "anm")])
        n = len(group)
        advantage_se = np.std(advantage, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        fitted_se = np.std(fitted, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "true_lambda": true_lambda,
            "replicates": n,
            "mean_anm_advantage_nll": np.mean(advantage),
            "advantage_ci95_low": np.mean(advantage) - 1.96 * advantage_se,
            "mean_fitted_lambda": np.mean(fitted),
            "fitted_lambda_ci95_low": np.mean(fitted) - 1.96 * fitted_se,
            "null_advantage_q95": null_advantage_upper,
            "null_fitted_lambda_q95": null_fitted_upper,
            "detectable": bool(n > 1
                               and np.mean(advantage) - 1.96*advantage_se > null_advantage_upper
                               and np.mean(fitted) - 1.96*fitted_se > null_fitted_upper),
        })
    return pd.DataFrame(rows)


def _gamma_truth_parameters() -> tuple[SimulationParameters, str]:
    target = np.load(TARGET_MODES)
    fitted = dict(zip(target["parameter_names"].astype(str), target["parameters"].astype(float)))
    parameters = SimulationParameters(
        sigma_exp=fitted["sigma_exp"], tau_z=0.0,
        tau_peptide=fitted["tau_peptide"], tau_time=0.0, kappa=0.0,
    )
    return parameters, common.sha256(TARGET_MODES)


def run_gamma(geometry, replicates, maxiter):
    """Run the finite-gating ladder, reverse null, and slow-rung confounding check."""
    base, _ = _gamma_truth_parameters()
    median_rate = float(np.median(geometry.k_int))
    initial_log_gamma = float(np.log(median_rate * 10.0))
    rows = []
    rungs = [("ex2", np.inf, None)] + [
        (f"{c:g}", c, float(np.log(c * median_rate))) for c in GAMMA_LADDER
    ]
    covariance = covariance_from_parameters(geometry, base)
    chol = np.linalg.cholesky(covariance)
    for rung_index, (rung, c, true_log_gamma) in enumerate(rungs):
        truth_mean = mean_for_kinetics(geometry, true_log_gamma)
        for replicate in range(replicates):
            rng = np.random.default_rng(5000 + 101 * replicate + rung_index)
            residual = chol @ rng.standard_normal(covariance.shape[0])
            surface = truth_mean + np.asarray(noise.unvectorize_time_major(
                residual, geometry.n_peptides, geometry.n_timepoints
            ))
            arm_results = {}
            for arm, initial, free in (
                ("k0", replace(base, log_gamma=None), ("sigma_exp", "tau_peptide")),
                ("k2", replace(base, log_gamma=initial_log_gamma),
                 ("log_gamma", "sigma_exp", "tau_peptide")),
            ):
                fitted, objective, success = fit_kinetic_parameters(
                    surface, geometry, initial, free=free, maxiter=maxiter
                )
                score, estimates = kinetic_cross_fitted_score(
                    surface, geometry, fitted, free=free, maxiter=maxiter
                )
                arm_results[arm] = (score, fitted, objective, success, estimates)
            advantage = (arm_results["k0"][0] - arm_results["k2"][0]) / surface.size
            for arm in ("k0", "k2"):
                score, fitted, objective, success, estimates = arm_results[arm]
                rows.append({
                    "question": "gamma", "rung": rung, "true_c": c,
                    "true_log_gamma": true_log_gamma, "replicate": replicate, "arm": arm,
                    "heldout_nll": score, "heldout_nll_per_cell": score / surface.size,
                    "advantage_k0_minus_k2_per_cell": advantage,
                    "fitted_log_gamma": fitted.log_gamma,
                    "mean_fold_log_gamma": np.nanmean([
                        x["log_gamma"] for x in estimates if x["log_gamma"] is not None
                    ]) if arm == "k2" else np.nan,
                    "fitted_sigma_exp": fitted.sigma_exp,
                    "fitted_tau_peptide": fitted.tau_peptide,
                    "fitted_tau_time": fitted.tau_time, "fitted_kappa": fitted.kappa,
                    "objective": objective, "success": success,
                })
            if np.isfinite(c) and c <= 3.0:
                free = ("log_gamma", "sigma_exp", "tau_peptide", "tau_time", "kappa")
                initial = replace(base, log_gamma=initial_log_gamma, tau_time=0.01, kappa=0.1)
                fitted, objective, success = fit_kinetic_parameters(
                    surface, geometry, initial, free=free, maxiter=maxiter
                )
                score, _ = kinetic_cross_fitted_score(
                    surface, geometry, fitted, free=free, maxiter=maxiter
                )
                rows.append({
                    "question": "gamma", "rung": rung, "true_c": c,
                    "true_log_gamma": true_log_gamma, "replicate": replicate,
                    "arm": "k2_confounding", "heldout_nll": score,
                    "heldout_nll_per_cell": score / surface.size,
                    "advantage_k0_minus_k2_per_cell": np.nan,
                    "fitted_log_gamma": fitted.log_gamma, "mean_fold_log_gamma": np.nan,
                    "fitted_sigma_exp": fitted.sigma_exp,
                    "fitted_tau_peptide": fitted.tau_peptide,
                    "fitted_tau_time": fitted.tau_time, "fitted_kappa": fitted.kappa,
                    "objective": objective, "success": success,
                })
    return pd.DataFrame(rows)


def summarize_gamma(frame: pd.DataFrame, median_rate: float) -> pd.DataFrame:
    """Calibrate finite-gating detection against the strict-EX2 reverse null."""
    primary = frame[frame["arm"] == "k2"]
    null = np.asarray(primary[primary["rung"] == "ex2"]["advantage_k0_minus_k2_per_cell"])
    null_q95 = float(np.quantile(null, 0.95))
    reverse = np.asarray(primary[primary["rung"] == "ex2"]["fitted_log_gamma"])
    reverse_relative = reverse - np.log(median_rate)
    rows = []
    for rung, group in primary.groupby("rung", sort=False):
        advantage = np.asarray(group["advantage_k0_minus_k2_per_cell"])
        c = float(group["true_c"].iloc[0])
        confounding = frame[(frame["rung"] == rung) & (frame["arm"] == "k2_confounding")]
        # Standing positive control: where gamma is identifiable the fitted rate must recover the
        # truth; on the ex2 null the load-bearing statistic is the MINIMUM fitted c, which must
        # stay above the detectability floor so the null cannot manufacture a false positive.
        fitted_c = np.exp(np.asarray(group["fitted_log_gamma"], dtype=float)) / median_rate
        fitted_c = fitted_c[np.isfinite(fitted_c)]
        row = {
            "rung": rung, "true_c": c, "replicates": len(group),
            "fitted_c_min": float(np.min(fitted_c)) if fitted_c.size else np.nan,
            "fitted_c_q05": float(np.quantile(fitted_c, 0.05)) if fitted_c.size else np.nan,
            "fitted_c_q50": float(np.quantile(fitted_c, 0.50)) if fitted_c.size else np.nan,
            "fitted_c_q95": float(np.quantile(fitted_c, 0.95)) if fitted_c.size else np.nan,
            "recovery_ratio_median": (
                float(np.median(fitted_c) / c) if fitted_c.size and np.isfinite(c) else np.nan
            ),
            "mean_advantage_nll_per_cell": float(np.mean(advantage)),
            "advantage_q05": float(np.quantile(advantage, 0.05)),
            "advantage_q95": float(np.quantile(advantage, 0.95)),
            "null_advantage_q95": null_q95,
            "detectable": bool(rung != "ex2" and len(group) > 1
                               and np.quantile(advantage, 0.05) > null_q95),
            "reverse_log_gamma_relative_q05": (
                float(np.quantile(reverse_relative, 0.05)) if rung == "ex2" else np.nan
            ),
            "reverse_log_gamma_relative_q50": (
                float(np.quantile(reverse_relative, 0.50)) if rung == "ex2" else np.nan
            ),
            "reverse_log_gamma_relative_q95": (
                float(np.quantile(reverse_relative, 0.95)) if rung == "ex2" else np.nan
            ),
        }
        for scale in ("sigma_exp", "tau_peptide", "tau_time", "kappa"):
            column = f"fitted_{scale}"
            row[f"confounding_corr_log_gamma_{scale}"] = (
                float(confounding["fitted_log_gamma"].corr(confounding[column]))
                if len(confounding) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_sign(geometry, base, replicates, maxiter):
    signs = np.where(np.arange(geometry.anm_correlation.shape[0]) % 3 == 0, -1.0, 1.0)
    arms = {
        "signed": geometry.anm_correlation,
        "flip": np.asarray(noise.domain_flip_correlation(geometry.anm_correlation, signs)),
        "unsigned": np.asarray(noise.schur_square_correlation(geometry.anm_correlation)),
    }
    rows = []
    for true_arm, truth in (("null", replace(base, anm_lambda=0.0)),
                            ("signed", replace(base, anm_lambda=0.7))):
        for replicate in range(replicates):
            surface, _ = simulate_surface(geometry, truth, 2000 + replicate, correlation=arms["signed"])
            for arm, corr in arms.items():
                score, estimates = cross_fitted_score(surface, geometry, truth,
                                                       free=("anm_lambda",), correlation=corr,
                                                       maxiter=maxiter)
                rows.append({"question": "sign", "true_arm": true_arm,
                             "replicate": replicate, "arm": arm, "heldout_nll": score,
                             "mean_fitted_lambda": np.mean([x["anm_lambda"] for x in estimates])})
    return pd.DataFrame(rows)


def summarize_sign(frame: pd.DataFrame) -> pd.DataFrame:
    """Report paired held-out NLL advantage of the generating signed arm."""
    pivot = frame.pivot(index=["true_arm", "replicate"], columns="arm", values="heldout_nll")
    rows = []
    for competitor in ("flip", "unsigned"):
        advantages = pivot[competitor] - pivot["signed"]
        null = np.asarray(advantages.loc["null"])
        advantage = np.asarray(advantages.loc["signed"])
        n = advantage.size
        se = np.std(advantage, ddof=1) / np.sqrt(n) if n > 1 else np.nan
        null_upper = float(np.quantile(null, 0.95))
        rows.append({"competitor": competitor, "replicates": n,
                     "mean_signed_advantage_nll": np.mean(advantage),
                     "advantage_ci95_low": np.mean(advantage) - 1.96*se,
                     "null_advantage_q95": null_upper,
                     "distinguishable": bool(n > 1
                                             and np.mean(advantage) - 1.96*se > null_upper)})
    return pd.DataFrame(rows)


def run_scale(geometry, base, ratios, replicates, maxiter):
    rows = []
    for varied in PARAMETER_NAMES:
        for ratio in ratios:
            truth = replace(base, **{varied: getattr(base, varied) * ratio})
            if varied == "kappa":
                truth = replace(truth, kappa=min(0.95, truth.kappa))
            for replicate in range(replicates):
                surface, _ = simulate_surface(geometry, truth, 3000 + replicate + int(100*ratio))
                fitted, objective, success = fit_parameters(surface, geometry, base,
                                                             maxiter=maxiter)
                row = {"question": "scale", "varied": varied, "ratio": ratio,
                       "replicate": replicate, "objective": objective, "success": success}
                for name in PARAMETER_NAMES:
                    row[f"true_{name}"] = getattr(truth, name)
                    row[f"fitted_{name}"] = getattr(fitted, name)
                    row[f"relative_error_{name}"] = (
                        (getattr(fitted, name) - getattr(truth, name))
                        / max(abs(getattr(truth, name)), 1e-12)
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_scale(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recovery bias and boundary frequency for each scale-grid cell."""
    rows = []
    for (varied, ratio), group in frame.groupby(["varied", "ratio"]):
        row = {"varied": varied, "ratio": ratio, "replicates": len(group),
               "optimizer_success_rate": float(group["success"].mean())}
        for name in PARAMETER_NAMES:
            fitted = np.asarray(group[f"fitted_{name}"])
            truth = np.asarray(group[f"true_{name}"])
            row[f"median_absolute_relative_error_{name}"] = float(
                np.median(np.abs(group[f"relative_error_{name}"]))
            )
            row[f"boundary_rate_{name}"] = float(np.mean(
                (fitted < 1e-4) | ((name == "kappa") & (fitted > 1 - 1e-4))
            ))
            row[f"median_bias_{name}"] = float(np.median(fitted - truth))
        rows.append(row)
    return pd.DataFrame(rows)


def run_delta(geometry, spreads, draws, seed=4000):
    rng = np.random.default_rng(seed)
    rows = []
    a0 = geometry.propagation
    r = geometry.logpf_scale.size
    base_correlation = np.eye(r)
    for spread in spreads:
        c_z = spread**2 * geometry.logpf_scale[:, None] * base_correlation * geometry.logpf_scale[None, :]
        linear = a0 @ c_z @ a0.T
        samples = rng.multivariate_normal(np.zeros(r), c_z, size=draws)
        for kind in ("mc_estimator_null", "nonlinear_ex2"):
            if kind == "mc_estimator_null":
                propagated = samples @ a0.T
            else:
                uptake = jax.vmap(
                    lambda delta: noise.vectorize_time_major(
                        noise.peptide_uptake(
                            noise.EX2Backend().residue_uptake(
                                jnp.asarray(geometry.log_pf) + delta,
                                geometry.k_int,
                                geometry.times,
                            ),
                            geometry.mapping,
                        )
                    )
                )(jnp.asarray(samples))
                propagated = np.asarray(uptake) - np.asarray(
                    noise.vectorize_time_major(geometry.mean)
                )
            empirical = np.cov(propagated, rowvar=False, ddof=0)
            fro_error = np.linalg.norm(empirical - linear, "fro") / np.linalg.norm(linear, "fro")
            diagonal_error = np.max(np.abs(np.diag(empirical) - np.diag(linear))
                                    / np.maximum(np.diag(linear), 1e-15))
            rows.append({"question": "delta", "spread": spread, "draws": draws,
                         "relative_frobenius_error": fro_error,
                         "max_relative_diagonal_error": diagonal_error, "kind": kind})
    return pd.DataFrame(rows)


def summarize_delta(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(index="spread", columns="kind",
                        values=["relative_frobenius_error", "max_relative_diagonal_error"])
    return pd.DataFrame({
        "spread": pivot.index,
        "nonlinear_excess_frobenius_error": (
            pivot[("relative_frobenius_error", "nonlinear_ex2")]
            - pivot[("relative_frobenius_error", "mc_estimator_null")]
        ).to_numpy(),
        "nonlinear_excess_max_diagonal_error": (
            pivot[("max_relative_diagonal_error", "nonlinear_ex2")]
            - pivot[("max_relative_diagonal_error", "mc_estimator_null")]
        ).to_numpy(),
    })


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=common.PACKAGE_ROOT.parent,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _anm_variant(question: str) -> dict[str, object]:
    return {
        "used_in_question": question in {"all", "anm", "sign"},
        "model": "C-alpha anisotropic network covariance converted to unit-diagonal correlation",
        "cutoff_angstrom": ANM_CUTOFF,
        "correlation_arms": (
            ["signed", "residue-permuted-shuffled"] if question == "anm"
            else ["signed", "domain-flipped", "Schur-square-unsigned"] if question == "sign"
            else []
        ),
    }


def _recordkeeping(question: str) -> dict[str, object]:
    """Return the identical §4-mandated manifest field set for every Phase-2 run."""
    return {
        "git_commit": _git_commit(),
        "pf_fit": {"performed": False, "start_count": None, "seed": None},
        "masks": {
            "construction": "76 peptide-covered residues selected before C_z, D_z, and R_ANM construction",
            "peptide_count": 14,
            "pf_conditioned_residue_count": 76,
            "excluded_mapping_columns_asserted_zero": True,
            "sentinels_forbidden_from_log_and_rate_equations": True,
        },
        "uptake_normalisation": UPTAKE_NORMALISATION,
        "anm_variant": _anm_variant(question),
        "numerical_floor": NUMERICAL_FLOOR,
        "seeds": ({**SEED_REGISTRY, "gamma_surface": GAMMA_SEED}
                  if question in {"gamma", "all"} else SEED_REGISTRY),
        "uptake_backend": "ll" if question in {"gamma", "all"} else "ex2",
        "ll_gamma_parameterisation": (
            "log_gamma; gamma = k_open + k_close; p_open = exp(-log_pf)"
            if question in {"gamma", "all"} else None
        ),
        "ll_log_gamma": list(GAMMA_LADDER) if question in {"gamma", "all"} else None,
    }


def backfill_manifests(root: Path) -> None:
    """Backfill existing Phase-2 manifests without rerunning the simulation grids."""
    for path in sorted(root.glob("*_full/manifest.json")):
        manifest = json.loads(path.read_text())
        question = path.parent.name.removesuffix("_full")
        manifest.update(_recordkeeping(question))
        for key in GAMMA_MANIFEST_KEYS:
            manifest.setdefault(key, None)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def run(args):
    output_dir = (args.output_dir / f"gamma_{args.profile}"
                  if args.question == "gamma" else args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_digest = common.sha256(STRUCTURE)
    if structure_digest != STRUCTURE_SHA256:
        raise ValueError(f"unexpected structure SHA-256: {structure_digest}")
    inputs = common.load_blinded_ensemble_inputs(args.ensemble, args.rate_source)
    geometry = build_geometry(inputs)
    base = SimulationParameters()
    if args.profile == "smoke":
        lambdas, ratios, spreads, replicates, draws, maxiter = [0.0, 0.5], [0.5, 2.0], [0.05, 0.2], 1, 300, 35
        gamma_replicates = 2
    else:
        lambdas, ratios, spreads, replicates, draws, maxiter = np.linspace(0, 1, 6), [0.25, 0.5, 1, 2, 4], [0.025, 0.05, 0.1, 0.2, 0.4], args.replicates, args.mc_draws, args.maxiter
        gamma_replicates = replicates
    frames = {}
    requested = ("anm", "sign", "scale", "delta", "gamma") if args.question == "all" else (args.question,)
    if "anm" in requested:
        frames["anm"] = run_anm(geometry, base, lambdas, replicates, maxiter)
        frames["anm_summary"] = summarize_anm(frames["anm"])
    if "sign" in requested:
        frames["sign"] = run_sign(geometry, base, replicates, maxiter)
        frames["sign_summary"] = summarize_sign(frames["sign"])
    if "scale" in requested:
        frames["scale"] = run_scale(geometry, base, ratios, replicates, maxiter)
        frames["scale_summary"] = summarize_scale(frames["scale"])
    if "delta" in requested:
        frames["delta"] = run_delta(geometry, spreads, draws)
        frames["delta_summary"] = summarize_delta(frames["delta"])
    k1_max_abs = None
    k1_scaling_check = None
    target_hash = None
    median_rate = None
    if "gamma" in requested:
        median_rate = float(np.median(geometry.k_int))
        fastest_log_gamma = float(np.log(K1_FAST_LIMIT_C * median_rate))
        k1_max_abs = float(np.max(np.abs(
            mean_for_kinetics(geometry, fastest_log_gamma) - geometry.mean
        )))
        if k1_max_abs > K1_TOLERANCE:
            raise AssertionError(f"K1 fast-limit control failed: max abs = {k1_max_abs:.3g}")
        k1_scaling_check = _k1_scaling_diagnostics(geometry, median_rate)
        frames["gamma"] = run_gamma(geometry, gamma_replicates, maxiter)
        frames["gamma_summary"] = summarize_gamma(frames["gamma"], median_rate)
        _, target_hash = _gamma_truth_parameters()
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "phase": 2,
        "profile": args.profile,
        "scientific_result": args.profile == "full",
        "ensemble": args.ensemble,
        "rate_provenance": common.rate_source_provenance(args.rate_source),
        "structure": str(STRUCTURE.resolve()),
        "structure_sha256": structure_digest,
        "structure_sha256_expected": STRUCTURE_SHA256,
        "vector_order": "time-major; index = j * P + p",
        "shape": [geometry.n_peptides, geometry.n_timepoints],
        "pf_conditioned_residues": geometry.logpf_scale.size,
        "held_out_scoring": "conditional Gaussian NLL; blocked 3-timepoint folds; refit per fold",
        "parameters": asdict(base),
        "outputs": {name: f"{name}.csv" for name in frames},
        "interpretation_rule": (
            "Only full-profile summaries are scientific. Detectability requires paired held-out "
            "NLL advantage and fitted effect lower 95% bounds above their registered nulls."
        ),
        "k1_fast_limit_max_abs": k1_max_abs,
        # The ladder is dimensionless (c = gamma / median k_int), so the absolute floor in min^-1
        # depends on the rate source. Record the scale that converts one to the other.
        "median_k_int_per_min": median_rate,
        "gamma_ladder_units": (
            "c is dimensionless; gamma = c * median_k_int_per_min (min^-1)"
            if median_rate is not None else None
        ),
        "k1_fast_limit_c": K1_FAST_LIMIT_C if "gamma" in requested else None,
        "k1_fast_limit_tolerance": K1_TOLERANCE if "gamma" in requested else None,
        "k1_scaling_check": k1_scaling_check,
        "k1_control_rationale": (
            "The fast-limit control is deliberately decoupled from GAMMA_LADDER[0] because "
            "the ladder's fastest rung carries real O(k_int/gamma) physics."
            if "gamma" in requested else None
        ),
        "ll_logpf_clip_count": int(np.count_nonzero(geometry.log_pf < 0)) if "gamma" in requested else None,
        "ll_covariance_sensitivity_approximation": (
            "EX2 sensitivities retained for tau_time/kappa confounding check; primary gamma covariance is gamma-independent"
            if "gamma" in requested else None
        ),
        "truth_parameter_provenance": ({
            "path": str(TARGET_MODES.resolve()), "sha256": target_hash,
            "parameter_names": ["sigma_exp", "tau_peptide"],
        } if "gamma" in requested else None),
        "post_hoc_provenance_additions": None,
        **_recordkeeping(args.question),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("_moprp_sigma_identifiability"))
    parser.add_argument("--ensemble", choices=tuple(common.ENSEMBLES), default="AF2_MSAss")
    parser.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default="hdxrate_pdla_validated")
    parser.add_argument("--question", choices=("all", "anm", "sign", "scale", "delta", "gamma"), default="all")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--mc-draws", type=int, default=5000)
    parser.add_argument("--maxiter", type=int, default=250)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
