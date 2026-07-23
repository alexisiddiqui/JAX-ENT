"""Geometry-regularised inference of HDX effective-rate variance amplitudes.

This module is intentionally upstream of ensemble reweighting.  It accepts fixed
effective-rate means, uptake curves, a peptide map, and population-free geometry;
it never accepts target frame weights or state populations.

Two estimators are provided:

``curve_moment``
    A positive Gamma two-moment closure whose mean rate is fixed and whose
    variance is inferred from the complete uptake curve.

``structured_residual``
    A Gaussian quasi-likelihood for residuals about the fixed-mean curve, with
    time-dependent covariance propagated through the uptake Jacobian.  Until
    qualified on ISO, this is model-discrepancy inference rather than an estimate
    of conformational variance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

jax.config.update("jax_enable_x64", True)


ARTIFACT_TYPE = "geometry_regularised_hdx_target_variance"
ARTIFACT_VERSION = 1
DEFAULT_DISTANCE_CUTOFF_ANGSTROM = 8.0
GEOMETRY_NAMES = (
    "covariance_only",
    "distance_only",
    "sequence_only",
    "covariance_distance_sequence",
    "identity",
    "shuffled_geometry",
)


@dataclass(frozen=True)
class TargetVarianceFit:
    """Result of one no-reweighting target-variance fit."""

    estimator: str
    geometry_name: str
    regularization: float
    variances: np.ndarray
    covariance: np.ndarray
    mapped_covariance: np.ndarray
    predicted_uptake: np.ndarray
    objective: float
    data_objective: float
    regularization_objective: float
    success: bool
    message: str
    iterations: int


def effective_rates(log_pf_by_frame: np.ndarray, k_ints: np.ndarray) -> np.ndarray:
    """Return ``k_int * exp(-log_pf)`` with shape ``(residues, frames)``."""

    log_pf = np.asarray(log_pf_by_frame, dtype=float)
    k_ints = np.asarray(k_ints, dtype=float)
    if log_pf.ndim != 2 or k_ints.shape != (log_pf.shape[0],):
        raise ValueError("log_pf_by_frame and k_ints are not residue-aligned")
    if not np.isfinite(log_pf).all() or not np.isfinite(k_ints).all() or np.any(k_ints <= 0):
        raise ValueError("log protection factors must be finite and intrinsic rates positive")
    return k_ints[:, None] * np.exp(-log_pf)


def population_covariance(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Population covariance over columns, using uniform weights by default."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("values must have shape (variables, observations)")
    if weights is None:
        weights = np.full(values.shape[1], 1.0 / values.shape[1])
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (values.shape[1],) or np.any(weights < 0) or not np.sum(weights) > 0:
        raise ValueError("weights must be non-negative and observation-aligned")
    weights = weights / weights.sum()
    centered = values - values @ weights[:, None]
    covariance = (centered * weights[None, :]) @ centered.T
    return 0.5 * (covariance + covariance.T)


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert a PSD covariance to a PSD correlation, including constant variables."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    scale = float(np.max(np.abs(np.diag(covariance)), initial=0.0))
    tolerance = 1e-10 * scale + np.finfo(float).tiny
    if float(np.linalg.eigvalsh(covariance).min()) < -tolerance:
        raise ValueError("covariance is not positive semidefinite")
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    active = diagonal > tolerance
    correlation = np.zeros_like(covariance)
    if active.any():
        scale = np.sqrt(diagonal[active])
        correlation[np.ix_(active, active)] = covariance[np.ix_(active, active)] / np.outer(
            scale, scale
        )
    # A constant coordinate is independent geometry, not a zero-diagonal correlation.
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


def uniform_rate_correlation(rates_by_frame: np.ndarray) -> np.ndarray:
    """Population-free rate correlation under uniform trajectory weights."""

    return covariance_to_correlation(population_covariance(rates_by_frame))


def wendland_distance_kernel(
    coordinates: np.ndarray,
    cutoff_angstrom: float = DEFAULT_DISTANCE_CUTOFF_ANGSTROM,
) -> np.ndarray:
    """Compact-support Wendland C2 kernel, PSD for 3-D Euclidean coordinates.

    ``phi(r) = (1-r)^4 (1+4r)`` for ``0 <= r < 1`` and zero otherwise.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a finite (residues, 3) array")
    if cutoff_angstrom <= 0:
        raise ValueError("cutoff_angstrom must be positive")
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    scaled = distances / float(cutoff_angstrom)
    remaining = np.clip(1.0 - scaled, 0.0, None)
    kernel = remaining**4 * (1.0 + 4.0 * scaled)
    kernel[scaled >= 1.0] = 0.0
    np.fill_diagonal(kernel, 1.0)
    return 0.5 * (kernel + kernel.T)


def nearest_neighbor_sequence_kernel(
    residue_ids: Sequence[int] | np.ndarray,
    neighbor_strength: float = 0.25,
) -> np.ndarray:
    """PSD kernel supported only on identical and sequential residues.

    The matrix is ``I + rho*A`` for disjoint chain paths.  ``rho <= 0.5``
    guarantees positive semidefiniteness because every path adjacency matrix has
    spectrum in ``[-2, 2]``.
    """

    residue_ids = np.asarray(residue_ids, dtype=int)
    if residue_ids.ndim != 1 or len(np.unique(residue_ids)) != residue_ids.size:
        raise ValueError("residue_ids must be a one-dimensional unique sequence")
    if not 0.0 <= neighbor_strength <= 0.5:
        raise ValueError("neighbor_strength must lie in [0, 0.5] for the PSD guarantee")
    adjacent = np.abs(residue_ids[:, None] - residue_ids[None, :]) == 1
    kernel = np.eye(residue_ids.size) + float(neighbor_strength) * adjacent.astype(float)
    return kernel


def distance_sequence_support_kernel(
    coordinates: np.ndarray,
    residue_ids: Sequence[int] | np.ndarray,
    *,
    cutoff_angstrom: float = DEFAULT_DISTANCE_CUTOFF_ANGSTROM,
    neighbor_strength: float = 0.25,
) -> np.ndarray:
    """PSD average of spatial and sequential support kernels."""

    distance = wendland_distance_kernel(coordinates, cutoff_angstrom)
    sequence = nearest_neighbor_sequence_kernel(residue_ids, neighbor_strength)
    return 0.5 * (distance + sequence)


def _validate_correlation(matrix: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ValueError(f"{name} must have unit diagonal")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def build_rate_geometries(
    rates_by_frame: np.ndarray,
    coordinates: np.ndarray,
    residue_ids: Sequence[int] | np.ndarray,
    *,
    cutoff_angstrom: float = DEFAULT_DISTANCE_CUTOFF_ANGSTROM,
    neighbor_strength: float = 0.25,
    shuffle_seed: int = 20260722,
) -> dict[str, np.ndarray]:
    """Build all preregistered, population-free PSD rate geometries.

    The combined geometry uses a Schur product, so PSD is preserved without a
    hard covariance mask.  The shuffled control applies a simultaneous row/column
    permutation and therefore also remains PSD.
    """

    rates = np.asarray(rates_by_frame, dtype=float)
    residue_ids = np.asarray(residue_ids, dtype=int)
    coordinates = np.asarray(coordinates, dtype=float)
    if rates.ndim != 2 or rates.shape[0] != residue_ids.size:
        raise ValueError("rates_by_frame and residue_ids are not aligned")
    if coordinates.shape != (rates.shape[0], 3):
        raise ValueError("coordinates and rates_by_frame are not aligned")
    correlation = uniform_rate_correlation(rates)
    distance = wendland_distance_kernel(coordinates, cutoff_angstrom)
    sequence = nearest_neighbor_sequence_kernel(residue_ids, neighbor_strength)
    support = 0.5 * (distance + sequence)
    tapered = correlation * support  # Schur product theorem preserves PSD.
    permutation = np.random.default_rng(shuffle_seed).permutation(rates.shape[0])
    shuffled = tapered[np.ix_(permutation, permutation)]
    geometries = {
        "covariance_only": correlation,
        "distance_only": distance,
        "sequence_only": sequence,
        "covariance_distance_sequence": tapered,
        "identity": np.eye(rates.shape[0]),
        "shuffled_geometry": shuffled,
    }
    return {name: _validate_correlation(value, name=name) for name, value in geometries.items()}


def build_hdx_covariance(variances: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """Construct ``D**1/2 R D**1/2`` and verify its defining invariants."""

    variances = np.asarray(variances, dtype=float)
    geometry = _validate_correlation(geometry, name="geometry")
    if variances.shape != (geometry.shape[0],) or np.any(~np.isfinite(variances)):
        raise ValueError("variances and geometry are not aligned")
    if np.any(variances < 0):
        raise ValueError("variances must be non-negative")
    root = np.sqrt(variances)
    covariance = root[:, None] * geometry * root[None, :]
    covariance = 0.5 * (covariance + covariance.T)
    if not np.allclose(np.diag(covariance), variances, rtol=1e-10, atol=1e-14):
        raise AssertionError("constructed covariance does not have diagonal D")
    tolerance = 1e-9 * max(float(np.max(variances, initial=0.0)), 1.0)
    if float(np.linalg.eigvalsh(covariance).min()) < -tolerance:
        raise AssertionError("constructed covariance is not positive semidefinite")
    return covariance


def map_hdx_covariance(covariance: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Map a complete residue covariance by the congruence ``M C M.T``."""

    covariance = np.asarray(covariance, dtype=float)
    mapping = np.asarray(mapping, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if mapping.ndim != 2 or mapping.shape[1] != covariance.shape[0]:
        raise ValueError("mapping and covariance are not residue-aligned")
    if not np.isfinite(mapping).all() or np.any(np.linalg.norm(mapping, axis=1) == 0):
        raise ValueError("mapping must contain finite, non-empty peptide rows")
    mapped = mapping @ covariance @ mapping.T
    return 0.5 * (mapped + mapped.T)


def _gamma_two_moment_uptake_jax(
    mean_rates: jax.Array, variances: jax.Array, timepoints: jax.Array
) -> jax.Array:
    tiny = jnp.finfo(mean_rates.dtype).eps
    safe_variance = jnp.maximum(variances, tiny * jnp.square(mean_rates))
    shape = jnp.square(mean_rates) / safe_variance
    scale = safe_variance / mean_rates
    gamma_survival = jnp.exp(-shape[None, :] * jnp.log1p(timepoints[:, None] * scale[None, :]))
    deterministic_survival = jnp.exp(-timepoints[:, None] * mean_rates[None, :])
    survival = jnp.where(
        variances[None, :] <= tiny * jnp.square(mean_rates)[None, :],
        deterministic_survival,
        gamma_survival,
    )
    return 1.0 - survival


def positive_two_moment_uptake(
    mean_rates: np.ndarray, variances: np.ndarray, timepoints: np.ndarray
) -> np.ndarray:
    """Expected residue uptake for a positive Gamma rate distribution.

    The Gamma shape and scale are chosen to match the supplied mean and variance.
    The exact deterministic ``variance=0`` limit is handled explicitly.
    """

    mean_rates = np.asarray(mean_rates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    timepoints = np.asarray(timepoints, dtype=float)
    if mean_rates.ndim != 1 or variances.shape != mean_rates.shape:
        raise ValueError("mean_rates and variances must be aligned vectors")
    if np.any(mean_rates <= 0) or np.any(variances < 0) or np.any(timepoints < 0):
        raise ValueError("rates must be positive; variances and times must be non-negative")
    return np.asarray(
        _gamma_two_moment_uptake_jax(
            jnp.asarray(mean_rates), jnp.asarray(variances), jnp.asarray(timepoints)
        )
    )


def predict_curve_moment_uptake(
    mean_rates: np.ndarray,
    variances: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    """Return peptide-by-time uptake under the positive two-moment model."""

    residue = positive_two_moment_uptake(mean_rates, variances, timepoints)
    mapping = np.asarray(mapping, dtype=float)
    if mapping.ndim != 2 or mapping.shape[1] != residue.shape[1]:
        raise ValueError("mapping and rates are not residue-aligned")
    return mapping @ residue.T


def predict_fixed_mean_uptake(
    mean_rates: np.ndarray, timepoints: np.ndarray, mapping: np.ndarray
) -> np.ndarray:
    """Return the zero-variance fixed-BV mean curve (peptides by time)."""

    return predict_curve_moment_uptake(
        mean_rates, np.zeros_like(np.asarray(mean_rates, dtype=float)), timepoints, mapping
    )


def _validate_fit_inputs(
    observed_uptake: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    observation_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed_uptake, dtype=float)
    mean_rates = np.asarray(mean_rates, dtype=float)
    timepoints = np.asarray(timepoints, dtype=float)
    mapping = np.asarray(mapping, dtype=float)
    geometry = _validate_correlation(geometry, name="geometry")
    if observed.shape != (mapping.shape[0], timepoints.size):
        raise ValueError("observed_uptake must have shape (peptides, timepoints)")
    if mean_rates.shape != (mapping.shape[1],) or geometry.shape[0] != mean_rates.size:
        raise ValueError("mean rates, mapping, and geometry are not residue-aligned")
    if np.any(mean_rates <= 0) or np.any(timepoints < 0) or not np.isfinite(observed).all():
        raise ValueError("fit inputs contain invalid rates, times, or uptake")
    mask = np.ones_like(observed, dtype=bool) if observation_mask is None else np.asarray(
        observation_mask, dtype=bool
    )
    if mask.shape != observed.shape or not mask.any():
        raise ValueError("observation_mask must select at least one observation")
    return observed, mean_rates, timepoints, mapping, geometry, mask


def _regularization_penalty(
    log_relative_variance: jax.Array, geometry: jax.Array
) -> jax.Array:
    dimension = log_relative_variance.size
    off_diagonal = jnp.abs(geometry) * (1.0 - jnp.eye(dimension, dtype=geometry.dtype))
    differences = log_relative_variance[:, None] - log_relative_variance[None, :]
    graph = jnp.sum(off_diagonal * jnp.square(differences)) / jnp.maximum(
        jnp.sum(off_diagonal), 1.0
    )
    centered = log_relative_variance - jnp.mean(log_relative_variance)
    return graph + 0.01 * jnp.mean(jnp.square(centered))


def _initial_log_relative_variance(
    mean_rates: np.ndarray, initial_variance: float | np.ndarray
) -> np.ndarray:
    initial = np.asarray(initial_variance, dtype=float)
    if initial.ndim == 0:
        initial = np.full(mean_rates.size, float(initial)) * np.square(mean_rates)
    if initial.shape != mean_rates.shape or np.any(initial <= 0):
        raise ValueError("initial_variance must be positive and scalar or residue-aligned")
    return np.log(initial / np.square(mean_rates))


def fit_curve_moment_variance(
    observed_uptake: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    *,
    geometry_name: str = "custom",
    regularization: float = 0.0,
    observation_mask: np.ndarray | None = None,
    initial_variance: float | np.ndarray = 0.1,
    maxiter: int = 1000,
) -> TargetVarianceFit:
    """Fit positive rate variances with the Gamma two-moment uptake model."""

    observed, means, times, peptide_map, geometry, mask = _validate_fit_inputs(
        observed_uptake, mean_rates, timepoints, mapping, geometry, observation_mask
    )
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    observed_j = jnp.asarray(observed)
    means_j = jnp.asarray(means)
    times_j = jnp.asarray(times)
    mapping_j = jnp.asarray(peptide_map)
    geometry_j = jnp.asarray(geometry)
    mask_j = jnp.asarray(mask)

    def components(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        variances = jnp.square(means_j) * jnp.exp(parameters)
        residue = _gamma_two_moment_uptake_jax(means_j, variances, times_j)
        predicted = mapping_j @ residue.T
        data = jnp.mean(jnp.square((predicted - observed_j)[mask_j]))
        penalty = _regularization_penalty(parameters, geometry_j)
        return data, penalty

    def objective(parameters: jax.Array) -> jax.Array:
        data, penalty = components(parameters)
        return data + regularization * penalty

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=float)

    initial = _initial_log_relative_variance(means, initial_variance)
    result = minimize(
        scipy_objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-18.0, 8.0)] * means.size,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    variances = np.square(means) * np.exp(np.asarray(result.x))
    covariance = build_hdx_covariance(variances, geometry)
    data_value, penalty_value = components(jnp.asarray(result.x))
    return TargetVarianceFit(
        estimator="curve_moment",
        geometry_name=geometry_name,
        regularization=float(regularization),
        variances=variances,
        covariance=covariance,
        mapped_covariance=map_hdx_covariance(covariance, peptide_map),
        predicted_uptake=predict_curve_moment_uptake(means, variances, times, peptide_map),
        objective=float(result.fun),
        data_objective=float(data_value),
        regularization_objective=float(penalty_value),
        success=bool(result.success and np.isfinite(result.fun)),
        message=str(result.message),
        iterations=int(result.nit),
    )


def propagated_uptake_covariance(
    mean_rates: np.ndarray,
    covariance: np.ndarray,
    timepoint: float,
    mapping: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Return ``M J_t C J_t M.T + epsilon I`` in peptide coordinates."""

    means = np.asarray(mean_rates, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    mapping = np.asarray(mapping, dtype=float)
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    jacobian = float(timepoint) * np.exp(-float(timepoint) * means)
    propagated = mapping @ (jacobian[:, None] * covariance * jacobian[None, :]) @ mapping.T
    return 0.5 * (propagated + propagated.T) + float(noise_variance) * np.eye(mapping.shape[0])


def structured_residual_nll(
    observed_uptake: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    covariance: np.ndarray,
    *,
    noise_variance: float = 1e-4,
    observation_mask: np.ndarray | None = None,
    predicted_uptake: np.ndarray | None = None,
) -> float:
    """Evaluate a normalized propagated-covariance Gaussian predictive score.

    By default the residual is about the fixed-mean curve, matching the structured-
    residual estimator.  Supplying ``predicted_uptake`` permits the curve-moment
    estimator to be compared with the same held-out proper scoring rule.
    """

    observed = np.asarray(observed_uptake, dtype=float)
    means = np.asarray(mean_rates, dtype=float)
    times = np.asarray(timepoints, dtype=float)
    peptide_map = np.asarray(mapping, dtype=float)
    if predicted_uptake is None:
        predicted = predict_fixed_mean_uptake(means, times, peptide_map)
    else:
        predicted = np.asarray(predicted_uptake, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("observed_uptake is not aligned to mapping and timepoints")
    mask = np.ones_like(observed, dtype=bool) if observation_mask is None else np.asarray(
        observation_mask, dtype=bool
    )
    if mask.shape != observed.shape or not mask.any():
        raise ValueError("observation_mask must select at least one observation")
    total = 0.0
    count = 0
    for time_index, timepoint in enumerate(times):
        indices = np.flatnonzero(mask[:, time_index])
        if indices.size == 0:
            continue
        selected_map = peptide_map[indices]
        sigma = propagated_uptake_covariance(
            means, covariance, float(timepoint), selected_map, noise_variance
        )
        residual = observed[indices, time_index] - predicted[indices, time_index]
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            return float("inf")
        total += 0.5 * (
            float(residual @ np.linalg.solve(sigma, residual))
            + float(logdet)
            + indices.size * np.log(2.0 * np.pi)
        )
        count += indices.size
    return float(total / count)


def fit_structured_residual_variance(
    observed_uptake: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    *,
    geometry_name: str = "custom",
    regularization: float = 0.0,
    noise_variance: float = 1e-4,
    observation_mask: np.ndarray | None = None,
    initial_variance: float | np.ndarray = 0.1,
    maxiter: int = 1000,
) -> TargetVarianceFit:
    """Fit structured-residual amplitudes with quadratic and log-det terms.

    This estimator describes model discrepancy until ISO qualification establishes
    correspondence with conformational rate variance.
    """

    observed, means, times, peptide_map, geometry, mask = _validate_fit_inputs(
        observed_uptake, mean_rates, timepoints, mapping, geometry, observation_mask
    )
    if regularization < 0 or noise_variance <= 0:
        raise ValueError("regularization must be non-negative and noise_variance positive")
    fixed_prediction = predict_fixed_mean_uptake(means, times, peptide_map)
    residual = observed - fixed_prediction
    means_j = jnp.asarray(means)
    mapping_j = jnp.asarray(peptide_map)
    geometry_j = jnp.asarray(geometry)
    residual_j = jnp.asarray(residual)
    selections = tuple(np.flatnonzero(mask[:, index]) for index in range(times.size))

    def components(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        variances = jnp.square(means_j) * jnp.exp(parameters)
        root = jnp.sqrt(variances)
        covariance = root[:, None] * geometry_j * root[None, :]
        nll = jnp.asarray(0.0, dtype=means_j.dtype)
        observations = 0
        for time_index, timepoint in enumerate(times):
            indices = selections[time_index]
            if indices.size == 0:
                continue
            selected_map = mapping_j[indices]
            jacobian = float(timepoint) * jnp.exp(-float(timepoint) * means_j)
            propagated = selected_map @ (
                jacobian[:, None] * covariance * jacobian[None, :]
            ) @ selected_map.T
            sigma = propagated + float(noise_variance) * jnp.eye(indices.size)
            factor = jnp.linalg.cholesky(sigma)
            selected_residual = residual_j[indices, time_index]
            solved = jax.scipy.linalg.cho_solve((factor, True), selected_residual)
            nll = nll + 0.5 * (
                selected_residual @ solved
                + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
                + indices.size * jnp.log(2.0 * jnp.pi)
            )
            observations += indices.size
        nll = nll / max(observations, 1)
        penalty = _regularization_penalty(parameters, geometry_j)
        return nll, penalty

    def objective(parameters: jax.Array) -> jax.Array:
        nll, penalty = components(parameters)
        return nll + regularization * penalty

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=float)

    initial = _initial_log_relative_variance(means, initial_variance)
    result = minimize(
        scipy_objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-18.0, 8.0)] * means.size,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-8},
    )
    variances = np.square(means) * np.exp(np.asarray(result.x))
    covariance = build_hdx_covariance(variances, geometry)
    data_value, penalty_value = components(jnp.asarray(result.x))
    return TargetVarianceFit(
        estimator="structured_residual_model_discrepancy",
        geometry_name=geometry_name,
        regularization=float(regularization),
        variances=variances,
        covariance=covariance,
        mapped_covariance=map_hdx_covariance(covariance, peptide_map),
        predicted_uptake=fixed_prediction,
        objective=float(result.fun),
        data_objective=float(data_value),
        regularization_objective=float(penalty_value),
        success=bool(result.success and np.isfinite(result.fun)),
        message=str(result.message),
        iterations=int(result.nit),
    )


def covariance_profiles(
    covariance: np.ndarray, *, relative_ridge: float = 1e-6, absolute_ridge: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return regularised marginal and conditional variance profiles."""

    covariance = np.asarray(covariance, dtype=float)
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = covariance + (relative_ridge * scale + absolute_ridge) * np.eye(
        covariance.shape[0]
    )
    marginal = np.diag(regularized)
    conditional = 1.0 / np.diag(np.linalg.inv(regularized))
    return marginal, conditional


def variance_recovery_metrics(
    inferred_variances: np.ndarray,
    true_variances: np.ndarray,
    *,
    floor: float = 1e-15,
) -> dict[str, float]:
    """Log-scale residue variance recovery metrics."""

    inferred = np.asarray(inferred_variances, dtype=float)
    truth = np.asarray(true_variances, dtype=float)
    if inferred.shape != truth.shape or np.any(inferred < 0) or np.any(truth < 0):
        raise ValueError("inferred and true variances must be aligned and non-negative")
    log_inferred = np.log(np.clip(inferred, floor, None))
    log_truth = np.log(np.clip(truth, floor, None))
    correlation = spearmanr(log_inferred, log_truth).statistic
    return {
        "log_variance_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "log_variance_rmse": float(np.sqrt(np.mean(np.square(log_inferred - log_truth)))),
    }


def qualification_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_panels: Sequence[str] = ("equal", "random_fixed", "random_variable"),
    required_ensembles: Sequence[str] = ("ISO_BI", "ISO_TRI"),
) -> dict[str, Any]:
    """Apply the preregistered TeaA/ISO qualification gate to summary rows."""

    if not rows:
        raise ValueError("qualification rows are empty")
    required = {
        "ensemble",
        "panel",
        "heldout_mean_mse_ratio",
        "log_variance_spearman",
        "mapped_variance_log_rmse",
        "constant_mapped_variance_log_rmse",
        "beats_shuffled_geometry",
        "psd",
        "finite_objective",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"qualification rows require fields {sorted(required)}")
    panels = sorted({str(row["panel"]) for row in rows})
    ensembles = sorted({str(row["ensemble"]) for row in rows})
    ratios = np.asarray([float(row["heldout_mean_mse_ratio"]) for row in rows])
    correlations = np.asarray([float(row["log_variance_spearman"]) for row in rows])
    mapped = np.asarray([float(row["mapped_variance_log_rmse"]) for row in rows])
    constant = np.asarray([float(row["constant_mapped_variance_log_rmse"]) for row in rows])
    reduction = 1.0 - np.median(mapped) / max(float(np.median(constant)), 1e-15)
    panel_shuffle = {
        panel: all(bool(row["beats_shuffled_geometry"]) for row in rows if str(row["panel"]) == panel)
        for panel in panels
    }
    checks = {
        "all_registered_panels_present": set(required_panels).issubset(panels),
        "both_iso_geometry_sources_present": set(required_ensembles).issubset(ensembles),
        "heldout_mean_mse_within_1p05": bool(np.median(ratios) <= 1.05),
        "median_log_variance_spearman_at_least_0p5": bool(np.median(correlations) >= 0.5),
        "mapped_variance_log_rmse_reduction_at_least_20pct": bool(reduction >= 0.20),
        "beats_shuffled_in_every_panel": bool(all(panel_shuffle.values())),
        "all_covariances_psd": bool(all(bool(row["psd"]) for row in rows)),
        "all_objectives_finite": bool(all(bool(row["finite_objective"]) for row in rows)),
    }
    return {
        "qualified": bool(all(checks.values())),
        "checks": checks,
        "median_heldout_mean_mse_ratio": float(np.median(ratios)),
        "median_log_variance_spearman": float(np.median(correlations)),
        "mapped_variance_log_rmse_reduction": float(reduction),
        "panel_beats_shuffled": panel_shuffle,
    }


def write_frozen_settings(
    path: str | Path,
    *,
    settings: Mapping[str, Any],
    qualification: Mapping[str, Any],
    input_hashes: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a qualified TeaA settings manifest without overwriting provenance."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen settings: {destination}")
    if not bool(qualification.get("qualified", False)):
        raise ValueError("cannot freeze settings that failed TeaA/ISO qualification")
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "qualified": True,
        "selection_source": "TeaA/ISO held-out HDX reconstruction only",
        "nmr_used_for_selection": False,
        "ensemble_reweighting_performed": False,
        "settings": dict(settings),
        "qualification": dict(qualification),
        "input_hashes": dict(input_hashes or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def load_frozen_settings(path: str | Path, *, require_qualified: bool = True) -> dict[str, Any]:
    """Load only artifacts belonging to this experiment; reject former Stage J."""

    source = Path(path)
    payload = json.loads(source.read_text())
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(
            "MoPrP target-variance validation requires a geometry-regularised HDX "
            "target-variance artifact; former Stage-J joint-geometry artifacts are prohibited"
        )
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported target-variance artifact version")
    if require_qualified and not bool(payload.get("qualified", False)):
        raise ValueError("MoPrP target-variance validation is blocked: TeaA qualification failed")
    if bool(payload.get("nmr_used_for_selection", True)):
        raise ValueError("frozen settings are invalid because NMR truth was used for selection")
    return payload
