"""Utilities for matching conformational log-PF conditional variances.

This module deliberately keeps conformational covariance separate from observation
noise.  Frame weights describe a discrete structural ensemble, so covariance is the
weighted population covariance (there is no sample/Bessel correction).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve
from jax.typing import ArrayLike


def weights_from_logits(logits: ArrayLike) -> Array:
    """Map unconstrained frame logits to probabilities exactly once."""

    return jax.nn.softmax(jnp.asarray(logits))


def weighted_population_covariance(values: ArrayLike, weights: ArrayLike) -> Array:
    """Return the weighted population covariance of ``values`` over frames.

    Args:
        values: Array shaped ``(n_variables, n_frames)``.
        weights: Non-negative frame probabilities shaped ``(n_frames,)``.
    """

    values = jnp.asarray(values)
    weights = jnp.asarray(weights)
    weights = weights / jnp.sum(weights)
    mean = values @ weights
    centered = values - mean[:, None]
    return (centered * weights[None, :]) @ centered.T


def shrink_covariance(
    covariance: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_epsilon: float = 1e-8,
    absolute_epsilon: float = 1e-12,
) -> Array:
    """Apply differentiable isotropic shrinkage and a strictly positive floor."""

    covariance = jnp.asarray(covariance)
    dimension = covariance.shape[0]
    scale = jnp.trace(covariance) / dimension
    identity = jnp.eye(dimension, dtype=covariance.dtype)
    ridge = relative_epsilon * scale + absolute_epsilon
    return (1.0 - alpha) * covariance + alpha * scale * identity + ridge * identity


def conditional_variances_from_covariance(
    covariance: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_epsilon: float = 1e-8,
    absolute_epsilon: float = 1e-12,
) -> Array:
    """Compute ``1 / diag(C_shrunk**-1)`` using a Cholesky solve."""

    regularized = shrink_covariance(
        covariance,
        alpha=alpha,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=absolute_epsilon,
    )
    factor = jnp.linalg.cholesky(regularized)
    precision = cho_solve((factor, True), jnp.eye(regularized.shape[0], dtype=regularized.dtype))
    return 1.0 / jnp.diag(precision)


def covariance_profiles_from_covariance(
    covariance: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_epsilon: float = 1e-8,
    absolute_epsilon: float = 1e-12,
) -> tuple[Array, Array, Array, Array]:
    """Return regularized covariance, marginal, precision, and conditional diagonals.

    Profiles must be extracted after any linear coordinate mapping has been applied to
    the full covariance.  In particular, neither marginal nor conditional variances
    transform linearly under a peptide sparse map.
    """

    regularized = shrink_covariance(
        covariance,
        alpha=alpha,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=absolute_epsilon,
    )
    factor = jnp.linalg.cholesky(regularized)
    precision = cho_solve(
        (factor, True), jnp.eye(regularized.shape[0], dtype=regularized.dtype)
    )
    precision_diagonal = jnp.diag(precision)
    return (
        regularized,
        jnp.diag(regularized),
        precision_diagonal,
        1.0 / precision_diagonal,
    )


def conditional_variance_profile(
    values: ArrayLike,
    weights: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_epsilon: float = 1e-8,
    absolute_epsilon: float = 1e-12,
) -> Array:
    """Compute a correlation-aware conditional-variance profile over frames."""

    covariance = weighted_population_covariance(values, weights)
    return conditional_variances_from_covariance(
        covariance,
        alpha=alpha,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=absolute_epsilon,
    )


def marginal_variance_profile(
    values: ArrayLike,
    weights: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_epsilon: float = 1e-8,
    absolute_epsilon: float = 1e-12,
) -> Array:
    """Compute the shrunk marginal variance ``diag(C)`` over frames."""

    covariance = weighted_population_covariance(values, weights)
    regularized = shrink_covariance(
        covariance,
        alpha=alpha,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=absolute_epsilon,
    )
    return jnp.diag(regularized)


def conditional_variance_log_ratio_loss(predicted: ArrayLike, target: ArrayLike) -> Array:
    """Symmetric, dimensionless squared log-ratio profile distance."""

    predicted = jnp.asarray(predicted)
    target = jnp.asarray(target)
    return jnp.mean(jnp.square(jnp.log(predicted / target)))


def peptide_overlap_similarity(mapping: ArrayLike) -> Array:
    """Return cosine similarity between row-normalized peptide maps."""

    mapping = jnp.asarray(mapping)
    norms = jnp.linalg.norm(mapping, axis=1)
    if bool(jnp.any(norms <= 0)):
        raise ValueError("Peptide mapping contains an empty row")
    normalized = mapping / norms[:, None]
    return normalized @ normalized.T


def inverse_overlap_degree_weights(mapping: ArrayLike) -> Array:
    """Return mean-one weights that downweight redundant peptide rows."""

    similarity = peptide_overlap_similarity(mapping)
    weights = 1.0 / jnp.sum(similarity, axis=1)
    return weights / jnp.mean(weights)


def weighted_variance_log_ratio_loss(
    predicted: ArrayLike, target: ArrayLike, weights: ArrayLike
) -> Array:
    """Symmetric squared log-ratio loss with fixed peptide weights."""

    predicted = jnp.asarray(predicted)
    target = jnp.asarray(target)
    weights = jnp.asarray(weights)
    residual = jnp.log(predicted / target)
    return jnp.sum(weights * jnp.square(residual)) / jnp.sum(weights)


def overlap_projection(
    mapping: ArrayLike, relative_threshold: float = 1e-6
) -> Array:
    """Return columns spanning non-redundant cosine-overlap modes."""

    similarity = peptide_overlap_similarity(mapping)
    eigenvalues, eigenvectors = jnp.linalg.eigh(similarity)
    keep = eigenvalues > relative_threshold * jnp.max(eigenvalues)
    return eigenvectors[:, keep]


def _symmetric_matrix_log(matrix: ArrayLike) -> Array:
    matrix = jnp.asarray(matrix)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvectors * jnp.log(eigenvalues)[None, :]) @ eigenvectors.T


def projected_log_euclidean_covariance_loss(
    predicted_covariance: ArrayLike,
    target_covariance: ArrayLike,
    projection: ArrayLike,
    alpha: float | ArrayLike = 0.05,
) -> Array:
    """Symmetric covariance distance after removing redundant peptide modes."""

    projection = jnp.asarray(projection)
    predicted = projection.T @ jnp.asarray(predicted_covariance) @ projection
    target = projection.T @ jnp.asarray(target_covariance) @ projection
    predicted = shrink_covariance(predicted, alpha=alpha)
    target = shrink_covariance(target, alpha=alpha)
    difference = _symmetric_matrix_log(predicted) - _symmetric_matrix_log(target)
    return jnp.mean(jnp.square(difference))


def average_first_uptake(
    log_pf_by_frame: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
    weights: ArrayLike,
) -> Array:
    """Production BV semantics: average log-PF, then transform to uptake."""

    mean_log_pf = jnp.asarray(log_pf_by_frame) @ jnp.asarray(weights)
    return uptake_from_log_pf(mean_log_pf, k_ints, timepoints)


def uptake_from_log_pf(
    log_pf: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
) -> Array:
    """Compute bounded residue uptake from one latent residue log-PF vector."""

    pf = jnp.exp(jnp.asarray(log_pf))
    return 1.0 - jnp.exp(
        -jnp.asarray(timepoints)[:, None] * jnp.asarray(k_ints)[None, :] / pf[None, :]
    )


def uptake_log_pf_jacobian(
    log_pf: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
) -> Array:
    """Return the diagonal derivative of residue uptake with respect to log-PF.

    The returned array is shaped ``(n_timepoints, n_residues)`` and contains the
    diagonal of each timepoint-specific Jacobian.  For
    ``u = 1 - exp(-k_int * t / exp(z))``, the derivative is ``-x * exp(-x)``
    where ``x = k_int * t / exp(z)``.
    """

    log_pf = jnp.asarray(log_pf)
    x = (
        jnp.asarray(timepoints)[:, None]
        * jnp.asarray(k_ints)[None, :]
        / jnp.exp(log_pf)[None, :]
    )
    return -x * jnp.exp(-x)


def average_after_uptake(
    log_pf_by_frame: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
    weights: ArrayLike,
) -> Array:
    """Sensitivity target: transform every frame, then average uptake."""

    pf = jnp.exp(jnp.asarray(log_pf_by_frame))
    uptake_by_frame = 1.0 - jnp.exp(
        -jnp.asarray(timepoints)[:, None, None]
        * jnp.asarray(k_ints)[None, :, None]
        / pf[None, :, :]
    )
    return jnp.einsum("tdf,f->td", uptake_by_frame, jnp.asarray(weights))


def framewise_uptake(
    log_pf_by_frame: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
) -> Array:
    """Transform every residue/frame log-PF into uptake without frame averaging.

    Returns an array shaped ``(n_timepoints, n_residues, n_frames)``.  This is
    useful for studying conformational uptake covariance; it is not the
    production average-first ensemble prediction.
    """

    pf = jnp.exp(jnp.asarray(log_pf_by_frame))
    return 1.0 - jnp.exp(
        -jnp.asarray(timepoints)[:, None, None]
        * jnp.asarray(k_ints)[None, :, None]
        / pf[None, :, :]
    )


def map_residue_uptake_to_peptides(mapping: ArrayLike, residue_uptake: ArrayLike) -> Array:
    """Average residue uptake into peptides after the nonlinear forward transform."""

    return jnp.asarray(residue_uptake) @ jnp.asarray(mapping).T


def map_framewise_residue_uptake_to_peptides(
    mapping: ArrayLike, residue_uptake_by_frame: ArrayLike
) -> Array:
    """Average framewise residue uptake into peptides after uptake conversion.

    Args:
        mapping: Row-normalized peptide-by-residue map.
        residue_uptake_by_frame: Array shaped
            ``(n_timepoints, n_residues, n_frames)``.

    Returns:
        Array shaped ``(n_timepoints, n_peptides, n_frames)``.
    """

    return jnp.einsum(
        "pr,trf->tpf", jnp.asarray(mapping), jnp.asarray(residue_uptake_by_frame)
    )


def map_frame_log_pf_to_peptides(mapping: ArrayLike, log_pf_by_frame: ArrayLike) -> Array:
    """Map per-frame residue log-PFs into peptide means before frame covariance."""

    return jnp.asarray(mapping) @ jnp.asarray(log_pf_by_frame)


def curve_precision_from_target_uptake(
    target_uptake: ArrayLike, alpha: float | ArrayLike = 0.05
) -> Array:
    """Build trace-normalized peptide curve geometry from target uptake over time."""

    target_uptake = jnp.asarray(target_uptake)
    covariance = weighted_population_covariance(
        target_uptake.T,
        jnp.full(target_uptake.shape[0], 1.0 / target_uptake.shape[0]),
    )
    regularized = shrink_covariance(covariance, alpha=alpha)
    factor = jnp.linalg.cholesky(regularized)
    precision = cho_solve((factor, True), jnp.eye(regularized.shape[0], dtype=regularized.dtype))
    return trace_normalize_precision(precision)


def trace_normalize_precision(precision: ArrayLike) -> Array:
    """Match the production convention ``trace(W) == dimension``."""

    precision = jnp.asarray(precision)
    dimension = precision.shape[0]
    return precision * (dimension / jnp.trace(precision))


def covariance_mse(predicted: ArrayLike, target: ArrayLike, precision: ArrayLike) -> Array:
    """Mean ``0.5 * r.T @ W @ r`` over timepoints and variables."""

    residual = jnp.asarray(predicted) - jnp.asarray(target)
    weighted_squares = jnp.einsum("ti,ij,tj->t", residual, precision, residual)
    return 0.5 * jnp.mean(weighted_squares) / residual.shape[1]


def kl_to_uniform(weights: ArrayLike) -> Array:
    """Return ``KL(weights || uniform)`` with numerically safe probabilities."""

    weights = jnp.asarray(weights)
    weights = weights / jnp.sum(weights)
    tiny = jnp.finfo(weights.dtype).tiny
    safe_weights = jnp.maximum(weights, tiny)
    return jnp.sum(safe_weights * (jnp.log(safe_weights) + jnp.log(weights.size)))


def jensen_shannon_divergence(probabilities: ArrayLike, target: ArrayLike) -> Array:
    """Return base-2 JSD, bounded between zero and one.

    The inputs may contain zero-probability populations, as occurs for the TRI
    decoy population in the known target.
    """

    probabilities = jnp.asarray(probabilities)
    target = jnp.asarray(target)
    probabilities = probabilities / jnp.sum(probabilities)
    target = target / jnp.sum(target)
    midpoint = 0.5 * (probabilities + target)

    def entropy(values: Array) -> Array:
        tiny = jnp.finfo(values.dtype).tiny
        return -jnp.sum(values * jnp.log2(jnp.maximum(values, tiny)))

    return entropy(midpoint) - 0.5 * entropy(probabilities) - 0.5 * entropy(target)


def jensen_shannon_recovery_percent(probabilities: ArrayLike, target: ArrayLike) -> Array:
    """Return ``100 * (1 - sqrt(JSD))`` for population probabilities."""

    divergence = jnp.clip(jensen_shannon_divergence(probabilities, target), 0.0, 1.0)
    return 100.0 * (1.0 - jnp.sqrt(divergence))


def conditional_subset_effective_sample_size(weights: ArrayLike, mask: ArrayLike) -> Array:
    """Return ESS after conditioning weights on membership in ``mask``."""

    selected = jnp.asarray(weights)[jnp.asarray(mask, dtype=bool)]
    selected = selected / jnp.sum(selected)
    return 1.0 / jnp.sum(jnp.square(selected))
