#!/usr/bin/env python3
"""Numerical machinery for the MoPrP joint HDX noise model.

This module contains only model primitives.  Model fitting, cross-validation, and
scientific activation of finite-gating kinetics are deliberately deferred to the
later phases described in ``plans/hdx_noise_model_implementation_handoff.md``.

All flattened peptide/time arrays use time-major order: ``index = j * P + p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


class UptakeBackend(Protocol):
    """Swappable residue-uptake and log-PF-sensitivity interface."""

    def residue_uptake(self, log_pf, k_int, times, kinetics=None) -> Array: ...

    def logpf_sensitivity(self, log_pf, k_int, times, kinetics=None) -> Array: ...


def _validated_vectors(log_pf, k_int, times):
    log_pf = jnp.asarray(log_pf)
    k_int = jnp.asarray(k_int)
    times = jnp.asarray(times)
    if log_pf.ndim != 1 or k_int.shape != log_pf.shape:
        raise ValueError("log_pf and k_int must be aligned residue vectors")
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    return log_pf, k_int, times


@dataclass(frozen=True)
class EX2Backend:
    """Strict EX2 uptake with analytic log-PF sensitivity."""

    def residue_uptake(self, log_pf, k_int, times, kinetics=None) -> Array:
        """Return residue uptake with shape ``(R, T)``."""
        log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
        x = k_int[:, None] * jnp.exp(-log_pf[:, None]) * times[None, :]
        return -jnp.expm1(-x)

    def logpf_sensitivity(self, log_pf, k_int, times, kinetics=None) -> Array:
        """Return ``d uptake / d log_pf`` with shape ``(R, T)``."""
        log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
        x = k_int[:, None] * jnp.exp(-log_pf[:, None]) * times[None, :]
        return -x * jnp.exp(-x)


def _two_state_probabilities(log_pf: Array, k_int: Array, times: Array, log_gamma: Array) -> tuple[Array, Array]:
    """Propagate closed/open survival using a stable closed-form 2x2 exponential.

    The equilibrium open fraction is ``exp(-log_pf)``.  Consequently this backend
    requires PF >= 1 (log_pf >= 0), matching the kinetic parameterisation frozen in
    the handoff.  The result has shape ``(R, T)``.
    """
    p_open = jnp.exp(-log_pf)
    gamma = jnp.exp(log_gamma)
    if gamma.ndim == 0:
        gamma = jnp.broadcast_to(gamma, log_pf.shape)
    if gamma.shape != log_pf.shape:
        raise ValueError("log_gamma must be scalar or aligned with log_pf")

    k_open = gamma * p_open
    k_close = gamma * (1.0 - p_open)
    # A = [[-ko, kc], [ko, -(kc + ki)]] acts on column probabilities.
    a = -k_open
    b = k_close
    c = k_open
    d = -(k_close + k_int)
    half_trace = 0.5 * (a + d)
    delta_sq = (0.5 * (a - d)) ** 2 + b * c

    # Avoid sqrt'(0) in the inactive branch.  The series is analytic in delta_sq
    # and is selected near an eigenvalue collision, keeping both value and AD finite.
    threshold = jnp.asarray(1e-12, dtype=delta_sq.dtype)
    delta = jnp.sqrt(jnp.maximum(delta_sq, threshold))
    t = times[None, :]
    ht = half_trace[:, None] * t
    dt = delta[:, None] * t
    e_plus = jnp.exp(ht + dt)
    e_minus = jnp.exp(ht - dt)
    c0_regular = 0.5 * (e_plus + e_minus)
    c1_regular = 0.5 * (e_plus - e_minus) / delta[:, None]
    base = jnp.exp(ht)
    u = delta_sq[:, None] * t**2
    c0_series = base * (1.0 + 0.5 * u + u**2 / 24.0)
    c1_series = base * t * (1.0 + u / 6.0 + u**2 / 120.0)
    near = delta_sq[:, None] < threshold
    c0 = jnp.where(near, c0_series, c0_regular)
    c1 = jnp.where(near, c1_series, c1_regular)

    p0_c = 1.0 - p_open
    p0_o = p_open
    bc = (a - half_trace) * p0_c + b * p0_o
    bo = c * p0_c + (d - half_trace) * p0_o
    p_c = c0 * p0_c[:, None] + c1 * bc[:, None]
    p_o = c0 * p0_o[:, None] + c1 * bo[:, None]
    return p_c, p_o


def ll_state_probabilities(log_pf, k_int, times, log_gamma) -> tuple[Array, Array, Array]:
    """Return finite-gating ``(p_closed, p_open, p_exchanged)`` arrays of shape ``(R,T)``."""
    log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
    p_c, p_o = _two_state_probabilities(log_pf, k_int, times, jnp.asarray(log_gamma))
    return p_c, p_o, 1.0 - p_c - p_o


@dataclass(frozen=True)
class LLBackend:
    """Finite-gating Linderstrøm-Lang backend with AD-derived sensitivity."""

    def residue_uptake(self, log_pf, k_int, times, kinetics=None) -> Array:
        """Return residue uptake ``(R,T)``; ``kinetics`` is scalar/per-residue log gamma."""
        log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
        if kinetics is None:
            raise ValueError("LLBackend requires log_gamma kinetics")
        # Eager validation catches sentinels while tracers remain JIT/AD compatible.
        if not isinstance(log_pf, Tracer) and np.any(np.asarray(log_pf) < 0):
            raise ValueError("LLBackend requires PF >= 1 (log_pf >= 0)")
        log_gamma = jnp.asarray(kinetics)
        p_c, p_o = _two_state_probabilities(log_pf, k_int, times, log_gamma)
        return jnp.clip(1.0 - p_c - p_o, 0.0, 1.0)

    def logpf_sensitivity(self, log_pf, k_int, times, kinetics=None) -> Array:
        """Return the diagonal residue sensitivity ``d uptake_r / d log_pf_r``."""
        log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
        jac = jax.jacfwd(lambda z: self.residue_uptake(z, k_int, times, kinetics))(log_pf)
        return jnp.einsum("rtr->rt", jac)


def peptide_uptake(residue_uptake, mapping) -> Array:
    """Map ``(R,T)`` residue uptake to ``(P,T)`` peptide uptake."""
    return jnp.asarray(mapping) @ jnp.asarray(residue_uptake)


def interval_hazard_uptake(log_pf, k_int, times, offsets=None) -> Array:
    """Return interval-recursion uptake ``(R,T)`` on an irregular time grid."""
    log_pf, k_int, times = _validated_vectors(log_pf, k_int, times)
    previous = jnp.concatenate((jnp.zeros((1,), dtype=times.dtype), times[:-1]))
    dt = times - previous
    offsets = jnp.zeros_like(times) if offsets is None else jnp.asarray(offsets)
    hazards = dt[None, :] * k_int[:, None] * jnp.exp(-log_pf[:, None] + offsets[None, :])
    return -jnp.expm1(-jnp.cumsum(hazards, axis=1))


def vectorize_time_major(values) -> Array:
    """Flatten a ``(P,T)`` array so flat index is ``j * P + p``."""
    values = jnp.asarray(values)
    if values.ndim != 2:
        raise ValueError("values must have shape (P,T)")
    return values.T.reshape(-1)


def unvectorize_time_major(values, n_peptides: int, n_timepoints: int) -> Array:
    """Restore a time-major vector to shape ``(P,T)``."""
    values = jnp.asarray(values)
    if values.shape != (n_peptides * n_timepoints,):
        raise ValueError("vector length does not match P*T")
    return values.reshape(n_timepoints, n_peptides).T


def extract_time_blocks(covariance, n_peptides: int, n_timepoints: int) -> Array:
    """Extract all diagonal time blocks as ``(T,P,P)``."""
    covariance = jnp.asarray(covariance)
    n = n_peptides * n_timepoints
    if covariance.shape != (n, n):
        raise ValueError("covariance must have shape (P*T,P*T)")
    indices = jnp.arange(n).reshape(n_timepoints, n_peptides)
    return jax.vmap(lambda index: covariance[jnp.ix_(index, index)])(indices)


def stack_propagation_matrix(sensitivity, mapping) -> Array:
    """Build ``A`` with shape ``(P*T,R)`` in time-major order."""
    sensitivity = jnp.asarray(sensitivity)
    mapping = jnp.asarray(mapping)
    if sensitivity.ndim != 2 or mapping.ndim != 2 or mapping.shape[1] != sensitivity.shape[0]:
        raise ValueError("expected sensitivity (R,T) and mapping (P,R)")
    return jnp.einsum("pr,rt->tpr", mapping, sensitivity).reshape(-1, sensitivity.shape[0])


def schur_square_correlation(correlation) -> Array:
    """Unsigned correlation arm, preserving PSD and a unit diagonal."""
    correlation = jnp.asarray(correlation)
    return correlation * correlation


def domain_flip_correlation(correlation, signs) -> Array:
    """Apply ``S R S`` for a vector of +/-1 domain signs."""
    correlation, signs = jnp.asarray(correlation), jnp.asarray(signs)
    return signs[:, None] * correlation * signs[None, :]


def heteroscedastic_diagonal(mean, kappa, epsilon_u=0.05) -> Array:
    """Return mean-one acquisition variance shape in time-major order."""
    mu = vectorize_time_major(mean)
    shape = (1.0 - kappa) + kappa * (epsilon_u + (1.0 - epsilon_u) * 4.0 * mu * (1.0 - mu))
    return jnp.diag(shape / jnp.mean(shape))


def build_joint_covariance(
    propagation,
    logpf_covariance,
    *,
    peptide_loading=None,
    time_loading=None,
    time_correlation=None,
    acquisition_diagonal=None,
    tau_peptide=0.0,
    tau_time=0.0,
    sigma_exp=0.0,
    numerical_floor=0.0,
) -> Array:
    """Build the unnormalised ``(P*T,P*T)`` covariance from PSD components."""
    a = jnp.asarray(propagation)
    covariance = a @ jnp.asarray(logpf_covariance) @ a.T
    if peptide_loading is not None:
        z = jnp.asarray(peptide_loading)
        covariance = covariance + tau_peptide**2 * (z @ z.T)
    if time_loading is not None:
        z = jnp.asarray(time_loading)
        kt = jnp.eye(z.shape[1]) if time_correlation is None else jnp.asarray(time_correlation)
        covariance = covariance + tau_time**2 * (z @ kt @ z.T)
    if acquisition_diagonal is not None:
        covariance = covariance + sigma_exp**2 * jnp.asarray(acquisition_diagonal)
    return covariance + numerical_floor * jnp.eye(covariance.shape[0], dtype=covariance.dtype)


def gaussian_nll_from_cholesky(residual, chol) -> Array:
    """Full Gaussian NLL from a lower Cholesky factor, without an inverse."""
    residual, chol = jnp.asarray(residual), jnp.asarray(chol)
    whitened = solve_triangular(chol, residual, lower=True)
    n = residual.size
    return 0.5 * jnp.vdot(whitened, whitened) + jnp.sum(jnp.log(jnp.diag(chol))) + 0.5 * n * jnp.log(2.0 * jnp.pi)


def mixture_nll(mode_residuals, mode_chols, log_weights) -> Array:
    """Stable normalized Gaussian-mixture NLL over the leading mode axis."""
    nlls = jax.vmap(gaussian_nll_from_cholesky)(jnp.asarray(mode_residuals), jnp.asarray(mode_chols))
    log_weights = jnp.asarray(log_weights) - logsumexp(jnp.asarray(log_weights))
    return -logsumexp(log_weights - nlls)
