"""Frame-averaging semantics for synthetic HDX uptake targets.

All functions are side-effect free and accept array-like inputs.  ``log_pf`` is
shaped ``(n_residues, n_frames)`` and uptake outputs are shaped
``(n_timepoints, n_residues)``.
"""

from __future__ import annotations

import numpy as np

from jaxent.src.analysis.pf_variance import uptake_from_log_pf


def weights_from_cluster_populations(
    assignments: np.ndarray, populations: dict[int, float]
) -> np.ndarray:
    """Spread each requested cluster mass uniformly over its member frames."""
    assignments = np.asarray(assignments)
    weights = np.zeros(assignments.shape, dtype=np.float64)
    if assignments.ndim != 1:
        raise ValueError("assignments must be one-dimensional")
    total = float(sum(populations.values()))
    if not np.isfinite(total) or total <= 0 or any(value < 0 for value in populations.values()):
        raise ValueError("populations must be finite, non-negative, and have positive total mass")
    for label, population in populations.items():
        mask = assignments == label
        count = int(mask.sum())
        mass = float(population) / total
        if mass > 0 and count == 0:
            raise ValueError(f"cluster {label} has positive mass but no frames")
        if count:
            weights[mask] = mass / count
    unrequested = set(np.unique(assignments)) - set(populations)
    if unrequested:
        raise ValueError(f"populations missing cluster labels: {sorted(unrequested)}")
    return weights


def effective_rates(log_pf: np.ndarray, k_ints: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """Return EX2 or full Linderstrom--Lang effective exchange rates."""
    log_pf = np.asarray(log_pf, dtype=np.float64)
    k_ints = np.asarray(k_ints, dtype=np.float64)
    if log_pf.ndim != 2 or k_ints.shape != (log_pf.shape[0],):
        raise ValueError("expected log_pf (n_residues, n_frames) and k_ints (n_residues,)")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if np.any(log_pf < 0):
        raise ValueError("log_pf must be non-negative so protection = 1/f_open is at least one")
    if tau == 0:
        return k_ints[:, None] * np.exp(-log_pf)
    protection = np.exp(log_pf)
    return k_ints[:, None] / (protection * (1.0 + tau * k_ints[:, None]))


def _validate_weights(log_pf: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (np.asarray(log_pf).shape[1],):
        raise ValueError("weights length must match the frame dimension")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    return weights


def residue_uptake_legacy(log_pf, k_ints, timepoints, weights, tau: float = 0.0):
    """Average log-PF first, then transform it to uptake."""
    weights = _validate_weights(log_pf, weights)
    mean_log_pf = np.asarray(log_pf, dtype=np.float64) @ weights
    if tau == 0:
        return np.asarray(uptake_from_log_pf(mean_log_pf, k_ints, timepoints))
    rate = effective_rates(mean_log_pf[:, None], k_ints, tau)[:, 0]
    return 1.0 - np.exp(-np.asarray(timepoints)[:, None] * rate[None, :])


def residue_uptake_fast(log_pf, k_ints, timepoints, weights, tau: float = 0.0):
    """Average effective frame rates, then transform the mean rate to uptake."""
    weights = _validate_weights(log_pf, weights)
    mean_rate = effective_rates(log_pf, k_ints, tau) @ weights
    return 1.0 - np.exp(-np.asarray(timepoints)[:, None] * mean_rate[None, :])


def residue_uptake_slow2(log_pf, k_ints, timepoints, weights, assignments, tau: float = 0.0):
    """Average rates within clusters, then mix cluster uptake curves."""
    weights = _validate_weights(log_pf, weights)
    assignments = np.asarray(assignments)
    if assignments.shape != weights.shape:
        raise ValueError("assignments length must match weights")
    rates = effective_rates(log_pf, k_ints, tau)
    uptake = np.zeros((len(timepoints), rates.shape[0]), dtype=np.float64)
    for label in np.unique(assignments):
        mask = assignments == label
        mass = weights[mask].sum()
        if mass == 0:
            continue
        cluster_rate = rates[:, mask] @ (weights[mask] / mass)
        uptake += mass * (1.0 - np.exp(-np.asarray(timepoints)[:, None] * cluster_rate[None, :]))
    return uptake
