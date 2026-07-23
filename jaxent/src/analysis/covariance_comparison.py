"""Metrics for comparing two covariance matrices.

Used to ask how close a candidate covariance (unweighted trajectory, elastic-network prior, or a
linear-model prediction) is to a target covariance, separating *structure* (correlation pattern)
from *magnitude* (scale).  The Mantel test gives a permutation null for structural similarity that
respects the non-independence of covariance entries.
"""

from __future__ import annotations

import numpy as np


def to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Return the correlation matrix of a covariance (diagonal scaled to one)."""

    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    return covariance / np.outer(scale, scale)


def offdiagonal(matrix: np.ndarray) -> np.ndarray:
    """Return the strict upper-triangle (off-diagonal) entries as a flat vector."""

    matrix = np.asarray(matrix, dtype=float)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def mantel_test(a: np.ndarray, b: np.ndarray, permutations: int = 999, seed: int = 0) -> tuple[float, float]:
    """Mantel test on the correlation structure of two symmetric matrices.

    Correlates the off-diagonal correlation entries of ``a`` and ``b`` and builds a permutation
    null by relabelling the rows/columns of ``b``.  Returns ``(mantel_r, p_value)``.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Mantel test requires two square matrices of equal shape")
    ca, cb = to_correlation(a), to_correlation(b)
    va = offdiagonal(ca)
    observed = float(np.corrcoef(va, offdiagonal(cb))[0, 1])
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        permuted = offdiagonal(cb[perm][:, perm])
        if abs(float(np.corrcoef(va, permuted)[0, 1])) >= abs(observed):
            count += 1
    return observed, (count + 1) / (permutations + 1)


def rebuild_covariance(structure: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Rebuild a covariance from a correlation *structure* and a *scale* (variance diagonal).

    ``structure`` supplies only the correlation pattern (it is renormalized to a correlation
    matrix internally, so any covariance may be passed); ``variances`` supplies the per-element
    magnitude.  Returns ``D^{1/2} R D^{1/2}``, whose diagonal equals ``variances`` and whose
    correlation equals ``structure``'s.  This is the Stage D construction: population-free
    structure combined with a separately sourced (e.g. mean-derived) scale.
    """

    correlation = to_correlation(structure)
    scale = np.sqrt(np.clip(np.asarray(variances, dtype=float), 0.0, None))
    return correlation * np.outer(scale, scale)


def trace_match_scale(structure: np.ndarray, target_trace: float) -> np.ndarray:
    """Return per-element variances that keep ``structure``'s variance *shape* but set the total.

    The variances are proportional to ``structure``'s own diagonal and sum to ``target_trace``,
    so ``rebuild_covariance(structure, trace_match_scale(structure, t))`` preserves the correlation
    pattern *and* the relative variance profile while matching an externally supplied total
    magnitude ``t`` (e.g. the trace of a mean-only-fit covariance).
    """

    diag = np.clip(np.diag(np.asarray(structure, dtype=float)), 1e-12, None)
    return diag * (float(target_trace) / diag.sum())


def _log_diag(matrix: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.diag(np.asarray(matrix, dtype=float)), 1e-12, None))


def covariance_metrics(candidate: np.ndarray, target: np.ndarray, permutations: int = 999, seed: int = 0) -> dict:
    """Return structure/magnitude comparison metrics between candidate and target.

    - ``norm_distance``: relative Frobenius distance (magnitude + structure);
    - ``offdiag_corr``: Pearson correlation of the raw off-diagonal covariance entries;
    - ``diag_log_corr``: correlation of the log-variance profiles (the diagonal / magnitude ranking);
    - ``mantel_r`` / ``mantel_p``: structural (correlation-pattern) similarity with a permutation null.
    """

    candidate = np.asarray(candidate, dtype=float)
    target = np.asarray(target, dtype=float)
    norm_distance = float(np.linalg.norm(candidate - target) / (np.linalg.norm(target) + 1e-12))
    offdiag_corr = float(np.corrcoef(offdiagonal(candidate), offdiagonal(target))[0, 1])
    diag_log_corr = float(np.corrcoef(_log_diag(candidate), _log_diag(target))[0, 1])
    mantel_r, mantel_p = mantel_test(candidate, target, permutations=permutations, seed=seed)
    return {
        "norm_distance": norm_distance,
        "offdiag_corr": offdiag_corr,
        "diag_log_corr": diag_log_corr,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
    }
