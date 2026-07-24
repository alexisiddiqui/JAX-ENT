"""Shared centroid/shape decomposition for HDX isotope envelopes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


def envelope_moments(probabilities: np.ndarray) -> tuple[float, float]:
    """Return the centroid and variance on nominal mass-bin support."""
    probabilities = np.asarray(probabilities, dtype=float)
    x = np.arange(len(probabilities), dtype=float)
    mean = float(x @ probabilities)
    return mean, float(((x - mean) ** 2) @ probabilities)


def observe_shifted_full(
    full: np.ndarray, delta: float, n_bins: int
) -> np.ndarray:
    """Translate full support, then select and normalize the fixed window."""
    full = np.asarray(full, dtype=float)
    positions = np.arange(len(full), dtype=float) + float(delta)
    lower = np.floor(positions).astype(int)
    fraction = positions - lower
    observed = np.zeros(n_bins, dtype=float)
    for indices, weights in (
        (lower, full * (1.0 - fraction)),
        (lower + 1, full * fraction),
    ):
        valid = (indices >= 0) & (indices < n_bins)
        np.add.at(observed, indices[valid], weights[valid])
    if observed.sum() <= 0:
        raise ValueError("shifted distribution has no mass in the observed window")
    return observed / observed.sum()


def centroid_aligned_shift(
    full: np.ndarray, observed: np.ndarray, n_bins: int
) -> dict[str, float]:
    """Align centroids using the physical shift-then-window observation model."""
    observed_mean, _ = envelope_moments(observed)

    def objective(delta: float) -> float:
        candidate_mean, _ = envelope_moments(
            observe_shifted_full(full, delta, n_bins)
        )
        return (candidate_mean - observed_mean) ** 2

    scan = np.linspace(-float(n_bins) + 1e-6, float(n_bins) - 1e-6, 4001)
    losses = np.asarray([objective(delta) for delta in scan])
    near_best = np.flatnonzero(
        losses <= max(float(losses.min()) + 1e-10, 1e-10)
    )
    scan_index = int(near_best[np.argmin(np.abs(scan[near_best]))])
    lower_index = max(0, scan_index - 1)
    upper_index = min(len(scan) - 1, scan_index + 1)
    result = minimize_scalar(
        objective,
        bounds=(float(scan[lower_index]), float(scan[upper_index])),
        method="bounded",
        options={"xatol": 1e-12},
    )
    delta = float(result.x)
    shifted = observe_shifted_full(full, delta, n_bins)
    predicted = observe_shifted_full(full, 0.0, n_bins)
    total = float(np.sum((predicted - observed) ** 2))
    residual = float(np.sum((shifted - observed) ** 2))
    return {
        "best_shift_bins": delta,
        "aligned_centroid_gap": float(
            envelope_moments(shifted)[0] - observed_mean
        ),
        "total_sse": total,
        "centroid_component_sse": max(0.0, total - residual),
        "shape_component_sse": residual,
        "centroid_explained_fraction": (
            max(0.0, total - residual) / total if total else 0.0
        ),
    }


def decompose_envelope(
    full: np.ndarray, observed: np.ndarray, n_bins: int
) -> dict[str, float]:
    """Return the shared Stage-(b) aligned width decomposition."""
    observed = np.asarray(observed, dtype=float)
    result = centroid_aligned_shift(full, observed, n_bins)
    aligned = observe_shifted_full(full, result["best_shift_bins"], n_bins)
    observed_mean, observed_var = envelope_moments(observed)
    predicted = observe_shifted_full(full, 0.0, n_bins)
    predicted_mean, predicted_var = envelope_moments(predicted)
    _, aligned_var = envelope_moments(aligned)
    return {
        "predicted_centroid": predicted_mean,
        "observed_centroid": observed_mean,
        "centroid_gap": predicted_mean - observed_mean,
        "predicted_width_var": predicted_var,
        "observed_width_var": observed_var,
        "width_ratio": predicted_var / observed_var,
        "centroid_aligned_width_var": aligned_var,
        "centroid_aligned_width_ratio": aligned_var / observed_var,
        **result,
    }


def precision_band_decision(
    average_first_ratio: float,
    frame_mixture_ratio: float,
    precision_floor: float,
    relative_half_width: float = 0.25,
) -> dict[str, Any]:
    """Apply the shared empirical precision-band / separation rule."""
    lower = (1.0 - relative_half_width) * float(precision_floor)
    upper = (1.0 + relative_half_width) * float(precision_floor)
    separation = bool(
        average_first_ratio < lower
        and lower <= frame_mixture_ratio <= upper
    )
    return {
        "precision_band_lower": lower,
        "precision_band_upper": upper,
        "separation_survives": separation,
        "detected_excess_width": max(
            0.0, lower - float(average_first_ratio)
        ),
    }
