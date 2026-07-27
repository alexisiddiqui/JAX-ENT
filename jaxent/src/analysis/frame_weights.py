"""Validation helpers for normalized frame-weight analysis inputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validated_frame_weight_simplex(
    values: ArrayLike, *, atol: float = 1e-5, context: str | None = None
) -> NDArray[np.float64]:
    weights = np.asarray(values, dtype=float)
    prefix = f"{context}: " if context else ""
    if weights.ndim != 1:
        raise ValueError(f"{prefix}Frame-weight simplex must be 1D, got {weights.shape}")
    if weights.size == 0:
        raise ValueError(f"{prefix}Frame-weight simplex is empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{prefix}Frame-weight simplex contains non-finite values")
    if np.any(weights < -atol):
        raise ValueError(f"{prefix}Frame-weight simplex contains negative values")
    total = weights.sum()
    if not np.isclose(total, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{prefix}Frame-weight simplex sums to {total}, expected 1")
    weights = np.clip(weights, 0.0, None)
    return weights / weights.sum()


def validated_frame_weight_simplex_rows(
    values: ArrayLike, *, atol: float = 1e-5, context: str | None = None
) -> NDArray[np.float64]:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"Frame-weight simplex rows must be 2D, got {rows.shape}")
    return np.stack(
        [
            validated_frame_weight_simplex(
                row, atol=atol, context=f"{context or 'frame weights'} row {index}"
            )
            for index, row in enumerate(rows)
        ]
    )
