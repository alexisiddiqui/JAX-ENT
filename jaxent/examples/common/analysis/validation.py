from __future__ import annotations

import numpy as np


def _is_valid_array(arr: np.ndarray) -> bool:
    """Return ``True`` if *arr* is non-None, non-empty, and not all-NaN."""
    if arr is None:
        return False
    if arr.size == 0:
        return False
    if np.all(np.isnan(arr)):
        return False
    return True


def _align_uptake_shape(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    """Remove leading singleton axes until ``arr.ndim == target_ndim``."""
    aligned = arr
    while aligned.ndim > target_ndim and aligned.shape[0] == 1:
        aligned = np.squeeze(aligned, axis=0)
    return aligned


def get_experimental_uptake(data) -> np.ndarray:
    """Extract ``.dfrac`` from a list of ``ExpD_Datapoint`` objects."""
    if not data:
        return np.array([])
    return np.array([d.dfrac for d in data]).squeeze()


def calculate_mse(pred: np.ndarray, exp: np.ndarray) -> float:
    """MSE between predicted and experimental uptake with shape alignment.

    Uses :func:`_is_valid_array` and :func:`_align_uptake_shape` for robustness.
    """
    if not _is_valid_array(np.asarray(pred) if pred is not None else None) or not _is_valid_array(
        np.asarray(exp) if exp is not None else None
    ):
        return np.nan

    pred = _align_uptake_shape(np.asarray(pred), np.asarray(exp).ndim)
    exp = np.asarray(exp)

    if pred.size == 0:
        return np.nan

    if pred.shape != exp.shape:
        if pred.ndim >= 2 and exp.ndim >= 2 and pred.shape[0] == exp.shape[0]:
            min_t = min(pred.shape[1], exp.shape[1])
            pred = pred[:, :min_t]
            exp = exp[:, :min_t]
        else:
            return np.nan

    return float(np.nanmean((pred - exp) ** 2))
