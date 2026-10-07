"""Array-only structural distances for the ISO policy sidecars."""
from __future__ import annotations

import numpy as np


def median_scale(distances):
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distances must be square")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
        raise ValueError("Distances must be symmetric with zero diagonal")
    values = matrix[np.triu_indices(len(matrix), 1)]
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("No positive pairwise distances to scale")
    scale = float(np.median(positive))
    return matrix / scale, scale


def pairwise_ca_rmsd(coordinates):
    """All-pairs optimal proper rotations (Kabsch), in Angstrom."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected finite frame x CA atom x xyz coordinates")
    centered = xyz - xyz.mean(axis=1, keepdims=True)
    norms = np.sum(centered**2, axis=(1, 2))
    distances = np.zeros((len(xyz), len(xyz)), dtype=np.float64)
    for i in range(len(xyz) - 1):
        cross = np.einsum("ra,frb->fab", centered[i], centered[i+1:])
        u, singular, vh = np.linalg.svd(cross)
        signs = np.linalg.det(u @ vh)
        overlap = singular[:, 0] + singular[:, 1] + signs * singular[:, 2]
        values = np.sqrt(np.maximum(norms[i] + norms[i+1:] - 2*overlap, 0) / xyz.shape[1])
        distances[i, i+1:] = values
        distances[i+1:, i] = values
    return distances
