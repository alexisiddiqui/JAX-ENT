"""Elastic-network model covariances from a single structure (ANM and GNM).

These give a *population-free* covariance prior from topology alone — no trajectory, no ensemble
weighting, no population.  Both return an ``(n_residues, n_residues)`` residue covariance so they
can be compared directly to a residue log-PF covariance and peptide-mapped with ``M C Mᵀ``.

- GNM (Gaussian network model): isotropic, scalar per residue.  Its Kirchhoff pseudo-inverse *is*
  the residue covariance, and it typically tracks residue flexibility (B-factors) better than ANM.
- ANM (anisotropic network model): 3N Hessian; the residue covariance is the trace of each 3x3
  block of the pseudo-inverse.
"""

from __future__ import annotations

import numpy as np

GNM_CUTOFF = 7.3  # Angstrom, standard GNM contact cutoff
ANM_CUTOFF = 15.0  # Angstrom, standard ANM contact cutoff


def _pairwise_sq_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def _pseudo_inverse(matrix: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Moore-Penrose inverse discarding near-zero eigen-modes (rigid body / degeneracy).

    Modes with eigenvalue at or below ``tol`` times the largest eigenvalue are dropped, so this is
    robust to degenerate (e.g. collinear) geometries that carry more than the nominal number of
    zero modes.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    threshold = tol * max(float(eigenvalues[-1]), 1e-30)
    inverse = np.zeros_like(matrix)
    for index in range(matrix.shape[0]):
        if eigenvalues[index] > threshold:
            inverse += np.outer(eigenvectors[:, index], eigenvectors[:, index]) / eigenvalues[index]
    return inverse


def kirchhoff(coords: np.ndarray, cutoff: float = GNM_CUTOFF) -> np.ndarray:
    """Return the GNM Kirchhoff (connectivity) matrix Γ."""

    coords = np.asarray(coords, dtype=float)
    within = (_pairwise_sq_distances(coords) <= cutoff**2).astype(float)
    np.fill_diagonal(within, 0.0)
    gamma = -within
    np.fill_diagonal(gamma, within.sum(axis=1))
    return gamma


def gnm_covariance(coords: np.ndarray, cutoff: float = GNM_CUTOFF) -> np.ndarray:
    """Return the GNM residue covariance (Kirchhoff pseudo-inverse; one zero mode removed)."""

    gamma = kirchhoff(coords, cutoff=cutoff)
    return _pseudo_inverse(gamma)


def anm_hessian(coords: np.ndarray, cutoff: float = ANM_CUTOFF, gamma: float = 1.0) -> np.ndarray:
    """Return the 3N x 3N ANM Hessian."""

    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    hessian = np.zeros((3 * n, 3 * n))
    sq = _pairwise_sq_distances(coords)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sq[i, j]
            if 1e-6 < d2 <= cutoff**2:
                rij = coords[j] - coords[i]
                super_element = -(gamma / d2) * np.outer(rij, rij)
                hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3] = super_element
                hessian[3 * j:3 * j + 3, 3 * i:3 * i + 3] = super_element
    for i in range(n):
        block = np.zeros((3, 3))
        for j in range(n):
            if j != i:
                block -= hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3]
        hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] = block
    return hessian


def anm_covariance(coords: np.ndarray, cutoff: float = ANM_CUTOFF, gamma: float = 1.0) -> np.ndarray:
    """Return the ANM residue covariance (trace of each 3x3 pseudo-inverse block; 6 zero modes)."""

    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    inverse = _pseudo_inverse(anm_hessian(coords, cutoff=cutoff, gamma=gamma))
    covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            covariance[i, j] = np.trace(inverse[3 * i:3 * i + 3, 3 * j:3 * j + 3])
    return covariance


def mean_square_fluctuations(covariance: np.ndarray) -> np.ndarray:
    """Return the per-residue MSF (diagonal of a residue covariance)."""

    return np.diag(np.asarray(covariance, dtype=float)).copy()
