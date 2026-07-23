"""Unit tests for the elastic-network covariance priors (ANM / GNM)."""

from __future__ import annotations

import numpy as np
import pytest

from jaxent.src.analysis import elastic_network as en


def _chain(n=8, spacing=3.8):
    # a straight Cα chain
    coords = np.zeros((n, 3))
    coords[:, 0] = np.arange(n) * spacing
    return coords


def _helix(n=12):
    # a non-collinear α-helix-like Cα trace (avoids ANM's collinear degeneracy)
    t = np.arange(n)
    angle = np.deg2rad(100.0) * t
    return np.column_stack([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * t])


def test_kirchhoff_rows_sum_to_zero_and_one_zero_mode():
    gamma = en.kirchhoff(_chain(), cutoff=5.0)
    np.testing.assert_allclose(gamma.sum(axis=1), 0.0, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(gamma)
    # exactly one (translation) zero mode for a connected network
    assert np.sum(eigenvalues < 1e-8) == 1


def test_kirchhoff_connectivity_counts_neighbours():
    # spacing 3.8, cutoff 5.0 -> only sequential neighbours are in contact
    gamma = en.kirchhoff(_chain(n=4, spacing=3.8), cutoff=5.0)
    # interior residues have degree 2, ends degree 1
    np.testing.assert_array_equal(np.diag(gamma), np.array([1, 2, 2, 1]))
    assert gamma[0, 1] == -1 and gamma[0, 2] == 0


def test_gnm_covariance_symmetric_and_finite():
    cov = en.gnm_covariance(_chain(), cutoff=8.0)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    assert np.all(en.mean_square_fluctuations(cov) > 0)


def test_anm_hessian_has_six_zero_modes():
    hessian = en.anm_hessian(_helix(n=12), cutoff=15.0)
    eigenvalues = np.linalg.eigvalsh(hessian)
    # six rigid-body modes (3 translation + 3 rotation) for a non-degenerate structure
    assert np.sum(eigenvalues < 1e-6 * eigenvalues.max()) == 6


def test_anm_covariance_symmetric_and_finite():
    cov = en.anm_covariance(_helix(n=12), cutoff=15.0)
    assert cov.shape == (12, 12)
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)


def test_gnm_ends_more_flexible_than_middle():
    # a classic GNM signature: chain termini fluctuate more than the interior
    cov = en.gnm_covariance(_chain(n=12, spacing=3.8), cutoff=8.0)
    msf = en.mean_square_fluctuations(cov)
    assert msf[0] > msf[6] and msf[-1] > msf[6]
