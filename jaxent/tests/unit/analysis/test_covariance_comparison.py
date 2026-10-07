"""Unit tests for the covariance-comparison metrics (incl. the Mantel test)."""

from __future__ import annotations

import numpy as np
import pytest

from jaxent.src.analysis import covariance_comparison as cc


def _random_covariance(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + np.eye(n)


def test_to_correlation_unit_diagonal():
    corr = cc.to_correlation(_random_covariance(6, 0))
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)


def test_mantel_identical_matrices_r_one_p_small():
    cov = _random_covariance(10, 1)
    r, p = cc.mantel_test(cov, cov, permutations=199, seed=0)
    assert r == pytest.approx(1.0, abs=1e-10)
    assert p <= 0.01


def test_mantel_independent_matrices_r_near_zero_p_large():
    r, p = cc.mantel_test(_random_covariance(20, 2), _random_covariance(20, 3), permutations=499, seed=0)
    assert abs(r) < 0.3
    assert p > 0.05


def test_mantel_is_seed_deterministic():
    a, b = _random_covariance(12, 4), _random_covariance(12, 5)
    assert cc.mantel_test(a, b, permutations=199, seed=7) == cc.mantel_test(a, b, permutations=199, seed=7)


def test_covariance_metrics_zero_distance_at_identity():
    cov = _random_covariance(8, 6)
    m = cc.covariance_metrics(cov, cov, permutations=99, seed=0)
    assert m["norm_distance"] == pytest.approx(0.0, abs=1e-10)
    assert m["offdiag_corr"] == pytest.approx(1.0, abs=1e-8)
    assert m["diag_log_corr"] == pytest.approx(1.0, abs=1e-8)
    assert m["mantel_r"] == pytest.approx(1.0, abs=1e-8)


def test_rebuild_covariance_sets_diagonal_and_preserves_structure():
    structure = _random_covariance(8, 10)
    variances = np.linspace(0.5, 4.0, 8)
    rebuilt = cc.rebuild_covariance(structure, variances)
    # diagonal equals the supplied variances
    np.testing.assert_allclose(np.diag(rebuilt), variances, atol=1e-10)
    # correlation structure is preserved
    np.testing.assert_allclose(cc.to_correlation(rebuilt), cc.to_correlation(structure), atol=1e-10)
    # symmetric and PSD (structure was PSD)
    np.testing.assert_allclose(rebuilt, rebuilt.T, atol=1e-12)
    assert np.linalg.eigvalsh(rebuilt).min() > -1e-8


def test_rebuild_covariance_scale_is_separable():
    structure = _random_covariance(6, 11)
    base = cc.rebuild_covariance(structure, np.ones(6))
    scaled = cc.rebuild_covariance(structure, 9.0 * np.ones(6))
    # scaling every variance by k scales the covariance by k
    np.testing.assert_allclose(scaled, 9.0 * base, atol=1e-10)


def test_trace_match_scale_sets_total_and_preserves_structure():
    structure = _random_covariance(7, 12)
    target_trace = 3.0
    variances = cc.trace_match_scale(structure, target_trace)
    rebuilt = cc.rebuild_covariance(structure, variances)
    assert np.trace(rebuilt) == pytest.approx(target_trace, rel=1e-10)
    # correlation pattern and relative variance profile preserved -> rebuilt is a scalar multiple
    ratio = rebuilt / structure
    np.testing.assert_allclose(ratio, ratio.flat[0], rtol=1e-8)


def test_covariance_metrics_scale_invariance_of_structure():
    # scaling a covariance changes magnitude (norm distance) but not correlation structure
    cov = _random_covariance(10, 7)
    m = cc.covariance_metrics(cov, 5.0 * cov, permutations=99, seed=0)
    assert m["offdiag_corr"] == pytest.approx(1.0, abs=1e-8)
    assert m["mantel_r"] == pytest.approx(1.0, abs=1e-8)
    assert m["diag_log_corr"] == pytest.approx(1.0, abs=1e-8)
    assert m["norm_distance"] > 0.1
