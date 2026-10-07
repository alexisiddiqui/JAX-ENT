"""Numerical invariants of the disposable synthetic Sigma experiment."""
import importlib
import json

import numpy as np
import pytest


sigma = importlib.import_module("jaxent.examples.1_IsoValidation_OMass.fitting.jaxENT.compute_sigma_synthetic")


def fixture_covariance():
    values = np.array([[1., 3., 2., 7.], [4., 1., 5., 2.], [2., 6., 1., 4.]])
    weights = np.array([.1, .2, .3, .4])
    return values, weights, sigma.compute_weighted_covariance(values, weights)


def test_default_matches_existing_calculation():
    _, weights, raw = fixture_covariance()
    expected = raw + np.diag(np.full(len(raw), 1e-6))
    np.testing.assert_array_equal(sigma.construct_covariance(raw, weights), expected)


def test_sample_correction_matches_numpy_and_preserves_shape():
    values, weights, raw = fixture_covariance()
    corrected = sigma.construct_covariance(raw, weights, "weighted_sample", ridge=0)
    np.testing.assert_allclose(corrected, np.cov(values, aweights=weights, ddof=1))
    metrics = sigma.matrix_shape_metrics(corrected, raw, np.arange(3), permutations=19)
    assert metrics["mantel_r"] == pytest.approx(1)
    assert metrics["correlation_distance"] < 1e-14
    assert metrics["normalized_covariance_distance"] < 1e-14
    assert sigma.shape_metrics(np.diag(corrected), np.diag(raw))["profile_distance"] < 1e-14


def test_shrinkage_precedes_inverse_and_preserves_diagonal():
    _, weights, raw = fixture_covariance()
    partial = sigma.construct_covariance(raw, weights, target="diagonal", alpha=.4, ridge=0)
    np.testing.assert_allclose(np.diag(partial), np.diag(raw))
    upper = np.triu_indices(3, 1)
    np.testing.assert_allclose(partial[upper], .6*raw[upper])
    full = sigma.construct_covariance(raw, weights, target="diagonal", alpha=1)
    np.testing.assert_allclose(np.linalg.inv(full), np.diag(1/(np.diag(raw)+1e-6)))
    assert not np.allclose(np.diag(np.linalg.inv(raw)), 1/np.diag(raw))
    identity = sigma.construct_covariance(raw, weights, target="identity", alpha=1, ridge=0)
    np.testing.assert_allclose(identity, np.eye(3)*np.trace(raw)/3)


def test_ridge_thresholds_bracket_singular_and_stable_boundaries():
    raw = np.diag([0., 1., 4.])
    thresholds = sigma.ridge_thresholds(raw)
    assert 0 < thresholds["ridge_pd"] < thresholds["ridge_stable"] < 1e-6
    for column in ("ridge_pd", "ridge_stable"):
        ridge = thresholds[column]
        above = np.linalg.eigvalsh(raw+ridge*(1+1e-6)*np.eye(3))
        below = np.linalg.eigvalsh(raw+ridge*(1-1e-6)*np.eye(3))
        if column == "ridge_pd":
            assert above[0] > 3*np.finfo(float).eps*above[-1]
            assert below[0] <= 3*np.finfo(float).eps*below[-1]
        else:
            assert above[-1]/above[0] <= 1e8
            assert below[-1]/below[0] > 1e8
    assert sigma.ridge_thresholds(np.eye(3))["ridge_stable"] == 0
    diagonal = sigma.construct_covariance(raw, np.ones(4), target="diagonal", alpha=1, ridge=0)
    assert sigma.ridge_thresholds(diagonal)["ridge_pd"] > 0
    identity = sigma.construct_covariance(raw, np.ones(4), target="identity", alpha=.1, ridge=0)
    assert sigma.ridge_thresholds(identity)["ridge_stable"] == 0


def test_log_pf_reference_and_undefined_statistics():
    values, weights, covariance = fixture_covariance()
    direct = np.sum(weights*(values-(values@weights)[:, None])**2, axis=1)
    np.testing.assert_allclose(np.diag(covariance), direct)
    assert np.isnan(sigma.shape_metrics(np.ones(3), direct)["pearson"])
    metrics = sigma.matrix_shape_metrics(np.eye(3), np.eye(3), np.arange(3), permutations=19)
    assert np.isnan(metrics["mantel_r"])
    assert np.isnan(metrics["mantel_p"])
    assert metrics["correlation_distance"] == 0


def test_shape_comparison_has_no_absolute_variance_floor():
    _, _, raw = fixture_covariance()
    raw = raw*1e-16
    metrics = sigma.matrix_shape_metrics(raw*1.01, raw, np.arange(3), permutations=19)
    assert metrics["mantel_r"] == pytest.approx(1)
    assert metrics["correlation_distance"] < 1e-14


def test_invalid_constructions():
    with pytest.raises(ValueError, match="effective observation"):
        sigma.construct_covariance(np.eye(2), np.array([1., 0.]), "weighted_sample")
    with pytest.raises(ValueError, match="shrinkage target"):
        sigma.construct_covariance(np.eye(2), np.ones(2), alpha=.1)
    with pytest.raises(ValueError, match="positive spectral scale"):
        sigma.ridge_thresholds(np.zeros((2, 2)))


def test_disposable_sweep_exports_consistent_results(tmp_path, monkeypatch):
    values, weights, raw = fixture_covariance()
    values = np.vstack((np.zeros(values.shape[1]), values))
    raw = sigma.compute_weighted_covariance(values, weights)
    monkeypatch.setattr(sigma, "plot_diagnostics", lambda *args: None)
    matrix, profile = sigma.run_diagnostics(raw, weights, values+.2, tmp_path, "test", {},
                                          alphas=(0., .05, 1.), permutations=19)
    assert len(matrix) == 24
    assert len(profile) == 72
    stable = matrix[matrix.stage=="stable_ridge"]
    assert np.all(stable.condition <= 1e8*(1+1e-12))
    assert np.all(matrix.inverse_residual < 1e-6)
    manifest = json.loads((tmp_path/"diagnostics/manifest.json").read_text())
    assert manifest["excluded_indices"] == [0]
    with np.load(tmp_path/"diagnostics/matrices.npz") as archive:
        np.testing.assert_allclose(archive["log_pf_marginal"], np.diag(raw))
        np.testing.assert_allclose(archive["raw_population"], raw, rtol=1e-14, atol=1e-14)
    assert (tmp_path/"diagnostics/diagonal_profiles.csv").exists()
