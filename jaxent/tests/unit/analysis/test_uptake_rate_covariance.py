import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[3]/"examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_uptake_rate_covariance.py"
SPEC = importlib.util.spec_from_file_location("uptake_rate_covariance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_intrinsic_rates_scale_inverse_pf_covariance_pairwise():
    z = np.asarray([[0., 1., 2.], [1., 1.5, 3.]])
    kint = np.asarray([0.4, 1.2])
    weights = np.asarray([0.2, 0.3, 0.5])
    rates = MODULE.effective_rates(z, kint)
    inverse_pf = np.exp(-z.T)
    expected = kint[:, None]*MODULE.weighted_covariance(inverse_pf, weights)*kint[None, :]
    np.testing.assert_allclose(MODULE.weighted_covariance(rates, weights), expected)


def test_apparent_rate_exactly_inverts_framewise_uptake():
    rates = np.asarray([[0.02, 0.4], [0.1, 0.7]])
    for time in (0.167, 10.0):
        uptake = 1-np.exp(-time*rates)
        np.testing.assert_allclose(MODULE.apparent_rate(uptake, time), rates, rtol=1e-12)


def test_equal_weight_permutation_preserves_marginals_but_changes_coupling():
    values = np.asarray([[0., 0.], [1., 1.], [2., 4.], [3., 9.]])
    weights = np.full(4, 0.25)
    permuted = MODULE.permute_equal_weight_frames(values, weights, seed=4)
    np.testing.assert_allclose(np.sort(permuted, axis=0), np.sort(values, axis=0))
    assert not np.allclose(MODULE.weighted_covariance(permuted, weights), MODULE.weighted_covariance(values, weights))
    times = np.asarray([0.1, 1., 10.])
    original_mean = MODULE.framewise_uptake(values, times).mean(axis=1)
    permuted_mean = MODULE.framewise_uptake(permuted, times).mean(axis=1)
    np.testing.assert_allclose(original_mean, permuted_mean)


def test_curve_covariance_rank_is_bounded_by_observation_count():
    curve = np.arange(75, dtype=float).reshape(5, 15)/100
    matrices = MODULE.curve_covariances(curve, MODULE.TIMEPOINTS)
    assert np.linalg.matrix_rank(matrices["curve_raw_uptake"], tol=1e-10) <= 4
    assert np.linalg.matrix_rank(matrices["curve_adjacent_survival_slope"], tol=1e-10) <= 3


def test_mapping_congruence_and_projected_distance_at_truth():
    values = np.asarray([[0., 1., 2.], [2., 3., 1.], [4., 0., 2.], [1., 2., 5.]])
    weights = np.full(4, 0.25)
    mapping = np.asarray([[0.5, 0.5, 0.], [0., 0.25, 0.75]])
    covariance = MODULE.weighted_covariance(values, weights)
    mapped = mapping@covariance@mapping.T
    np.testing.assert_allclose(mapped, MODULE.weighted_covariance(values@mapping.T, weights))
    projection = MODULE.overlap_projection(mapping)
    assert MODULE.log_euclidean_distance(mapped, mapped, projection) < 1e-12


def test_cumulants_recover_small_time_mean_and_variance():
    rates = np.asarray([[0.2, 0.8], [0.4, 1.2], [0.7, 1.6]])
    weights = np.asarray([0.2, 0.3, 0.5])
    times = np.asarray([1, 2, 4, 8, 16])*1e-5
    survival = np.einsum("tfr,f->tr", np.exp(-times[:, None, None]*rates[None, :, :]), weights)
    mean, variance = MODULE.cumulants_from_survival(survival, times)
    np.testing.assert_allclose(mean, weights@rates, rtol=2e-5)
    np.testing.assert_allclose(variance, np.diag(MODULE.weighted_covariance(rates, weights)), rtol=3e-4)


def test_no_fitting_integration(tmp_path):
    MODULE.run(argparse.Namespace(results_dir=SCRIPT.parent/"_pf_peptide_moment_final", output_dir=tmp_path))
    manifest = __import__("json").loads((tmp_path/"manifest.json").read_text())
    metrics = pd.read_csv(tmp_path/"matrix_metrics.csv")
    assert manifest["fitting_performed"] is False
    assert set(metrics.weights) == {"uniform", "known_40_60"}
    assert set(metrics.panel) == {"equal", "random_fixed", "random_variable"}
    positive = metrics[(metrics.weights == "known_40_60") & (metrics.construction == "oracle_inverse_uptake_asymptotic")]
    assert positive.raw_relative_error.max() < 1e-5
    negative = metrics[(metrics.weights == "known_40_60") & (metrics.construction == "control_logpf")]
    assert negative.raw_relative_error.min() > 0.9
