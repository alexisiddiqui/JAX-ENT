import importlib.util
import sys
import argparse
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[3] / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_iso_mean_rate_uptake.py"
SPEC = importlib.util.spec_from_file_location("iso_physics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_pf_ordering_and_small_time_limit():
    z = np.asarray([[0.0, 1.0, 2.0], [1.0, 1.5, 3.0]])
    weights = np.asarray([0.2, 0.3, 0.5])
    k = np.asarray([0.7, 1.2])
    pfs = MODULE.effective_pfs(z, weights)
    assert np.all(pfs["harmonic_mean_rate"] <= pfs["geometric"])
    assert np.all(pfs["geometric"] <= pfs["arithmetic"])
    time = np.asarray([1e-9])
    exact, _ = MODULE.exact_frame_uptake(z, k, weights, time)
    harmonic = MODULE.single_rate_uptake(pfs["harmonic_mean_rate"], k, time)
    np.testing.assert_allclose(harmonic, exact, rtol=1e-7, atol=1e-15)


def test_gaussian_rate_moments_and_normalized_identity():
    rng = np.random.default_rng(4)
    covariance = np.asarray([[0.3, 0.12], [0.12, 0.2]])
    values = rng.multivariate_normal([1.0, 1.5], covariance, size=400_000)
    result = MODULE.gaussian_rate_closures(values.T, np.asarray([0.8, 1.1]), np.full(len(values), 1 / len(values)))
    assert MODULE.relative_error(result["gaussian_mean"], result["exact_mean"]) < 0.004
    assert MODULE.relative_error(result["gaussian_full_covariance"], result["exact_covariance"]) < 0.02
    quotient = result["exact_covariance"] / np.outer(result["exact_mean"], result["exact_mean"])
    np.testing.assert_allclose(quotient, np.exp(result["covariance_z"]) - 1, rtol=0.03, atol=0.003)


def test_delta_covariance_mapping_and_correlation_behavior():
    covariance = np.asarray([[0.5, 0.2, -0.1], [0.2, 0.4, 0.12], [-0.1, 0.12, 0.3]])
    delta, jacobian = MODULE.uptake_delta_covariance(np.asarray([1.0, 1.4, 0.8]), covariance, np.asarray([1.0, 0.7, 1.3]), 0.5)
    expected = jacobian[:, None] * covariance * jacobian[None, :]
    np.testing.assert_allclose(delta, expected)
    np.testing.assert_allclose(MODULE.covariance_to_correlation(delta), MODULE.covariance_to_correlation(covariance), atol=1e-12)
    mapping = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.4, 0.6]])
    np.testing.assert_allclose(mapping @ delta @ mapping.T, (mapping @ np.diag(jacobian)) @ covariance @ (mapping @ np.diag(jacobian)).T)
    assert MODULE.relative_error(MODULE.covariance_to_correlation(mapping @ delta @ mapping.T), MODULE.covariance_to_correlation(mapping @ covariance @ mapping.T)) > 1e-4


def test_exact_delta_limit_for_small_log_pf_spread():
    z = np.asarray([[1.0, 1.001, 0.999], [1.5, 1.499, 1.501]])
    weights = np.asarray([0.2, 0.3, 0.5])
    k = np.asarray([0.8, 1.2])
    result = MODULE.gaussian_rate_closures(z, k, weights)
    _, exact = MODULE.exact_frame_uptake(z, k, weights, np.asarray([0.2]))
    delta, _ = MODULE.uptake_delta_covariance(result["mean_z"], result["covariance_z"], k, 0.2)
    assert MODULE.relative_error(delta, exact[0]) < 0.003


def test_no_fitting_integration_covers_both_weights_and_three_panels(tmp_path):
    results = SCRIPT.parent / "_pf_peptide_moment_final"
    MODULE.run(argparse.Namespace(results_dir=results, output_dir=tmp_path))
    import json
    import pandas as pd
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    metrics = pd.read_csv(tmp_path / "uptake_covariance_metrics.csv")
    assert manifest["fitting_performed"] is False
    assert manifest["panel_design_performed"] is False
    assert set(metrics.weights) == {"uniform", "known_40_60"}
    assert set(metrics.panel) == {"equal", "random_fixed", "random_variable"}
    assert len(metrics) == 2 * 3 * 5
