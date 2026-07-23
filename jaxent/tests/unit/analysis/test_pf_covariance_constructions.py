import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/analyze_pf_covariance_constructions.py"
)
SPEC = importlib.util.spec_from_file_location("pf_covariance_constructions_test_module", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_streaming_moments_match_population_covariance():
    values = np.asarray(
        [[0.0, 1.0, 2.0], [2.0, 4.0, 1.0], [5.0, 0.0, 3.0], [1.0, 2.0, 8.0]]
    )
    moments = MODULE.Moments(values.shape[1])
    moments.update(values[:2])
    moments.update(values[2:])

    np.testing.assert_allclose(
        moments.covariance(), MODULE.population_covariance(values), rtol=1e-12, atol=1e-12
    )


def test_covariance_across_five_timepoints_has_rank_at_most_four():
    target_curve = np.arange(75, dtype=np.float64).reshape(5, 15)
    covariance = MODULE.population_covariance(target_curve)

    assert np.linalg.matrix_rank(covariance, tol=1e-10) <= 4


def test_sparse_mapping_full_covariance_precedes_profile_extraction():
    values = np.asarray(
        [[0.0, 1.0, 2.0], [2.0, 4.0, 1.0], [5.0, 0.0, 3.0], [1.0, 2.0, 8.0]]
    )
    mapping = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    residue_covariance = MODULE.population_covariance(values)
    mapped_covariance = mapping @ residue_covariance @ mapping.T
    direct_covariance = MODULE.population_covariance(values @ mapping.T)

    np.testing.assert_allclose(mapped_covariance, direct_covariance, rtol=1e-12, atol=1e-12)
    profiles = MODULE.regularized_profiles(mapped_covariance, alpha=0.05)
    np.testing.assert_allclose(
        profiles["conditional"],
        1.0 / np.diag(np.linalg.inv(profiles["regularized"])),
        rtol=1e-12,
    )


def test_identical_matrix_comparison_has_zero_distance_and_unit_trace_ratio():
    covariance = np.asarray([[2.0, 0.4, 0.1], [0.4, 1.0, -0.2], [0.1, -0.2, 0.8]])
    metrics = MODULE.matrix_metrics(covariance, covariance)

    assert metrics["raw_relative_error"] == 0.0
    assert metrics["normalized_frobenius_distance"] == 0.0
    np.testing.assert_allclose(metrics["off_diagonal_correlation"], 1.0)
    np.testing.assert_allclose(metrics["raw_trace_ratio"], 1.0)
