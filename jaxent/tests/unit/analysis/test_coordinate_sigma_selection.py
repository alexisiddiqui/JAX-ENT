"""Scientific invariants for coordinate Sigma selection conventions."""

import numpy as np

from jaxent.examples.common.analysis.rescore_crossval_coordinate_sigma import (
    sigma_mse,
    split_precisions,
)


def test_split_precision_matches_conditional_and_marginal_covariances():
    covariance = np.array([[2.0, 0.4, 0.8], [0.4, 1.0, 0.1], [0.8, 0.1, 1.5]])
    indices = [0, 1]
    actual = split_precisions(covariance, indices)
    marginal = covariance[:2, :2]
    conditional = marginal - covariance[:2, 2:] @ np.linalg.solve(
        covariance[2:, 2:], covariance[2:, :2]
    )
    for name, matrix in [
        ("inverse_then_split", conditional),
        ("split_then_inverse", marginal),
    ]:
        expected = np.linalg.inv(matrix)
        expected *= 2 / np.trace(expected)
        np.testing.assert_allclose(actual[name], expected, atol=1e-12)
        np.testing.assert_allclose(np.trace(actual[name]), 2)
    assert not np.allclose(actual["inverse_then_split"], actual["split_then_inverse"])


def test_independent_excluded_peptides_make_selectors_identical():
    covariance = np.array([[2.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.5]])
    actual = split_precisions(covariance, [0, 1])
    np.testing.assert_allclose(
        actual["inverse_then_split"], actual["split_then_inverse"]
    )
    residual = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(
        sigma_mse(residual, np.eye(2)), 0.5 * np.mean(residual**2)
    )
