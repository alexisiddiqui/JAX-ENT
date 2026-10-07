import numpy as np

from jaxent.examples.ATLAS_BV.analysis.variance_graph_checkpoint29 import (
    graph_feature,
    robust_location_scale,
)


def test_robust_location_scale_is_finite_for_constant_component():
    center, scale = robust_location_scale(np.array([[1.0, 2.0], [1.0, 4.0]]))
    assert np.all(np.isfinite(center))
    assert np.all(scale > 0)


def test_variance_graph_magnitude_is_endpoint_swap_invariant():
    matrix = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]).astype(float)
    energy = np.array([0.0, 0.5, 1.0, 0.7, 1.5])
    variance = np.array([0.2, 0.3, 0.6, 0.4, 0.8])
    left = np.array([0, 1])
    right = np.array([4, 3])
    forward, _, _ = graph_feature(
        matrix, energy, variance, left, right, 2, "energy_variance", 1.0, 0.5, 0.4
    )
    reverse, _, _ = graph_feature(
        matrix, energy, variance, right, left, 2, "energy_variance", 1.0, 0.5, 0.4
    )
    np.testing.assert_allclose(forward, reverse)


def test_directed_path_imbalance_is_antisymmetric():
    matrix = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]).astype(float)
    energy = np.array([0.0, 0.5, 1.0, 0.7, 1.5])
    variance = np.array([0.2, 0.3, 0.6, 0.4, 0.8])
    left = np.array([0])
    right = np.array([4])
    _, signed_forward, _ = graph_feature(
        matrix, energy, variance, left, right, 2, "directed_basin", 1.0, 0.5, 0.4
    )
    _, signed_reverse, _ = graph_feature(
        matrix, energy, variance, right, left, 2, "directed_basin", 1.0, 0.5, 0.4
    )
    np.testing.assert_allclose(signed_forward, -signed_reverse)
