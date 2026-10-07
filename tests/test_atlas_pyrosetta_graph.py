import numpy as np

from jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 import (
    fitted_scale,
    graph_is_connected,
    knn_edges,
    path_feature,
    robust_energy_scale,
    sparse_graph,
)


def line_matrix() -> np.ndarray:
    positions = np.arange(4, dtype=float)
    return np.abs(positions[:, None] - positions[None, :])


def test_knn_union_is_connected_for_line():
    left, right = knn_edges(line_matrix(), 1)
    graph = sparse_graph(4, left, right, np.ones(len(left)))
    assert graph_is_connected(graph)


def test_total_variation_shortest_path_has_endpoint_energy_sign():
    signed, magnitude, _ = path_feature(
        line_matrix(),
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0]),
        np.array([3]),
        1,
        "total_variation",
        None,
        1.0,
    )
    np.testing.assert_allclose(magnitude, [3.0], rtol=1e-7)
    np.testing.assert_allclose(signed, [3.0], rtol=1e-7)


def test_uphill_action_uses_forward_and_reverse_paths():
    signed, magnitude, _ = path_feature(
        line_matrix(),
        np.array([0.0, 2.0, 1.0, 3.0]),
        np.array([0]),
        np.array([3]),
        1,
        "uphill_action",
        1.0,
        1.0,
    )
    assert signed[0] > 0
    assert magnitude[0] > 3.0


def test_geometry_only_does_not_use_energy_magnitude():
    args = (line_matrix(), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0]), np.array([3]), 1)
    _, first, _ = path_feature(*args, "geometry_only", None, 1.0)
    changed = args[1] * 1000
    _, second, _ = path_feature(args[0], changed, *args[2:], "geometry_only", None, 1.0)
    np.testing.assert_allclose(first, second)


def test_robust_scale_and_nonnegative_fit_are_finite():
    center, scale = robust_energy_scale(np.ones(5))
    assert center == 1.0
    assert scale > 0
    assert fitted_scale(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == 0.0
