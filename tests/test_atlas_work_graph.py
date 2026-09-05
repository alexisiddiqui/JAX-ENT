import numpy as np

from jaxent.examples.ATLAS_BV.analysis.work_graph_checkpoint27 import (
    edge_metric,
    path_candidate,
)


def test_edge_metric_matches_scalar_and_vector_work_definitions():
    left = np.array([0, 1])
    right = np.array([1, 2])
    np.testing.assert_allclose(edge_metric(np.array([0.0, 2.0, 5.0]), left, right), [2.0, 3.0])
    values = np.array([[0.0, 2.0, 4.0], [1.0, 4.0, 8.0]])
    np.testing.assert_allclose(edge_metric(values, left, right), [2.5, 3.0])


def test_accumulated_work_path_on_line():
    positions = np.arange(4, dtype=float)
    matrix = np.abs(positions[:, None] - positions[None, :])
    feature, edges = path_candidate(
        matrix,
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0]),
        np.array([3]),
        1,
        "accumulated",
        None,
        1.0,
        1.0,
    )
    np.testing.assert_allclose(feature, [3.0], rtol=1e-7)
    assert edges >= 3


def test_weighted_work_path_exceeds_geometry_only():
    positions = np.arange(4, dtype=float)
    matrix = np.abs(positions[:, None] - positions[None, :])
    args = (
        matrix,
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0]),
        np.array([3]),
        1,
    )
    geometry, _ = path_candidate(*args, "geometry_only", None, 1.0, 1.0)
    weighted, _ = path_candidate(*args, "weighted_w1", 1.0, 1.0, 1.0)
    assert weighted[0] > geometry[0]
