import numpy as np

from jaxent.examples.ATLAS_BV.analysis.cluster_stratified_checkpoint22 import (
    pair_matrix,
    relation,
    select_hdbscan,
    select_kmeans,
)


def test_pair_matrix_is_symmetric_with_zero_diagonal():
    points = np.array([0.0, 1.0, 4.0])
    observed = pair_matrix(
        len(points), lambda left, right: np.abs(points[left] - points[right])
    )

    np.testing.assert_allclose(observed, observed.T)
    np.testing.assert_allclose(np.diag(observed), 0.0)


def test_relation_partitions_pairs_and_flags_novel_support():
    labels = np.array([0, 0, 1, 1])
    supported = np.array([True, True, True, False])
    left = np.array([0, 0, 2])
    right = np.array([1, 2, 3])
    observed = relation(labels, supported, left, right)

    np.testing.assert_array_equal(observed["within"], [True, False, True])
    np.testing.assert_array_equal(observed["between"], [False, True, False])
    np.testing.assert_array_equal(observed["common_within"], [True, False, False])
    np.testing.assert_array_equal(observed["common_between"], [False, True, False])
    np.testing.assert_array_equal(observed["novel"], [False, False, True])


def test_kmeans_selection_is_deterministic_and_recovers_two_groups():
    rng = np.random.default_rng(4)
    features = np.r_[rng.normal(-3, 0.1, (60, 3)), rng.normal(3, 0.1, (60, 3))]

    first = select_kmeans(features, "test", "system")
    second = select_kmeans(features, "test", "system")

    assert first["clusters"] == 2
    np.testing.assert_array_equal(first["labels"], second["labels"])


def test_hdbscan_accepts_small_floating_point_diagonal():
    rng = np.random.default_rng(2)
    features = np.r_[rng.normal(-2, 0.1, (60, 2)), rng.normal(2, 0.1, (60, 2))]
    distance = np.linalg.norm(features[:, None] - features[None, :], axis=2)
    np.fill_diagonal(distance, np.finfo(float).eps)

    observed = select_hdbscan(distance, "test", "system")

    assert observed["valid"]
    assert observed["clusters"] >= 2
