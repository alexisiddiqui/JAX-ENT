import numpy as np

from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import (
    local_statistics,
    pair_local_features,
)


def test_local_statistics_and_floor_are_finite():
    values = np.array([[0.0, 1.0, 2.0, 3.0]])
    neighbours = np.array([[1, 2], [0, 2], [1, 3], [1, 2]])
    mean, variance, reference = local_statistics(values, neighbours, None, 0.01)
    assert mean.shape == variance.shape == (1, 4)
    assert np.all(variance > 0)
    assert reference.shape == (1,)


def test_magnitude_is_pair_swap_invariant_and_signed_terms_are_antisymmetric():
    values = np.array([[0.0, 1.0, 3.0], [1.0, 2.0, 2.0]])
    mean = values.copy()
    variance = np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 1.0]])
    forward = pair_local_features(
        values, mean, variance, np.array([0]), np.array([2]), "l1"
    )
    reverse = pair_local_features(
        values, mean, variance, np.array([2]), np.array([0]), "l1"
    )
    for name in ("pooled", "two_sided_z", "variance_magnitude"):
        np.testing.assert_allclose(forward[name], reverse[name])
    for name in ("variance_contrast", "direct_signed", "variance_signed_magnitude"):
        np.testing.assert_allclose(forward[name], -reverse[name])
