import numpy as np
from scipy.stats import wasserstein_distance

from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import (
    mean_absolute_pair_distance,
    periodic_dihedral_pair_distance,
    pf_w1_frame_profiles,
    pooled_analysis_frame_indices,
    regularized_residue_variance,
    training_circular_variance,
    training_residue_variance,
    variance_scaled_dihedral_pair_distance,
    variance_scaled_pair_distance,
)


def test_pf_w1_matches_equal_weight_scipy_wasserstein():
    z = np.array([[3.0, -2.0], [0.0, 4.0], [1.0, 1.0], [7.0, 3.0]])
    raw, _ = pf_w1_frame_profiles(z)
    observed = mean_absolute_pair_distance(raw, np.array([0]), np.array([1]))[0]
    expected = wasserstein_distance(z[:, 0], z[:, 1])
    assert np.isclose(observed, expected)


def test_pf_w1_is_permutation_invariant_and_centered_removes_uniform_shift():
    prior = np.array([1.0, 3.0, 8.0, 10.0])
    permuted = prior[[2, 0, 3, 1]]
    shifted = prior + 5.0

    raw, centered = pf_w1_frame_profiles(np.stack([prior, permuted, shifted], axis=1))
    permutation_distance = mean_absolute_pair_distance(raw, np.array([0]), np.array([1]))[0]
    centered_shift_distance = mean_absolute_pair_distance(centered, np.array([0]), np.array([2]))[0]
    raw_shift_distance = mean_absolute_pair_distance(raw, np.array([0]), np.array([2]))[0]

    assert np.isclose(permutation_distance, 0.0)
    assert np.isclose(centered_shift_distance, 0.0)
    assert np.isclose(raw_shift_distance, 5.0)


def test_training_variance_uses_only_declared_replica():
    z = np.array([[0.0, 2.0, 100.0, 200.0], [3.0, 3.0, -50.0, 50.0]])
    replicas = np.array([1, 1, 2, 3])
    observed = training_residue_variance(z, replicas, fit_replica=1)

    np.testing.assert_allclose(observed, np.array([2.0, 0.0]))


def test_regularized_variance_is_finite_for_zero_and_nonfinite_values():
    regularized, reference = regularized_residue_variance(
        np.array([0.0, 2.0, 8.0, np.nan]), shrinkage=0.01
    )

    assert np.isclose(reference, 5.0)
    assert np.all(np.isfinite(regularized))
    assert np.all(regularized > 0)
    assert np.isclose(regularized[0], 0.05)


def test_information_cost_is_symmetric_and_root_is_exact():
    z = np.array([[0.0, 2.0], [1.0, 5.0], [-1.0, -1.0]])
    variance = np.array([2.0, 4.0, 0.5])
    left = np.array([0])
    right = np.array([1])
    quadratic = variance_scaled_pair_distance(z, left, right, variance, root=False)
    reverse = variance_scaled_pair_distance(z, right, left, variance, root=False)
    root = variance_scaled_pair_distance(z, left, right, variance, root=True)

    np.testing.assert_allclose(quadratic, reverse)
    np.testing.assert_allclose(np.square(root), quadratic)
    assert quadratic[0] > 0
    assert variance_scaled_pair_distance(z, left, left, variance, root=False)[0] == 0.0


def test_periodic_dihedral_distance_wraps_across_180_degrees():
    angles = np.deg2rad(np.array([[179.0, -179.0]]))
    observed = periodic_dihedral_pair_distance(angles, np.array([0]), np.array([1]))

    assert np.isclose(observed[0], 2.0)
    reverse = periodic_dihedral_pair_distance(angles, np.array([1]), np.array([0]))
    np.testing.assert_allclose(reverse, observed)
    assert periodic_dihedral_pair_distance(angles, np.array([0]), np.array([0]))[0] == 0.0


def test_periodic_dihedral_distance_is_rms_over_all_phi_psi_angles():
    angles = np.deg2rad(np.array([[0.0, 3.0], [0.0, -4.0]]))
    observed = periodic_dihedral_pair_distance(angles, np.array([0]), np.array([1]))

    assert np.isclose(observed[0], np.sqrt((3.0**2 + 4.0**2) / 2.0))


def test_pooled_dihedral_frames_match_replica_major_post_equilibration_order():
    observed = pooled_analysis_frame_indices(
        total_frames=12,
        replicas=3,
        equilibration_ns=1.0,
        frame_interval_ns=1.0,
    )

    np.testing.assert_array_equal(observed, np.array([2, 3, 6, 7, 10, 11]))


def test_circular_training_variance_is_small_across_wrap_and_uses_a_only():
    angles = np.deg2rad(np.array([[179.0, -179.0, 0.0, 90.0]]))
    replicas = np.array([1, 1, 2, 3])
    variance = training_circular_variance(angles, replicas, fit_replica=1)

    np.testing.assert_allclose(variance, np.deg2rad(1.0) ** 2, rtol=1e-10)


def test_variance_scaled_dihedral_distance_matches_circular_zscore_rms():
    angles = np.deg2rad(np.array([[179.0, -179.0], [0.0, 4.0]]))
    variance = np.deg2rad(np.array([1.0, 2.0])) ** 2
    quadratic = variance_scaled_dihedral_pair_distance(
        angles, np.array([0]), np.array([1]), variance, root=False
    )
    root = variance_scaled_dihedral_pair_distance(
        angles, np.array([0]), np.array([1]), variance, root=True
    )

    np.testing.assert_allclose(quadratic, np.array([(2.0**2 + 2.0**2) / 2.0]))
    np.testing.assert_allclose(np.square(root), quadratic)
