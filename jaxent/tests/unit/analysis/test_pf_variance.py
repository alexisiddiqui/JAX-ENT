import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from jaxent.src.analysis.pf_variance import (
    average_after_uptake,
    average_first_uptake,
    conditional_variance_log_ratio_loss,
    conditional_variance_profile,
    conditional_variances_from_covariance,
    covariance_profiles_from_covariance,
    conditional_subset_effective_sample_size,
    curve_precision_from_target_uptake,
    framewise_uptake,
    jensen_shannon_recovery_percent,
    marginal_variance_profile,
    inverse_overlap_degree_weights,
    map_frame_log_pf_to_peptides,
    map_framewise_residue_uptake_to_peptides,
    map_residue_uptake_to_peptides,
    overlap_projection,
    peptide_overlap_similarity,
    projected_log_euclidean_covariance_loss,
    shrink_covariance,
    uptake_from_log_pf,
    uptake_log_pf_jacobian,
    weighted_population_covariance,
    weighted_variance_log_ratio_loss,
    weights_from_logits,
)
from jaxent.src.data.splitting.sparse_map import normalize_sparse_map_rows
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.HDX.forward import BV_uptake_ForwardPass
from jaxent.src.utils.jax_fn import frame_average_features


def test_weighted_population_covariance_uses_probability_weights_without_bessel_correction():
    values = jnp.asarray([[0.0, 2.0], [1.0, 5.0]])
    weights = jnp.asarray([0.25, 0.75])

    covariance = weighted_population_covariance(values, weights)

    mean = np.asarray(values) @ np.asarray(weights)
    centered = np.asarray(values) - mean[:, None]
    expected = (centered * np.asarray(weights)[None, :]) @ centered.T
    np.testing.assert_allclose(covariance, expected, rtol=1e-6, atol=1e-7)


def test_conditional_variance_of_diagonal_covariance_is_regularized_diagonal():
    covariance = jnp.diag(jnp.asarray([1.0, 4.0, 9.0]))
    regularized = shrink_covariance(covariance, alpha=0.05)

    conditional = conditional_variances_from_covariance(covariance, alpha=0.05)

    np.testing.assert_allclose(conditional, jnp.diag(regularized), rtol=1e-5, atol=1e-6)


def test_cholesky_conditional_variance_matches_explicit_inverse():
    covariance = jnp.asarray([[2.0, 0.4, 0.2], [0.4, 1.5, -0.1], [0.2, -0.1, 1.0]])
    regularized = shrink_covariance(covariance, alpha=0.05)
    expected = 1.0 / jnp.diag(jnp.linalg.inv(regularized))

    actual = conditional_variances_from_covariance(covariance, alpha=0.05)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_covariance_profiles_are_extracted_from_the_regularized_full_matrix():
    covariance = jnp.asarray([[2.0, 0.7], [0.7, 1.0]])
    regularized, marginal, precision_diagonal, conditional = (
        covariance_profiles_from_covariance(covariance, alpha=0.05)
    )
    expected_precision = jnp.linalg.inv(regularized)

    np.testing.assert_allclose(marginal, jnp.diag(regularized), rtol=1e-6)
    np.testing.assert_allclose(
        precision_diagonal, jnp.diag(expected_precision), rtol=1e-5
    )
    np.testing.assert_allclose(conditional, 1.0 / jnp.diag(expected_precision), rtol=1e-5)


def test_marginal_variance_profile_is_regularized_covariance_diagonal():
    values = jnp.asarray([[0.0, 2.0, 4.0], [1.0, 5.0, 2.0]])
    weights = jnp.asarray([0.2, 0.3, 0.5])
    covariance = weighted_population_covariance(values, weights)

    actual = marginal_variance_profile(values, weights, alpha=0.05)

    np.testing.assert_allclose(actual, jnp.diag(shrink_covariance(covariance, alpha=0.05)))


def test_log_ratio_variance_loss_is_zero_at_truth_and_swap_symmetric():
    target = jnp.asarray([0.2, 1.0, 3.0])
    predicted = jnp.asarray([0.4, 0.8, 2.0])

    assert float(conditional_variance_log_ratio_loss(target, target)) == 0.0
    np.testing.assert_allclose(
        conditional_variance_log_ratio_loss(predicted, target),
        conditional_variance_log_ratio_loss(target, predicted),
        rtol=1e-6,
    )


def test_profile_and_gradient_remain_finite_for_identical_frames():
    values = jnp.ones((4, 3))

    def loss(logits):
        weights = weights_from_logits(logits)
        profile = conditional_variance_profile(values, weights, alpha=0.05)
        return jnp.sum(profile)

    value, gradient = jax.value_and_grad(loss)(jnp.asarray([20.0, -20.0, -20.0]))

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.all(weights_from_logits(jnp.asarray([2.0, -1.0, 0.5])) > 0)
    np.testing.assert_allclose(
        jnp.sum(weights_from_logits(jnp.asarray([2.0, -1.0, 0.5]))), 1.0, atol=1e-7
    )


def test_average_first_helper_matches_bv_forward_pass():
    heavy = jnp.asarray([[1.0, 3.0], [2.0, 5.0]])
    acceptor = jnp.asarray([[0.5, 2.0], [1.5, 0.25]])
    k_ints = jnp.asarray([0.4, 0.7])
    timepoints = jnp.asarray([0.167, 1.0, 10.0])
    weights = jnp.asarray([0.25, 0.75])
    features = BV_input_features(heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints)
    averaged_features = frame_average_features(features, weights)
    parameters = BV_Model_Parameters(
        bv_bc=jnp.asarray(0.35), bv_bh=jnp.asarray(2.0), timepoints=timepoints
    )

    production = BV_uptake_ForwardPass()(averaged_features, parameters).uptake
    helper = average_first_uptake(0.35 * heavy + 2.0 * acceptor, k_ints, timepoints, weights)

    np.testing.assert_allclose(helper, production, rtol=1e-6, atol=1e-7)


def test_average_after_is_only_a_distinct_target_sensitivity():
    log_pf = jnp.asarray([[0.0, 4.0], [1.0, 2.0]])
    k_ints = jnp.asarray([0.4, 0.7])
    timepoints = jnp.asarray([1.0, 10.0])
    weights = jnp.asarray([0.5, 0.5])

    production = average_first_uptake(log_pf, k_ints, timepoints, weights)
    sensitivity_target = average_after_uptake(log_pf, k_ints, timepoints, weights)

    assert production.shape == sensitivity_target.shape == (2, 2)
    assert not np.allclose(production, sensitivity_target)


def test_uptake_log_pf_jacobian_matches_finite_difference():
    log_pf = jnp.asarray([1.0, 2.0])
    k_ints = jnp.asarray([0.2, 0.8])
    timepoints = jnp.asarray([0.167, 1.0, 10.0])
    epsilon = 1e-3
    analytic = uptake_log_pf_jacobian(log_pf, k_ints, timepoints)
    columns = []
    for residue in range(log_pf.size):
        direction = jnp.zeros_like(log_pf).at[residue].set(epsilon)
        plus = uptake_from_log_pf(log_pf + direction, k_ints, timepoints)
        minus = uptake_from_log_pf(log_pf - direction, k_ints, timepoints)
        columns.append(((plus - minus) / (2.0 * epsilon))[:, residue])
    finite_difference = jnp.stack(columns, axis=1)

    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-3, atol=2e-5)


def test_cluster_recovery_is_100_at_truth_and_zero_for_disjoint_populations():
    target = jnp.asarray([0.4, 0.6, 0.0])

    np.testing.assert_allclose(jensen_shannon_recovery_percent(target, target), 100.0)
    np.testing.assert_allclose(
        jensen_shannon_recovery_percent(jnp.asarray([0.0, 0.0, 1.0]), target),
        0.0,
        atol=1e-5,
    )


def test_conditional_subset_ess_ignores_population_mass_outside_subset():
    weights = jnp.asarray([0.1, 0.1, 0.8])
    mask = jnp.asarray([True, True, False])

    np.testing.assert_allclose(conditional_subset_effective_sample_size(weights, mask), 2.0)


def test_sparse_map_row_normalization_averages_only_represented_residues():
    mapping = sparse.bcoo_fromdense(jnp.asarray([[0.25, 0.25, 0.0], [0.0, 0.5, 0.25]]))

    normalized = normalize_sparse_map_rows(mapping).todense()

    np.testing.assert_allclose(jnp.sum(normalized, axis=1), jnp.ones(2))
    np.testing.assert_allclose(normalized, jnp.asarray([[0.5, 0.5, 0.0], [0.0, 2 / 3, 1 / 3]]))


def test_sparse_map_row_normalization_rejects_empty_peptide():
    mapping = sparse.bcoo_fromdense(jnp.asarray([[0.5, 0.5], [0.0, 0.0]]))

    with pytest.raises(ValueError, match="empty fragment row"):
        normalize_sparse_map_rows(mapping)


def test_peptide_uptake_maps_after_residue_transform():
    mapping = jnp.asarray([[0.5, 0.5]])
    log_pf = jnp.asarray([[0.0], [4.0]])
    k_ints = jnp.asarray([0.4, 0.7])
    timepoints = jnp.asarray([1.0, 10.0])
    weights = jnp.asarray([1.0])

    residue_uptake = average_first_uptake(log_pf, k_ints, timepoints, weights)
    peptide_uptake = map_residue_uptake_to_peptides(mapping, residue_uptake)

    np.testing.assert_allclose(peptide_uptake[:, 0], jnp.mean(residue_uptake, axis=1))
    mapped_log_pf = map_frame_log_pf_to_peptides(mapping, log_pf)
    wrong_order = average_first_uptake(mapped_log_pf, jnp.asarray([jnp.mean(k_ints)]), timepoints, weights)
    assert not np.allclose(peptide_uptake, wrong_order)


def test_peptide_covariance_mapping_matches_congruence_transform():
    mapping = jnp.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    values = jnp.asarray([[0.0, 1.0, 2.0], [1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])
    weights = jnp.asarray([0.2, 0.3, 0.5])

    mapped_covariance = weighted_population_covariance(
        map_frame_log_pf_to_peptides(mapping, values), weights
    )
    residue_covariance = weighted_population_covariance(values, weights)

    np.testing.assert_allclose(
        mapped_covariance, mapping @ residue_covariance @ mapping.T, rtol=1e-6, atol=1e-7
    )


def test_framewise_uptake_is_mapped_to_peptides_after_residue_transform():
    mapping = jnp.asarray([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    log_pf = jnp.asarray([[0.0, 1.0], [2.0, 3.0], [1.0, 4.0]])
    k_ints = jnp.asarray([0.2, 0.5, 0.8])
    timepoints = jnp.asarray([0.167, 10.0])

    residue = framewise_uptake(log_pf, k_ints, timepoints)
    peptide = map_framewise_residue_uptake_to_peptides(mapping, residue)

    assert residue.shape == (2, 3, 2)
    assert peptide.shape == (2, 2, 2)
    np.testing.assert_allclose(peptide, jnp.einsum("pr,trf->tpf", mapping, residue))


def test_curve_precision_is_finite_symmetric_and_trace_normalized():
    target = jnp.asarray([[0.1, 0.2, 0.4], [0.2, 0.5, 0.7], [0.4, 0.8, 0.9]])

    precision = curve_precision_from_target_uptake(target, alpha=0.05)

    assert jnp.all(jnp.isfinite(precision))
    np.testing.assert_allclose(precision, precision.T, atol=1e-6)
    np.testing.assert_allclose(jnp.trace(precision), target.shape[1], rtol=1e-5)


def test_overlap_weights_are_uniform_without_overlap_and_share_duplicate_weight():
    independent = jnp.eye(3)
    duplicated = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(peptide_overlap_similarity(independent), jnp.eye(3))
    np.testing.assert_allclose(inverse_overlap_degree_weights(independent), jnp.ones(3))
    weights = np.asarray(inverse_overlap_degree_weights(duplicated))
    np.testing.assert_allclose(weights[0], weights[1])
    np.testing.assert_allclose(weights[2], 2.0 * weights[0])
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_weighted_profile_loss_is_symmetric_and_zero_at_truth():
    target = jnp.asarray([1.0, 2.0, 4.0])
    predicted = jnp.asarray([2.0, 1.0, 8.0])
    weights = jnp.asarray([0.5, 1.0, 1.5])

    assert float(weighted_variance_log_ratio_loss(target, target, weights)) == 0.0
    np.testing.assert_allclose(
        weighted_variance_log_ratio_loss(predicted, target, weights),
        weighted_variance_log_ratio_loss(target, predicted, weights),
    )


def test_projected_covariance_loss_is_symmetric_zero_and_finite_gradient():
    mapping = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    projection = overlap_projection(mapping)
    target = jnp.asarray([[2.0, 2.0, 0.1], [2.0, 2.0, 0.1], [0.1, 0.1, 1.0]])
    predicted = target + 0.2 * jnp.eye(3)

    assert projection.shape == (3, 2)
    assert float(projected_log_euclidean_covariance_loss(target, target, projection)) < 1e-12
    np.testing.assert_allclose(
        projected_log_euclidean_covariance_loss(predicted, target, projection),
        projected_log_euclidean_covariance_loss(target, predicted, projection),
        rtol=1e-5,
    )
    gradient = jax.grad(
        lambda scale: projected_log_euclidean_covariance_loss(
            target + scale * jnp.eye(3), target, projection
        )
    )(jnp.asarray(0.2))
    assert jnp.isfinite(gradient)
