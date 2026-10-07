import jax.numpy as jnp
import numpy as np

from jaxent.src.data.covariance import ObservationCovariance


def test_stacked_subset_is_marginal_precision_not_precision_subblock():
    sigma = jnp.array([[2.0, 0.8], [0.8, 1.5]])
    cov = ObservationCovariance.from_stacked(sigma, n_timepoints=1)
    train, _ = cov.subset(jnp.array([0]), jnp.array([1]))
    np.testing.assert_allclose(train.covariance, sigma[:1, :1])
    assert not np.allclose(
        np.linalg.inv(np.asarray(sigma))[:1, :1], np.linalg.inv(np.asarray(sigma[:1, :1]))
    )


def test_conditional_matches_schur_complement():
    sigma = jnp.array([[2.0, 0.4], [0.4, 1.5]])
    cov = ObservationCovariance.from_stacked(sigma, n_timepoints=1)
    _, val = cov.subset(jnp.array([0]), jnp.array([1]))
    mean, conditional = val.conditional(jnp.array([0.7]))
    expected_mean = sigma[1, 0] / sigma[0, 0] * 0.7
    expected_var = sigma[1, 1] - sigma[1, 0] ** 2 / sigma[0, 0]
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(conditional.covariance, jnp.array([[expected_var]]))


def test_diagonal_constructor_stores_variances_and_logdet():
    cov = ObservationCovariance.from_diagonal(jnp.array([[2.0, 3.0]]))
    np.testing.assert_allclose(cov.sigma_diag, jnp.array([[4.0, 9.0]]))
    np.testing.assert_allclose(cov.log_det, jnp.log(36.0))


def test_diagonal_subsetting_commutes_bitwise_while_correlated_marginal_changes():
    sigma = jnp.array([[0.5, 0.7], [0.8, 0.6]])
    full = ObservationCovariance.from_diagonal(sigma)
    subset, _ = full.subset(jnp.array([1]), jnp.array([0]))
    expected = ObservationCovariance.from_diagonal(sigma[1:2])
    np.testing.assert_array_equal(subset.sigma_diag, expected.sigma_diag)
    np.testing.assert_array_equal(subset.log_det, expected.log_det)

    correlated = ObservationCovariance.from_stacked(
        jnp.array([[1.0, 0.8], [0.8, 1.0]]), n_timepoints=1
    )
    marginal, _ = correlated.subset(jnp.array([0]), jnp.array([1]))
    precision_subblock = jnp.linalg.inv(correlated.covariance)[jnp.ix_(jnp.array([0]), jnp.array([0]))]
    assert not np.array_equal(marginal.covariance, jnp.linalg.inv(precision_subblock))
