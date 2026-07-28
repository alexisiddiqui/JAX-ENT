from types import SimpleNamespace

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from jaxent.src.data.covariance import ObservationCovariance
from jaxent.src.data.loader import Dataset
from jaxent.src.data.splitting.mapping import SparseFragmentMapping
from jaxent.src.opt.loss.gaussian import gaussian_joint_nll, sigma_diag_noise


def _dataset(y_true, covariance):
    mapping = SparseFragmentMapping(
        sparse_map=sparse.BCOO.fromdense(jnp.eye(y_true.shape[0]))
    )
    return SimpleNamespace(
        train=Dataset([], y_true, mapping, obs_covariance=covariance),
        val=Dataset([], y_true, mapping, obs_covariance=covariance),
    )


def _model(uptake):
    return SimpleNamespace(outputs=(SimpleNamespace(uptake=uptake),))


def test_identity_joint_nll_matches_summed_gaussian_formula():
    uptake = jnp.array([[0.2, 0.8], [0.5, 0.1]])
    y_true = jnp.array([[[0.0], [1.0]], [[0.25], [0.0]]])
    covariance = ObservationCovariance.from_stacked(jnp.eye(4), n_timepoints=2)
    train, _ = gaussian_joint_nll()(_model(uptake), _dataset(y_true, covariance), 0)
    residual = (uptake.T - y_true[..., 0]).reshape(-1)
    expected = 0.5 * jnp.sum(residual**2) + 0.5 * 4 * jnp.log(2 * jnp.pi)
    np.testing.assert_allclose(train, expected, rtol=1e-10, atol=1e-10)


def test_diagonal_and_stacked_likelihoods_agree():
    uptake = jnp.array([[0.2, 0.8], [0.5, 0.1]])
    y_true = jnp.array([[[0.0], [1.0]], [[0.25], [0.0]]])
    sigma = jnp.array([[0.5, 0.7], [0.8, 0.6]])
    diag = ObservationCovariance.from_diagonal(sigma)
    stacked = ObservationCovariance.from_stacked(jnp.diag((sigma**2).reshape(-1)), n_timepoints=2)
    dataset = _dataset(y_true, diag)
    train_diag, _ = sigma_diag_noise()(_model(uptake), dataset, 0)
    train_stacked, _ = gaussian_joint_nll()(_model(uptake), _dataset(y_true, stacked), 0)
    np.testing.assert_allclose(train_diag, train_stacked, rtol=1e-10, atol=1e-10)
