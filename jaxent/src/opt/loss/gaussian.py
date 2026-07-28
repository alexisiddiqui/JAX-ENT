"""Physical Gaussian likelihoods for HDX observations."""

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.data.covariance import ObservationCovariance
from jaxent.src.opt.loss.base import register_loss
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping


def _residual(predictions, dataset, split: str) -> Array:
    data = getattr(dataset, split)
    y_true = jnp.asarray(data.y_true)
    if y_true.ndim == 3:
        y_true = y_true[..., 0]
    if y_true.ndim != 2:
        raise ValueError("HDX observations must have shape (fragments, timepoints[, 1])")
    mapped = jax.vmap(
        lambda uptake: apply_sparse_mapping(data.residue_feature_ouput_mapping, uptake)
    )(jnp.asarray(predictions.uptake))
    if mapped.shape != (y_true.shape[1], y_true.shape[0]):
        raise ValueError(
            f"Prediction/observation shape mismatch: predicted {mapped.shape}, "
            f"expected {(y_true.shape[1], y_true.shape[0])}"
        )
    return (mapped.T - y_true).reshape(-1)


def _check_covariance(cov: ObservationCovariance | None) -> ObservationCovariance:
    if cov is None:
        raise ValueError("Physical Gaussian losses require dataset.obs_covariance")
    if cov.trace_normalised:
        raise ValueError("Physical Gaussian losses reject trace-normalised covariance")
    return cov


def _nll(residual: Array, cov: ObservationCovariance, normalisation: str, include_logdet: bool) -> Array:
    n = residual.shape[0]
    if normalisation not in ("sum", "count_rescaled"):
        raise ValueError("normalisation must be 'sum' or 'count_rescaled'")
    if cov.kind == "diag":
        variances = cov.sigma_diag.reshape(-1)
        quadratic = jnp.sum(residual**2 / variances)
    else:
        whitened = jax.scipy.linalg.cho_solve((cov.chol, True), residual)
        quadratic = residual @ whitened
    value = 0.5 * quadratic
    if include_logdet:
        value = value + 0.5 * cov.log_det + 0.5 * n * jnp.log(2 * jnp.pi)
    if normalisation == "count_rescaled":
        value = value / n
    return value


def _make_loss(normalisation: str, include_logdet: bool, diagonal_only: bool):
    def loss(model, dataset, prediction_index: int):
        predictions = model.outputs[prediction_index]
        train_cov = _check_covariance(dataset.train.obs_covariance)
        val_cov = _check_covariance(dataset.val.obs_covariance)
        if diagonal_only and (train_cov.kind != "diag" or val_cov.kind != "diag"):
            raise ValueError("sigma_diag_noise requires diagonal ObservationCovariance")
        return (
            _nll(_residual(predictions, dataset, "train"), train_cov, normalisation, include_logdet),
            _nll(_residual(predictions, dataset, "val"), val_cov, normalisation, include_logdet),
        )
    return loss


@register_loss("gaussian_joint_nll")
def gaussian_joint_nll(normalisation: str = "sum", include_logdet: bool = True):
    """Build a summed or count-rescaled joint Gaussian negative log likelihood."""
    return _make_loss(normalisation, include_logdet, diagonal_only=False)


@register_loss("sigma_diag_noise")
def sigma_diag_noise(normalisation: str = "sum", include_logdet: bool = True):
    """Build the diagonal Gaussian specialization."""
    return _make_loss(normalisation, include_logdet, diagonal_only=True)
