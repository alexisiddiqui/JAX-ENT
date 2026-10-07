"""MoPrP peptide × timepoint weighted MSE for example fits."""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxent.examples.common.loading import load_hdx_timepoints_minutes


@lru_cache(maxsize=1)
def source_surface():
    source = Path(__file__).resolve().parents[1] / "2_CrossValidation/data/_MoPrP"
    weights = np.loadtxt(source / "moprp.weights")
    uptake = np.loadtxt(source / "moprp.dexp")
    np.testing.assert_array_equal(weights[:, 0], uptake[:, 0])
    np.testing.assert_allclose(
        weights[:, 0] * 60,
        load_hdx_timepoints_minutes(source / "moprp.times"),
        rtol=0,
        atol=1e-12,
    )
    if (
        weights.shape != uptake.shape
        or not np.isfinite(weights).all()
        or np.any(weights[:, 1:] <= 0)
    ):
        raise ValueError("Require finite positive MoPrP weights aligned with uptake")
    return weights[:, 1:].T, uptake[:, 1:].T


def peptide_time_weights(data, power):
    if power not in (1, 2):
        raise ValueError("MoPrP weight power must be 1 or 2")
    inverse_sd, uptake = source_surface()
    ids = [int(d.top.fragment_index) for d in data]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or min(ids) < 0
        or max(ids) >= len(inverse_sd)
    ):
        raise ValueError("Require unique valid source peptide IDs")
    np.testing.assert_allclose(
        np.array([d.dfrac for d in data]), uptake[ids], rtol=0, atol=1e-7
    )
    return jnp.asarray(inverse_sd[ids] ** power)


def _weighted_loss(model, dataset, prediction_index, power, normalize_weights=True):
    uptake = model.outputs[prediction_index].uptake

    def score(split):
        weights = peptide_time_weights(split.data, power)
        observed = split.y_true[:, :, 0]
        predicted = split.residue_feature_ouput_mapping.todense() @ uptake.T
        if predicted.shape != observed.shape or weights.shape != observed.shape:
            raise ValueError("MoPrP predictions, uptake and weights must align exactly")
        # Existing example MSE losses include a factor of 1/2. Preserve their
        # scale relative to the unchanged MaxEnt and BV regularization weights.
        denominator = jnp.sum(weights) if normalize_weights else observed.size
        return 0.5 * jnp.sum(weights * (predicted - observed) ** 2) / denominator

    return score(dataset.train), score(dataset.val)


def hdx_uptake_inv_sd_MSE_loss(model, dataset, prediction_index):
    return _weighted_loss(model, dataset, prediction_index, 1)


def hdx_uptake_inv_variance_MSE_loss(model, dataset, prediction_index):
    return _weighted_loss(model, dataset, prediction_index, 2)


def hdx_uptake_moprp_raw_weighted_MSE_loss(model, dataset, prediction_index):
    """Use file weights directly: 0.5 * mean(w * residual²), no weight normalization."""
    return _weighted_loss(model, dataset, prediction_index, 1, normalize_weights=False)
