from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from jaxent.examples.common import moprp_weighted_loss as loss
from jaxent.examples.common.losses import hdx_uptake_MSE_loss


@pytest.fixture
def surface(monkeypatch):
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
    uptake = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    monkeypatch.setattr(loss, "source_surface", lambda: (weights, uptake))
    data = [
        SimpleNamespace(top=SimpleNamespace(fragment_index=i), dfrac=uptake[i])
        for i in (1, 0)
    ]
    split = SimpleNamespace(
        data=data,
        y_true=jnp.asarray(uptake[[1, 0], :, None]),
        residue_feature_ouput_mapping=BCOO.fromdense(jnp.eye(2)),
    )
    return weights[[1, 0]], split


@pytest.mark.parametrize("power", [1, 2])
def test_weighted_value_and_gradient(surface, power):
    weights, split = surface
    predicted = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    dataset = SimpleNamespace(train=split, val=split)

    def score(pred):
        model = SimpleNamespace(outputs=[SimpleNamespace(uptake=pred)])
        return loss._weighted_loss(model, dataset, 0, power)[0]

    residual = np.asarray(predicted).T - np.asarray(split.y_true[:, :, 0])
    expected = 0.5 * np.sum(weights**power * residual**2) / np.sum(weights**power)
    np.testing.assert_allclose(score(predicted), expected, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(score)(predicted), expected, rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(score)(predicted),
        (weights**power * residual / np.sum(weights**power)).T,
        rtol=1e-6,
        atol=1e-8,
    )


def test_uniform_weights_match_existing_loss(surface, monkeypatch):
    _, split = surface
    uptake = np.asarray(split.y_true[[1, 0], :, 0])
    monkeypatch.setattr(loss, "source_surface", lambda: (np.ones_like(uptake), uptake))
    model = SimpleNamespace(outputs=[SimpleNamespace(uptake=jnp.ones((3, 2)) * 0.5)])
    dataset = SimpleNamespace(train=split, val=split)
    for power in (1, 2):
        np.testing.assert_allclose(
            loss._weighted_loss(model, dataset, 0, power),
            hdx_uptake_MSE_loss(model, dataset, 0),
            rtol=1e-6,
        )


def test_bad_peptide_alignment_rejected(surface):
    _, split = surface
    split.data[0].top.fragment_index = 0
    with pytest.raises(ValueError, match="unique"):
        loss.peptide_time_weights(split.data, 1)


def test_raw_weights_value_gradient_and_scale(surface, monkeypatch):
    weights, split = surface
    predicted = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    dataset = SimpleNamespace(train=split, val=split)

    def score(pred):
        model = SimpleNamespace(outputs=[SimpleNamespace(uptake=pred)])
        return loss.hdx_uptake_moprp_raw_weighted_MSE_loss(model, dataset, 0)[0]

    residual = np.asarray(predicted).T - np.asarray(split.y_true[:, :, 0])
    expected = 0.5 * np.mean(weights * residual**2)
    np.testing.assert_allclose(jax.jit(score)(predicted), expected, rtol=1e-6)
    np.testing.assert_allclose(
        jax.grad(score)(predicted),
        (weights * residual / residual.size).T,
        rtol=1e-6,
        atol=1e-8,
    )
    original_uptake = np.asarray(split.y_true[[1, 0], :, 0])
    monkeypatch.setattr(
        loss, "source_surface", lambda: (weights[[1, 0]] * 10, original_uptake)
    )
    np.testing.assert_allclose(score(predicted), expected * 10, rtol=1e-6)


def test_raw_train_and_validation_use_observation_counts(surface):
    weights, split = surface
    train = SimpleNamespace(
        data=split.data[:1],
        y_true=split.y_true[:1],
        residue_feature_ouput_mapping=BCOO.fromdense(jnp.array([[1.0, 0.0]])),
    )
    val = SimpleNamespace(
        data=split.data[1:],
        y_true=split.y_true[1:],
        residue_feature_ouput_mapping=BCOO.fromdense(jnp.array([[0.0, 1.0]])),
    )
    predicted = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    model = SimpleNamespace(outputs=[SimpleNamespace(uptake=predicted)])
    residual = np.asarray(predicted).T - np.asarray(split.y_true[:, :, 0])
    expected = [0.5 * np.mean(w * r**2) for w, r in zip(weights, residual)]
    np.testing.assert_allclose(
        loss.hdx_uptake_moprp_raw_weighted_MSE_loss(
            model, SimpleNamespace(train=train, val=val), 0
        ),
        expected,
        rtol=1e-6,
    )


@pytest.mark.parametrize("power", [1, 2])
def test_train_and_validation_normalize_their_own_weights(surface, power):
    weights, split = surface
    train = SimpleNamespace(
        data=split.data[:1],
        y_true=split.y_true[:1],
        residue_feature_ouput_mapping=BCOO.fromdense(jnp.array([[1.0, 0.0]])),
    )
    val = SimpleNamespace(
        data=split.data[1:],
        y_true=split.y_true[1:],
        residue_feature_ouput_mapping=BCOO.fromdense(jnp.array([[0.0, 1.0]])),
    )
    predicted = jnp.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    model = SimpleNamespace(outputs=[SimpleNamespace(uptake=predicted)])
    dataset = SimpleNamespace(train=train, val=val)
    residual = np.asarray(predicted).T - np.asarray(split.y_true[:, :, 0])
    expected = [
        0.5 * np.sum(w**power * r**2) / np.sum(w**power)
        for w, r in zip(weights, residual)
    ]
    np.testing.assert_allclose(
        loss._weighted_loss(model, dataset, 0, power), expected, rtol=1e-6
    )
