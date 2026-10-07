"""Tests for the original weight-weighted OMC Laplacian."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.loss.base import LossRegistry
from jaxent.src.opt.loss.original_omc_laplacian import (
    build_omc_kernel,
    create_original_omc_loss,
    omc_graph_energy,
)


def _parameters(logits, model_parameters=()):
    return Simulation_Parameters(
        frame_weight_logits=logits,
        model_parameters=model_parameters,
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )


class _SimulationStub:
    outputs = ()

    def __init__(self, params):
        self.params = params

    @staticmethod
    def forward(sim, params, mutate=True):
        del params, mutate
        return sim


def _explicit_energy(weights, similarity):
    weighted = similarity * weights[:, None] * weights[None, :]
    laplacian = np.diag(weighted.sum(axis=1)) - weighted
    return weights.size**2 * weights @ laplacian @ weights


def test_primary_energy_matches_weighted_laplacian_built_with_loops():
    weights = np.array([0.07, 0.18, 0.29, 0.46])
    similarity = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            similarity[i, j] = np.exp(-abs(i - j) / 2)
    expected = _explicit_energy(weights, similarity)
    actual = omc_graph_energy(jnp.asarray(weights), jnp.asarray(similarity))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_normalised_energy_has_weight_dependent_denominator():
    similarity = np.exp(-np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]))
    ratios = []
    for weights in (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.6, 0.1, 0.1, 0.2])):
        weighted = similarity * weights[:, None] * weights[None, :]
        primary = float(omc_graph_energy(weights, similarity))
        normalised = float(omc_graph_energy(weights, similarity, normalise=True))
        np.testing.assert_allclose(normalised, primary / weighted.sum(), rtol=1e-6)
        ratios.append(normalised / primary)
    assert not np.isclose(ratios[0], ratios[1])


def test_full_degeneracy_is_documented():
    similarity = jnp.ones((4, 4))
    for weights in (
        jnp.full(4, 0.25),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jnp.array([0.5, 0.5, 0.0, 0.0]),
    ):
        np.testing.assert_allclose(
            omc_graph_energy(weights, similarity), 0.0, atol=1e-7
        )


def test_n_squared_energy_is_invariant_to_exact_frame_duplication():
    weights = np.array([0.15, 0.25, 0.60])
    distance = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]).astype(float)
    similarity = np.exp(-(distance**2) / 2)
    duplicated_weights = np.repeat(weights / 2, 2)
    duplicated_similarity = np.repeat(np.repeat(similarity, 2, axis=0), 2, axis=1)
    np.testing.assert_allclose(
        omc_graph_energy(duplicated_weights, duplicated_similarity),
        omc_graph_energy(weights, similarity),
        rtol=1e-6,
    )


def test_energy_is_equivariant_to_every_node_permutation():
    weights = np.array([0.05, 0.15, 0.3, 0.5])
    distance = np.abs(
        np.array([0.0, 0.2, 1.1, 3.0])[:, None]
        - np.array([0.0, 0.2, 1.1, 3.0])[None, :]
    )
    similarity = np.exp(-(distance**2) / 0.7)
    expected = omc_graph_energy(weights, similarity)
    for permutation in itertools.permutations(range(4)):
        p = np.asarray(permutation)
        actual = omc_graph_energy(weights[p], similarity[np.ix_(p, p)])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_bandwidth_monotonically_increases_similarity_and_energy_stays_finite():
    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    similarities = []
    for bandwidth in np.logspace(-2, 2, 9):
        kernel = build_omc_kernel(distance, bandwidth=bandwidth, metric="test")
        similarities.append(np.asarray(kernel.similarity))
        assert np.isfinite(float(omc_graph_energy(weights, kernel.similarity)))
    for left, right in zip(similarities[:-1], similarities[1:], strict=True):
        assert np.all(left <= right)


def test_float32_and_float64_agree_on_pilot_sized_input():
    x = np.linspace(0, 4, 128)
    similarity = np.exp(-np.square(x[:, None] - x[None, :]) / 0.3)
    weights = np.exp(np.sin(x))
    weights /= weights.sum()
    value32 = omc_graph_energy(
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.asarray(similarity, dtype=jnp.float32),
    )
    with jax.experimental.enable_x64():
        value64 = omc_graph_energy(
            jnp.asarray(weights, dtype=jnp.float64),
            jnp.asarray(similarity, dtype=jnp.float64),
        )
    np.testing.assert_allclose(value32, value64, rtol=1e-4)


def test_kernel_rejects_invalid_inputs():
    valid = np.array([[0.0, 1.0], [1.0, 0.0]])
    invalid = (
        np.ones((2, 3)),
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[0.0, np.inf], [np.inf, 0.0]]),
    )
    for matrix in invalid:
        with np.testing.assert_raises(ValueError):
            build_omc_kernel(matrix, bandwidth=1.0, metric="test")
    for bandwidth in (0.0, -1.0, np.nan, np.inf):
        with np.testing.assert_raises_regex(ValueError, "bandwidth"):
            build_omc_kernel(valid, bandwidth=bandwidth, metric="test")


def test_registry_round_trip_is_jittable_and_gradient_is_scoped_to_logits():
    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    kernel = build_omc_kernel(distance, bandwidth=1.0, metric="test")
    loss_fn = LossRegistry.get("original_omc_laplacian")

    def pure(logits, target):
        return loss_fn(_SimulationStub(_parameters(logits)), target, None)[0]

    logits = jnp.array([-1.0, 0.2, 0.5, 1.4])
    eager = pure(logits, kernel)
    compiled = jax.jit(pure)(logits, kernel)
    np.testing.assert_allclose(compiled, eager, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(jax.grad(pure)(logits, kernel))))
    assert "original_omc_laplacian_norm" in LossRegistry.list_losses()

    params = _parameters(logits, model_parameters=())
    gradient = jax.grad(lambda p: loss_fn(_SimulationStub(p), kernel, None)[0])(params)
    assert not gradient.model_parameters


def test_registry_loss_rejects_node_count_mismatch():
    kernel = build_omc_kernel(np.zeros((3, 3)), bandwidth=1.0, metric="test")
    with np.testing.assert_raises_regex(ValueError, "node counts"):
        create_original_omc_loss()(
            _SimulationStub(_parameters(jnp.zeros(4))), kernel, None
        )
