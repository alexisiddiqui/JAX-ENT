import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
)


def _params_from_weights(weights) -> Simulation_Parameters:
    return Simulation_Parameters.from_frame_weights(
        weights,
        model_parameters=[],
        forward_model_weights=jnp.asarray([]),
        normalise_loss_functions=jnp.asarray([]),
        forward_model_scaling=jnp.asarray([]),
    )


@pytest.mark.parametrize("n_frames", [5, 500, 5000])
def test_uniform_weights_round_trip_bit_exactly(n_frames):
    weights = jnp.ones(n_frames) / n_frames
    result = _params_from_weights(weights).frame_weight_simplex
    assert jnp.array_equal(result, weights)


def test_sharp_nonuniform_weights_round_trip_without_flattening():
    weights = jnp.asarray([0.97, 0.02, 0.01])
    result = _params_from_weights(weights).frame_weight_simplex
    np.testing.assert_allclose(result, weights, rtol=1e-6)
    assert not jnp.allclose(jax.nn.softmax(weights), weights)
    assert not jnp.allclose(jax.nn.softmax(weights * weights.size), weights)


def test_dirichlet_weights_round_trip():
    weights = np.random.default_rng(7).dirichlet(np.ones(500)).astype(np.float32)
    result = _params_from_weights(weights).frame_weight_simplex
    np.testing.assert_allclose(result, weights, rtol=1e-6, atol=1e-9)


def test_property_cannot_disagree_after_arithmetic_or_pytree_round_trip():
    params = _params_from_weights(jnp.asarray([0.7, 0.2, 0.1]))
    for candidate in (params * 0.5, params + params):
        expected = jax.nn.softmax(candidate.frame_weight_logits)
        assert jnp.array_equal(candidate.frame_weight_simplex, expected)
        assert jnp.isclose(candidate.frame_weight_simplex.sum(), 1.0)

        leaves, treedef = jax.tree_util.tree_flatten(candidate)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.array_equal(
            restored.frame_weight_simplex,
            jax.nn.softmax(restored.frame_weight_logits),
        )


def test_optimizer_parameter_tree_has_five_leaves_and_no_derived_label():
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(optimizer="adamw")
    state = optimizer.initialise(simulation)
    assert len(jax.tree_util.tree_leaves(state.params)) == 5
    labels = Simulation_Parameters.param_labels(state.params)
    assert "derived" not in jax.tree_util.tree_leaves(labels)


def test_nonuniform_seed_survives_optimizer_initialise():
    simulation, _ = _create_synthetic_simulation()
    weights = jnp.asarray([0.5, 0.125, 0.125, 0.125, 0.125])
    simulation.params = Simulation_Parameters.from_frame_weights(
        weights,
        model_parameters=simulation.params.model_parameters,
        forward_model_weights=simulation.params.forward_model_weights,
        normalise_loss_functions=simulation.params.normalise_loss_functions,
        forward_model_scaling=simulation.params.forward_model_scaling,
    )
    state = OptaxOptimizer().initialise(simulation)
    np.testing.assert_allclose(state.params.frame_weight_simplex, weights, rtol=1e-6)
