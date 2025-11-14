from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.tree_util import register_pytree_node_class, tree_map

from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import LossComponents, OptimizationState


# Dummy Model_Parameters for testing
@register_pytree_node_class
@dataclass(frozen=True)
class MockModelParameters(Model_Parameters):
    param1: jnp.ndarray
    param2: jnp.ndarray

    def tree_flatten(self):
        return (self.param1, self.param2), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@pytest.fixture
def mock_model_params1():
    return MockModelParameters(param1=jnp.array([1.0, 2.0]), param2=jnp.array([3.0, 4.0]))


@pytest.fixture
def mock_model_params2():
    return MockModelParameters(param1=jnp.array([5.0, 6.0]), param2=jnp.array([7.0, 8.0]))


@pytest.fixture
def sim_params1(mock_model_params1):
    return Simulation_Parameters(
        frame_weights=jnp.array([0.1, 0.9]),
        frame_mask=jnp.array([1, 1]),
        model_parameters=[mock_model_params1],
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.array([0]),
        forward_model_scaling=jnp.array([1.0]),
    )


@pytest.fixture
def sim_params2(mock_model_params2):
    return Simulation_Parameters(
        frame_weights=jnp.array([0.2, 0.8]),
        frame_mask=jnp.array([0, 1]),
        model_parameters=[mock_model_params2],
        forward_model_weights=jnp.array([0.5]),
        normalise_loss_functions=jnp.array([1]),
        forward_model_scaling=jnp.array([0.5]),
    )


@pytest.fixture
def loss_components1():
    return LossComponents(
        train_losses=jnp.array([10.0]),
        val_losses=jnp.array([12.0]),
        scaled_train_losses=jnp.array([10.0]),
        scaled_val_losses=jnp.array([12.0]),
        total_train_loss=jnp.array(10.0),
        total_val_loss=jnp.array(12.0),
    )


@pytest.fixture
def loss_components2():
    return LossComponents(
        train_losses=jnp.array([5.0]),
        val_losses=jnp.array([6.0]),
        scaled_train_losses=jnp.array([2.5]),
        scaled_val_losses=jnp.array([3.0]),
        total_train_loss=jnp.array(2.5),
        total_val_loss=jnp.array(3.0),
    )


@pytest.fixture
def opt_state1():
    return optax.adam(1e-3).init(jnp.zeros(1))


@pytest.fixture
def opt_state2():
    return optax.adam(1e-4).init(jnp.ones(1))


@pytest.fixture
def state1(sim_params1, opt_state1, loss_components1):
    return OptimizationState(
        params=sim_params1,
        opt_state=opt_state1,
        step=10,
        losses=loss_components1,
        gradients=tree_map(lambda x: x * 0.1, sim_params1),
    )


@pytest.fixture
def state2(sim_params2, opt_state2, loss_components2):
    return OptimizationState(
        params=sim_params2,
        opt_state=opt_state2,
        step=20,
        losses=loss_components2,
        gradients=tree_map(lambda x: x * 0.2, sim_params2),
    )


def assert_pytrees_allclose(tree1, tree2):
    """Assert that two pytrees are close."""
    leaves1, treedef1 = jax.tree_util.tree_flatten(tree1)
    leaves2, treedef2 = jax.tree_util.tree_flatten(tree2)
    assert treedef1 == treedef2
    for l1, l2 in zip(leaves1, leaves2):
        assert jnp.allclose(l1, l2)


def test_addition(state1, state2):
    result = state1 + state2

    # Check params
    expected_params = tree_map(jnp.add, state1.params, state2.params)
    assert_pytrees_allclose(result.params, expected_params)

    # Check losses
    expected_losses = tree_map(jnp.add, state1.losses, state2.losses)
    assert_pytrees_allclose(result.losses, expected_losses)

    # Check gradients
    expected_gradients = tree_map(jnp.add, state1.gradients, state2.gradients)
    assert_pytrees_allclose(result.gradients, expected_gradients)

    # Check metadata from right operand
    assert result.step == state2.step
    assert result.opt_state == state2.opt_state


def test_subtraction(state1, state2):
    result = state1 - state2

    expected_params = tree_map(jnp.subtract, state1.params, state2.params)
    assert_pytrees_allclose(result.params, expected_params)

    expected_losses = tree_map(jnp.subtract, state1.losses, state2.losses)
    assert_pytrees_allclose(result.losses, expected_losses)

    expected_gradients = tree_map(jnp.subtract, state1.gradients, state2.gradients)
    assert_pytrees_allclose(result.gradients, expected_gradients)

    assert result.step == state2.step
    assert result.opt_state == state2.opt_state


def test_multiplication(state1, state2):
    result = state1 * state2

    expected_params = tree_map(jnp.multiply, state1.params, state2.params)
    assert_pytrees_allclose(result.params, expected_params)

    expected_losses = tree_map(jnp.multiply, state1.losses, state2.losses)
    assert_pytrees_allclose(result.losses, expected_losses)

    expected_gradients = tree_map(jnp.multiply, state1.gradients, state2.gradients)
    assert_pytrees_allclose(result.gradients, expected_gradients)

    assert result.step == state2.step
    assert result.opt_state == state2.opt_state


def test_division(state1, state2):
    result = state1 / state2

    # To avoid division by zero in mock data, we add a small epsilon
    safe_state2_params = tree_map(lambda x: x + 1e-6, state2.params)
    safe_state2_losses = tree_map(lambda x: x + 1e-6, state2.losses)
    safe_state2_gradients = tree_map(lambda x: x + 1e-6, state2.gradients)

    # Recreate state2 with safe values for division
    safe_state2 = OptimizationState(
        params=safe_state2_params,
        opt_state=state2.opt_state,
        step=state2.step,
        losses=safe_state2_losses,
        gradients=safe_state2_gradients,
    )

    result = state1 / safe_state2

    expected_params = tree_map(jnp.divide, state1.params, safe_state2.params)
    assert_pytrees_allclose(result.params, expected_params)

    expected_losses = tree_map(jnp.divide, state1.losses, safe_state2.losses)
    assert_pytrees_allclose(result.losses, expected_losses)

    expected_gradients = tree_map(jnp.divide, state1.gradients, safe_state2.gradients)
    assert_pytrees_allclose(result.gradients, expected_gradients)

    assert result.step == safe_state2.step
    assert result.opt_state == safe_state2.opt_state


def test_operations_with_none_fields(state1, state2):
    state1_no_grads = state1._replace(gradients=None)
    state2_no_losses = state2._replace(losses=None)

    # Test addition
    add_result = state1_no_grads + state2
    assert add_result.gradients is None
    assert add_result.losses is not None

    add_result2 = state1 + state2_no_losses
    assert add_result2.gradients is not None
    assert add_result2.losses is None

    # Test subtraction
    sub_result = state1_no_grads - state2
    assert sub_result.gradients is None
    sub_result2 = state1 - state2_no_losses
    assert sub_result2.losses is None

    # Test multiplication
    mul_result = state1_no_grads * state2
    assert mul_result.gradients is None
    mul_result2 = state1 * state2_no_losses
    assert mul_result2.losses is None

    # Test division
    div_result = state1_no_grads / state2
    assert div_result.gradients is None
    div_result2 = state1 / state2_no_losses
    assert div_result2.losses is None
