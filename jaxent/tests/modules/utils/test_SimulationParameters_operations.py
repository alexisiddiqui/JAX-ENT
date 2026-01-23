from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import register_pytree_node_class, tree_map

from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters


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
def params1(mock_model_params1):
    return Simulation_Parameters(
        frame_weights=jnp.array([0.1, 0.9]),
        frame_mask=jnp.array([1, 1]),
        model_parameters=[mock_model_params1],
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.array([0]),
        forward_model_scaling=jnp.array([1.0]),
    )


@pytest.fixture
def params2(mock_model_params2):
    return Simulation_Parameters(
        frame_weights=jnp.array([0.2, 0.8]),
        frame_mask=jnp.array([0, 1]),
        model_parameters=[mock_model_params2],
        forward_model_weights=jnp.array([0.5]),
        normalise_loss_functions=jnp.array([1]),
        forward_model_scaling=jnp.array([0.5]),
    )


def assert_pytrees_allclose(tree1, tree2):
    """Assert that two pytrees are close."""
    leaves1, treedef1 = jax.tree_util.tree_flatten(tree1)
    leaves2, treedef2 = jax.tree_util.tree_flatten(tree2)
    assert treedef1 == treedef2
    for l1, l2 in zip(leaves1, leaves2):
        assert jnp.allclose(l1, l2)


def test_addition(params1, params2):
    result = params1 + params2

    # Check all fields are added correctly
    expected = tree_map(jnp.add, params1, params2)
    assert_pytrees_allclose(result, expected)


def test_subtraction(params1, params2):
    result = params1 - params2

    expected = tree_map(jnp.subtract, params1, params2)
    assert_pytrees_allclose(result, expected)


def test_multiplication(params1, params2):
    result = params1 * params2

    expected = tree_map(jnp.multiply, params1, params2)
    assert_pytrees_allclose(result, expected)


def test_division(params1, params2):
    # To avoid division by zero in mock data, we add a small epsilon
    safe_params2 = tree_map(lambda x: x + 1e-6, params2)

    result = params1 / safe_params2

    expected = tree_map(jnp.divide, params1, safe_params2)
    assert_pytrees_allclose(result, expected)


def test_scalar_addition(params1):
    scalar = 2.0
    result = params1 + scalar

    expected = tree_map(lambda x: x + scalar, params1)
    assert_pytrees_allclose(result, expected)


def test_scalar_multiplication(params1):
    scalar = 2.0
    result = params1 * scalar

    expected = tree_map(lambda x: x * scalar, params1)
    assert_pytrees_allclose(result, expected)


def test_scalar_subtraction(params1):
    scalar = 1.0
    result = params1 - scalar

    expected = tree_map(lambda x: x - scalar, params1)
    assert_pytrees_allclose(result, expected)


def test_scalar_division(params1):
    scalar = 2.0
    result = params1 / scalar

    expected = tree_map(lambda x: x / scalar, params1)
    assert_pytrees_allclose(result, expected)


def test_reverse_addition(params1):
    scalar = 2.0
    result = scalar + params1

    expected = tree_map(lambda x: scalar + x, params1)
    assert_pytrees_allclose(result, expected)


def test_reverse_multiplication(params1):
    scalar = 2.0
    result = scalar * params1

    expected = tree_map(lambda x: scalar * x, params1)
    assert_pytrees_allclose(result, expected)
