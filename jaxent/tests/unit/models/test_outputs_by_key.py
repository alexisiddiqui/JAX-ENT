import chex
import jax
import jax.numpy as jnp
import pytest
from dataclasses import dataclass, field

from jaxent.src.custom_types.key import m_key
from jaxent.src.models.core import Simulation
from jaxent.src.custom_types.protocols import SimulationLike

# Import mock infrastructure from test_simulation
from jaxent.tests.modules.model.test_simulation import (
    MockForwardModel,
    MockForwardModelConfig,
    MockInputFeatures,
    MockOutputFeatures,
    create_default_simulation_params,
    register_pytree_node,
)


# Create a second variant of OutputFeatures with a different key
class MockOutputFeatures2(MockOutputFeatures):
    key = m_key("mock_output_2")


register_pytree_node(
    MockOutputFeatures2,
    MockOutputFeatures2.tree_flatten,
    MockOutputFeatures2.tree_unflatten,
)


class MockForwardPass2:
    def __call__(self, input_features, parameters):
        return MockOutputFeatures2(
            output_data=jnp.full((input_features.data.shape[0], 1), 42.0)
        )


@dataclass(frozen=True)
class MockForwardModelConfig2(MockForwardModelConfig):
    key: m_key = m_key("mock_model_config_2")


class MockForwardModel2(MockForwardModel):
    def __init__(self, config):
        super().__init__(config)
        self._forwardpass = MockForwardPass2()


@pytest.fixture
def base_components():
    """Provides common base components for the tests."""
    input_features = [MockInputFeatures(data=jnp.ones((10, 5)))]
    params = create_default_simulation_params(num_models=1)
    params = params.normalize_weights(params)  # Keep parameter counts matching default structure
    return input_features, params


def test_single_output_by_key(base_components):
    input_features, params = base_components
    forward_models = [MockForwardModel(MockForwardModelConfig())]
    
    sim = Simulation(input_features, forward_models, params)
    sim.initialise()
    sim = Simulation.forward(sim, params)
    
    # Check outputs_by_key
    obk = sim.outputs_by_key
    assert len(obk) == 1
    assert m_key("mock_output") in obk
    assert isinstance(obk[m_key("mock_output")], MockOutputFeatures)


def test_multiple_outputs_by_key():
    input_features = [
        MockInputFeatures(data=jnp.ones((10, 5))),
        MockInputFeatures(data=jnp.ones((10, 5))),
    ]
    # Needs 2 models, so create params for 2 models
    params = create_default_simulation_params(num_models=2)
    params = params.normalize_weights(params)
    
    forward_models = [
        MockForwardModel(MockForwardModelConfig()),
        MockForwardModel2(MockForwardModelConfig2())
    ]
    
    sim = Simulation(input_features, forward_models, params)
    sim.initialise()
    sim = Simulation.forward(sim, params)
    
    # Check outputs_by_key
    obk = sim.outputs_by_key
    assert len(obk) == 2
    assert m_key("mock_output") in obk
    assert m_key("mock_output_2") in obk
    
    assert isinstance(obk[m_key("mock_output")], MockOutputFeatures)
    assert isinstance(obk[m_key("mock_output_2")], MockOutputFeatures2)


def test_duplicate_key_raises_error():
    input_features = [
        MockInputFeatures(data=jnp.ones((10, 5))),
        MockInputFeatures(data=jnp.ones((10, 5))),
    ]
    params = create_default_simulation_params(num_models=2)
    params = params.normalize_weights(params)
    
    # Use the same model twice, so it produces outputs with the exact same m_key
    forward_models = [
        MockForwardModel(MockForwardModelConfig()),
        MockForwardModel(MockForwardModelConfig())
    ]
    
    sim = Simulation(input_features, forward_models, params)
    sim.initialise()
    sim = Simulation.forward(sim, params)
    
    # Extracting outputs_by_key should raise KeyError due to duplicate key
    with pytest.raises(KeyError, match="Duplicate output key"):
        _ = sim.outputs_by_key


def test_simulation_protocol_conformance(base_components):
    input_features, params = base_components
    forward_models = [MockForwardModel(MockForwardModelConfig())]
    
    sim = Simulation(input_features, forward_models, params)
    sim.initialise()
    sim = Simulation.forward(sim, params)
    
    # Should conform to SimulationLike
    assert isinstance(sim, SimulationLike)
