"""
JIT-compatible simulation module for JAX-ENT.

This module provides a JIT-compatible implementation of the simulation forward pass,
separating state management from pure computation to enable JIT compilation.
"""

from typing import Any, List, Optional, Sequence
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import optax
import pytest

from jaxent.src.utils.jit_fn import jit_Guard

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.utils.jax_fn import frame_average_features, single_pass


def forward_pure(
    params: Simulation_Parameters,
    input_features: List[Input_Features],
    forwardpass: Sequence[ForwardPass],
    model_parameters: List[Any],
) -> List[Output_Features]:
    """
    Pure implementation of the forward method that is JIT-compatible.
    Does not modify any state and uses only JAX operations.

    Args:
        params: Simulation parameters
        input_features: List of input features
        forwardpass: List of forward pass functions
        model_parameters: List of model parameters

    Returns:
        List of output features
    """
    # Normalize weights using a pure function (ensure this doesn't modify inputs)
    params = Simulation_Parameters.normalize_weights(params)

    # Mask the frame weights using pure JAX operations
    masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
    masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

    # Process features one by one instead of using tree_map
    average_features = []
    for feature in input_features:
        # Extract features and apply frame averaging
        # We need to access the .features attribute directly
        feature_data = feature.features
        avg_feature = jnp.average(feature_data, weights=params.frame_weights, axis=0)
        average_features.append(avg_feature)

    # Process each model explicitly rather than using tree_map
    output_features = []
    for i in range(len(forwardpass)):
        fp = forwardpass[i]
        feat = average_features[i]
        param = model_parameters[i]
        # Call the forward pass function
        output = single_pass(fp, feat, param)
        output_features.append(output)

    return output_features


class JITSimulation:
    """
    A JIT-compatible version of the Simulation class that separates state
    management from computation to allow JIT compilation.
    """

    def __init__(
        self,
        input_features: List[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters] = None,
    ) -> None:
        """
        Initialize the JIT-compatible simulation.

        Args:
            input_features: List of input features
            forward_models: Sequence of forward models
            params: Optional simulation parameters
        """
        self.input_features = input_features
        self.forward_models = forward_models
        self.params = params
        self.forwardpass = [model.forwardpass for model in self.forward_models]
        self.outputs = None
        self.length = None  # Will be set during initialization

        # We'll create the JIT-compiled function during initialization
        # after we know all parameters are valid
        self._forward_jit = None

    def initialise(self) -> bool:
        """
        Initialize the simulation and prepare the JIT-compiled function.

        Returns:
            True if initialization was successful
        """
        # Validate input features
        lengths = [feature.features_shape[-1] for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
        self.length = lengths[0]

        # Validate parameters
        if self.params is None:
            raise ValueError("No simulation parameters were provided. Exiting.")

        # Validate model parameters
        assert len(self.forward_models) == len(self.params.model_parameters), (
            "Number of forward models must be equal to number of forward model parameters"
        )

        # Create a JIT-compiled version of the forward method
        self._forward_jit = jax.jit(
            lambda params, model_params: forward_pure(
                params, self.input_features, self.forwardpass, model_params
            )
        )
        return True

    def forward(self, params: Simulation_Parameters) -> List[Output_Features]:
        """
        JIT-compatible forward method that uses the JIT-compiled pure function.
        Updates state after the computation.

        Args:
            params: Simulation parameters

        Returns:
            List of output features
        """
        if self._forward_jit is None:
            raise RuntimeError("Simulation not initialized. Call initialise() first.")

        # Call the JIT-compiled pure function
        # We pass the model parameters as an argument to enable JIT compilation
        outputs = self._forward_jit(params, self.params.model_parameters)

        # Update state (this happens outside the JIT-compiled function)
        self.params = params
        self.outputs = outputs

        return outputs


# --- Test Suite ---

# Dummy classes for testing purposes
class DummyFeature:
    def __init__(self, num_frames=10, num_features=5):
        self.features = jnp.ones((num_frames, num_features))
        self.features_shape = (num_frames, num_features)

class DummyForwardPass(ForwardPass):
    def __call__(self, features, params):
        return features * 2

class DummyModel(ForwardModel):
    def __init__(self, config=None):
        super().__init__(config)

    @property
    def forwardpass(self):
        return DummyForwardPass()

    def initialise(self, *args, **kwargs):
        pass

    def featurise(self, *args, **kwargs):
        pass

@pytest.fixture
def simulation_params():
    """Fixture for creating dummy simulation parameters."""
    return Simulation_Parameters(
        frame_weights=jnp.ones(10) / 10,
        frame_mask=jnp.ones(10),
        model_parameters=[{}],  # Dummy object
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

@pytest.fixture
def test_data():
    """Fixture for creating dummy input features and forward models."""
    input_features = [DummyFeature()]
    mock_config = MagicMock()
    mock_config.forward_parameters = {}
    forward_models = [DummyModel(config=mock_config)]
    model_parameters = [{}]
    return input_features, forward_models, model_parameters

def test_forward_pure_runs_successfully(simulation_params, test_data):
    """Test that the pure forward function executes without errors."""
    input_features, forward_models, model_parameters = test_data
    forward_passes = [model.forwardpass for model in forward_models]

    result = forward_pure(
        simulation_params, input_features, forward_passes, model_parameters
    )
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    expected_output = jnp.ones(5) * 2
    assert jnp.allclose(result[0], expected_output)

def test_jitsimulation_initialise_success(simulation_params, test_data):
    """Test that JITSimulation initializes successfully with valid data."""
    input_features, forward_models, _ = test_data
    sim = JITSimulation(input_features, forward_models, params=simulation_params)
    assert sim.initialise() is True
    assert sim._forward_jit is not None

def test_jitsimulation_initialise_raises_error_without_params(test_data):
    """Test that JITSimulation raises a ValueError if params are not provided."""
    input_features, forward_models, _ = test_data
    sim = JITSimulation(input_features, forward_models, params=None)
    with pytest.raises(ValueError, match="No simulation parameters were provided"):
        sim.initialise()

def test_jitsimulation_forward_raises_error_before_initialise(simulation_params, test_data):
    """Test that calling forward() before initialise() raises a RuntimeError."""
    input_features, forward_models, _ = test_data
    sim = JITSimulation(input_features, forward_models, params=simulation_params)
    with pytest.raises(RuntimeError, match="Simulation not initialized"):
        sim.forward(simulation_params)

def test_jitsimulation_forward_produces_output(simulation_params, test_data):
    """Test that the forward method produces the expected output after initialization."""
    input_features, forward_models, _ = test_data
    sim = JITSimulation(input_features, forward_models, params=simulation_params)
    sim.initialise()
    outputs = sim.forward(simulation_params)

    assert outputs is not None
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    expected_output = jnp.ones(5) * 2
    assert jnp.allclose(outputs[0], expected_output)
    assert sim.outputs is not None

def test_jitsimulation_handles_mismatched_models_and_params(test_data):
    """Test assertion error for mismatched forward models and model parameters."""
    input_features, forward_models, _ = test_data
    mismatched_params = Simulation_Parameters(
        frame_weights=jnp.ones(10) / 10,
        frame_mask=jnp.ones(10),
        model_parameters=[{}, {}],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )
    sim = JITSimulation(input_features, forward_models, params=mismatched_params)
    with pytest.raises(AssertionError, match="Number of forward models must be equal"):
        sim.initialise()

@jit_Guard.test_isolation()
def test_jitsimulation_handles_mismatched_feature_shapes():
    """Test assertion error for input features with different shapes."""
    input_features = [DummyFeature(num_features=5), DummyFeature(num_features=6)]
    mock_config = MagicMock()
    mock_config.forward_parameters = {}
    forward_models = [DummyModel(config=mock_config), DummyModel(config=mock_config)]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(10) / 10,
        frame_mask=jnp.ones(10),
        model_parameters=[{}, {}],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
        normalise_loss_functions=jnp.ones(2),
    )
    sim = JITSimulation(input_features, forward_models, params=params)
    with pytest.raises(AssertionError, match="Input features have different shapes"):
        sim.initialise()
