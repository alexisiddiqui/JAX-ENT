"""
JIT-compatible simulation module for JAX-ENT.

This module provides a JIT-compatible implementation of the simulation forward pass,
separating state management from pure computation to enable JIT compilation.
"""

from typing import Any, List, Optional, Sequence

import jax
import jax.numpy as jnp
import optax

from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.types.base import ForwardModel, ForwardPass
from jaxent.types.features import Input_Features, Output_Features
from jaxent.utils.jax_fn import frame_average_features, single_pass


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
        avg_feature = frame_average_features(feature_data, params.frame_weights)
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

        print("JIT Simulation initialised successfully.")
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


# Testing function to validate the module
def _test_module():
    """Simple validation test to ensure the module is correctly implemented."""
    # Create dummy data for testing
    from jaxent.interfaces.simulation import Simulation_Parameters

    # Create a dummy parameter object
    params = Simulation_Parameters(
        frame_weights=jnp.ones(10) / 10,
        frame_mask=jnp.ones(10),
        model_parameters=[{}],  # Dummy object
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    # Create a dummy input feature
    class DummyFeature:
        def __init__(self):
            self.features = jnp.ones((10, 5))
            self.features_shape = (10, 5)

    # Create a dummy forward pass
    class DummyForwardPass:
        def __call__(self, features, params):
            return features * 2

    # Create a dummy forward model
    class DummyModel:
        def __init__(self):
            self.forwardpass = DummyForwardPass()

    # Test forward_pure function
    input_features = [DummyFeature()]
    forward_models = [DummyModel()]
    model_parameters = [{}]

    try:
        result = forward_pure(
            params, input_features, [forward_models[0].forwardpass], model_parameters
        )
        print("forward_pure function test: PASSED")
    except Exception as e:
        print(f"forward_pure function test: FAILED - {str(e)}")


if __name__ == "__main__":
    # Run a simple test when this module is executed directly
    _test_module()
