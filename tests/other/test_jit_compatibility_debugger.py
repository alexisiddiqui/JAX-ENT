import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax

# Set environment variable to prevent excessive memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Import custom modules - uncomment these and adjust the path
# import sys
# sys.path.insert(0, "/path/to/jaxent")
# from jaxent.interfaces.simulation import Simulation_Parameters
# from jaxent.models.core import Simulation
# from jaxent.utils.jax_fn import frame_average_features, single_pass


# Mock classes for testing if you can't import
class Input_Features:
    def __init__(self, features):
        self.features = features
        self.features_shape = features.shape


class ForwardPass:
    def __call__(self, features, params):
        return features * params.params


class ForwardModel:
    def __init__(self):
        self.forwardpass = ForwardPass()


class Model_Parameters:
    def __init__(self, params):
        self.params = params


class Simulation_Parameters:
    def __init__(
        self,
        frame_weights,
        frame_mask,
        model_parameters,
        forward_model_weights,
        forward_model_scaling,
        normalise_loss_functions,
    ):
        self.frame_weights = frame_weights
        self.frame_mask = frame_mask
        self.model_parameters = model_parameters
        self.forward_model_weights = forward_model_weights
        self.forward_model_scaling = forward_model_scaling
        self.normalise_loss_functions = normalise_loss_functions

    @staticmethod
    def normalize_weights(params):
        # This would normally normalize weights, but for testing we just return params
        return params


# Mock utility functions
def frame_average_features(feature, weights):
    return jnp.average(feature, axis=0, weights=weights)


def single_pass(fp, feat, param):
    return fp(feat, param)


# Simplified Simulation class for testing
class Simulation:
    def __init__(self, input_features, forward_models, params):
        self.input_features = input_features
        self.forward_models = forward_models
        self.params = params
        self.forwardpass = [model.forwardpass for model in self.forward_models]
        self.outputs = None

    def initialise(self):
        lengths = [feature.features_shape[-1] for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes."
        self.length = lengths[0]
        assert len(self.forward_models) == len(self.params.model_parameters)
        print("Simulation initialized successfully.")
        return True

    def forward(self, params):
        """The method we want to make JIT-compatible"""
        self.params = Simulation_Parameters.normalize_weights(params)

        # Mask the frame weights
        masked_frame_weights = jnp.where(self.params.frame_mask < 0.5, 0, self.params.frame_weights)
        masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

        # First map operation using tree_map
        average_features = jax.tree_util.tree_map(
            lambda feature: frame_average_features(feature.features, self.params.frame_weights),
            self.input_features,
        )

        # Second map operation using tree_map
        output_features = jax.tree_util.tree_map(
            lambda fp, feat, param: single_pass(fp, feat, param),
            self.forwardpass,
            average_features,
            self.params.model_parameters,
        )

        self.outputs = output_features
        return output_features


# Create a JIT-friendly version of the forward method
def forward_jit(params, input_features, forwardpass, model_parameters):
    """JIT-friendly version of forward without class attributes"""
    params = Simulation_Parameters.normalize_weights(params)

    # Mask the frame weights
    masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
    masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

    # Instead of tree_map, use explicit operations
    average_features = []
    for feature in input_features:
        avg = frame_average_features(feature.features, params.frame_weights)
        average_features.append(avg)

    # Process each model separately
    output_features = []
    for i in range(len(forwardpass)):
        fp = forwardpass[i]
        feat = average_features[i]
        param = model_parameters[i]
        output = single_pass(fp, feat, param)
        output_features.append(output)

    return output_features


def trace_and_profile():
    """Test JIT compatibility and profile the forward method"""
    print("Tracing the computation graph for forward method...")

    # Create test data
    num_frames = 5
    feature_dim = 3
    num_models = 2

    # Create input features
    features1 = jnp.ones((num_frames, feature_dim))
    features2 = jnp.ones((num_frames, feature_dim)) * 2
    input_features = [Input_Features(features1), Input_Features(features2)]

    # Create forward models
    forward_models = [ForwardModel() for _ in range(num_models)]

    # Create parameters
    frame_weights = jnp.ones(num_frames) / num_frames
    frame_mask = jnp.ones(num_frames)
    model_parameters = [Model_Parameters(jnp.ones(feature_dim)) for _ in range(num_models)]
    forward_model_weights = jnp.ones(num_models)
    forward_model_scaling = jnp.ones(num_models)
    normalise_loss_functions = jnp.ones(num_models)

    params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=frame_mask,
        model_parameters=model_parameters,
        forward_model_weights=forward_model_weights,
        forward_model_scaling=forward_model_scaling,
        normalise_loss_functions=normalise_loss_functions,
    )

    # Create simulation
    simulation = Simulation(input_features, forward_models, params)
    simulation.initialise()

    # Test original forward method
    print("\n1. Testing original forward method...")
    start_time = time.time()
    try:
        for _ in range(100):  # Run multiple times for better timing
            outputs = simulation.forward(params)
        elapsed = time.time() - start_time
        print(f"✓ Original method ran successfully in {elapsed:.4f}s")
    except Exception as e:
        print(f"✗ Original method failed: {str(e)}")

    # Test JIT compilation directly (expected to fail)
    print("\n2. Attempting to JIT-compile class method directly...")
    try:
        jitted_forward = jax.jit(simulation.forward)
        _ = jitted_forward(params)
        print("✓ Direct JIT compilation succeeded (unexpected)")
    except Exception as e:
        print(f"✗ Direct JIT compilation failed: {type(e).__name__}")
        print(f"  Error message: {str(e)[:150]}...")

    # Test JIT-friendly version
    print("\n3. Testing JIT-friendly version...")
    forward_fn = partial(
        forward_jit,
        input_features=input_features,
        forwardpass=simulation.forwardpass,
        model_parameters=params.model_parameters,
    )

    try:
        # Test without JIT first
        start_time = time.time()
        for _ in range(100):
            _ = forward_fn(params)
        elapsed_no_jit = time.time() - start_time
        print(f"✓ JIT-friendly version ran in {elapsed_no_jit:.4f}s without JIT")

        # Now test with JIT
        jitted_fn = jax.jit(forward_fn)

        # Warmup
        _ = jitted_fn(params)

        # Timing
        start_time = time.time()
        for _ in range(100):
            _ = jitted_fn(params)
        elapsed_jit = time.time() - start_time
        print(f"✓ JIT-friendly version ran in {elapsed_jit:.4f}s with JIT")
        print(f"  Speedup: {elapsed_no_jit / elapsed_jit:.2f}x")
    except Exception as e:
        print(f"✗ JIT-friendly version failed: {type(e).__name__}")
        print(f"  Error message: {str(e)[:150]}...")

    # Test each component separately to identify issues
    print("\n4. Testing individual components...")

    # Test tree_map with average_features
    def test_tree_map(input_features, weights):
        return jax.tree_util.tree_map(
            lambda feature: frame_average_features(feature.features, weights),
            input_features,
        )

    try:
        jitted_tree_map = jax.jit(test_tree_map)
        _ = jitted_tree_map(input_features, frame_weights)
        print("✓ tree_map with frame_average_features is JIT-compatible")
    except Exception as e:
        print(f"✗ tree_map with frame_average_features is JIT-incompatible: {type(e).__name__}")

    # Analyze Python objects for JIT compatibility
    print("\n5. Analyzing Python object compatibility...")
    for obj_name, obj in [
        ("input_features", input_features),
        ("forwardpass", simulation.forwardpass),
        ("model_parameters", model_parameters),
        ("params", params),
    ]:
        try:
            jax.tree_util.tree_structure(obj)
            print(f"✓ {obj_name} has valid JAX tree structure")
        except Exception as e:
            print(f"✗ {obj_name} is not a valid JAX pytree: {type(e).__name__}")

    # Summarize findings
    print("\n=== JIT Compatibility Analysis ===")
    print("\nIssues in the original forward method:")
    print("1. Modifies instance attributes (self.params, self.outputs)")
    print("   Solution: Return values instead of modifying state")

    print("\n2. Uses Python objects that may not be JAX-traceable")
    print("   Solution: Use JAX arrays and registered pytrees")

    print("\n3. Uses dynamic operations via tree_map")
    print("   Solution: Use explicit loops or JAX's vmap/scan")

    print("\nRecommended rewrite for JIT compatibility:")
    print("""
def forward_jit(params, input_features, forwardpass, model_parameters):
    # 1. Don't modify class attributes
    # 2. Use only JAX operations and arrays
    # 3. Return results instead of storing them
    
    # Normalize weights (keep as pure function)
    params = normalize_weights(params)
    
    # Use JAX operations for frame weights masking
    masked_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
    masked_weights = optax.projections.projection_simplex(masked_weights)
    
    # Use explicit loops instead of tree_map
    average_features = []
    for feature in input_features:
        avg = jnp.average(feature.features, axis=0, weights=params.frame_weights)
        average_features.append(avg)
    
    # Process each model separately
    output_features = []
    for i in range(len(forwardpass)):
        output = forwardpass[i](average_features[i], model_parameters[i])
        output_features.append(output)
    
    return output_features
    """)


if __name__ == "__main__":
    trace_and_profile()
