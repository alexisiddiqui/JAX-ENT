import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest
from jaxent.src.utils.jit_fn import jit_Guard

# Set environment variable to prevent excessive memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- Mock classes and functions for testing ---

class Input_Features:
    def __init__(self, features):
        self.features = features
        self.features_shape = features.shape

    def tree_flatten(self):
        children = (self.features,)
        aux_data = (self.features_shape,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

class ForwardPass:
    def __call__(self, features, params):
        return features * params.params

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

class ForwardModel:
    def __init__(self):
        self.forwardpass = ForwardPass()

    def tree_flatten(self):
        children = (self.forwardpass,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

class Model_Parameters:
    def __init__(self, params):
        self.params = params

    def tree_flatten(self):
        children = (self.params,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

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

    def tree_flatten(self):
        children = (
            self.frame_weights,
            self.frame_mask,
            self.model_parameters,
            self.forward_model_weights,
            self.forward_model_scaling,
            self.normalise_loss_functions,
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def normalize_weights(params):
        return params

# Register all classes as PyTrees
for cls in [Input_Features, ForwardPass, ForwardModel, Model_Parameters, Simulation_Parameters]:
    jax.tree_util.register_pytree_node_class(cls)


def frame_average_features(feature, weights):
    return jnp.average(feature, axis=0, weights=weights)

def single_pass(fp, feat, param):
    return fp(feat, param)

# --- Original Simulation class (the one to be debugged) ---

class Simulation:
    def __init__(self, input_features, forward_models, params):
        self.input_features = input_features
        self.forward_models = forward_models
        self.params = params
        self.forwardpass = [model.forwardpass for model in self.forward_models]
        self.outputs = None

    def initialise(self):
        # Simplified for JIT compatibility testing, avoiding complex topology operations
        return True

    def forward(self, params):
        """The method that is not JIT-compatible."""
        self.params = Simulation_Parameters.normalize_weights(params)
        masked_frame_weights = jnp.where(self.params.frame_mask < 0.5, 0, self.params.frame_weights)
        masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)
        average_features = [
            frame_average_features(feature.features, self.params.frame_weights)
            for feature in self.input_features
        ]
        output_features = [
            single_pass(fp, feat, param)
            for fp, feat, param in zip(self.forwardpass, average_features, self.params.model_parameters)
        ]
        self.outputs = output_features
        return output_features

# --- JIT-friendly version of the forward method ---

def forward_jit(params, input_features, forwardpass, model_parameters):
    """JIT-friendly version of forward without class attributes."""
    params = Simulation_Parameters.normalize_weights(params)
    masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
    masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)
    average_features = [
        frame_average_features(feature.features, params.frame_weights)
        for feature in input_features
    ]
    output_features = [
        single_pass(fp, feat, param)
        for fp, feat, param in zip(forwardpass, average_features, model_parameters)
    ]
    return output_features

# --- Pytest Test Suite ---

@pytest.fixture(scope="module")
def test_data():
    """Provides all necessary data for the tests."""
    num_frames = 5
    feature_dim = 3
    num_models = 2

    features1 = jnp.ones((num_frames, feature_dim))
    features2 = jnp.ones((num_frames, feature_dim)) * 2
    input_features = [Input_Features(features1), Input_Features(features2)]

    forward_models = [ForwardModel() for _ in range(num_models)]

    frame_weights = jnp.ones(num_frames) / num_frames
    frame_mask = jnp.ones(num_frames)
    model_parameters = [Model_Parameters(jnp.ones(feature_dim)) for _ in range(num_models)]
    
    params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=frame_mask,
        model_parameters=model_parameters,
        forward_model_weights=jnp.ones(num_models),
        forward_model_scaling=jnp.ones(num_models),
        normalise_loss_functions=jnp.ones(num_models),
    )

    simulation = Simulation(input_features, forward_models, params)
    simulation.initialise()

    return {
        "simulation": simulation,
        "params": params,
        "input_features": input_features,
        "frame_weights": frame_weights,
    }

def test_original_forward_method_runs(test_data):
    """Tests that the original forward method runs without errors."""
    simulation = test_data["simulation"]
    params = test_data["params"]
    try:
        outputs = simulation.forward(params)
        assert outputs is not None
        assert len(outputs) == 2
    except Exception as e:
        pytest.fail(f"Original forward method failed: {e}")



def test_jit_friendly_version_runs(test_data):
    """Tests that the JIT-friendly version runs without errors."""
    simulation = test_data["simulation"]
    params = test_data["params"]
    input_features = test_data["input_features"]
    
    forward_fn = partial(
        forward_jit,
        input_features=input_features,
        forwardpass=simulation.forwardpass,
        model_parameters=params.model_parameters,
    )
    
    try:
        outputs = forward_fn(params)
        assert outputs is not None
        assert len(outputs) == 2
    except Exception as e:
        pytest.fail(f"JIT-friendly version failed: {e}")

def test_jit_friendly_version_is_faster(test_data):
    """Compares the performance of the JIT-compiled vs. non-JIT-compiled friendly version."""
    simulation = test_data["simulation"]
    params = test_data["params"]
    input_features = test_data["input_features"]

    forward_fn = partial(
        forward_jit,
        input_features=input_features,
        forwardpass=simulation.forwardpass,
        model_parameters=params.model_parameters,
    )

    # Time non-JIT version
    start_time = time.time()
    for _ in range(100):
        _ = forward_fn(params)
    elapsed_no_jit = time.time() - start_time

    # Time JIT version
    jitted_fn = jax.jit(forward_fn)
    _ = jitted_fn(params)  # Warmup
    start_time = time.time()
    for _ in range(100):
        _ = jitted_fn(params)
    elapsed_jit = time.time() - start_time

    print(f"JIT-friendly (no JIT): {elapsed_no_jit:.4f}s | JIT-friendly (with JIT): {elapsed_jit:.4f}s")
    print(f"Speedup: {elapsed_no_jit / elapsed_jit:.2f}x")
    assert elapsed_jit < elapsed_no_jit

def test_treemap_jit_compatibility(test_data):
    """Tests if the tree_map operation with frame_average_features is JIT-compatible."""
    input_features = test_data["input_features"]
    frame_weights = test_data["frame_weights"]

    def test_fn(features, weights):
        return jax.tree_util.tree_map(
            lambda feature: frame_average_features(feature, weights),
            features,
        )

    try:
        jitted_fn = jax.jit(test_fn)
        _ = jitted_fn(input_features, frame_weights)
    except Exception as e:
        pytest.fail(f"tree_map with frame_average_features is not JIT-compatible: {e}")

@jit_Guard.test_isolation()
def test_pytree_compatibility(test_data):
    """Analyzes if the main Python objects are valid JAX pytrees."""
    simulation = test_data["simulation"]
    params = test_data["params"]
    input_features = test_data["input_features"]

    for name, obj in [
        ("input_features", input_features),
        ("forwardpass", simulation.forwardpass),
        ("model_parameters", params.model_parameters),
        ("params", params),
    ]:
        try:
            jax.tree_util.tree_structure(obj)
        except Exception as e:
            pytest.fail(f"Object '{name}' is not a valid JAX pytree: {e}")
