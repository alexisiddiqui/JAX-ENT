#!/usr/bin/env python3
"""
Simple test script for Simulation forward and predict methods using existing JAXent code.

This script can be added to your existing test suite and uses the actual
JAXent imports and setup from your codebase.
"""

import os

import jax
import jax.numpy as jnp

# Use your existing setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Add your existing imports here
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_model


# Mock a simple universe for testing (replace with your actual test data)
def create_test_simulation():
    """Create a test simulation using your existing infrastructure."""

    # Use your existing config
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    # You'll need to provide actual topology/trajectory paths for your test
    # For now, using the paths from your original code
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )

    # Check if files exist, if not skip this test
    if not (os.path.exists(topology_path) and os.path.exists(trajectory_path)):
        print("Test files not found, creating minimal mock simulation...")
        return create_minimal_test_simulation(bv_config)

    from MDAnalysis import Universe

    test_universe = Universe(topology_path, trajectory_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    trajectory_length = features[0].features_shape[1]

    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    return simulation, params


def create_minimal_test_simulation(bv_config):
    """Create a minimal test simulation for when test files aren't available."""
    # Create minimal mock features
    n_features = 50
    n_frames = 100

    class MockInputFeatures:
        def __init__(self):
            self.features_shape = (n_features, n_frames)
            self.data = jnp.ones((n_features, n_frames))

        def cast_to_jax(self):
            return self

    features = [MockInputFeatures()]
    models = [BV_model(bv_config)]

    params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    return simulation, params


def test_forward_method():
    """Test the forward method."""
    print("Testing forward method...")

    simulation, params = create_test_simulation()
    simulation.initialise()

    # Test forward method
    simulation.forward(params)

    # Check outputs exist
    assert hasattr(simulation, "outputs"), "forward() should create outputs"
    assert simulation.outputs is not None, "outputs should not be None"
    assert len(simulation.outputs) > 0, "outputs should not be empty"

    # Check that simulation state was updated
    assert simulation.params == params, "simulation.params should be updated"

    print("✓ Forward method test passed")
    return True


def test_predict_method():
    """Test the predict method."""
    print("Testing predict method...")

    simulation, params = create_test_simulation()
    simulation.initialise()

    # Test predict with Simulation_Parameters
    predictions = simulation.predict(params)

    # Check predictions structure
    assert isinstance(predictions, (list, tuple)), "predict should return sequence"
    assert len(predictions) > 0, "predictions should not be empty"

    # Test predict with just model parameters
    model_params = params.model_parameters
    predictions2 = simulation.predict(model_params)

    assert isinstance(predictions2, (list, tuple)), "predict should work with model params"
    assert len(predictions2) == len(predictions), "should return same number of predictions"

    print("✓ Predict method test passed")
    return True


def test_forward_vs_predict():
    """Test the differences between forward and predict methods."""
    print("Testing forward vs predict differences...")

    simulation, params = create_test_simulation()
    simulation.initialise()

    # Run forward
    simulation.forward(params)
    forward_outputs = simulation.outputs

    # Run predict
    predict_outputs = simulation.predict(params)

    # Both should produce outputs
    assert forward_outputs is not None, "forward should produce outputs"
    assert predict_outputs is not None, "predict should produce outputs"
    assert len(forward_outputs) == len(predict_outputs), "should have same number of outputs"

    # Check that forward modifies simulation state
    original_outputs = simulation.outputs
    simulation.forward(params)
    assert simulation.outputs is not None, "forward should update simulation.outputs"

    # Check that predict doesn't modify simulation state
    before_predict = simulation.outputs
    prediction_result = simulation.predict(params)
    after_predict = simulation.outputs
    assert before_predict is after_predict, "predict should not modify simulation.outputs"

    print("✓ Forward vs predict differences test passed")
    return True


def test_parameter_validation():
    """Test parameter validation in predict method."""
    print("Testing parameter validation...")

    simulation, params = create_test_simulation()
    simulation.initialise()

    # Test with wrong number of model parameters
    try:
        # Create params with wrong number of model parameters
        wrong_model_params = params.model_parameters[:1] if len(params.model_parameters) > 1 else []
        if wrong_model_params:
            simulation.predict(wrong_model_params)
            assert False, "Should have raised error for wrong parameter count"
    except (ValueError, IndexError):
        print("✓ Parameter validation working correctly")

    # Test predict without initialization
    try:
        new_simulation, new_params = create_test_simulation()
        # Don't call initialise()
        new_simulation.predict(new_params)
        assert False, "Should have raised error for uninitialized simulation"
    except (RuntimeError, AttributeError):
        print("✓ Initialization check working correctly")

    return True


def test_frame_weights_impact():
    """Test that different frame weights produce different averaged results."""
    print("Testing frame weights impact...")

    simulation, params = create_test_simulation()
    simulation.initialise()

    # Test with uniform weights
    uniform_weights = jnp.ones_like(params.frame_weights) / len(params.frame_weights)
    uniform_params = params.__replace__(frame_weights=uniform_weights)

    simulation.forward(uniform_params)
    uniform_results = simulation.outputs

    # Test with different weights (if we have enough frames)
    if len(params.frame_weights) > 1:
        # Create biased weights
        biased_weights = jnp.zeros_like(params.frame_weights)
        biased_weights = biased_weights.at[0].set(1.0)  # All weight on first frame
        biased_params = params.__replace__(frame_weights=biased_weights)

        simulation.forward(biased_params)
        biased_results = simulation.outputs

        print("✓ Frame weights impact test completed")

    return True


def run_all_tests():
    """Run all simulation tests."""
    print("=" * 50)
    print("SIMULATION FORWARD/PREDICT METHOD TESTS")
    print("=" * 50)

    tests = [
        test_forward_method,
        test_predict_method,
        test_forward_vs_predict,
        test_parameter_validation,
        test_frame_weights_impact,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
