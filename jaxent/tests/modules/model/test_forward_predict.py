#!/usr/bin/env python3
"""
Test script for Simulation object's forward and predict methods.

This script tests the core functionality of the Simulation class using
actual JAXent classes with minimal setup to avoid mocking complexities.

Final fixed version that passes all tests.
"""

import os
import sys
import tempfile
from dataclasses import dataclass
import dataclasses
from typing import ClassVar, Optional

from utils.jit_fn import jit_Guard

# JAX setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config

# Import JAXent modules
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_model


def create_minimal_topology_file():
    """Create a minimal PDB file for testing."""
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  10.000  10.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      11.000  11.000  10.000  1.00 20.00           C
ATOM      6  N   GLY A   2      12.000  11.000  10.000  1.00 20.00           N
ATOM      7  CA  GLY A   2      13.000  11.000  10.000  1.00 20.00           C
ATOM      8  C   GLY A   2      14.000  11.000  10.000  1.00 20.00           C
ATOM      9  O   GLY A   2      15.000  11.000  10.000  1.00 20.00           O
END
"""
    fd, path = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.write(pdb_content)
    return path


def create_minimal_trajectory():
    """Create minimal trajectory coordinates."""
    # Simple 2-frame trajectory with 9 atoms
    coords = np.array(
        [
            # Frame 1
            [
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [12.0, 10.0, 10.0],
                [13.0, 10.0, 10.0],
                [11.0, 11.0, 10.0],
                [12.0, 11.0, 10.0],
                [13.0, 11.0, 10.0],
                [14.0, 11.0, 10.0],
                [15.0, 11.0, 10.0],
            ],
            # Frame 2
            [
                [10.1, 10.1, 10.1],
                [11.1, 10.1, 10.1],
                [12.1, 10.1, 10.1],
                [13.1, 10.1, 10.1],
                [11.1, 11.1, 10.1],
                [12.1, 11.1, 10.1],
                [13.1, 11.1, 10.1],
                [14.1, 11.1, 10.1],
                [15.1, 11.1, 10.1],
            ],
        ]
    )
    return coords


@dataclass
class FixedBVFeatures:
    """
    Final corrected version of BV Features mock object. It's a dataclass
    with an intelligent __post_init__ to handle reconstruction when only
    feature arrays are provided.
    """

    __features__: ClassVar = ("heavy_contacts", "acceptor_contacts")

    heavy_contacts: jnp.ndarray
    acceptor_contacts: jnp.ndarray
    k_ints: Optional[jnp.ndarray] = None
    features_shape: Optional[tuple[int, ...]] = None

    def __post_init__(self):
        """
        This method is called by the dataclass after __init__. It sets
        default or derived values for fields that might not have been
        provided, which handles reconstruction in `frame_average_features`.
        """
        if self.features_shape is None:
            self.features_shape = self.heavy_contacts.shape
        if self.k_ints is None:
            n_features = self.features_shape[0]
            self.k_ints = jnp.ones(n_features) * 0.1

    def _get_ordered_slots(self):
        """Return dataclass fields in order."""
        return [f.name for f in dataclasses.fields(self)]

    def cast_to_jax(self):
        return self

    def tree_flatten(self):
        children = (self.heavy_contacts, self.acceptor_contacts)
        aux_data = (self.k_ints, self.features_shape)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        heavy, acceptor = children
        k_ints, features_shape = aux_data
        return cls(heavy, acceptor, k_ints, features_shape)


register_pytree_node(FixedBVFeatures, FixedBVFeatures.tree_flatten, FixedBVFeatures.tree_unflatten)


def create_test_simulation_fixed():
    """Create a test simulation with fixed feature objects."""
    bv_config = BV_model_Config()
    model = BV_model(bv_config)
    n_features, n_frames = 50, 10

    # Create feature values that VARY across frames.
    heavy_contacts_frames = jnp.linspace(0.5, 1.0, n_frames)
    acceptor_contacts_frames = jnp.linspace(0.4, 0.8, n_frames)

    features = [
        FixedBVFeatures(
            heavy_contacts=jnp.ones((n_features, n_frames)) * heavy_contacts_frames,
            acceptor_contacts=jnp.ones((n_features, n_frames)) * acceptor_contacts_frames,
            k_ints=jnp.ones(n_features) * 0.1,
            features_shape=(n_features, n_frames),
        )
    ]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames, dtype=jnp.bool_),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.bool_),
    )
    simulation = Simulation(
        forward_models=[model], input_features=features, params=params, raise_jit_failure=True
    )
    return simulation, params


def create_test_simulation_real():
    """Create a test simulation using real MDAnalysis universe."""
    try:
        import MDAnalysis as mda

        pdb_path = create_minimal_topology_file()
        try:
            coords = create_minimal_trajectory()
            u = mda.Universe(pdb_path)
            u.trajectory.n_frames = len(coords)
            bv_config = BV_model_Config()
            model = BV_model(bv_config)
            n_features, n_frames = 50, len(coords)

            heavy_contacts_frames = jnp.linspace(0.5, 1.0, n_frames)
            acceptor_contacts_frames = jnp.linspace(0.4, 0.8, n_frames)

            features = [
                FixedBVFeatures(
                    heavy_contacts=jnp.ones((n_features, n_frames)) * heavy_contacts_frames,
                    acceptor_contacts=jnp.ones((n_features, n_frames)) * acceptor_contacts_frames,
                    k_ints=jnp.ones(n_features) * 0.1,
                    features_shape=(n_features, n_frames),
                )
            ]
            params = Simulation_Parameters(
                frame_weights=jnp.ones(n_frames) / n_frames,
                frame_mask=jnp.ones(n_frames, dtype=jnp.bool_),
                model_parameters=[bv_config.forward_parameters],
                forward_model_weights=jnp.ones(1),
                forward_model_scaling=jnp.ones(1),
                normalise_loss_functions=jnp.ones(1, dtype=jnp.bool_),
            )
            simulation = Simulation(
                forward_models=[model],
                input_features=features,
                params=params,
                raise_jit_failure=True,
            )
            return simulation, params
        finally:
            os.unlink(pdb_path)
    except ImportError:
        return create_test_simulation_fixed()


class TestSimulationMethods:
    """Test class for Simulation forward and predict methods."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        try:
            cls.simulation, cls.params = create_test_simulation_real()
            print("Using real-ish simulation setup")
        except Exception as e:
            print(f"Failed to create real simulation, using fixed: {e}")
            cls.simulation, cls.params = create_test_simulation_fixed()

    def test_simulation_initialization(self):
        """Test that Simulation object initializes correctly."""
        assert self.simulation.input_features is not None
        assert self.simulation.forward_models is not None
        assert self.simulation.params is not None
        assert self.simulation.forwardpass and self.simulation.forwardpass[0]
        print("✓ Simulation initialization test passed")

    def test_simulation_initialise_method(self):
        """Test the initialise method."""
        result = self.simulation.initialise()
        assert result is True
        assert hasattr(self.simulation, "_input_features")
        assert hasattr(self.simulation, "_jit_forward_pure")
        print("✓ Simulation initialise method test passed")

    def test_forward_method(self):
        """Test the forward method functionality."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        self.simulation.forward(self.params)
        assert hasattr(self.simulation, "outputs")
        assert len(self.simulation.outputs) > 0
        assert self.simulation.params == self.params
        print("✓ Forward method test passed")

    def test_predict_method_with_simulation_parameters(self):
        """Test predict method with Simulation_Parameters input."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        predictions = self.simulation.predict(self.params)
        assert isinstance(predictions, (list, tuple))
        assert len(predictions) > 0
        assert predictions[0].y_pred().shape == self.simulation.input_features[0].features_shape
        print("✓ Predict method with Simulation_Parameters test passed")

    def test_predict_method_with_model_parameters(self):
        """Test predict method with sequence of Model_Parameters."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        model_params = self.params.model_parameters
        predictions = self.simulation.predict(model_params)
        assert isinstance(predictions, (list, tuple))
        assert len(predictions) > 0
        assert predictions[0].y_pred().shape == self.simulation.input_features[0].features_shape
        print("✓ Predict method with Model_Parameters test passed")

    def test_forward_vs_predict_differences(self):
        """Test the key differences between forward and predict methods."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        self.simulation.forward(self.params)
        forward_outputs = self.simulation.outputs
        expected_forward_shape = (self.simulation.input_features[0].features_shape[0],)
        assert forward_outputs[0].y_pred().shape == expected_forward_shape
        predict_outputs = self.simulation.predict(self.params)
        expected_predict_shape = self.simulation.input_features[0].features_shape
        assert predict_outputs[0].y_pred().shape == expected_predict_shape
        assert forward_outputs is not None
        assert predict_outputs is not None
        assert len(forward_outputs) == len(predict_outputs)
        assert self.simulation.outputs is not None
        print("✓ Forward vs Predict differences test passed")

    def test_predict_parameter_validation(self):
        """Test parameter validation in predict method."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        if len(self.params.model_parameters) > 1:
            wrong_params = self.params.model_parameters[:1]
            try:
                self.simulation.predict(wrong_params)
                assert False, "Should have raised ValueError"
            except (ValueError, IndexError, TypeError):
                print("✓ Parameter validation correctly caught mismatch")
        else:
            print("✓ Parameter validation test skipped (only 1 model)")

    def test_predict_without_initialization(self):
        """Test predict method behavior when simulation not initialized."""
        fresh_sim, fresh_params = create_test_simulation_fixed()
        try:
            fresh_sim.predict(fresh_params)
            assert False, "Should have raised RuntimeError"
        except (RuntimeError, AttributeError):
            print("✓ Predict method correctly requires initialization")

    @jit_Guard.test_isolation()
    def test_parameter_updates(self):
        """Test that parameters are properly updated between calls."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        n_frames = len(self.params.frame_weights)

        # FIX: Create non-uniform weights. The `forward` method normalizes
        # weights, so a simple scalar multiplication won't change the outcome.
        # We must change the weight *distribution*.
        modified_weights = jnp.ones(n_frames)
        if n_frames > 1:
            modified_weights = modified_weights.at[0].set(5.0)  # Skew the weights

        modified_params = self.params.__replace__(frame_weights=modified_weights)

        # Run with original (uniform) params
        self.simulation.forward(self.params)
        original_output = self.simulation.outputs[0].y_pred()

        # Run with modified (non-uniform) params
        self.simulation.forward(modified_params)
        modified_output = self.simulation.outputs[0].y_pred()

        # The internal state of simulation.params should be updated
        # Note: We compare against the normalized weights that the model actually uses.
        normalized_modified_weights = modified_params.frame_weights / jnp.sum(
            modified_params.frame_weights
        )
        # FIX: Compare to normalized weights, not raw weights
        sim_normalized_weights = self.simulation.params.frame_weights / jnp.sum(
            self.simulation.params.frame_weights
        )
        assert jnp.allclose(sim_normalized_weights, normalized_modified_weights)

        # Because the weight distribution changed, the outputs should be different.
        assert not jnp.allclose(original_output, modified_output)
        print("✓ Parameter updates test passed")

    def test_frame_weights_effect(self):
        """Test that different frame weights produce different results."""
        if not hasattr(self.simulation, "_input_features"):
            self.simulation.initialise()
        n_frames = len(self.params.frame_weights)
        uniform_weights = jnp.ones(n_frames) / n_frames
        uniform_params = self.params.__replace__(frame_weights=uniform_weights)
        self.simulation.forward(uniform_params)
        uniform_results = self.simulation.outputs[0].y_pred()
        if n_frames > 1:
            biased_weights = jnp.zeros(n_frames).at[0].set(1.0)
            biased_params = self.params.__replace__(frame_weights=biased_weights)
            self.simulation.forward(biased_params)
            biased_results = self.simulation.outputs[0].y_pred()
            assert not jnp.allclose(uniform_results, biased_results)
        print("✓ Frame weights effect test completed")


def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("COMPREHENSIVE SIMULATION FORWARD/PREDICT METHOD TESTS")
    print("(Final fixed version)")
    print("=" * 60)
    test_instance = TestSimulationMethods()
    TestSimulationMethods.setup_class()
    test_methods = [
        test_instance.test_simulation_initialization,
        test_instance.test_simulation_initialise_method,
        test_instance.test_forward_method,
        test_instance.test_predict_method_with_simulation_parameters,
        test_instance.test_predict_method_with_model_parameters,
        test_instance.test_forward_vs_predict_differences,
        test_instance.test_predict_parameter_validation,
        test_instance.test_predict_without_initialization,
        test_instance.test_parameter_updates,
        test_instance.test_frame_weights_effect,
    ]
    passed_tests, failed_tests = 0, 0
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method.__name__}...")
            test_method()
            passed_tests += 1
        except Exception as e:
            print(f"✗ {test_method.__name__} failed: {str(e)}")
            import traceback

            traceback.print_exc()
            failed_tests += 1
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(test_methods)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    if failed_tests == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed")
    return failed_tests == 0


def demonstrate_usage():
    """Demonstrate the expected usage patterns of forward and predict methods."""
    print("\n" + "=" * 60)
    print("USAGE DEMONSTRATION")
    print("=" * 60)
    simulation, params = create_test_simulation_fixed()
    simulation.initialise()
    print("\n1. Using forward() method:")
    print("   - Applies forward models to FRAME-AVERAGED features.")
    print("   - Stores a single, averaged result per model in `simulation.outputs`.")
    print("   - This is typically used during an optimization loop.")
    simulation.forward(params)
    output = simulation.outputs[0].y_pred()
    print(
        f"   - Generated {len(simulation.outputs)} output(s). Shape of first output: {output.shape}"
    )
    print("\n2. Using predict() method:")
    print("   - Applies forward models to FRAME-WISE features without averaging.")
    print("   - Returns frame-wise predictions directly.")
    print("   - This is typically used for inference or analyzing per-frame behavior.")
    predictions = simulation.predict(params)
    prediction_output = predictions[0].y_pred()
    print(
        f"   - Generated {len(predictions)} prediction(s). Shape of first prediction: {prediction_output.shape}"
    )
    print("\n3. Key differences summary:")
    print(f"   - forward() output shape: {output.shape} (averaged over frames)")
    print(f"   - predict() output shape: {prediction_output.shape} (preserves frame dimension)")
    print("   - `forward()` modifies the simulation's internal state (`simulation.outputs`).")
    print("   - `predict()` is a pure function that returns results without modifying state.")


if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        demonstrate_usage()
    sys.exit(0 if success else 1)
