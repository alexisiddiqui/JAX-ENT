import jax.numpy as jnp
import pytest
from jax import Array

from jaxent.src.custom_types.features import (
    AbstractFeatures,
    Input_Features,
    Output_Features,
)
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)

# Import the classes (assuming they're in a module called features)
# You'll need to adjust the import path based on your actual module structure


class TestAbstractFeatures:
    """Test the AbstractFeatures base class."""

    def create_test_features_class(self):
        """Create a test implementation of AbstractFeatures for testing."""

        class TestFeatures(AbstractFeatures):
            __slots__ = ("param1", "param2", "static_param")
            __features__ = {"param1", "param2"}

            def __init__(self, param1, param2, static_param="default"):
                self.param1 = param1
                self.param2 = param2
                self.static_param = static_param

        return TestFeatures

    def test_get_ordered_slots(self):
        """Test _get_ordered_slots method."""
        TestFeatures = self.create_test_features_class()
        slots = TestFeatures._get_ordered_slots()

        # Should include slots from the class
        assert "param1" in slots
        assert "param2" in slots
        assert "static_param" in slots

        # Should be a tuple
        assert isinstance(slots, tuple)

    def test_get_grouped_slots(self):
        """Test _get_grouped_slots method."""
        TestFeatures = self.create_test_features_class()
        dynamic_slots, static_slots = TestFeatures._get_grouped_slots()

        # Dynamic slots should be those in __features__
        assert set(dynamic_slots) == {"param1", "param2"}

        # Static slots should be the rest
        assert "static_param" in static_slots

        # Both should be tuples
        assert isinstance(dynamic_slots, tuple)
        assert isinstance(static_slots, tuple)

    def test_tree_flatten(self):
        """Test tree_flatten method."""
        TestFeatures = self.create_test_features_class()
        instance = TestFeatures(
            param1=jnp.array([1.0, 2.0]), param2=jnp.array([3.0, 4.0]), static_param="test"
        )

        arrays, static_data = instance.tree_flatten()

        # Should have arrays for dynamic parameters
        assert len(arrays) == 2
        assert all(isinstance(arr, Array) for arr in arrays)
        assert all(arr.dtype == jnp.float32 for arr in arrays)

        # Should have static data
        assert "test" in static_data

    def test_tree_unflatten(self):
        """Test tree_unflatten method."""
        TestFeatures = self.create_test_features_class()

        # Create test data
        arrays = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
        static_data = ("test",)

        # Unflatten
        instance = TestFeatures.tree_unflatten(static_data, arrays)

        # Check instance properties
        assert isinstance(instance, TestFeatures)
        assert jnp.array_equal(instance.param1, arrays[0])
        assert jnp.array_equal(instance.param2, arrays[1])
        assert instance.static_param == "test"

    def test_cast_to_jax(self):
        """Test cast_to_jax method."""
        TestFeatures = self.create_test_features_class()
        instance = TestFeatures(
            param1=[1.0, 2.0],  # Regular list
            param2=[3.0, 4.0],  # Regular list
            static_param="test",
        )

        jax_instance = instance.cast_to_jax()

        # Should return new instance with JAX arrays
        assert isinstance(jax_instance, TestFeatures)
        assert isinstance(jax_instance.param1, Array)
        assert isinstance(jax_instance.param2, Array)
        assert jax_instance.static_param == "test"

    # Save/load tests have been moved to test_features_save_load.py for comprehensive testing


class TestBVInputFeatures:
    """Test BV_input_features class."""

    def create_sample_data(self):
        """Create sample data for testing."""
        return {
            "heavy_contacts": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "acceptor_contacts": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "k_ints": [1, 2, 3],
        }

    def test_initialization_with_lists(self):
        """Test initialization with regular lists."""
        data = self.create_sample_data()
        features = BV_input_features(**data)

        assert features.heavy_contacts == data["heavy_contacts"]
        assert features.acceptor_contacts == data["acceptor_contacts"]
        assert features.k_ints == data["k_ints"]

    def test_initialization_with_arrays(self):
        """Test initialization with JAX arrays."""
        heavy_contacts = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        acceptor_contacts = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        k_ints = jnp.array([1, 2, 3])

        features = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=k_ints
        )

        assert jnp.array_equal(features.heavy_contacts, heavy_contacts)
        assert jnp.array_equal(features.acceptor_contacts, acceptor_contacts)
        assert jnp.array_equal(features.k_ints, k_ints)

    def test_initialization_without_k_ints(self):
        """Test initialization without optional k_ints parameter."""
        data = self.create_sample_data()
        del data["k_ints"]

        features = BV_input_features(**data)

        assert features.k_ints is None

    def test_features_shape_with_lists(self):
        """Test features_shape property with list inputs."""
        data = self.create_sample_data()
        features = BV_input_features(**data)

        shape = features.features_shape
        assert shape == (2, 3)  # (n_residues, n_frames)

    def test_features_shape_with_arrays(self):
        """Test features_shape property with array inputs."""
        heavy_contacts = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        acceptor_contacts = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        features = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts
        )

        shape = features.features_shape
        assert shape == (3, 2)  # (n_residues, n_frames)

    def test_features_shape_empty_data(self):
        """Test features_shape with empty data."""
        features = BV_input_features(heavy_contacts=[], acceptor_contacts=[])

        shape = features.features_shape
        assert shape == (0, 0)

    def test_features_shape_type_mismatch(self):
        """Test features_shape raises TypeError for mismatched types."""
        with pytest.raises(TypeError):
            features = BV_input_features(
                heavy_contacts=[[1.0, 2.0]],  # List
                acceptor_contacts=jnp.array([[0.1, 0.2]]),  # Array
            )
            _ = features.features_shape

    def test_feat_pred(self):
        """Test feat_pred property."""
        data = self.create_sample_data()
        features = BV_input_features(**data)

        pred = features.feat_pred

        assert len(pred) == 2
        assert isinstance(pred[0], Array)
        assert isinstance(pred[1], Array)
        assert jnp.array_equal(pred[0], jnp.asarray(data["heavy_contacts"]))
        assert jnp.array_equal(pred[1], jnp.asarray(data["acceptor_contacts"]))

    # Save/load tests have been moved to test_features_save_load.py for comprehensive testing


class TestBVOutputFeatures:
    """Test BV_output_features class."""

    def test_initialization_with_list(self):
        """Test initialization with list."""
        log_Pf = [1.0, 2.0, 3.0]
        features = BV_output_features(log_Pf=log_Pf)

        assert features.log_Pf == log_Pf
        assert features.k_ints is None

    def test_initialization_with_array(self):
        """Test initialization with JAX array."""
        log_Pf = jnp.array([1.0, 2.0, 3.0])
        k_ints = jnp.array([1, 2])

        features = BV_output_features(log_Pf=log_Pf, k_ints=k_ints)

        assert jnp.array_equal(features.log_Pf, log_Pf)
        assert jnp.array_equal(features.k_ints, k_ints)

    def test_output_shape(self):
        """Test output_shape property."""
        log_Pf = [1.0, 2.0, 3.0, 4.0]
        features = BV_output_features(log_Pf=log_Pf)

        shape = features.output_shape
        assert shape == (1, 4)  # (1, n_residues)

    def test_y_pred(self):
        """Test y_pred method."""
        log_Pf = [1.0, 2.0, 3.0]
        features = BV_output_features(log_Pf=log_Pf)

        pred = features.y_pred()

        assert isinstance(pred, Array)
        assert jnp.array_equal(pred, jnp.asarray(log_Pf))

    # Save/load tests have been moved to test_features_save_load.py for comprehensive testing


class TestUptakeBVOutputFeatures:
    """Test uptake_BV_output_features class."""

    def test_initialization_with_nested_list(self):
        """Test initialization with nested list."""
        uptake = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        features = uptake_BV_output_features(uptake=uptake)

        assert features.uptake == uptake

    def test_initialization_with_array(self):
        """Test initialization with JAX array."""
        uptake = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        features = uptake_BV_output_features(uptake=uptake)

        assert jnp.array_equal(features.uptake, uptake)

    def test_output_shape(self):
        """Test output_shape property."""
        uptake = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]  # (1, 2 peptides, 3 timepoints)
        features = uptake_BV_output_features(uptake=uptake)

        shape = features.output_shape
        assert shape == (1, 2, 3)

    def test_y_pred(self):
        """Test y_pred method."""
        uptake = [[[1.0, 2.0], [3.0, 4.0]]]
        features = uptake_BV_output_features(uptake=uptake)

        pred = features.y_pred()

        assert isinstance(pred, Array)
        assert jnp.array_equal(pred, jnp.asarray(uptake))

    # Save/load tests have been moved to test_features_save_load.py for comprehensive testing


class TestJAXTreeIntegration:
    """Test JAX tree integration for all feature classes."""

    def test_bv_input_features_tree_operations(self):
        """Test JAX tree operations for BV_input_features."""
        features = BV_input_features(
            heavy_contacts=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            acceptor_contacts=jnp.array([[0.1, 0.2], [0.3, 0.4]]),
            k_ints=jnp.array([1, 2]),
        )

        # Test tree flatten/unflatten
        arrays, static_data = features.tree_flatten()
        reconstructed = BV_input_features.tree_unflatten(static_data, arrays)

        assert isinstance(reconstructed, BV_input_features)
        assert jnp.array_equal(reconstructed.heavy_contacts, features.heavy_contacts)
        assert jnp.array_equal(reconstructed.acceptor_contacts, features.acceptor_contacts)
        assert jnp.array_equal(reconstructed.k_ints, features.k_ints)

    def test_bv_output_features_tree_operations(self):
        """Test JAX tree operations for BV_output_features."""
        features = BV_output_features(log_Pf=jnp.array([1.0, 2.0, 3.0]), k_ints=jnp.array([1, 2]))

        arrays, static_data = features.tree_flatten()
        reconstructed = BV_output_features.tree_unflatten(static_data, arrays)

        assert isinstance(reconstructed, BV_output_features)
        assert jnp.array_equal(reconstructed.log_Pf, features.log_Pf)
        assert jnp.array_equal(reconstructed.k_ints, features.k_ints)

    def test_uptake_output_features_tree_operations(self):
        """Test JAX tree operations for uptake_BV_output_features."""
        features = uptake_BV_output_features(uptake=jnp.array([[[1.0, 2.0], [3.0, 4.0]]]))

        arrays, static_data = features.tree_flatten()
        reconstructed = uptake_BV_output_features.tree_unflatten(static_data, arrays)

        assert isinstance(reconstructed, uptake_BV_output_features)
        assert jnp.array_equal(reconstructed.uptake, features.uptake)


class TestClassMetadata:
    """Test class metadata and attributes."""

    def test_bv_input_features_metadata(self):
        """Test BV_input_features class metadata."""
        assert hasattr(BV_input_features, "__features__")
        assert BV_input_features.__features__ == {"heavy_contacts", "acceptor_contacts"}

        assert hasattr(BV_input_features, "key")
        assert isinstance(BV_input_features.key, set)
        assert len(BV_input_features.key) == 2

    def test_bv_output_features_metadata(self):
        """Test BV_output_features class metadata."""
        assert hasattr(BV_output_features, "__features__")
        assert BV_output_features.__features__ == {"log_Pf", "k_ints"}

        assert hasattr(BV_output_features, "key")

    def test_uptake_output_features_metadata(self):
        """Test uptake_BV_output_features class metadata."""
        assert hasattr(uptake_BV_output_features, "__features__")
        assert uptake_BV_output_features.__features__ == {"uptake"}

        assert hasattr(uptake_BV_output_features, "key")

    def test_inheritance(self):
        """Test proper inheritance relationships."""
        assert issubclass(BV_input_features, Input_Features)
        assert issubclass(BV_input_features, AbstractFeatures)

        assert issubclass(BV_output_features, Output_Features)
        assert issubclass(BV_output_features, AbstractFeatures)

        assert issubclass(uptake_BV_output_features, Output_Features)
        assert issubclass(uptake_BV_output_features, AbstractFeatures)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_bv_input_features_with_single_residue(self):
        """Test BV_input_features with single residue."""
        features = BV_input_features(
            heavy_contacts=[[1.0, 2.0, 3.0]], acceptor_contacts=[[0.1, 0.2, 0.3]]
        )

        assert features.features_shape == (1, 3)
        pred = features.feat_pred
        assert pred[0].shape == (1, 3)
        assert pred[1].shape == (1, 3)

    def test_bv_output_features_with_single_residue(self):
        """Test BV_output_features with single residue."""
        features = BV_output_features(log_Pf=[1.0])

        assert features.output_shape == (1, 1)
        pred = features.y_pred()
        assert pred.shape == (1,)

    def test_uptake_features_with_single_values(self):
        """Test uptake_BV_output_features with minimal data."""
        features = uptake_BV_output_features(uptake=[[[1.0]]])

        assert features.output_shape == (1, 1, 1)
        pred = features.y_pred()
        assert pred.shape == (1, 1, 1)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
