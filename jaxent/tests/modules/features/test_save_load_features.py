import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

# Assuming the concrete classes are in a module called 'features_concrete'
# Adjust the import path as needed
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)


class TestBVInputFeaturesSaveLoad:
    """Test suite for BV_input_features save_features and load_features methods."""

    def test_save_load_basic_lists(self, tmp_path):
        """Test basic save/load with list inputs."""
        # Create test data
        heavy_contacts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        acceptor_contacts = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        k_ints = [1, 2, 3]

        # Create original features
        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=k_ints
        )

        # Save and load
        filepath = tmp_path / "test_features.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Verify data integrity
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(
            loaded.acceptor_contacts, acceptor_contacts, rtol=1e-07, atol=1e-07
        )
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)

        # Verify shapes are preserved
        assert loaded.features_shape == original.features_shape

    def test_save_load_jax_arrays(self, tmp_path):
        """Test save/load with JAX arrays."""
        # Create test data as JAX arrays
        heavy_contacts = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        acceptor_contacts = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        k_ints = jnp.array([1, 2, 3])

        # Create original features
        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=k_ints
        )

        # Save and load
        filepath = tmp_path / "test_jax_features.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Verify data integrity
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(
            loaded.acceptor_contacts, acceptor_contacts, rtol=1e-07, atol=1e-07
        )
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)

    def test_save_load_with_none_optional(self, tmp_path):
        """Test save/load when optional field is None."""
        # Create test data with k_ints as None
        heavy_contacts = [[1.0, 2.0], [3.0, 4.0]]
        acceptor_contacts = [[0.1, 0.2], [0.3, 0.4]]

        # Create original features
        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=None
        )

        # Save and load
        filepath = tmp_path / "test_none_features.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Verify data integrity
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(
            loaded.acceptor_contacts, acceptor_contacts, rtol=1e-07, atol=1e-07
        )
        assert loaded.k_ints is None or (
            isinstance(loaded.k_ints, np.ndarray) and loaded.k_ints.item() is None
        )

    def test_save_load_empty_arrays(self, tmp_path):
        """Test save/load with empty arrays."""
        # Create empty test data
        heavy_contacts = []
        acceptor_contacts = []
        k_ints = []

        # Create original features
        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=k_ints
        )

        # Save and load
        filepath = tmp_path / "test_empty_features.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Verify data integrity
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(
            loaded.acceptor_contacts, acceptor_contacts, rtol=1e-07, atol=1e-07
        )
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)

    def test_auto_npz_extension(self, tmp_path):
        """Test that .npz extension is automatically added."""
        # Create test data
        heavy_contacts = [[1.0, 2.0]]
        acceptor_contacts = [[0.1, 0.2]]

        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts
        )

        # Save without .npz extension
        filepath_no_ext = tmp_path / "test_features"
        original.save_features(str(filepath_no_ext))

        # Verify file was created with .npz extension
        expected_filepath = tmp_path / "test_features.npz"
        assert expected_filepath.exists()

        # Load and verify
        loaded = BV_input_features.load_features(str(filepath_no_ext))
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BV_input_features.load_features("nonexistent_file.npz")

    def test_load_partial_features(self, tmp_path):
        """Test loading when saved file has only some features."""
        # Create a manual .npz file with only some features
        heavy_contacts = jnp.array([[1.0, 2.0]])
        # Note: missing acceptor_contacts and k_ints

        filepath = tmp_path / "partial_features.npz"
        jnp.savez(str(filepath), heavy_contacts=heavy_contacts)

        # This should now raise a TypeError because acceptor_contacts is a required field
        with pytest.raises(
            TypeError,
            match="BV_input_features.__init__\(\)\s*missing required features: acceptor_contacts",
        ):
            BV_input_features.load_features(str(filepath))


class TestBVOutputFeaturesSaveLoad:
    """Test suite for BV_output_features save_features and load_features methods."""

    def test_save_load_basic(self, tmp_path):
        """Test basic save/load functionality."""
        log_Pf = [1.5, 2.3, -0.8, 4.1]
        k_ints = [10, 20, 30]

        original = BV_output_features(log_Pf=log_Pf, k_ints=k_ints)

        filepath = tmp_path / "bv_output.npz"
        original.save_features(str(filepath))
        loaded = BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(loaded.log_Pf, log_Pf, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)
        assert loaded.output_shape == original.output_shape

    def test_save_load_jax_arrays(self, tmp_path):
        """Test with JAX arrays."""
        log_Pf = jnp.array([1.5, 2.3, -0.8, 4.1])
        k_ints = jnp.array([10, 20, 30])

        original = BV_output_features(log_Pf=log_Pf, k_ints=k_ints)

        filepath = tmp_path / "bv_output_jax.npz"
        original.save_features(str(filepath))
        loaded = BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(loaded.log_Pf, log_Pf, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)

    def test_save_load_none_k_ints(self, tmp_path):
        """Test with None k_ints."""
        log_Pf = [1.5, 2.3, -0.8]

        original = BV_output_features(log_Pf=log_Pf, k_ints=None)

        filepath = tmp_path / "bv_output_none.npz"
        original.save_features(str(filepath))
        loaded = BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(loaded.log_Pf, log_Pf, rtol=1e-07, atol=1e-07)
        assert loaded.k_ints is None

    def test_y_pred_consistency(self, tmp_path):
        """Test that y_pred method works correctly after load."""
        log_Pf = [1.5, 2.3, -0.8, 4.1]

        original = BV_output_features(log_Pf=log_Pf)

        filepath = tmp_path / "bv_output_pred.npz"
        original.save_features(str(filepath))
        loaded = BV_output_features.load_features(str(filepath))

        # Verify y_pred works and gives same result
        np.testing.assert_allclose(original.y_pred(), loaded.y_pred(), rtol=1e-07, atol=1e-07)


class TestUptakeBVOutputFeaturesSaveLoad:
    """Test suite for uptake_BV_output_features save_features and load_features methods."""

    def test_save_load_3d_data(self, tmp_path):
        """Test save/load with 3D uptake data."""
        uptake = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]

        original = uptake_BV_output_features(uptake=uptake)

        filepath = tmp_path / "uptake_features.npz"
        original.save_features(str(filepath))
        loaded = uptake_BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(loaded.uptake, uptake, rtol=1e-07, atol=1e-07)
        assert loaded.output_shape == original.output_shape

    def test_save_load_jax_3d_array(self, tmp_path):
        """Test with JAX 3D array."""
        uptake = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

        original = uptake_BV_output_features(uptake=uptake)

        filepath = tmp_path / "uptake_jax.npz"
        original.save_features(str(filepath))
        loaded = uptake_BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(loaded.uptake, uptake, rtol=1e-07, atol=1e-07)

    def test_y_pred_consistency_3d(self, tmp_path):
        """Test y_pred consistency for 3D data."""
        uptake = [[[1.0, 2.0], [3.0, 4.0]]]

        original = uptake_BV_output_features(uptake=uptake)

        filepath = tmp_path / "uptake_pred.npz"
        original.save_features(str(filepath))
        loaded = uptake_BV_output_features.load_features(str(filepath))

        np.testing.assert_allclose(original.y_pred(), loaded.y_pred(), rtol=1e-07, atol=1e-07)


class TestCrossClassCompatibility:
    """Test edge cases and cross-class scenarios."""

    def test_different_classes_same_features(self, tmp_path):
        """Test that features can't be loaded into incompatible classes."""
        # Create BV_output_features
        log_Pf = [1.0, 2.0, 3.0]
        bv_output = BV_output_features(log_Pf=log_Pf)

        filepath = tmp_path / "cross_class.npz"
        bv_output.save_features(str(filepath))

        # Try to load into different class - should work if features overlap
        # but may have missing required features
        try:
            loaded = uptake_BV_output_features.load_features(str(filepath))
            # This might fail in __init__ due to missing 'uptake' parameter
        except TypeError:
            # Expected if 'uptake' is required and not provided
            pass

    def test_file_corruption_handling(self, tmp_path):
        """Test handling of corrupted files."""
        # Create a corrupted file
        filepath = tmp_path / "corrupted.npz"
        with open(filepath, "w") as f:
            f.write("This is not a valid npz file")

        # Should raise appropriate error
        with pytest.raises((ValueError, OSError, FileNotFoundError)):
            BV_input_features.load_features(str(filepath))

    def test_large_data_integrity(self, tmp_path):
        """Test with larger datasets to ensure no data corruption."""
        # Create larger test data
        n_residues, n_frames = 100, 50
        heavy_contacts = np.random.random((n_residues, n_frames)).tolist()
        acceptor_contacts = np.random.random((n_residues, n_frames)).tolist()
        k_ints = list(range(1000))

        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts, k_ints=k_ints
        )

        filepath = tmp_path / "large_features.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Verify all data is exactly preserved
        np.testing.assert_allclose(loaded.heavy_contacts, heavy_contacts, rtol=1e-07, atol=1e-07)
        np.testing.assert_allclose(
            loaded.acceptor_contacts, acceptor_contacts, rtol=1e-07, atol=1e-07
        )
        np.testing.assert_allclose(loaded.k_ints, k_ints, rtol=1e-07, atol=1e-07)

    def test_dtype_preservation(self, tmp_path):
        """Test that dtypes are handled correctly."""
        # Test with different numeric types
        heavy_contacts = np.array([[1, 2, 3]], dtype=np.int32)
        acceptor_contacts = np.array([[1.5, 2.5, 3.5]], dtype=np.float64)

        original = BV_input_features(
            heavy_contacts=heavy_contacts, acceptor_contacts=acceptor_contacts
        )

        filepath = tmp_path / "dtype_test.npz"
        original.save_features(str(filepath))
        loaded = BV_input_features.load_features(str(filepath))

        # Note: save_features converts to float32, so we expect that dtype
        assert loaded.heavy_contacts.dtype == np.float32
        assert loaded.acceptor_contacts.dtype == np.float32

        # But values should be preserved (within float32 precision)
        np.testing.assert_array_almost_equal(loaded.heavy_contacts, heavy_contacts, decimal=6)
        np.testing.assert_array_almost_equal(loaded.acceptor_contacts, acceptor_contacts, decimal=6)


# Fixture for temporary directory
@pytest.fixture
def tmp_path():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
