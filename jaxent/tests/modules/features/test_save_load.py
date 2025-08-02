import os
import pickle
import tempfile
import threading
import time
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.custom_types.features import AbstractFeatures
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)


class TestAbstractFeaturesSaveLoad:
    """Comprehensive tests for AbstractFeatures save/load functionality using concrete classes."""

    @pytest.fixture
    def sample_input_features(self):
        """Create a sample BV_input_features instance for testing."""
        return BV_input_features(
            heavy_contacts=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            acceptor_contacts=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            k_ints=jnp.array([1, 2, 3]),
        )

    @pytest.fixture
    def sample_output_features(self):
        """Create a sample BV_output_features instance for testing."""
        return BV_output_features(
            log_Pf=jnp.array([1.5, 2.3, -0.8, 4.1]), k_ints=jnp.array([10, 20, 30])
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_basic_functionality(self, sample_input_features, temp_dir):
        """Test basic save functionality."""
        filepath = os.path.join(temp_dir, "test_basic")
        sample_input_features.save(filepath)
        assert os.path.exists(filepath + ".npz")
        assert os.path.getsize(filepath + ".npz") > 0

    def test_save_with_npz_extension(self, sample_input_features, temp_dir):
        """Test save when .npz extension is already provided."""
        filepath = os.path.join(temp_dir, "test_with_extension.npz")
        sample_input_features.save(filepath)
        assert os.path.exists(filepath)
        assert not os.path.exists(filepath + ".npz")

    def test_save_overwrites_existing_file(self, sample_input_features, temp_dir):
        """Test that save overwrites existing files."""
        filepath = os.path.join(temp_dir, "test_overwrite")
        sample_input_features.save(filepath)
        initial_mtime = os.path.getmtime(filepath + ".npz")
        time.sleep(0.01)
        sample_input_features.save(filepath)
        new_mtime = os.path.getmtime(filepath + ".npz")
        assert new_mtime > initial_mtime

    def test_save_creates_intermediate_directories(self, sample_input_features, temp_dir):
        """Test that save creates intermediate directories."""
        nested_path = os.path.join(temp_dir, "nested", "dir", "test_nested")
        sample_input_features.save(nested_path)
        assert os.path.exists(nested_path + ".npz")
        assert os.path.isdir(os.path.join(temp_dir, "nested", "dir"))

    def test_save_with_complex_paths(self, sample_input_features, temp_dir):
        """Test save with various path formats."""
        test_paths = [
            "simple_name",
            "name_with_underscores",
            "name-with-dashes",
            "name.with.dots",
            "name with spaces",
            "../relative_path",
            "./current_dir_file",
        ]
        for path_name in test_paths:
            filepath = os.path.join(temp_dir, path_name)
            sample_input_features.save(filepath)
            assert os.path.exists(filepath + ".npz"), f"Failed for path: {path_name}"

    def test_save_preserves_data_types(self, temp_dir):
        """Test that save preserves different data types correctly."""
        instance = BV_input_features(
            heavy_contacts=jnp.array([1, 2, 3], dtype=jnp.int32),
            acceptor_contacts=jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64),
            k_ints=jnp.array([True, False, True], dtype=jnp.bool),
        )
        filepath = os.path.join(temp_dir, "test_types")
        instance.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")

        assert loaded.heavy_contacts.dtype == jnp.float32
        assert loaded.acceptor_contacts.dtype == jnp.float32
        assert loaded.k_ints.dtype == jnp.bool_

        np.testing.assert_array_almost_equal(
            loaded.heavy_contacts, jnp.array([1, 2, 3], dtype=jnp.float32)
        )
        np.testing.assert_array_almost_equal(
            loaded.acceptor_contacts, jnp.array([1.5, 2.5, 3.5], dtype=jnp.float32)
        )
        np.testing.assert_array_equal(
            loaded.k_ints, jnp.array([True, False, True], dtype=jnp.bool_)
        )

    def test_save_with_none_values(self, temp_dir):
        """Test save with None values in optional parameters."""
        instance = BV_input_features(
            heavy_contacts=jnp.array([1.0, 2.0]),
            acceptor_contacts=jnp.array([3.0, 4.0]),
            k_ints=None,
        )
        filepath = os.path.join(temp_dir, "test_none")
        instance.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert loaded.k_ints is None

    def test_save_file_permissions(self, sample_input_features, temp_dir):
        """Test that saved files have appropriate permissions."""
        filepath = os.path.join(temp_dir, "test_permissions")
        sample_input_features.save(filepath)
        file_path = filepath + ".npz"
        file_stat = os.stat(file_path)
        assert file_stat.st_mode & 0o400
        assert file_stat.st_mode & 0o200

    def test_load_basic_functionality(self, sample_input_features, temp_dir):
        """Test basic load functionality."""
        filepath = os.path.join(temp_dir, "test_load_basic")
        sample_input_features.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert isinstance(loaded, BV_input_features)
        assert jnp.array_equal(loaded.heavy_contacts, sample_input_features.heavy_contacts)
        assert jnp.array_equal(loaded.acceptor_contacts, sample_input_features.acceptor_contacts)
        assert jnp.array_equal(loaded.k_ints, sample_input_features.k_ints)

    def test_load_without_npz_extension(self, sample_input_features, temp_dir):
        """Test load when .npz extension is not provided."""
        filepath = os.path.join(temp_dir, "test_load_no_ext")
        sample_input_features.save(filepath)
        with pytest.raises(FileNotFoundError):
            BV_input_features.load(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert isinstance(loaded, BV_input_features)

    def test_load_nonexistent_file(self):
        """Test load with nonexistent file."""
        with pytest.raises(FileNotFoundError, match="Features file not found"):
            BV_input_features.load("nonexistent_file.npz")

    def test_load_corrupted_file(self, temp_dir):
        """Test load with corrupted/invalid file."""
        filepath = os.path.join(temp_dir, "corrupted.npz")
        with open(filepath, "wb") as f:
            f.write(b"corrupted data")
        import zipfile

        with pytest.raises(
            (KeyError, ValueError, OSError, EOFError, zipfile.BadZipFile, pickle.UnpicklingError)
        ):
            BV_input_features.load(filepath)

    def test_load_missing_metadata(self, temp_dir):
        """Test load with file missing required metadata."""
        filepath = os.path.join(temp_dir, "missing_metadata.npz")
        jnp.savez(filepath, some_array=jnp.array([1, 2, 3]), __class_module__="test_module")
        with pytest.raises(KeyError, match="missing required metadata"):
            BV_input_features.load(filepath)

    def test_load_invalid_class_module(self, sample_input_features, temp_dir):
        """Test load with invalid class module."""
        filepath = os.path.join(temp_dir, "invalid_module")
        sample_input_features.save(filepath)
        data = dict(jnp.load(filepath + ".npz", allow_pickle=True))
        data["__class_module__"] = "nonexistent_module"
        jnp.savez(filepath + "_corrupted.npz", **data)
        with pytest.raises(ImportError, match="Cannot import class from saved file"):
            BV_input_features.load(filepath + "_corrupted.npz")

    def test_load_incompatible_class(self, temp_dir):
        """Test load with incompatible class type."""
        instance = BV_input_features(heavy_contacts=[[1.0]], acceptor_contacts=[[0.1]])
        filepath = os.path.join(temp_dir, "incompatible")
        instance.save(filepath)
        with pytest.raises(TypeError, match="is not a subclass of"):
            BV_output_features.load(filepath + ".npz")

    def test_load_class_verification(self, sample_input_features, temp_dir):
        """Test that load verifies class compatibility correctly."""
        filepath = os.path.join(temp_dir, "class_verification")
        sample_input_features.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert isinstance(loaded, BV_input_features)
        loaded_abstract = AbstractFeatures.load(filepath + ".npz")
        assert isinstance(loaded_abstract, BV_input_features)

    def test_save_load_roundtrip_data_integrity(self, temp_dir):
        """Test complete roundtrip data integrity."""
        original = BV_input_features(
            heavy_contacts=jnp.array([1.0, 2.0, 3.0, float("inf"), -float("inf")]),
            acceptor_contacts=jnp.array([[1e-10, 1e10], [0.0, -0.0]]),
            k_ints=jnp.array([]),
        )
        filepath = os.path.join(temp_dir, "integrity_test")
        original.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert jnp.array_equal(loaded.heavy_contacts, original.heavy_contacts, equal_nan=True)
        assert jnp.array_equal(loaded.acceptor_contacts, original.acceptor_contacts, equal_nan=True)
        assert jnp.array_equal(loaded.k_ints, original.k_ints)

    def test_save_load_with_large_data(self, temp_dir):
        """Test save/load with large data arrays."""
        large_array1 = jnp.ones((100, 100))
        large_array2 = jnp.arange(5000).reshape(50, 100)
        instance = BV_input_features(
            heavy_contacts=large_array1, acceptor_contacts=large_array2, k_ints=None
        )
        filepath = os.path.join(temp_dir, "large_data")
        instance.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")
        assert jnp.array_equal(loaded.heavy_contacts, instance.heavy_contacts)
        assert jnp.array_equal(loaded.acceptor_contacts, instance.acceptor_contacts)
        assert loaded.k_ints is None

    def test_concurrent_save_load(self, temp_dir):
        """Test concurrent save/load operations."""
        instances = [
            BV_input_features(
                heavy_contacts=jnp.array([[i, i + 1]]),
                acceptor_contacts=jnp.array([[i * 2, i * 3]]),
                k_ints=jnp.array([i]),
            )
            for i in range(5)
        ]
        filepaths = [os.path.join(temp_dir, f"concurrent_{i}") for i in range(5)]
        results = {}

        def save_and_load(instance, filepath, thread_id):
            try:
                instance.save(filepath)
                time.sleep(0.01)
                loaded = BV_input_features.load(filepath + ".npz")
                results[thread_id] = loaded
            except Exception as e:
                results[thread_id] = e

        threads = []
        for i, (instance, filepath) in enumerate(zip(instances, filepaths)):
            thread = threading.Thread(target=save_and_load, args=(instance, filepath, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for i, (thread_id, result) in enumerate(results.items()):
            assert not isinstance(result, Exception), f"Thread {thread_id} failed: {result}"
            assert isinstance(result, BV_input_features)
            assert jnp.array_equal(result.heavy_contacts, instances[thread_id].heavy_contacts)


class TestConcreteClassesSaveLoad:
    """Comprehensive save/load tests for concrete feature classes."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_bv_input_features_save_load_comprehensive(self, temp_dir):
        """Comprehensive save/load test for BV_input_features."""
        test_cases = [
            # Basic case
            {
                "heavy_contacts": [[1.0, 2.0], [3.0, 4.0]],
                "acceptor_contacts": [[0.1, 0.2], [0.3, 0.4]],
                "k_ints": [1, 2, 3],
            },
            # With JAX arrays
            {
                "heavy_contacts": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "acceptor_contacts": jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                "k_ints": jnp.array([1, 2]),
            },
            # Without optional k_ints
            {"heavy_contacts": [[1.0], [2.0], [3.0]], "acceptor_contacts": [[0.1], [0.2], [0.3]]},
            # Edge case: empty sequences
            {"heavy_contacts": [], "acceptor_contacts": [], "k_ints": []},
            # Single residue, multiple frames
            {
                "heavy_contacts": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "acceptor_contacts": [[0.1, 0.2, 0.3, 0.4, 0.5]],
                "k_ints": None,
            },
        ]

        for i, case_data in enumerate(test_cases):
            filepath = os.path.join(temp_dir, f"bv_input_case_{i}")
            original = BV_input_features(**case_data)
            original.save(filepath)
            loaded = BV_input_features.load(filepath + ".npz")
            assert isinstance(loaded, BV_input_features)
            assert jnp.array_equal(
                jnp.asarray(loaded.heavy_contacts), jnp.asarray(original.heavy_contacts)
            )
            assert jnp.array_equal(
                jnp.asarray(loaded.acceptor_contacts), jnp.asarray(original.acceptor_contacts)
            )
            if original.k_ints is not None:
                assert jnp.array_equal(jnp.asarray(loaded.k_ints), jnp.asarray(original.k_ints))
            else:
                assert loaded.k_ints is None
            assert loaded.features_shape == original.features_shape

    def test_bv_output_features_save_load_comprehensive(self, temp_dir):
        """Comprehensive save/load test for BV_output_features."""
        test_cases = [
            # Basic case
            {"log_Pf": [1.0, 2.0, 3.0], "k_ints": [1, 2]},
            # With JAX arrays
            {"log_Pf": jnp.array([1.5, 2.5, 3.5, 4.5]), "k_ints": jnp.array([10, 20, 30])},
            # Without optional k_ints
            {"log_Pf": [0.1, 0.2, 0.3, 0.4, 0.5]},
            # Single residue
            {"log_Pf": [42.0], "k_ints": None},
            # Large array
            {"log_Pf": list(range(1000)), "k_ints": list(range(500))},
        ]

        for i, case_data in enumerate(test_cases):
            filepath = os.path.join(temp_dir, f"bv_output_case_{i}")
            original = BV_output_features(**case_data)
            original.save(filepath)
            loaded = BV_output_features.load(filepath + ".npz")
            assert isinstance(loaded, BV_output_features)
            assert jnp.array_equal(jnp.asarray(loaded.log_Pf), jnp.asarray(original.log_Pf))
            if original.k_ints is not None:
                assert jnp.array_equal(jnp.asarray(loaded.k_ints), jnp.asarray(original.k_ints))
            else:
                assert loaded.k_ints is None
            assert loaded.output_shape == original.output_shape
            assert jnp.array_equal(loaded.y_pred(), original.y_pred())

    def test_uptake_bv_output_features_save_load_comprehensive(self, temp_dir):
        """Comprehensive save/load test for uptake_BV_output_features."""
        test_cases = [
            # Basic 3D case
            {"uptake": [[[1.0, 2.0], [3.0, 4.0]]]},
            # With JAX array
            {"uptake": jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])},
            # Single residue, single timepoint
            {"uptake": [[[42.0]]]},
            # Multiple residues, single timepoint
            {"uptake": [[[1.0], [2.0], [3.0], [4.0]]]},
            # Single residue, multiple timepoints
            {"uptake": [[[1.0, 2.0, 3.0, 4.0, 5.0]]]},
            # Complex case
            {
                "uptake": [
                    [[i * j * k for k in range(1, 4)] for j in range(1, 6)] for i in range(1, 3)
                ]
            },
        ]

        for i, case_data in enumerate(test_cases):
            filepath = os.path.join(temp_dir, f"uptake_case_{i}")
            original = uptake_BV_output_features(**case_data)
            original.save(filepath)
            loaded = uptake_BV_output_features.load(filepath + ".npz")
            assert isinstance(loaded, uptake_BV_output_features)
            assert jnp.array_equal(jnp.asarray(loaded.uptake), jnp.asarray(original.uptake))
            assert loaded.output_shape == original.output_shape
            assert jnp.array_equal(loaded.y_pred(), original.y_pred())

    def test_cross_class_loading_prevention(self, temp_dir):
        """Test that loading prevents cross-class contamination."""
        bv_input = BV_input_features(heavy_contacts=[[1.0, 2.0]], acceptor_contacts=[[0.1, 0.2]])
        bv_output = BV_output_features(log_Pf=[1.0, 2.0])
        uptake_output = uptake_BV_output_features(uptake=[[[1.0, 2.0]]])

        bv_input.save(os.path.join(temp_dir, "bv_input"))
        bv_output.save(os.path.join(temp_dir, "bv_output"))
        uptake_output.save(os.path.join(temp_dir, "uptake_output"))

        with pytest.raises(TypeError):
            BV_output_features.load(os.path.join(temp_dir, "bv_input.npz"))

        with pytest.raises(TypeError):
            BV_input_features.load(os.path.join(temp_dir, "bv_output.npz"))

        with pytest.raises(TypeError):
            uptake_BV_output_features.load(os.path.join(temp_dir, "bv_input.npz"))

        loaded_input = BV_input_features.load(os.path.join(temp_dir, "bv_input.npz"))
        loaded_output = BV_output_features.load(os.path.join(temp_dir, "bv_output.npz"))
        loaded_uptake = uptake_BV_output_features.load(os.path.join(temp_dir, "uptake_output.npz"))

        assert isinstance(loaded_input, BV_input_features)
        assert isinstance(loaded_output, BV_output_features)
        assert isinstance(loaded_uptake, uptake_BV_output_features)


class TestSaveLoadErrorHandling:
    """Test error handling and edge cases for save/load operations."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_to_readonly_directory(self, temp_dir):
        """Test save to read-only directory."""
        readonly_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)

        features = BV_input_features(heavy_contacts=[[1.0]], acceptor_contacts=[[0.1]])

        try:
            filepath = os.path.join(readonly_dir, "test")
            with pytest.raises(PermissionError):
                features.save(filepath)
        finally:
            os.chmod(readonly_dir, 0o755)

    def test_save_to_invalid_path(self):
        """Test save to invalid path."""
        features = BV_input_features(heavy_contacts=[[1.0]], acceptor_contacts=[[0.1]])

        invalid_paths = [
            "/root/invalid_path",
            "con",
            "aux",
            "\x00invalid",
        ]

        for invalid_path in invalid_paths:
            try:
                with pytest.raises((OSError, ValueError, PermissionError)):
                    features.save(invalid_path)
            except NotImplementedError:
                pass

    def test_load_from_directory(self, temp_dir):
        """Test load from directory instead of file."""
        dir_path = os.path.join(temp_dir, "test_dir.npz")
        os.makedirs(dir_path)

        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            BV_input_features.load(dir_path)

    def test_save_load_with_special_characters(self, temp_dir):
        """Test save/load with special characters in data."""
        special_data = BV_input_features(
            heavy_contacts=jnp.array([[float("inf"), float("-inf"), float("nan")]]),
            acceptor_contacts=jnp.array([[1e-100, 1e100, 0.0]]),
        )

        filepath = os.path.join(temp_dir, "special_chars")
        special_data.save(filepath)
        loaded = BV_input_features.load(filepath + ".npz")

        assert jnp.isinf(loaded.heavy_contacts[0, 0])
        assert jnp.isinf(loaded.heavy_contacts[0, 1])
        assert jnp.isnan(loaded.heavy_contacts[0, 2])

    def test_file_corruption_recovery(self, temp_dir):
        """Test handling of various file corruption scenarios."""
        features = BV_input_features(heavy_contacts=[[1.0, 2.0]], acceptor_contacts=[[0.1, 0.2]])

        filepath = os.path.join(temp_dir, "corruption_test")
        features.save(filepath)

        corruption_tests = [
            (b"", "empty file"),
            (b"corrupted", "invalid data"),
            (b"\x00" * 100, "null bytes"),
            (b"PK\x03\x04", "zip header without valid content"),
        ]

        for corrupt_data, description in corruption_tests:
            corrupt_filepath = os.path.join(
                temp_dir, f"corrupt_{description.replace(' ', '_')}.npz"
            )

            with open(corrupt_filepath, "wb") as f:
                f.write(corrupt_data)

            with pytest.raises(
                (
                    ValueError,
                    OSError,
                    KeyError,
                    EOFError,
                    zipfile.BadZipFile,
                    pickle.UnpicklingError,
                )
            ):
                BV_input_features.load(corrupt_filepath)

    def test_save_load_memory_efficiency(self, temp_dir):
        """Test memory efficiency of save/load operations."""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        large_features = BV_input_features(
            heavy_contacts=jnp.ones((1000, 1000)), acceptor_contacts=jnp.zeros((1000, 1000))
        )

        filepath = os.path.join(temp_dir, "memory_test")

        large_features.save(filepath)

        del large_features
        gc.collect()

        loaded = BV_input_features.load(filepath + ".npz")

        assert loaded.heavy_contacts.shape == (1000, 1000)
        assert jnp.all(loaded.heavy_contacts == 1.0)
        assert jnp.all(loaded.acceptor_contacts == 0.0)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        expected_size = 1000 * 1000 * 4 * 2
        assert memory_increase < expected_size * 10


class TestSaveLoadMetadata:
    """Test metadata handling in save/load operations."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_metadata_completeness(self, temp_dir):
        """Test that all required metadata is saved."""
        features = BV_input_features(heavy_contacts=[[1.0]], acceptor_contacts=[[0.1]])

        filepath = os.path.join(temp_dir, "metadata_test")
        features.save(filepath)

        raw_data = jnp.load(filepath + ".npz", allow_pickle=True)

        required_metadata = [
            "__class_module__",
            "__class_name__",
            "__static_data__",
            "__dynamic_slots__",
        ]

        for metadata_key in required_metadata:
            assert metadata_key in raw_data, f"Missing metadata: {metadata_key}"

    def test_metadata_values(self, temp_dir):
        """Test that metadata values are correct."""
        features = BV_input_features(
            heavy_contacts=[[1.0, 2.0]], acceptor_contacts=[[0.1, 0.2]], k_ints=[1, 2]
        )

        filepath = os.path.join(temp_dir, "metadata_values")
        features.save(filepath)

        raw_data = jnp.load(filepath + ".npz", allow_pickle=True)

        assert str(raw_data["__class_module__"].item()) == "jaxent.src.models.HDX.BV.features"
        assert str(raw_data["__class_name__"].item()) == "BV_input_features"

        dynamic_slots = tuple(raw_data["__dynamic_slots__"])
        expected_dynamic = {"heavy_contacts", "acceptor_contacts"}
        assert set(dynamic_slots) == expected_dynamic

        for slot in dynamic_slots:
            assert slot in raw_data

    def test_metadata_tampering_detection(self, temp_dir):
        """Test detection of tampered metadata."""
        features = BV_input_features(heavy_contacts=[[1.0]], acceptor_contacts=[[0.1]])

        filepath = os.path.join(temp_dir, "tamper_test")
        features.save(filepath)

        raw_data = dict(jnp.load(filepath + ".npz", allow_pickle=True))

        tamper_tests = [
            ("__class_module__", "fake_module"),
            ("__class_name__", "FakeClass"),
            ("__dynamic_slots__", ("fake_slot",)),
        ]

        for key, fake_value in tamper_tests:
            tampered_data = raw_data.copy()
            tampered_data[key] = fake_value

            tampered_filepath = os.path.join(temp_dir, f"tampered_{key}.npz")
            jnp.savez(tampered_filepath, **tampered_data)

            with pytest.raises((ImportError, TypeError, KeyError, AttributeError)):
                BV_input_features.load(tampered_filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
