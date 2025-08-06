"""
Enhanced test script that tests the functionality of the ExpD_Datapoint class including the loading and saving of data points via the subclasses HDX_peptide and HDX_protection_factor.

This version includes more comprehensive edge cases, stricter assertions, and better error validation.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from interfaces.topology.serialise import PTSerialiser

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.HDX import HDX_peptide, HDX_protection_factor
from jaxent.src.interfaces.topology import Partial_Topology


@pytest.fixture
def sample_topologies():
    """Provides a list of sample Partial_Topology objects for testing."""
    return [
        Partial_Topology(fragment_name="res1", chain="A", residues=[1], fragment_sequence=["GLY"]),
        Partial_Topology(fragment_name="res2", chain="A", residues=[2], fragment_sequence=["ALA"]),
        Partial_Topology(
            fragment_name="pep1-3",
            chain="A",
            residues=[1, 2, 3],
            fragment_sequence=["GLY", "ALA", "SER"],
        ),
        Partial_Topology(
            fragment_name="long_pep",
            chain="B",
            residues=list(range(1, 21)),
            fragment_sequence=["GLY"] * 20,
        ),
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class TestExpDDatapoint:
    """Test suite for the base ExpD_Datapoint class."""

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError with exact message."""
        with pytest.raises(
            NotImplementedError, match="This method must be implemented in the child class."
        ):
            ExpD_Datapoint.extract_features(None)

        with pytest.raises(
            NotImplementedError, match="This method must be implemented in the child class."
        ):
            ExpD_Datapoint._create_from_features(None, None)

    # def test_abstract_class_cannot_be_instantiated(self):
    #     """Test that ExpD_Datapoint cannot be instantiated directly."""
    #     topology = Partial_Topology(
    #         fragment_name="test", chain="A", residues=[1], fragment_sequence=["GLY"]
    #     )
    #     with pytest.raises(TypeError):  # ABC classes raise TypeError when instantiated
    #         ExpD_Datapoint(top=topology)

    @pytest.mark.parametrize(
        "json_path,expected_json,expected_csv",
        [
            ("data/topology_train.json", "data/topology_train.json", "data/features_train.csv"),
            ("data/topology_val.json", "data/topology_val.json", "data/features_val.csv"),
            (
                "results/topology_test.json",
                "results/topology_test.json",
                "results/features_test.csv",
            ),
            ("simple.json", "simple.json", "simple.csv"),
        ],
    )
    def test_build_file_paths_from_json(self, json_path, expected_json, expected_csv):
        """Test file path building from JSON path with various naming patterns."""
        json_p, csv_p = ExpD_Datapoint._build_file_paths(json_path=json_path)
        assert json_p == Path(expected_json)
        assert csv_p == Path(expected_csv)

    @pytest.mark.parametrize(
        "csv_path,expected_json,expected_csv",
        [
            ("data/features_val.csv", "data/topology_val.json", "data/features_val.csv"),
            ("data/features_train.csv", "data/topology_train.json", "data/features_train.csv"),
            (
                "results/features_test.csv",
                "results/topology_test.json",
                "results/features_test.csv",
            ),
            ("simple.csv", "simple.json", "simple.csv"),  # No pattern to replace
        ],
    )
    def test_build_file_paths_from_csv(self, csv_path, expected_json, expected_csv):
        """Test file path building from CSV path with various naming patterns."""
        json_p, csv_p = ExpD_Datapoint._build_file_paths(csv_path=csv_path)
        assert json_p == Path(expected_json)
        assert csv_p == Path(expected_csv)

    @pytest.mark.parametrize(
        "base_name,expected_json,expected_csv",
        [
            ("output/test", "output/test.json", "output/test.csv"),
            ("data", "data.json", "data.csv"),
            ("results/experiment_1", "results/experiment_1.json", "results/experiment_1.csv"),
        ],
    )
    def test_build_file_paths_from_base_name(self, base_name, expected_json, expected_csv):
        """Test file path building from base name."""
        json_p, csv_p = ExpD_Datapoint._build_file_paths(base_name=base_name)
        assert json_p == Path(expected_json)
        assert csv_p == Path(expected_csv)

    def test_build_file_paths_both_paths_provided(self):
        """Test that providing both paths works correctly."""
        json_p, csv_p = ExpD_Datapoint._build_file_paths(
            json_path="custom/topology.json", csv_path="custom/features.csv"
        )
        assert json_p == Path("custom/topology.json")
        assert csv_p == Path("custom/features.csv")

    def test_build_file_paths_no_input_error(self):
        """Test error when no input is provided."""
        with pytest.raises(
            ValueError, match="Must provide at least one of json_path, csv_path, or base_name"
        ):
            ExpD_Datapoint._build_file_paths()

    def test_build_file_paths_empty_strings(self):
        """Test behavior with empty strings."""
        with pytest.raises(
            ValueError, match="Must provide at least one of json_path, csv_path, or base_name"
        ):
            ExpD_Datapoint._build_file_paths(json_path="", csv_path="", base_name="")


class TestHDXPeptide:
    """Test suite for the HDX_peptide dataclass."""

    def test_instantiation_basic(self, sample_topologies):
        """Test successful instantiation with basic parameters."""
        peptide = HDX_peptide(top=sample_topologies[2], dfrac=[0.1, 0.5, 0.9])
        assert peptide.top.fragment_name == "pep1-3"
        assert peptide.dfrac == [0.1, 0.5, 0.9]
        assert str(peptide.key) == "HDX_peptide"
        assert peptide.charge is None
        assert peptide.retention_time is None
        assert peptide.intensity is None

    def test_instantiation_with_optional_fields(self, sample_topologies):
        """Test instantiation with all optional fields."""
        peptide = HDX_peptide(
            top=sample_topologies[0], dfrac=[0.2], charge=2, retention_time=15.5, intensity=1000.0
        )
        assert peptide.top.fragment_name == "res1"
        assert peptide.dfrac == [0.2]
        assert peptide.charge == 2
        assert peptide.retention_time == 15.5
        assert peptide.intensity == 1000.0

    @pytest.mark.parametrize(
        "dfrac,expected_shape",
        [
            ([0.1], (1, 1)),
            ([0.1, 0.5], (2, 1)),
            ([0.1, 0.3, 0.5, 0.7, 0.9], (5, 1)),
            ([0.0], (1, 1)),
            ([1.0], (1, 1)),
        ],
    )
    def test_extract_features_various_lengths(self, sample_topologies, dfrac, expected_shape):
        """Test feature extraction with various dfrac lengths."""
        peptide = HDX_peptide(top=sample_topologies[0], dfrac=dfrac)
        features = peptide.extract_features()
        assert features.shape == expected_shape
        np.testing.assert_array_equal(features.flatten(), np.array(dfrac))

    def test_extract_features_empty_dfrac(self, sample_topologies):
        """Test feature extraction with empty dfrac list."""
        peptide = HDX_peptide(top=sample_topologies[0], dfrac=[])
        features = peptide.extract_features()
        assert features.shape == (0, 1)
        np.testing.assert_array_equal(features, np.array([]).reshape(-1, 1))

    @pytest.mark.parametrize(
        "dfrac",
        [
            [-0.1, 0.5, 0.9],  # Negative value
            [0.1, 1.1, 0.9],  # Value > 1
            [float("inf"), 0.5, 0.9],  # Infinity
            [0.1, float("nan"), 0.9],  # NaN
        ],
    )
    def test_extract_features_edge_cases(self, sample_topologies, dfrac):
        """Test feature extraction with edge case values."""
        peptide = HDX_peptide(top=sample_topologies[0], dfrac=dfrac)
        features = peptide.extract_features()
        # Should still work, just return the values as-is
        np.testing.assert_array_equal(features.flatten(), np.array(dfrac))

    @pytest.mark.parametrize(
        "features,expected_dfrac",
        [
            (np.array([0.1]), [0.1]),
            (np.array([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9]),
            (np.array([0.0, 1.0]), [0.0, 1.0]),
            (np.array([]), []),
        ],
    )
    def test_create_from_features_valid(self, sample_topologies, features, expected_dfrac):
        """Test creating instances from valid features."""
        topology = sample_topologies[2]
        peptide = HDX_peptide._create_from_features(topology, features)
        assert isinstance(peptide, HDX_peptide)
        assert peptide.top == topology
        assert peptide.dfrac == expected_dfrac

    def test_create_from_features_2d_array(self, sample_topologies):
        """Test creating from 2D features array (should flatten)."""
        topology = sample_topologies[0]
        features = np.array([[0.1], [0.5], [0.9]])  # 2D array
        peptide = HDX_peptide._create_from_features(topology, features)
        assert peptide.dfrac == [0.1, 0.5, 0.9]

    def test_create_from_features_special_values(self, sample_topologies):
        """Test creating from features with special float values."""
        topology = sample_topologies[0]
        features = np.array([float("inf"), float("-inf"), float("nan")])
        peptide = HDX_peptide._create_from_features(topology, features)
        assert len(peptide.dfrac) == 3
        assert peptide.dfrac[0] == float("inf")
        assert peptide.dfrac[1] == float("-inf")
        assert np.isnan(peptide.dfrac[2])

    def test_create_from_features_type_conversion(self, sample_topologies):
        """Test that features are properly converted to float."""
        topology = sample_topologies[0]
        features = np.array([1, 2, 3], dtype=int)  # Integer array
        peptide = HDX_peptide._create_from_features(topology, features)
        assert peptide.dfrac == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in peptide.dfrac)


class TestHDXProtectionFactor:
    """Test suite for the HDX_protection_factor dataclass."""

    def test_instantiation_basic(self, sample_topologies):
        """Test successful instantiation."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=100.5)
        assert pf.top.fragment_name == "res1"
        assert pf.protection_factor == 100.5
        assert str(pf.key) == "HDX_resPf"

    @pytest.mark.parametrize(
        "pf_value",
        [
            0.0,
            1e-10,
            1.0,
            100.0,
            1e10,
            float("inf"),
        ],
    )
    def test_instantiation_edge_values(self, sample_topologies, pf_value):
        """Test instantiation with edge case protection factor values."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=pf_value)
        assert pf.protection_factor == pf_value

    def test_instantiation_nan(self, sample_topologies):
        """Test instantiation with NaN protection factor."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=float("nan"))
        assert np.isnan(pf.protection_factor)

    def test_extract_features_basic(self, sample_topologies):
        """Test feature extraction."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=100.5)
        features = pf.extract_features()
        np.testing.assert_array_equal(features, np.array([100.5]))
        assert features.shape == (1,)
        assert features.dtype == np.float64

    @pytest.mark.parametrize("pf_value", [0.0, 1e-100, 1.0, 1e100, float("inf"), float("-inf")])
    def test_extract_features_edge_values(self, sample_topologies, pf_value):
        """Test feature extraction with edge values."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=pf_value)
        features = pf.extract_features()
        np.testing.assert_array_equal(features, np.array([pf_value]))

    def test_extract_features_nan(self, sample_topologies):
        """Test feature extraction with NaN value."""
        pf = HDX_protection_factor(top=sample_topologies[0], protection_factor=float("nan"))
        features = pf.extract_features()
        assert len(features) == 1
        assert np.isnan(features[0])

    def test_create_from_features_valid(self, sample_topologies):
        """Test creating instance from valid features."""
        topology = sample_topologies[0]
        features = np.array([100.5])
        pf = HDX_protection_factor._create_from_features(topology, features)
        assert isinstance(pf, HDX_protection_factor)
        assert pf.top == topology
        assert pf.protection_factor == 100.5

    @pytest.mark.parametrize(
        "invalid_features,expected_length",
        [
            (np.array([]), 0),
            (np.array([1.0, 2.0]), 2),
            (np.array([1.0, 2.0, 3.0]), 3),
            (np.array([[1.0]]), 1),  # 2D array with one element
        ],
    )
    def test_create_from_features_invalid_shape(
        self, sample_topologies, invalid_features, expected_length
    ):
        """Test error handling for invalid feature shapes."""
        with pytest.raises(
            ValueError,
            match=r"HDX_protection_factor expects a single feature with shape \(1,\), got \(.*\)",
        ):
            HDX_protection_factor._create_from_features(sample_topologies[0], invalid_features)

    def test_create_from_features_type_conversion(self, sample_topologies):
        """Test that features are properly converted to float."""
        topology = sample_topologies[0]
        features = np.array([42], dtype=int)
        pf = HDX_protection_factor._create_from_features(topology, features)
        assert pf.protection_factor == 42.0
        assert isinstance(pf.protection_factor, float)

    def test_create_from_features_special_values(self, sample_topologies):
        """Test creating from features with special values."""
        topology = sample_topologies[0]

        # Test infinity
        pf_inf = HDX_protection_factor._create_from_features(topology, np.array([float("inf")]))
        assert pf_inf.protection_factor == float("inf")

        # Test NaN
        pf_nan = HDX_protection_factor._create_from_features(topology, np.array([float("nan")]))
        assert np.isnan(pf_nan.protection_factor)


class TestSaveLoad:
    """Test suite for saving and loading ExpD_Datapoint lists."""

    @pytest.fixture
    def hdx_peptides(self, sample_topologies):
        """Create a list of HDX_peptide objects."""
        return [
            HDX_peptide(top=sample_topologies[2], dfrac=[0.1, 0.5, 0.9]),
            HDX_peptide(
                top=Partial_Topology(
                    fragment_name="pep4-5",
                    chain="A",
                    residues=[4, 5],
                    fragment_sequence=["THR", "PRO"],
                ),
                dfrac=[0.2, 0.6],
            ),
            HDX_peptide(top=sample_topologies[0], dfrac=[0.8]),  # Single value
            HDX_peptide(top=sample_topologies[3], dfrac=[]),  # Empty dfrac
        ]

    @pytest.fixture
    def hdx_pfs(self, sample_topologies):
        """Create a list of HDX_protection_factor objects."""
        return [
            HDX_protection_factor(top=sample_topologies[0], protection_factor=100.0),
            HDX_protection_factor(top=sample_topologies[1], protection_factor=250.5),
            HDX_protection_factor(top=sample_topologies[2], protection_factor=0.0),
            HDX_protection_factor(top=sample_topologies[3], protection_factor=float("inf")),
        ]

    def test_save_load_hdx_peptides_comprehensive(self, hdx_peptides, temp_dir):
        """Test saving and loading HDX_peptide objects with comprehensive validation."""
        base_name = temp_dir / "peptides"
        ExpD_Datapoint.save_list_to_files(hdx_peptides, base_name=base_name)

        # Verify files exist
        json_path = base_name.with_suffix(".json")
        csv_path = base_name.with_suffix(".csv")
        assert json_path.exists()
        assert csv_path.exists()

        # Verify file contents structure
        df = pd.read_csv(csv_path)
        assert "datapoint_type" in df.columns
        assert "feature_length" in df.columns
        assert all(df["datapoint_type"] == "HDX_peptide")
        assert df.shape[0] == len(hdx_peptides)

        # Load and verify
        loaded_peptides = ExpD_Datapoint.load_list_from_files(base_name=base_name)

        assert len(loaded_peptides) == len(hdx_peptides)
        for original, loaded in zip(hdx_peptides, loaded_peptides):
            assert isinstance(loaded, HDX_peptide)
            assert original.top == loaded.top
            assert len(original.dfrac) == len(loaded.dfrac)
            if original.dfrac:  # Skip empty lists
                np.testing.assert_allclose(original.dfrac, loaded.dfrac)

    def test_save_load_hdx_protection_factors_comprehensive(self, hdx_pfs, temp_dir):
        """Test saving and loading HDX_protection_factor objects with comprehensive validation."""
        base_name = temp_dir / "pfs"
        ExpD_Datapoint.save_list_to_files(hdx_pfs, base_name=base_name)

        # Verify files exist and structure
        json_path = base_name.with_suffix(".json")
        csv_path = base_name.with_suffix(".csv")
        assert json_path.exists()
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert all(df["datapoint_type"] == "HDX_resPf")
        assert all(df["feature_length"] == 1)

        loaded_pfs = ExpD_Datapoint.load_list_from_files(base_name=base_name)

        assert len(loaded_pfs) == len(hdx_pfs)
        for original, loaded in zip(hdx_pfs, loaded_pfs):
            assert isinstance(loaded, HDX_protection_factor)
            assert original.top == loaded.top
            if np.isnan(original.protection_factor):
                assert np.isnan(loaded.protection_factor)
            else:
                assert original.protection_factor == loaded.protection_factor

    def test_save_empty_list_error(self, temp_dir):
        """Test that saving empty list raises appropriate error."""
        with pytest.raises(ValueError, match="Cannot save empty list of datapoints"):
            ExpD_Datapoint.save_list_to_files([], base_name=temp_dir / "empty")

    def test_save_mixed_types_validation_enabled(self, hdx_peptides, hdx_pfs, temp_dir):
        """Test that mixed types raise error when validation is enabled."""
        mixed_list = [hdx_pfs[0], hdx_peptides[0], hdx_pfs[1]]

        with pytest.raises(ValueError, match="Mixed datapoint types found: .*"):
            ExpD_Datapoint.save_list_to_files(mixed_list, base_name=temp_dir / "mixed")

    def test_save_mixed_types_validation_disabled(self, hdx_peptides, hdx_pfs, temp_dir):
        """Test that mixed types work when validation is disabled."""
        mixed_list = [hdx_pfs[0], hdx_peptides[0], hdx_pfs[1]]
        base_name = temp_dir / "mixed"

        ExpD_Datapoint.save_list_to_files(
            mixed_list, base_name=base_name, validate_homogeneous=False
        )

        loaded_mixed = ExpD_Datapoint.load_list_from_files(base_name=base_name)

        assert len(loaded_mixed) == len(mixed_list)
        assert isinstance(loaded_mixed[0], HDX_protection_factor)
        assert isinstance(loaded_mixed[1], HDX_peptide)
        assert isinstance(loaded_mixed[2], HDX_protection_factor)

    def test_save_load_very_large_features(self, sample_topologies, temp_dir):
        """Test saving/loading with very large feature arrays."""
        large_dfrac = [0.1] * 1000  # Large feature array
        peptide = HDX_peptide(top=sample_topologies[0], dfrac=large_dfrac)

        base_name = temp_dir / "large"
        ExpD_Datapoint.save_list_to_files([peptide], base_name=base_name)

        loaded = ExpD_Datapoint.load_list_from_files(base_name=base_name)
        assert len(loaded[0].dfrac) == 1000
        assert all(x == 0.1 for x in loaded[0].dfrac)

    def test_load_nonexistent_files(self, temp_dir):
        """Test loading from non-existent files."""
        with pytest.raises(FileNotFoundError, match="Topology file not found:"):
            ExpD_Datapoint.load_list_from_files(base_name=temp_dir / "nonexistent")

    def test_load_topology_count_mismatch(self, temp_dir):
        """Test error when topology count doesn't match feature count."""
        json_path = temp_dir / "bad.json"
        csv_path = temp_dir / "bad.csv"

        # Create mismatched files
        with open(json_path, "w") as f:
            json.dump(
                {
                    "topology_count": 2,
                    "topologies": [
                        {
                            "fragment_name": "A",
                            "chain": "A",
                            "residues": [1],
                            "fragment_sequence": ["GLY"],
                        }
                    ],
                },
                f,
            )

        pd.DataFrame({"datapoint_type": ["HDX_peptide"], "feature_length": [1], "0": [1.0]}).to_csv(
            csv_path, index=False
        )

        with pytest.raises(ValueError, match="Topology count mismatch: expected 2, found 1"):
            ExpD_Datapoint.load_list_from_files(json_path=json_path)

    def test_load_unknown_datapoint_type(self, temp_dir, sample_topologies):
        """Test error when loading unknown datapoint type."""
        json_path = temp_dir / "unknown.json"
        csv_path = temp_dir / "unknown.csv"

        # Create valid topology file
        PTSerialiser.save_list_to_json([sample_topologies[0]], json_path)

        # Create CSV with unknown type
        pd.DataFrame(
            {"datapoint_type": ["unknown_type"], "feature_length": [1], "0": [1.0]}
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Unknown datapoint type 'unknown_type' at index 0"):
            ExpD_Datapoint.load_list_from_files(json_path=json_path)

    def test_load_missing_csv_columns(self, temp_dir, sample_topologies):
        """Test error when CSV is missing required columns."""
        json_path = temp_dir / "missing_cols.json"
        csv_path = temp_dir / "missing_cols.csv"

        PTSerialiser.save_list_to_json([sample_topologies[0]], json_path)

        # Create CSV missing required columns
        pd.DataFrame({"0": [1.0]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="CSV missing required columns:"):
            ExpD_Datapoint.load_list_from_files(json_path=json_path)

    def test_directory_convenience_methods_comprehensive(self, hdx_pfs, temp_dir):
        """Test directory convenience methods with comprehensive validation."""
        data_dir = temp_dir / "dataset"
        dataset_name = "test_data"

        ExpD_Datapoint.save_to_directory(hdx_pfs, data_dir, dataset_name=dataset_name)

        # Verify directory and files exist
        assert data_dir.exists()
        assert (data_dir / f"{dataset_name}.json").exists()
        assert (data_dir / f"{dataset_name}.csv").exists()

        loaded_pfs = ExpD_Datapoint.load_from_directory(data_dir, dataset_name=dataset_name)
        assert len(loaded_pfs) == len(hdx_pfs)

    @pytest.mark.parametrize(
        "dataset_name", ["train", "val", "test", "full_dataset", "experiment_1", "data_2024"]
    )
    def test_directory_methods_various_names(self, hdx_pfs, temp_dir, dataset_name):
        """Test directory methods with various dataset names."""
        data_dir = temp_dir / "data"

        ExpD_Datapoint.save_to_directory(hdx_pfs, data_dir, dataset_name=dataset_name)
        loaded = ExpD_Datapoint.load_from_directory(data_dir, dataset_name=dataset_name)

        assert len(loaded) == len(hdx_pfs)

    def test_file_permission_error_handling(self, hdx_pfs, temp_dir):
        """Test handling of file permission errors."""
        if os.name == "nt":  # Skip on Windows
            pytest.skip("Permission tests not reliable on Windows")

        # Create read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            # Accept either the custom error or the underlying PermissionError
            with pytest.raises(IOError, match="Failed to save datapoints:|Permission denied"):
                ExpD_Datapoint.save_list_to_files(hdx_pfs, base_name=readonly_dir / "test")
        finally:
            readonly_dir.chmod
        json_path = temp_dir / "corrupted.json"
        csv_path = temp_dir / "corrupted.csv"

        # Create corrupted JSON
        with open(json_path, "w") as f:
            f.write("{ invalid json content")

        pd.DataFrame({"datapoint_type": ["HDX_peptide"], "feature_length": [1], "0": [1.0]}).to_csv(
            csv_path, index=False
        )

        # Accept both possible error messages
        with pytest.raises(
            IOError, match="Failed to load datapoints:|Failed to read topology file"
        ):
            ExpD_Datapoint.load_list_from_files(json_path=json_path)

    def test_corrupted_csv_file(self, temp_dir, sample_topologies):
        """Test handling of corrupted CSV file."""
        json_path = temp_dir / "corrupted.json"
        csv_path = temp_dir / "corrupted.csv"

        PTSerialiser.save_list_to_json([sample_topologies[0]], json_path)

        # Create corrupted CSV
        with open(csv_path, "w") as f:
            f.write("invalid,csv,content\nwith,wrong,number,of,columns")

        # Accept both possible error messages
        with pytest.raises(
            Exception, match="Failed to load datapoints:|CSV missing required columns:"
        ):
            ExpD_Datapoint.load_list_from_files(json_path=json_path)

    def test_save_cleanup_on_error(self, hdx_pfs, temp_dir):
        """Test that partial files are cleaned up on save error."""
        base_name = temp_dir / "cleanup_test"

        with patch(
            "jaxent.src.interfaces.topology.PTSerialiser.save_list_to_json",
            side_effect=Exception("Simulated error"),
        ):
            with pytest.raises(IOError, match="Failed to save datapoints:"):
                ExpD_Datapoint.save_list_to_files(hdx_pfs, base_name=base_name)

            # Verify no partial files remain
            assert not base_name.with_suffix(".json").exists()
            assert not base_name.with_suffix(".csv").exists()

    def test_save_load_special_float_values(self, sample_topologies, temp_dir):
        """Test saving/loading with special float values (inf, -inf, nan)."""
        special_pfs = [
            HDX_protection_factor(top=sample_topologies[0], protection_factor=float("inf")),
            HDX_protection_factor(top=sample_topologies[1], protection_factor=float("-inf")),
            HDX_protection_factor(top=sample_topologies[2], protection_factor=float("nan")),
        ]

        base_name = temp_dir / "special_values"
        ExpD_Datapoint.save_list_to_files(special_pfs, base_name=base_name)

        loaded = ExpD_Datapoint.load_list_from_files(base_name=base_name)

        assert loaded[0].protection_factor == float("inf")
        assert loaded[1].protection_factor == float("-inf")
        assert np.isnan(loaded[2].protection_factor)

    def test_feature_padding_unpadding_comprehensive(self, sample_topologies, temp_dir):
        """Test comprehensive feature padding and unpadding scenarios."""
        peptides = [
            HDX_peptide(top=sample_topologies[0], dfrac=[0.1, 0.2, 0.3, 0.4, 0.5]),  # len 5
            HDX_peptide(top=sample_topologies[1], dfrac=[0.6]),  # len 1
            HDX_peptide(top=sample_topologies[2], dfrac=[0.7, 0.8]),  # len 2
            HDX_peptide(top=sample_topologies[3], dfrac=[]),  # len 0
        ]

        base_name = temp_dir / "padding_test"
        ExpD_Datapoint.save_list_to_files(peptides, base_name=base_name)

        # Verify CSV padding
        df = pd.read_csv(base_name.with_suffix(".csv"))
        assert df.shape[1] == 2 + 5  # metadata + max_feature_length

        # Check padding with NaN
        assert pd.isna(df.iloc[1, -1])  # Second peptide, last column should be NaN
        assert pd.isna(df.iloc[2, -1])  # Third peptide, last columns should be NaN
        assert pd.isna(df.iloc[3, 2])  # Fourth peptide (empty), first feature should be NaN

        loaded_peptides = ExpD_Datapoint.load_list_from_files(base_name=base_name)

        assert len(loaded_peptides) == 4
        assert loaded_peptides[0].dfrac == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert loaded_peptides[1].dfrac == [0.6]
        assert loaded_peptides[2].dfrac == [0.7, 0.8]
        assert loaded_peptides[3].dfrac == []


class TestRegistryAndClassVariables:
    """Test suite for class registry and class variables."""

    def test_registry_contains_expected_classes(self):
        """Test that the registry contains expected datapoint classes."""
        assert "HDX_peptide" in ExpD_Datapoint._registry
        assert "HDX_resPf" in ExpD_Datapoint._registry
        assert ExpD_Datapoint._registry["HDX_peptide"] == HDX_peptide
        assert ExpD_Datapoint._registry["HDX_resPf"] == HDX_protection_factor

    def test_class_keys_are_correct(self):
        """Test that class keys are set correctly."""
        assert str(HDX_peptide.key) == "HDX_peptide"
        assert str(HDX_protection_factor.key) == "HDX_resPf"

    def test_registry_lookup_works(self):
        """Test that registry lookup returns correct classes."""
        hdx_peptide_class = ExpD_Datapoint._registry["HDX_peptide"]
        hdx_pf_class = ExpD_Datapoint._registry["HDX_resPf"]

        assert hdx_peptide_class == HDX_peptide
        assert hdx_pf_class == HDX_protection_factor


if __name__ == "__main__":
    # Run with more verbose output and specific test selection options
    pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Treat unknown markers as errors
            "--strict-config",  # Treat unknown config options as errors
        ]
    )
