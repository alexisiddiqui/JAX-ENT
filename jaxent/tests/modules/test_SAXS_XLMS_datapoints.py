import numpy as np
import pytest
from pathlib import Path

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.SAXS import SAXS_curve
from jaxent.src.custom_types.XLMS import XLMS_distance_restraint
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


class TestSAXSAndXLMSDatapoints:
    """Test suite for the SAXS_curve and XLMS_distance_restraint types."""

    def test_registry_registration(self):
        """Verify the new types are correctly registered in the ExpD_Datapoint registry."""
        assert "SAXS_Iq" in ExpD_Datapoint._registry
        assert "XLMS_distance" in ExpD_Datapoint._registry
        assert ExpD_Datapoint._registry["SAXS_Iq"] == SAXS_curve
        assert ExpD_Datapoint._registry["XLMS_distance"] == XLMS_distance_restraint

    def test_saxs_curve_instantiation(self, sample_topologies):
        """Verify instantiation of a SAXS_curve object."""
        q_vals = np.array([0.01, 0.02, 0.03])
        intensities = np.array([100.0, 50.0, 10.0])
        errors = np.array([1.0, 0.5, 0.1])
        
        curve = SAXS_curve(
            top=sample_topologies[3],
            intensities=intensities,
            q_values=q_vals,
            errors=errors
        )
        
        assert curve.top.fragment_name == "long_pep"
        np.testing.assert_array_equal(curve.intensities, intensities)
        np.testing.assert_array_equal(curve.q_values, q_vals)
        np.testing.assert_array_equal(curve.errors, errors)
        assert str(curve.key) == "SAXS_Iq"

    def test_saxs_curve_extract_features(self, sample_topologies):
        """Verify extraction of features from a SAXS_curve object."""
        intensities = np.array([100.0, 50.0, 10.0])
        curve = SAXS_curve(top=sample_topologies[3], intensities=intensities)
        
        features = curve.extract_features()
        assert features.shape == (3,)
        np.testing.assert_array_equal(features, intensities)

    def test_saxs_curve_create_from_features(self, sample_topologies):
        """Verify recreation of a SAXS_curve object from a flat features array."""
        topology = sample_topologies[3]
        features = np.array([100.0, 50.0, 10.0])
        
        curve = SAXS_curve._create_from_features(topology, features)
        assert isinstance(curve, SAXS_curve)
        assert curve.top == topology
        np.testing.assert_array_equal(curve.intensities, features)
        # Note: q_values and errors are not round-tripped
        assert len(curve.q_values) == 0
        assert curve.errors is None

    def test_saxs_curve_save_load(self, sample_topologies, temp_dir):
        """Verify save_list_to_files/load_list_from_files functionality for SAXS curves."""
        base_name = temp_dir / "saxs_curves"
        
        curves = [
            SAXS_curve(top=sample_topologies[3], intensities=np.array([100.0, 50.0, 10.0])),
            SAXS_curve(top=sample_topologies[3], intensities=np.array([90.0, 45.0, 9.0]))
        ]
        
        ExpD_Datapoint.save_list_to_files(curves, base_name=base_name)
        loaded_curves = ExpD_Datapoint.load_list_from_files(base_name=base_name)
        
        assert len(loaded_curves) == len(curves)
        for original, loaded in zip(curves, loaded_curves):
            assert isinstance(loaded, SAXS_curve)
            assert original.top == loaded.top
            np.testing.assert_array_equal(original.intensities, loaded.intensities)

    def test_xlms_restraint_instantiation(self, sample_topologies):
        """Verify instantiation of an XLMS_distance_restraint object."""
        restraint = XLMS_distance_restraint(
            top=sample_topologies[0],
            top_j=sample_topologies[1],
            distance=15.5,
            lower_bound=10.0,
            upper_bound=20.0
        )
        
        assert restraint.top.fragment_name == "res1"
        assert restraint.top_j is not None
        assert restraint.top_j.fragment_name == "res2"
        assert restraint.distance == 15.5
        assert restraint.lower_bound == 10.0
        assert restraint.upper_bound == 20.0
        assert str(restraint.key) == "XLMS_distance"

    def test_xlms_restraint_extract_features(self, sample_topologies):
        """Verify extraction of features from an XLMS_distance_restraint object."""
        restraint = XLMS_distance_restraint(top=sample_topologies[0], top_j=sample_topologies[1], distance=15.5)
        features = restraint.extract_features()
        assert features.shape == (1,)
        np.testing.assert_array_equal(features, np.array([15.5]))

    def test_xlms_restraint_create_from_features(self, sample_topologies):
        """Verify recreation of an XLMS_distance_restraint object from features."""
        topology = sample_topologies[0]
        features = np.array([15.5])
        
        restraint = XLMS_distance_restraint._create_from_features(topology, features)
        assert isinstance(restraint, XLMS_distance_restraint)
        assert restraint.top == topology
        assert restraint.top_j is None  # Known limitation: top_j is not round-tripped
        assert restraint.distance == 15.5

    def test_xlms_restraint_invalid_shape(self, sample_topologies):
        """Verify XLMS_distance_restraint creation fails on invalid feature shapes."""
        with pytest.raises(ValueError, match=r"XLMS_distance_restraint expects a single feature with shape \(1,\), got \(2,\)"):
            XLMS_distance_restraint._create_from_features(sample_topologies[0], np.array([15.5, 10.0]))

    def test_xlms_restraint_save_load(self, sample_topologies, temp_dir):
        """Verify save_list_to_files/load_list_from_files applies to XLMS points."""
        base_name = temp_dir / "xlms_restraints"
        
        restraints = [
            XLMS_distance_restraint(top=sample_topologies[0], top_j=sample_topologies[1], distance=15.5),
            XLMS_distance_restraint(top=sample_topologies[1], top_j=sample_topologies[0], distance=20.0)
        ]
        
        ExpD_Datapoint.save_list_to_files(restraints, base_name=base_name)
        loaded_restraints = ExpD_Datapoint.load_list_from_files(base_name=base_name)
        
        assert len(loaded_restraints) == len(restraints)
        for original, loaded in zip(restraints, loaded_restraints):
            assert isinstance(loaded, XLMS_distance_restraint)
            assert original.top == loaded.top
            assert loaded.top_j is None  # Known limitation
            assert original.distance == loaded.distance

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
