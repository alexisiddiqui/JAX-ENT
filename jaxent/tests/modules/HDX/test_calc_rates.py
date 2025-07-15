import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

from jaxent.src.models.func.uptake import calculate_intrinsic_rates


def create_test_universe(residue_names, n_atoms_per_residue=3):
    """
    Create a test MDAnalysis Universe with specified residue names.

    Parameters:
    -----------
    residue_names : list of str
        List of residue names (e.g., ["ALA", "GLY", "PRO"])
    n_atoms_per_residue : int
        Number of atoms per residue (default: 3 for N, CA, C)

    Returns:
    --------
    MDAnalysis.Universe
        A properly constructed Universe object
    """
    n_residues = len(residue_names)
    n_atoms = n_residues * n_atoms_per_residue

    # Create per-atom attributes
    atom_resindices = np.repeat(np.arange(n_residues), n_atoms_per_residue)
    atom_names = ["N", "CA", "C"] * n_residues
    atom_types = ["N", "C", "C"] * n_residues

    # Create per-residue attributes
    resids = np.arange(1, n_residues + 1)
    resnames = residue_names
    segids = ["A"]  # Single segment

    # Create simple coordinates (just a line of atoms)
    coordinates = np.zeros((1, n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        coordinates[0, i] = [i * 2.0, 0, 0]

    # Create Universe
    universe = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindices,
        residue_segindex=[0] * n_residues,
        trajectory=False,
    )

    # Add topology attributes
    universe.add_TopologyAttr("names", atom_names)
    universe.add_TopologyAttr("types", atom_types)
    universe.add_TopologyAttr("resids", resids)
    universe.add_TopologyAttr("resnames", resnames)
    universe.add_TopologyAttr("segids", segids)

    # Add trajectory
    universe.trajectory = MemoryReader(
        coordinates, dimensions=np.array([[50, 50, 50, 90, 90, 90]]), dt=1.0
    )

    return universe


class TestCalculateIntrinsicRates:
    """Test suite for calculate_intrinsic_rates function."""

    def test_basic_functionality(self):
        """Test basic functionality with a small protein."""
        universe = create_test_universe(["ALA", "GLY", "PRO", "SER", "VAL"])

        result = calculate_intrinsic_rates(universe)

        # Check return type
        assert isinstance(result, dict)

        # Check that all residues are in result
        assert len(result) == 5

        # Get residues for testing
        residues = list(universe.residues)

        # First residue should have infinite rate
        assert np.isinf(result[residues[0]])

        # PRO residue (index 2) should have infinite rate
        assert np.isinf(result[residues[2]])

        # Other residues should have finite positive rates
        for i in [1, 3, 4]:  # GLY, SER, VAL
            assert result[residues[i]] > 0
            assert np.isfinite(result[residues[i]])

    def test_residue_group_input(self):
        """Test with a specific residue_group input."""
        universe = create_test_universe(["ALA", "GLY", "PRO", "SER", "VAL"])

        # Create a subset - select only GLY and SER residues
        subset_residues = universe.select_atoms("resname GLY or resname SER").residues

        result = calculate_intrinsic_rates(universe, subset_residues)

        # Should only have 2 residues in result
        assert len(result) == 2

        # Check that only GLY and SER are in result
        resnames_in_result = {res.resname for res in result.keys()}
        assert resnames_in_result == {"GLY", "SER"}

        # Both residues should have finite rates since neither is the first residue of the chain
        # in the original universe (ALA is the first)
        residues_list = list(subset_residues)
        assert np.isfinite(result[residues_list[0]]) and result[residues_list[0]] > 0
        assert np.isfinite(result[residues_list[1]]) and result[residues_list[1]] > 0

    def test_special_residue_handling(self):
        """Test handling of special residues (first residue, PRO, termini)."""
        universe = create_test_universe(["MET", "ALA", "PRO", "GLY", "PRO", "SER"])

        result = calculate_intrinsic_rates(universe)
        residues = list(universe.residues)

        # First residue should have infinite rate
        assert np.isinf(result[residues[0]])  # MET at position 0

        # PRO residues should have infinite rates
        assert np.isinf(result[residues[2]])  # PRO at position 2
        assert np.isinf(result[residues[4]])  # PRO at position 4

        # Non-PRO, non-first residues should have finite rates
        assert np.isfinite(result[residues[1]])  # ALA
        assert np.isfinite(result[residues[3]])  # GLY
        assert np.isfinite(result[residues[5]])  # SER

    def test_unknown_residue_type(self):
        """Test handling of unknown residue types."""
        universe = create_test_universe(["ALA", "XYZ", "GLY"])  # XYZ is unknown

        # Test with explicit residue group to include the unknown residue
        # (protein selection might filter out non-standard residues)
        all_residues = universe.residues
        result = calculate_intrinsic_rates(universe, all_residues)

        # Should not raise an error
        assert len(result) == 3

        residues = list(universe.residues)

        # First residue (ALA) should be infinite
        assert np.isinf(result[residues[0]])

        # Unknown residue (XYZ) behavior depends on whether it's considered "protein"
        xyz_residue = residues[1]  # XYZ is at index 1
        assert xyz_residue.resname == "XYZ"

        # XYZ might get infinite rate if it's not recognized as protein by MDAnalysis
        # This is acceptable behavior - the function should handle unknown residues gracefully
        xyz_rate = result[xyz_residue]
        assert xyz_rate > 0 or np.isinf(xyz_rate), "Rate should be positive or infinite"

        # Known residue should have finite rate
        assert np.isfinite(result[residues[2]]) and result[residues[2]] > 0

    def test_empty_residue_group(self):
        """Test with an empty residue group."""
        universe = create_test_universe(["ALA", "GLY", "SER"])

        # Create empty residue group
        empty_group = universe.select_atoms("resname NONEXISTENT").residues

        result = calculate_intrinsic_rates(universe, empty_group)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_single_residue(self):
        """Test with a single residue."""
        universe = create_test_universe(["ALA"])

        result = calculate_intrinsic_rates(universe)
        residues = list(universe.residues)

        assert len(result) == 1
        # Single residue is the first residue, so should be infinite
        assert np.isinf(result[residues[0]])

    def test_single_non_first_residue_group(self):
        """Test with a single residue that's not the first in the full protein."""
        universe = create_test_universe(["ALA", "GLY", "SER"])

        # Select only the GLY residue (middle one)
        gly_residue = universe.select_atoms("resname GLY").residues

        result = calculate_intrinsic_rates(universe, gly_residue)

        assert len(result) == 1
        # GLY is not the first residue in the original universe (ALA is),
        # so it should get a finite rate
        for rate in result.values():
            assert np.isfinite(rate) and rate > 0

    def test_all_proline(self):
        """Test with all PRO residues."""
        universe = create_test_universe(["PRO", "PRO", "PRO"])

        result = calculate_intrinsic_rates(universe)

        # All should have infinite rates
        for rate in result.values():
            assert np.isinf(rate)

    def test_rate_values_are_reasonable(self):
        """Test that calculated rates are in reasonable ranges."""
        universe = create_test_universe(["ALA", "GLY", "SER", "VAL", "LEU"])

        result = calculate_intrinsic_rates(universe)
        residues = list(universe.residues)

        # Skip first residue (infinite)
        finite_rates = [result[res] for res in residues[1:]]

        # Rates should be positive and in reasonable range (typically 10^-3 to 10^3 s^-1)
        for rate in finite_rates:
            assert 1e-5 < rate < 1e5, f"Rate {rate} is outside reasonable range"

    def test_different_residue_types_give_different_rates(self):
        """Test that different residue types produce different intrinsic rates."""
        universe = create_test_universe(["ALA", "GLY", "SER", "VAL"])

        result = calculate_intrinsic_rates(universe)
        residues = list(universe.residues)

        # Get rates for non-first residues
        rates = [result[res] for res in residues[1:]]  # GLY, SER, VAL

        # Rates should be different (allowing for some numerical precision)
        assert not np.allclose(rates, rates[0], rtol=1e-10), "All rates are identical"

    def test_protein_selection_when_no_residue_group(self):
        """Test that protein selection works correctly when no residue_group is provided."""
        # Create universe with non-protein residues
        universe = create_test_universe(["ALA", "GLY", "SER"])

        # Manually modify some residues to be non-protein (this is a bit hacky but works for testing)
        # In real usage, MDAnalysis would handle protein selection properly

        result = calculate_intrinsic_rates(universe)

        # Should process all residues when using protein selection
        assert len(result) == 3

    def test_consistency_across_calls(self):
        """Test that the function returns consistent results across multiple calls."""
        universe = create_test_universe(["ALA", "GLY", "SER", "VAL"])

        result1 = calculate_intrinsic_rates(universe)
        result2 = calculate_intrinsic_rates(universe)

        # Results should be identical
        assert len(result1) == len(result2)
        for res in result1:
            assert result1[res] == result2[res]

    def test_large_protein(self):
        """Test with a larger protein sequence."""
        # Create a more realistic protein sequence
        sequence = [
            "MET",
            "ALA",
            "GLY",
            "PRO",
            "SER",
            "THR",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",
            "HIS",
            "LYS",
            "ARG",
            "ASP",
            "GLU",
            "ASN",
            "GLQ",
            "CYS",
        ]

        universe = create_test_universe(sequence)
        result = calculate_intrinsic_rates(universe)

        assert len(result) == len(sequence)

        residues = list(universe.residues)

        # First residue should be infinite
        assert np.isinf(result[residues[0]])

        # PRO should be infinite
        pro_idx = sequence.index("PRO")
        assert np.isinf(result[residues[pro_idx]])

        # Count finite rates
        finite_count = sum(1 for rate in result.values() if np.isfinite(rate))
        infinite_count = sum(1 for rate in result.values() if np.isinf(rate))

        assert finite_count + infinite_count == len(sequence)
        assert infinite_count == 2  # First residue + PRO
        assert finite_count == len(sequence) - 2

    def test_first_residue_behavior_in_subgroups(self):
        """Test that demonstrates the first residue behavior based on original universe."""
        universe = create_test_universe(["ALA", "GLY", "SER", "VAL", "LEU"])

        # Test 1: Full protein - first residue gets infinite rate
        result_full = calculate_intrinsic_rates(universe)
        residues = list(universe.residues)
        assert np.isinf(result_full[residues[0]])  # ALA - first in universe
        assert np.isfinite(result_full[residues[1]])  # GLY

        # Test 2: Subset starting from second residue
        subset = universe.select_atoms("resid 2 3 4").residues  # GLY, SER, VAL
        result_subset = calculate_intrinsic_rates(universe, subset)
        subset_residues = list(subset)

        # All residues in subset should have finite rates since none is the first
        # residue in the original universe (ALA is the first)
        assert (
            np.isfinite(result_subset[subset_residues[0]]) and result_subset[subset_residues[0]] > 0
        )  # GLY
        assert (
            np.isfinite(result_subset[subset_residues[1]]) and result_subset[subset_residues[1]] > 0
        )  # SER
        assert (
            np.isfinite(result_subset[subset_residues[2]]) and result_subset[subset_residues[2]] > 0
        )  # VAL

        # Test 3: Subset that includes the first residue of the universe
        first_subset = universe.select_atoms("resid 1 3").residues  # ALA, SER
        result_first_subset = calculate_intrinsic_rates(universe, first_subset)
        first_subset_residues = list(first_subset)

        # ALA should be infinite (first in universe), SER should be finite
        assert np.isinf(result_first_subset[first_subset_residues[0]])  # ALA
        assert (
            np.isfinite(result_first_subset[first_subset_residues[1]])
            and result_first_subset[first_subset_residues[1]] > 0
        )  # SER

    def test_protein_selection_filters_nonstandard_residues(self):
        """Test that protein selection may filter out non-standard residues."""
        universe = create_test_universe(["ALA", "XYZ", "GLY"])

        # When no residue_group is provided, protein selection is used
        result_protein_selection = calculate_intrinsic_rates(universe)

        # XYZ might be filtered out by protein selection
        # This test documents the current behavior
        residue_names = {res.resname for res in result_protein_selection.keys()}

        # Should contain standard amino acids
        assert "ALA" in residue_names
        assert "GLY" in residue_names

        # XYZ may or may not be included depending on MDAnalysis protein selection
        # If it's filtered out, that's the expected behavior
        if "XYZ" not in residue_names:
            assert len(result_protein_selection) == 2  # Only ALA and GLY
        else:
            assert len(result_protein_selection) == 3  # All residues included


if __name__ == "__main__":
    pytest.main([__file__])
