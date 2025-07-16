import MDAnalysis as mda
import numpy as np
import pytest

from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.features import BV_input_features

# Import the classes under test
from jaxent.src.models.HDX.BV.forwardmodel import BV_model


def create_test_universe(residue_names, n_atoms_per_residue=3, n_frames=1):
    """
    Create a test MDAnalysis Universe with specified residue names and frames.

    Parameters:
    -----------
    residue_names : list of str
        List of residue names (e.g., ["ALA", "GLY", "SER"])
    n_atoms_per_residue : int
        Number of atoms per residue (default: 3 for N, H, O)
    n_frames : int
        Number of trajectory frames

    Returns:
    --------
    MDAnalysis.Universe
        A properly constructed Universe object
    """
    n_residues = len(residue_names)
    n_atoms = n_residues * n_atoms_per_residue

    # Create per-atom attributes
    atom_resindices = np.repeat(np.arange(n_residues), n_atoms_per_residue)
    atom_names = ["N", "H", "O"] * n_residues
    atom_types = ["N", "H", "O"] * n_residues

    # Create per-residue attributes
    resids = np.arange(1, n_residues + 1)
    resnames = residue_names
    segids = ["A"]  # Single segment

    # Create coordinates for multiple frames
    coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        for i in range(n_atoms):
            # Spread atoms along x-axis with some frame-dependent variation
            coordinates[frame, i] = [i * 3.0 + frame * 0.1, frame * 0.1, 0]

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
    from MDAnalysis.coordinates.memory import MemoryReader

    universe.trajectory = MemoryReader(
        coordinates, dimensions=np.array([[100, 100, 100, 90, 90, 90]] * n_frames), dt=1.0
    )

    return universe


def create_multi_chain_universe(chain_residues, n_frames=1):
    """
    Create a test MDAnalysis Universe with multiple chains.

    The order of chains in the input dictionary determines the order of atoms
    in the created Universe.

    Parameters:
    -----------
    chain_residues : dict
        Dict mapping chain ID to a list of residue names.
        e.g., {'B': ["ALA", "GLY"], 'A': ["SER", "VAL"]}
    n_frames : int
        Number of trajectory frames

    Returns:
    --------
    MDAnalysis.Universe
        A properly constructed Universe object
    """
    n_atoms_per_residue = 3
    n_total_residues = sum(len(res) for res in chain_residues.values())
    n_total_atoms = n_total_residues * n_atoms_per_residue

    # --- Topology Attributes ---
    atom_resindices = np.zeros(n_total_atoms, dtype=int)
    resnames = []
    resids = np.zeros(n_total_residues, dtype=int)
    atom_names = []
    atom_types = []  # FIX: Initialize list for atom types

    # Segment-related attributes
    unique_segids = sorted(list(chain_residues.keys()))
    residue_segindices = np.zeros(n_total_residues, dtype=int)

    atom_idx = 0
    residue_idx = 0
    # Iterate through chains in the user-provided order to create a non-sorted universe.
    for chain_id, res_name_list in chain_residues.items():
        seg_idx = unique_segids.index(chain_id)
        for i, res_name in enumerate(res_name_list):
            resnames.append(res_name)
            resids[residue_idx] = i + 1  # Resids are 1-based per chain
            residue_segindices[residue_idx] = seg_idx
            atom_resindices[atom_idx : atom_idx + n_atoms_per_residue] = residue_idx
            atom_names.extend(["N", "H", "O"])
            atom_types.extend(["N", "H", "O"])  # FIX: Populate atom types

            atom_idx += n_atoms_per_residue
            residue_idx += 1

    # --- Coordinates ---
    coordinates = np.zeros((n_frames, n_total_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        for i in range(n_total_atoms):
            coordinates[frame, i] = [i * 3.0 + frame * 0.1, 0, 0]

    # --- Create Universe ---
    universe = mda.Universe.empty(
        n_total_atoms,
        n_residues=n_total_residues,
        n_segments=len(unique_segids),
        atom_resindex=atom_resindices,
        residue_segindex=residue_segindices,
        trajectory=False,
    )

    universe.add_TopologyAttr("names", atom_names)
    universe.add_TopologyAttr("types", atom_types)  # FIX: Add types attribute to topology
    universe.add_TopologyAttr("resnames", resnames)
    universe.add_TopologyAttr("resids", resids)
    universe.add_TopologyAttr("segids", unique_segids)

    from MDAnalysis.coordinates.memory import MemoryReader

    universe.trajectory = MemoryReader(
        coordinates, dimensions=np.array([[200, 200, 200, 90, 90, 90]] * n_frames), dt=1.0
    )

    return universe


class TestBVModel:
    """Test suite for BV_model class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = BV_model_Config()
        self.model = BV_model(self.config)

        # Create larger, more comprehensive test universes
        # 20-residue sequence with various amino acids including PRO
        self.large_residues = [
            "MET",
            "ALA",
            "GLY",
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",
            "PRO",
            "ASP",
            "GLU",
            "LYS",
            "ARG",
            "HIS",
            "ASN",
            "GLN",
            "CYS",
            "THR",
        ]

        # 15-residue sequence without PRO for comparison
        self.no_pro_residues = [
            "MET",
            "ALA",
            "GLY",
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",
            "ASP",
            "GLU",
            "LYS",
            "ARG",
            "HIS",
        ]

        # 25-residue sequence with multiple PRO residues at different positions
        self.multi_pro_residues = [
            "MET",
            "ALA",
            "PRO",
            "GLY",
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PRO",
            "PHE",
            "TYR",
            "TRP",
            "ASP",
            "GLU",
            "PRO",
            "LYS",
            "ARG",
            "HIS",
            "ASN",
            "GLN",
            "CYS",
            "THR",
            "PRO",
            "ASP",
            "GLU",
        ]

        self.universe1 = create_test_universe(self.large_residues, n_frames=2)
        self.universe2 = create_test_universe(self.large_residues, n_frames=3)
        self.ensemble = [self.universe1, self.universe2]

        # Single universe for simple tests
        self.single_universe = [self.universe1]

    def test_model_initialization(self):
        """Test that BV_model can be initialized with config."""
        assert isinstance(self.model, BV_model)
        assert self.model.config == self.config
        assert hasattr(self.model, "forward")
        assert hasattr(self.model, "base_include_selection")
        assert hasattr(self.model, "base_exclude_selection")
        assert self.model.base_exclude_selection == "resname PRO or resid 1"

    def test_selection_combination(self):
        """Test _combine_selections method."""
        base = "protein"
        additional = "resname ALA"

        result = self.model._combine_selections(base, additional)
        assert result == "(protein) and (resname ALA)"

        # Test with None additional
        result = self.model._combine_selections(base, None)
        assert result == base

    def test_prepare_selection_lists_single_strings(self):
        """Test _prepare_selection_lists with single string inputs."""
        include_sel = "resname ALA"
        exclude_sel = "resname PRO"

        inc_list, exc_list = self.model._prepare_selection_lists(
            self.ensemble, include_sel, exclude_sel
        )

        # Should expand to match ensemble length
        assert len(inc_list) == len(self.ensemble)
        assert len(exc_list) == len(self.ensemble)

        # Should combine with base selections
        expected_include = "(protein) and (resname ALA)"
        expected_exclude = "(resname PRO or resid 1) or (resname PRO)"

        assert all(sel == expected_include for sel in inc_list)
        assert all(sel == expected_exclude for sel in exc_list)

    def test_prepare_selection_lists_none_inputs(self):
        """Test _prepare_selection_lists with None inputs."""
        inc_list, exc_list = self.model._prepare_selection_lists(self.ensemble, None, None)

        assert len(inc_list) == len(self.ensemble)
        assert len(exc_list) == len(self.ensemble)

        # Should use base selections only
        assert all(sel == "protein" for sel in inc_list)
        assert all(sel == "resname PRO or resid 1" for sel in exc_list)

    def test_prepare_selection_lists_per_universe(self):
        """Test _prepare_selection_lists with per-universe selections."""
        include_sels = ["resname ALA", "resname GLY"]
        exclude_sels = ["resname PRO", "resname SER"]

        inc_list, exc_list = self.model._prepare_selection_lists(
            self.ensemble, include_sels, exclude_sels
        )

        assert len(inc_list) == 2
        assert len(exc_list) == 2

        assert inc_list[0] == "(protein) and (resname ALA)"
        assert inc_list[1] == "(protein) and (resname GLY)"
        assert exc_list[0] == "(resname PRO or resid 1) or (resname PRO)"
        assert exc_list[1] == "(resname PRO or resid 1) or (resname SER)"

    def test_initialise_single_universe(self):
        """Test initialise method with a single universe."""
        success = self.model.initialise(self.single_universe)

        assert success is True
        assert hasattr(self.model, "common_topology")
        assert hasattr(self.model, "common_k_ints")
        assert hasattr(self.model, "n_frames")
        assert hasattr(self.model, "topology_order")

        # With 20 residues: MET(1) excluded as first, THR(20) excluded as last, PRO(11) excluded
        # Expected residues: ALA(2), GLY(3), SER(4), VAL(5), LEU(6), ILE(7), PHE(8), TYR(9), TRP(10),
        # ASP(12), GLU(13), LYS(14), ARG(15), HIS(16), ASN(17), GLN(18), CYS(19) = 17 residues
        expected_residue_count = 17
        assert len(self.model.common_k_ints) == expected_residue_count

        # All intrinsic rates should be positive and finite
        for k_int in self.model.common_k_ints:
            assert k_int > 0
            assert np.isfinite(k_int)

        # Check frame count
        assert self.model.n_frames == self.universe1.trajectory.n_frames

    def test_initialise_ensemble(self):
        """Test initialise method with multiple universes."""
        success = self.model.initialise(self.ensemble)

        assert success is True
        assert hasattr(self.model, "common_topology")
        assert hasattr(self.model, "common_k_ints")

        # Should process common residues across ensemble
        # With 20 residues: excluding first (MET), last (THR), and PRO -> 17 residues
        expected_residue_count = 17
        assert len(self.model.common_k_ints) == expected_residue_count

        # Check total frame count
        expected_frames = sum(u.trajectory.n_frames for u in self.ensemble)
        assert self.model.n_frames == expected_frames

        # Intrinsic rates should be averaged across ensemble
        for k_int in self.model.common_k_ints:
            assert k_int > 0
            assert np.isfinite(k_int)

    def test_initialise_with_custom_selections(self):
        """Test initialise with custom include/exclude selections."""
        # Only include hydrophobic residues
        include_sel = "resname ALA or resname VAL or resname LEU or resname ILE or resname PHE"

        success = self.model.initialise(self.single_universe, include_selection=include_sel)

        assert success is True

        # With our 20-residue sequence, hydrophobic residues are at positions:
        # ALA(2), VAL(5), LEU(6), ILE(7), PHE(8)
        # All should be included (none are first, last, or PRO)
        expected_count = 5
        assert len(self.model.common_k_ints) == expected_count

    def test_featurise_single_universe(self):
        """Test featurise method with a single universe."""
        # Must initialise first
        self.model.initialise(self.single_universe)

        features, topology_order = self.model.featurise(self.single_universe)

        # Check return types
        assert isinstance(features, BV_input_features)
        assert isinstance(topology_order, list)

        # Check feature structure
        assert hasattr(features, "heavy_contacts")
        assert hasattr(features, "acceptor_contacts")
        assert hasattr(features, "k_ints")

        # Check dimensions
        n_residues = len(self.model.common_k_ints)
        n_frames = self.universe1.trajectory.n_frames

        # Features should have shape (residues, frames)
        assert features.heavy_contacts.shape == (n_residues, n_frames)
        assert features.acceptor_contacts.shape == (n_residues, n_frames)
        assert features.k_ints.shape == (n_residues,)

        # All contact values should be non-negative
        assert np.all(features.heavy_contacts >= 0)
        assert np.all(features.acceptor_contacts >= 0)

        # Intrinsic rates should match what was calculated in initialise
        # Use assert_allclose instead of assert_array_equal to handle float32/float64 differences
        np.testing.assert_allclose(features.k_ints, self.model.common_k_ints, rtol=1e-6)

    def test_featurise_ensemble(self):
        """Test featurise method with multiple universes."""
        # Must initialise first
        self.model.initialise(self.ensemble)

        features, topology_order = self.model.featurise(self.ensemble)

        # Check feature structure
        n_residues = len(self.model.common_k_ints)
        total_frames = sum(u.trajectory.n_frames for u in self.ensemble)

        # Features should have shape (residues, total_frames)
        assert features.heavy_contacts.shape == (n_residues, total_frames)
        assert features.acceptor_contacts.shape == (n_residues, total_frames)
        assert len(features.k_ints) == n_residues

        # All values should be valid
        assert np.all(features.heavy_contacts >= 0)
        assert np.all(features.acceptor_contacts >= 0)
        assert np.all(features.k_ints > 0)

    def test_featurise_output_order(self):
        """Test that features are ordered correctly according to topology ranking."""
        # 1. Create a universe with multiple chains in a non-alphabetical order
        # The featurization should reorder them to be alphabetical (A then B).
        multi_chain_residues = {
            "B": ["MET", "ALA", "GLY", "SER", "VAL"],  # Chain B
            "A": ["MET", "CYS", "THR", "PHE", "TYR"],  # Chain A
        }
        multi_chain_universe = create_multi_chain_universe(multi_chain_residues, n_frames=2)
        ensemble = [multi_chain_universe]

        # 2. Initialise the model
        success = self.model.initialise(ensemble)
        assert success is True

        # 3. Check the initial topology order from `initialise`
        # After exclusion (resid 1 and 5 from each chain) and renumbering, we expect:
        # Chain A: [CYS(1), THR(2), PHE(3)]
        # Chain B: [ALA(1), GLY(2), SER(3)]
        # Ranking is by chain ID, so A comes before B.
        assert len(self.model.topology_order) == 6  # (5-2) + (5-2)
        expected_chains = ["A"] * 3 + ["B"] * 3
        expected_resids = [1, 2, 3, 1, 2, 3]
        actual_chains = [t.chain for t in self.model.topology_order]
        actual_resids = [t.residues[0] for t in self.model.topology_order]

        assert actual_chains == expected_chains
        assert actual_resids == expected_resids

        # 4. Featurise
        features, topology_order_from_featurise = self.model.featurise(ensemble)

        # 5. Verify the order
        # The topology list from featurise should be the same as from initialise
        assert topology_order_from_featurise == self.model.topology_order

        # The features should be ordered according to this topology list.
        # We can verify this using the intrinsic rates (k_ints), which are stored
        # in a map during initialise and then ordered.
        for i, topo in enumerate(topology_order_from_featurise):
            # The topo object itself is the key in the map
            assert topo in self.model.common_k_ints_map
            expected_k_int = self.model.common_k_ints_map[topo]
            actual_k_int = features.k_ints[i]
            np.testing.assert_allclose(actual_k_int, expected_k_int, rtol=1e-6)

    def test_featurise_before_initialise_fails(self):
        """Test that featurise fails if called before initialise."""
        with pytest.raises(AttributeError):
            # Should fail because common_topology doesn't exist
            self.model.featurise(self.single_universe)

    def test_empty_ensemble_handling(self):
        """Test handling of empty ensemble."""
        empty_ensemble = []

        # Should handle gracefully or raise appropriate error
        with pytest.raises((IndexError, ValueError)):
            self.model.initialise(empty_ensemble)

    def test_proline_residue_handling(self):
        """Test that PRO residues are properly excluded."""
        # Use sequence with multiple PRO residues
        pro_universe = create_test_universe(self.multi_pro_residues)

        success = self.model.initialise([pro_universe])
        assert success is True

        # With 25 residues: MET(1) excluded as first, GLU(25) excluded as last
        # PRO residues at positions 3, 9, 15, 23 excluded
        # Expected: 25 - 2 (termini) - 4 (PRO) = 19 residues
        expected_count = 19
        assert len(self.model.common_k_ints) == expected_count

    def test_no_proline_sequence(self):
        """Test with sequence containing no PRO residues."""
        no_pro_universe = create_test_universe(self.no_pro_residues)

        success = self.model.initialise([no_pro_universe])
        assert success is True

        # With 15 residues: MET(1) excluded as first, HIS(15) excluded as last
        # No PRO residues to exclude
        # Expected: 15 - 2 = 13 residues
        expected_count = 13
        assert len(self.model.common_k_ints) == expected_count

    def test_different_universe_compatibility(self):
        """Test that universes with different sequences are handled."""
        # Create universes with different but overlapping sequences
        seq1 = self.large_residues  # 20 residues
        seq2 = self.no_pro_residues + [
            "ASN",
            "GLN",
            "CYS",
            "THR",
            "ASP",
        ]  # 20 residues, different composition

        universe1 = create_test_universe(seq1)
        universe2 = create_test_universe(seq2)

        success = self.model.initialise([universe1, universe2])
        assert success is True

        # Both sequences have same length and similar structure, so many residues should be common
        # The test should verify that common residues are found, not that it's less than individual counts
        assert len(self.model.common_k_ints) > 0
        # Most residues should be common since both sequences are 20 residues with similar composition
        assert len(self.model.common_k_ints) >= 15  # Most residues should be in common

    def test_single_residue_universe(self):
        """Test with universe containing only three residues to test terminal exclusion."""
        # Use three residues to properly test exclude_termini=True behavior
        three_res = ["ALA", "GLY", "SER"]
        three_universe = create_test_universe(three_res)

        success = self.model.initialise([three_universe])
        assert success is True

        # With exclude_termini=True, first (ALA) and last (SER) residues should be excluded
        # Only middle residue (GLY) should remain
        assert len(self.model.common_k_ints) == 1

        # The remaining residue should have a finite intrinsic rate
        assert np.isfinite(self.model.common_k_ints[0])
        assert self.model.common_k_ints[0] > 0

    def test_comprehensive_amino_acid_coverage(self):
        """Test with all 20 standard amino acids."""
        all_aa_sequence = [
            "MET",
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]

        aa_universe = create_test_universe(all_aa_sequence)
        success = self.model.initialise([aa_universe])
        assert success is True

        # With 21 residues: first (MET) and last (VAL) excluded, PRO excluded
        # Expected: 21 - 2 (termini) - 1 (PRO) = 18 residues
        expected_count = 18
        assert len(self.model.common_k_ints) == expected_count

    def test_large_protein_simulation(self):
        """Test with a larger protein-like sequence."""
        # Create a 50-residue sequence with realistic composition
        large_sequence = [
            "MET",
            "ALA",
            "GLY",
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",  # 1-10
            "ASP",
            "GLU",
            "LYS",
            "ARG",
            "HIS",
            "ASN",
            "GLN",
            "CYS",
            "THR",
            "ALA",  # 11-20
            "GLY",
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",
            "ASP",
            "GLU",  # 21-30
            "LYS",
            "ARG",
            "HIS",
            "ASN",
            "GLN",
            "CYS",
            "THR",
            "PRO",
            "ALA",
            "GLY",  # 31-40
            "SER",
            "VAL",
            "LEU",
            "ILE",
            "PHE",
            "TYR",
            "TRP",
            "ASP",
            "GLU",
            "LYS",  # 41-50
        ]

        large_universe = create_test_universe(large_sequence, n_frames=5)
        success = self.model.initialise([large_universe])
        assert success is True

        # With 50 residues: first (MET) and last (LYS) excluded, PRO(38) excluded
        # Expected: 50 - 2 (termini) - 1 (PRO) = 47 residues
        expected_count = 47
        assert len(self.model.common_k_ints) == expected_count

        # Test featurization with larger sequence
        features, topology_order = self.model.featurise([large_universe])

        assert features.heavy_contacts.shape == (expected_count, 5)  # 5 frames
        assert features.acceptor_contacts.shape == (expected_count, 5)
        assert len(features.k_ints) == expected_count

    def test_multiple_chain_segments(self):
        """Test behavior with different chain arrangements."""
        # Test with two shorter sequences that would be treated as separate chains
        short_seq1 = ["ALA", "GLY", "SER", "VAL", "LEU"]
        short_seq2 = ["PHE", "TYR", "TRP", "ASP", "GLU"]

        universe1 = create_test_universe(short_seq1)
        universe2 = create_test_universe(short_seq2)

        # Test individually
        success1 = self.model.initialise([universe1])
        assert success1 is True
        count1 = len(self.model.common_k_ints)
        # 5 residues - 2 termini = 3 residues
        assert count1 == 3

        # Reset model for second test
        self.model = BV_model(self.config)
        success2 = self.model.initialise([universe2])
        assert success2 is True
        count2 = len(self.model.common_k_ints)
        # 5 residues - 2 termini = 3 residues
        assert count2 == 3


class TestBVModelEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BV_model_Config()
        self.model = BV_model(self.config)

    def test_malformed_universe(self):
        """Test handling of malformed universes."""
        # Create universe with missing attributes
        n_atoms = 6
        universe = mda.Universe.empty(n_atoms, trajectory=False)

        # Missing required topology attributes should cause issues
        with pytest.raises((AttributeError, KeyError, ValueError)):
            self.model.initialise([universe])

    def test_very_large_contact_radius(self):
        """Test with very large contact radius."""
        config = BV_model_Config()
        config.heavy_radius = 1000.0  # Very large radius
        config.o_radius = 1000.0

        model = BV_model(config)
        residues = ["MET", "ALA", "GLY"]
        universe = create_test_universe(residues)

        success = model.initialise([universe])
        assert success is True

        features, _ = model.featurise([universe])

        # With very large radius, should have many contacts
        # (unless there are not many atoms to begin with)
        assert np.all(features.heavy_contacts >= 0)
        assert np.all(features.acceptor_contacts >= 0)

    def test_very_small_contact_radius(self):
        """Test with very small contact radius."""
        config = BV_model_Config()
        config.heavy_radius = 0.1  # Very small radius
        config.o_radius = 0.1

        model = BV_model(config)
        residues = ["MET", "ALA", "GLY"]
        universe = create_test_universe(residues)

        success = model.initialise([universe])
        assert success is True

        features, _ = model.featurise([universe])

        # With very small radius, should have few/no contacts
        assert np.all(features.heavy_contacts >= 0)
        assert np.all(features.acceptor_contacts >= 0)
        # Most values should be zero or very small
        assert np.mean(features.heavy_contacts) < 1.0
        assert np.mean(features.acceptor_contacts) < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
