from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from jaxent.src.interfaces.topology import (
    Partial_Topology,
    TopologyFactory,
    mda_TopologyAdapter,
)
from jaxent.tests.test_utils import get_inst_path


@pytest.fixture
def bpti_universe():
    """Load BPTI structure as an MDAnalysis Universe"""
    inst_dir = get_inst_path(Path(__file__).parent.parent.parent.parent)
    pdb_path = inst_dir / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    if not pdb_path.exists():
        pytest.skip(f"Test PDB file not found: {pdb_path}")
    return mda.Universe(str(pdb_path))


@pytest.fixture
def chain_groups(bpti_universe):
    """Get various chain groups for testing"""
    protein_atoms = bpti_universe.select_atoms("protein")
    chain_id = mda_TopologyAdapter._get_chain_id(protein_atoms[0])

    single_residue_atoms = protein_atoms.residues[0].atoms
    multi_residue_atoms = protein_atoms.residues[:3].atoms
    residue_group = protein_atoms.residues[:5]

    return {
        "single_atoms": single_residue_atoms,
        "multi_atoms": multi_residue_atoms,
        "residue_group": residue_group,
        "chain_id": chain_id,
        "all_residues": list(protein_atoms.residues),
    }


class TestMdaGroupToTopology:
    """Focused tests for _mda_group_to_topology method"""

    def test_single_residue_atom_group(self, chain_groups):
        """Test topology creation from single residue AtomGroup"""
        topology = mda_TopologyAdapter._mda_group_to_topology(
            chain_groups["single_atoms"], exclude_termini=False
        )

        assert isinstance(topology, Partial_Topology)
        assert topology.chain == chain_groups["chain_id"]
        assert len(topology.residues) == 1
        assert topology.fragment_name.startswith(chain_groups["chain_id"])

    def test_multi_residue_atom_group(self, chain_groups):
        """Test topology creation from multi-residue AtomGroup"""
        topology = mda_TopologyAdapter._mda_group_to_topology(
            chain_groups["multi_atoms"], exclude_termini=False
        )

        assert isinstance(topology, Partial_Topology)
        assert topology.chain == chain_groups["chain_id"]
        assert len(topology.residues) == 3
        assert topology.fragment_name.startswith(chain_groups["chain_id"])

    def test_residue_group(self, chain_groups):
        """Test topology creation from ResidueGroup"""
        topology = mda_TopologyAdapter._mda_group_to_topology(
            chain_groups["residue_group"], exclude_termini=False
        )

        assert isinstance(topology, Partial_Topology)
        assert topology.chain == chain_groups["chain_id"]
        assert len(topology.residues) == 5

    def test_custom_fragment_name_template(self, chain_groups):
        """Test custom fragment naming"""
        template = "{chain}_custom_{resid}_{resname}"
        topology = mda_TopologyAdapter._mda_group_to_topology(
            chain_groups["single_atoms"], fragment_name_template=template, exclude_termini=False
        )

        assert "custom" in topology.fragment_name
        assert chain_groups["chain_id"] in topology.fragment_name

    def test_empty_group_error(self, bpti_universe):
        """Test error handling for empty groups"""
        empty_atoms = bpti_universe.select_atoms("resname NONEXISTENT")

        with pytest.raises(ValueError, match="contains no residues"):
            mda_TopologyAdapter._mda_group_to_topology(empty_atoms)

    def test_mixed_chain_error(self, bpti_universe):
        """Test error for groups with multiple chains"""
        # Create a mock group with different chain IDs
        all_atoms = bpti_universe.select_atoms("protein")
        if len(set(mda_TopologyAdapter._get_chain_id(atom) for atom in all_atoms)) > 1:
            with pytest.raises(ValueError, match="multiple chains"):
                mda_TopologyAdapter._mda_group_to_topology(all_atoms)
        else:
            pytest.skip("Only one chain in test structure")

    def test_unsupported_group_type(self):
        """Test error for unsupported group types"""
        with pytest.raises(TypeError, match="Unsupported group type"):
            mda_TopologyAdapter._mda_group_to_topology("invalid_type")


class TestCreateMdaGroupLookupKey:
    """Focused tests for _create_mda_group_lookup_key method"""

    def test_atom_group_lookup_key(self, chain_groups):
        """Test lookup key creation from AtomGroup"""
        key = mda_TopologyAdapter._create_mda_group_lookup_key(chain_groups["single_atoms"])

        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)  # chain_id
        assert isinstance(key[1], frozenset)  # resids
        assert len(key[1]) == 1

    def test_residue_group_lookup_key(self, chain_groups):
        """Test lookup key creation from ResidueGroup"""
        key = mda_TopologyAdapter._create_mda_group_lookup_key(chain_groups["residue_group"])

        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], str)
        assert isinstance(key[1], frozenset)
        assert len(key[1]) == 5

    def test_multi_residue_atom_group(self, chain_groups):
        """Test lookup key for multi-residue AtomGroup"""
        key = mda_TopologyAdapter._create_mda_group_lookup_key(chain_groups["multi_atoms"])

        assert isinstance(key, tuple)
        assert len(key[1]) == 3

    def test_with_renumber_mapping(self, chain_groups):
        """Test lookup key with renumbering mapping"""
        original_key = mda_TopologyAdapter._create_mda_group_lookup_key(
            chain_groups["single_atoms"]
        )
        chain_id = original_key[0]
        original_resid = next(iter(original_key[1]))

        renumber_mapping = {(chain_id, 100): original_resid}

        key_with_mapping = mda_TopologyAdapter._create_mda_group_lookup_key(
            chain_groups["single_atoms"], renumber_mapping
        )

        assert key_with_mapping is not None
        assert key_with_mapping[0] == chain_id

    def test_empty_group_returns_none(self, bpti_universe):
        """Test that empty groups return None"""
        empty_atoms = bpti_universe.select_atoms("resname NONEXISTENT")
        key = mda_TopologyAdapter._create_mda_group_lookup_key(empty_atoms)
        assert key is None

    def test_invalid_group_type_returns_none(self):
        """Test that invalid group types return None"""
        key = mda_TopologyAdapter._create_mda_group_lookup_key("invalid")
        assert key is None

    def test_mixed_chain_group_error(self, bpti_universe):
        """Test error for groups spanning multiple chains"""
        all_atoms = bpti_universe.select_atoms("protein")
        unique_chains = set(mda_TopologyAdapter._get_chain_id(atom) for atom in all_atoms)

        if len(unique_chains) > 1:
            with pytest.raises(ValueError, match="multiple chains"):
                mda_TopologyAdapter._create_mda_group_lookup_key(all_atoms)
        else:
            pytest.skip("Only one chain in test structure")


class TestBuildRenumberingMapping:
    """Focused tests for _build_renumbering_mapping method"""

    def test_basic_renumbering_mapping(self, bpti_universe):
        """Test basic renumbering mapping creation"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(bpti_universe)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Check key format: (chain_id, new_resid)
        for key in mapping.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], (int, np.integer))

        # Check value format: original_resid
        for value in mapping.values():
            assert isinstance(value, (int, np.integer))

    def test_exclude_termini_mapping(self, bpti_universe):
        """Test mapping with terminal exclusion"""
        mapping_with_termini = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )
        mapping_without_termini = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Should have fewer mappings when excluding termini
        assert len(mapping_without_termini) <= len(mapping_with_termini)

    def test_custom_termini_selection(self, bpti_universe):
        """Test mapping with custom termini selection"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True, termini_chain_selection="name CA"
        )

        assert isinstance(mapping, dict)

    def test_mapping_consistency(self, bpti_universe):
        """Test that mapping is consistent across calls"""
        mapping1 = mda_TopologyAdapter._build_renumbering_mapping(bpti_universe)
        mapping2 = mda_TopologyAdapter._build_renumbering_mapping(bpti_universe)

        assert mapping1 == mapping2

    def test_sequential_renumbering(self, bpti_universe):
        """Test that renumbering creates sequential indices"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )
        # Group by chain
        by_chain = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            if chain_id not in by_chain:
                by_chain[chain_id] = []
            by_chain[chain_id].append(new_resid)

        # Check each chain has sequential numbering starting from 1
        for chain_id, new_resids in by_chain.items():
            sorted_resids = sorted(new_resids)
            expected = list(range(1, len(sorted_resids) + 1))
            assert sorted_resids == expected, f"Chain {chain_id} not sequentially numbered"

    def test_sequential_renumbering_termini(self, bpti_universe):
        """Test that renumbering creates sequential indices"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )
        # Group by chain
        by_chain = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            if chain_id not in by_chain:
                by_chain[chain_id] = []
            by_chain[chain_id].append(new_resid)

        # Check each chain has sequential numbering starting from 1
        for chain_id, new_resids in by_chain.items():
            sorted_resids = sorted(new_resids)
            expected = list(range(2, max(sorted_resids) + 1))
            assert sorted_resids == expected, f"Chain {chain_id} not sequentially numbered"


class TestValidateTopologyContainment:
    """Focused tests for _validate_topology_containment method"""

    def test_valid_topology_passes(self, bpti_universe):
        """Test that valid topologies pass validation"""
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", exclude_termini=False
        )

        if topologies:
            # Should not raise any exception
            mda_TopologyAdapter._validate_topology_containment(
                topologies[0], bpti_universe, exclude_termini=False
            )

    def test_invalid_residue_fails(self, bpti_universe, chain_groups):
        """Test that topology with invalid residues fails validation"""
        invalid_topology = TopologyFactory.from_single(
            chain=chain_groups["chain_id"],
            residue=9999,  # Non-existent residue
            fragment_sequence="A",
            fragment_name="invalid",
            peptide=False,
        )

        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(invalid_topology, bpti_universe)

    def test_validation_with_termini_exclusion(self, bpti_universe):
        """Test validation respects termini exclusion settings"""
        topologies_no_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", exclude_termini=True, renumber_residues=True
        )

        if topologies_no_termini:
            # Should pass when using same exclusion settings
            mda_TopologyAdapter._validate_topology_containment(
                topologies_no_termini[0],
                bpti_universe,
                exclude_termini=True,
                renumber_residues=True,
            )

    def test_validation_with_renumbering(self, bpti_universe):
        """Test validation respects renumbering settings"""
        topologies_renumbered = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", renumber_residues=True
        )

        if topologies_renumbered:
            # Should pass with renumbering enabled
            mda_TopologyAdapter._validate_topology_containment(
                topologies_renumbered[0], bpti_universe, renumber_residues=True
            )

    def test_validation_no_renumbering(self, bpti_universe):
        """Test validation without renumbering"""
        topologies_original = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", renumber_residues=False
        )

        if topologies_original:
            # Should pass without renumbering
            mda_TopologyAdapter._validate_topology_containment(
                topologies_original[0], bpti_universe, renumber_residues=False
            )

    def test_validation_empty_chain_fails(self, bpti_universe):
        """Test validation fails for non-existent chains"""
        invalid_topology = TopologyFactory.from_single(
            chain="NONEXISTENT",
            residue=1,
            fragment_sequence="A",
            fragment_name="invalid",
            peptide=False,
        )

        with pytest.raises(ValueError, match="No residues found for chain"):
            mda_TopologyAdapter._validate_topology_containment(invalid_topology, bpti_universe)


class TestIntegratedBehavior:
    """Integration tests for the four utility methods working together"""

    def test_topology_roundtrip(self, bpti_universe, chain_groups):
        """Test creating topology from group and validating it"""
        # Create topology from MDA group
        topology = mda_TopologyAdapter._mda_group_to_topology(
            chain_groups["single_atoms"], exclude_termini=False
        )

        # Validate it exists in universe
        mda_TopologyAdapter._validate_topology_containment(
            topology, bpti_universe, exclude_termini=False, renumber_residues=False
        )

        # Create lookup key
        lookup_key = mda_TopologyAdapter._create_mda_group_lookup_key(chain_groups["single_atoms"])
        assert lookup_key is not None

    def test_renumbering_integration(self, bpti_universe, chain_groups):
        """Test renumbering mapping with topology validation"""
        # Build renumbering mapping
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )

        # Create topology with renumbering
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", renumber_residues=True, exclude_termini=False
        )

        if topologies:
            # Validate with renumbering
            mda_TopologyAdapter._validate_topology_containment(
                topologies[1], bpti_universe, renumber_residues=True, exclude_termini=False
            )

            # Create lookup key with mapping
            lookup_key = mda_TopologyAdapter._create_mda_group_lookup_key(
                chain_groups["single_atoms"],
                mapping,
            )
            assert lookup_key is not None, (
                f"Lookup key creation failed for {chain_groups['single_atoms']}",
                f"Lookup key: {lookup_key}",
            )

    def test_error_propagation(self, bpti_universe):
        """Test that errors propagate correctly between methods"""
        # Create invalid topology
        invalid_topology = TopologyFactory.from_single(
            chain="INVALID", residue=1, fragment_sequence="A", fragment_name="test", peptide=False
        )

        # Should fail validation
        with pytest.raises(ValueError):
            mda_TopologyAdapter._validate_topology_containment(invalid_topology, bpti_universe)
