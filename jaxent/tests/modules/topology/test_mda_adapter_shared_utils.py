from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from jaxent.src.interfaces.topology import (
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
def sample_residues(bpti_universe):
    """Get sample residues from BPTI for testing"""
    protein_atoms = bpti_universe.select_atoms("protein")
    return list(protein_atoms.residues[:5])  # First 5 residues


class TestSharedUtilityMethods:
    """Test suite for mda_TopologyAdapter shared utility methods"""

    def test_get_chain_id(self, bpti_universe):
        """Test _get_chain_id method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        first_atom = protein_atoms[0]
        first_residue = protein_atoms.residues[0]

        # Test with atom
        chain_id_atom = mda_TopologyAdapter._get_chain_id(first_atom)
        assert isinstance(chain_id_atom, str)
        assert len(chain_id_atom) > 0

        # Test with residue
        chain_id_residue = mda_TopologyAdapter._get_chain_id(first_residue)
        assert isinstance(chain_id_residue, str)
        assert chain_id_atom == chain_id_residue

    def test_extract_sequence(self, sample_residues):
        """Test _extract_sequence method"""
        # Test string output
        sequence_str = mda_TopologyAdapter._extract_sequence(sample_residues, return_list=False)
        assert isinstance(sequence_str, str)
        assert len(sequence_str) == len(sample_residues)
        assert all(aa in "ACDEFGHIKLMNPQRSTVWYX" for aa in sequence_str)

        # Test list output
        sequence_list = mda_TopologyAdapter._extract_sequence(sample_residues, return_list=True)
        assert isinstance(sequence_list, list)
        assert len(sequence_list) == len(sample_residues)
        assert sequence_str == "".join(sequence_list)

        # Test empty input
        empty_sequence = mda_TopologyAdapter._extract_sequence([], return_list=False)
        assert empty_sequence == ""

    def test_check_chain(self, bpti_universe):
        """Test _check_chain method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        sample_atom = protein_atoms[0]

        # Get the actual chain ID from the atom
        if hasattr(sample_atom, "chainid"):
            actual_chain_id = sample_atom.chainid
        elif hasattr(sample_atom, "segid"):
            actual_chain_id = sample_atom.segid
        else:
            pytest.skip("No chain identifier available")

        chain_id, (has_chainid, has_segid) = mda_TopologyAdapter._check_chain(
            actual_chain_id, sample_atom
        )

        assert chain_id == actual_chain_id
        assert isinstance(has_chainid, bool)
        assert isinstance(has_segid, bool)
        assert has_chainid or has_segid  # At least one should be True

    def test_extract_chain_identifier(self, bpti_universe):
        """Test _extract_chain_identifier method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        sample_atom = protein_atoms[0]

        # Test without providing chain_id
        chain_id, (has_chainid, has_segid) = mda_TopologyAdapter._extract_chain_identifier(
            sample_atom
        )
        assert isinstance(chain_id, str)
        assert len(chain_id) > 0
        assert isinstance(has_chainid, bool)
        assert isinstance(has_segid, bool)

        # Test with universe
        chain_id_universe, attrs = mda_TopologyAdapter._extract_chain_identifier(bpti_universe)
        assert isinstance(chain_id_universe, str)

        # Test providing specific chain_id
        chain_id_provided, attrs_provided = mda_TopologyAdapter._extract_chain_identifier(
            sample_atom, chain_id
        )
        assert chain_id_provided == chain_id

    def test_build_residue_selection_string(self, bpti_universe):
        """Test _build_residue_selection_string method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        chain_id = mda_TopologyAdapter._get_chain_id(protein_atoms[0])
        resids = [1, 2, 3, 5, 10]

        selection_string = mda_TopologyAdapter._build_residue_selection_string(
            chain_id, resids, bpti_universe
        )

        assert isinstance(selection_string, str)
        assert len(selection_string) > 0
        assert "resid" in selection_string
        assert "1 2 3 5 10" in selection_string

        # Test empty resids
        empty_selection = mda_TopologyAdapter._build_residue_selection_string(
            chain_id, [], bpti_universe
        )
        assert empty_selection == ""

    def test_normalize_parameters(self, bpti_universe):
        """Test _normalize_parameters method"""
        ensemble = [bpti_universe, bpti_universe]

        # Test with string parameters
        include_list, exclude_list, termini_list, exclude_termini_list = (
            mda_TopologyAdapter._normalize_parameters(
                ensemble,
                include_selection="protein",
                exclude_selection="resname SOL",
                termini_chain_selection="protein",
                exclude_termini=True,
            )
        )

        assert len(include_list) == 2
        assert len(exclude_list) == 2
        assert len(termini_list) == 2
        assert len(exclude_termini_list) == 2
        assert all(sel == "protein" for sel in include_list)
        assert all(sel == "resname SOL" for sel in exclude_list)
        assert all(exc for exc in exclude_termini_list)

        # Test with list parameters
        include_list_input = ["protein", "backbone"]
        exclude_list_input = ["resname SOL", "resname HOH"]

        include_list, exclude_list, _, _ = mda_TopologyAdapter._normalize_parameters(
            ensemble, include_selection=include_list_input, exclude_selection=exclude_list_input
        )

        assert include_list == include_list_input
        assert exclude_list == exclude_list_input

        # Test mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            mda_TopologyAdapter._normalize_parameters(
                ensemble,
                include_selection=["protein"],  # Wrong length
            )

    def test_apply_selection_pipeline(self, bpti_universe):
        """Test _apply_selection_pipeline method"""
        # Test basic selection
        selected_atoms, chains = mda_TopologyAdapter._apply_selection_pipeline(
            bpti_universe, "protein"
        )

        assert isinstance(selected_atoms, mda.AtomGroup)
        assert len(selected_atoms) > 0
        assert isinstance(chains, dict)
        assert len(chains) > 0

        # Test with exclude selection
        selected_atoms_excl, chains_excl = mda_TopologyAdapter._apply_selection_pipeline(
            bpti_universe, "protein", "name CA"
        )

        assert len(selected_atoms_excl) < len(selected_atoms)

        # Test invalid selection
        with pytest.raises(ValueError, match="Invalid include selection"):
            mda_TopologyAdapter._apply_selection_pipeline(bpti_universe, "invalid_selection")

    def test_process_chain_residues(self, bpti_universe):
        """Test _process_chain_residues method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        chain_id = mda_TopologyAdapter._get_chain_id(protein_atoms[0])
        chain_atoms = [
            atom for atom in protein_atoms if mda_TopologyAdapter._get_chain_id(atom) == chain_id
        ]

        # Test without terminal exclusion
        sorted_residues, residue_mapping, included_resids = (
            mda_TopologyAdapter._process_chain_residues(
                bpti_universe, chain_id, chain_atoms, exclude_termini=False, renumber_residues=True
            )
        )

        assert isinstance(sorted_residues, list)
        assert len(sorted_residues) > 0
        assert isinstance(residue_mapping, dict)
        assert isinstance(included_resids, set)
        assert len(included_resids) == len(sorted_residues)

        # Test with terminal exclusion
        sorted_residues_excl, mapping_excl, resids_excl = (
            mda_TopologyAdapter._process_chain_residues(
                bpti_universe, chain_id, chain_atoms, exclude_termini=True, renumber_residues=True
            )
        )

        # Should have fewer residues when excluding termini
        assert len(sorted_residues_excl) <= len(sorted_residues)

        # Test without renumbering
        _, mapping_no_renum, _ = mda_TopologyAdapter._process_chain_residues(
            bpti_universe, chain_id, chain_atoms, exclude_termini=False, renumber_residues=False
        )

        # Mapping should be identity mapping
        for orig_resid, mapped_resid in mapping_no_renum.items():
            assert orig_resid == mapped_resid

    def test_create_topology_from_residues(self, sample_residues):
        """Test _create_topology_from_residues method"""
        chain_id = mda_TopologyAdapter._get_chain_id(sample_residues[0])
        residue_mapping = {i + 1: res.resid for i, res in enumerate(sample_residues)}

        # Test chain mode
        chain_topologies = mda_TopologyAdapter._create_topology_from_residues(
            chain_id, sample_residues, residue_mapping, mode="chain"
        )

        assert len(chain_topologies) == 1
        assert chain_topologies[0].chain == chain_id
        assert len(chain_topologies[0].residues) == len(sample_residues)

        # Test residue mode
        residue_topologies = mda_TopologyAdapter._create_topology_from_residues(
            chain_id, sample_residues, residue_mapping, mode="residue"
        )

        assert len(residue_topologies) == len(sample_residues)
        for topo in residue_topologies:
            assert topo.chain == chain_id
            assert len(topo.residues) == 1

    def test_mda_group_to_topology(self, bpti_universe):
        """Test _mda_group_to_topology method"""
        protein_atoms = bpti_universe.select_atoms("protein")

        # Test with single residue
        single_residue = protein_atoms.residues[0]
        single_res_group = single_residue.atoms

        topology_single = mda_TopologyAdapter._mda_group_to_topology(
            single_res_group, exclude_termini=False
        )
        assert len(topology_single.residues) == 1

        # Test with multiple residues (same chain)
        multi_residues = protein_atoms.residues[:3]
        multi_res_group = multi_residues.atoms

        topology_multi = mda_TopologyAdapter._mda_group_to_topology(
            multi_res_group, exclude_termini=False
        )
        assert len(topology_multi.residues) == 3

        # Test with ResidueGroup
        residue_group = protein_atoms.residues[:2]
        topology_resgroup = mda_TopologyAdapter._mda_group_to_topology(
            residue_group, exclude_termini=False
        )
        assert len(topology_resgroup.residues) == 2

    def test_create_mda_group_lookup_key(self, bpti_universe):
        """Test _create_mda_group_lookup_key method"""
        protein_atoms = bpti_universe.select_atoms("protein")

        # Test with AtomGroup
        atom_group = protein_atoms.residues[0].atoms
        lookup_key = mda_TopologyAdapter._create_mda_group_lookup_key(atom_group)

        assert isinstance(lookup_key, tuple)
        assert len(lookup_key) == 2
        assert isinstance(lookup_key[0], str)  # chain_id
        assert isinstance(lookup_key[1], frozenset)  # resids

        # Test with ResidueGroup
        residue_group = protein_atoms.residues[:2]
        lookup_key_res = mda_TopologyAdapter._create_mda_group_lookup_key(residue_group)

        assert isinstance(lookup_key_res, tuple)
        assert len(lookup_key_res[1]) == 2  # Two residues

        # Test with renumber mapping
        first_resid = next(iter(lookup_key[1]))  # Get first element from frozenset
        renumber_mapping = {(lookup_key[0], 1): first_resid}
        lookup_key_renum = mda_TopologyAdapter._create_mda_group_lookup_key(
            atom_group, renumber_mapping
        )
        assert lookup_key_renum is not None

    def test_build_chain_selection_string(self, bpti_universe):
        """Test _build_chain_selection_string method"""
        protein_atoms = bpti_universe.select_atoms("protein")
        chain_id = mda_TopologyAdapter._get_chain_id(protein_atoms[0])

        # Test without base selection
        selection_string, fallback_atoms = mda_TopologyAdapter._build_chain_selection_string(
            bpti_universe, chain_id
        )

        assert isinstance(selection_string, str)
        assert isinstance(fallback_atoms, mda.AtomGroup)

        # Test with base selection
        selection_with_base, _ = mda_TopologyAdapter._build_chain_selection_string(
            bpti_universe, chain_id, "protein"
        )

        assert "protein" in selection_with_base or selection_with_base == ""

    def test_build_renumbering_mapping(self, bpti_universe):
        """Test _build_renumbering_mapping method"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        assert isinstance(mapping, dict)

        # Keys should be (chain_id, new_resid) tuples
        for key in mapping.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)  # chain_id
            assert isinstance(key[1], (int, np.integer))  # new_resid

        # Values should be original resids (could be numpy integers)
        for value in mapping.values():
            assert isinstance(value, (int, np.integer))

    def test_get_mda_group_sort_key(self, bpti_universe):
        """Test _get_mda_group_sort_key method"""
        protein_atoms = bpti_universe.select_atoms("protein")

        # Test with single residue
        residue = protein_atoms.residues[0]
        sort_key = mda_TopologyAdapter._get_mda_group_sort_key(residue)

        assert isinstance(sort_key, tuple)
        assert len(sort_key) == 4
        assert isinstance(sort_key[0], int)  # chain_id length
        assert isinstance(sort_key[1], tuple)  # chain score
        assert isinstance(sort_key[2], float)  # avg residue
        assert isinstance(sort_key[3], int)  # negative length

        # Test with ResidueGroup
        residue_group = protein_atoms.residues[:3]
        sort_key_group = mda_TopologyAdapter._get_mda_group_sort_key(residue_group)
        assert isinstance(sort_key_group, tuple)

        # Test with AtomGroup
        atom_group = protein_atoms.residues[0].atoms
        sort_key_atoms = mda_TopologyAdapter._get_mda_group_sort_key(atom_group)
        assert isinstance(sort_key_atoms, tuple)

        # Test error cases
        with pytest.raises(ValueError, match="contains no residues"):
            empty_residue_group = bpti_universe.select_atoms("resname XXX").residues
            mda_TopologyAdapter._get_mda_group_sort_key(empty_residue_group)

    def test_validate_topology_containment(self, bpti_universe):
        """Test _validate_topology_containment method"""
        # Create a simple topology from the universe
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", exclude_termini=False
        )

        if topologies:
            test_topology = topologies[0]

            # This should not raise an error
            mda_TopologyAdapter._validate_topology_containment(
                test_topology, bpti_universe, exclude_termini=False
            )

            # Create an invalid topology (with non-existent residues)
            invalid_topology = TopologyFactory.from_single(
                chain=test_topology.chain,
                residue=9999,  # Non-existent residue
                fragment_sequence="A",
                fragment_name="invalid",
                peptide=False,
            )

            with pytest.raises(ValueError, match="contains residues .* that are not available"):
                mda_TopologyAdapter._validate_topology_containment(
                    invalid_topology, bpti_universe, exclude_termini=False
                )


class TestUtilityMethodsIntegration:
    """Integration tests for utility methods working together"""

    def test_from_mda_universe_uses_utilities(self, bpti_universe):
        """Test that from_mda_universe correctly uses utility methods"""
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )

        assert len(topologies) > 0
        for topo in topologies:
            assert isinstance(topo.chain, str)
            assert len(topo.residues) > 0

    def test_to_mda_group_uses_utilities(self, bpti_universe):
        """Test that to_mda_group correctly uses utility methods"""
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", exclude_termini=False
        )

        if topologies:
            subset = topologies[:3]  # Test with subset

            mda_group = mda_TopologyAdapter.to_mda_group(
                subset, bpti_universe, exclude_termini=False
            )

            assert isinstance(mda_group, (mda.ResidueGroup, mda.AtomGroup))
            assert len(mda_group) > 0
