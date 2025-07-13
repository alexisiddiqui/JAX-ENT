import os

import MDAnalysis as mda
import pytest

from jaxent.src.interfaces.topology import Partial_Topology


class TestPartialTopologyFromMDA:
    """Test extraction of Partial_Topology from MDAnalysis Universe"""

    @pytest.fixture
    def bpti_universe(self):
        """Load BPTI structure as an MDAnalysis Universe"""
        pdb_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        if not os.path.exists(pdb_path):
            pytest.skip(f"Test PDB file not found: {pdb_path}")
        return mda.Universe(pdb_path)

    def test_extract_by_chain_mode(self, bpti_universe):
        """Test extraction of topologies by chain mode"""
        # Extract topologies by chain
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", include_selection="protein"
        )

        # BPTI should have one chain
        assert len(topologies) == 1, "Should extract one topology for BPTI (single chain)"

        # Verify the chain topology properties
        chain_topo = topologies[0]
        assert chain_topo.chain is not None, "Chain identifier should be set"
        assert chain_topo.is_contiguous is True, "BPTI chain should be contiguous"
        assert chain_topo.length > 50, "BPTI should have 50+ residues"
        assert len(chain_topo.fragment_sequence) > 0, "Fragment sequence should be extracted"

        # Check fragment name
        assert "chain" in chain_topo.fragment_name.lower(), "Fragment name should contain 'chain'"

    def test_extract_by_residue_mode(self, bpti_universe):
        """Test extraction of topologies by residue mode"""
        # Extract topologies by residue
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # Check we get multiple topologies (one per residue)
        assert len(topologies) > 50, "Should extract 50+ topologies (one per residue)"

        # Test properties of individual residue topologies
        for topo in topologies[:5]:  # Check first few
            assert topo.chain is not None, "Chain identifier should be set"
            assert topo.length == 1, "Each topology should represent a single residue"
            assert len(topo.residues) == 1, "Should have exactly one residue"

            # Check fragment name includes residue info
            assert topo.fragment_name is not None, "Fragment name should be set"
            assert "_" in topo.fragment_name, "Fragment name should be formatted as chain_residue"

    def test_custom_selection(self, bpti_universe):
        """Test custom atom selection criteria"""
        # Extract only CA atoms from residues 10-20
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and name CA and resid 10-20",
            exclude_termini=False,
        )

        # Should only get topologies for residues 10-20
        assert 10 <= len(topologies) <= 11, "Should extract ~11 topologies (residues 10-20)"

        # Check residue numbers
        residue_ids = sorted([topo.residues[0] for topo in topologies])
        for i, resid in enumerate(range(1, len(residue_ids) + 1)):
            assert residue_ids[i] == resid, "Expected sequential residue numbering starting from 1"

    def test_exclude_selection(self, bpti_universe):
        """Test exclusion criteria"""
        # First get all residues
        all_topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein",
            exclude_termini=False,
        )

        # Then get with exclusion
        exclude_topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein",
            exclude_selection="resid 1-5",
            exclude_termini=False,
        )

        # Should have fewer topologies with exclusion
        assert len(exclude_topologies) < len(all_topologies), (
            "Exclusion should reduce topology count"
        )

        # Check no excluded residues are present
        for topo in exclude_topologies:
            assert topo.residues[0] > 5, (
                f"Should not include residues 1-5 but found {topo.residues}"
            )

    def test_custom_fragment_naming(self, bpti_universe):
        """Test custom fragment naming template"""
        # Use custom naming template
        template = "res_{chain}_{resname}{resid}"
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 1-5",
            fragment_name_template=template,
        )

        # Check custom naming
        for topo in topologies:
            assert topo.fragment_name.startswith("res_"), "Should use custom name template"
            assert "_" in topo.fragment_name, "Should contain chain and residue info"

    def test_exclude_termini(self, bpti_universe):
        """Test excluding termini from chains"""
        # Extract with and without terminal exclusion
        with_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )

        without_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )

        # Chain with termini excluded should be shorter
        assert with_termini[0].length > without_termini[0].length, (
            "Excluding termini should reduce length"
        )

        # Should be exactly 2 residues shorter (N and C termini)
        assert with_termini[0].length == without_termini[0].length + 2, (
            f"Should be exactly 2 residues shorter. "
            f"With termini: {with_termini[0].length}, without: {without_termini[0].length}"
        )

        # Check sequence content - without termini should be missing first and last amino acids
        if isinstance(with_termini[0].fragment_sequence, str) and isinstance(
            without_termini[0].fragment_sequence, str
        ):
            with_seq = with_termini[0].fragment_sequence
            without_seq = without_termini[0].fragment_sequence

            # Without termini sequence should be the middle portion of the full sequence
            assert without_seq == with_seq[1:-1], (
                f"Without termini sequence should be middle of full sequence.\n"
                f"Full: {with_seq}\n"
                f"Trimmed: {without_seq}\n"
                f"Expected: {with_seq[1:-1]}"
            )

        # Even with renumbering, we should see differences in residue start/end
        # because terminal exclusion removes residues from the beginning and end
        assert with_termini[0].residue_start < without_termini[0].residue_start, (
            f"First residue should be different even with renumbering. "
            f"With termini start: {with_termini[0].residue_start}, "
            f"without termini start: {without_termini[0].residue_start}"
        )
        assert with_termini[0].residue_end > without_termini[0].residue_end, (
            f"Last residue should be different even with renumbering. "
            f"With termini end: {with_termini[0].residue_end}, "
            f"without termini end: {without_termini[0].residue_end}"
        )

        # Test with renumber_residues=False to see actual residue number differences
        with_termini_no_renumber = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False, renumber_residues=False
        )

        without_termini_no_renumber = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True, renumber_residues=False
        )

        # With no renumbering, the differences should be even more pronounced
        assert (
            with_termini_no_renumber[0].residue_start < without_termini_no_renumber[0].residue_start
        ), (
            f"First residue should be different when not renumbering. "
            f"With termini start: {with_termini_no_renumber[0].residue_start}, "
            f"without termini start: {without_termini_no_renumber[0].residue_start}"
        )
        assert (
            with_termini_no_renumber[0].residue_end > without_termini_no_renumber[0].residue_end
        ), (
            f"Last residue should be different when not renumbering. "
            f"With termini end: {with_termini_no_renumber[0].residue_end}, "
            f"without termini end: {without_termini_no_renumber[0].residue_end}"
        )


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
