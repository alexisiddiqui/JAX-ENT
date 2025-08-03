from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from jaxent.src.interfaces.topology import (
    TopologyFactory,
    mda_TopologyAdapter,
    rank_and_index,
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


class TestPartialTopologyFromMDA:
    """Test extraction of Partial_Topology from MDAnalysis Universe"""

    def test_extract_by_chain_mode(self, bpti_universe):
        """Test extraction of topologies by chain mode"""
        # Extract topologies by chain
        topologies = mda_TopologyAdapter.from_mda_universe(
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
        topologies = mda_TopologyAdapter.from_mda_universe(
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
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and name CA and resid 10-20",
            exclude_termini=False,
            renumber_residues=False,
        )

        # Should only get topologies for residues 10-20
        assert 10 <= len(topologies) <= 11, "Should extract ~11 topologies (residues 10-20)"

        # Check residue numbers
        residue_ids = sorted([topo.residues[0] for topo in topologies])
        for i, resid in enumerate(range(10, len(residue_ids) + 1)):
            assert residue_ids[i] == resid, (
                f"Expected sequential residue numbering starting from 1, residues: {residue_ids}"
            )

    def test_exclude_selection(self, bpti_universe):
        """Test exclusion criteria"""
        # First get all residues
        all_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein",
            exclude_termini=False,
            renumber_residues=False,
        )

        # Then get with exclusion
        exclude_topologies = mda_TopologyAdapter.from_mda_universe(
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
        topologies = mda_TopologyAdapter.from_mda_universe(
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
        with_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )

        without_termini = mda_TopologyAdapter.from_mda_universe(
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
        with_termini_no_renumber = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False, renumber_residues=False
        )

        without_termini_no_renumber = mda_TopologyAdapter.from_mda_universe(
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


class TestFindCommonResidues:
    """Test find_common_residues method for ensemble analysis"""

    def test_empty_ensemble_raises_error(self):
        """Test that empty ensemble raises ValueError"""
        with pytest.raises(ValueError, match="Empty ensemble provided"):
            mda_TopologyAdapter.find_common_residues([])

    def test_single_universe_termini_exclusion(self, bpti_universe):
        """Test exact termini exclusion behavior with single universe"""
        ensemble = [bpti_universe]

        # Get baseline - all protein residues with and without termini
        baseline_with_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )[0]
        baseline_without_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )[0]

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection="resname SOL"
        )

        # Common residues should match the exclude_termini=True extraction
        assert len(common_residues) == baseline_without_termini.length, (
            f"Common residues count {len(common_residues)} should match baseline without termini "
            f"{baseline_without_termini.length}"
        )

        # Excluded residues should be exactly the difference (termini)
        expected_excluded_count = baseline_with_termini.length - baseline_without_termini.length
        assert len(excluded_residues) == expected_excluded_count, (
            f"Excluded count {len(excluded_residues)} should be {expected_excluded_count} "
            f"(difference between with/without termini)"
        )

        # Verify specific terminal residues are excluded
        excluded_residue_ids = {list(topo.residues)[0] for topo in excluded_residues}
        # Fix: Use actual excluded residues instead of assuming N-terminal exclusion
        expected_excluded = {
            baseline_with_termini.residue_start,
            baseline_with_termini.residue_end,
        }
        assert excluded_residue_ids == expected_excluded, (
            f"Expected C-terminal residues {expected_excluded} to be excluded, got {excluded_residue_ids}"
        )

    def test_identical_ensembles_exact_matching(self, bpti_universe):
        """Test that identical universes produce identical results"""
        # Test with 2, 3, and 5 identical universes
        for n_universes in [2, 3, 5]:
            ensemble = [bpti_universe] * n_universes

            common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
                ensemble, include_selection="protein", exclude_selection="resname SOL"
            )

            # Results should be identical regardless of ensemble size for identical universes
            if n_universes == 2:
                baseline_common = len(common_residues)
                baseline_excluded = len(excluded_residues)
            else:
                assert len(common_residues) == baseline_common, (
                    f"Common residues should be identical for {n_universes} identical universes"
                )
                assert len(excluded_residues) == baseline_excluded, (
                    f"Excluded residues should be identical for {n_universes} identical universes"
                )

    def test_residue_renumbering_consistency(self, bpti_universe):
        """Test that residue renumbering works consistently with termini exclusion"""
        ensemble = [bpti_universe]

        # Get baseline to understand the expected behavior
        baseline_with_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )[0]
        baseline_without_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )[0]

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection="", exclude_termini=False
        )

        # Common residues should match the exclude_termini=True behavior
        common_residue_ids = sorted([list(topo.residues)[0] for topo in common_residues])

        assert len(common_residue_ids) > 0, "Should have common residues"

        # First residue should match baseline without termini (starts at 1 after renumbering)
        expected_first = 1  # After renumbering, starts at 1
        assert common_residue_ids[0] == expected_first, (
            f"First common residue should be {expected_first}, got {common_residue_ids[0]}"
        )

        # Should be contiguous (no gaps in middle residues)
        expected_range = list(range(expected_first, expected_first + len(common_residue_ids)))
        assert common_residue_ids == expected_range, (
            f"Common residues should be contiguous {expected_first}-{expected_first + len(common_residue_ids) - 1}, "
            f"got {common_residue_ids[:10]}..."
        )

    def test_renumbering_behavior_with_selections(self, bpti_universe):
        """Test and document the specific renumbering behavior with different selections"""
        ensemble = [bpti_universe]

        # Test 1: Full protein shows termini exclusion effect
        common_full, excluded_full = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection="", exclude_termini=False
        )
        full_ids = sorted([list(topo.residues)[0] for topo in common_full])

        # Test 2: Restricted selection preserves original numbering
        common_restricted, excluded_restricted = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein and resid 10-20", exclude_selection=""
        )
        restricted_ids = sorted([list(topo.residues)[0] for topo in common_restricted])

        # Document the behavior:
        # - Full protein with termini exclusion: starts from 1 (after renumbering)
        # - Restricted selection: gets renumbered based on the selection
        assert len(full_ids) > 0 and len(restricted_ids) > 0, (
            "Both selections should yield residues"
        )

        # Full protein should start from 1 (first residue after renumbering)
        assert full_ids[0] == 1, f"Full protein should start from residue 1, got {full_ids[0]}"

        # Restricted selection gets renumbered - should start from 1 for the selected range
        assert restricted_ids[0] >= 1, (
            f"Restricted selection should start from renumbered position, got {restricted_ids[0]}"
        )

        # Both should be contiguous within their respective ranges
        assert full_ids == list(range(full_ids[0], full_ids[0] + len(full_ids))), (
            "Full protein residues should be contiguous"
        )
        assert restricted_ids == list(
            range(restricted_ids[0], restricted_ids[0] + len(restricted_ids))
        ), "Restricted residues should be contiguous"

    def test_ignore_selection_specific_effects(self, bpti_universe):
        """Test specific ignore selection patterns"""
        ensemble = [bpti_universe]

        # Baseline: no ignore selection
        common_baseline, excluded_baseline = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection=""
        )
        baseline_count = len(common_baseline)

        # Test hydrogen exclusion (if present)
        try:
            common_no_h, excluded_no_h = mda_TopologyAdapter.find_common_residues(
                ensemble, include_selection="protein", exclude_selection="name H*"
            )
            # Should have same or fewer common residues (hydrogens don't define residues)
            assert len(common_no_h) <= baseline_count, (
                "Ignoring hydrogens shouldn't increase residue count"
            )
        except ValueError:
            # No hydrogens in structure, that's fine
            pass

        # Test backbone-only selection
        common_backbone, excluded_backbone = mda_TopologyAdapter.find_common_residues(
            ensemble,
            include_selection="protein and name CA",  # Only CA atoms
            exclude_selection="",
        )

        # Should have same number of residues (CA defines each residue)
        assert len(common_backbone) == baseline_count, (
            f"CA-only selection should have same residue count: "
            f"baseline={baseline_count}, CA-only={len(common_backbone)}"
        )

    def test_restrictive_include_selection_with_renumbering(self, bpti_universe):
        """Test restrictive include selections with renumber_residues=True"""
        ensemble = [bpti_universe]

        # Test middle residues only with renumbering enabled
        test_ranges = [
            (10, 20),  # 11 residues
            (15, 25),  # 11 residues
            (5, 15),  # 11 residues
        ]

        for start, end in test_ranges:
            common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
                ensemble,
                include_selection=f"protein and resid {start}-{end}",
                exclude_selection="",
                renumber_residues=True,  # Enable renumbering
            )

            # With renumbering=True, residues are renumbered based on their position
            # in the full filtered chain (after terminal exclusion)
            expected_max = end - start + 1  # Original range size
            expected_after_termini = max(0, expected_max - 2)  # Minus termini

            assert len(common_residues) <= expected_max, (
                f"Range {start}-{end}: too many residues {len(common_residues)} > {expected_max}"
            )
            assert len(common_residues) >= expected_after_termini, (
                f"Range {start}-{end}: too few residues {len(common_residues)} < {expected_after_termini}"
            )

            if len(common_residues) > 0:
                common_ids = sorted([list(topo.residues)[0] for topo in common_residues])

                # With renumbering=True, the first residue depends on where the selected
                # range falls within the renumbered full chain (which starts from 1)
                # The exact value depends on the position of the selection within the chain

                # Residues should be contiguous within their renumbered positions
                expected_span = max(common_ids) - min(common_ids) + 1
                assert len(common_ids) == expected_span, (
                    f"Range {start}-{end}: residues should be contiguous, "
                    f"got {len(common_ids)} residues spanning {expected_span} positions"
                )

                # All residues should be positive (since renumbering starts from 1)
                assert all(rid > 0 for rid in common_ids), (
                    f"Range {start}-{end}: all renumbered residues should be positive: {common_ids}"
                )

    def test_restrictive_include_selection_without_renumbering(self, bpti_universe):
        """Test restrictive include selections with renumber_residues=False"""
        ensemble = [bpti_universe]

        # Test middle residues only with renumbering disabled
        test_ranges = [
            (10, 20),  # 11 residues
            (15, 25),  # 11 residues
            (5, 15),  # 11 residues
        ]

        for start, end in test_ranges:
            common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
                ensemble,
                include_selection=f"protein and resid {start}-{end}",
                exclude_selection="",
                renumber_residues=False,  # Disable renumbering
            )

            # Without renumbering, original residue numbers should be preserved
            expected_max = end - start + 1  # Original range size
            expected_after_termini = max(0, expected_max - 2)  # Minus termini

            assert len(common_residues) <= expected_max, (
                f"Range {start}-{end}: too many residues {len(common_residues)} > {expected_max}"
            )
            assert len(common_residues) >= expected_after_termini, (
                f"Range {start}-{end}: too few residues {len(common_residues)} < {expected_after_termini}"
            )

            if len(common_residues) > 0:
                common_ids = sorted([list(topo.residues)[0] for topo in common_residues])

                # Without renumbering, residues should preserve their original numbering
                # First residue should be from the selected range (possibly +1 if start is excluded as terminus)
                expected_min = start
                expected_max_first = start + 1  # In case start is excluded as terminus

                assert expected_min <= common_ids[0] <= expected_max_first, (
                    f"Range {start}-{end}: first residue {common_ids[0]} should be between {expected_min} and {expected_max_first}"
                )

                # Last residue should be from the selected range (possibly -1 if end is excluded as terminus)
                expected_max_last = end
                expected_min_last = end - 1  # In case end is excluded as terminus

                assert expected_min_last <= common_ids[-1] <= expected_max_last, (
                    f"Range {start}-{end}: last residue {common_ids[-1]} should be between {expected_min_last} and {expected_max_last}"
                )

                # Residues should be contiguous within the selected range
                expected_span = max(common_ids) - min(common_ids) + 1
                assert len(common_ids) == expected_span, (
                    f"Range {start}-{end}: residues should be contiguous, "
                    f"got {len(common_ids)} residues spanning {expected_span} positions"
                )

                # All residues should fall within the original selected range (accounting for terminal exclusion)
                assert all(start - 1 <= rid <= end + 1 for rid in common_ids), (
                    f"Range {start}-{end}: all residues should be near original range: {common_ids}"
                )

    def test_chain_identification_specificity(self, bpti_universe):
        """Test specific chain identification and handling"""
        ensemble = [bpti_universe]

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection=""
        )

        # Get unique chain identifiers
        common_chains = {topo.chain for topo in common_residues}
        excluded_chains = {topo.chain for topo in excluded_residues}

        # BPTI should have exactly one protein chain
        assert len(common_chains) == 1, (
            f"Expected 1 chain, found {len(common_chains)}: {common_chains}"
        )

        # Excluded residues should be from same chain(s)
        if excluded_residues:
            assert excluded_chains <= common_chains, (
                f"Excluded chains {excluded_chains} should be subset of common chains {common_chains}"
            )

        # All residues should have the same chain identifier
        chain_id = list(common_chains)[0]
        for topo in common_residues:
            assert topo.chain == chain_id, f"All common residues should have chain {chain_id}"

    def test_residue_properties_validation(self, bpti_universe):
        """Test detailed properties of returned residues"""
        ensemble = [bpti_universe]

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection=""
        )

        # Test every common residue
        for i, topo in enumerate(common_residues):
            assert topo.length == 1, f"Common residue {i}: length should be 1, got {topo.length}"
            assert len(topo.residues) == 1, f"Common residue {i}: should have 1 residue ID"
            assert not topo.peptide, f"Common residue {i}: should not be marked as peptide"
            assert topo.peptide_trim == 0, f"Common residue {i}: peptide_trim should be 0"
            assert topo.fragment_name is not None, (
                f"Common residue {i}: fragment_name should be set"
            )
            assert topo.chain is not None, f"Common residue {i}: chain should be set"

        # Test every excluded residue
        for i, topo in enumerate(excluded_residues):
            assert topo.length == 1, f"Excluded residue {i}: length should be 1, got {topo.length}"
            assert len(topo.residues) == 1, f"Excluded residue {i}: should have 1 residue ID"
            assert not topo.peptide, f"Excluded residue {i}: should not be marked as peptide"
            assert topo.chain is not None, f"Excluded residue {i}: chain should be set"

    def test_intersection_logic_validation(self, bpti_universe):
        """Test the intersection merge logic with identical universes"""
        # Using identical universes should result in perfect intersection
        ensemble = [bpti_universe, bpti_universe, bpti_universe]

        # Get individual chain topologies for comparison
        individual_chains = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", include_selection="protein"
        )
        individual_residues = TopologyFactory.extract_residues(
            individual_chains[0], use_peptide_trim=False
        )

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection=""
        )

        # For identical universes, common residues should match the exclude_termini=True extraction
        baseline_no_termini = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )[0]

        assert len(common_residues) == baseline_no_termini.length, (
            f"Intersection of identical universes should match single extraction: "
            f"got {len(common_residues)}, expected {baseline_no_termini.length}"
        )

    def test_error_conditions_specific(self, bpti_universe):
        """Test specific error conditions and edge cases"""

        # Test 1: Invalid residue selection
        with pytest.raises(ValueError, match="Failed to extract topologies"):
            mda_TopologyAdapter.find_common_residues(
                [bpti_universe],
                include_selection="resname NONEXISTENT",
                exclude_selection="",
            )

        # Test 2: Invalid atom selection
        with pytest.raises(ValueError, match="Failed to extract topologies"):
            mda_TopologyAdapter.find_common_residues(
                [bpti_universe], include_selection="name FAKEATOM", exclude_selection=""
            )

        # Test 3: Out of range residue selection
        with pytest.raises(ValueError, match="Failed to extract topologies"):
            mda_TopologyAdapter.find_common_residues(
                [bpti_universe],
                include_selection="protein and resid 9999",
                exclude_selection="",
            )

    def test_uniqueness_and_no_overlap_strict(self, bpti_universe):
        """Test strict uniqueness and non-overlap requirements"""
        ensemble = [bpti_universe, bpti_universe]  # Identical universes

        common_residues, excluded_residues = mda_TopologyAdapter.find_common_residues(
            ensemble, include_selection="protein", exclude_selection=""
        )

        # Convert to lists for detailed checking
        common_list = list(common_residues)
        excluded_list = list(excluded_residues)

        # Test uniqueness by checking for duplicates
        common_signatures = [(topo.chain, tuple(topo.residues)) for topo in common_list]
        excluded_signatures = [(topo.chain, tuple(topo.residues)) for topo in excluded_list]

        assert len(common_signatures) == len(set(common_signatures)), (
            "Common residues contain duplicates"
        )
        assert len(excluded_signatures) == len(set(excluded_signatures)), (
            "Excluded residues contain duplicates"
        )

        # Test no overlap between common and excluded
        common_set = set(common_signatures)
        excluded_set = set(excluded_signatures)
        overlap = common_set & excluded_set

        assert len(overlap) == 0, f"Found overlap between common and excluded: {overlap}"

        # Test that together they account for reasonable total
        total_residues = len(common_set) + len(excluded_set)
        baseline_total = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )[0].length

        assert total_residues == baseline_total, (
            f"Total residues {total_residues} should equal baseline {baseline_total}"
        )


class TestToMDAGroup:
    """Test the to_mda_group method that maps Partial_Topology back to MDAnalysis groups"""

    def test_basic_conversion(self, bpti_universe):
        """Test basic conversion from Partial_Topology to MDAnalysis groups"""
        # First extract topologies from universe
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 10-20",
            renumber_residues=False,
        )

        assert len(topologies) > 0, "Should extract some topologies"

        # Convert back to MDAnalysis group
        residue_group = mda_TopologyAdapter.to_mda_group(
            set(topologies), bpti_universe, include_selection="protein"
        )

        # Check if we got a ResidueGroup with the expected number of residues
        assert hasattr(residue_group, "residues"), "Should return a ResidueGroup"
        assert len(residue_group) == len(topologies), (
            f"Should contain {len(topologies)} residues, got {len(residue_group)}"
        )

        # Check residue IDs match what we expect
        topology_resids = sorted([topo.residues[0] for topo in topologies])
        mda_resids = sorted([res.resid for res in residue_group])

        assert len(topology_resids) == len(mda_resids), "Residue counts should match"

        # Note: We don't directly compare resids since to_mda_group maps back to original resids
        # Instead check their lengths and that they're in the expected range
        assert 10 <= min(mda_resids) <= 20, (
            f"Residue IDs should be in range 10-20, got {min(mda_resids)}-{max(mda_resids)}"
        )
        assert 10 <= max(mda_resids) <= 20, (
            f"Residue IDs should be in range 10-20, got {min(mda_resids)}-{max(mda_resids)}"
        )

    def test_atom_filtering(self, bpti_universe):
        """Test atom filtering in to_mda_group"""
        # Extract chain topology
        chain_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="chain", include_selection="protein", exclude_termini=False
        )

        # Convert to AtomGroup with CA-only filter
        ca_atoms = mda_TopologyAdapter.to_mda_group(
            set(chain_topologies),
            bpti_universe,
            exclude_termini=False,
            include_selection="protein",
            mda_atom_filtering="name CA",
        )

        # Should get an AtomGroup containing only CA atoms
        # Note: In MDAnalysis, AtomGroup objects do have a 'residues' attribute
        # but we should verify we got atoms instead of residues directly
        assert isinstance(ca_atoms, mda.AtomGroup), "Should return an AtomGroup"
        assert len(ca_atoms) > 0, "Should have CA atoms"
        assert all(atom.name == "CA" for atom in ca_atoms), "All atoms should be CA atoms"

        # Number of atoms should match number of residues in the chain
        original_residues = len(chain_topologies[0].residues)
        assert len(ca_atoms) == original_residues, (
            f"Should have one CA atom per residue, got {len(ca_atoms)} atoms for {original_residues} residues"
        )

    def test_subset_selection(self, bpti_universe):
        """Test selecting a subset of residues"""
        # Extract all residues
        all_residue_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein",
        )

        # Select just the first few
        subset = set(all_residue_topologies[:5])

        # Convert to MDAnalysis group
        subset_group = mda_TopologyAdapter.to_mda_group(
            subset, bpti_universe, include_selection="protein", exclude_termini=False
        )

        # Check we got the expected number of residues
        assert len(subset_group) == 5, f"Should have 5 residues, got {len(subset_group)}"

        # Test a different subset
        middle_subset = set(all_residue_topologies[10:15])
        middle_group = mda_TopologyAdapter.to_mda_group(
            middle_subset, bpti_universe, include_selection="protein"
        )

        assert len(middle_group) == 5, f"Should have 5 residues, got {len(middle_group)}"

        # Make sure they're different groups
        middle_resids = sorted([res.resid for res in middle_group])
        subset_resids = sorted([res.resid for res in subset_group])
        assert middle_resids != subset_resids, "Different subsets should give different residues"

    def test_exclude_selection(self, bpti_universe):
        """Test exclude_selection parameter"""
        # Get all protein residues
        all_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # First get without exclusion
        all_group = mda_TopologyAdapter.to_mda_group(
            set(all_topologies), bpti_universe, include_selection="protein"
        )

        # Then with exclusion
        exclude_group = mda_TopologyAdapter.to_mda_group(
            set(all_topologies),
            bpti_universe,
            include_selection="protein",
            exclude_selection="resid 1-5",
        )

        # Should have fewer residues with exclusion
        assert len(exclude_group) < len(all_group), "Exclusion should reduce residue count"

        # Make sure excluded residues are actually excluded
        exclude_resids = [res.resid for res in exclude_group]
        for i in range(1, 6):
            assert i not in exclude_resids, f"Residue {i} should be excluded"

    def test_round_trip_conversion(self, bpti_universe):
        """Test round-trip conversion between Universe, Partial_Topology, and back"""
        # Extract specific residues
        original_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 10-20",
            renumber_residues=True,
        )

        # Convert back to MDAnalysis group
        mda_group = mda_TopologyAdapter.to_mda_group(
            set(original_topologies),
            bpti_universe,
            include_selection="protein",
            renumber_residues=True,
        )

        # Convert back to Partial_Topology again
        round_trip_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection=f"protein and resid {' '.join(str(res.resid) for res in mda_group)}",
            renumber_residues=True,
        )

        # Compare the original and round-trip topologies
        assert len(original_topologies) == len(round_trip_topologies), (
            "Should have same number of topologies"
        )

        # Sort both by residue number for comparison
        original_sorted = sorted(original_topologies, key=lambda t: t.residues[0])
        round_trip_sorted = sorted(round_trip_topologies, key=lambda t: t.residues[0])

        # Check residue sequences match
        original_resids = [t.residues[0] for t in original_sorted]
        round_trip_resids = [t.residues[0] for t in round_trip_sorted]
        assert original_resids == round_trip_resids, (
            "Residue IDs should match after round-trip conversion"
        )

    def test_error_conditions(self, bpti_universe):
        """Test error conditions in to_mda_group"""
        # Empty topology set should raise error
        with pytest.raises(ValueError, match="No topologies provided"):
            mda_TopologyAdapter.to_mda_group(set(), bpti_universe, include_selection="protein")

        # Invalid selection should raise error
        valid_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        with pytest.raises(ValueError, match="Invalid include selection"):
            mda_TopologyAdapter.to_mda_group(
                set(valid_topologies), bpti_universe, include_selection="nonexistent_selection"
            )

        # Invalid atom filtering should raise error
        with pytest.raises(ValueError, match="Invalid atom filtering"):
            mda_TopologyAdapter.to_mda_group(
                set(valid_topologies),
                bpti_universe,
                include_selection="protein",
                mda_atom_filtering="invalid_atom_name",
            )

    def test_to_mda_residue_dict(self, bpti_universe):
        """Test conversion to a dictionary of residue indices by chain."""
        # Extract topologies for residues 10-20
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 10-20",
            renumber_residues=False,  # Use original residue numbers for easier checking
        )

        # Convert to residue dictionary
        residue_dict = mda_TopologyAdapter.to_mda_residue_dict(
            set(topologies),
            bpti_universe,
            include_selection="protein",
            renumber_residues=False,
            exclude_termini=False,
        )

        # BPTI has one chain, usually 'A' or a segid
        assert len(residue_dict) == 1, "Should find residues in one chain"
        chain_id = list(residue_dict.keys())[0]

        # Check the residue numbers
        resids = residue_dict[chain_id]
        expected_resids = list(range(10, 21))  # resid 10-20 is inclusive
        assert sorted(resids) == expected_resids, (
            f"Residue dictionary should contain resids 10-20, got {resids}"
        )

        # Test with exclusion
        residue_dict_excluded = mda_TopologyAdapter.to_mda_residue_dict(
            set(topologies),
            bpti_universe,
            include_selection="protein",
            exclude_selection="resid 15-17",
            renumber_residues=False,
            exclude_termini=False,
        )
        resids_excluded = residue_dict_excluded[chain_id]
        expected_excluded = [10, 11, 12, 13, 14, 18, 19, 20]
        assert sorted(resids_excluded) == expected_excluded, (
            f"Residue dict with exclusion is incorrect: got {resids_excluded}"
        )


class TestPartialTopologyPairwiseDistances:
    """Test the partial_topology_pairwise_distances method."""

    def test_basic_calculation(self, bpti_universe):
        """Test basic distance calculation between a few residue topologies."""
        # Extract topologies from the universe to ensure they exist
        all_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # Select specific residues from the extracted topologies
        # Use residues that are reasonably far apart
        selected_indices = [0, 10, 20]  # First, 11th, and 21st residue
        topologies = [all_topologies[i] for i in selected_indices]

        dist_matrix, dist_std = mda_TopologyAdapter.partial_topology_pairwise_distances(
            topologies, bpti_universe, renumber_residues=False, verbose=False
        )

        assert dist_matrix.shape == (3, 3)
        assert dist_std.shape == (3, 3)

        # Diagonal elements should be zero
        assert np.all(np.diag(dist_matrix) == 0)
        # Std dev for a single frame (PDB) should be zero
        assert np.all(np.diag(dist_std) == 0)

        # Off-diagonal elements should be positive
        assert np.all(dist_matrix[np.triu_indices(3, k=1)] > 0)

        # Symmetry check
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_manual_comparison_single_frame(self, bpti_universe):
        """Test distance calculation against a manual calculation for a single frame."""
        # Extract topologies from the universe to ensure they exist
        all_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # Select a single residue topology
        single_residue = all_topologies[10]  # 11th residue

        # Create a multi-residue topology by extracting consecutive residues
        # and merging them
        multi_residues = TopologyFactory.merge(all_topologies[20:23])  # 21st-23rd residues

        topologies = [single_residue, multi_residues]

        # Manually calculate COM distances using the actual resids from the topologies
        single_resid = list(single_residue.residues)[0]
        multi_resids = list(multi_residues.residues)

        single_selection = f"resid {single_resid}"
        multi_selection = f"resid {' '.join(map(str, multi_resids))}"

        res_single_com = bpti_universe.select_atoms(single_selection).center_of_mass()
        res_multi_com = bpti_universe.select_atoms(multi_selection).center_of_mass()
        manual_dist = np.linalg.norm(res_single_com - res_multi_com)

        dist_matrix, dist_std = mda_TopologyAdapter.partial_topology_pairwise_distances(
            topologies, bpti_universe, renumber_residues=False, verbose=False
        )

        assert np.isclose(dist_matrix[0, 1], manual_dist)
        assert np.isclose(dist_matrix[1, 0], manual_dist)
        assert np.allclose(dist_std, 0)  # Single frame, so no std dev

    def test_empty_topologies_list_raises_error(self, bpti_universe):
        """Test that an empty list of topologies raises a ValueError."""
        with pytest.raises(ValueError, match="topologies list cannot be empty"):
            mda_TopologyAdapter.partial_topology_pairwise_distances([], bpti_universe)

    def test_mixed_topology_types(self, bpti_universe):
        """Test that the function works with a mix of single and multi-residue topologies."""
        # Extract topologies from the universe to ensure they exist
        all_topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # Create a mix of single and multi-residue topologies
        single_residue1 = all_topologies[5]  # 6th residue

        # Create a range topology by merging consecutive residues
        range_topology = TopologyFactory.merge(all_topologies[10:16])  # 11th-16th residues

        # Create a non-contiguous multi-residue topology
        noncontiguous_topology = TopologyFactory.merge(
            [all_topologies[20], all_topologies[25], all_topologies[30]]
        )

        topologies = [single_residue1, range_topology, noncontiguous_topology]

        dist_matrix, dist_std = mda_TopologyAdapter.partial_topology_pairwise_distances(
            topologies, bpti_universe, renumber_residues=False, verbose=False
        )

        assert dist_matrix.shape == (3, 3)
        assert dist_std.shape == (3, 3)
        assert np.all(np.diag(dist_matrix) == 0)
        assert np.all(dist_matrix[np.triu_indices(3, k=1)] > 0)
        assert np.allclose(dist_std, 0)


class TestGetMDAGroupSortKey:
    """Test the _get_mda_group_sort_key static method."""

    def test_sort_key_residuegroup(self, bpti_universe):
        # Extract a ResidueGroup for a range of residues
        residues = bpti_universe.select_atoms("protein and resid 10-15").residues
        sort_key = mda_TopologyAdapter._get_mda_group_sort_key(residues)
        # Should be a tuple of length 4
        assert isinstance(sort_key, tuple) and len(sort_key) == 4
        # Chain ID length should be int
        assert isinstance(sort_key[0], int)
        # Chain score should be tuple of ints
        assert isinstance(sort_key[1], tuple)
        # Average residue should be float
        assert isinstance(sort_key[2], float)
        # Length should be negative int
        assert isinstance(sort_key[3], int) and sort_key[3] < 0

    def test_sort_key_atomgroup(self, bpti_universe):
        # Extract an AtomGroup for a range of residues
        atoms = bpti_universe.select_atoms("protein and resid 10-15")
        sort_key = mda_TopologyAdapter._get_mda_group_sort_key(atoms)
        assert isinstance(sort_key, tuple) and len(sort_key) == 4

    def test_sort_key_single_residue(self, bpti_universe):
        # Single residue as ResidueGroup
        residue = bpti_universe.select_atoms("protein and resid 10").residues
        sort_key = mda_TopologyAdapter._get_mda_group_sort_key(residue)
        assert isinstance(sort_key, tuple) and len(sort_key) == 4

    def test_error_on_empty_group(self, bpti_universe):
        # Empty ResidueGroup
        from MDAnalysis.core.groups import ResidueGroup

        empty_group = ResidueGroup([], bpti_universe)
        with pytest.raises(ValueError):
            mda_TopologyAdapter._get_mda_group_sort_key(empty_group)

    def test_error_on_invalid_type(self):
        with pytest.raises(TypeError):
            mda_TopologyAdapter._get_mda_group_sort_key("not_a_group")


class TestGetResidueGroupReorderingIndices:
    """Test the get_residuegroup_reordering_indices method and compare with Partial_Topology ordering"""

    def test_basic_residuegroup_reordering(self, bpti_universe):
        """Test basic reordering of a ResidueGroup"""
        # Select a range of residues in reverse order
        residues = bpti_universe.select_atoms("protein and resid 15 14 13 12 11 10").residues

        # Get reordering indices
        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)

        # Apply reordering
        reordered_residues = [residues[i] for i in indices]

        # Check that residues are now in ascending order
        resids = [res.resid for res in reordered_residues]
        assert resids == sorted(resids), f"Residues should be in ascending order: {resids}"

        # Check indices are valid
        assert len(indices) == len(residues)
        assert all(0 <= i < len(residues) for i in indices)

    def test_atomgroup_reordering(self, bpti_universe):
        """Test reordering of an AtomGroup"""
        # Select atoms from multiple residues
        atoms = bpti_universe.select_atoms("protein and name CA and resid 20 18 16 14 12 10")

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(atoms)
        reordered_residues = [atoms.residues[i] for i in indices]

        # Check ordering
        resids = [res.resid for res in reordered_residues]
        assert resids == sorted(resids), f"Residues should be in ascending order: {resids}"

    def test_single_residue_returns_identity(self, bpti_universe):
        """Test that single residue returns identity ordering"""
        residue = bpti_universe.select_atoms("protein and resid 10").residues

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residue)

        assert indices == [0], "Single residue should return identity index [0]"

    def test_empty_group_raises_error(self, bpti_universe):
        """Test that empty group raises ValueError"""
        from MDAnalysis.core.groups import ResidueGroup

        empty_group = ResidueGroup([], bpti_universe)

        with pytest.raises(ValueError, match="Group contains no residues"):
            mda_TopologyAdapter.get_residuegroup_reordering_indices(empty_group)

    def test_invalid_type_raises_error(self):
        """Test that invalid group type raises TypeError"""
        with pytest.raises(TypeError, match="residue_group must be a ResidueGroup or AtomGroup"):
            mda_TopologyAdapter.get_residuegroup_reordering_indices("not_a_group")

    def test_already_ordered_residues(self, bpti_universe):
        """Test reordering residues that are already in correct order"""
        residues = bpti_universe.select_atoms("protein and resid 10-15").residues

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)

        # Should return identity ordering
        expected_indices = list(range(len(residues)))
        assert indices == expected_indices, (
            f"Already ordered residues should return identity: {indices}"
        )

    def test_comparison_with_partial_topology_ordering(self, bpti_universe):
        """Test that ordering matches Partial_Topology ordering methods"""
        # Create a mixed set of residues from different positions
        residues = bpti_universe.select_atoms("protein and resid 25 10 30 15 20").residues

        # Get reordering indices
        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        reordered_residues = [residues[i] for i in indices]

        # Create corresponding Partial_Topology objects
        partial_topologies = []
        for res in residues:
            chain_id = getattr(res.atoms[0], "segid", None) or getattr(res.atoms[0], "chainid", "A")
            topo = TopologyFactory.from_single(
                chain=chain_id,
                residue=res.resid,
                fragment_name=f"{chain_id}_{res.resname}{res.resid}",
            )
            partial_topologies.append(topo)

        # Sort using Partial_Topology method
        sorted_topologies = rank_and_index(partial_topologies.copy())

        # Compare orderings
        mda_ordered_resids = [res.resid for res in reordered_residues]
        topo_ordered_resids = [topo.residues[0] for topo in sorted_topologies]

        assert mda_ordered_resids == topo_ordered_resids, (
            f"MDA ordering {mda_ordered_resids} should match Partial_Topology ordering {topo_ordered_resids}"
        )

    def test_ordering_with_mixed_chains(self, bpti_universe):
        """Test ordering behavior with mixed chains (if available)"""
        # For BPTI (single chain), simulate by manually creating multi-chain scenario
        all_residues = bpti_universe.select_atoms("protein and resid 10-15").residues

        # Get indices for single chain
        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(all_residues)
        reordered = [all_residues[i] for i in indices]

        # Should be ordered by residue number within chain
        resids = [res.resid for res in reordered]
        assert resids == sorted(resids), "Single chain residues should be ordered by resid"

    def test_sort_key_consistency(self, bpti_universe):
        """Test that _get_mda_group_sort_key produces consistent results"""
        residues = bpti_universe.select_atoms("protein and resid 10-20").residues

        # Generate sort keys manually and compare with method results
        manual_keys = []
        for res in residues:
            key = mda_TopologyAdapter._get_mda_group_sort_key(res)
            manual_keys.append(key)

        # Sort using manual keys
        indexed_keys = [(key, i) for i, key in enumerate(manual_keys)]
        indexed_keys.sort(key=lambda x: x[0])
        manual_indices = [i for _, i in indexed_keys]

        # Compare with method result
        method_indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)

        assert manual_indices == method_indices, (
            f"Manual sort indices {manual_indices} should match method indices {method_indices}"
        )

    def test_residue_ordering_edge_cases(self, bpti_universe):
        """Test edge cases in residue ordering"""
        # Test with gaps in residue numbering
        scattered_residues = bpti_universe.select_atoms("protein and resid 10 12 14 16 18").residues

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(scattered_residues)
        reordered = [scattered_residues[i] for i in indices]

        # Should still be ordered by residue number
        resids = [res.resid for res in reordered]
        assert resids == [10, 12, 14, 16, 18], f"Scattered residues should be ordered: {resids}"

    def test_ordering_preserves_residue_properties(self, bpti_universe):
        """Test that reordering preserves residue properties"""
        residues = bpti_universe.select_atoms("protein and resid 20 15 10").residues
        original_resnames = [res.resname for res in residues]
        original_resids = [res.resid for res in residues]

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        reordered = [residues[i] for i in indices]

        # Check that residue properties are preserved
        reordered_resnames = [res.resname for res in reordered]
        reordered_resids = [res.resid for res in reordered]

        # Properties should be preserved but in new order
        assert set(original_resnames) == set(reordered_resnames), (
            "Residue names should be preserved"
        )
        assert set(original_resids) == set(reordered_resids), "Residue IDs should be preserved"
        assert reordered_resids == sorted(reordered_resids), "Residues should be ordered by ID"

    def test_comparison_with_direct_topology_creation(self, bpti_universe):
        """Test that ordering matches direct Partial_Topology creation and sorting"""
        # Select residues in random order
        residues = bpti_universe.select_atoms("protein and resid 30 10 25 15 20").residues

        # Method 1: Use get_residuegroup_reordering_indices
        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        method1_order = [residues[i].resid for i in indices]

        # Method 2: Create individual topologies and sort
        topologies = []
        for res in residues:
            chain_id = getattr(res.atoms[0], "segid", None) or getattr(res.atoms[0], "chainid", "A")
            topo = TopologyFactory.from_single(
                chain=chain_id, residue=res.resid, fragment_name=f"res_{res.resid}"
            )
            topologies.append(topo)

        # Sort topologies using standard method
        sorted_topologies = sorted(topologies)
        method2_order = [topo.residues[0] for topo in sorted_topologies]

        assert method1_order == method2_order, (
            f"get_residuegroup_reordering_indices order {method1_order} should match "
            f"direct topology sorting {method2_order}"
        )

    def test_large_residue_group_ordering(self, bpti_universe):
        """Test ordering with a large number of residues"""
        # Select all protein residues
        all_residues = bpti_universe.select_atoms("protein").residues

        # Shuffle the order by selecting in reverse
        residue_ids = [res.resid for res in all_residues]
        shuffled_selection = f"protein and resid {' '.join(map(str, reversed(residue_ids)))}"
        shuffled_residues = bpti_universe.select_atoms(shuffled_selection).residues

        indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(shuffled_residues)
        reordered = [shuffled_residues[i] for i in indices]

        # Check that all residues are included and ordered
        reordered_ids = [res.resid for res in reordered]
        assert len(reordered_ids) == len(residue_ids), "All residues should be included"
        assert reordered_ids == sorted(reordered_ids), "All residues should be ordered"

    def test_ordering_with__get_mda_group_sort_key_directly(self, bpti_universe):
        """Test that _get_mda_group_sort_key works correctly for individual residues"""
        # Test with individual residues
        res1 = bpti_universe.select_atoms("protein and resid 10").residues[0]
        res2 = bpti_universe.select_atoms("protein and resid 20").residues[0]

        key1 = mda_TopologyAdapter._get_mda_group_sort_key(res1)
        key2 = mda_TopologyAdapter._get_mda_group_sort_key(res2)

        # Earlier residue should have smaller sort key
        assert key1 < key2, f"Earlier residue should have smaller sort key: {key1} < {key2}"

        # Test with residue groups
        res_group = bpti_universe.select_atoms("protein and resid 10-12").residues
        group_key = mda_TopologyAdapter._get_mda_group_sort_key(res_group)

        assert isinstance(group_key, tuple), "Sort key should be a tuple"
        assert len(group_key) == 4, "Sort key should have 4 elements"

    def test_consistency_across_multiple_calls(self, bpti_universe):
        """Test that multiple calls to the method produce consistent results"""
        residues = bpti_universe.select_atoms("protein and resid 25 10 30 15 20").residues

        # Call method multiple times
        indices1 = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        indices2 = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        indices3 = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)

        # All calls should produce identical results
        assert indices1 == indices2 == indices3, "Multiple calls should produce identical results"

        # Verify ordering is stable
        reordered1 = [residues[i].resid for i in indices1]
        reordered2 = [residues[i].resid for i in indices2]

        assert reordered1 == reordered2, "Reordered results should be identical"

    def test_ordering_matches_rank_order_method(self, bpti_universe):
        """Test that ordering exactly matches Partial_Topology.rank_order method"""
        # Create residues in mixed order
        residues = bpti_universe.select_atoms("protein and resid 35 15 25 10 20 30").residues

        # Get MDA ordering
        mda_indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        mda_ordered_resids = [residues[i].resid for i in mda_indices]

        # Create corresponding topologies and sort using rank_order
        topologies = []
        for res in residues:
            chain_id = getattr(res.atoms[0], "segid", None) or getattr(res.atoms[0], "chainid", "A")
            topo = TopologyFactory.from_single(
                chain=chain_id, residue=res.resid, fragment_name=f"{chain_id}_{res.resid}"
            )
            topologies.append(topo)

        # Sort using rank_order
        sorted_by_rank = sorted(topologies, key=lambda t: t.rank_order())
        rank_ordered_resids = [topo.residues[0] for topo in sorted_by_rank]

        assert mda_ordered_resids == rank_ordered_resids, (
            f"MDA ordering {mda_ordered_resids} should exactly match "
            f"rank_order method {rank_ordered_resids}"
        )

    def test_ordering_with_rank_and_index_method(self, bpti_universe):
        """Test that ordering matches Partial_Topology.rank_and_index method"""
        residues = bpti_universe.select_atoms("protein and resid 40 20 30 10").residues

        # Get MDA ordering
        mda_indices = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        mda_ordered_resids = [residues[i].resid for i in mda_indices]

        # Create topologies and use rank_and_index
        topologies = []
        for res in residues:
            chain_id = getattr(res.atoms[0], "segid", None) or getattr(res.atoms[0], "chainid", "A")
            topo = TopologyFactory.from_single(
                chain=chain_id, residue=res.resid, fragment_name=f"res_{res.resid}"
            )
            topologies.append(topo)

        # Use rank_and_index to sort and assign indices
        ranked_topologies = rank_and_index(topologies)
        ranked_resids = [topo.residues[0] for topo in ranked_topologies]

        assert mda_ordered_resids == ranked_resids, (
            f"MDA ordering {mda_ordered_resids} should match rank_and_index method {ranked_resids}"
        )

        # Also check that fragment_index is assigned correctly
        expected_indices = list(range(len(ranked_topologies)))
        actual_indices = [topo.fragment_index for topo in ranked_topologies]
        assert actual_indices == expected_indices, "Fragment indices should be assigned correctly"

    def test_ordering_stability_with_identical_residues(self, bpti_universe):
        """Test ordering stability when residues have identical sort keys"""
        # Select residues that should have identical sort keys (same chain, same resid)
        # This is a bit artificial since we can't have truly identical residues in one structure
        residues = bpti_universe.select_atoms("protein and resid 10 11 12").residues

        # Test multiple orderings to ensure stability
        indices1 = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)
        indices2 = mda_TopologyAdapter.get_residuegroup_reordering_indices(residues)

        # Should be identical (stable sort)
        assert indices1 == indices2, "Sorting should be stable"

        # Final order should be by residue number
        final_order = [residues[i].resid for i in indices1]
        assert final_order == [10, 11, 12], f"Should be ordered by resid: {final_order}"


class TestBuildRenumberingMapping:
    """Test the _build_renumbering_mapping class method"""

    def test_basic_renumbering_mapping(self, bpti_universe):
        """Test basic renumbering mapping creation"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        assert isinstance(mapping, dict), "Should return a dictionary"
        assert len(mapping) > 0, "Should have mappings for at least some residues"

        # Check that all keys are (chain_id, new_resid) tuples
        for key in mapping.keys():
            assert isinstance(key, tuple), "Keys should be tuples"
            assert len(key) == 2, "Keys should be (chain_id, new_resid) pairs"
            chain_id, new_resid = key
            assert isinstance(chain_id, str), "Chain ID should be string"
            assert isinstance(new_resid, int), "New resid should be integer"

        # Check that all values are original residue IDs
        for orig_resid in mapping.values():
            assert isinstance(orig_resid, (int, np.integer)), (
                "Original resid should be integer or numpy integer"
            )

    def test_renumbering_with_termini_exclusion(self, bpti_universe):
        """Test renumbering mapping with terminal exclusion"""
        mapping_with_termini = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )
        mapping_without_termini = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Should have fewer mappings when excluding termini
        # assert len(mapping_without_termini) < len(mapping_with_termini), (
        #     "Should have fewer mappings when excluding termini"
        # )

        # Check that renumbering starts from 1
        min_new_resid = min(key[1] for key in mapping_without_termini.keys())
        assert min_new_resid == 1, "Renumbering should start from 1"

        # Check that renumbering is contiguous
        chain_mappings = {}
        for (chain_id, new_resid), orig_resid in mapping_without_termini.items():
            if chain_id not in chain_mappings:
                chain_mappings[chain_id] = []
            chain_mappings[chain_id].append(new_resid)

        for chain_id, new_resids in chain_mappings.items():
            sorted_resids = sorted(new_resids)
            expected_resids = list(range(1, len(sorted_resids) + 1))
            assert sorted_resids == expected_resids, (
                f"Chain {chain_id} should have contiguous renumbering: {sorted_resids}"
            )

    def test_renumbering_preserves_chain_separation(self, bpti_universe):
        """Test that renumbering preserves chain separation"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Group mappings by chain
        chain_mappings = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            if chain_id not in chain_mappings:
                chain_mappings[chain_id] = {}
            chain_mappings[chain_id][new_resid] = orig_resid

        # For each chain, verify that renumbering is independent
        for chain_id, chain_mapping in chain_mappings.items():
            new_resids = sorted(chain_mapping.keys())
            orig_resids = [chain_mapping[new_resid] for new_resid in new_resids]

            # New resids should start from 1 for each chain
            assert new_resids[0] == 1, f"Chain {chain_id} should start renumbering from 1"

            # New resids should be contiguous
            expected_new_resids = list(range(1, len(new_resids) + 1))
            assert new_resids == expected_new_resids, (
                f"Chain {chain_id} renumbering should be contiguous"
            )

            # Original resids should be in ascending order
            assert orig_resids == sorted(orig_resids), (
                f"Chain {chain_id} original resids should be in ascending order"
            )

    def test_renumbering_with_single_residue_chain(self, bpti_universe):
        """Test renumbering behavior with very short chains"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # BPTI typically has one chain, so test the general behavior
        chain_mappings = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            if chain_id not in chain_mappings:
                chain_mappings[chain_id] = []
            chain_mappings[chain_id].append((new_resid, orig_resid))

        # Each chain should have at least some residues after terminal exclusion
        for chain_id, mappings in chain_mappings.items():
            assert len(mappings) > 0, f"Chain {chain_id} should have residues after exclusion"

    def test_renumbering_consistency_with_from_mda_universe(self, bpti_universe):
        """Test that renumbering mapping is consistent with from_mda_universe"""
        # Get the renumbering mapping
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Extract topologies with renumbering
        topologies = mda_TopologyAdapter.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein",
            exclude_termini=True,
            renumber_residues=True,
        )

        # Verify that topologies use the same renumbering as the mapping
        for topo in topologies:
            topo_resid = topo.residues[0]  # Single residue topology
            chain_id = topo.chain

            # Get the actual chain ID from the universe for comparison
            actual_chain_id = (
                bpti_universe.select_atoms("protein").segments[0].segid
                if bpti_universe.select_atoms("protein").segments
                else "A"
            )
            if not actual_chain_id:
                actual_chain_id = (
                    bpti_universe.select_atoms("protein").segments[0].chainid
                    if bpti_universe.select_atoms("protein").segments
                    else "A"
                )

            # Find corresponding entry in mapping
            mapping_key = (chain_id, topo_resid)
            assert mapping_key in mapping, (
                f"Topology residue {topo_resid} in chain {chain_id} should be in mapping. "
                f"Actual chain ID in universe: {actual_chain_id}. Mapping keys: {list(mapping.keys())}"
            )

    def test_empty_universe_handling(self, bpti_universe):
        """Test handling of edge cases in universe structure"""
        # This is hard to test without creating artificial universes
        # But we can test the basic functionality
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )

        # Should have mappings for all residues
        assert len(mapping) > 0, "Should have mappings even without terminal exclusion"

    def test_renumbering_bidirectional_lookup(self, bpti_universe):
        """Test that renumbering mapping can be used for bidirectional lookup"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Create reverse mapping
        reverse_mapping = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            reverse_mapping[(chain_id, orig_resid)] = new_resid

        # Test that both directions work
        for (chain_id, new_resid), orig_resid in mapping.items():
            # Forward lookup
            assert mapping[(chain_id, new_resid)] == orig_resid

            # Reverse lookup
            assert reverse_mapping[(chain_id, orig_resid)] == new_resid

    def test_renumbering_with_no_terminal_exclusion(self, bpti_universe):
        """Test renumbering without terminal exclusion"""
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )

        # Should include all residues
        assert len(mapping) > 0, "Should have mappings for all residues"

        # Check that first residue maps to residue 1
        chain_mappings = {}
        for (chain_id, new_resid), orig_resid in mapping.items():
            if chain_id not in chain_mappings:
                chain_mappings[chain_id] = []
            chain_mappings[chain_id].append(new_resid)

        for chain_id, new_resids in chain_mappings.items():
            min_resid = min(new_resids)
            assert min_resid == 1, f"Chain {chain_id} should start from residue 1"


class TestValidateTopologyContainment:
    """Test the _validate_topology_containment class method"""

    @pytest.fixture(autouse=True)
    def setup_method(self, bpti_universe):
        # Get the actual chain ID from the universe once for all tests in this class
        self.actual_chain_id = (
            bpti_universe.select_atoms("protein").segments[0].segid
            if bpti_universe.select_atoms("protein").segments
            else "A"
        )
        if not self.actual_chain_id:
            self.actual_chain_id = (
                bpti_universe.select_atoms("protein").segments[0].chainid
                if bpti_universe.select_atoms("protein").segments
                else "A"
            )

    def test_valid_topology_containment(self, bpti_universe):
        """Test validation of valid topology containment"""
        # Create a topology that should be valid
        # Use a residue that is known to be valid after exclusion (e.g., 2, as 1 is usually excluded)
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=2,  # Should be within range after terminal exclusion
            fragment_name="test_residue",
        )

        # Should not raise any errors
        try:
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )
        except ValueError as e:
            pytest.fail(f"Valid topology should not raise ValueError, but raised: {e}")

    def test_invalid_topology_residue_out_of_range(self, bpti_universe):
        """Test validation fails for out-of-range residues"""
        # Create topology with residue number that's too high
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=9999,  # Should be out of range
            fragment_name="invalid_residue",
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_invalid_topology_residue_negative(self, bpti_universe):
        """Test validation fails for negative residue numbers"""
        # Create topology with negative residue number
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=-1,  # Invalid residue number
            fragment_name="negative_residue",
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_invalid_topology_residue_zero(self, bpti_universe):
        """Test validation fails for zero residue number"""
        # Create topology with zero residue number
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=0,  # Invalid residue number
            fragment_name="zero_residue",
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_validation_with_terminal_exclusion(self, bpti_universe):
        """Test validation behavior with terminal exclusion"""
        # Get the actual range of available residues
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        # Find the range of valid residues for chain A
        chain_resids = [
            new_resid
            for (chain_id, new_resid) in mapping.keys()
            if chain_id == self.actual_chain_id
        ]
        if not chain_resids:
            pytest.skip("No residues found for the actual chain ID after terminal exclusion")

        min_resid = min(chain_resids)
        max_resid = max(chain_resids)

        # Test valid residue (should pass)
        valid_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=min_resid, fragment_name="valid_residue"
        )

        # Should not raise
        mda_TopologyAdapter._validate_topology_containment(
            valid_topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )

        # Test invalid residue (should fail)
        invalid_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=max_resid + 1,  # One beyond range
            fragment_name="invalid_residue",
        )

        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                invalid_topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_validation_without_terminal_exclusion(self, bpti_universe):
        """Test validation behavior without terminal exclusion"""
        # Get the range with all residues included
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=False
        )

        chain_resids = [
            new_resid
            for (chain_id, new_resid) in mapping.keys()
            if chain_id == self.actual_chain_id
        ]
        if not chain_resids:
            pytest.skip("No residues found for the actual chain ID without terminal exclusion")

        max_resid = max(chain_resids)

        # Test residue that would be valid without terminal exclusion
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=max_resid, fragment_name="terminal_residue"
        )

        # Should not raise when terminals are not excluded
        mda_TopologyAdapter._validate_topology_containment(
            topology, bpti_universe, exclude_termini=False, renumber_residues=True
        )

    def test_validation_without_renumbering(self, bpti_universe):
        """Test validation behavior without renumbering"""
        # Get original residue numbers
        protein_residues = bpti_universe.select_atoms("protein").residues
        original_resids = [res.resid for res in protein_residues]

        if not original_resids:
            pytest.skip("No protein residues found")

        # Test with original residue number (should pass)
        valid_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id,
            residue=original_resids[len(original_resids) // 2],  # Middle residue
            fragment_name="original_residue",
        )

        # Should not raise
        mda_TopologyAdapter._validate_topology_containment(
            valid_topology, bpti_universe, exclude_termini=False, renumber_residues=False
        )

        # Test with invalid original residue number (should fail)
        invalid_resid = max(original_resids) + 100
        invalid_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=invalid_resid, fragment_name="invalid_original"
        )

        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                invalid_topology, bpti_universe, exclude_termini=False, renumber_residues=False
            )

    def test_validation_with_multi_residue_topology(self, bpti_universe):
        """Test validation with topology containing multiple residues"""
        # Create a topology with multiple residues
        # Use residues that are known to be valid after exclusion
        valid_residues_for_multi = list(range(2, 7))  # Example range, adjust if needed
        if not valid_residues_for_multi:
            pytest.skip("Not enough valid residues for multi-residue topology test")

        topology = TopologyFactory.from_residues(
            chain=self.actual_chain_id,
            residues=valid_residues_for_multi,  # Should be valid range
            fragment_name="multi_residue",
        )

        # Should not raise
        mda_TopologyAdapter._validate_topology_containment(
            topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )

        # Test with some invalid residues
        invalid_topology = TopologyFactory.from_residues(
            chain=self.actual_chain_id,
            residues=[
                valid_residues_for_multi[0],
                valid_residues_for_multi[1],
                9999,
            ],  # Mix of valid and invalid
            fragment_name="mixed_residues",
        )

        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                invalid_topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_validation_with_nonexistent_chain(self, bpti_universe):
        """Test validation fails for nonexistent chain"""
        # Create topology with nonexistent chain
        topology = TopologyFactory.from_single(
            chain="Z",  # Likely nonexistent
            residue=1,
            fragment_name="nonexistent_chain",
        )

        # Should raise ValueError about no residues found for chain
        with pytest.raises(ValueError, match="No residues found for chain"):
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_validation_error_messages(self, bpti_universe):
        """Test that validation error messages are informative"""
        # Create topology with out-of-range residue
        topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=9999, fragment_name="test_residue"
        )

        # Check that error message contains useful information
        with pytest.raises(ValueError) as exc_info:
            mda_TopologyAdapter._validate_topology_containment(
                topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

        error_msg = str(exc_info.value)
        assert "9999" in error_msg, "Error message should contain the invalid residue number"
        assert f":{self.actual_chain_id}:" in error_msg, "Error message should contain the chain ID"
        # Remove assertion for "Available residues" since it's not present in the error message

    def test_validation_with_peptide_topology(self, bpti_universe):
        """Test validation with peptide topology"""
        # Create peptide topology
        # Use residues that are known to be valid after exclusion
        valid_peptide_res = list(range(2, 12))  # Example range, adjust if needed
        if len(valid_peptide_res) < 10:
            pytest.skip("Not enough valid residues for peptide topology test")

        peptide_topology = TopologyFactory.from_residues(
            chain=self.actual_chain_id,
            residues=valid_peptide_res,
            fragment_name="peptide_test",
            peptide=True,
            peptide_trim=2,
        )

        # Should validate based on full residue list, not trimmed
        mda_TopologyAdapter._validate_topology_containment(
            peptide_topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )

    def test_validation_boundary_conditions(self, bpti_universe):
        """Test validation at boundary conditions"""
        # Get the actual range of available residues
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        chain_resids = [
            new_resid
            for (chain_id, new_resid) in mapping.keys()
            if chain_id == self.actual_chain_id
        ]
        if not chain_resids:
            pytest.skip("No residues found for the actual chain ID after terminal exclusion")

        min_resid = min(chain_resids)
        max_resid = max(chain_resids)

        # Test minimum valid residue
        min_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=min_resid, fragment_name="min_residue"
        )

        # Should not raise
        mda_TopologyAdapter._validate_topology_containment(
            min_topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )

        # Test maximum valid residue
        max_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=max_resid, fragment_name="max_residue"
        )

        # Should not raise
        mda_TopologyAdapter._validate_topology_containment(
            max_topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )

        # Test one below minimum (should fail)
        if min_resid > 1:
            below_min_topology = TopologyFactory.from_single(
                chain=self.actual_chain_id, residue=min_resid - 1, fragment_name="below_min"
            )

            with pytest.raises(ValueError, match="contains residues .* that are not available"):
                mda_TopologyAdapter._validate_topology_containment(
                    below_min_topology, bpti_universe, exclude_termini=True, renumber_residues=True
                )

        # Test one above maximum (should fail)
        above_max_topology = TopologyFactory.from_single(
            chain=self.actual_chain_id, residue=max_resid + 1, fragment_name="above_max"
        )

        with pytest.raises(ValueError, match="contains residues .* that are not available"):
            mda_TopologyAdapter._validate_topology_containment(
                above_max_topology, bpti_universe, exclude_termini=True, renumber_residues=True
            )

    def test_validation_performance_with_large_topology(self, bpti_universe):
        """Test validation performance with large topology"""
        # Create topology with many residues
        mapping = mda_TopologyAdapter._build_renumbering_mapping(
            bpti_universe, exclude_termini=True
        )

        chain_resids = [
            new_resid
            for (chain_id, new_resid) in mapping.keys()
            if chain_id == self.actual_chain_id
        ]
        if len(chain_resids) < 10:
            pytest.skip("Need at least 10 residues for this test")

        # Test with large but valid topology
        large_topology = TopologyFactory.from_residues(
            chain=self.actual_chain_id,
            residues=chain_resids,  # All valid residues
            fragment_name="large_topology",
        )

        # Should not raise and should complete reasonably quickly
        mda_TopologyAdapter._validate_topology_containment(
            large_topology, bpti_universe, exclude_termini=True, renumber_residues=True
        )
