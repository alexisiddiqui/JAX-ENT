import os

import MDAnalysis as mda
import pytest

from jaxent.src.interfaces.topology import Partial_Topology


@pytest.fixture
def bpti_universe():
    """Load BPTI structure as an MDAnalysis Universe"""
    pdb_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    if not os.path.exists(pdb_path):
        pytest.skip(f"Test PDB file not found: {pdb_path}")
    return mda.Universe(pdb_path)


class TestPartialTopologyFromMDA:
    """Test extraction of Partial_Topology from MDAnalysis Universe"""

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
        for i, resid in enumerate(range(10, len(residue_ids) + 1)):
            assert residue_ids[i] == resid, (
                f"Expected sequential residue numbering starting from 1, residues: {residue_ids}"
            )

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


class TestFindCommonResidues:
    """Test find_common_residues method for ensemble analysis"""

    def test_empty_ensemble_raises_error(self):
        """Test that empty ensemble raises ValueError"""
        with pytest.raises(ValueError, match="Empty ensemble provided"):
            Partial_Topology.find_common_residues([])

    def test_single_universe_termini_exclusion(self, bpti_universe):
        """Test exact termini exclusion behavior with single universe"""
        ensemble = [bpti_universe]

        # Get baseline - all protein residues with and without termini
        baseline_with_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )[0]
        baseline_without_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )[0]

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection="resname SOL"
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
        assert 1 in excluded_residue_ids, "N-terminal residue (1) should be excluded"
        assert baseline_with_termini.residue_end in excluded_residue_ids, (
            f"C-terminal residue ({baseline_with_termini.residue_end}) should be excluded"
        )

    def test_identical_ensembles_exact_matching(self, bpti_universe):
        """Test that identical universes produce identical results"""
        # Test with 2, 3, and 5 identical universes
        for n_universes in [2, 3, 5]:
            ensemble = [bpti_universe] * n_universes

            common_residues, excluded_residues = Partial_Topology.find_common_residues(
                ensemble, include_mda_selection="protein", ignore_mda_selection="resname SOL"
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
        baseline_with_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=False
        )[0]
        baseline_without_termini = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", exclude_termini=True
        )[0]

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
        )

        # Common residues should match the exclude_termini=True behavior
        common_residue_ids = sorted([list(topo.residues)[0] for topo in common_residues])

        assert len(common_residue_ids) > 0, "Should have common residues"

        # First residue should match baseline without termini (likely 2, not 1)
        expected_first = baseline_without_termini.residue_start
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
        common_full, excluded_full = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
        )
        full_ids = sorted([list(topo.residues)[0] for topo in common_full])

        # Test 2: Restricted selection preserves original numbering
        common_restricted, excluded_restricted = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein and resid 10-20", ignore_mda_selection=""
        )
        restricted_ids = sorted([list(topo.residues)[0] for topo in common_restricted])

        # Document the behavior:
        # - Full protein with termini exclusion: starts from 2 (not 1)
        # - Restricted selection: preserves original position numbers
        assert len(full_ids) > 0 and len(restricted_ids) > 0, (
            "Both selections should yield residues"
        )

        # Full protein should start from 2 (first residue after N-terminus exclusion)
        assert full_ids[0] == 2, f"Full protein should start from residue 2, got {full_ids[0]}"

        # Restricted selection should preserve original numbering in the 10-20 range
        assert 10 <= restricted_ids[0] <= 11, (
            f"Restricted selection should start around residue 10-11, got {restricted_ids[0]}"
        )
        assert all(10 <= rid <= 20 for rid in restricted_ids), (
            f"All restricted residues should be in range 10-20, got {restricted_ids}"
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
        common_baseline, excluded_baseline = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
        )
        baseline_count = len(common_baseline)

        # Test hydrogen exclusion (if present)
        try:
            common_no_h, excluded_no_h = Partial_Topology.find_common_residues(
                ensemble, include_mda_selection="protein", ignore_mda_selection="name H*"
            )
            # Should have same or fewer common residues (hydrogens don't define residues)
            assert len(common_no_h) <= baseline_count, (
                "Ignoring hydrogens shouldn't increase residue count"
            )
        except ValueError:
            # No hydrogens in structure, that's fine
            pass

        # Test backbone-only selection
        common_backbone, excluded_backbone = Partial_Topology.find_common_residues(
            ensemble,
            include_mda_selection="protein and name CA",  # Only CA atoms
            ignore_mda_selection="",
        )

        # Should have same number of residues (CA defines each residue)
        assert len(common_backbone) == baseline_count, (
            f"CA-only selection should have same residue count: "
            f"baseline={baseline_count}, CA-only={len(common_backbone)}"
        )

    def test_restrictive_include_selection(self, bpti_universe):
        """Test restrictive include selections preserve original residue numbering"""
        ensemble = [bpti_universe]

        # Test middle residues only
        test_ranges = [
            (10, 20),  # 11 residues
            (15, 25),  # 11 residues
            (5, 15),  # 11 residues
        ]

        for start, end in test_ranges:
            common_residues, excluded_residues = Partial_Topology.find_common_residues(
                ensemble,
                include_mda_selection=f"protein and resid {start}-{end}",
                ignore_mda_selection="",
            )

            # After renumbering and terminal exclusion, should have reasonable count
            expected_max = end - start + 1  # Original range size
            expected_after_termini = max(0, expected_max - 2)  # Minus termini

            assert len(common_residues) <= expected_max, (
                f"Range {start}-{end}: too many residues {len(common_residues)} > {expected_max}"
            )
            assert len(common_residues) >= expected_after_termini, (
                f"Range {start}-{end}: too few residues {len(common_residues)} < {expected_after_termini}"
            )

            # Residues should preserve their original numbering from the selection
            # The renumbering is based on full protein, so "resid 10-20" keeps positions 10-20
            if len(common_residues) > 0:
                common_ids = sorted([list(topo.residues)[0] for topo in common_residues])

                # First residue should be from the selected range (possibly +1 if terminus excluded)
                # For "resid 10-20" with exclude_termini=True, first residue could be 11 (if 10 is excluded as terminus)
                expected_min = start
                expected_max_first = start + 1  # In case start is excluded as terminus

                assert expected_min <= common_ids[0] <= expected_max_first, (
                    f"Range {start}-{end}: first residue {common_ids[0]} should be between {expected_min} and {expected_max_first}"
                )

                # Residues should be contiguous within the selected range
                expected_span = max(common_ids) - min(common_ids) + 1
                assert len(common_ids) == expected_span, (
                    f"Range {start}-{end}: residues should be contiguous, "
                    f"got {len(common_ids)} residues spanning {expected_span} positions"
                )

    def test_chain_identification_specificity(self, bpti_universe):
        """Test specific chain identification and handling"""
        ensemble = [bpti_universe]

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
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

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
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
        individual_chains = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", include_selection="protein"
        )
        individual_residues = individual_chains[0].extract_residues(use_peptide_trim=False)

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
        )

        # For identical universes, common residues should match the exclude_termini=True extraction
        baseline_no_termini = Partial_Topology.from_mda_universe(
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
            Partial_Topology.find_common_residues(
                [bpti_universe],
                include_mda_selection="resname NONEXISTENT",
                ignore_mda_selection="",
            )

        # Test 2: Invalid atom selection
        with pytest.raises(ValueError, match="Failed to extract topologies"):
            Partial_Topology.find_common_residues(
                [bpti_universe], include_mda_selection="name FAKEATOM", ignore_mda_selection=""
            )

        # Test 3: Out of range residue selection
        with pytest.raises(ValueError, match="Failed to extract topologies"):
            Partial_Topology.find_common_residues(
                [bpti_universe],
                include_mda_selection="protein and resid 9999",
                ignore_mda_selection="",
            )

    def test_uniqueness_and_no_overlap_strict(self, bpti_universe):
        """Test strict uniqueness and non-overlap requirements"""
        ensemble = [bpti_universe, bpti_universe]  # Identical universes

        common_residues, excluded_residues = Partial_Topology.find_common_residues(
            ensemble, include_mda_selection="protein", ignore_mda_selection=""
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
        baseline_total = Partial_Topology.from_mda_universe(
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
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein and resid 10-20"
        )

        assert len(topologies) > 0, "Should extract some topologies"

        # Convert back to MDAnalysis group
        residue_group = Partial_Topology.to_mda_group(
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
        chain_topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="chain", include_selection="protein"
        )

        # Convert to AtomGroup with CA-only filter
        ca_atoms = Partial_Topology.to_mda_group(
            set(chain_topologies),
            bpti_universe,
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
        all_residue_topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # Select just the first few
        subset = set(all_residue_topologies[:5])

        # Convert to MDAnalysis group
        subset_group = Partial_Topology.to_mda_group(
            subset, bpti_universe, include_selection="protein"
        )

        # Check we got the expected number of residues
        assert len(subset_group) == 5, f"Should have 5 residues, got {len(subset_group)}"

        # Test a different subset
        middle_subset = set(all_residue_topologies[10:15])
        middle_group = Partial_Topology.to_mda_group(
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
        all_topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        # First get without exclusion
        all_group = Partial_Topology.to_mda_group(
            set(all_topologies), bpti_universe, include_selection="protein"
        )

        # Then with exclusion
        exclude_group = Partial_Topology.to_mda_group(
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
        original_topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 10-20",
            renumber_residues=True,
        )

        # Convert back to MDAnalysis group
        mda_group = Partial_Topology.to_mda_group(
            set(original_topologies),
            bpti_universe,
            include_selection="protein",
            renumber_residues=True,
        )

        # Convert back to Partial_Topology again
        round_trip_topologies = Partial_Topology.from_mda_universe(
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
            Partial_Topology.to_mda_group(set(), bpti_universe, include_selection="protein")

        # Invalid selection should raise error
        valid_topologies = Partial_Topology.from_mda_universe(
            bpti_universe, mode="residue", include_selection="protein"
        )

        with pytest.raises(ValueError, match="Invalid include selection"):
            Partial_Topology.to_mda_group(
                set(valid_topologies), bpti_universe, include_selection="nonexistent_selection"
            )

        # Invalid atom filtering should raise error
        with pytest.raises(ValueError, match="Invalid atom filtering"):
            Partial_Topology.to_mda_group(
                set(valid_topologies),
                bpti_universe,
                include_selection="protein",
                mda_atom_filtering="invalid_atom_name",
            )

    def test_to_mda_residue_dict(self, bpti_universe):
        """Test conversion to a dictionary of residue indices by chain."""
        # Extract topologies for residues 10-20
        topologies = Partial_Topology.from_mda_universe(
            bpti_universe,
            mode="residue",
            include_selection="protein and resid 10-20",
            renumber_residues=False,  # Use original residue numbers for easier checking
        )

        # Convert to residue dictionary
        residue_dict = Partial_Topology.to_mda_residue_dict(
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
        residue_dict_excluded = Partial_Topology.to_mda_residue_dict(
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
