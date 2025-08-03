from jaxent.src.interfaces.topology import (
    TopologyFactory,
)
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons


class TestTopologyComparisons:
    """Test advanced topology comparison methods"""

    def test_intersects_same_chain(self):
        """Test intersection with same chain"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4])
        topo2 = TopologyFactory.from_residues("A", [3, 4, 5, 6])

        assert PairwiseTopologyComparisons.intersects(topo1, topo2) is True
        assert PairwiseTopologyComparisons.intersects(topo2, topo1) is True

    def test_intersects_different_chain(self):
        """Test intersection with different chains"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3])
        topo2 = TopologyFactory.from_residues("B", [1, 2, 3])

        assert PairwiseTopologyComparisons.intersects(topo1, topo2) is False

    def test_intersects_no_overlap(self):
        """Test intersection with no overlap"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3])
        topo2 = TopologyFactory.from_residues("A", [4, 5, 6])

        assert PairwiseTopologyComparisons.intersects(topo1, topo2) is False

    def test_intersects_with_peptide_trim(self):
        """Test intersection considering peptide trimming"""
        # Create two peptides where full residues overlap but peptide residues don't
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=3)
        # Full: [10, 11, 12, 13, 14, 15], Peptide: [13, 14, 15]

        peptide2 = TopologyFactory.from_range("A", 13, 18, peptide=True, peptide_trim=4)
        # Full: [13, 14, 15, 16, 17, 18], Peptide: [17, 18]

        # With check_trim=True (default), peptide residues don't overlap
        assert PairwiseTopologyComparisons.intersects(peptide1, peptide2, check_trim=True) is False

        # With check_trim=False, full residues do overlap ([13, 14, 15])
        assert PairwiseTopologyComparisons.intersects(peptide1, peptide2, check_trim=False) is True

    def test_contains_topology_full_containment(self):
        """Test topology containment - full containment"""
        large_topo = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5, 6, 7, 8])
        small_topo = TopologyFactory.from_residues("A", [3, 4, 5])

        assert PairwiseTopologyComparisons.contains_topology(large_topo, small_topo) is True
        assert PairwiseTopologyComparisons.contains_topology(small_topo, large_topo) is False

    def test_contains_topology_with_peptide_trim(self):
        """Test topology containment with peptide trimming"""
        # Large peptide that contains a smaller region after trimming
        large_peptide = TopologyFactory.from_range("A", 10, 20, peptide=True, peptide_trim=2)
        # Full: [10-20], Peptide: [12-20]

        small_topo = TopologyFactory.from_residues("A", [11, 12, 13])

        # With check_trim=True, only checks if [11,12,13] ⊆ [12-20] → False (11 not in peptide)
        assert (
            PairwiseTopologyComparisons.contains_topology(
                large_peptide, small_topo, check_trim=True
            )
            is False
        )

        # With check_trim=False, checks if [11,12,13] ⊆ [10-20] → True
        assert (
            PairwiseTopologyComparisons.contains_topology(
                large_peptide, small_topo, check_trim=False
            )
            is True
        )

        # Small topology that fits within peptide residues
        small_peptide_topo = TopologyFactory.from_residues("A", [15, 16, 17])
        assert (
            PairwiseTopologyComparisons.contains_topology(
                large_peptide, small_peptide_topo, check_trim=True
            )
            is True
        )
        assert (
            PairwiseTopologyComparisons.contains_topology(
                large_peptide, small_peptide_topo, check_trim=False
            )
            is True
        )

    def test_contains_topology_partial_overlap(self):
        """Test topology containment - partial overlap"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4])
        topo2 = TopologyFactory.from_residues("A", [3, 4, 5, 6])

        assert PairwiseTopologyComparisons.contains_topology(topo1, topo2) is False
        assert PairwiseTopologyComparisons.contains_topology(topo2, topo1) is False

    def test_contains_topology_different_chains(self):
        """Test topology containment - different chains"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = TopologyFactory.from_residues("B", [2, 3])

        assert PairwiseTopologyComparisons.contains_topology(topo1, topo2) is False

    def test_is_subset_of(self):
        """Test subset relationship"""
        small_topo = TopologyFactory.from_residues("A", [2, 3, 4])
        large_topo = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])

        assert PairwiseTopologyComparisons.is_subset_of(small_topo, large_topo) is True
        assert PairwiseTopologyComparisons.is_subset_of(large_topo, small_topo) is False

        # Same topology should be subset of itself
        assert PairwiseTopologyComparisons.is_subset_of(small_topo, small_topo) is True

    def test_is_subset_of_with_peptide_trim(self):
        """Test subset relationship with peptide trimming"""
        peptide = TopologyFactory.from_range("A", 10, 20, peptide=True, peptide_trim=3)
        # Full: [10-20], Peptide: [13-20]

        # Subset that includes trimmed residues
        subset_with_trimmed = TopologyFactory.from_residues("A", [10, 11, 15])

        # With check_trim=True, [10,11,15] ⊄ [13-20] → False
        assert (
            PairwiseTopologyComparisons.is_subset_of(subset_with_trimmed, peptide, check_trim=True)
            is False
        )

        # With check_trim=False, [10,11,15] ⊆ [10-20] → True
        assert (
            PairwiseTopologyComparisons.is_subset_of(subset_with_trimmed, peptide, check_trim=False)
            is True
        )

        # Subset within peptide residues
        subset_peptide = TopologyFactory.from_residues("A", [15, 16, 17])
        assert (
            PairwiseTopologyComparisons.is_subset_of(subset_peptide, peptide, check_trim=True)
            is True
        )
        assert (
            PairwiseTopologyComparisons.is_subset_of(subset_peptide, peptide, check_trim=False)
            is True
        )

    def test_is_superset_of(self):
        """Test superset relationship"""
        small_topo = TopologyFactory.from_residues("A", [2, 3, 4])
        large_topo = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])

        assert PairwiseTopologyComparisons.is_superset_of(large_topo, small_topo) is True
        assert PairwiseTopologyComparisons.is_superset_of(small_topo, large_topo) is False

        # Same topology should be superset of itself
        assert PairwiseTopologyComparisons.is_superset_of(large_topo, large_topo) is True

    def test_get_overlap(self):
        """Test getting overlapping residues"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = TopologyFactory.from_residues("A", [3, 4, 5, 6, 7])

        overlap = PairwiseTopologyComparisons.get_overlap(topo1, topo2)
        assert overlap == [3, 4, 5]

        # Test no overlap
        topo3 = TopologyFactory.from_residues("A", [10, 11, 12])
        no_overlap = PairwiseTopologyComparisons.get_overlap(topo1, topo3)
        assert no_overlap == []

        # Test different chains
        topo4 = TopologyFactory.from_residues("B", [3, 4, 5])
        different_chain = PairwiseTopologyComparisons.get_overlap(topo1, topo4)
        assert different_chain == []

    def test_get_overlap_with_peptide_trim(self):
        """Test getting overlap with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = TopologyFactory.from_range("A", 13, 18, peptide=True, peptide_trim=3)
        # Full: [13-18], Peptide: [16-18]

        # With check_trim=True, overlap of peptide residues [12-15] ∩ [16-18] = []
        overlap_trimmed = PairwiseTopologyComparisons.get_overlap(
            peptide1, peptide2, check_trim=True
        )
        assert overlap_trimmed == []

        # With check_trim=False, overlap of full residues [10-15] ∩ [13-18] = [13,14,15]
        overlap_full = PairwiseTopologyComparisons.get_overlap(peptide1, peptide2, check_trim=False)
        assert overlap_full == [13, 14, 15]

    def test_get_difference(self):
        """Test getting residue differences"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = TopologyFactory.from_residues("A", [3, 4, 5, 6, 7])

        diff1 = PairwiseTopologyComparisons.get_difference(topo1, topo2)
        assert diff1 == [1, 2]

        diff2 = PairwiseTopologyComparisons.get_difference(topo2, topo1)
        assert diff2 == [6, 7]

        # Test no overlap (should return all residues)
        topo3 = TopologyFactory.from_residues("A", [10, 11, 12])
        diff3 = PairwiseTopologyComparisons.get_difference(topo1, topo3)
        assert diff3 == [1, 2, 3, 4, 5]

        # Test different chains (should return all residues)
        topo4 = TopologyFactory.from_residues("B", [1, 2, 3])
        diff4 = PairwiseTopologyComparisons.get_difference(topo1, topo4)
        assert diff4 == [1, 2, 3, 4, 5]

    def test_get_difference_with_peptide_trim(self):
        """Test getting difference with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        regular_topo = TopologyFactory.from_residues("A", [13, 14, 16])

        # With check_trim=True, difference of [12-15] - [13,14,16] = [12,15]
        diff_trimmed = PairwiseTopologyComparisons.get_difference(
            peptide1, regular_topo, check_trim=True
        )
        assert diff_trimmed == [12, 15]

        # With check_trim=False, difference of [10-15] - [13,14,16] = [10,11,12,15]
        diff_full = PairwiseTopologyComparisons.get_difference(
            peptide1, regular_topo, check_trim=False
        )
        assert diff_full == [10, 11, 12, 15]

    def test_is_adjacent_to(self):
        """Test adjacency checking"""
        topo1 = TopologyFactory.from_range("A", 10, 15)
        topo2 = TopologyFactory.from_range("A", 16, 20)  # Adjacent after topo1
        topo3 = TopologyFactory.from_range("A", 5, 9)  # Adjacent before topo1
        topo4 = TopologyFactory.from_range("A", 18, 25)  # Not adjacent

        assert PairwiseTopologyComparisons.is_adjacent_to(topo1, topo2) is True
        assert PairwiseTopologyComparisons.is_adjacent_to(topo1, topo3) is True
        assert PairwiseTopologyComparisons.is_adjacent_to(topo1, topo4) is False

        # Test non-contiguous topology
        scattered = TopologyFactory.from_residues("A", [1, 3, 5])
        adjacent_scattered = TopologyFactory.from_residues("A", [6, 7, 8])
        assert (
            PairwiseTopologyComparisons.is_adjacent_to(scattered, adjacent_scattered) is True
        )  # 5 + 1 = 6

    def test_is_adjacent_to_with_peptide_trim(self):
        """Test adjacency with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=3)
        # Full: [10-15], Peptide: [13-15]

        peptide2 = TopologyFactory.from_range("A", 16, 20, peptide=True, peptide_trim=2)
        # Full: [16-20], Peptide: [18-20]

        # With check_trim=True, peptide residues [13-15] and [18-20] are NOT adjacent
        assert (
            PairwiseTopologyComparisons.is_adjacent_to(peptide1, peptide2, check_trim=True) is False
        )

        # With check_trim=False, full residues [10-15] and [16-20] ARE adjacent
        assert (
            PairwiseTopologyComparisons.is_adjacent_to(peptide1, peptide2, check_trim=False) is True
        )

        # Create adjacent peptide residues
        peptide3 = TopologyFactory.from_range("A", 16, 20, peptide=True, peptide_trim=0)
        # Full: [16-20], Peptide: [16-20] (no trimming)

        assert (
            PairwiseTopologyComparisons.is_adjacent_to(peptide1, peptide3, check_trim=True) is True
        )  # 15 + 1 = 16

    def test_get_gap_to(self):
        """Test gap calculation between topologies"""
        topo1 = TopologyFactory.from_range("A", 10, 15)  # ends at 15
        topo2 = TopologyFactory.from_range("A", 16, 20)  # starts at 16 (adjacent)
        topo3 = TopologyFactory.from_range("A", 18, 25)  # starts at 18 (gap of 2)
        topo4 = TopologyFactory.from_range("A", 5, 8)  # ends at 8 (gap of 1)

        # Adjacent topologies
        assert PairwiseTopologyComparisons.get_gap_to(topo1, topo2) == 0
        assert PairwiseTopologyComparisons.get_gap_to(topo2, topo1) == 0

        # Gap of 2 (16, 17 are missing)
        assert PairwiseTopologyComparisons.get_gap_to(topo1, topo3) == 2
        assert PairwiseTopologyComparisons.get_gap_to(topo3, topo1) == 2

        # Gap of 1 (9 is missing)
        assert PairwiseTopologyComparisons.get_gap_to(topo1, topo4) == 1
        assert PairwiseTopologyComparisons.get_gap_to(topo4, topo1) == 1

        # Overlapping topologies
        overlapping = TopologyFactory.from_range("A", 12, 18)
        assert PairwiseTopologyComparisons.get_gap_to(topo1, overlapping) is None

        # Different chains
        different_chain = TopologyFactory.from_range("B", 16, 20)
        assert PairwiseTopologyComparisons.get_gap_to(topo1, different_chain) is None

    def test_get_gap_to_with_peptide_trim(self):
        """Test gap calculation with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = TopologyFactory.from_range("A", 18, 22, peptide=True, peptide_trim=1)
        # Full: [18-22], Peptide: [19-22]

        # With check_trim=True, gap between peptide residues [12-15] and [19-22]
        # Gap = 19 - 15 - 1 = 3 (residues 16, 17, 18)
        gap_trimmed = PairwiseTopologyComparisons.get_gap_to(peptide1, peptide2, check_trim=True)
        assert gap_trimmed == 3

        # With check_trim=False, gap between full residues [10-15] and [18-22]
        # Gap = 18 - 15 - 1 = 2 (residues 16, 17)
        gap_full = PairwiseTopologyComparisons.get_gap_to(peptide1, peptide2, check_trim=False)
        assert gap_full == 2

    def test_protein_domain_analysis(self):
        """Test real-world protein domain comparison scenario"""
        # Define some protein domains
        signal_peptide = TopologyFactory.from_range("A", 1, 25, fragment_name="signal")
        n_terminal = TopologyFactory.from_range("A", 26, 150, fragment_name="N_terminal")
        binding_domain = TopologyFactory.from_range("A", 100, 200, fragment_name="binding")
        c_terminal = TopologyFactory.from_range("A", 201, 350, fragment_name="C_terminal")

        # Test domain relationships
        assert PairwiseTopologyComparisons.is_adjacent_to(signal_peptide, n_terminal) is True
        assert PairwiseTopologyComparisons.is_adjacent_to(binding_domain, c_terminal) is True
        assert PairwiseTopologyComparisons.is_adjacent_to(signal_peptide, c_terminal) is False

        # Test domain overlaps
        assert PairwiseTopologyComparisons.intersects(n_terminal, binding_domain) is True
        overlap = PairwiseTopologyComparisons.get_overlap(n_terminal, binding_domain)
        assert overlap == list(range(100, 151))  # residues 100-150

        # Test containment
        assert PairwiseTopologyComparisons.contains_topology(binding_domain, n_terminal) is False
        assert (
            PairwiseTopologyComparisons.contains_topology(
                n_terminal, TopologyFactory.from_range("A", 120, 130)
            )
            is True
        )

    def test_active_site_analysis(self):
        """Test active site residue analysis"""
        # Enzyme with scattered active site
        catalytic_triad = TopologyFactory.from_residues(
            "A", [57, 102, 195], fragment_name="catalytic_triad"
        )
        oxyanion_hole = TopologyFactory.from_residues(
            "A", [58, 195, 196], fragment_name="oxyanion_hole"
        )
        substrate_binding = TopologyFactory.from_residues(
            "A", [89, 90, 156, 189, 190], fragment_name="substrate_binding"
        )

        # Check which residues are shared between functional sites
        triad_oxyanion_overlap = PairwiseTopologyComparisons.get_overlap(
            catalytic_triad, oxyanion_hole
        )
        assert triad_oxyanion_overlap == [195]  # Shared catalytic residue

        # Check if substrate binding site contains any catalytic residues
        assert PairwiseTopologyComparisons.intersects(catalytic_triad, substrate_binding) is False

        # Find residues unique to each site
        unique_to_triad = PairwiseTopologyComparisons.get_difference(catalytic_triad, oxyanion_hole)
        assert unique_to_triad == [57, 102]

    def test_multi_residue_queries(self):
        """Test querying multiple residues at once"""
        protein_region = TopologyFactory.from_residues("A", [10, 12, 14, 16, 18, 20, 22])

        # Check specific residues of interest
        query_residues = [10, 11, 12, 13, 14, 15, 16]
        containment_map = protein_region.contains_which_residue(query_residues)
        expected = {10: True, 11: False, 12: True, 13: False, 14: True, 15: False, 16: True}
        assert containment_map == expected

        # Check if all key residues are present
        key_residues = [10, 12, 14]
        assert protein_region.contains_all_residues(key_residues) is True

        # Check if any problematic residues are present
        problematic_residues = [11, 13, 15, 99]
        assert protein_region.contains_any_residues(problematic_residues) is False

    def test_topology_gap_analysis(self):
        """Test gap analysis for topology fragments"""
        # Simulate exons in a gene
        exon1 = TopologyFactory.from_range("A", 1, 50, fragment_name="exon1")
        exon2 = TopologyFactory.from_range("A", 101, 150, fragment_name="exon2")
        exon3 = TopologyFactory.from_range("A", 201, 300, fragment_name="exon3")

        # Calculate intron sizes
        intron1_size = PairwiseTopologyComparisons.get_gap_to(exon1, exon2)
        intron2_size = PairwiseTopologyComparisons.get_gap_to(exon2, exon3)

        assert intron1_size == 50  # residues 51-100 (50 residues)
        assert intron2_size == 50  # residues 151-200 (50 residues)

        # Check adjacent exons would have gap of 0
        adjacent_exon = TopologyFactory.from_range("A", 51, 100)
        assert PairwiseTopologyComparisons.get_gap_to(exon1, adjacent_exon) == 0

    def test_subset_superset_relationships(self):
        """Test subset/superset relationships in protein analysis"""
        # Full protein sequence
        full_protein = TopologyFactory.from_range("A", 1, 500, fragment_name="full_protein")

        # Various domains
        signal_peptide = TopologyFactory.from_range("A", 1, 25, fragment_name="signal")
        mature_protein = TopologyFactory.from_range("A", 26, 500, fragment_name="mature")
        binding_domain = TopologyFactory.from_range("A", 100, 200, fragment_name="binding")

        # Test hierarchical relationships
        assert PairwiseTopologyComparisons.is_subset_of(signal_peptide, full_protein) is True
        assert PairwiseTopologyComparisons.is_subset_of(mature_protein, full_protein) is True
        assert PairwiseTopologyComparisons.is_subset_of(binding_domain, full_protein) is True
        assert PairwiseTopologyComparisons.is_subset_of(binding_domain, mature_protein) is True

        # Test superset relationships
        assert PairwiseTopologyComparisons.is_superset_of(full_protein, signal_peptide) is True
        assert PairwiseTopologyComparisons.is_superset_of(full_protein, mature_protein) is True
        assert PairwiseTopologyComparisons.is_superset_of(mature_protein, binding_domain) is True

        # Signal peptide and mature protein should not overlap
        assert PairwiseTopologyComparisons.intersects(signal_peptide, mature_protein) is False
