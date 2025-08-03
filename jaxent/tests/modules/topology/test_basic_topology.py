import pytest

# Import the class being tested
# from your_module import Partial_Topology
# For this example, assume the class is in the same file or properly imported
from interfaces.topology.pairwise import PairwiseTopologyComparisons

from jaxent.src.interfaces.topology import (
    TopologyFactory,
)


class TestPartialTopologyConstruction:
    """Test different construction methods"""

    def test_from_range_basic(self):
        """Test basic range construction"""
        topo = TopologyFactory.from_range("A", 1, 5, "MKLIV", "N_terminus")

        assert topo.chain == "A"
        assert topo.residues == [1, 2, 3, 4, 5]
        assert topo.fragment_sequence == "MKLIV"
        assert topo.fragment_name == "N_terminus"
        assert topo.residue_start == 1
        assert topo.residue_end == 5
        assert topo.length == 5
        assert topo.is_contiguous is True
        assert topo.peptide is False
        assert topo.peptide_residues == []

    def test_from_residues_contiguous(self):
        """Test construction from contiguous residue list"""
        topo = TopologyFactory.from_residues("B", [10, 11, 12, 13], "GLYK", "test_frag")

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12, 13]
        assert topo.is_contiguous is True
        assert topo.length == 4
        assert topo.residue_start == 10
        assert topo.residue_end == 13

    def test_from_residues_scattered(self):
        """Test construction from non-contiguous residue list"""
        topo = TopologyFactory.from_residues("C", [20, 22, 24, 26, 28], "ARGLY", "scattered")

        assert topo.chain == "C"
        assert topo.residues == [20, 22, 24, 26, 28]
        assert topo.is_contiguous is False
        assert topo.length == 5
        assert topo.residue_start == 20
        assert topo.residue_end == 28

    def test_from_single(self):
        """Test single residue construction"""
        topo = TopologyFactory.from_single("D", 100, "L", "single_lys")

        assert topo.chain == "D"
        assert topo.residues == [100]
        assert topo.length == 1
        assert topo.is_contiguous is True
        assert topo.residue_start == 100
        assert topo.residue_end == 100

    def test_residue_deduplication_and_sorting(self):
        """Test that residues are deduplicated and sorted"""
        topo = TopologyFactory.from_residues("E", [3, 1, 2, 3, 1], "MKL", "test")

        assert topo.residues == [1, 2, 3]
        assert topo.length == 3

    def test_empty_residues_raises_error(self):
        """Test that empty residue list raises ValueError"""
        with pytest.raises(ValueError, match="At least one residue must be specified"):
            TopologyFactory.from_residues("A", [], "SEQ", "test")


class TestPeptideBehavior:
    """Test peptide-specific functionality"""

    def test_peptide_creation_with_trim(self):
        """Test peptide creation with trimming"""
        topo = TopologyFactory.from_range(
            chain="A",
            start=10,
            end=20,
            fragment_sequence="AKLMQWERTYP",
            fragment_name="signal_peptide",
            peptide=True,
            peptide_trim=2,
        )

        assert topo.peptide is True
        assert topo.peptide_trim == 2
        assert topo.length == 11
        assert topo.peptide_residues == [12, 13, 14, 15, 16, 17, 18, 19, 20]
        assert len(topo.peptide_residues) == 9

    def test_peptide_no_trim_when_too_short(self):
        """Test that peptide trimming doesn't apply to short fragments"""
        topo = TopologyFactory.from_range(chain="A", start=1, end=2, peptide=True, peptide_trim=5)

        assert topo.peptide is True
        assert topo.peptide_trim == 5
        assert topo.length == 2
        assert topo.peptide_residues == []  # Too short to trim

    def test_non_peptide_no_trimming(self):
        """Test that non-peptides don't have peptide_residues"""
        topo = TopologyFactory.from_range(chain="A", start=1, end=10, peptide=False, peptide_trim=2)

        assert topo.peptide is False
        assert topo.peptide_residues == []

    def test_set_peptide_enable(self):
        """Test enabling peptide mode"""
        topo = TopologyFactory.from_range("A", 1, 5, peptide=False)

        assert topo.peptide is False
        assert topo.peptide_residues == []

        topo._set_peptide(True, trim=1)

        assert topo.peptide is True
        assert topo.peptide_trim == 1
        assert topo.peptide_residues == [2, 3, 4, 5]

    def test_set_peptide_disable(self):
        """Test disabling peptide mode"""
        topo = TopologyFactory.from_range("A", 1, 10, peptide=True, peptide_trim=2)

        assert topo.peptide is True
        assert len(topo.peptide_residues) == 8

        topo._set_peptide(False)

        assert topo.peptide is False
        assert topo.peptide_residues == []

    def test_set_peptide_change_trim(self):
        """Test changing peptide trim value"""
        topo = TopologyFactory.from_range("A", 1, 10, peptide=True, peptide_trim=2)

        initial_peptide_length = len(topo.peptide_residues)

        topo._set_peptide(True, trim=3)

        assert topo.peptide_trim == 3
        assert len(topo.peptide_residues) == initial_peptide_length - 1


class TestStringRepresentation:
    """Test string representations"""

    def test_str_single_residue(self):
        """Test string representation for single residue"""
        topo = TopologyFactory.from_single("A", 100, "L", "test", fragment_index=5)
        result = str(topo)

        assert "5:test:A:100" in result

    def test_str_contiguous_range(self):
        """Test string representation for contiguous range"""
        topo = TopologyFactory.from_range("B", 10, 15, "MKLIVQ", "helix")
        result = str(topo)

        assert "None:helix:B:10-15" in result

    def test_str_scattered_residues(self):
        """Test string representation for scattered residues"""
        topo = TopologyFactory.from_residues(
            "C", [1, 3, 5, 7, 9, 11, 13], fragment_name="scattered"
        )
        result = str(topo)

        # Should show first 3 and last 3 with ellipsis
        assert "scattered:C:[1,3,5...9,11,13]" in result

    def test_str_peptide_annotation(self):
        """Test peptide annotation in string representation"""
        topo = TopologyFactory.from_range(
            "A", 1, 10, "MKLIVQWERT", "peptide", peptide=True, peptide_trim=2
        )
        result = str(topo)

        assert "[peptide: 8 residues]" in result

    def test_repr_excludes_computed_fields(self):
        """Test that repr excludes computed fields"""
        topo = TopologyFactory.from_range("A", 1, 5, "MKLIV", "test")
        result = repr(topo)

        # Should include core fields
        assert "chain='A'" in result
        assert "fragment_name='test'" in result

        # Should not include computed fields
        assert "residue_start" not in result
        assert "length" not in result
        assert "is_contiguous" not in result


class TestTopologyOperations:
    """Test topology manipulation operations"""

    def test_get_residue_ranges_contiguous(self):
        """Test residue ranges for contiguous topology"""
        topo = TopologyFactory.from_range("A", 10, 15)
        ranges = topo.get_residue_ranges()

        assert ranges == [(10, 15)]

    def test_get_residue_ranges_scattered(self):
        """Test residue ranges for scattered topology"""
        topo = TopologyFactory.from_residues("A", [1, 2, 3, 7, 8, 15, 16, 17])
        ranges = topo.get_residue_ranges()

        assert ranges == [(1, 3), (7, 8), (15, 17)]

    def test_extract_residues_non_peptide(self):
        """Test extracting residues from non-peptide"""
        topo = TopologyFactory.from_range("A", 10, 12, peptide=False)
        extracted = TopologyFactory.extract_residues(topo, use_peptide_trim=False)

        assert len(extracted) == 3
        for i, res_topo in enumerate(extracted):
            assert res_topo.residues == [10 + i]
            assert res_topo.peptide is False

    def test_extract_residues_peptide_with_trim(self):
        """Test extracting residues from peptide with trimming"""
        topo = TopologyFactory.from_range("A", 1, 10, peptide=True, peptide_trim=2)
        extracted = TopologyFactory.extract_residues(topo, use_peptide_trim=True)

        # Should only get trimmed residues
        assert len(extracted) == 8  # 10 - 2 = 8
        assert extracted[0].residues == [3]  # First trimmed residue
        assert extracted[-1].residues == [10]  # Last residue

    def test_extract_residues_single(self):
        """Test extracting from single residue returns self-like"""
        topo = TopologyFactory.from_single("A", 100)
        extracted = TopologyFactory.extract_residues(topo)

        assert len(extracted) == 1
        assert extracted[0].residues == [100]

    def test_contains_residue_single(self):
        """Test residue containment check for single residue"""
        topo = TopologyFactory.from_residues("A", [1, 3, 5, 7])

        assert topo.contains_which_residue(1) is True
        assert topo.contains_which_residue(3) is True
        assert topo.contains_which_residue(2) is False
        assert topo.contains_which_residue(8) is False

    def test_contains_residue_multiple(self):
        """Test residue containment check for multiple residues"""
        topo = TopologyFactory.from_residues("A", [1, 3, 5, 7])

        result = topo.contains_which_residue([1, 2, 3, 4, 5])
        expected = {1: True, 2: False, 3: True, 4: False, 5: True}

        assert result == expected
        assert isinstance(result, dict)

    def test_contains_residue_with_peptide_trim(self):
        """Test residue containment with peptide trimming"""
        # Create a peptide that trims first 2 residues
        peptide = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full residues: [10, 11, 12, 13, 14, 15]
        # Peptide residues: [12, 13, 14, 15] (after trimming first 2)

        # With check_trim=True (default), should only check peptide_residues
        assert peptide.contains_which_residue(10, check_trim=True) is False  # Trimmed
        assert peptide.contains_which_residue(11, check_trim=True) is False  # Trimmed
        assert peptide.contains_which_residue(12, check_trim=True) is True  # Active
        assert peptide.contains_which_residue(15, check_trim=True) is True  # Active

        # With check_trim=False, should check all residues
        assert peptide.contains_which_residue(10, check_trim=False) is True  # In full residues
        assert peptide.contains_which_residue(11, check_trim=False) is True  # In full residues
        assert peptide.contains_which_residue(12, check_trim=False) is True  # In full residues
        assert peptide.contains_which_residue(15, check_trim=False) is True  # In full residues

    def test_contains_all_residues(self):
        """Test checking if topology contains all specified residues"""
        topo = TopologyFactory.from_residues("A", [1, 2, 3, 4, 5])

        assert topo.contains_all_residues([1, 3, 5]) is True
        assert topo.contains_all_residues([1, 2, 3, 4, 5]) is True
        assert topo.contains_all_residues([1, 6]) is False
        assert topo.contains_all_residues([6, 7, 8]) is False
        assert topo.contains_all_residues([]) is True  # Empty list

    def test_contains_all_residues_peptide_trim(self):
        """Test contains_all_residues with peptide trimming"""
        peptide = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Peptide residues: [12, 13, 14, 15]

        # Check with trimming
        assert (
            peptide.contains_all_residues([12, 13], check_trim=True) is True
        )  # In peptide residues
        assert peptide.contains_all_residues([10, 11], check_trim=True) is False  # Trimmed out
        assert peptide.contains_all_residues([10, 12], check_trim=True) is False  # 10 is trimmed

        # Check without trimming (default behavior)
        assert peptide.contains_all_residues([10, 11], check_trim=False) is True
        assert peptide.contains_all_residues([10, 12], check_trim=False) is True

    def test_contains_any_residues(self):
        """Test checking if topology contains any of specified residues"""
        topo = TopologyFactory.from_residues("A", [1, 3, 5, 7])

        assert topo.contains_any_residues([1, 2]) is True
        assert topo.contains_any_residues([2, 4, 6]) is False
        assert topo.contains_any_residues([7, 8, 9]) is True
        assert topo.contains_any_residues([]) is False  # Empty list

    def test_contains_any_residues_peptide_trim(self):
        """Test contains_any_residues with peptide trimming"""
        peptide = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Peptide residues: [12, 13, 14, 15]

        # Check with trimming
        assert peptide.contains_any_residues([9, 10, 11], check_trim=True) is False  # All trimmed
        assert (
            peptide.contains_any_residues([10, 12], check_trim=True) is True
        )  # 12 is in peptide residues
        assert peptide.contains_any_residues([16, 17], check_trim=True) is False  # Not in topology

        # Check without trimming (default behavior)
        assert (
            peptide.contains_any_residues([9, 10, 11], check_trim=False) is True
        )  # 10, 11 in full
        assert (
            peptide.contains_any_residues([16, 17], check_trim=False) is False
        )  # Still not in topology

    def test_union_same_chain(self):
        """Test union of topologies on same chain"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3], fragment_name="frag1")
        topo2 = TopologyFactory.from_residues("A", [4, 5, 6], fragment_name="frag2")

        union = TopologyFactory.union(topo1, topo2)

        assert union.chain == "A"
        assert union.residues == [1, 2, 3, 4, 5, 6]
        assert union.fragment_name == "frag1"  # Takes from first topology

    def test_union_with_peptide_trim(self):
        """Test union with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = TopologyFactory.from_range("A", 18, 22, peptide=True, peptide_trim=1)
        # Full: [18-22], Peptide: [19-22]

        # With check_trim=True, union based on peptide residues [12-15] ∪ [19-22]
        # Should create range [12-22] to encompass both
        union_trimmed = TopologyFactory.union(peptide1, peptide2, check_trim=True)
        expected_range = list(range(12, 23))  # [12, 13, ..., 22]
        assert union_trimmed.residues == expected_range

        # With check_trim=False, union based on full residues [10-15] ∪ [18-22]
        # Should create range [10-22] to encompass both
        union_full = TopologyFactory.union(peptide1, peptide2, check_trim=False)
        expected_full_range = list(range(10, 23))  # [10, 11, ..., 22]
        assert union_full.residues == expected_full_range

    def test_union_different_chain_raises_error(self):
        """Test that union with different chains raises error"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3])
        topo2 = TopologyFactory.from_residues("B", [1, 2, 3])

        with pytest.raises(ValueError, match="Cannot combine topologies with different chains"):
            TopologyFactory.union(topo1, topo2)

    def test_union_overlapping_residues(self):
        """Test union with overlapping residues"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3, 4])
        topo2 = TopologyFactory.from_residues("A", [3, 4, 5, 6])

        union = TopologyFactory.union(topo1, topo2)

        # Should create contiguous range to encompass both
        assert union.residues == [1, 2, 3, 4, 5, 6]


class TestBiophysicalExamples:
    """Test real-world biophysical use cases"""

    def test_signal_peptide(self):
        """Test signal peptide modeling"""
        signal = TopologyFactory.from_range(
            chain="A",
            start=1,
            end=25,
            fragment_sequence="MKLLIVLLAFGAILFVVPGCGASS",
            fragment_name="signal_peptide",
            peptide=True,
            peptide_trim=3,
        )

        assert signal.length == 25
        assert signal.peptide is True
        assert len(signal.peptide_residues) == 22  # 25 - 3
        assert signal.peptide_residues[0] == 4  # First after trim

    def test_active_site_scattered(self):
        """Test active site with scattered residues"""
        active_site = TopologyFactory.from_residues(
            chain="A",
            residues=[45, 78, 123, 156, 234],
            fragment_sequence=["H", "D", "S", "H", "E"],
            fragment_name="active_site",
            fragment_index=1,
        )

        assert active_site.is_contiguous is False
        assert active_site.length == 5
        assert active_site.peptide is False
        assert len(active_site.get_residue_ranges()) == 5  # All separate ranges

    def test_transmembrane_helix(self):
        """Test transmembrane helix modeling"""
        tm_helix = TopologyFactory.from_range(
            chain="A",
            start=89,
            end=112,
            fragment_sequence="LVVFGAILFVVPGCGASSLMKDT",
            fragment_name="TM_helix_1",
            fragment_index=2,
        )

        assert tm_helix.is_contiguous is True
        assert tm_helix.length == 24
        assert len(tm_helix.get_residue_ranges()) == 1

    def test_protein_domain_merging(self):
        """Test merging protein domains for real-world scenarios"""
        # Simulate protein with signal peptide + multiple domains
        signal_peptide = TopologyFactory.from_range(
            "A", 1, 25, fragment_name="signal_peptide", peptide=True, peptide_trim=3
        )

        n_domain = TopologyFactory.from_range("A", 26, 150, fragment_name="N_domain")
        binding_domain = TopologyFactory.from_range("A", 151, 280, fragment_name="binding_domain")
        c_domain = TopologyFactory.from_range("A", 281, 400, fragment_name="C_domain")

        # Merge mature protein (excluding signal)
        mature_protein = TopologyFactory.merge_contiguous(
            [n_domain, binding_domain, c_domain], merged_name="mature_protein"
        )

        assert mature_protein.residues == list(range(26, 401))
        assert mature_protein.fragment_name == "mature_protein"
        assert mature_protein.length == 375

        # Full protein including signal (with trimming to get mature sequence)
        full_protein_trimmed = TopologyFactory.merge(
            [signal_peptide, n_domain, binding_domain, c_domain],
            trim=True,
            merged_name="full_protein_active",
        )

        # Should exclude trimmed signal peptide residues (1-3)
        expected_residues = list(range(4, 26)) + list(range(26, 401))  # [4-25] + [26-400]
        assert full_protein_trimmed.residues == expected_residues

        # Full protein without trimming (includes all residues)
        full_protein_complete = TopologyFactory.merge(
            [signal_peptide, n_domain, binding_domain, c_domain],
            trim=False,
            merged_name="full_protein_complete",
        )

        assert full_protein_complete.residues == list(range(1, 401))
        assert full_protein_complete.peptide is False  # trim=False removes peptide flag

    def test_active_site_assembly(self):
        """Test assembling active site from scattered catalytic residues"""
        # Individual catalytic residues
        his57 = TopologyFactory.from_single("A", 57, fragment_sequence="H", fragment_name="His57")
        asp102 = TopologyFactory.from_single(
            "A", 102, fragment_sequence="D", fragment_name="Asp102"
        )
        ser195 = TopologyFactory.from_single(
            "A", 195, fragment_sequence="S", fragment_name="Ser195"
        )

        # Assemble catalytic triad
        catalytic_triad = TopologyFactory.merge(
            [his57, asp102, ser195], merged_name="catalytic_triad", merged_sequence=["H", "D", "S"]
        )

        assert catalytic_triad.residues == [57, 102, 195]
        assert catalytic_triad.fragment_name == "catalytic_triad"
        assert catalytic_triad.fragment_sequence == ["H", "D", "S"]
        assert catalytic_triad.is_contiguous is False

        # Check relationships with substrate binding site
        substrate_site = TopologyFactory.from_residues(
            "A", [189, 190, 214, 226], fragment_name="substrate_binding"
        )

        assert not PairwiseTopologyComparisons.intersects(
            catalytic_triad, substrate_site
        )  # Should be separate

        # Merge into complete active site
        complete_active_site = TopologyFactory.merge(
            [catalytic_triad, substrate_site], merged_name="complete_active_site"
        )

        expected_active_residues = [57, 102, 189, 190, 195, 214, 226]
        assert complete_active_site.residues == expected_active_residues


@pytest.fixture
def sample_topology():
    """Fixture providing a sample topology for testing"""
    return TopologyFactory.from_range(
        chain="A",
        start=10,
        end=20,
        fragment_sequence="AKLMQWERTYP",
        fragment_name="test_fragment",
        fragment_index=1,
        peptide=True,
        peptide_trim=2,
    )


class TestFixtures:
    """Test using pytest fixtures"""

    def test_sample_topology_properties(self, sample_topology):
        """Test the sample topology fixture"""
        assert sample_topology.chain == "A"
        assert sample_topology.length == 11
        assert sample_topology.peptide is True
        assert len(sample_topology.peptide_residues) == 9

    def test_modify_sample_topology(self, sample_topology):
        """Test modifying the sample topology"""
        original_trim = sample_topology.peptide_trim
        sample_topology._set_peptide(True, trim=3)

        assert sample_topology.peptide_trim == 3
        assert sample_topology.peptide_trim != original_trim


class TestTopologyMerging:
    """Test topology merging functionality"""

    def test_basic_merge(self):
        """Test basic merge functionality"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3], fragment_name="frag1")
        topo2 = TopologyFactory.from_residues("A", [5, 6, 7], fragment_name="frag2")
        topo3 = TopologyFactory.from_residues("A", [10, 11], fragment_name="frag3")

        merged = TopologyFactory.merge([topo1, topo2, topo3])

        assert merged.chain == "A"
        assert merged.residues == [1, 2, 3, 5, 6, 7, 10, 11]
        assert merged.fragment_name == "frag1+frag2+frag3"
        assert merged.peptide is False  # Default when no peptides in input

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises error"""
        with pytest.raises(ValueError, match="Cannot merge empty list of topologies"):
            TopologyFactory.merge([])

    def test_merge_different_chains_raises_error(self):
        """Test that merging different chains raises error"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3])
        topo2 = TopologyFactory.from_residues("B", [4, 5, 6])

        with pytest.raises(ValueError, match="All topologies must be on the same chain"):
            TopologyFactory.merge([topo1, topo2])

    def test_merge_with_custom_metadata(self):
        """Test merge with custom name and sequence"""
        topo1 = TopologyFactory.from_residues(
            "A", [1, 2, 3], fragment_sequence="MKL", fragment_name="start"
        )
        topo2 = TopologyFactory.from_residues(
            "A", [5, 6, 7], fragment_sequence="QWE", fragment_name="end"
        )

        merged = TopologyFactory.merge(
            [topo1, topo2], merged_name="custom_name", merged_sequence="CUSTOM_SEQ", merged_index=42
        )

        assert merged.fragment_name == "custom_name"
        assert merged.fragment_sequence == "CUSTOM_SEQ"
        assert merged.fragment_index == 42

    def test_merge_sequence_concatenation(self):
        """Test automatic sequence concatenation"""
        topo1 = TopologyFactory.from_residues("A", [1, 2, 3], fragment_sequence="MKL")
        topo2 = TopologyFactory.from_residues("A", [4, 5, 6], fragment_sequence="QWE")
        topo3 = TopologyFactory.from_residues("A", [7, 8, 9], fragment_sequence="RTY")

        merged = TopologyFactory.merge([topo1, topo2, topo3])

        assert merged.fragment_sequence == "MKLQWERTY"

    def test_merge_with_peptide_trim_true(self):
        """Test merge with peptide trimming enabled"""
        peptide1 = TopologyFactory.from_range(
            "A", 10, 15, peptide=True, peptide_trim=2, fragment_name="pep1"
        )
        # Full: [10-15], Active: [12-15]

        peptide2 = TopologyFactory.from_range(
            "A", 20, 25, peptide=True, peptide_trim=3, fragment_name="pep2"
        )
        # Full: [20-25], Active: [23-25]

        merged = TopologyFactory.merge([peptide1, peptide2], trim=True)

        # Should only include active residues: [12, 13, 14, 15, 23, 24, 25]
        assert merged.residues == [12, 13, 14, 15, 23, 24, 25]
        assert merged.peptide is True  # Result is a peptide
        assert merged.peptide_trim == 2  # Minimum trim from inputs

    def test_merge_with_peptide_trim_false(self):
        """Test merge with peptide trimming disabled"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        peptide2 = TopologyFactory.from_range("A", 20, 25, peptide=True, peptide_trim=3)

        merged = TopologyFactory.merge([peptide1, peptide2], trim=False)

        # Should include all residues: [10-15] + [20-25]
        assert merged.residues == [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]
        assert merged.peptide is False  # Result is not a peptide when trim=False
        assert merged.peptide_trim == 0  # No trimming when trim=False

    def test_merge_mixed_peptide_non_peptide(self):
        """Test merge with mix of peptides and non-peptides"""
        peptide = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Active: [12-15]

        regular = TopologyFactory.from_residues("A", [20, 21, 22])

        merged = TopologyFactory.merge([peptide, regular], trim=True)

        # Should include peptide active residues + all regular residues
        assert merged.residues == [12, 13, 14, 15, 20, 21, 22]
        assert merged.peptide is True  # Has peptide input

    def test_merge_contiguous_success(self):
        """Test successful contiguous merge"""
        topo1 = TopologyFactory.from_range("A", 10, 15, fragment_name="part1")
        topo2 = TopologyFactory.from_range("A", 16, 20, fragment_name="part2")
        topo3 = TopologyFactory.from_range("A", 21, 25, fragment_name="part3")

        merged = TopologyFactory.merge_contiguous([topo1, topo2, topo3])

        assert merged.residues == list(range(10, 26))  # [10, 11, ..., 25]
        assert merged.fragment_name == "part1+part2+part3"

    def test_merge_contiguous_with_gap_tolerance(self):
        """Test contiguous merge with gap tolerance"""
        topo1 = TopologyFactory.from_range("A", 10, 15)
        topo2 = TopologyFactory.from_range("A", 18, 22)  # Gap of 2 (16, 17 missing)

        # Should fail with gap_tolerance=0
        with pytest.raises(ValueError, match="Gap of 2 residues.*exceeds tolerance of 0"):
            TopologyFactory.merge_contiguous([topo1, topo2], gap_tolerance=0)

        # Should succeed with gap_tolerance=2
        merged = TopologyFactory.merge_contiguous([topo1, topo2], gap_tolerance=2)
        assert merged.residues == [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]

    def test_merge_contiguous_auto_sort(self):
        """Test that contiguous merge automatically sorts topologies"""
        topo1 = TopologyFactory.from_range("A", 20, 25, fragment_name="second")
        topo2 = TopologyFactory.from_range("A", 10, 15, fragment_name="first")
        topo3 = TopologyFactory.from_range(
            "A", 16, 19, fragment_name="middle"
        )  # Make them actually contiguous

        # Input in wrong order, should be auto-sorted
        merged = TopologyFactory.merge_contiguous([topo1, topo2, topo3])

        assert merged.residues == list(range(10, 26))
        # Name should reflect the sorted order
        assert "first" in merged.fragment_name

    def test_merge_contiguous_overlapping_allowed(self):
        """Test that overlapping topologies are allowed in contiguous merge"""
        topo1 = TopologyFactory.from_range("A", 10, 15)
        topo2 = TopologyFactory.from_range("A", 13, 18)  # Overlaps with topo1 at 13,14,15

        merged = TopologyFactory.merge_contiguous([topo1, topo2])

        assert merged.residues == list(range(10, 19))  # [10, 11, ..., 18]

    def test_merge_overlapping_success(self):
        """Test successful overlapping merge"""
        topo1 = TopologyFactory.from_residues("A", [10, 11, 12, 13], fragment_name="overlap1")
        topo2 = TopologyFactory.from_residues("A", [12, 13, 14, 15], fragment_name="overlap2")
        topo3 = TopologyFactory.from_residues("A", [14, 15, 16, 17], fragment_name="overlap3")

        # Each topology should overlap with at least one other by 2 residues
        # topo1 and topo2 overlap at [12, 13] (2 residues)
        # topo2 and topo3 overlap at [14, 15] (2 residues)
        merged = TopologyFactory.merge_overlapping([topo1, topo2, topo3], min_overlap=2)

        assert merged.residues == list(range(10, 18))  # [10, 11, ..., 17]
        assert merged.fragment_name == "overlap1+overlap2+overlap3"

    def test_merge_overlapping_insufficient_overlap(self):
        """Test overlapping merge with insufficient overlap"""
        topo1 = TopologyFactory.from_residues("A", [10, 11, 12], fragment_name="no_overlap1")
        topo2 = TopologyFactory.from_residues("A", [20, 21, 22], fragment_name="no_overlap2")

        with pytest.raises(ValueError, match="does not have at least 1 overlapping residues"):
            TopologyFactory.merge_overlapping([topo1, topo2], min_overlap=1)

    def test_merge_overlapping_single_topology(self):
        """Test overlapping merge with single topology"""
        topo = TopologyFactory.from_residues("A", [10, 11, 12])

        merged = TopologyFactory.merge_overlapping([topo])

        assert merged.residues == [10, 11, 12]

    def test_merge_with_peptide_trimming_contiguous(self):
        """Test contiguous merge with peptide trimming"""
        peptide1 = TopologyFactory.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Active: [12-15]

        peptide2 = TopologyFactory.from_range("A", 16, 20, peptide=True, peptide_trim=1)
        # Full: [16-20], Active: [17-20]

        # With trim=True, active regions [12-15] and [17-20] have a gap, so we need tolerance
        merged_trimmed = TopologyFactory.merge_contiguous(
            [peptide1, peptide2], trim=True, gap_tolerance=1
        )
        assert merged_trimmed.residues == [12, 13, 14, 15, 17, 18, 19, 20]

        # With trim=False, full regions [10-15] and [16-20] are adjacent
        merged_full = TopologyFactory.merge_contiguous([peptide1, peptide2], trim=False)
        assert merged_full.residues == list(range(10, 21))
        assert merged_full.peptide is False  # trim=False creates non-peptide


class TestRemoveResidues:
    """Test the remove_residues functionality"""

    def test_basic_remove(self):
        """Test basic residue removal"""
        topo = TopologyFactory.from_range("A", 1, 10, fragment_name="original")
        to_remove = TopologyFactory.from_residues("A", [3, 4, 5], fragment_name="to_remove")

        result = TopologyFactory.remove_residues_by_topologies(topo, [to_remove])

        assert result.chain == "A"
        assert result.residues == [1, 2, 6, 7, 8, 9, 10]
        assert result.fragment_name == "original"  # Should preserve metadata

    def test_remove_non_existent_residues(self):
        """Test removing residues that don't exist in the original topology"""
        topo = TopologyFactory.from_range("A", 1, 5, fragment_name="original")
        to_remove = TopologyFactory.from_residues("A", [6, 7, 8], fragment_name="non_existent")

        result = TopologyFactory.remove_residues_by_topologies(topo, [to_remove])

        # Should be unchanged since none of the residues exist in the original
        assert result.residues == [1, 2, 3, 4, 5]

    def test_remove_from_different_chain_raises_error(self):
        """Test that removing from different chain raises error"""
        topo = TopologyFactory.from_range("A", 1, 5)
        to_remove = TopologyFactory.from_residues("B", [1, 2, 3])

        with pytest.raises(ValueError, match="Cannot remove residues from different chain"):
            TopologyFactory.remove_residues_by_topologies(topo, [to_remove])

    def test_remove_all_residues_raises_error(self):
        """Test that removing all residues raises error"""
        topo = TopologyFactory.from_range("A", 1, 5)
        to_remove = TopologyFactory.from_range("A", 1, 5)

        with pytest.raises(ValueError, match="No residues remaining after removal"):
            TopologyFactory.remove_residues_by_topologies(topo, [to_remove])

    def test_remove_from_multiple_topologies(self):
        """Test removing residues from multiple topologies at once"""
        topo = TopologyFactory.from_range("A", 1, 20, fragment_name="original")
        to_remove1 = TopologyFactory.from_residues("A", [3, 4, 5], fragment_name="remove_first")
        to_remove2 = TopologyFactory.from_residues("A", [10, 11, 12], fragment_name="remove_middle")
        to_remove3 = TopologyFactory.from_residues("A", [18, 19], fragment_name="remove_end")

        result = TopologyFactory.remove_residues_by_topologies(
            topo, [to_remove1, to_remove2, to_remove3]
        )

        expected_residues = [1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17, 20]
        assert result.residues == expected_residues
        assert result.fragment_name == "original"

    def test_remove_with_peptide_trim(self):
        """Test removing residues with peptide trimming behavior"""
        # Create a peptide with trimming
        peptide = TopologyFactory.from_range("A", 1, 10, peptide=True, peptide_trim=2)
        # Full residues: [1-10], Peptide residues: [3-10]

        # Remove some residues
        to_remove = TopologyFactory.from_residues("A", [3, 4, 7, 8])

        result = TopologyFactory.remove_residues_by_topologies(peptide, [to_remove])

        # Check that metadata is preserved
        assert result.peptide is True
        assert result.peptide_trim == 2

        # Check residues
        assert result.residues == [1, 2, 5, 6, 9, 10]
        # Peptide residues should be recalculated after trim
        assert result.peptide_residues == [5, 6, 9, 10]  # Skips first 2 residues

        # Even with all peptide residues removed, if base residues remain, it should work
        all_peptide_residues = TopologyFactory.from_residues("A", [3, 4, 5, 6, 7, 8, 9, 10])
        result_no_active = TopologyFactory.remove_residues_by_topologies(
            peptide, [all_peptide_residues]
        )

        # Only has untrimmed residues remaining
        assert result_no_active.residues == [1, 2]
        # No peptide residues after trimming
        assert result_no_active.peptide_residues == []
