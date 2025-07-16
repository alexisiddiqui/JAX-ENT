import json
from pathlib import Path

import pytest

# Import the class being tested
# from your_module import Partial_Topology
# For this example, assume the class is in the same file or properly imported
from jaxent.src.interfaces.topology import Partial_Topology


class TestPartialTopologyConstruction:
    """Test different construction methods"""

    def test_from_range_basic(self):
        """Test basic range construction"""
        topo = Partial_Topology.from_range("A", 1, 5, "MKLIV", "N_terminus")

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
        topo = Partial_Topology.from_residues("B", [10, 11, 12, 13], "GLYK", "test_frag")

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12, 13]
        assert topo.is_contiguous is True
        assert topo.length == 4
        assert topo.residue_start == 10
        assert topo.residue_end == 13

    def test_from_residues_scattered(self):
        """Test construction from non-contiguous residue list"""
        topo = Partial_Topology.from_residues("C", [20, 22, 24, 26, 28], "ARGLY", "scattered")

        assert topo.chain == "C"
        assert topo.residues == [20, 22, 24, 26, 28]
        assert topo.is_contiguous is False
        assert topo.length == 5
        assert topo.residue_start == 20
        assert topo.residue_end == 28

    def test_from_single(self):
        """Test single residue construction"""
        topo = Partial_Topology.from_single("D", 100, "L", "single_lys")

        assert topo.chain == "D"
        assert topo.residues == [100]
        assert topo.length == 1
        assert topo.is_contiguous is True
        assert topo.residue_start == 100
        assert topo.residue_end == 100

    def test_residue_deduplication_and_sorting(self):
        """Test that residues are deduplicated and sorted"""
        topo = Partial_Topology.from_residues("E", [3, 1, 2, 3, 1], "MKL", "test")

        assert topo.residues == [1, 2, 3]
        assert topo.length == 3

    def test_empty_residues_raises_error(self):
        """Test that empty residue list raises ValueError"""
        with pytest.raises(ValueError, match="At least one residue must be specified"):
            Partial_Topology.from_residues("A", [], "SEQ", "test")


class TestPeptideBehavior:
    """Test peptide-specific functionality"""

    def test_peptide_creation_with_trim(self):
        """Test peptide creation with trimming"""
        topo = Partial_Topology.from_range(
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
        topo = Partial_Topology.from_range(chain="A", start=1, end=2, peptide=True, peptide_trim=5)

        assert topo.peptide is True
        assert topo.peptide_trim == 5
        assert topo.length == 2
        assert topo.peptide_residues == []  # Too short to trim

    def test_non_peptide_no_trimming(self):
        """Test that non-peptides don't have peptide_residues"""
        topo = Partial_Topology.from_range(
            chain="A", start=1, end=10, peptide=False, peptide_trim=2
        )

        assert topo.peptide is False
        assert topo.peptide_residues == []

    def test_set_peptide_enable(self):
        """Test enabling peptide mode"""
        topo = Partial_Topology.from_range("A", 1, 5, peptide=False)

        assert topo.peptide is False
        assert topo.peptide_residues == []

        topo.set_peptide(True, trim=1)

        assert topo.peptide is True
        assert topo.peptide_trim == 1
        assert topo.peptide_residues == [2, 3, 4, 5]

    def test_set_peptide_disable(self):
        """Test disabling peptide mode"""
        topo = Partial_Topology.from_range("A", 1, 10, peptide=True, peptide_trim=2)

        assert topo.peptide is True
        assert len(topo.peptide_residues) == 8

        topo.set_peptide(False)

        assert topo.peptide is False
        assert topo.peptide_residues == []

    def test_set_peptide_change_trim(self):
        """Test changing peptide trim value"""
        topo = Partial_Topology.from_range("A", 1, 10, peptide=True, peptide_trim=2)

        initial_peptide_length = len(topo.peptide_residues)

        topo.set_peptide(True, trim=3)

        assert topo.peptide_trim == 3
        assert len(topo.peptide_residues) == initial_peptide_length - 1


class TestSerialization:
    """Test JSON serialization and deserialization"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        topo = Partial_Topology.from_range(
            chain="A",
            start=1,
            end=5,
            fragment_sequence="MKLIV",
            fragment_name="test_fragment",
            fragment_index=42,
            peptide=True,
            peptide_trim=2,
        )

        data = topo.to_dict()
        expected_keys = {
            "chain",
            "residues",
            "fragment_sequence",
            "fragment_name",
            "fragment_index",
            "peptide",
            "peptide_trim",
        }

        assert set(data.keys()) == expected_keys
        assert data["chain"] == "A"
        assert data["residues"] == [1, 2, 3, 4, 5]
        assert data["fragment_sequence"] == "MKLIV"
        assert data["fragment_name"] == "test_fragment"
        assert data["fragment_index"] == 42
        assert data["peptide"] is True
        assert data["peptide_trim"] == 2

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "chain": "B",
            "residues": [10, 11, 12],
            "fragment_sequence": "GLY",
            "fragment_name": "test",
            "fragment_index": None,
            "peptide": False,
            "peptide_trim": 2,
        }

        topo = Partial_Topology.from_dict(data)

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12]
        assert topo.fragment_sequence == "GLY"
        assert topo.fragment_name == "test"
        assert topo.fragment_index is None
        assert topo.peptide is False
        assert topo.peptide_trim == 0

        data["peptide"] = True

        topo = Partial_Topology.from_dict(data)

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12]
        assert topo.fragment_sequence == "GLY"
        assert topo.fragment_name == "test"
        assert topo.fragment_index is None
        assert topo.peptide is True
        assert topo.peptide_trim == 2

    def test_json_round_trip(self):
        """Test JSON serialization round trip"""
        original = Partial_Topology.from_residues(
            chain="C",
            residues=[1, 3, 5, 7],
            fragment_sequence=["M", "K", "L", "I"],
            fragment_name="scattered_fragment",
            fragment_index=123,
            peptide=True,
            peptide_trim=1,
        )

        # Serialize to JSON
        json_str = original.to_json()

        # Deserialize from JSON
        restored = Partial_Topology.from_json(json_str)

        # Check all core attributes match
        assert restored.chain == original.chain
        assert restored.residues == original.residues
        assert restored.fragment_sequence == original.fragment_sequence
        assert restored.fragment_name == original.fragment_name
        assert restored.fragment_index == original.fragment_index
        assert restored.peptide == original.peptide
        assert restored.peptide_trim == original.peptide_trim

        # Check computed properties are recalculated correctly
        assert restored.residue_start == original.residue_start
        assert restored.residue_end == original.residue_end
        assert restored.length == original.length
        assert restored.is_contiguous == original.is_contiguous
        assert restored.peptide_residues == original.peptide_residues

    def test_json_format(self):
        """Test that JSON output is properly formatted"""
        topo = Partial_Topology.from_single("A", 1, "M", "test")
        json_str = topo.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Should be formatted (contain newlines)
        assert "\n" in json_str


class TestStringRepresentation:
    """Test string representations"""

    def test_str_single_residue(self):
        """Test string representation for single residue"""
        topo = Partial_Topology.from_single("A", 100, "L", "test", fragment_index=5)
        result = str(topo)

        assert "5:test:A:100" in result

    def test_str_contiguous_range(self):
        """Test string representation for contiguous range"""
        topo = Partial_Topology.from_range("B", 10, 15, "MKLIVQ", "helix")
        result = str(topo)

        assert "None:helix:B:10-15" in result

    def test_str_scattered_residues(self):
        """Test string representation for scattered residues"""
        topo = Partial_Topology.from_residues(
            "C", [1, 3, 5, 7, 9, 11, 13], fragment_name="scattered"
        )
        result = str(topo)

        # Should show first 3 and last 3 with ellipsis
        assert "scattered:C:[1,3,5...9,11,13]" in result

    def test_str_peptide_annotation(self):
        """Test peptide annotation in string representation"""
        topo = Partial_Topology.from_range(
            "A", 1, 10, "MKLIVQWERT", "peptide", peptide=True, peptide_trim=2
        )
        result = str(topo)

        assert "[peptide: 8 residues]" in result

    def test_repr_excludes_computed_fields(self):
        """Test that repr excludes computed fields"""
        topo = Partial_Topology.from_range("A", 1, 5, "MKLIV", "test")
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
        topo = Partial_Topology.from_range("A", 10, 15)
        ranges = topo.get_residue_ranges()

        assert ranges == [(10, 15)]

    def test_get_residue_ranges_scattered(self):
        """Test residue ranges for scattered topology"""
        topo = Partial_Topology.from_residues("A", [1, 2, 3, 7, 8, 15, 16, 17])
        ranges = topo.get_residue_ranges()

        assert ranges == [(1, 3), (7, 8), (15, 17)]

    def test_extract_residues_non_peptide(self):
        """Test extracting residues from non-peptide"""
        topo = Partial_Topology.from_range("A", 10, 12, peptide=False)
        extracted = topo.extract_residues(use_peptide_trim=False)

        assert len(extracted) == 3
        for i, res_topo in enumerate(extracted):
            assert res_topo.residues == [10 + i]
            assert res_topo.peptide is False

    def test_extract_residues_peptide_with_trim(self):
        """Test extracting residues from peptide with trimming"""
        topo = Partial_Topology.from_range("A", 1, 10, peptide=True, peptide_trim=2)
        extracted = topo.extract_residues(use_peptide_trim=True)

        # Should only get trimmed residues
        assert len(extracted) == 8  # 10 - 2 = 8
        assert extracted[0].residues == [3]  # First trimmed residue
        assert extracted[-1].residues == [10]  # Last residue

    def test_extract_residues_single(self):
        """Test extracting from single residue returns self-like"""
        topo = Partial_Topology.from_single("A", 100)
        extracted = topo.extract_residues()

        assert len(extracted) == 1
        assert extracted[0].residues == [100]

    def test_contains_residue_single(self):
        """Test residue containment check for single residue"""
        topo = Partial_Topology.from_residues("A", [1, 3, 5, 7])

        assert topo.contains_residue(1) is True
        assert topo.contains_residue(3) is True
        assert topo.contains_residue(2) is False
        assert topo.contains_residue(8) is False

    def test_contains_residue_multiple(self):
        """Test residue containment check for multiple residues"""
        topo = Partial_Topology.from_residues("A", [1, 3, 5, 7])

        result = topo.contains_residue([1, 2, 3, 4, 5])
        expected = {1: True, 2: False, 3: True, 4: False, 5: True}

        assert result == expected
        assert isinstance(result, dict)

    def test_contains_residue_with_peptide_trim(self):
        """Test residue containment with peptide trimming"""
        # Create a peptide that trims first 2 residues
        peptide = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full residues: [10, 11, 12, 13, 14, 15]
        # Peptide residues: [12, 13, 14, 15] (after trimming first 2)

        # With check_trim=True (default), should only check peptide_residues
        assert peptide.contains_residue(10, check_trim=True) is False  # Trimmed
        assert peptide.contains_residue(11, check_trim=True) is False  # Trimmed
        assert peptide.contains_residue(12, check_trim=True) is True  # Active
        assert peptide.contains_residue(15, check_trim=True) is True  # Active

        # With check_trim=False, should check all residues
        assert peptide.contains_residue(10, check_trim=False) is True  # In full residues
        assert peptide.contains_residue(11, check_trim=False) is True  # In full residues
        assert peptide.contains_residue(12, check_trim=False) is True  # In full residues
        assert peptide.contains_residue(15, check_trim=False) is True  # In full residues

    def test_contains_all_residues(self):
        """Test checking if topology contains all specified residues"""
        topo = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])

        assert topo.contains_all_residues([1, 3, 5]) is True
        assert topo.contains_all_residues([1, 2, 3, 4, 5]) is True
        assert topo.contains_all_residues([1, 6]) is False
        assert topo.contains_all_residues([6, 7, 8]) is False
        assert topo.contains_all_residues([]) is True  # Empty list

    def test_contains_all_residues_peptide_trim(self):
        """Test contains_all_residues with peptide trimming"""
        peptide = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
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
        topo = Partial_Topology.from_residues("A", [1, 3, 5, 7])

        assert topo.contains_any_residues([1, 2]) is True
        assert topo.contains_any_residues([2, 4, 6]) is False
        assert topo.contains_any_residues([7, 8, 9]) is True
        assert topo.contains_any_residues([]) is False  # Empty list

    def test_contains_any_residues_peptide_trim(self):
        """Test contains_any_residues with peptide trimming"""
        peptide = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
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

    def test_intersects_same_chain(self):
        """Test intersection with same chain"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4])
        topo2 = Partial_Topology.from_residues("A", [3, 4, 5, 6])

        assert topo1.intersects(topo2) is True
        assert topo2.intersects(topo1) is True

    def test_intersects_different_chain(self):
        """Test intersection with different chains"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3])
        topo2 = Partial_Topology.from_residues("B", [1, 2, 3])

        assert topo1.intersects(topo2) is False

    def test_intersects_no_overlap(self):
        """Test intersection with no overlap"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3])
        topo2 = Partial_Topology.from_residues("A", [4, 5, 6])

        assert topo1.intersects(topo2) is False

    def test_intersects_with_peptide_trim(self):
        """Test intersection considering peptide trimming"""
        # Create two peptides where full residues overlap but peptide residues don't
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=3)
        # Full: [10, 11, 12, 13, 14, 15], Peptide: [13, 14, 15]

        peptide2 = Partial_Topology.from_range("A", 13, 18, peptide=True, peptide_trim=4)
        # Full: [13, 14, 15, 16, 17, 18], Peptide: [17, 18]

        # With check_trim=True (default), peptide residues don't overlap
        assert peptide1.intersects(peptide2, check_trim=True) is False

        # With check_trim=False, full residues do overlap ([13, 14, 15])
        assert peptide1.intersects(peptide2, check_trim=False) is True

    def test_contains_topology_full_containment(self):
        """Test topology containment - full containment"""
        large_topo = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5, 6, 7, 8])
        small_topo = Partial_Topology.from_residues("A", [3, 4, 5])

        assert large_topo.contains_topology(small_topo) is True
        assert small_topo.contains_topology(large_topo) is False

    def test_contains_topology_with_peptide_trim(self):
        """Test topology containment with peptide trimming"""
        # Large peptide that contains a smaller region after trimming
        large_peptide = Partial_Topology.from_range("A", 10, 20, peptide=True, peptide_trim=2)
        # Full: [10-20], Peptide: [12-20]

        small_topo = Partial_Topology.from_residues("A", [11, 12, 13])

        # With check_trim=True, only checks if [11,12,13] ⊆ [12-20] → False (11 not in peptide)
        assert large_peptide.contains_topology(small_topo, check_trim=True) is False

        # With check_trim=False, checks if [11,12,13] ⊆ [10-20] → True
        assert large_peptide.contains_topology(small_topo, check_trim=False) is True

        # Small topology that fits within peptide residues
        small_peptide_topo = Partial_Topology.from_residues("A", [15, 16, 17])
        assert large_peptide.contains_topology(small_peptide_topo, check_trim=True) is True
        assert large_peptide.contains_topology(small_peptide_topo, check_trim=False) is True

    def test_contains_topology_partial_overlap(self):
        """Test topology containment - partial overlap"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4])
        topo2 = Partial_Topology.from_residues("A", [3, 4, 5, 6])

        assert topo1.contains_topology(topo2) is False
        assert topo2.contains_topology(topo1) is False

    def test_contains_topology_different_chains(self):
        """Test topology containment - different chains"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = Partial_Topology.from_residues("B", [2, 3])

        assert topo1.contains_topology(topo2) is False

    def test_is_subset_of(self):
        """Test subset relationship"""
        small_topo = Partial_Topology.from_residues("A", [2, 3, 4])
        large_topo = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])

        assert small_topo.is_subset_of(large_topo) is True
        assert large_topo.is_subset_of(small_topo) is False

        # Same topology should be subset of itself
        assert small_topo.is_subset_of(small_topo) is True

    def test_is_subset_of_with_peptide_trim(self):
        """Test subset relationship with peptide trimming"""
        peptide = Partial_Topology.from_range("A", 10, 20, peptide=True, peptide_trim=3)
        # Full: [10-20], Peptide: [13-20]

        # Subset that includes trimmed residues
        subset_with_trimmed = Partial_Topology.from_residues("A", [10, 11, 15])

        # With check_trim=True, [10,11,15] ⊄ [13-20] → False
        assert subset_with_trimmed.is_subset_of(peptide, check_trim=True) is False

        # With check_trim=False, [10,11,15] ⊆ [10-20] → True
        assert subset_with_trimmed.is_subset_of(peptide, check_trim=False) is True

        # Subset within peptide residues
        subset_peptide = Partial_Topology.from_residues("A", [15, 16, 17])
        assert subset_peptide.is_subset_of(peptide, check_trim=True) is True
        assert subset_peptide.is_subset_of(peptide, check_trim=False) is True

    def test_is_superset_of(self):
        """Test superset relationship"""
        small_topo = Partial_Topology.from_residues("A", [2, 3, 4])
        large_topo = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])

        assert large_topo.is_superset_of(small_topo) is True
        assert small_topo.is_superset_of(large_topo) is False

        # Same topology should be superset of itself
        assert large_topo.is_superset_of(large_topo) is True

    def test_get_overlap(self):
        """Test getting overlapping residues"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = Partial_Topology.from_residues("A", [3, 4, 5, 6, 7])

        overlap = topo1.get_overlap(topo2)
        assert overlap == [3, 4, 5]

        # Test no overlap
        topo3 = Partial_Topology.from_residues("A", [10, 11, 12])
        no_overlap = topo1.get_overlap(topo3)
        assert no_overlap == []

        # Test different chains
        topo4 = Partial_Topology.from_residues("B", [3, 4, 5])
        different_chain = topo1.get_overlap(topo4)
        assert different_chain == []

    def test_get_overlap_with_peptide_trim(self):
        """Test getting overlap with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = Partial_Topology.from_range("A", 13, 18, peptide=True, peptide_trim=3)
        # Full: [13-18], Peptide: [16-18]

        # With check_trim=True, overlap of peptide residues [12-15] ∩ [16-18] = []
        overlap_trimmed = peptide1.get_overlap(peptide2, check_trim=True)
        assert overlap_trimmed == []

        # With check_trim=False, overlap of full residues [10-15] ∩ [13-18] = [13,14,15]
        overlap_full = peptide1.get_overlap(peptide2, check_trim=False)
        assert overlap_full == [13, 14, 15]

    def test_get_difference(self):
        """Test getting residue differences"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4, 5])
        topo2 = Partial_Topology.from_residues("A", [3, 4, 5, 6, 7])

        diff1 = topo1.get_difference(topo2)
        assert diff1 == [1, 2]

        diff2 = topo2.get_difference(topo1)
        assert diff2 == [6, 7]

        # Test no overlap (should return all residues)
        topo3 = Partial_Topology.from_residues("A", [10, 11, 12])
        diff3 = topo1.get_difference(topo3)
        assert diff3 == [1, 2, 3, 4, 5]

        # Test different chains (should return all residues)
        topo4 = Partial_Topology.from_residues("B", [1, 2, 3])
        diff4 = topo1.get_difference(topo4)
        assert diff4 == [1, 2, 3, 4, 5]

    def test_get_difference_with_peptide_trim(self):
        """Test getting difference with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        regular_topo = Partial_Topology.from_residues("A", [13, 14, 16])

        # With check_trim=True, difference of [12-15] - [13,14,16] = [12,15]
        diff_trimmed = peptide1.get_difference(regular_topo, check_trim=True)
        assert diff_trimmed == [12, 15]

        # With check_trim=False, difference of [10-15] - [13,14,16] = [10,11,12,15]
        diff_full = peptide1.get_difference(regular_topo, check_trim=False)
        assert diff_full == [10, 11, 12, 15]

    def test_is_adjacent_to(self):
        """Test adjacency checking"""
        topo1 = Partial_Topology.from_range("A", 10, 15)
        topo2 = Partial_Topology.from_range("A", 16, 20)  # Adjacent after topo1
        topo3 = Partial_Topology.from_range("A", 5, 9)  # Adjacent before topo1
        topo4 = Partial_Topology.from_range("A", 18, 25)  # Not adjacent

        assert topo1.is_adjacent_to(topo2) is True
        assert topo1.is_adjacent_to(topo3) is True
        assert topo1.is_adjacent_to(topo4) is False

        # Test non-contiguous topology
        scattered = Partial_Topology.from_residues("A", [1, 3, 5])
        adjacent_scattered = Partial_Topology.from_residues("A", [6, 7, 8])
        assert scattered.is_adjacent_to(adjacent_scattered) is True  # 5 + 1 = 6

    def test_is_adjacent_to_with_peptide_trim(self):
        """Test adjacency with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=3)
        # Full: [10-15], Peptide: [13-15]

        peptide2 = Partial_Topology.from_range("A", 16, 20, peptide=True, peptide_trim=2)
        # Full: [16-20], Peptide: [18-20]

        # With check_trim=True, peptide residues [13-15] and [18-20] are NOT adjacent
        assert peptide1.is_adjacent_to(peptide2, check_trim=True) is False

        # With check_trim=False, full residues [10-15] and [16-20] ARE adjacent
        assert peptide1.is_adjacent_to(peptide2, check_trim=False) is True

        # Create adjacent peptide residues
        peptide3 = Partial_Topology.from_range("A", 16, 20, peptide=True, peptide_trim=0)
        # Full: [16-20], Peptide: [16-20] (no trimming)

        assert peptide1.is_adjacent_to(peptide3, check_trim=True) is True  # 15 + 1 = 16

    def test_get_gap_to(self):
        """Test gap calculation between topologies"""
        topo1 = Partial_Topology.from_range("A", 10, 15)  # ends at 15
        topo2 = Partial_Topology.from_range("A", 16, 20)  # starts at 16 (adjacent)
        topo3 = Partial_Topology.from_range("A", 18, 25)  # starts at 18 (gap of 2)
        topo4 = Partial_Topology.from_range("A", 5, 8)  # ends at 8 (gap of 1)

        # Adjacent topologies
        assert topo1.get_gap_to(topo2) == 0
        assert topo2.get_gap_to(topo1) == 0

        # Gap of 2 (16, 17 are missing)
        assert topo1.get_gap_to(topo3) == 2
        assert topo3.get_gap_to(topo1) == 2

        # Gap of 1 (9 is missing)
        assert topo1.get_gap_to(topo4) == 1
        assert topo4.get_gap_to(topo1) == 1

        # Overlapping topologies
        overlapping = Partial_Topology.from_range("A", 12, 18)
        assert topo1.get_gap_to(overlapping) is None

        # Different chains
        different_chain = Partial_Topology.from_range("B", 16, 20)
        assert topo1.get_gap_to(different_chain) is None

    def test_get_gap_to_with_peptide_trim(self):
        """Test gap calculation with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = Partial_Topology.from_range("A", 18, 22, peptide=True, peptide_trim=1)
        # Full: [18-22], Peptide: [19-22]

        # With check_trim=True, gap between peptide residues [12-15] and [19-22]
        # Gap = 19 - 15 - 1 = 3 (residues 16, 17, 18)
        gap_trimmed = peptide1.get_gap_to(peptide2, check_trim=True)
        assert gap_trimmed == 3

        # With check_trim=False, gap between full residues [10-15] and [18-22]
        # Gap = 18 - 15 - 1 = 2 (residues 16, 17)
        gap_full = peptide1.get_gap_to(peptide2, check_trim=False)
        assert gap_full == 2

    def test_union_same_chain(self):
        """Test union of topologies on same chain"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3], fragment_name="frag1")
        topo2 = Partial_Topology.from_residues("A", [4, 5, 6], fragment_name="frag2")

        union = topo1.union(topo2)

        assert union.chain == "A"
        assert union.residues == [1, 2, 3, 4, 5, 6]
        assert union.fragment_name == "frag1"  # Takes from first topology

    def test_union_with_peptide_trim(self):
        """Test union with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Peptide: [12-15]

        peptide2 = Partial_Topology.from_range("A", 18, 22, peptide=True, peptide_trim=1)
        # Full: [18-22], Peptide: [19-22]

        # With check_trim=True, union based on peptide residues [12-15] ∪ [19-22]
        # Should create range [12-22] to encompass both
        union_trimmed = peptide1.union(peptide2, check_trim=True)
        expected_range = list(range(12, 23))  # [12, 13, ..., 22]
        assert union_trimmed.residues == expected_range

        # With check_trim=False, union based on full residues [10-15] ∪ [18-22]
        # Should create range [10-22] to encompass both
        union_full = peptide1.union(peptide2, check_trim=False)
        expected_full_range = list(range(10, 23))  # [10, 11, ..., 22]
        assert union_full.residues == expected_full_range

    def test_union_different_chain_raises_error(self):
        """Test that union with different chains raises error"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3])
        topo2 = Partial_Topology.from_residues("B", [1, 2, 3])

        with pytest.raises(ValueError, match="Cannot combine topologies with different chains"):
            topo1.union(topo2)

    def test_union_overlapping_residues(self):
        """Test union with overlapping residues"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3, 4])
        topo2 = Partial_Topology.from_residues("A", [3, 4, 5, 6])

        union = topo1.union(topo2)

        # Should create contiguous range to encompass both
        assert union.residues == [1, 2, 3, 4, 5, 6]


class TestTopologyComparisons:
    """Test advanced topology comparison methods"""

    def test_protein_domain_analysis(self):
        """Test real-world protein domain comparison scenario"""
        # Define some protein domains
        signal_peptide = Partial_Topology.from_range("A", 1, 25, fragment_name="signal")
        n_terminal = Partial_Topology.from_range("A", 26, 150, fragment_name="N_terminal")
        binding_domain = Partial_Topology.from_range("A", 100, 200, fragment_name="binding")
        c_terminal = Partial_Topology.from_range("A", 201, 350, fragment_name="C_terminal")

        # Test domain relationships
        assert signal_peptide.is_adjacent_to(n_terminal) is True
        assert (
            binding_domain.is_adjacent_to(c_terminal) is True
        )  # binding domain (ends at 200) is adjacent to c_terminal (starts at 201)
        assert (
            signal_peptide.is_adjacent_to(c_terminal) is False
        )  # signal peptide is far from c_terminal

        # Test domain overlaps
        assert n_terminal.intersects(binding_domain) is True
        overlap = n_terminal.get_overlap(binding_domain)
        assert overlap == list(range(100, 151))  # residues 100-150

        # Test containment
        assert binding_domain.contains_topology(n_terminal) is False
        assert n_terminal.contains_topology(Partial_Topology.from_range("A", 120, 130)) is True

    def test_active_site_analysis(self):
        """Test active site residue analysis"""
        # Enzyme with scattered active site
        catalytic_triad = Partial_Topology.from_residues(
            "A", [57, 102, 195], fragment_name="catalytic_triad"
        )
        oxyanion_hole = Partial_Topology.from_residues(
            "A", [58, 195, 196], fragment_name="oxyanion_hole"
        )
        substrate_binding = Partial_Topology.from_residues(
            "A", [89, 90, 156, 189, 190], fragment_name="substrate_binding"
        )

        # Check which residues are shared between functional sites
        triad_oxyanion_overlap = catalytic_triad.get_overlap(oxyanion_hole)
        assert triad_oxyanion_overlap == [195]  # Shared catalytic residue

        # Check if substrate binding site contains any catalytic residues
        assert catalytic_triad.intersects(substrate_binding) is False

        # Find residues unique to each site
        unique_to_triad = catalytic_triad.get_difference(oxyanion_hole)
        assert unique_to_triad == [57, 102]

    def test_multi_residue_queries(self):
        """Test querying multiple residues at once"""
        protein_region = Partial_Topology.from_residues("A", [10, 12, 14, 16, 18, 20, 22])

        # Check specific residues of interest
        query_residues = [10, 11, 12, 13, 14, 15, 16]
        containment_map = protein_region.contains_residue(query_residues)
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
        exon1 = Partial_Topology.from_range("A", 1, 50, fragment_name="exon1")
        exon2 = Partial_Topology.from_range("A", 101, 150, fragment_name="exon2")
        exon3 = Partial_Topology.from_range("A", 201, 300, fragment_name="exon3")

        # Calculate intron sizes
        intron1_size = exon1.get_gap_to(exon2)
        intron2_size = exon2.get_gap_to(exon3)

        assert intron1_size == 50  # residues 51-100 (50 residues)
        assert intron2_size == 50  # residues 151-200 (50 residues)

        # Check adjacent exons would have gap of 0
        adjacent_exon = Partial_Topology.from_range("A", 51, 100)
        assert exon1.get_gap_to(adjacent_exon) == 0

    def test_subset_superset_relationships(self):
        """Test subset/superset relationships in protein analysis"""
        # Full protein sequence
        full_protein = Partial_Topology.from_range("A", 1, 500, fragment_name="full_protein")

        # Various domains
        signal_peptide = Partial_Topology.from_range("A", 1, 25, fragment_name="signal")
        mature_protein = Partial_Topology.from_range("A", 26, 500, fragment_name="mature")
        binding_domain = Partial_Topology.from_range("A", 100, 200, fragment_name="binding")

        # Test hierarchical relationships
        assert signal_peptide.is_subset_of(full_protein) is True
        assert mature_protein.is_subset_of(full_protein) is True
        assert binding_domain.is_subset_of(full_protein) is True
        assert binding_domain.is_subset_of(mature_protein) is True

        # Test superset relationships
        assert full_protein.is_superset_of(signal_peptide) is True
        assert full_protein.is_superset_of(mature_protein) is True
        assert mature_protein.is_superset_of(binding_domain) is True

        # Signal peptide and mature protein should not overlap
        assert signal_peptide.intersects(mature_protein) is False


class TestTopologyMerging:
    """Test topology merging functionality"""

    def test_basic_merge(self):
        """Test basic merge functionality"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3], fragment_name="frag1")
        topo2 = Partial_Topology.from_residues("A", [5, 6, 7], fragment_name="frag2")
        topo3 = Partial_Topology.from_residues("A", [10, 11], fragment_name="frag3")

        merged = Partial_Topology.merge([topo1, topo2, topo3])

        assert merged.chain == "A"
        assert merged.residues == [1, 2, 3, 5, 6, 7, 10, 11]
        assert merged.fragment_name == "frag1+frag2+frag3"
        assert merged.peptide is False  # Default when no peptides in input

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises error"""
        with pytest.raises(ValueError, match="Cannot merge empty list of topologies"):
            Partial_Topology.merge([])

    def test_merge_different_chains_raises_error(self):
        """Test that merging different chains raises error"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3])
        topo2 = Partial_Topology.from_residues("B", [4, 5, 6])

        with pytest.raises(ValueError, match="All topologies must be on the same chain"):
            Partial_Topology.merge([topo1, topo2])

    def test_merge_with_custom_metadata(self):
        """Test merge with custom name and sequence"""
        topo1 = Partial_Topology.from_residues(
            "A", [1, 2, 3], fragment_sequence="MKL", fragment_name="start"
        )
        topo2 = Partial_Topology.from_residues(
            "A", [5, 6, 7], fragment_sequence="QWE", fragment_name="end"
        )

        merged = Partial_Topology.merge(
            [topo1, topo2], merged_name="custom_name", merged_sequence="CUSTOM_SEQ", merged_index=42
        )

        assert merged.fragment_name == "custom_name"
        assert merged.fragment_sequence == "CUSTOM_SEQ"
        assert merged.fragment_index == 42

    def test_merge_sequence_concatenation(self):
        """Test automatic sequence concatenation"""
        topo1 = Partial_Topology.from_residues("A", [1, 2, 3], fragment_sequence="MKL")
        topo2 = Partial_Topology.from_residues("A", [4, 5, 6], fragment_sequence="QWE")
        topo3 = Partial_Topology.from_residues("A", [7, 8, 9], fragment_sequence="RTY")

        merged = Partial_Topology.merge([topo1, topo2, topo3])

        assert merged.fragment_sequence == "MKLQWERTY"

    def test_merge_with_peptide_trim_true(self):
        """Test merge with peptide trimming enabled"""
        peptide1 = Partial_Topology.from_range(
            "A", 10, 15, peptide=True, peptide_trim=2, fragment_name="pep1"
        )
        # Full: [10-15], Active: [12-15]

        peptide2 = Partial_Topology.from_range(
            "A", 20, 25, peptide=True, peptide_trim=3, fragment_name="pep2"
        )
        # Full: [20-25], Active: [23-25]

        merged = Partial_Topology.merge([peptide1, peptide2], trim=True)

        # Should only include active residues: [12, 13, 14, 15, 23, 24, 25]
        assert merged.residues == [12, 13, 14, 15, 23, 24, 25]
        assert merged.peptide is True  # Result is a peptide
        assert merged.peptide_trim == 2  # Minimum trim from inputs

    def test_merge_with_peptide_trim_false(self):
        """Test merge with peptide trimming disabled"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        peptide2 = Partial_Topology.from_range("A", 20, 25, peptide=True, peptide_trim=3)

        merged = Partial_Topology.merge([peptide1, peptide2], trim=False)

        # Should include all residues: [10-15] + [20-25]
        assert merged.residues == [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]
        assert merged.peptide is False  # Result is not a peptide when trim=False
        assert merged.peptide_trim == 0  # No trimming when trim=False

    def test_merge_mixed_peptide_non_peptide(self):
        """Test merge with mix of peptides and non-peptides"""
        peptide = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Active: [12-15]

        regular = Partial_Topology.from_residues("A", [20, 21, 22])

        merged = Partial_Topology.merge([peptide, regular], trim=True)

        # Should include peptide active residues + all regular residues
        assert merged.residues == [12, 13, 14, 15, 20, 21, 22]
        assert merged.peptide is True  # Has peptide input

    def test_merge_contiguous_success(self):
        """Test successful contiguous merge"""
        topo1 = Partial_Topology.from_range("A", 10, 15, fragment_name="part1")
        topo2 = Partial_Topology.from_range("A", 16, 20, fragment_name="part2")
        topo3 = Partial_Topology.from_range("A", 21, 25, fragment_name="part3")

        merged = Partial_Topology.merge_contiguous([topo1, topo2, topo3])

        assert merged.residues == list(range(10, 26))  # [10, 11, ..., 25]
        assert merged.fragment_name == "part1+part2+part3"

    def test_merge_contiguous_with_gap_tolerance(self):
        """Test contiguous merge with gap tolerance"""
        topo1 = Partial_Topology.from_range("A", 10, 15)
        topo2 = Partial_Topology.from_range("A", 18, 22)  # Gap of 2 (16, 17 missing)

        # Should fail with gap_tolerance=0
        with pytest.raises(ValueError, match="Gap of 2 residues.*exceeds tolerance of 0"):
            Partial_Topology.merge_contiguous([topo1, topo2], gap_tolerance=0)

        # Should succeed with gap_tolerance=2
        merged = Partial_Topology.merge_contiguous([topo1, topo2], gap_tolerance=2)
        assert merged.residues == [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]

    def test_merge_contiguous_auto_sort(self):
        """Test that contiguous merge automatically sorts topologies"""
        topo1 = Partial_Topology.from_range("A", 20, 25, fragment_name="second")
        topo2 = Partial_Topology.from_range("A", 10, 15, fragment_name="first")
        topo3 = Partial_Topology.from_range(
            "A", 16, 19, fragment_name="middle"
        )  # Make them actually contiguous

        # Input in wrong order, should be auto-sorted
        merged = Partial_Topology.merge_contiguous([topo1, topo2, topo3])

        assert merged.residues == list(range(10, 26))
        # Name should reflect the sorted order
        assert "first" in merged.fragment_name

    def test_merge_contiguous_overlapping_allowed(self):
        """Test that overlapping topologies are allowed in contiguous merge"""
        topo1 = Partial_Topology.from_range("A", 10, 15)
        topo2 = Partial_Topology.from_range("A", 13, 18)  # Overlaps with topo1 at 13,14,15

        merged = Partial_Topology.merge_contiguous([topo1, topo2])

        assert merged.residues == list(range(10, 19))  # [10, 11, ..., 18]

    def test_merge_overlapping_success(self):
        """Test successful overlapping merge"""
        topo1 = Partial_Topology.from_residues("A", [10, 11, 12, 13], fragment_name="overlap1")
        topo2 = Partial_Topology.from_residues("A", [12, 13, 14, 15], fragment_name="overlap2")
        topo3 = Partial_Topology.from_residues("A", [14, 15, 16, 17], fragment_name="overlap3")

        # Each topology should overlap with at least one other by 2 residues
        # topo1 and topo2 overlap at [12, 13] (2 residues)
        # topo2 and topo3 overlap at [14, 15] (2 residues)
        merged = Partial_Topology.merge_overlapping([topo1, topo2, topo3], min_overlap=2)

        assert merged.residues == list(range(10, 18))  # [10, 11, ..., 17]
        assert merged.fragment_name == "overlap1+overlap2+overlap3"

    def test_merge_overlapping_insufficient_overlap(self):
        """Test overlapping merge with insufficient overlap"""
        topo1 = Partial_Topology.from_residues("A", [10, 11, 12], fragment_name="no_overlap1")
        topo2 = Partial_Topology.from_residues("A", [20, 21, 22], fragment_name="no_overlap2")

        with pytest.raises(ValueError, match="does not have at least 1 overlapping residues"):
            Partial_Topology.merge_overlapping([topo1, topo2], min_overlap=1)

    def test_merge_overlapping_single_topology(self):
        """Test overlapping merge with single topology"""
        topo = Partial_Topology.from_residues("A", [10, 11, 12])

        merged = Partial_Topology.merge_overlapping([topo])

        assert merged.residues == [10, 11, 12]

    def test_merge_with_peptide_trimming_contiguous(self):
        """Test contiguous merge with peptide trimming"""
        peptide1 = Partial_Topology.from_range("A", 10, 15, peptide=True, peptide_trim=2)
        # Full: [10-15], Active: [12-15]

        peptide2 = Partial_Topology.from_range("A", 16, 20, peptide=True, peptide_trim=1)
        # Full: [16-20], Active: [17-20]

        # With trim=True, active regions [12-15] and [17-20] have a gap, so we need tolerance
        merged_trimmed = Partial_Topology.merge_contiguous(
            [peptide1, peptide2], trim=True, gap_tolerance=1
        )
        assert merged_trimmed.residues == [12, 13, 14, 15, 17, 18, 19, 20]

        # With trim=False, full regions [10-15] and [16-20] are adjacent
        merged_full = Partial_Topology.merge_contiguous([peptide1, peptide2], trim=False)
        assert merged_full.residues == list(range(10, 21))
        assert merged_full.peptide is False  # trim=False creates non-peptide


class TestBiophysicalExamples:
    """Test real-world biophysical use cases"""

    def test_signal_peptide(self):
        """Test signal peptide modeling"""
        signal = Partial_Topology.from_range(
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
        active_site = Partial_Topology.from_residues(
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
        tm_helix = Partial_Topology.from_range(
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
        signal_peptide = Partial_Topology.from_range(
            "A", 1, 25, fragment_name="signal_peptide", peptide=True, peptide_trim=3
        )

        n_domain = Partial_Topology.from_range("A", 26, 150, fragment_name="N_domain")
        binding_domain = Partial_Topology.from_range("A", 151, 280, fragment_name="binding_domain")
        c_domain = Partial_Topology.from_range("A", 281, 400, fragment_name="C_domain")

        # Merge mature protein (excluding signal)
        mature_protein = Partial_Topology.merge_contiguous(
            [n_domain, binding_domain, c_domain], merged_name="mature_protein"
        )

        assert mature_protein.residues == list(range(26, 401))
        assert mature_protein.fragment_name == "mature_protein"
        assert mature_protein.length == 375

        # Full protein including signal (with trimming to get mature sequence)
        full_protein_trimmed = Partial_Topology.merge(
            [signal_peptide, n_domain, binding_domain, c_domain],
            trim=True,
            merged_name="full_protein_active",
        )

        # Should exclude trimmed signal peptide residues (1-3)
        expected_residues = list(range(4, 26)) + list(range(26, 401))  # [4-25] + [26-400]
        assert full_protein_trimmed.residues == expected_residues

        # Full protein without trimming (includes all residues)
        full_protein_complete = Partial_Topology.merge(
            [signal_peptide, n_domain, binding_domain, c_domain],
            trim=False,
            merged_name="full_protein_complete",
        )

        assert full_protein_complete.residues == list(range(1, 401))
        assert full_protein_complete.peptide is False  # trim=False removes peptide flag

    def test_active_site_assembly(self):
        """Test assembling active site from scattered catalytic residues"""
        # Individual catalytic residues
        his57 = Partial_Topology.from_single("A", 57, fragment_sequence="H", fragment_name="His57")
        asp102 = Partial_Topology.from_single(
            "A", 102, fragment_sequence="D", fragment_name="Asp102"
        )
        ser195 = Partial_Topology.from_single(
            "A", 195, fragment_sequence="S", fragment_name="Ser195"
        )

        # Assemble catalytic triad
        catalytic_triad = Partial_Topology.merge(
            [his57, asp102, ser195], merged_name="catalytic_triad", merged_sequence=["H", "D", "S"]
        )

        assert catalytic_triad.residues == [57, 102, 195]
        assert catalytic_triad.fragment_name == "catalytic_triad"
        assert catalytic_triad.fragment_sequence == ["H", "D", "S"]
        assert catalytic_triad.is_contiguous is False

        # Check relationships with substrate binding site
        substrate_site = Partial_Topology.from_residues(
            "A", [189, 190, 214, 226], fragment_name="substrate_binding"
        )

        assert not catalytic_triad.intersects(substrate_site)  # Should be separate

        # Merge into complete active site
        complete_active_site = Partial_Topology.merge(
            [catalytic_triad, substrate_site], merged_name="complete_active_site"
        )

        expected_active_residues = [57, 102, 189, 190, 195, 214, 226]
        assert complete_active_site.residues == expected_active_residues


@pytest.fixture
def sample_topology():
    """Fixture providing a sample topology for testing"""
    return Partial_Topology.from_range(
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
        sample_topology.set_peptide(True, trim=3)

        assert sample_topology.peptide_trim == 3
        assert sample_topology.peptide_trim != original_trim


class TestRemoveResidues:
    """Test the remove_residues functionality"""

    def test_basic_remove(self):
        """Test basic residue removal"""
        topo = Partial_Topology.from_range("A", 1, 10, fragment_name="original")
        to_remove = Partial_Topology.from_residues("A", [3, 4, 5], fragment_name="to_remove")

        result = topo.remove_residues([to_remove])

        assert result.chain == "A"
        assert result.residues == [1, 2, 6, 7, 8, 9, 10]
        assert result.fragment_name == "original"  # Should preserve metadata

    def test_remove_non_existent_residues(self):
        """Test removing residues that don't exist in the original topology"""
        topo = Partial_Topology.from_range("A", 1, 5, fragment_name="original")
        to_remove = Partial_Topology.from_residues("A", [6, 7, 8], fragment_name="non_existent")

        result = topo.remove_residues([to_remove])

        # Should be unchanged since none of the residues exist in the original
        assert result.residues == [1, 2, 3, 4, 5]

    def test_remove_from_different_chain_raises_error(self):
        """Test that removing from different chain raises error"""
        topo = Partial_Topology.from_range("A", 1, 5)
        to_remove = Partial_Topology.from_residues("B", [1, 2, 3])

        with pytest.raises(ValueError, match="Cannot remove residues from different chain"):
            topo.remove_residues([to_remove])

    def test_remove_all_residues_raises_error(self):
        """Test that removing all residues raises error"""
        topo = Partial_Topology.from_range("A", 1, 5)
        to_remove = Partial_Topology.from_range("A", 1, 5)

        with pytest.raises(ValueError, match="No residues remaining after removal"):
            topo.remove_residues([to_remove])

    def test_remove_from_multiple_topologies(self):
        """Test removing residues from multiple topologies at once"""
        topo = Partial_Topology.from_range("A", 1, 20, fragment_name="original")
        to_remove1 = Partial_Topology.from_residues("A", [3, 4, 5], fragment_name="remove_first")
        to_remove2 = Partial_Topology.from_residues(
            "A", [10, 11, 12], fragment_name="remove_middle"
        )
        to_remove3 = Partial_Topology.from_residues("A", [18, 19], fragment_name="remove_end")

        result = topo.remove_residues([to_remove1, to_remove2, to_remove3])

        expected_residues = [1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17, 20]
        assert result.residues == expected_residues
        assert result.fragment_name == "original"

    def test_remove_with_peptide_trim(self):
        """Test removing residues with peptide trimming behavior"""
        # Create a peptide with trimming
        peptide = Partial_Topology.from_range("A", 1, 10, peptide=True, peptide_trim=2)
        # Full residues: [1-10], Peptide residues: [3-10]

        # Remove some residues
        to_remove = Partial_Topology.from_residues("A", [3, 4, 7, 8])

        result = peptide.remove_residues([to_remove])

        # Check that metadata is preserved
        assert result.peptide is True
        assert result.peptide_trim == 2

        # Check residues
        assert result.residues == [1, 2, 5, 6, 9, 10]
        # Peptide residues should be recalculated after trim
        assert result.peptide_residues == [5, 6, 9, 10]  # Skips first 2 residues

        # Even with all peptide residues removed, if base residues remain, it should work
        all_peptide_residues = Partial_Topology.from_residues("A", [3, 4, 5, 6, 7, 8, 9, 10])
        result_no_active = peptide.remove_residues([all_peptide_residues])

        # Only has untrimmed residues remaining
        assert result_no_active.residues == [1, 2]
        # No peptide residues after trimming
        assert result_no_active.peptide_residues == []


class TestRankingAndSorting:
    """Test the ranking and sorting functionality"""

    def test_rank_order_chain_priority(self):
        """Test sorting priority by chain ID (length then value)"""
        topo_A = Partial_Topology.from_single("A", 1)
        topo_B = Partial_Topology.from_single("B", 1)
        topo_1 = Partial_Topology.from_single("1", 1)
        topo_10 = Partial_Topology.from_single("10", 1)

        # Shorter chains come first
        assert topo_A.rank_order() < topo_10.rank_order()
        assert topo_1.rank_order() < topo_10.rank_order()

        # For same length, alphanumeric sorting applies
        assert topo_1.rank_order() < topo_A.rank_order()
        assert topo_A.rank_order() < topo_B.rank_order()

    def test_rank_order_residue_position_priority(self):
        """Test sorting priority by average residue position"""
        topo_early = Partial_Topology.from_range("A", 1, 5)  # avg = 3
        topo_late = Partial_Topology.from_range("A", 10, 14)  # avg = 12

        assert topo_early.rank_order() < topo_late.rank_order()

    def test_rank_order_length_priority(self):
        """Test sorting priority by fragment length (longer first)"""
        # Same chain, same average residue position (5)
        topo_short = Partial_Topology.from_range("A", 4, 6)  # length 3
        topo_long = Partial_Topology.from_range("A", 3, 7)  # length 5

        # Longer fragment should come first (smaller rank order value)
        assert topo_long.rank_order() < topo_short.rank_order()

    def test_rank_order_with_peptide_trim(self):
        """Test rank_order respects peptide trimming"""
        # Peptide where trimming changes the average position and length
        peptide = Partial_Topology.from_range("A", 1, 10, peptide=True, peptide_trim=5)
        # Full: [1-10], avg=5.5, len=10
        # Trimmed: [6-10], avg=8, len=5

        rank_full = peptide.rank_order(check_trim=False)
        rank_trimmed = peptide.rank_order(check_trim=True)

        assert rank_full != rank_trimmed
        # Check avg residue position part of the key
        assert rank_full[2] == 5.5
        assert rank_trimmed[2] == 8.0
        # Check length part of the key
        assert rank_full[3] == -10
        assert rank_trimmed[3] == -5

    def test_direct_sorting_with_lt(self):
        """Test direct sorting of topologies using the __lt__ method"""
        topo1 = Partial_Topology.from_range("A", 10, 15)  # avg 12.5, len 6
        topo2 = Partial_Topology.from_range("A", 1, 5)  # avg 3, len 5
        topo3 = Partial_Topology.from_range("B", 1, 5)  # chain B
        topo4 = Partial_Topology.from_range("A", 1, 10)  # avg 5.5, len 10

        topologies = [topo1, topo2, topo3, topo4]
        sorted_topos = sorted(topologies)

        # Expected order based on rank_order():
        # 1. topo2 (A, avg 3, len 5)
        # 2. topo4 (A, avg 5.5, len 10)
        # 3. topo1 (A, avg 12.5, len 6)
        # 4. topo3 (B, chain B is after A)
        expected_order = [topo2, topo4, topo1, topo3]
        assert sorted_topos == expected_order

    def test_rank_and_index_method(self):
        """Test the rank_and_index class method"""
        topo_B1 = Partial_Topology.from_range("B", 1, 5, fragment_name="B1")
        topo_A2 = Partial_Topology.from_range("A", 10, 15, fragment_name="A2")
        topo_A1 = Partial_Topology.from_range("A", 1, 5, fragment_name="A1")

        topologies = [topo_B1, topo_A2, topo_A1]

        # Before ranking, indices are None
        assert all(t.fragment_index is None for t in topologies)

        ranked_topologies = Partial_Topology.rank_and_index(topologies)

        # Check order
        assert ranked_topologies[0].fragment_name == "A1"
        assert ranked_topologies[1].fragment_name == "A2"
        assert ranked_topologies[2].fragment_name == "B1"

        # Check indices
        assert ranked_topologies[0].fragment_index == 0
        assert ranked_topologies[1].fragment_index == 1
        assert ranked_topologies[2].fragment_index == 2

    def test_rank_and_index_stability(self):
        """Test that sorting is stable for equally ranked items"""
        # Create two identical topologies, but they are different objects
        topo1 = Partial_Topology.from_range("A", 1, 5, fragment_name="first")
        topo2 = Partial_Topology.from_range("B", 1, 5, fragment_name="third")
        topo3 = Partial_Topology.from_range("A", 1, 5, fragment_name="second")

        # topo1 and topo3 have identical ranking keys
        assert topo1.rank_order() == topo3.rank_order()

        topologies = [topo1, topo2, topo3]

        # Since sort is stable, topo1 should appear before topo3 in the sorted list
        ranked = Partial_Topology.rank_and_index(topologies)

        # Expected order: topo1, topo3, topo2
        assert ranked[0].fragment_name == "first"
        assert ranked[1].fragment_name == "second"
        assert ranked[2].fragment_name == "third"

        # Check indices
        assert ranked[0].fragment_index == 0
        assert ranked[1].fragment_index == 1
        assert ranked[2].fragment_index == 2


class TestListSerialization:
    """Test list save/load functionality"""

    def test_save_list_to_json_basic(self, tmp_path):
        """Test basic save functionality"""
        topo1 = Partial_Topology.from_range("A", 1, 5, fragment_name="frag1")
        topo2 = Partial_Topology.from_range(
            "B", 10, 15, fragment_name="frag2", peptide=True, peptide_trim=2
        )
        topo3 = Partial_Topology.from_residues("C", [20, 22, 24], fragment_name="scattered")

        topologies = [topo1, topo2, topo3]
        filepath = tmp_path / "test_topologies.json"

        # Save the list
        Partial_Topology.save_list_to_json(topologies, filepath)

        # Verify file exists
        assert filepath.exists()

        # Verify file content structure
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "topology_count" in data
        assert "topologies" in data
        assert data["topology_count"] == 3
        assert len(data["topologies"]) == 3

        # Verify individual topology data
        assert data["topologies"][0]["chain"] == "A"
        assert data["topologies"][0]["fragment_name"] == "frag1"
        assert data["topologies"][1]["peptide"] is True
        assert data["topologies"][1]["peptide_trim"] == 2
        assert data["topologies"][2]["residues"] == [20, 22, 24]

    def test_save_list_to_json_with_path_string(self, tmp_path):
        """Test save with string path instead of Path object"""
        topo = Partial_Topology.from_single("A", 100, fragment_name="single")
        filepath_str = str(tmp_path / "string_path.json")

        Partial_Topology.save_list_to_json([topo], filepath_str)

        assert Path(filepath_str).exists()

    def test_save_list_empty_raises_error(self, tmp_path):
        """Test that saving empty list raises error"""
        filepath = tmp_path / "empty.json"

        with pytest.raises(ValueError, match="Cannot save empty list of topologies"):
            Partial_Topology.save_list_to_json([], filepath)

    def test_save_list_creates_directory(self, tmp_path):
        """Test that save creates parent directories"""
        nested_path = tmp_path / "nested" / "directories" / "test.json"
        topo = Partial_Topology.from_single("A", 1)

        Partial_Topology.save_list_to_json([topo], nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_list_file_permission_error(self, tmp_path):
        """Test handling of file permission errors"""
        topo = Partial_Topology.from_single("A", 1)

        # Try to save to a directory (should fail)
        with pytest.raises(IOError, match="Failed to write topologies"):
            Partial_Topology.save_list_to_json([topo], tmp_path)

    def test_load_list_from_json_basic(self, tmp_path):
        """Test basic load functionality"""
        # Create test data
        original_topos = [
            Partial_Topology.from_range("A", 1, 5, fragment_name="test1", fragment_index=0),
            Partial_Topology.from_residues(
                "B",
                [10, 12, 14],
                fragment_name="test2",
                peptide=True,
                peptide_trim=1,
                fragment_index=1,
            ),
            Partial_Topology.from_single(
                "C", 100, fragment_sequence="M", fragment_name="test3", fragment_index=2
            ),
        ]

        # Save to file
        filepath = tmp_path / "load_test.json"
        Partial_Topology.save_list_to_json(original_topos, filepath)

        # Load from file
        loaded_topos = Partial_Topology.load_list_from_json(filepath)

        # Verify count
        assert len(loaded_topos) == 3

        # Verify content matches
        for orig, loaded in zip(original_topos, loaded_topos):
            assert loaded.chain == orig.chain
            assert loaded.residues == orig.residues
            assert loaded.fragment_name == orig.fragment_name
            assert loaded.fragment_sequence == orig.fragment_sequence
            assert loaded.fragment_index == orig.fragment_index
            assert loaded.peptide == orig.peptide
            assert loaded.peptide_trim == orig.peptide_trim

            # Verify computed properties are recalculated correctly
            assert loaded.length == orig.length
            assert loaded.is_contiguous == orig.is_contiguous
            assert loaded.peptide_residues == orig.peptide_residues

    def test_load_list_from_json_with_path_string(self, tmp_path):
        """Test load with string path instead of Path object"""
        topo = Partial_Topology.from_single("A", 50)
        filepath = tmp_path / "string_load.json"

        Partial_Topology.save_list_to_json([topo], filepath)
        loaded = Partial_Topology.load_list_from_json(str(filepath))

        assert len(loaded) == 1
        assert loaded[0].residues == [50]

    def test_load_list_file_not_found(self, tmp_path):
        """Test handling of missing file"""
        nonexistent_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Topology file not found"):
            Partial_Topology.load_list_from_json(nonexistent_path)

    def test_load_list_invalid_json(self, tmp_path):
        """Test handling of invalid JSON"""
        filepath = tmp_path / "invalid.json"

        # Write invalid JSON
        with open(filepath, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(IOError, match="Failed to read topology file"):
            Partial_Topology.load_list_from_json(filepath)

    def test_load_list_wrong_format_not_dict(self, tmp_path):
        """Test handling of wrong file format (not a dict)"""
        filepath = tmp_path / "wrong_format.json"

        # Write valid JSON but wrong format
        with open(filepath, "w") as f:
            json.dump(["not", "a", "dict"], f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: root must be a dictionary"
        ):
            Partial_Topology.load_list_from_json(filepath)

    def test_load_list_missing_topologies_key(self, tmp_path):
        """Test handling of missing 'topologies' key"""
        filepath = tmp_path / "missing_key.json"

        with open(filepath, "w") as f:
            json.dump({"topology_count": 1, "wrong_key": []}, f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: missing 'topologies' key"
        ):
            Partial_Topology.load_list_from_json(filepath)

    def test_load_list_topologies_not_list(self, tmp_path):
        """Test handling of 'topologies' not being a list"""
        filepath = tmp_path / "topologies_not_list.json"

        with open(filepath, "w") as f:
            json.dump({"topologies": "not a list"}, f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: 'topologies' must be a list"
        ):
            Partial_Topology.load_list_from_json(filepath)

    def test_load_list_count_mismatch(self, tmp_path):
        """Test handling of count mismatch"""
        filepath = tmp_path / "count_mismatch.json"

        # Create valid topology data but wrong count
        topo_data = Partial_Topology.from_single("A", 1).to_dict()
        data = {
            "topology_count": 5,  # Wrong count
            "topologies": [topo_data, topo_data],  # Only 2 topologies
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Topology count mismatch: expected 5, found 2"):
            Partial_Topology.load_list_from_json(filepath)

    def test_load_list_invalid_topology_data(self, tmp_path):
        """Test handling of invalid topology data"""
        filepath = tmp_path / "invalid_topology.json"

        # Create data with invalid topology (missing required field)
        invalid_topo_data = {"chain": "A"}  # Missing residues
        data = {"topology_count": 1, "topologies": [invalid_topo_data]}

        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Failed to parse topology data"):
            Partial_Topology.load_list_from_json(filepath)

    def test_save_load_round_trip_complex(self, tmp_path):
        """Test complete round trip with complex topologies"""
        # Create diverse set of topologies
        topologies = [
            Partial_Topology.from_range(
                "A", 1, 10, "MKLIVQWERT", "signal", peptide=True, peptide_trim=3
            ),
            Partial_Topology.from_residues(
                "B", [15, 17, 19, 21], ["M", "K", "L", "I"], "active_site"
            ),
            Partial_Topology.from_single("C", 100, "W", "binding_site"),
            Partial_Topology.from_range("A", 200, 250, fragment_name="domain", fragment_index=5),
            Partial_Topology.from_residues(
                "D", list(range(1, 101, 5)), fragment_name="large_scattered"
            ),
        ]

        filepath = tmp_path / "complex_round_trip.json"

        # Save
        Partial_Topology.save_list_to_json(topologies, filepath)

        # Load
        loaded_topologies = Partial_Topology.load_list_from_json(filepath)

        # Verify everything matches
        assert len(loaded_topologies) == len(topologies)

        for orig, loaded in zip(topologies, loaded_topologies):
            # Core attributes
            assert loaded.chain == orig.chain
            assert loaded.residues == orig.residues
            assert loaded.fragment_sequence == orig.fragment_sequence
            assert loaded.fragment_name == orig.fragment_name
            assert loaded.fragment_index == orig.fragment_index
            assert loaded.peptide == orig.peptide
            assert loaded.peptide_trim == orig.peptide_trim

            # Computed properties
            assert loaded.residue_start == orig.residue_start
            assert loaded.residue_end == orig.residue_end
            assert loaded.length == orig.length
            assert loaded.is_contiguous == orig.is_contiguous
            assert loaded.peptide_residues == orig.peptide_residues

    def test_save_load_empty_fields(self, tmp_path):
        """Test save/load with empty or None fields"""
        topo = Partial_Topology.from_residues(
            chain="A",
            residues=[1, 2, 3],
            fragment_sequence="",  # Empty sequence
            fragment_name="test",
            fragment_index=None,  # None index
            peptide=False,
            peptide_trim=0,
        )

        filepath = tmp_path / "empty_fields.json"

        # Save and load
        Partial_Topology.save_list_to_json([topo], filepath)
        loaded = Partial_Topology.load_list_from_json(filepath)

        assert len(loaded) == 1
        loaded_topo = loaded[0]

        assert loaded_topo.fragment_sequence == ""
        assert loaded_topo.fragment_index is None
        assert loaded_topo.peptide is False
        assert loaded_topo.peptide_trim == 0

    def test_load_list_without_count_field(self, tmp_path):
        """Test loading file without topology_count field (should still work)"""
        topo = Partial_Topology.from_single("A", 42)

        # Manually create JSON without count field
        data = {"topologies": [topo.to_dict()]}
        filepath = tmp_path / "no_count.json"

        with open(filepath, "w") as f:
            json.dump(data, f)

        # Should load successfully
        loaded = Partial_Topology.load_list_from_json(filepath)
        assert len(loaded) == 1
        assert loaded[0].residues == [42]

    def test_save_load_large_list(self, tmp_path):
        """Test save/load with large number of topologies"""
        # Create 100 topologies
        topologies = []
        for i in range(100):
            chain = chr(ord("A") + (i % 26))  # Cycle through A-Z
            start_res = i * 10 + 1
            end_res = start_res + 5
            topo = Partial_Topology.from_range(
                chain=chain,
                start=start_res,
                end=end_res,
                fragment_name=f"frag_{i}",
                fragment_index=i,
            )
            topologies.append(topo)

        filepath = tmp_path / "large_list.json"

        # Save and load
        Partial_Topology.save_list_to_json(topologies, filepath)
        loaded = Partial_Topology.load_list_from_json(filepath)

        assert len(loaded) == 100

        # Spot check a few
        assert loaded[0].fragment_name == "frag_0"
        assert loaded[0].fragment_index == 0
        assert loaded[50].fragment_name == "frag_50"
        assert loaded[50].fragment_index == 50
        assert loaded[99].fragment_name == "frag_99"
        assert loaded[99].fragment_index == 99

    def test_save_load_unicode_content(self, tmp_path):
        """Test save/load with unicode characters"""
        topo = Partial_Topology.from_single(
            chain="A",
            residue=1,
            fragment_sequence="α",  # Greek alpha
            fragment_name="π_helix",  # Greek pi
        )

        filepath = tmp_path / "unicode.json"

        # Save and load
        Partial_Topology.save_list_to_json([topo], filepath)
        loaded = Partial_Topology.load_list_from_json(filepath)

        assert len(loaded) == 1
        assert loaded[0].fragment_sequence == "α"
        assert loaded[0].fragment_name == "π_helix"
