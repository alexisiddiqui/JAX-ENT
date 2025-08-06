import json
from pathlib import Path

import pytest

# Import the class being tested
# from your_module import Partial_Topology
# For this example, assume the class is in the same file or properly imported
from jaxent.src.interfaces.topology import (
    Partial_Topology,
    PTSerialiser,
    TopologyFactory,
)


class TestListSerialization:
    """Test list save/load functionality"""

    def test_save_list_to_json_basic(self, tmp_path):
        """Test basic save functionality"""
        topo1 = TopologyFactory.from_range("A", 1, 5, fragment_name="frag1")
        topo2 = TopologyFactory.from_range(
            "B", 10, 15, fragment_name="frag2", peptide=True, peptide_trim=2
        )
        topo3 = TopologyFactory.from_residues("C", [20, 22, 24], fragment_name="scattered")

        topologies = [topo1, topo2, topo3]
        filepath = tmp_path / "test_topologies.json"

        # Save the list
        PTSerialiser.save_list_to_json(topologies, filepath)

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
        topo = TopologyFactory.from_single("A", 100, fragment_name="single")
        filepath_str = str(tmp_path / "string_path.json")

        PTSerialiser.save_list_to_json([topo], filepath_str)

        assert Path(filepath_str).exists()

    def test_save_list_empty_raises_error(self, tmp_path):
        """Test that saving empty list raises error"""
        filepath = tmp_path / "empty.json"

        with pytest.raises(ValueError, match="Cannot save empty list of topologies"):
            PTSerialiser.save_list_to_json([], filepath)

    def test_save_list_creates_directory(self, tmp_path):
        """Test that save creates parent directories"""
        nested_path = tmp_path / "nested" / "directories" / "test.json"
        topo = TopologyFactory.from_single("A", 1)

        PTSerialiser.save_list_to_json([topo], nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_list_file_permission_error(self, tmp_path):
        """Test handling of file permission errors"""
        topo = TopologyFactory.from_single("A", 1)

        # Try to save to a directory (should fail)
        with pytest.raises(IOError, match="Failed to write topologies"):
            PTSerialiser.save_list_to_json([topo], tmp_path)

    def test_load_list_from_json_basic(self, tmp_path):
        """Test basic load functionality"""
        # Create test data
        original_topos = [
            TopologyFactory.from_range("A", 1, 5, fragment_name="test1", fragment_index=0),
            TopologyFactory.from_residues(
                "B",
                [10, 12, 14],
                fragment_name="test2",
                peptide=True,
                peptide_trim=1,
                fragment_index=1,
            ),
            TopologyFactory.from_single(
                "C", 100, fragment_sequence="M", fragment_name="test3", fragment_index=2
            ),
        ]

        # Save to file
        filepath = tmp_path / "load_test.json"
        PTSerialiser.save_list_to_json(original_topos, filepath)

        # Load from file
        loaded_topos = PTSerialiser.load_list_from_json(filepath)

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
        topo = TopologyFactory.from_single("A", 50)
        filepath = tmp_path / "string_load.json"

        PTSerialiser.save_list_to_json([topo], filepath)
        loaded = PTSerialiser.load_list_from_json(str(filepath))

        assert len(loaded) == 1
        assert loaded[0].residues == [50]

    def test_load_list_file_not_found(self, tmp_path):
        """Test handling of missing file"""
        nonexistent_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Topology file not found"):
            PTSerialiser.load_list_from_json(nonexistent_path)

    def test_load_list_invalid_json(self, tmp_path):
        """Test handling of invalid JSON"""
        filepath = tmp_path / "invalid.json"

        # Write invalid JSON
        with open(filepath, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(IOError, match="Failed to read topology file"):
            PTSerialiser.load_list_from_json(filepath)

    def test_load_list_wrong_format_not_dict(self, tmp_path):
        """Test handling of wrong file format (not a dict)"""
        filepath = tmp_path / "wrong_format.json"

        # Write valid JSON but wrong format
        with open(filepath, "w") as f:
            json.dump(["not", "a", "dict"], f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: root must be a dictionary"
        ):
            PTSerialiser.load_list_from_json(filepath)

    def test_load_list_missing_topologies_key(self, tmp_path):
        """Test handling of missing 'topologies' key"""
        filepath = tmp_path / "missing_key.json"

        with open(filepath, "w") as f:
            json.dump({"topology_count": 1, "wrong_key": []}, f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: missing 'topologies' key"
        ):
            PTSerialiser.load_list_from_json(filepath)

    def test_load_list_topologies_not_list(self, tmp_path):
        """Test handling of 'topologies' not being a list"""
        filepath = tmp_path / "topologies_not_list.json"

        with open(filepath, "w") as f:
            json.dump({"topologies": "not a list"}, f)

        with pytest.raises(
            ValueError, match="Invalid topology file format: 'topologies' must be a list"
        ):
            PTSerialiser.load_list_from_json(filepath)

    def test_load_list_count_mismatch(self, tmp_path):
        """Test handling of count mismatch"""
        filepath = tmp_path / "count_mismatch.json"

        # Create valid topology data but wrong count
        topo_data = TopologyFactory.from_single("A", 1)._to_dict()
        data = {
            "topology_count": 5,  # Wrong count
            "topologies": [topo_data, topo_data],  # Only 2 topologies
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Topology count mismatch: expected 5, found 2"):
            PTSerialiser.load_list_from_json(filepath)

    def test_load_list_invalid_topology_data(self, tmp_path):
        """Test handling of invalid topology data"""
        filepath = tmp_path / "invalid_topology.json"

        # Create data with invalid topology (missing required field)
        invalid_topo_data = {"chain": "A"}  # Missing residues
        data = {"topology_count": 1, "topologies": [invalid_topo_data]}

        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="Failed to parse topology data"):
            PTSerialiser.load_list_from_json(filepath)

    def test_save_load_round_trip_complex(self, tmp_path):
        """Test complete round trip with complex topologies"""
        # Create diverse set of topologies
        topologies = [
            TopologyFactory.from_range(
                "A", 1, 10, "MKLIVQWERT", "signal", peptide=True, peptide_trim=3
            ),
            TopologyFactory.from_residues(
                "B", [15, 17, 19, 21], ["M", "K", "L", "I"], "active_site"
            ),
            TopologyFactory.from_single("C", 100, "W", "binding_site"),
            TopologyFactory.from_range("A", 200, 250, fragment_name="domain", fragment_index=5),
            TopologyFactory.from_residues(
                "D", list(range(1, 101, 5)), fragment_name="large_scattered"
            ),
        ]

        filepath = tmp_path / "complex_round_trip.json"

        # Save
        PTSerialiser.save_list_to_json(topologies, filepath)

        # Load
        loaded_topologies = PTSerialiser.load_list_from_json(filepath)

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
        topo = TopologyFactory.from_residues(
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
        PTSerialiser.save_list_to_json([topo], filepath)
        loaded = PTSerialiser.load_list_from_json(filepath)

        assert len(loaded) == 1
        loaded_topo = loaded[0]

        assert loaded_topo.fragment_sequence == ""
        assert loaded_topo.fragment_index is None
        assert loaded_topo.peptide is False
        assert loaded_topo.peptide_trim == 0

    def test_load_list_without_count_field(self, tmp_path):
        """Test loading file without topology_count field (should still work)"""
        topo = TopologyFactory.from_single("A", 42)

        # Manually create JSON without count field
        data = {"topologies": [topo._to_dict()]}
        filepath = tmp_path / "no_count.json"

        with open(filepath, "w") as f:
            json.dump(data, f)

        # Should load successfully
        loaded = PTSerialiser.load_list_from_json(filepath)
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
            topo = TopologyFactory.from_range(
                chain=chain,
                start=start_res,
                end=end_res,
                fragment_name=f"frag_{i}",
                fragment_index=i,
            )
            topologies.append(topo)

        filepath = tmp_path / "large_list.json"

        # Save and load
        PTSerialiser.save_list_to_json(topologies, filepath)
        loaded = PTSerialiser.load_list_from_json(filepath)

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
        topo = TopologyFactory.from_single(
            chain="A",
            residue=1,
            fragment_sequence="α",  # Greek alpha
            fragment_name="π_helix",  # Greek pi
        )

        filepath = tmp_path / "unicode.json"

        # Save and load
        PTSerialiser.save_list_to_json([topo], filepath)
        loaded = PTSerialiser.load_list_from_json(filepath)

        assert len(loaded) == 1
        assert loaded[0].fragment_sequence == "α"
        assert loaded[0].fragment_name == "π_helix"


class TestSerialization:
    """Test JSON serialization and deserialization"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        topo = TopologyFactory.from_range(
            chain="A",
            start=1,
            end=5,
            fragment_sequence="MKLIV",
            fragment_name="test_fragment",
            fragment_index=42,
            peptide=True,
            peptide_trim=2,
        )

        data = topo._to_dict()
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

        topo = Partial_Topology._from_dict(data)

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12]
        assert topo.fragment_sequence == "GLY"
        assert topo.fragment_name == "test"
        assert topo.fragment_index is None
        assert topo.peptide is False
        assert topo.peptide_trim == 0

        data["peptide"] = True

        topo = Partial_Topology._from_dict(data)

        assert topo.chain == "B"
        assert topo.residues == [10, 11, 12]
        assert topo.fragment_sequence == "GLY"
        assert topo.fragment_name == "test"
        assert topo.fragment_index is None
        assert topo.peptide is True
        assert topo.peptide_trim == 2

    def test_json_round_trip(self):
        """Test JSON serialization round trip"""
        original = TopologyFactory.from_residues(
            chain="C",
            residues=[1, 3, 5, 7],
            fragment_sequence=["M", "K", "L", "I"],
            fragment_name="scattered_fragment",
            fragment_index=123,
            peptide=True,
            peptide_trim=1,
        )

        # Serialize to JSON
        json_str = PTSerialiser.to_json(original)

        # Deserialize from JSON
        restored = PTSerialiser.from_json(json_str)

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
        topo = TopologyFactory.from_single("A", 1, "M", "test")
        json_str = PTSerialiser.to_json(topo)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Should be formatted (contain newlines)
        assert "\n" in json_str
