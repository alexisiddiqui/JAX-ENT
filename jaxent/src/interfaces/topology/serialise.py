import json
from pathlib import Path
from typing import Union

from jaxent.src.interfaces.topology.core import Partial_Topology


class PTSerialiser:
    """Function container for serializing and deserializing topology objects"""

    ### Serialization
    @staticmethod
    def to_json(top: Partial_Topology) -> str:
        """Serialize to JSON string"""
        return json.dumps(top._to_dict(), indent=2)

    @staticmethod
    def from_json(json_str: str) -> Partial_Topology:
        """Deserialize from JSON string"""
        return Partial_Topology._from_dict(json.loads(json_str))

    @staticmethod
    def save_list_to_json(topologies: list[Partial_Topology], filepath: Union[str, Path]) -> None:
        """Save a list of Partial_Topology objects to a JSON file

        Args:
            topologies: List of Partial_Topology objects to save
            filepath: Path to the output JSON file

        Raises:
            ValueError: If topologies list is empty
            IOError: If file cannot be written
        """
        if not topologies:
            raise ValueError("Cannot save empty list of topologies")

        filepath = Path(filepath)

        # Convert all topologies to dictionaries
        topology_dicts = [topo._to_dict() for topo in topologies]

        # Create output data structure
        output_data = {"topology_count": len(topology_dicts), "topologies": topology_dicts}

        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            raise IOError(f"Failed to write topologies to {filepath}: {e}")

    @staticmethod
    def load_list_from_json(filepath: Union[str, Path]) -> list[Partial_Topology]:
        """Load a list of Partial_Topology objects from a JSON file

        Args:
            filepath: Path to the JSON file to load

        Returns:
            List of Partial_Topology objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
            IOError: If file cannot be read
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Topology file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise IOError(f"Failed to read topology file {filepath}: {e}")

        # Validate file format
        if not isinstance(data, dict):
            raise ValueError("Invalid topology file format: root must be a dictionary")

        if "topologies" not in data:
            raise ValueError("Invalid topology file format: missing 'topologies' key")

        if not isinstance(data["topologies"], list):
            raise ValueError("Invalid topology file format: 'topologies' must be a list")

        # Optional validation of count
        if "topology_count" in data:
            expected_count = data["topology_count"]
            actual_count = len(data["topologies"])
            if expected_count != actual_count:
                raise ValueError(
                    f"Topology count mismatch: expected {expected_count}, found {actual_count}"
                )

        # Convert dictionaries back to Partial_Topology objects
        try:
            topologies = [
                Partial_Topology._from_dict(topo_dict) for topo_dict in data["topologies"]
            ]
        except Exception as e:
            raise ValueError(f"Failed to parse topology data: {e}")

        return topologies
