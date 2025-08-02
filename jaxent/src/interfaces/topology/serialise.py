
    ### Serialization
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self._to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Partial_Topology":
        """Deserialize from JSON string"""
        return cls._from_dict(json.loads(json_str))

    @classmethod
    def save_list_to_json(
        cls, topologies: list["Partial_Topology"], filepath: Union[str, Path]
    ) -> None:
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

    @classmethod
    def load_list_from_json(cls, filepath: Union[str, Path]) -> list["Partial_Topology"]:
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
            topologies = [cls._from_dict(topo_dict) for topo_dict in data["topologies"]]
        except Exception as e:
            raise ValueError(f"Failed to parse topology data: {e}")

        return topologies
