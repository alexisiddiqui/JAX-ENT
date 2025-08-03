import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.groups import Residue
from tqdm import tqdm

from jaxent.src.models.func.common import compute_trajectory_average_com_distances


@dataclass
class Partial_Topology:
    """
    Flexible dataclass for biophysical topology fragments.
    Can represent contiguous ranges or arbitrary residue selections.
    Fully JSON-serializable.
    """

    chain: Union[str, int] = "A"  # Chain identifier (string or integer)
    residues: list[int] = field(default_factory=list)  # Canonical representation
    fragment_sequence: Union[str, list[str]] = ""  # Actual amino acid sequence
    fragment_name: str = "seg"  # Fragment identifier/name
    fragment_index: Optional[int] = None
    peptide: bool = False  # Whether this is treated as a peptide
    peptide_trim: Optional[int] = None  # Residues to trim from start when peptide=True

    # Computed properties (not stored in JSON)
    residue_start: int = field(init=False, repr=False)
    residue_end: int = field(init=False, repr=False)
    length: int = field(init=False, repr=False)
    is_contiguous: bool = field(init=False, repr=False)
    peptide_residues: list[int] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        """Compute derived properties from the residue list"""
        if not self.residues:
            raise ValueError("At least one residue must be specified")

        # Ensure chain is capitalized if it's a string
        if isinstance(self.chain, str):
            self.chain = self.chain.upper()
        elif isinstance(self.chain, int):
            self.chain = str(self.chain)
        else:
            raise TypeError("Chain must be a string or integer")

        # Sort residues and remove duplicates
        self.residues = sorted(set(self.residues))

        # Compute basic properties
        object.__setattr__(self, "residue_start", min(self.residues))
        object.__setattr__(self, "residue_end", max(self.residues))
        object.__setattr__(self, "length", len(self.residues))

        # Check if residues are contiguous
        expected_range = list(range(self.residue_start, self.residue_end + 1))
        object.__setattr__(self, "is_contiguous", self.residues == expected_range)

        if not self.peptide:
            self.peptide_trim = 0  # No trimming for non-peptides

        # Compute peptide residues only if peptide=True and length > trim
        if self.peptide and self.length > self.peptide_trim:
            peptide_res = self.residues[self.peptide_trim :]
            object.__setattr__(self, "peptide_residues", peptide_res)
        else:
            object.__setattr__(self, "peptide_residues", [])

    def set_peptide(self, val: bool, trim: int = None) -> None:
        """Update peptide settings and recompute peptide residues"""
        self.peptide = val
        if trim is not None:
            self.peptide_trim = trim

        # Recompute peptide residues
        if self.peptide and self.length > self.peptide_trim:
            peptide_res = self.residues[self.peptide_trim :]
            object.__setattr__(self, "peptide_residues", peptide_res)
        else:
            object.__setattr__(self, "peptide_residues", [])

    @classmethod
    def from_range(
        cls,
        chain: Union[str, int],
        start: int,
        end: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: Optional[int] = None,
    ) -> "Partial_Topology":
        """Create from a contiguous range of residues"""
        residues = list(range(int(start), int(end) + 1))
        return cls(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @classmethod
    def from_residues(
        cls,
        chain: Union[str, int],
        residues: list[int],
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: Optional[int] = None,
    ) -> "Partial_Topology":
        """Create from an arbitrary list of residues"""
        return cls(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @classmethod
    def from_single(
        cls,
        chain: Union[str, int],
        residue: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: Optional[int] = None,
    ) -> "Partial_Topology":
        """Create from a single residue"""
        return cls(
            chain=chain,
            residues=[residue],
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        # Include all the core data fields that should be persisted
        return {
            "chain": self.chain,
            "residues": self.residues,
            "fragment_sequence": self.fragment_sequence,
            "fragment_name": self.fragment_name,
            "fragment_index": self.fragment_index,
            "peptide": self.peptide,
            "peptide_trim": self.peptide_trim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Partial_Topology":
        """Create from dictionary (e.g., from JSON)"""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Partial_Topology":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))

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
        topology_dicts = [topo.to_dict() for topo in topologies]

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
            topologies = [cls.from_dict(topo_dict) for topo_dict in data["topologies"]]
        except Exception as e:
            raise ValueError(f"Failed to parse topology data: {e}")

        return topologies

    def __str__(self) -> str:
        """Pretty string representation"""
        index_part = f"{self.fragment_index}:" if self.fragment_index is not None else "None:"

        if self.length == 1:
            desc = f"{index_part}{self.fragment_name}:{self.chain}:{self.residues[0]}"
        elif self.is_contiguous:
            desc = f"{index_part}{self.fragment_name}:{self.chain}:{self.residue_start}-{self.residue_end}"
        else:
            # Show first few and last few residues for non-contiguous
            if len(self.residues) <= 6:
                res_str = ",".join(map(str, self.residues))
            else:
                res_str = f"{','.join(map(str, self.residues[:3]))}...{','.join(map(str, self.residues[-3:]))}"
            desc = f"{index_part}{self.fragment_name}:{self.chain}:[{res_str}]"

        if self.peptide and self.peptide_residues:
            desc += f" [peptide: {len(self.peptide_residues)} residues]"

        return desc

    def __repr__(self) -> str:
        """Detailed representation with all attributes"""
        attrs = []
        for field_name, field_value in self.__dict__.items():
            # Skip computed attributes for cleaner output
            if field_name in (
                "residue_start",
                "residue_end",
                "length",
                "is_contiguous",
                "peptide_residues",
            ):
                continue
            if isinstance(field_value, str):
                attrs.append(f"{field_name}='{field_value}'")
            else:
                attrs.append(f"{field_name}={field_value}")
        return f"Partial_Topology({', '.join(attrs)})"

    def __lt__(self, other: Any) -> bool:
        """
        Compare this topology with another for sorting.

        Comparison is based on the `rank_order` method (with default settings).
        This allows direct sorting of Partial_Topology objects.
        e.g., `sorted(list_of_topologies)`
        """
        if not isinstance(other, Partial_Topology):
            return NotImplemented
        # Default to check_trim=False for direct comparison
        return self.rank_order() < other.rank_order()

    def __hash__(self) -> int:
        """Allow topologies to be added to sets"""
        # Hash based on
        return hash((self.chain, tuple(self.peptide_residues if self.peptide else self.residues)))

    def _get_chain_score(self) -> tuple:
        """
        Generates a score for a chain ID to allow for proper sorting.
        Digits are scored lower than letters.
        e.g., '1' < 'A', 'B1' < 'C0'
        """
        # Create a tuple of ordinal values for each character.
        # This correctly handles alphanumeric sorting when comparing tuples.
        return tuple(ord(c) for c in str(self.chain))

    def rank_order(self, check_trim: bool = False) -> tuple:
        """
        Generate a sort key for ranking topologies.

        Objects are sorted by:
        1. Chain ID (shorter chains first, then alphanumerically).
        2. Average position of active residues (lower first).
        3. Length of the fragment (longer fragments first, descending).

        Args:
            check_trim: If True, use peptide_residues for ranking.

        Returns:
            A tuple that can be used as a sort key.
        """
        active_residues = self._get_active_residues(check_trim)
        if not active_residues:
            # Handle cases with no active residues to avoid errors
            avg_residue = float("inf")
            length = 0
        else:
            avg_residue = sum(active_residues) / len(active_residues)
            length = len(active_residues)

        # Sort key: (chain_len, chain_score, avg_residue_pos, -length)
        # Negative length to sort longer fragments first (descending).
        return (len(str(self.chain)), self._get_chain_score(), avg_residue, -length)

    @classmethod
    def rank_and_index(
        cls, topologies: list["Partial_Topology"], check_trim: bool = False
    ) -> list["Partial_Topology"]:
        """
        Sorts a list of topologies and assigns fragment_index.

        The list is sorted based on the key from the `rank_order` method.
        The sort is stable, so equally ranked objects retain their
        relative order from the input list. After sorting, the
        `fragment_index` of each topology is updated to its
        position in the sorted list.

        Args:
            topologies: A list of Partial_Topology objects to sort and index.
            check_trim: If True, use peptide_residues for ranking.

        Returns:
            The sorted and indexed list of topologies.
        """
        # Sort the list using the rank_order method as the key.
        # Python's sort is stable.
        topologies.sort(key=lambda t: t.rank_order(check_trim=check_trim))

        # Assign fragment_index based on the new order
        for i, topo in enumerate(topologies):
            topo.fragment_index = i

        return topologies

    def get_residue_ranges(self) -> list[tuple[int, int]]:
        """Get contiguous ranges within the residue list"""
        if not self.residues:
            return []

        ranges = []
        start = self.residues[0]
        prev = start

        for residue in self.residues[1:]:
            if residue != prev + 1:
                ranges.append((start, prev))
                start = residue
            prev = residue

        ranges.append((start, prev))
        return ranges

    def extract_residues(self, use_peptide_trim: bool = True) -> list["Partial_Topology"]:
        """Extract individual residues

        Args:
            use_peptide_trim: If True and this is a peptide, use peptide_residues.
                             If False, use all residues regardless of peptide settings.
        """
        if self.length == 1:
            return [self]

        if use_peptide_trim and self.peptide and self.peptide_residues:
            residues_to_extract = self.peptide_residues
        else:
            residues_to_extract = self.residues

        return [
            Partial_Topology.from_single(
                chain=self.chain,
                residue=res,
                fragment_sequence=self.fragment_sequence,
                fragment_name=self.fragment_name,
                fragment_index=self.fragment_index,
                peptide=False,  # Individual residues are not peptides
                peptide_trim=self.peptide_trim,
            )
            for res in residues_to_extract
        ]

    def remove_residues(self, topologies_to_remove: list["Partial_Topology"]) -> "Partial_Topology":
        """Remove residues from this topology based on other topologies

        Args:
            topologies_to_remove: List of Partial_Topology objects whose residues
                              should be removed from this topology

        Returns:
            A new Partial_Topology with the specified residues removed

        Raises:
            ValueError: If any topology has a different chain than this one
            ValueError: If no residues would remain after removal
        """
        # Verify all topologies are on the same chain
        for topo in topologies_to_remove:
            if topo.chain != self.chain:
                raise ValueError(
                    f"Cannot remove residues from different chain: {topo.chain} != {self.chain}"
                )

        # Collect all residues to remove
        residues_to_remove = set()
        for topo in topologies_to_remove:
            residues_to_remove.update(topo.residues)

        # Create new residue list with specified residues removed
        new_residues = [res for res in self.residues if res not in residues_to_remove]

        # If no residues left, raise error
        if not new_residues:
            raise ValueError("No residues remaining after removal")

        # Create new topology with remaining residues
        return Partial_Topology.from_residues(
            chain=self.chain,
            residues=new_residues,
            fragment_sequence=self.fragment_sequence,
            fragment_name=self.fragment_name,
            fragment_index=self.fragment_index,
            peptide=self.peptide,
            peptide_trim=self.peptide_trim,
        )

    def _get_active_residues(self, check_trim: bool = False) -> list[int]:
        """Get the active residues for this topology

        Args:
            check_trim: If True and this is a peptide, return peptide_residues.
                       If False, return all residues.
        """
        if check_trim and self.peptide and self.peptide_residues:
            return self.peptide_residues
        return self.residues

    def contains_residue(
        self, residues: Union[int, list[int]], check_trim: bool = False
    ) -> Union[bool, dict[int, bool]]:
        """Check if this topology contains specific residue(s)

        Args:
            residues: Single residue number or list of residue numbers
            check_trim: If True and this is a peptide, check against peptide_residues

        Returns:
            For single residue: bool
            For multiple residues: dict mapping residue -> bool
        """
        active_residues = self._get_active_residues(check_trim)

        if isinstance(residues, int):
            return residues in active_residues
        else:
            return {res: res in active_residues for res in residues}

    def contains_all_residues(self, residues: list[int], check_trim: bool = False) -> bool:
        """Check if this topology contains ALL of the specified residues

        Args:
            residues: List of residue numbers to check
            check_trim: If True and this is a peptide, check against peptide_residues
        """
        active_residues = self._get_active_residues(check_trim)
        return all(res in active_residues for res in residues)

    def contains_any_residues(self, residues: list[int], check_trim: bool = False) -> bool:
        """Check if this topology contains ANY of the specified residues

        Args:
            residues: List of residue numbers to check
            check_trim: If True and this is a peptide, check against peptide_residues
        """
        active_residues = self._get_active_residues(check_trim)
        return any(res in active_residues for res in residues)

    def intersects(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology has any residues in common with another

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return bool(set(self_residues) & set(other_residues))

    def contains_topology(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology fully contains another topology

        Args:
            other: The other topology to check
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(other_residues).issubset(set(self_residues))

    def is_subset_of(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is a subset of another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(self_residues).issubset(set(other_residues))

    def is_superset_of(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is a superset of another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        return other.is_subset_of(self, check_trim)

    def get_overlap(self, other: "Partial_Topology", check_trim: bool = False) -> list[int]:
        """Get the residues that overlap between this and another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return []

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(self_residues) & set(other_residues))

    def get_difference(self, other: "Partial_Topology", check_trim: bool = False) -> list[int]:
        """Get residues in this topology but not in the other

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return self._get_active_residues(check_trim).copy()

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(self_residues) - set(other_residues))

    def is_adjacent_to(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is adjacent to another (residues are consecutive)

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        self_end = max(self_residues)
        self_start = min(self_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        return self_end + 1 == other_start or other_end + 1 == self_start

    def get_gap_to(self, other: "Partial_Topology", check_trim: bool = False) -> Optional[int]:
        """Get the gap size between this and another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides

        Returns:
            None if topologies overlap or are on different chains
            0 if topologies are adjacent
            Positive integer for the gap size
        """
        if self.chain != other.chain:
            return None

        if self.intersects(other, check_trim):
            return None  # They overlap

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        self_end = max(self_residues)
        self_start = min(self_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        if self_end < other_start:
            gap = other_start - self_end - 1
        elif other_end < self_start:
            gap = self_start - other_end - 1
        else:
            return None  # Shouldn't reach here if logic is correct

        return gap

    @classmethod
    def merge(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
        intersection: bool = False,
    ) -> "Partial_Topology":
        """Merge multiple topologies into a single topology

        Args:
            topologies: List of Partial_Topology objects to merge
            trim: If True, respect peptide trimming of input topologies.
                    If False, use all residues and create non-peptide result.
            merged_name: Name for the merged topology (default: concatenate names)
            merged_sequence: Sequence for merged topology (default: concatenate sequences)
            merged_index: Index for merged topology (default: use first topology's index)
            intersection: If True, only include residues that are present in all topologies

        Returns:
            New Partial_Topology containing merged residues

        Raises:
            ValueError: If topologies list is empty or topologies have different chains
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")

        # Validate all topologies are on the same chain
        first_chain = topologies[0].chain
        for topo in topologies:
            if topo.chain != first_chain:
                raise ValueError(
                    f"All topologies must be on the same chain. "
                    f"Found chains: {set(t.chain for t in topologies)}"
                )

        # Process residues based on intersection or union and trim setting
        if intersection:
            # Start with first topology's residues for intersection
            all_residues = set(topologies[0]._get_active_residues(check_trim=trim))

            # Intersect with each subsequent topology
            for topo in topologies[1:]:
                residues = set(topo._get_active_residues(check_trim=trim))
                all_residues.intersection_update(residues)
        else:
            # Perform union of all residues (original behavior)
            all_residues = set()
            for topo in topologies:
                # Use check_trim=trim to respect peptide trimming when requested
                residues = topo._get_active_residues(check_trim=trim)
                all_residues.update(residues)

        # If intersection is empty, raise a more informative error
        if intersection and not all_residues:
            raise ValueError("No common residues found between all topologies")

        merged_residues = sorted(all_residues)

        # Handle merged metadata
        if merged_name is None:
            # Concatenate unique names
            names = [t.fragment_name for t in topologies if t.fragment_name != "seg"]
            if names:
                merged_name = "+".join(dict.fromkeys(names))  # Remove duplicates, preserve order
            else:
                merged_name = "merged"

        if merged_sequence is None:
            # Concatenate sequences if they're strings, otherwise use first non-empty
            sequences = [t.fragment_sequence for t in topologies if t.fragment_sequence]
            if sequences:
                if all(isinstance(seq, str) for seq in sequences):
                    merged_sequence = "".join(sequences)
                else:
                    # If mixed types or lists, use first sequence
                    merged_sequence = sequences[0]
            else:
                merged_sequence = ""

        if merged_index is None:
            # Use first topology's index that isn't None
            for topo in topologies:
                if topo.fragment_index is not None:
                    merged_index = topo.fragment_index
                    break

        # Determine peptide settings for merged result
        if trim:
            # If we respected trimming during merge, result can be a peptide
            # Use the minimum trim value from peptide inputs, or 0 if no peptides
            peptide_trims = [t.peptide_trim for t in topologies if t.peptide]
            merged_peptide = any(t.peptide for t in topologies)
            merged_peptide_trim = min(peptide_trims) if peptide_trims else 0
        else:
            # If trim=False, result is not a peptide
            merged_peptide = False
            merged_peptide_trim = 0

        return cls.from_residues(
            chain=first_chain,
            residues=merged_residues,
            fragment_sequence=merged_sequence,
            fragment_name=merged_name,
            fragment_index=merged_index,
            peptide=merged_peptide,
            peptide_trim=merged_peptide_trim,
        )

    @classmethod
    def merge_contiguous(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        gap_tolerance: int = 0,
        **kwargs,
    ) -> "Partial_Topology":
        """Merge topologies that are contiguous or nearly contiguous

        Args:
            topologies: List of topologies to merge (will be sorted by residue position)
            trim: Whether to respect peptide trimming
            gap_tolerance: Maximum gap size to allow between topologies (0 = must be adjacent)
            **kwargs: Additional arguments passed to merge()

        Returns:
            Merged topology

        Raises:
            ValueError: If topologies are not contiguous within gap_tolerance
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")

        # Sort topologies by their start position
        sorted_topos = sorted(topologies, key=lambda t: min(t._get_active_residues(trim)))

        # Check contiguity
        for i in range(len(sorted_topos) - 1):
            current = sorted_topos[i]
            next_topo = sorted_topos[i + 1]

            gap = current.get_gap_to(next_topo, check_trim=trim)
            if gap is None:  # Overlapping is OK
                continue
            elif gap > gap_tolerance:
                current_residues = current._get_active_residues(trim)
                next_residues = next_topo._get_active_residues(trim)
                raise ValueError(
                    f"Gap of {gap} residues between topologies {current.fragment_name} "
                    f"(ends at {max(current_residues)}) and {next_topo.fragment_name} "
                    f"(starts at {min(next_residues)}) exceeds tolerance of {gap_tolerance}"
                )

        return cls.merge(sorted_topos, trim=trim, **kwargs)

    @classmethod
    def merge_overlapping(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        min_overlap: int = 1,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
    ) -> "Partial_Topology":
        """Merge topologies that have overlapping residues

        Args:
            topologies: List of topologies to merge
            trim: Whether to respect peptide trimming
            min_overlap: Minimum number of overlapping residues required
            merged_name: Name for the merged topology (default: concatenate names)
            merged_sequence: Sequence for merged topology (default: concatenate sequences)
            merged_index: Index for merged topology (default: use first topology's index)

        Returns:
            Merged topology

        Raises:
            ValueError: If not all topologies overlap sufficiently
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")

        if len(topologies) < 2:
            return cls.merge(
                topologies,
                trim=trim,
                merged_name=merged_name,
                merged_sequence=merged_sequence,
                merged_index=merged_index,
            )

        # Check that all topologies have sufficient overlap with at least one other
        for i, topo1 in enumerate(topologies):
            has_sufficient_overlap = False
            for j, topo2 in enumerate(topologies):
                if i != j:
                    overlap = topo1.get_overlap(topo2, check_trim=trim)
                    if len(overlap) >= min_overlap:
                        has_sufficient_overlap = True
                        break

            if not has_sufficient_overlap:
                topo1_residues = topo1._get_active_residues(trim)
                raise ValueError(
                    f"Topology {topo1.fragment_name} (residues {min(topo1_residues)}-{max(topo1_residues)}) "
                    f"does not have at least {min_overlap} overlapping residues with any other topology"
                )

        return cls.merge(
            topologies,
            trim=trim,
            merged_name=merged_name,
            merged_sequence=merged_sequence,
            merged_index=merged_index,
        )

    def union(self, other: "Partial_Topology", check_trim: bool = False) -> "Partial_Topology":
        """Combine this topology with another (union of residues)

        Args:
            other: The other topology to combine with
            check_trim: If True, use peptide_residues for peptides when determining union
        """
        if self.chain != other.chain:
            raise ValueError("Cannot combine topologies with different chains")

        # For union, we need to be careful about which residues to combine
        # If check_trim is True and either topology is a peptide, we use their active residues
        # But the result should contain the full residue ranges that encompass both active regions

        self_active = self._get_active_residues(check_trim)
        other_active = other._get_active_residues(check_trim)

        # Determine the full range needed to encompass both active regions
        all_active = sorted(set(self_active) | set(other_active))
        min_res, max_res = min(all_active), max(all_active)

        # Create the union with the full range
        combined_residues = list(range(min_res, max_res + 1))

        return Partial_Topology.from_residues(
            chain=self.chain,
            residues=combined_residues,
            fragment_sequence=self.fragment_sequence,
            fragment_name=self.fragment_name,
            fragment_index=self.fragment_index,
            peptide=self.peptide,
            peptide_trim=self.peptide_trim,
        )

    @classmethod
    def _build_chain_selection_string(
        cls, universe: mda.Universe, chain_id: str, base_selection: Optional[str] = None
    ) -> tuple[str, mda.AtomGroup]:
        """
        Utility method to build a selection string for a specific chain,
        accounting for available attributes in the universe.

        Args:
            universe: MDAnalysis Universe object
            chain_id: Chain ID to select
            base_selection: Optional base selection to combine with chain selection

        Returns:
            tuple: (selection_string, fallback_atomgroup)
            - selection_string: MDAnalysis selection string for the chain
            - fallback_atomgroup: AtomGroup to use if selection fails
        """
        # Check available attributes for chain selection
        has_segid = hasattr(universe.atoms[0], "segid") if len(universe.atoms) > 0 else False
        has_chainid = hasattr(universe.atoms[0], "chainid") if len(universe.atoms) > 0 else False

        # Build appropriate selection string based on available attributes
        chain_selection_parts = []
        if has_segid:
            chain_selection_parts.append(f"segid {chain_id}")
        if has_chainid:
            chain_selection_parts.append(f"chainid {chain_id}")

        # Initialize fallback_atoms as empty
        fallback_atoms = mda.AtomGroup([], universe)

        if not chain_selection_parts:
            # If neither attribute is available, we can't create a selection string
            selection_string = ""
        else:
            # Join selection parts with OR
            chain_selection = " or ".join(chain_selection_parts)

            # Combine with base selection if provided
            if base_selection:
                selection_string = f"({base_selection}) and ({chain_selection})"
            else:
                selection_string = chain_selection

        return selection_string, fallback_atoms

    @classmethod
    def from_mda_universe(
        cls,
        universe: mda.Universe,
        mode: str = "residue",
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        fragment_name_template: str = "auto",
        renumber_residues: bool = False,
    ) -> list["Partial_Topology"]:
        """Extract Partial_Topology objects from an MDAnalysis Universe

        Args:
            universe: MDAnalysis Universe object
            mode: Extraction mode - "by_chain" (one topology per chain) or
                "by_residue" (one topology per residue)
            include_selection: MDAnalysis selection string for atoms to include
            exclude_selection: Optional MDAnalysis selection string for atoms to exclude
            exclude_termini: If True, exclude N and C terminal residues from each chain
            termini_chain_selection: str
            fragment_name_template: Naming template - "auto" for automatic naming,
                                or custom template with {chain}, {resid}, {resname} placeholders
            renumber_residues: If True, renumber residues starting from 1 for each chain.
                            If False, preserve original residue numbers from MDAnalysis.

        Returns:
            List of Partial_Topology objects

        Raises:
            ValueError: If mode is not "by_chain" or "by_residue"
            ImportError: If MDAnalysis is not available
        """
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for this method. Install with: pip install MDAnalysis"
            )

        if mode not in ("chain", "residue"):
            raise ValueError("Mode must be either 'by_chain' or 'by_residue'")

        # Apply include selection
        try:
            selected_atoms = universe.select_atoms(include_selection)
        except Exception as e:
            raise ValueError(f"Invalid include selection '{include_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError(f"No atoms found with include selection '{include_selection}'")

        # Apply exclude selection if provided
        if exclude_selection:
            try:
                exclude_atoms = universe.select_atoms(exclude_selection)
                # Remove excluded atoms from selected atoms
                selected_atoms = selected_atoms - exclude_atoms
            except Exception as e:
                raise ValueError(f"Invalid exclude selection '{exclude_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError("No atoms remaining after applying exclude selection")

        # Group atoms by chain (try segid first, then chainid)
        chains = {}
        for atom in selected_atoms:
            # Prefer segid, fall back to chainid, then use 'A' as default
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        # Process each chain
        partial_topologies = []

        for chain_id, chain_atoms in chains.items():
            # Use utility method to build chain selection string
            chain_selection_string, fallback_atoms = cls._build_chain_selection_string(
                universe, chain_id, termini_chain_selection
            )

            if chain_selection_string:
                try:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        # Try without base selection
                        chain_only_selection, _ = cls._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                except Exception:
                    # Fallback if selection fails
                    termini_atoms = mda.AtomGroup(chain_atoms)
            else:
                # If we couldn't build a selection string, use the chain atoms directly
                termini_atoms = mda.AtomGroup(chain_atoms)

            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion to determine which residues to include
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]  # Remove first and last
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter selected atoms to only include residues that pass terminal exclusion
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            if not sorted_residues:
                continue  # Skip if no residues left after filtering

            # Create renumbering mapping from the list of ALL included residues for the chain,
            # not just the ones in the current selection. This ensures consistent numbering.
            if renumber_residues:
                final_residue_mapping = {res.resid: i for i, res in enumerate(included_residues, 1)}
            else:
                # When not renumbering, the mapping is identity for all residues in the chain
                final_residue_mapping = {res.resid: res.resid for res in full_chain_residues}

            # Extract sequence information
            try:
                # Try to get single-letter amino acid codes
                sequence = []
                for residue in sorted_residues:
                    resname = residue.resname
                    # Convert common 3-letter to 1-letter codes
                    aa_map = {
                        "ALA": "A",
                        "ARG": "R",
                        "ASN": "N",
                        "ASP": "D",
                        "CYS": "C",
                        "GLN": "Q",
                        "GLU": "E",
                        "GLY": "G",
                        "HIS": "H",
                        "ILE": "I",
                        "LEU": "L",
                        "LYS": "K",
                        "MET": "M",
                        "PHE": "F",
                        "PRO": "P",
                        "SER": "S",
                        "THR": "T",
                        "TRP": "W",
                        "TYR": "Y",
                        "VAL": "V",
                        "MSE": "M",  # Selenomethionine
                    }
                    sequence.append(aa_map.get(resname, "X"))  # X for unknown
                sequence_str = "".join(sequence)
            except:
                # Fallback to residue names
                sequence_str = [res.resname for res in sorted_residues]

            if mode == "chain":
                # Create one topology per chain
                # Use the corrected mapping to get residue numbers
                residue_numbers = [final_residue_mapping[res.resid] for res in sorted_residues]

                # Generate fragment name
                if fragment_name_template == "auto":
                    fragment_name = f"chain_{chain_id}"
                else:
                    fragment_name = fragment_name_template.format(
                        chain=chain_id,
                        resid=f"{min(residue_numbers)}-{max(residue_numbers)}",
                        resname="chain",
                    )

                topology = cls.from_residues(
                    chain=chain_id,
                    residues=residue_numbers,
                    fragment_sequence=sequence_str,
                    fragment_name=fragment_name,
                    peptide=False,
                )
                partial_topologies.append(topology)

            elif mode == "residue":
                # Create one topology per residue
                for i, residue in enumerate(sorted_residues):
                    # Use the corrected mapping to get the new resid
                    new_resid = final_residue_mapping[residue.resid]

                    # Get single residue sequence
                    if isinstance(sequence_str, str):
                        res_sequence = sequence_str[i]
                    else:
                        res_sequence = sequence_str[i]

                    # Generate fragment name
                    if fragment_name_template == "auto":
                        fragment_name = f"{chain_id}_{residue.resname}{new_resid}"
                    else:
                        fragment_name = fragment_name_template.format(
                            chain=chain_id, resid=new_resid, resname=residue.resname
                        )

                    topology = cls.from_single(
                        chain=chain_id,
                        residue=new_resid,
                        fragment_sequence=res_sequence,
                        fragment_name=fragment_name,
                        peptide=False,
                    )
                    partial_topologies.append(topology)

        if not partial_topologies:
            raise ValueError("No partial topologies could be created from the selection")

        return partial_topologies

    @classmethod
    def to_mda_group(
        cls,
        topologies: Union[set["Partial_Topology"], list["Partial_Topology"]],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = False,
        mda_atom_filtering: Optional[str] = None,
    ) -> Union["mda.ResidueGroup", "mda.AtomGroup"]:
        """Create MDAnalysis ResidueGroup or AtomGroup from Partial_Topology objects."""
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for this method. Install with: pip install MDAnalysis"
            )

        if not topologies:
            raise ValueError("No topologies provided")

        # Convert to list if set
        if isinstance(topologies, set):
            topologies = list(topologies)

        # Apply include selection - SAME as from_mda_universe
        try:
            selected_atoms = universe.select_atoms(include_selection)
        except Exception as e:
            raise ValueError(f"Invalid include selection '{include_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError(f"No atoms found with include selection '{include_selection}'")

        # Apply exclude selection if provided - SAME as from_mda_universe
        if exclude_selection:
            try:
                exclude_atoms = universe.select_atoms(exclude_selection)
                selected_atoms = selected_atoms - exclude_atoms
            except Exception as e:
                raise ValueError(f"Invalid exclude selection '{exclude_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError("No atoms remaining after applying exclude selection")

        # Group atoms by chain - SAME as from_mda_universe
        chains = {}
        for atom in selected_atoms:
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        # Group topologies by chain
        topologies_by_chain = {}
        for topo in topologies:
            if topo.chain not in topologies_by_chain:
                topologies_by_chain[topo.chain] = []
            topologies_by_chain[topo.chain].append(topo)

        # Create selection parts for each chain
        per_chain_selection_parts = []

        for chain_id, chain_topologies in topologies_by_chain.items():
            if chain_id not in chains:
                continue  # Skip if chain not found in universe

            # Get chain_atoms (from include/exclude selections) - SAME as from_mda_universe
            chain_atoms = chains[chain_id]

            # Use utility method to build chain selection string for terminal exclusion
            chain_selection_string, fallback_atoms = cls._build_chain_selection_string(
                universe, chain_id, termini_chain_selection
            )

            try:
                if chain_selection_string:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        chain_only_selection, _ = cls._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                else:
                    termini_atoms = mda.AtomGroup(chain_atoms)
            except Exception as e:
                raise ValueError(f"Failed to select chain {chain_id}: {e}")

            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion to determine which residues to include - SAME as from_mda_universe
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]  # Remove first and last
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter chain atoms to only include residues that pass terminal exclusion - SAME as from_mda_universe
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid - SAME as from_mda_universe
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            if not sorted_residues:
                continue

            # Create renumbering mapping from the FINAL list of residues after all filtering - SAME as from_mda_universe
            if renumber_residues:
                renumber_to_original = {i: res.resid for i, res in enumerate(included_residues, 1)}
            else:
                renumber_to_original = {res.resid: res.resid for res in full_chain_residues}

            # Map topology residues to original residues
            chain_target_resids = []
            for topo in chain_topologies:
                for topo_resid in topo.residues:
                    original_resid = renumber_to_original.get(topo_resid)
                    if original_resid is not None:
                        chain_target_resids.append(original_resid)

            if not chain_target_resids:
                continue

            unique_resids = sorted(set(chain_target_resids))
            resid_selection = f"resid {' '.join(map(str, unique_resids))}"

            chain_sel_str, _ = cls._build_chain_selection_string(universe, chain_id)
            if chain_sel_str:
                per_chain_selection_parts.append(f"(({chain_sel_str}) and ({resid_selection}))")
            else:
                per_chain_selection_parts.append(f"({resid_selection})")

        if not per_chain_selection_parts:
            raise ValueError("No matching residues found in universe")

        final_resid_selection = " or ".join(per_chain_selection_parts)

        # Select residues from universe
        try:
            # Combine with original selection to ensure consistency
            combined_selection = f"({include_selection}) and ({final_resid_selection})"
            if exclude_selection:
                combined_selection = f"({combined_selection}) and not ({exclude_selection})"

            target_atoms = universe.select_atoms(combined_selection)

            if len(target_atoms) == 0:
                raise ValueError("No atoms found matching the topology residues")

        except Exception as e:
            raise ValueError(f"Failed to select target atoms: {e}")

        # Apply atom filtering if requested
        if mda_atom_filtering:
            try:
                target_atoms = target_atoms.select_atoms(mda_atom_filtering)
                if len(target_atoms) == 0:
                    raise ValueError(f"No atoms found after applying filter '{mda_atom_filtering}'")
            except Exception as e:
                raise ValueError(f"Invalid atom filtering '{mda_atom_filtering}': {e}")

        # Return atoms or residues as requested
        if mda_atom_filtering:
            return target_atoms
        else:
            return target_atoms.residues

    @classmethod
    def to_mda_residue_dict(
        cls,
        topologies: Union[set["Partial_Topology"], list["Partial_Topology"]],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
    ) -> dict[Union[str, int], list[int]]:
        """
        Extract residue information as a dictionary of chain:[residue_indices].

        This method uses to_mda_group to get the residues and then formats them
        into a dictionary.

        Args:
            topologies: Set or list of Partial_Topology objects to convert.
            universe: MDAnalysis Universe object to select from.
            include_selection: MDAnalysis selection string for atoms to include.
            exclude_selection: Optional MDAnalysis selection string for atoms to exclude.
            exclude_termini: If True, account for N and C terminal residue exclusion.
            termini_chain_selection: Selection string used for terminal identification.
            renumber_residues: If True, assume residue numbers were renumbered from 1.
                              If False, use original residue numbers.

        Returns:
            A dictionary mapping chain ID to a list of residue indices.
        """
        residue_group = cls.to_mda_group(
            topologies=topologies,
            universe=universe,
            include_selection=include_selection,
            exclude_selection=exclude_selection,
            exclude_termini=exclude_termini,
            termini_chain_selection=termini_chain_selection,
            renumber_residues=renumber_residues,
            mda_atom_filtering=None,  # We want residues, not atoms
        )

        if isinstance(residue_group, mda.AtomGroup):
            residues = residue_group.residues
        else:
            residues = residue_group

        residue_dict = {}
        for res in residues:
            chain_id = getattr(res, "segid", None) or getattr(res, "chainid", "A")
            if chain_id not in residue_dict:
                residue_dict[chain_id] = []
            residue_dict[chain_id].append(res.resid)

        # Sort residue indices for each chain
        for chain_id in residue_dict:
            residue_dict[chain_id].sort()

        return residue_dict

    @classmethod
    def group_set_by_chain(
        cls, topologies: set["Partial_Topology"]
    ) -> dict[Union[str, int], set["Partial_Topology"]]:
        """Group a set of Partial_Topology objects by chain.

        Args:
            topologies: Set of Partial_Topology objects to group

        Returns:
            Dictionary mapping chain ID to sets of Partial_Topology objects for that chain
        """
        result = {}
        for topo in topologies:
            if topo.chain not in result:
                result[topo.chain] = set()
            result[topo.chain].add(topo)
        return result

    @classmethod
    def calculate_fragment_redundancy(
        cls,
        topologies: list["Partial_Topology"],
        mode: str = "mean",
        check_trim: bool = False,
    ) -> list[float]:
        """Calculate overlap redundancy between fragments in a list

        Args:
            topologies: List of Partial_Topology objects to compare
            mode: Either "max" or "mean" to determine how to calculate overlap scores
            check_trim: If True, use peptide_residues for peptides; if False, use all residues

        Returns:
            List of overlap scores for each fragment (same order as input)

        Raises:
            ValueError: If mode is not "max" or "mean"
            ValueError: If topologies list is empty
        """
        if not topologies:
            raise ValueError("Cannot calculate redundancy for empty topology list")

        if mode not in ("max", "mean"):
            raise ValueError("Mode must be either 'max' or 'mean'")

        # Group topologies by chain for more efficient comparison
        topology_set = set(topologies)
        grouped_by_chain = cls.group_set_by_chain(topology_set)

        redundancy_scores = []

        for i, fragment in enumerate(topologies):
            # Get active residues for current fragment
            current_residues = set(fragment._get_active_residues(check_trim))

            # Only compare with fragments from the same chain
            same_chain_fragments = grouped_by_chain.get(fragment.chain, set())
            fragment_overlaps = []

            # Compare with other fragments in the same chain
            for other in same_chain_fragments:
                if other is not fragment:  # Skip self-comparison
                    # Get active residues for other fragment
                    other_residues = set(other._get_active_residues(check_trim))

                    # Calculate overlap
                    overlap_size = len(current_residues & other_residues)
                    if overlap_size > 0:
                        fragment_overlaps.append(overlap_size)

            # Calculate final redundancy score based on mode
            if fragment_overlaps:
                if mode == "max":
                    redundancy_scores.append(max(fragment_overlaps))
                elif mode == "mean":
                    redundancy_scores.append(sum(fragment_overlaps) / len(fragment_overlaps))
            else:
                redundancy_scores.append(0.0)

        return redundancy_scores

    @classmethod
    def find_common_residues(
        cls,
        ensemble: list[mda.Universe],
        include_selection: Union[str, list[str]] = "protein",
        exclude_selection: Union[str, list[str]] = "resname SOL",
        renumber_residues: bool = True,  # Changed default to False
    ) -> tuple[set["Partial_Topology"], set["Partial_Topology"]]:
        """Find common residues across an ensemble of MDAnalysis Universe objects.

        Args:
            ensemble: List of MDAnalysis Universe objects to analyze
            include_selection: Selection string or list of strings for atoms to include.
                            If a list, it must have the same length as the ensemble.
            exclude_selection: Selection string or list of strings for atoms to exclude.
                            If a list, it must have the same length as the ensemble.
            renumber_residues: If True, renumber residues starting from 1 for each chain.
                            If False, preserve original residue numbers.

        Returns:
            Tuple containing:
            - Set of common residues present in all universes
            - Set of excluded residues (present in some but not all universes)

        Raises:
            ValueError: If no common residues are found or if selection list lengths are incorrect.
            ImportError: If MDAnalysis is not available
        """
        if not ensemble:
            raise ValueError("Empty ensemble provided")

        # Normalize selection strings to lists
        if isinstance(include_selection, str):
            include_selection = [include_selection] * len(ensemble)
        if isinstance(exclude_selection, str):
            exclude_selection = [exclude_selection] * len(ensemble)

        if len(include_selection) != len(ensemble):
            raise ValueError("include_selection list must have the same length as the ensemble")
        if len(exclude_selection) != len(ensemble):
            raise ValueError("exclude_selection list must have the same length as the ensemble")

        # Extract included chain topologies from each universe
        included_chain_topologies_by_universe = []
        for i, universe in enumerate(ensemble):
            try:
                # Extract chain topologies with filtering
                chain_topos = cls.from_mda_universe(
                    universe,
                    mode="chain",
                    include_selection=include_selection[i],
                    exclude_selection=exclude_selection[i],
                    exclude_termini=True,
                    renumber_residues=renumber_residues,  # Use parameter
                )
                included_chain_topologies_by_universe.append(chain_topos)
            except Exception as e:
                raise ValueError(f"Failed to extract topologies from universe {i}: {e}")

        # Group chain topologies by chain ID across universes using the new method
        chain_ids = set()
        for chain_topos in included_chain_topologies_by_universe:
            chain_ids.update(topo.chain for topo in chain_topos)

        # Group all chain topologies by chain ID across all universes
        all_topos_by_chain = {}
        for chain_topos in included_chain_topologies_by_universe:
            for topo in chain_topos:
                if topo.chain not in all_topos_by_chain:
                    all_topos_by_chain[topo.chain] = []
                all_topos_by_chain[topo.chain].append(topo)

        # Find common residues by chain using intersection merge
        common_residue_topologies = set()
        for chain_id, topos_for_chain in all_topos_by_chain.items():
            if not topos_for_chain:
                continue

            try:
                # Find intersection of residues across all universes for this chain
                common_chain_topo = cls.merge(
                    topos_for_chain,
                    trim=False,
                    intersection=True,
                    merged_name=f"common_chain_{chain_id}",
                )

                # Extract individual residues from the common chain
                residue_topos = common_chain_topo.extract_residues(use_peptide_trim=False)
                common_residue_topologies.update(residue_topos)

            except ValueError as e:
                # If merge fails because there are no common residues, continue
                if "No common residues found" in str(e):
                    continue
                # Re-raise other ValueErrors
                raise e

        # Extract excluded residues
        excluded_residue_topologies = set()

        # Get all possible residues (without ignore selection, without termini exclusion)
        all_chain_topologies_by_universe = []
        for i, universe in enumerate(ensemble):
            try:
                all_chain_topos = cls.from_mda_universe(
                    universe,
                    mode="chain",
                    include_selection=include_selection[i],
                    exclude_selection="",  # No exclusions
                    exclude_termini=False,  # Include termini
                    renumber_residues=renumber_residues,  # Use parameter
                )
                all_chain_topologies_by_universe.append(all_chain_topos)
            except Exception as e:
                raise ValueError(f"Failed to extract all topologies from universe {i}: {e}")

        # For each universe, find residues that are excluded
        for i, (all_chain_topos, included_chain_topos) in enumerate(
            zip(all_chain_topologies_by_universe, included_chain_topologies_by_universe)
        ):
            # Group chain topologies by chain using the new method
            included_by_chain = {topo.chain: topo for topo in included_chain_topos}

            for all_chain_topo in all_chain_topos:
                chain_id = all_chain_topo.chain

                if chain_id in included_by_chain:
                    # Remove included residues from this chain
                    try:
                        excluded_chain_topo = all_chain_topo.remove_residues(
                            [included_by_chain[chain_id]]
                        )
                        excluded_residues = excluded_chain_topo.extract_residues(
                            use_peptide_trim=False
                        )
                        excluded_residue_topologies.update(excluded_residues)
                    except ValueError:
                        # No residues remaining after removal
                        continue
                else:
                    # Entire chain was excluded
                    excluded_residues = all_chain_topo.extract_residues(use_peptide_trim=False)
                    excluded_residue_topologies.update(excluded_residues)

        # Remove duplicates from excluded set using contains_topology
        # Group by chain first for efficiency
        excluded_by_chain = cls.group_set_by_chain(excluded_residue_topologies)
        excluded_unique = set()

        for chain_id, chain_topos in tqdm(
            excluded_by_chain.items(), desc="Processing excluded residues by chain"
        ):
            chain_unique = set()
            for excluded_topo in chain_topos:
                # Check if this topology is already represented in the same chain
                is_duplicate = False
                for existing_topo in chain_unique:
                    if excluded_topo.contains_topology(existing_topo):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    chain_unique.add(excluded_topo)

            excluded_unique.update(chain_unique)

        excluded_residue_topologies = excluded_unique

        if not common_residue_topologies:
            raise ValueError("No common residues found in the ensemble")

        return common_residue_topologies, excluded_residue_topologies

    @classmethod
    def partial_topology_pairwise_distances(
        cls,
        topologies: list["Partial_Topology"],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        compound: str = "group",
        pbc: bool = True,
        backend: str = "OpenMP",
        verbose: bool = True,
        check_trim: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute trajectory-averaged pairwise center-of-mass distances between Partial_Topology objects.

        Parameters
        ----------
        topologies : List[Partial_Topology]
            List of Partial_Topology objects to compute distances between
        universe : mda.Universe
            MDAnalysis Universe containing the trajectory
        include_selection : str, default "protein"
            MDAnalysis selection string for atoms to include
        exclude_selection : str, optional
            MDAnalysis selection string for atoms to exclude
        exclude_termini : bool, default True
            If True, account for N and C terminal residue exclusion
        termini_chain_selection : str, default "protein"
            Selection string used for terminal identification
        renumber_residues : bool, default True
            If True, assume residue numbers were renumbered from 1
        start : int, optional
            Start frame for analysis
        stop : int, optional
            Stop frame for analysis
        step : int, optional
            Step size for frame iteration
        compound : str, default 'group'
            How to compute center of mass for each topology
        pbc : bool, default True
            Whether to account for periodic boundary conditions
        backend : str, default 'OpenMP'
            Backend for distance calculations
        verbose : bool, default True
            Whether to show progress bar
        check_trim : bool, default False
            If True, use peptide_residues for peptides

        Returns
        -------
        distance_matrix : np.ndarray
            Trajectory-averaged distance matrix of shape (n_topologies, n_topologies)
        distance_std : np.ndarray
            Standard deviation of distances over trajectory
        """
        if not topologies:
            raise ValueError("topologies list cannot be empty")

        # Convert Partial_Topology objects to MDAnalysis groups
        mda_groups = []
        for topo in topologies:
            try:
                # Use existing to_mda_group method from Partial_Topology
                group = cls.to_mda_group(
                    topologies={topo},  # Convert single topology to set
                    universe=universe,
                    include_selection=include_selection,
                    exclude_selection=exclude_selection,
                    exclude_termini=exclude_termini,
                    termini_chain_selection=termini_chain_selection,
                    renumber_residues=renumber_residues,
                    mda_atom_filtering=None,  # Get residues, not atoms
                )

                # Convert ResidueGroup to AtomGroup for COM calculations
                if isinstance(group, mda.ResidueGroup):
                    group = group.atoms

                mda_groups.append(group)

            except Exception as e:
                raise ValueError(f"Failed to convert topology {topo} to MDAnalysis group: {e}")

        if verbose:
            print(f"Converted {len(topologies)} topologies to MDAnalysis groups")

        # Compute distances
        distance_matrix, distance_std = compute_trajectory_average_com_distances(
            universe=universe,
            group_list=mda_groups,
            start=start,
            stop=stop,
            step=step,
            compound=compound,
            pbc=pbc,
            backend=backend,
            verbose=verbose,
        )

        return distance_matrix, distance_std

    @classmethod
    def get_atomgroup_reordering_indices(
        cls,
        mda_groups: list[Union[mda.ResidueGroup, mda.AtomGroup]],
        universe: mda.Universe,
        target_topologies: Optional[list["Partial_Topology"]] = None,
        # include_selection: str = "protein",
        # exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
    ) -> list[int]:
        """Get indices to reorder a list of MDAnalysis groups to match topology order.

        This method is essential for the BV_model.featurise() method to ensure that
        feature calculation results are ordered consistently with Partial_Topology ranking.

        Args:
            mda_groups: List of MDAnalysis ResidueGroup or AtomGroup objects to be reordered
            universe: MDAnalysis Universe object
            target_topologies: Optional list of Partial_Topology objects in desired order.
                             If None, will extract topologies from mda_groups and use default ranking.
            include_selection: Selection string for atoms to include
            exclude_selection: Selection string for atoms to exclude
            exclude_termini: Whether to exclude terminal residues
            termini_chain_selection: Selection for terminal identification
            renumber_residues: Whether residues were renumbered from 1

        Returns:
            List of indices for reordering mda_groups to match target topology order.
            Use as: reordered_features = [original_features[i] for i in indices]

        Raises:
            ValueError: If any topology cannot be mapped to mda_groups, or if
                       topology residues are not contained within the chain bounds
        """
        if not mda_groups:
            return []

        # If no target topologies provided, extract them from mda_groups and rank them
        if target_topologies is None:
            target_topologies = []
            for mda_group in mda_groups:
                # Convert each MDA group to a Partial_Topology
                if isinstance(mda_group, mda.ResidueGroup):
                    residues = [res for res in mda_group]  # Iterate explicitly
                    if not residues:
                        continue

                    if len(residues) == 1:
                        # Single residue
                        res = residues[0]
                        chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                            res.atoms[0], "chainid", "A"
                        )
                        topo = cls.from_single(
                            chain=chain_id,
                            residue=res.resid,
                            fragment_name=f"{chain_id}_{res.resname}{res.resid}",
                        )
                    else:
                        # Multiple residues
                        chain_ids = set()
                        resids = []
                        for res in residues:
                            chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                                res.atoms[0], "chainid", "A"
                            )
                            chain_ids.add(chain_id)
                            resids.append(res.resid)

                        if len(chain_ids) > 1:
                            raise ValueError(
                                f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                            )

                        chain_id = list(chain_ids)[0]
                        topo = cls.from_residues(
                            chain=chain_id,
                            residues=resids,
                            fragment_name=f"{chain_id}_multi",
                        )
                elif isinstance(mda_group, mda.AtomGroup):
                    residues = list(mda_group.residues)
                    if len(residues) == 1:
                        # Single residue
                        res = residues[0]
                        chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                            res.atoms[0], "chainid", "A"
                        )
                        topo = cls.from_single(
                            chain=chain_id,
                            residue=res.resid,
                            fragment_name=f"{chain_id}_{res.resname}{res.resid}",
                        )
                    else:
                        # Multiple residues
                        chain_ids = set()
                        resids = []
                        for res in residues:
                            chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                                res.atoms[0], "chainid", "A"
                            )
                            chain_ids.add(chain_id)
                            resids.append(res.resid)

                        if len(chain_ids) > 1:
                            raise ValueError(
                                f"AtomGroup contains residues from multiple chains: {chain_ids}"
                            )

                        chain_id = list(chain_ids)[0]
                        topo = cls.from_residues(
                            chain=chain_id,
                            residues=resids,
                            fragment_name=f"{chain_id}_multi",
                        )
                else:
                    raise TypeError(f"Unsupported group type: {type(mda_group)}")

                target_topologies.append(topo)

            # Rank and index the topologies using default ordering
            target_topologies = cls.rank_and_index(target_topologies, check_trim=False)

        # Build a mapping from topology characteristics to mda_groups index
        mda_group_lookup = {}

        # Build renumbering mapping if needed
        if renumber_residues:
            renumber_mapping = cls._build_renumbering_mapping(
                universe, exclude_termini, termini_chain_selection
            )

        for i, mda_group in enumerate(mda_groups):
            if isinstance(mda_group, mda.ResidueGroup):
                residues = [res for res in mda_group]  # Iterate explicitly
                if not residues:
                    continue

                chain_ids = set()
                resids = []
                for res in residues:
                    chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                        res.atoms[0], "chainid", "A"
                    )
                    chain_ids.add(chain_id)

                    if renumber_residues:
                        original_resid = res.resid
                        renumbered_resid = None
                        for (ch, new_id), orig_id in renumber_mapping.items():
                            if ch == chain_id and orig_id == original_resid:
                                renumbered_resid = new_id
                                break
                        if renumbered_resid is not None:
                            resids.append(renumbered_resid)
                    else:
                        resids.append(res.resid)

                if len(chain_ids) > 1:
                    raise ValueError(
                        f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                    )

                if not resids:
                    continue  # Skip if no valid resids

                chain_id = list(chain_ids)[0]
                lookup_key = (chain_id, frozenset(resids))

            elif isinstance(mda_group, mda.AtomGroup):
                residues = list(mda_group.residues)
                if not residues:
                    continue

                chain_ids = set()
                resids = []
                for res in residues:
                    chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                        res.atoms[0], "chainid", "A"
                    )
                    chain_ids.add(chain_id)

                    if renumber_residues:
                        original_resid = res.resid
                        renumbered_resid = None
                        for (ch, new_id), orig_id in renumber_mapping.items():
                            if ch == chain_id and orig_id == original_resid:
                                renumbered_resid = new_id
                                break
                        if renumbered_resid is not None:
                            resids.append(renumbered_resid)
                    else:
                        resids.append(res.resid)

                if len(chain_ids) > 1:
                    raise ValueError(
                        f"AtomGroup contains residues from multiple chains: {chain_ids}"
                    )

                if not resids:
                    continue  # Skip if no valid resids

                chain_id = list(chain_ids)[0]
                lookup_key = (chain_id, frozenset(resids))

            mda_group_lookup[lookup_key] = i

        # Find indices for each target topology
        result_indices = []

        for target_topo in target_topologies:
            # Validate topology containment
            cls._validate_topology_containment(
                target_topo, universe, exclude_termini, termini_chain_selection, renumber_residues
            )

            chain_id = target_topo.chain
            active_residues = target_topo._get_active_residues(check_trim=False)
            lookup_key = (chain_id, frozenset(active_residues))

            if lookup_key not in mda_group_lookup:
                available_keys = list(mda_group_lookup.keys())
                raise ValueError(
                    f"Cannot find MDA group for topology {target_topo} "
                    f"(chain={chain_id}, resids={sorted(active_residues)}). "
                    f"Available groups: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}"
                )

            result_indices.append(mda_group_lookup[lookup_key])

        return result_indices

    @classmethod
    def get_residuegroup_reordering_indices(
        cls,
        residue_group: Union[mda.ResidueGroup, mda.AtomGroup],
    ) -> list[int]:
        """Get indices to reorder individual residues in a ResidueGroup/AtomGroup by topology ranking.

        This method takes a single ResidueGroup or AtomGroup containing multiple residues
        and returns indices to sort the individual residues according to Partial_Topology
        ranking logic.

        Args:
            residue_group: MDAnalysis ResidueGroup or AtomGroup containing multiple residues

        Returns:
            List of indices for reordering individual residues within the group.
            Use as: sorted_residues = [residue_group.residues[i] for i in indices]

        Raises:
            ValueError: If group contains no residues
            TypeError: If group is not a ResidueGroup or AtomGroup

        Example:
            >>> indices = Partial_Topology.get_residuegroup_reordering_indices(protein_residues)
            >>> sorted_residues = [protein_residues.residues[i] for i in indices]
        """
        if isinstance(residue_group, mda.AtomGroup):
            residues = list(residue_group.residues)
        elif isinstance(residue_group, mda.ResidueGroup):
            residues = [res for res in residue_group]  # Iterate explicitly
        else:
            raise TypeError("residue_group must be a ResidueGroup or AtomGroup")

        if not residues:
            raise ValueError("Group contains no residues")

        # Create (sort_key, original_index) pairs
        keyed_residues = []
        for i, residue in enumerate(residues):
            sort_key = cls.get_mda_group_sort_key(residue)
            keyed_residues.append((sort_key, i))

        # Sort by sort_key and extract original indices
        keyed_residues.sort(key=lambda x: x[0])
        reorder_indices = [original_idx for _, original_idx in keyed_residues]

        return reorder_indices

    @classmethod
    def _build_renumbering_mapping(
        cls,
        universe: mda.Universe,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
    ) -> dict[tuple[str, int], int]:
        """Build mapping from (chain, renumbered_resid) to original_resid."""
        renumber_mapping = {}

        # Apply the same selection logic as from_mda_universe
        try:
            selected_atoms = universe.select_atoms(termini_chain_selection)
        except Exception:
            # Fallback to all atoms if selection fails
            selected_atoms = universe.atoms

        if len(selected_atoms) == 0:
            return renumber_mapping

        # Group atoms by chain
        chains = {}
        for atom in selected_atoms:
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        for chain_id, chain_atoms in chains.items():
            # Get chain selection for terminal exclusion
            chain_selection_string, _ = cls._build_chain_selection_string(
                universe, chain_id, termini_chain_selection
            )

            if chain_selection_string:
                try:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        chain_only_selection, _ = cls._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                except:
                    termini_atoms = mda.AtomGroup(chain_atoms)
            else:
                termini_atoms = mda.AtomGroup(chain_atoms)

            # Sort by original resid
            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion BEFORE creating the renumbering mapping
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter chain atoms to only include residues that pass terminal exclusion
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            # Create renumbering mapping from the final filtered residues
            for new_resid, residue in enumerate(sorted_residues, 1):
                renumber_mapping[(chain_id, new_resid)] = residue.resid

        return renumber_mapping

    @classmethod
    def _validate_topology_containment(
        cls,
        topology: "Partial_Topology",
        universe: mda.Universe,
        exclude_termini: bool = True,
        renumber_residues: bool = True,
    ) -> None:
        """Validate that topology residues are contained within chain bounds."""
        chain_id = topology.chain

        # Get chain selection
        chain_selection_string, _ = cls._build_chain_selection_string(universe, chain_id)

        if chain_selection_string:
            try:
                chain_residues = universe.select_atoms(chain_selection_string).residues
            except:
                chain_residues = [
                    r
                    for r in universe.residues
                    if (getattr(r.atoms[0], "segid", None) or getattr(r.atoms[0], "chainid", "A"))
                    == chain_id
                ]
        else:
            chain_residues = [
                r
                for r in universe.residues
                if (getattr(r.atoms[0], "segid", None) or getattr(r.atoms[0], "chainid", "A"))
                == chain_id
            ]

        if not chain_residues:
            raise ValueError(f"No residues found for chain {chain_id}")

        # Sort and apply terminal exclusion
        full_chain_residues = sorted(chain_residues, key=lambda r: r.resid)
        if exclude_termini and len(full_chain_residues) > 2:
            available_residues = full_chain_residues[1:-1]
        else:
            available_residues = full_chain_residues

        if renumber_residues:
            # Check that topology residues fall within the renumbered range
            expected_range = list(range(1, len(available_residues) + 1))
            available_resids = set(expected_range)
        else:
            # Check that topology residues exist in the available residues
            available_resids = {r.resid for r in available_residues}

        # Check topology residues (use active residues based on check_trim)
        active_residues = topology._get_active_residues(check_trim=False)

        missing_residues = set(active_residues) - available_resids
        if missing_residues:
            raise ValueError(
                f"Topology {topology} contains residues {missing_residues} "
                f"that are not available in chain {chain_id}. "
                f"Available residues: {sorted(available_resids)}"
            )

    @staticmethod
    def get_mda_group_sort_key(
        group: Union[mda.ResidueGroup, mda.AtomGroup, Residue],
    ) -> tuple[int, tuple[int, ...], float, int]:
        """Generate a sort key for an MDAnalysis group that matches Partial_Topology ranking."""
        if isinstance(group, Residue):
            residues = [group]
            chain_id = getattr(group.atoms[0], "segid", None) or getattr(
                group.atoms[0], "chainid", "A"
            )
        elif isinstance(group, mda.ResidueGroup):
            residues = [res for res in group.residues]  # Iterate explicitly
            if len(residues) == 0:
                raise ValueError("ResidueGroup contains no residues")

            # Check that all residues are from the same chain
            chain_ids = set()
            for residue in residues:
                residue_chain_id = getattr(residue.atoms[0], "segid", None) or getattr(
                    residue.atoms[0], "chainid", "A"
                )
                chain_ids.add(residue_chain_id)

            if len(chain_ids) > 1:
                raise ValueError(
                    f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                )

            chain_id = list(chain_ids)[0]
        elif isinstance(group, mda.AtomGroup):
            residues = list(group.residues)
            if len(residues) == 0:
                raise ValueError("AtomGroup contains no residues")

            # Check that all atoms are from the same chain
            chain_ids = set()
            for atom in group:
                atom_chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
                chain_ids.add(atom_chain_id)

            if len(chain_ids) > 1:
                raise ValueError(f"AtomGroup contains atoms from multiple chains: {chain_ids}")

            chain_id = list(chain_ids)[0]
        else:
            raise TypeError("group must be a Residue, ResidueGroup, or AtomGroup")

        # Ensure chain is string and uppercase
        if isinstance(chain_id, str):
            chain_id = chain_id.upper()
        else:
            chain_id = str(chain_id)

        # Get residue numbers
        resids = [r.resid for r in residues]

        # Calculate average residue position
        avg_residue = sum(resids) / len(resids)

        # Calculate length
        length = len(resids)

        # Generate chain score (same logic as Partial_Topology._get_chain_score)
        # This allows proper alphanumeric sorting: A < B < ... < 1 < 2 < ...
        chain_score = tuple(ord(c) for c in chain_id)

        # Return sort key: (chain_len, chain_score, avg_residue, -length)
        # Negative length to sort longer fragments first (descending)
        return (len(chain_id), chain_score, avg_residue, -length)
