from dataclasses import dataclass, field
from typing import Any, Optional, Union


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
    peptide_trim: int = 2  # Residues to trim from start when peptide=True

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

    def _set_peptide(self, val: bool, trim: int = None) -> None:
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

    def _get_active_residues(self, check_trim: bool = False) -> list[int]:
        """Get the active residues for this topology

        Args:
            check_trim: If True and this is a peptide, return peptide_residues.
                       If False, return all residues.
        """
        if check_trim and self.peptide and self.peptide_residues:
            return self.peptide_residues
        return self.residues

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

    def _to_dict(self) -> dict[str, Any]:
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
    def _from_dict(cls, data: dict[str, Any]) -> "Partial_Topology":
        """Create from dictionary (e.g., from JSON)"""
        return cls(**data)

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

    def contains_all_residues(self, residues: list[int], check_trim: bool = True) -> bool:
        """Check if this topology contains ALL of the specified residues

        Args:
            residues: List of residue numbers to check
            check_trim: If True and this is a peptide, check against peptide_residues
        """
        active_residues = self._get_active_residues(check_trim)
        return all(res in active_residues for res in residues)

    def contains_any_residues(self, residues: list[int], check_trim: bool = True) -> bool:
        """Check if this topology contains ANY of the specified residues

        Args:
            residues: List of residue numbers to check
            check_trim: If True and this is a peptide, check against peptide_residues
        """
        active_residues = self._get_active_residues(check_trim)
        return any(res in active_residues for res in residues)

    def contains_which_residue(
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
