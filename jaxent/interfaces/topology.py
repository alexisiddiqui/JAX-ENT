from dataclasses import dataclass
from typing import Optional, cast


@dataclass()
class Partial_Topology:
    """
    Dataclass that holds the information of a single topology fragment.
    This is usually a residue but may also be a group of residues such as peptides.
    """

    chain: str | int
    fragment_sequence: (
        str | list[str]
    )  # resname (residue) or single letter codes (peptide) or atom name (atom)
    residue_start: int  # inclusive, if a peptide, this is the first residue - not the
    residue_end: int | None = None  # inclusive - if None, then this is a single residue
    peptide_trim: int = 2
    fragment_index: Optional[int] = (
        None  # atom or peptide index - optional for residues - required for peptides and atoms
    )
    length: int | None = None  # length of the fragment

    def __post_init__(self):
        if self.residue_end is None:
            self.residue_end = self.residue_start
        self.length = (self.residue_end - self.residue_start) + 1
        if self.length > self.peptide_trim:
            self.peptide_residues = [
                i
                for i in range(
                    self.residue_start + self.peptide_trim,
                    self.residue_end + 1,
                )
            ]
            self.peptide_length = len(self.peptide_residues)

    def __eq__(self, other) -> bool:
        """Equal comparison - checks if all attributes are the same"""
        if not isinstance(other, Partial_Topology):
            return NotImplemented
        return (
            self.chain == other.chain
            and self.fragment_index == other.fragment_index
            and self.fragment_sequence == other.fragment_sequence
            and self.residue_start == other.residue_start
            and self.residue_end == other.residue_end
        )

    def __hash__(self) -> int:
        """Hash function - uses the hash of chain, index, sequence, residue start and end"""
        return hash(
            (
                self.chain,
                self.fragment_index,
                self.fragment_sequence,
                self.residue_start,
                self.residue_end,
            )
        )

    def __str__(self) -> str:
        """Pretty string representation for printing"""
        # Construct index part - show fragment index first, or "none" if None
        index_part = f"{self.fragment_index}:" if self.fragment_index is not None else "None:"

        # For single residue
        if self.length == 1:
            desc = f"{index_part}{self.fragment_sequence}:{self.chain}:{self.residue_start}"
            return desc

        # For peptides or multi-residue fragments
        desc = f"{index_part}{self.fragment_sequence}:{self.chain}:{self.residue_start}-{self.residue_end}"
        if hasattr(self, "peptide_residues"):
            desc += f" [peptide: {len(self.peptide_residues)} residues]"
        return desc

    def __repr__(self) -> str:
        """Detailed representation with all attributes"""
        attrs = []
        for field_name, field_value in self.__dict__.items():
            # Skip certain attributes for cleaner output
            if field_name in ("peptide_residues", "peptide_length"):
                continue

            if isinstance(field_value, str):
                attrs.append(f"{field_name}='{field_value}'")
            else:
                attrs.append(f"{field_name}={field_value}")

        return f"Partial_Topology({', '.join(attrs)})"

    def extract_residues(self, peptide: bool = True) -> list["Partial_Topology"]:
        """
        Extracts the residues from a peptide or atom fragment
        """
        if self.length == 1:
            return [self]
        else:
            if peptide is True:  # this needs to ignore the first two residues
                return [
                    Partial_Topology(self.chain, self.fragment_sequence, res)
                    for res in self.peptide_residues
                ]
            else:
                return [
                    Partial_Topology(self.chain, self.fragment_sequence, res)
                    for res in range(self.residue_start, cast(int, self.residue_end) + 1)
                ]

    def __add__(self, other: "Partial_Topology") -> "Partial_Topology":
        """Operator overloading for + to combine two topologies if they are adjacent"""
        if self.chain != other.chain or self.fragment_sequence != other.fragment_sequence:
            raise ValueError("Cannot combine topologies with different chains or sequences")

        # Check if they are adjacent
        if self.residue_end + 1 == other.residue_start:
            return Partial_Topology(
                self.chain,
                self.fragment_sequence,
                self.residue_start,
                other.residue_end,
                self.peptide_trim,
                self.fragment_index,
            )
        elif other.residue_end + 1 == self.residue_start:
            return Partial_Topology(
                self.chain,
                self.fragment_sequence,
                other.residue_start,
                self.residue_end,
                self.peptide_trim,
                self.fragment_index,
            )
        else:
            raise ValueError("Topologies are not adjacent")

    def __sub__(self, other: "Partial_Topology") -> "Partial_Topology":
        """Operator overloading for - to subtract a topology from another (splitting it)"""
        if self.chain != other.chain or self.fragment_sequence != other.fragment_sequence:
            raise ValueError("Cannot subtract topologies with different chains or sequences")

        # Check if the other topology is fully contained within this one
        if other.residue_start >= self.residue_start and other.residue_end <= self.residue_end:
            # Create remaining fragments
            result = []
            # Left fragment if exists
            if other.residue_start > self.residue_start:
                result.append(
                    Partial_Topology(
                        self.chain,
                        self.fragment_sequence,
                        self.residue_start,
                        other.residue_start - 1,
                        self.peptide_trim,
                        self.fragment_index,
                    )
                )
            # Right fragment if exists
            if other.residue_end < self.residue_end:
                result.append(
                    Partial_Topology(
                        self.chain,
                        self.fragment_sequence,
                        other.residue_end + 1,
                        self.residue_end,
                        self.peptide_trim,
                        self.fragment_index,
                    )
                )
            # If we have exactly one fragment, return it directly
            if len(result) == 1:
                return result[0]
            # If we have two fragments, we can't return both, so we'll raise an error
            # and suggest using extract_residues
            elif len(result) > 1:
                raise ValueError(
                    "Subtraction would result in multiple fragments. Use extract_residues instead."
                )
            else:
                raise ValueError("Subtraction would remove the entire topology")
        else:
            raise ValueError(
                "Cannot subtract a topology that is not fully contained within this one"
            )

    def copy(self) -> "Partial_Topology":
        """Create a copy of this topology"""
        return Partial_Topology(
            self.chain,
            self.fragment_sequence,
            self.residue_start,
            self.residue_end,
            self.peptide_trim,
            self.fragment_index,
        )


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING PARTIAL TOPOLOGY CLASS")
    print("=" * 80)

    print("\n1. BASIC STRING REPRESENTATION TESTS")
    print("-" * 50)
    # Test the Partial_Topology class string representation
    pt = Partial_Topology("A", "ALA", 1, 5)
    print(f"Multi-residue topology: {pt}")
    print(f"Detailed representation: {repr(pt)}")

    # Test with single residue
    single_res = Partial_Topology("B", "GLY", 10)
    print(f"Single residue topology: {single_res}")

    # Test with fragment index
    indexed = Partial_Topology("C", "ARG", 20, 25, fragment_index=3)
    print(f"Topology with index: {indexed}")

    print("\n2. EXTRACT RESIDUES METHOD TESTS (EXPECTED TO SUCCEED)")
    print("-" * 50)
    print("Extract with peptide=True (skips first two residues):")
    for res in pt.extract_residues():
        print(f"  {res}")

    print("\nExtract with peptide=False (includes all residues):")
    for res in pt.extract_residues(peptide=False):
        print(f"  {res}")

    print("\n3. OPERATOR TESTS")
    print("-" * 50)

    print("\n3.1 Addition operator (+) - EXPECTED TO SUCCEED")
    # Should work: adjacent residues
    res1 = Partial_Topology("D", "PHE", 30)
    res2 = Partial_Topology("D", "PHE", 31)
    try:
        combined = res1 + res2
        print(f"✓ SUCCESS: Combined adjacent topologies: {combined}")
    except ValueError as e:
        print(f"✗ UNEXPECTED FAILURE: {e}")

    # Should fail: non-adjacent residues
    res3 = Partial_Topology("D", "PHE", 33)
    try:
        combined = res1 + res3
        print(f"✗ UNEXPECTED SUCCESS: {combined}")
    except ValueError as e:
        print(f"✓ EXPECTED FAILURE: {e}")

    # Should fail: different chains
    res4 = Partial_Topology("E", "PHE", 31)
    try:
        combined = res1 + res4
        print(f"✗ UNEXPECTED SUCCESS: {combined}")
    except ValueError as e:
        print(f"✓ EXPECTED FAILURE: {e}")

    print("\n3.2 Subtraction operator (-) - VARIOUS OUTCOMES")

    # Case 1: Should succeed - removing residues from the end
    full1 = Partial_Topology("E", "TRP", 40, 45, fragment_index=5)
    end = Partial_Topology("E", "TRP", 44, 45)
    try:
        remaining = full1 - end
        print(f"✓ SUCCESS: Removed end section: {remaining}")
    except ValueError as e:
        print(f"✗ UNEXPECTED FAILURE: {e}")

    # Case 2: Should succeed - removing residues from the beginning
    full2 = Partial_Topology("E", "TRP", 40, 45, fragment_index=5)
    start = Partial_Topology("E", "TRP", 40, 41)
    try:
        remaining = full2 - start
        print(f"✓ SUCCESS: Removed start section: {remaining}")
    except ValueError as e:
        print(f"✗ UNEXPECTED FAILURE: {e}")

    # Case 3: Should fail - removing from the middle (would create two fragments)
    full3 = Partial_Topology("E", "TRP", 40, 45, fragment_index=5)
    middle = Partial_Topology("E", "TRP", 42, 43)
    try:
        remaining = full3 - middle
        print(f"✗ UNEXPECTED SUCCESS: {remaining}")
    except ValueError as e:
        print(f"✓ EXPECTED FAILURE: Removing middle section: {e}")

    # Case 4: Should fail - removing entire topology
    full4 = Partial_Topology("E", "TRP", 40, 45, fragment_index=5)
    all_of_it = Partial_Topology("E", "TRP", 40, 45)
    try:
        remaining = full4 - all_of_it
        print(f"✗ UNEXPECTED SUCCESS: {remaining}")
    except ValueError as e:
        print(f"✓ EXPECTED FAILURE: Removing entire topology: {e}")

    print("\n4. COPY METHOD TEST (EXPECTED TO SUCCEED)")
    print("-" * 50)
    original = Partial_Topology("F", "LYS", 50, 55, fragment_index=6)
    copy = original.copy()
    print(f"Original: {original}")
    print(f"Copy: {copy}")
    print(f"Are they equal? {original == copy}")
    print(f"Are they the same object? {original is copy}")

    print("\n" + "=" * 80)
