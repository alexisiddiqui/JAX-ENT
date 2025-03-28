from dataclasses import dataclass
from typing import Optional, cast


###################################################################################################
# TODO add add and subtract methods to combine or split topologies
# TODO add copy method to create a new topology
@dataclass()
class Partial_Topology:
    """
    Dataclass that holds the information of a single topology fragment.
    This is usually a residue but may also be a group of residues such as peptides.
    """

    chain: str | int
    fragment_sequence: str  # resname (residue) or single letter codes (peptide) or atom name (atom)
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


###################################################################################################
