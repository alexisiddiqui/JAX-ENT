from abc import abstractmethod
from dataclasses import dataclass
from typing import NewType, Optional

import numpy as np

m_key = NewType("m_key", str)
m_id = NewType("m_id", str)


@dataclass()
class Topology_Fragment:
    """
    Dataclass that holds the information of a single topology fragment.
    This is usually a residue but may also be a group of residues such as peptides.
    """

    chain: str | int
    fragment_sequence: str  # resname (residue) or single letter codes (peptide) or atom name (atom)
    residue_start: int  # inclusive, if a peptide, this is the first residue - not the
    residue_end: int | None = None  # inclusive - if None, then this is a single residue
    fragment_index: Optional[int] = (
        None  # atom or peptide index - optional for residues - required for peptides and atoms
    )

    def __post_init__(self):
        if self.residue_end is None:
            self.residue_end: int = self.residue_start
        self.length = self.residue_end - self.residue_start + 1
        if self.length > 2:
            self.peptide_residues = [i for i in range(self.residue_start + 2, self.residue_end + 1)]

    def __eq__(self, other) -> bool:
        """Equal comparison - checks if all attributes are the same"""
        if not isinstance(other, Topology_Fragment):
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

    def extract_residues(self, peptide: bool = True) -> list["Topology_Fragment"]:
        """
        Extracts the residues from a peptide or atom fragment
        """
        if self.length == 1:
            return [self]
        else:
            if peptide is True:
                return [
                    Topology_Fragment(self.chain, self.fragment_sequence, res)
                    for res in self.peptide_residues
                ]
            else:
                return [
                    Topology_Fragment(self.chain, self.fragment_sequence, res)
                    for res in range(self.residue_start, self.residue_end + 1)
                ]


@dataclass()
class Experimental_Fragment:
    """
    Base class for experimental data - grouped into subdomain fragments
    Limtation is that it only covers a single chain - which should be fine in most cases.
    """

    top: Topology_Fragment | None
    # key: m_key

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the child class.")
