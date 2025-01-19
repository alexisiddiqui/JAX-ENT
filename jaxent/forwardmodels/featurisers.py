from hdxrate import k_int_from_sequence
from MDAnalysis import Universe

from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import ForwardModel


class BV_model(ForwardModel):
    """
    The BV or Best-Vendruscolo model for HDX-MS data.
    This computes protection factors using heavy and h bond acceptor (O) contacts.
    The list of universes must contain compatible residue sequences.
    """

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.common_residues: set[tuple[str, int]]
        self.common_k_ints: list[float]
        self.compatability = {HDX_peptide, HDX_protection_factor}
        self.temperature = temperature

    def initialise(self, ensemble: list[Universe]) -> bool:
        # find the common residue sequence

        residue_sequences = [(u.residues.resnames, u.residues.resids) for u in ensemble]
        # find the set of common residues in the ensemble
        common_residues = set.intersection(*[set(r) for r in residue_sequences])

        # find the set of all residues in the ensemble
        all_residues = set.union(*[set(r) for r in residue_sequences])

        if len(common_residues) == 0:
            raise ValueError("No common residues found in the ensemble.")

        if common_residues < all_residues:
            UserWarning("Some residues are not present in all universes. These will be ignored.")

        self.common_residues = common_residues

        self.common_k_ints = [
            k_int_from_sequence(
                list(*common_residues),
                self.temperature,
            )
        ]

        return True
