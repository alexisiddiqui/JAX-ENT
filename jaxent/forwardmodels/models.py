from dataclasses import dataclass
from typing import Optional

import numpy as np
from MDAnalysis import Universe

from jaxent.config.base import BaseConfig
from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import ForwardModel
from jaxent.forwardmodels.functions import (
    calc_BV_contacts_universe,
    calculate_intrinsic_rates,
    get_residue_atom_pairs,
)


@dataclass()
class BV_model_Config(BaseConfig):
    temperature: float = 300
    bv_bc: float = 0.35
    bv_bh: float = 2.0
    ph: float = 7
    heavy_radius: float = 6.5
    o_radius: float = 2.4


class BV_input_features(Input_Features):
    heavy_conacts: list  # (frames, residues)
    acceptor_contacts: list  # (frames, residues)


class BV_output_features(Output_Features):
    log_Pf: list
    k_ints: Optional[list]


def BV_forward(
    input_features: BV_input_features,
    parameters: tuple[float, float] = (BV_model_Config.bv_bc, BV_model_Config.bv_bh),
) -> BV_output_features:
    pass


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
        self.compatability: HDX_peptide | HDX_protection_factor
        self.temperature = temperature

    def initialise(self, ensemble: list[Universe]) -> bool:
        """
        To do: probably move this to the forward model?
        """

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

        # find the total number of frmes in the ensemble
        self.n_frames = sum([u.trajectory.n_frames for u in ensemble])
        # common residue indices
        common_residue_indices = [u.residues.resids for u in ensemble]

        # calculate intrinsic rates for the common residues
        k_ints = []
        for u in ensemble:
            k_ints.append(calculate_intrinsic_rates(u))

        self.common_k_ints = np.mean(k_ints, axis=0)[common_residue_indices]

        return True

    def featurise(self, ensemble: list[Universe]) -> list[list[list[float]]]:
        """
        Calculate BV model features (heavy atom contacts and H-bond acceptor contacts)
        for each residue in the ensemble.

        Args:
            ensemble: List of MDAnalysis Universe objects

        Returns:
            List of features per universe, per residue, per frame:
            [[[heavy_contacts, o_contacts], ...], ...]

        Notes:
            - Heavy atom cutoff: 0.65 nm
            - O atom cutoff: 0.24 nm
            - Excludes residues within ±2 positions
        """
        features = []

        # Constants
        HEAVY_RADIUS = 6.5  # 0.65 nm in Angstroms
        O_RADIUS = 2.4  # 0.24 nm in Angstroms

        for universe in ensemble:
            universe_features = []

            # Get residue indices and atom indices for amide N and H atoms
            NH_residue_atom_pairs = get_residue_atom_pairs(universe, self.common_residues, "N")
            HN_residue_atom_pairs = get_residue_atom_pairs(universe, self.common_residues, "H")

            # Calculate heavy atom contacts
            heavy_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=NH_residue_atom_pairs,
                contact_selection="heavy",
                radius=HEAVY_RADIUS,
            )

            # Calculate O atom contacts (H-bond acceptors)
            o_contacts = calc_BV_contacts_universe(
                universe=universe,
                residue_atom_index=HN_residue_atom_pairs,
                contact_selection="oxygen",
                radius=O_RADIUS,
            )

            # Combine features for each residue
            for h_contacts, o_contacts in zip(heavy_contacts, o_contacts):
                universe_features.append([float(h_contacts), float(o_contacts)])

            features.append(universe_features)

        return features
