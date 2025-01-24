from dataclasses import dataclass
from typing import Literal

import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array

from jaxent.config.base import BaseConfig
from jaxent.datatypes import HDX_peptide, HDX_protection_factor
from jaxent.forwardmodels.base import ForwardModel


def calculate_intrinsic_rates(universe_or_atomgroup):
    """
    Calculate intrinsic exchange rates for each residue in an MDAnalysis Universe or AtomGroup.

    Parameters:
    -----------
    universe_or_atomgroup : MDAnalysis.Universe or MDAnalysis.AtomGroup
        MDAnalysis Universe or AtomGroup containing protein structure information

    Returns:
    --------
    tuple : (rates, residue_ids)
        rates : numpy array of intrinsic rates for each residue
        residue_ids : numpy array of corresponding residue sequence numbers
    """

    # Default rate parameters from Bai et al., Proteins, 1993, 17, 75-86
    rate_params = {
        "lgkAref": 2.04,
        "lgkBref": 10.36,
        "lgkWref": -1.5,
        "EaA": 14.0,
        "EaB": 17.0,
        "EaW": 19.0,
        "R": 0.001987,
        "Tref": 293,
        "Texp": 298,
        "pKD": 14.87,
        "pD": 7.4,
    }

    # Rate adjustments from Nguyen et al., J. Am. Soc. Mass Spec., 2018, 29, 1936-1939
    # Each value is [lgAL, lgBL, lgAR, lgBR]
    rate_adjs = {
        "ALA": [0.00, 0.00, 0.00, 0.00],
        "ARG": [-0.59, 0.08, -0.32, 0.22],
        "ASN": [-0.58, 0.49, -0.13, 0.32],
        "ASP": [0.90, 0.10, 0.58, -0.18],
        "ASH": [-0.90, 0.69, -0.12, 0.60],  # Protonated ASP
        "CYS": [-0.54, 0.62, -0.46, 0.55],
        "CYS2": [-0.74, 0.55, -0.58, 0.46],  # Disulfide
        "GLY": [-0.22, -0.03, 0.22, 0.17],
        "GLN": [-0.47, 0.06, -0.27, 0.20],
        "GLU": [-0.90, -0.11, 0.31, -0.15],
        "GLH": [-0.60, 0.24, -0.27, 0.39],  # Protonated GLU
        "HIS": [0.00, -0.10, 0.00, 0.14],
        "HIP": [-0.80, 0.80, -0.51, 0.83],
        "ILE": [-0.91, -0.73, -0.59, -0.23],
        "LEU": [-0.57, -0.58, -0.13, -0.21],
        "LYS": [-0.56, -0.04, -0.29, 0.12],
        "MET": [-0.64, -0.01, -0.28, 0.11],
        "PHE": [-0.52, -0.24, -0.43, 0.06],
        "PRO": [0.00, 0.00, -0.19, -0.24],
        "SER": [-0.44, 0.37, -0.39, 0.30],
        "THR": [-0.79, -0.07, -0.47, 0.20],
        "TRP": [-0.40, -0.41, -0.44, -0.11],
        "TYR": [-0.41, -0.27, -0.37, 0.05],
        "VAL": [-0.74, -0.70, -0.30, -0.14],
        "NT": [0.00, 0.00, -1.32, 1.62],  # N-term NH3+
        "CT": [0.96, -1.80, 0.00, 0.00],  # C-term COO-
        "CTH": [0.05, 0.00, 0.00, 0.00],  # C-term COOH
    }

    def _adj_to_rates(rate_adjs):
        """Helper function to calculate intrinsic rates from adjustments"""
        # Calculate reference rates at experimental temperature
        lgkAexp = rate_params["lgkAref"] - (rate_params["EaA"] / np.log(10) / rate_params["R"]) * (
            1.0 / rate_params["Texp"] - 1.0 / rate_params["Tref"]
        )
        lgkBexp = rate_params["lgkBref"] - (rate_params["EaB"] / np.log(10) / rate_params["R"]) * (
            1.0 / rate_params["Texp"] - 1.0 / rate_params["Tref"]
        )
        lgkWexp = rate_params["lgkWref"] - (rate_params["EaW"] / np.log(10) / rate_params["R"]) * (
            1.0 / rate_params["Texp"] - 1.0 / rate_params["Tref"]
        )

        # Calculate log(kA||kB||kW)
        lgkA = lgkAexp + rate_adjs[0] + rate_adjs[2] - rate_params["pD"]
        lgkB = lgkBexp + rate_adjs[1] + rate_adjs[3] - rate_params["pKD"] + rate_params["pD"]
        lgkW = lgkWexp + rate_adjs[1] + rate_adjs[3]

        return 10**lgkA + 10**lgkB + 10**lgkW

    # Handle input type
    if isinstance(universe_or_atomgroup, mda.Universe):
        u = universe_or_atomgroup
        protein = u.select_atoms("protein")
    else:  # AtomGroup
        protein = universe_or_atomgroup.select_atoms("protein")

    # Get unique residues (MDAnalysis specific)
    residues = protein.residues
    n_residues = len(residues)

    # Initialize arrays
    kints = np.zeros(n_residues)
    residue_ids = np.zeros(n_residues, dtype=int)

    # Calculate rates for each residue
    for i, curr in enumerate(residues):
        residue_ids[i] = curr.resid

        # Skip first residue or PRO
        if i == 0 or curr.resname == "PRO":
            kints[i] = np.inf
            continue

        prev = residues[i - 1]

        # Get rate adjustments
        curr_name = (
            curr.resname if curr.resname in rate_adjs else "ALA"
        )  # Default to ALA if unknown
        prev_name = prev.resname if prev.resname in rate_adjs else "ALA"

        # Special handling for termini
        if i == 1:  # First non-PRO residue
            prev_name = "NT"
        if i == n_residues - 1:  # Last residue
            curr_name = "CT"

        # Combine adjustments
        curr_adjs = rate_adjs[curr_name][:2]  # Take first two values
        curr_adjs.extend(rate_adjs[prev_name][2:])  # Add last two values from previous

        # Calculate rate
        kints[i] = _adj_to_rates(curr_adjs)

    return kints, residue_ids


@dataclass()
class BV_model_Config(BaseConfig):
    temperature: float
    bv_bc: float
    bv_bh: float
    ph: float
    heavy_radius: float = 6.5
    o_radius: float = 2.4


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

            # Get residue indices and atom indices for amide N atoms in common residues
            NH_residue_atom_pairs = []
            for residue in universe.residues:
                if (residue.resname, residue.resid) in self.common_residues:
                    try:
                        N_idx = residue.atoms.select_atoms("name N")[0].index
                        NH_residue_atom_pairs.append((residue.resid, N_idx))
                    except IndexError:
                        continue

            HN_residue_atom_pairs = []
            for residue in universe.residues:
                if (residue.resname, residue.resid) in self.common_residues:
                    try:
                        H_idx = residue.atoms.select_atoms("name H")[0].index
                        HN_residue_atom_pairs.append((residue.resid, H_idx))
                    except IndexError:
                        continue

            # Calculate heavy atom contacts
            heavy_contacts = calc_contacts_universe(
                universe=universe,
                residue_atom_index=NH_residue_atom_pairs,
                contact_selection="heavy",
                radius=HEAVY_RADIUS,
            )

            # Calculate O atom contacts (H-bond acceptors)
            o_contacts = calc_contacts_universe(
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


def calc_contacts_universe(
    universe: mda.Universe,
    residue_atom_index: list[tuple[int, int]],
    contact_selection: Literal["heavy", "oxygen"],
    radius: float,
    residue_ignore: tuple[int, int] = (-2, 2),
    switch: bool = False,
) -> list[list[float]]:
    """Calculate contacts between specified atoms and surrounding atoms using MDAnalysis.

    Implements contact calculation for Best-Vendruscolo HDX prediction model.

    Args:
        universe: MDAnalysis Universe object containing trajectory
        residue_atom_index: List of (residue_idx, atom_idx) tuples for target atoms (usually amide N)
        contact_selection: Type of atoms to count contacts with:
            - "heavy": All heavy atoms
            - "oxygen": Only oxygen atoms
        radius: Cutoff distance for contacts in Angstroms
        residue_ignore: Range of residues to ignore relative to target (start, end)
        switch: Whether to use switching function instead of hard cutoff

    Returns:
        List of contact counts per frame for each target atom

    Note:
        For switching function, implements rational 6-12 switching as in original code.
        The switching function calculates a weighted contact value between 0 and 1
        for atoms beyond the cutoff radius.
    """
    # Set up selection string based on contact type
    if contact_selection == "heavy":
        sel_string = "not name H*"  # Select all non-hydrogen atoms
    elif contact_selection == "oxygen":
        sel_string = "element O"  # Select all oxygen atoms
    else:
        raise ValueError("contact_selection must be either 'heavy' or 'oxygen'")

    # Initialize results array
    n_frames = len(universe.trajectory)
    n_targets = len(residue_atom_index)
    results = np.zeros((n_targets, n_frames))

    # Switching function if needed
    def rational_switch(r: np.ndarray, r0: float) -> np.ndarray:
        """Rational 6-12 switching function"""
        x = (r / r0) ** 6
        return (1 - x) / (1 - x * x)

    # Process each frame
    for ts in universe.trajectory:
        # For each target atom
        for i, (res_idx, atom_idx) in enumerate(residue_atom_index):
            # Get the target atom
            target_atom = universe.atoms[atom_idx]
            target_res = target_atom.residue

            # Get residue range to exclude
            exclude_start = target_res.resid + residue_ignore[0]
            exclude_end = target_res.resid + residue_ignore[1]

            # Select contact atoms, excluding residue range
            contact_atoms = universe.select_atoms(
                f"{sel_string} and not (resid {exclude_start}:{exclude_end})"
            )

            if len(contact_atoms) == 0:
                continue

            # Calculate distances
            distances = distance_array(
                target_atom.position[np.newaxis, :],
                contact_atoms.positions,
                box=universe.dimensions,
            )[0]

            if switch:
                # Apply switching function to all distances
                contact_values = rational_switch(distances, radius)
                # Sum up contact values (weighted contacts)
                results[i, ts.frame] = np.sum(contact_values[distances <= radius * 1.5])
            else:
                # Count contacts within radius
                results[i, ts.frame] = np.sum(distances <= radius)

    return results.tolist()
