from typing import Dict, List, Optional

import MDAnalysis as mda
import numpy as np
from hdxrate import k_int_from_sequence
from MDAnalysis.core.groups import Residue  # Import Residue class

from jaxent.src.interfaces.topology.mda_adapter import mda_TopologyAdapter


def calculate_HDXrate(
    residue_group: mda.ResidueGroup, temperature: float = 300.0, pD: float = 7.0
) -> Dict[Residue, float]:
    """
    Calculate the intrinsic rate for a group of residues.
    This is using the HDXrate from sequence implementation:

    k_int_from_sequence('HHHHH', 300, 7.)
    array([0.00000000e+00, 2.62430718e+03, 6.29527446e+01, 6.29527446e+01,
    9.97734191e-01])


    Parameters
    ----------
    residue_group : mda.ResidueGroup
        The group of residues to calculate the HDX rate for.
    temperature : float
        The temperature in Kelvin.
    pD : float
        The pD value.

    Returns
    -------
    Dict[Residue, float]
        A dictionary mapping each residue to its HDX rate.
    """

    # Fix: residue_list should be a list of Residue objects, not a list of lists
    residue_list: list[Residue] = list(residue_group.residues)

    chains = set()

    for residue in residue_list:
        chain_id = mda_TopologyAdapter._get_chain_id(residue.atoms)
        if chain_id not in chains:
            chains.add(chain_id)

    assert len(chains) == 1, (
        "All residues must belong to the same chain.",
        f"Found chains: {chains}",
    )

    sequence = mda_TopologyAdapter._extract_sequence(residue_list, return_list=True)

    if not sequence:
        raise ValueError("No sequence could be extracted from the residue group.")

    k_int = k_int_from_sequence(sequence, temperature, pD)

    if not isinstance(k_int, np.ndarray):
        raise TypeError("k_int should be a numpy array.")

    if len(k_int) != len(residue_list):
        raise ValueError(
            f"Length of k_int ({len(k_int)}) does not match number of residues ({len(residue_list)})."
        )
    return {residue: k_int[i] for i, residue in enumerate(residue_list)}


def calculate_intrinsic_rates(
    universe: mda.Universe, residue_group: Optional[mda.ResidueGroup] = None
) -> Dict[Residue, float]:
    """
    Calculate intrinsic exchange rates for each residue in an MDAnalysis Universe or a specific ResidueGroup.

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe containing protein structure information
    residue_group : MDAnalysis.ResidueGroup, optional
        Specific residue group to calculate rates for. If None, all protein residues will be used.

    Returns:
    --------
    Dict[mda.Residue, float]: A dictionary mapping residue objects to their intrinsic rates.
    """

    # Default rate parameters from Bai et al., Proteins, 1993, 17, 75-86
    rate_params: Dict[str, float] = {
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
    rate_adjs: Dict[str, List[float]] = {
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

    def _adj_to_rates(rate_adjs: List[float]) -> float:
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

    # Identify first residues of each chain using original universe topology
    first_residues_per_chain = set()
    for segment in universe.segments:
        # Get protein atoms in this segment, then their residues
        segment_protein_atoms = segment.atoms.select_atoms("protein")
        if len(segment_protein_atoms) > 0:
            protein_residues_in_segment = segment_protein_atoms.residues
            if len(protein_residues_in_segment) > 0:
                first_residues_per_chain.add(protein_residues_in_segment[0])

    # Handle input selection for which residues to calculate rates for
    if residue_group is None:
        # No residue group provided, select all protein residues
        protein = universe.select_atoms("protein")
        residues = protein.residues
    else:
        # Use the provided residue group directly
        residues = residue_group

    # Initialize dictionary
    kint_dict: Dict[Residue, float] = {}

    # Calculate rates for each residue
    for curr in residues:
        # Skip first residue of each chain or PRO
        if curr in first_residues_per_chain or curr.resname == "PRO":
            kint_dict[curr] = np.inf
            continue

        # Find previous residue in the same chain/segment
        curr_segment = curr.segment
        curr_segment_protein_atoms = curr_segment.atoms.select_atoms("protein")
        curr_segment_residues = curr_segment_protein_atoms.residues

        # Find current residue index within its segment
        try:
            curr_idx_in_segment = list(curr_segment_residues).index(curr)
        except ValueError:
            # Current residue not found in protein residues of its segment
            kint_dict[curr] = np.inf
            continue

        if curr_idx_in_segment == 0:
            # This is the first residue in the segment (already handled above)
            kint_dict[curr] = np.inf
            continue

        # Get previous residue
        prev = curr_segment_residues[curr_idx_in_segment - 1]

        # Get rate adjustments
        curr_name = (
            curr.resname if curr.resname in rate_adjs else "ALA"
        )  # Default to ALA if unknown
        prev_name = prev.resname if prev.resname in rate_adjs else "ALA"

        # Special handling for termini
        if curr_idx_in_segment == 1:  # Second residue in chain (first non-terminal)
            prev_name = "NT"
        if curr_idx_in_segment == len(curr_segment_residues) - 1:  # Last residue in chain
            curr_name = "CT"

        # Combine adjustments
        curr_adjs = rate_adjs[curr_name][:2]  # Take first two values
        curr_adjs.extend(rate_adjs[prev_name][2:])  # Add last two values from previous

        # Calculate rate
        rate = _adj_to_rates(curr_adjs)
        kint_dict[curr] = rate

    return kint_dict
