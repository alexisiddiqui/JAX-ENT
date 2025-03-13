import warnings
from typing import List, cast

import MDAnalysis as mda
from MDAnalysis import Universe
from MDAnalysis.core.groups import ResidueGroup

from jaxent.types.topology import Partial_Topology


####################################################################################################
# TODO this needs to return lists or numbered dictionaries instead of sets
# perhaps this is fine and we simply just add an ignore str to the function
def find_common_residues(
    ensemble: List[Universe],
    include_mda_selection: str = "protein",
    ignore_mda_selection: str = "resname SOL",
) -> tuple[set[Partial_Topology], set[Partial_Topology]]:
    """
    Find the common residues across an ensemble of MDAnalysis Universe objects.

    Args:
        ensemble: List of MDAnalysis Universe objects to analyze

    Returns:
        Tuple containing:
        - Set of common residues (residue name, residue ID) present in all universes
        - Set of all residues present in any universe

    Raises:
        ValueError: If no common residues are found in the ensemble
    """
    # Extract residue sequences from each universe - this generates a tuple of topology fragments for each universe
    ensemble = [u.select_atoms(include_mda_selection) for u in ensemble]
    ensemble = [u.select_atoms(f"not {ignore_mda_selection}") for u in ensemble]

    ensemble_residue_sequences = [
        [
            Partial_Topology(
                chain=res.segid,
                fragment_sequence=res.resname,
                residue_start=res.resid,
            )
            for res in cast(ResidueGroup, u.residues)
        ]
        for u in ensemble
    ]

    # check that each list of topology fragments are unique within each universe

    ensemble_residue_sequences_set = [
        set(residue_sequence) for residue_sequence in ensemble_residue_sequences
    ]

    for resi_list, resi_set, universe in zip(
        ensemble_residue_sequences, ensemble_residue_sequences_set, ensemble
    ):
        if len(resi_list) != len(resi_set):
            raise ValueError(f"Residue sequences are not unique in universe {universe}")

    # Find common residues (intersection of all sets)
    common_residues = set.intersection(*ensemble_residue_sequences_set)

    # Find residues that don't match the common set
    excluded_residues = set.union(*ensemble_residue_sequences_set) - common_residues

    if len(common_residues) == 0:
        raise ValueError("No common residues found in the ensemble.")

    if len(excluded_residues) > 0:
        warnings.warn(
            f"Excluded {len(excluded_residues)} residues that are not common across all universes."
        )

    return common_residues, excluded_residues


# def get_residue_indices(universe: Universe, common_residues: set[Partial_Topology]) -> List[int]:
#     """
#     Get the indices of common residues in a specific universe.

#     Args:
#         universe: MDAnalysis Universe object
#         common_residues: Set of Topology Fragments for common residues
#     Returns:
#         sorted List of residue indices for the common residues in this universe
#     """


def get_residue_atom_pairs(
    universe: mda.Universe, common_residues: set[tuple[str, int]], atom_name: str
) -> list[tuple[int, int]]:
    """Generate residue and atom index pairs for specified atoms in common residues.

    Args:
        universe: MDAnalysis Universe containing the structure
        common_residues: Set of (resname, resid) tuples indicating residues to process
        atom_name: Name of the atom to select (e.g., "N" for amide nitrogen, "H" for amide hydrogen)

    Returns:
        List of (residue_id, atom_index) tuples for matching atoms in common residues

    Example:
        NH_pairs = get_residue_atom_pairs(universe, common_residues, "N")
        HN_pairs = get_residue_atom_pairs(universe, common_residues, "H")
    """
    residue_atom_pairs = []

    for residue in cast(ResidueGroup, universe.residues):
        if (residue.resname, residue.resid) in common_residues:
            # skip the first residue
            if residue.resid == 1:
                continue

            # skip PRo residues
            if residue.resname == "PRO":
                continue
            try:
                atom_idx = residue.atoms.select_atoms(f"name {atom_name}")[0].index
                residue_atom_pairs.append((residue.resid, atom_idx))
            except IndexError:
                continue

    return residue_atom_pairs
