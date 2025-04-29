import warnings
from typing import List, cast

import MDAnalysis as mda
from icecream import ic  # Import the icecream debugging library
from MDAnalysis import Universe
from MDAnalysis.core.groups import ResidueGroup

from jaxent.interfaces.topology import Partial_Topology


####################################################################################################
# TODO this needs to return lists or numbered dictionaries instead of sets
# perhaps this is fine and we simply just add an ignore str to the function
# we need an easier way to get direct access to the residues
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
    ic(len(ensemble), "Starting residue comparison across ensemble")

    # Extract residue sequences from each universe while applying filters
    ic("Applying selection filters:", include_mda_selection, ignore_mda_selection)

    # First, select atoms based on include criteria
    ensemble = [u.select_atoms(include_mda_selection) for u in ensemble]
    ic(len(ensemble), "Universes after include filter")

    # Then exclude atoms based on ignore criteria
    ensemble = [u.select_atoms(f"not {ignore_mda_selection}") for u in ensemble]
    ic(len(ensemble), "Universes after exclude filter")

    # For each universe, create a list of Partial_Topology objects representing each residue
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

    # Log the number of residues found in each universe
    for i, residues in enumerate(ensemble_residue_sequences):
        ic(i, len(residues), "Residues in universe")

    # Convert lists to sets for efficient comparison operations
    # Sets can only contain unique elements, which helps validate that residue sequences are unique
    ensemble_residue_sequences_set = [
        set(residue_sequence) for residue_sequence in ensemble_residue_sequences
    ]

    # Verify that each universe has unique residue identifiers (no duplicates)
    # If the length of the list and set differ, there are duplicate residues
    for i, (resi_list, resi_set, universe) in enumerate(
        zip(ensemble_residue_sequences, ensemble_residue_sequences_set, ensemble)
    ):
        ic(i, len(resi_list), len(resi_set), "Checking for duplicates")
        if len(resi_list) != len(resi_set):
            ic("DUPLICATE RESIDUES FOUND", len(resi_list) - len(resi_set), "duplicates")
            raise ValueError(f"Residue sequences are not unique in universe {universe}")

    # Find residues common to all universes using set intersection
    common_residues = set.intersection(*ensemble_residue_sequences_set)
    ic(len(common_residues), "Common residues found across all universes")

    # Find residues present in at least one universe but not in all (the difference)
    excluded_residues = set(set.union(*ensemble_residue_sequences_set) - common_residues)
    ic(len(excluded_residues), "Residues excluded (not common across all universes)")

    # Raise error if no common residues were found
    if len(common_residues) == 0:
        ic("ERROR: No common residues found")
        raise ValueError("No common residues found in the ensemble.")

    # Warn if some residues were excluded
    if len(excluded_residues) > 0:
        ic("WARNING", len(excluded_residues), "residues excluded")
        warnings.warn(
            f"Excluded {len(excluded_residues)} residues that are not common across all universes."
        )

    return common_residues, excluded_residues


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
    ic(len(common_residues), atom_name, "Starting residue-atom pair collection")

    residue_atom_pairs = []
    skipped_first = False
    skipped_pro = 0
    missing_atoms = 0

    for residue in cast(ResidueGroup, universe.residues):
        res_identifier = (residue.resname, residue.resid)
        # Check if this residue is in our set of common residues
        if res_identifier in common_residues:
            ic(residue.resname, residue.resid, "Processing residue")

            # Skip the first residue (typically doesn't have the specified atoms in protein chains)
            if residue.resid == 1:
                ic("Skipping first residue", residue.resid, residue.resname)
                skipped_first = True
                continue

            # Skip proline residues (e.g., no amide hydrogen)
            if residue.resname == "PRO":
                ic("Skipping PRO residue", residue.resid)
                skipped_pro += 1
                continue

            try:
                # Try to find the specified atom in this residue
                atom_selection = residue.atoms.select_atoms(f"name {atom_name}")
                ic(residue.resid, f"Found {len(atom_selection)} atoms named", atom_name)

                atom_idx = atom_selection[0].index
                residue_atom_pairs.append((residue.resid, atom_idx))
                ic(residue.resid, atom_idx, "Added residue-atom pair")

            except IndexError:
                # If the atom is not found in this residue
                missing_atoms += 1
                ic(residue.resid, residue.resname, f"Missing {atom_name} atom")
                continue

    ic(len(residue_atom_pairs), "Total residue-atom pairs found")
    ic(
        skipped_first,
        skipped_pro,
        missing_atoms,
        "Summary: first_residue_skipped, prolines_skipped, missing_atoms",
    )

    return residue_atom_pairs
