from typing import List, Tuple

import mdtraj as md
import numpy as np


def get_contact_atoms(
    universe: md.Trajectory,
    residue_idx: int,
    contact_selection: str,
    residue_ignore: Tuple[int, int] = (-2, 2),
) -> np.ndarray:
    """Get indices of atoms to check contacts with for a given residue."""
    # Get range of residues to exclude (n-2 to n+2 by default)
    exclude_start = residue_idx + residue_ignore[0]  # typically n-2
    exclude_end = residue_idx + residue_ignore[1] + 1  # typically n+2
    excluded_residues = range(max(0, exclude_start), exclude_end)

    # Get all matching atoms except those in excluded residues
    all_atoms = []
    for atom in universe.topology.atoms:
        # Skip atoms in excluded residues
        if atom.residue.index in excluded_residues:
            continue

        # Apply selection criteria based on contact type
        if contact_selection == "oxygen":
            if atom.element.symbol == "O":
                all_atoms.append(atom.index)
        else:  # heavy atoms
            # Match original BV behavior: include all non-hydrogen atoms
            if atom.element.symbol != "H":
                all_atoms.append(atom.index)

    return np.array(all_atoms)


def calc_BV_contacts_mdtraj(
    universe: md.Trajectory,
    residue_atom_index: List[Tuple[int, int]],
    contact_selection: str,
    radius: float,
    residue_ignore: Tuple[int, int] = (-2, 2),
    switch: bool = False,
) -> List[List[float]]:
    """Calculate contacts for multiple residue/atom pairs using Best-Vendruscolo method."""
    if switch:
        raise NotImplementedError("Switch function not implemented in this version")

    all_contacts = []

    for resid, atom_idx in residue_atom_index:
        # Get the actual residue index from the topology
        residue = next(r for r in universe.topology.residues if r.resSeq == resid)

        # Get atoms to check contacts with
        contact_atoms = get_contact_atoms(
            universe,
            residue.index,  # Use residue.index instead of resSeq
            contact_selection,
            residue_ignore,
        )

        # Calculate contacts
        contacts = calc_contacts(universe, atom_idx, contact_atoms, radius)

        all_contacts.append(contacts)

    return all_contacts


def calc_contacts(
    universe: md.Trajectory, query_idx: int, contact_indices: np.ndarray, radius: float
) -> List[int]:
    """Calculate contacts using a hard cutoff."""
    # Handle single atom case
    query_idx = np.array([query_idx])

    # Use MDTraj neighbor search
    neighbors = md.compute_neighbors(universe, radius, query_idx, haystack_indices=contact_indices)

    # Count contacts per frame
    return [len(frame_neighbors) for frame_neighbors in neighbors]
