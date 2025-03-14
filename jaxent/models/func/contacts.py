from typing import Literal, Sequence, cast

import MDAnalysis as mda
import numpy as np
from icecream import ic  # Import icecream for debugging
from MDAnalysis.core.groups import AtomGroup  # type: ignore
from MDAnalysis.lib.distances import distance_array  # type: ignore


def calc_BV_contacts_universe(
    universe: mda.Universe,
    residue_atom_index: list[tuple[int, int]],
    contact_selection: Literal["heavy", "oxygen"],
    radius: float,
    residue_ignore: tuple[int, int] = (-2, 2),
    switch: bool = False,
) -> Sequence[Sequence[float]]:
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
    ic.configureOutput(prefix="DEBUG | ")
    ic(
        universe.trajectory.n_frames,
        len(residue_atom_index),
        contact_selection,
        radius,
        residue_ignore,
        switch,
    )

    # Set up selection string based on contact type
    if contact_selection == "heavy":
        sel_string = "not type H"  # Select all non-hydrogen atoms
        ic(sel_string)
    elif contact_selection == "oxygen":
        sel_string = "type O"  # Select all oxygen atoms
        ic(sel_string)
    else:
        raise ValueError("contact_selection must be either 'heavy' or 'oxygen'")

    # Initialize results array
    n_frames = len(universe.trajectory)
    n_targets = len(residue_atom_index)
    results = np.zeros((n_targets, n_frames))
    ic(n_frames, n_targets, results.shape)

    # Switching function if needed
    def rational_switch(r: np.ndarray, r0: float) -> np.ndarray:
        """Rational 6-12 switching function"""
        x = (r / r0) ** 6
        return (1 - x) / (1 - x * x)

    # Process each frame
    for ts in universe.trajectory:
        ic(f"Processing frame {ts.frame} of {n_frames}")
        # For each target atom
        for i, (res_idx, atom_idx) in enumerate(residue_atom_index):
            ic(f"Target {i}: res_idx={res_idx}, atom_idx={atom_idx}")
            # Get the target atom
            target_atom = cast(AtomGroup, universe.atoms)[atom_idx]
            target_res = target_atom.residue
            ic(f"Target atom: {target_atom}, residue: {target_res.resid}")

            # Get residue range to exclude
            exclude_start = target_res.resid + residue_ignore[0]
            exclude_end = target_res.resid + residue_ignore[1]
            ic(f"Excluding residues {exclude_start} to {exclude_end}")

            # Select contact atoms, excluding residue range
            selection_query = f"{sel_string} and not (resid {exclude_start}:{exclude_end})"
            ic(selection_query)
            contact_atoms = universe.select_atoms(selection_query)
            ic(f"Found {len(contact_atoms)} potential contact atoms")

            if len(contact_atoms) == 0:
                ic("No contact atoms found, skipping")
                continue

            # Calculate distances
            ic(f"Calculating distances for target position {target_atom.position}")
            distances = distance_array(
                target_atom.position[np.newaxis, :],
                contact_atoms.positions,
                box=universe.dimensions,
            )[0]
            ic(f"Distance range: min={np.min(distances):.3f}, max={np.max(distances):.3f}")

            if switch:
                # Apply switching function to all distances
                contact_values = rational_switch(distances, radius)
                # Sum up contact values (weighted contacts)
                results[i, ts.frame] = np.sum(contact_values[distances <= radius * 1])
                ic(f"Switch mode: counted {results[i, ts.frame]:.2f} weighted contacts")
            else:
                # Count contacts within radius
                contacts_count = np.sum(distances <= radius)
                results[i, ts.frame] = contacts_count
                ic(f"Hard cutoff mode: counted {contacts_count} contacts within {radius}Å")

    ic(f"Final results shape: {np.array(results).shape}")
    ################################################################################
    return results.tolist()


# sort this out to use numpy arrays
################################################################################
