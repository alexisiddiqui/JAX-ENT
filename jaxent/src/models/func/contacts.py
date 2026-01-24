from collections.abc import Sequence
from typing import Literal

import MDAnalysis as mda
from MDAnalysis import Universe
import numpy as np
from icecream import ic  # Import icecream for debugging
from MDAnalysis.core.groups import AtomGroup  # type: ignore
from MDAnalysis.lib.distances import distance_array  # type: ignore
from tqdm import tqdm  # Add tqdm import


# def calc_BV_contacts_universe(
#     universe: Universe,
#     target_atoms: AtomGroup,
#     contact_selection: Literal["heavy", "oxygen"],
#     radius: float,
#     residue_ignore: tuple[int, int] = (-2, 2),
#     switch: bool = False,
# ) -> Sequence[Sequence[float]]:
#     """Calculate contacts between specified atoms and surrounding atoms using MDAnalysis.

#     Implements contact calculation for Best-Vendruscolo HDX prediction model.

#     Args:
#         universe: MDAnalysis Universe object containing trajectory
#         target_atoms: AtomGroup containing target atoms (usually amide N or H atoms)
#         contact_selection: Type of atoms to count contacts with:
#             - "heavy": All heavy atoms
#             - "oxygen": Only oxygen atoms
#         radius: Cutoff distance for contacts in Angstroms
#         residue_ignore: Range of residues to ignore relative to target (start, end)
#         switch: Whether to use switching function instead of hard cutoff

#     Returns:
#         List of contact counts per frame for each target atom

#     Note:
#         For switching function, implements rational 6-12 switching as in original code.
#         The switching function calculates a weighted contact value between 0 and 1
#         for atoms beyond the cutoff radius.
#     """
#     ic.configureOutput(prefix="DEBUG | ")
#     ic(
#         universe.trajectory.n_frames,
#         len(target_atoms),
#         contact_selection,
#         radius,
#         residue_ignore,
#         switch,
#     )

#     # Set up selection string based on contact type
#     if contact_selection == "heavy":
#         sel_string = "not type H"  # Select all non-hydrogen atoms
#         ic(sel_string)
#     elif contact_selection == "oxygen":
#         sel_string = "type O"  # Select all oxygen atoms
#         ic(sel_string)
#     else:
#         raise ValueError("contact_selection must be either 'heavy' or 'oxygen'")

#     # Initialize results array
#     n_frames = len(universe.trajectory)
#     n_targets = len(target_atoms)
#     results = np.zeros((n_targets, n_frames))
#     ic(n_frames, n_targets, results.shape)

#     # Switching function if needed
#     def rational_switch(r: np.ndarray, r0: float) -> np.ndarray:
#         """Rational 6-12 switching function"""
#         x = (r / r0) ** 6
#         return (1 - x) / (1 - x * x)

#     # Process each frame
#     for ts in tqdm(universe.trajectory, desc="Frames"):  # Wrap trajectory with tqdm
#         ic(f"Processing frame {ts.frame} of {n_frames}")

#         # For each target atom
#         for i, target_atom in enumerate(target_atoms):
#             ic(f"Target {i}: atom={target_atom}, residue={target_atom.residue.resid}")

#             target_res = target_atom.residue

#             # Get residue range to exclude
#             exclude_start = target_res.resid + residue_ignore[0]
#             exclude_end = target_res.resid + residue_ignore[1]
#             ic(f"Excluding residues {exclude_start} to {exclude_end}")

#             # Select contact atoms, excluding residue range
#             selection_query = f"{sel_string} and not (resid {exclude_start}:{exclude_end})"
#             ic(selection_query)
#             contact_atoms = universe.select_atoms(selection_query)
#             ic(f"Found {len(contact_atoms)} potential contact atoms")

#             if len(contact_atoms) == 0:
#                 ic("No contact atoms found, skipping")
#                 continue

#             # Calculate distances
#             ic(f"Calculating distances for target position {target_atom.position}")
#             distances = distance_array(
#                 target_atom.position[np.newaxis, :],
#                 contact_atoms.positions,
#                 box=universe.dimensions,
#             )[0]
#             ic(f"Distance range: min={np.min(distances):.3f}, max={np.max(distances):.3f}")

#             if switch:
#                 # Apply switching function to all distances
#                 contact_values = rational_switch(distances, radius)
#                 # Sum up contact values (weighted contacts)
#                 results[i, ts.frame] = np.sum(contact_values[distances <= radius * 1])
#                 ic(f"Switch mode: counted {results[i, ts.frame]:.2f} weighted contacts")
#             else:
#                 # Count contacts within radius
#                 contacts_count = np.sum(distances <= radius)
#                 results[i, ts.frame] = contacts_count
#                 ic(f"Hard cutoff mode: counted {contacts_count} contacts within {radius}Å")

#     ic(f"Final results shape: {np.array(results).shape}")  # results (residues, frames)
#     return results.tolist()


def calc_BV_contacts_universe(
    universe: Universe,
    target_atoms: AtomGroup,
    contact_selection: Literal["heavy", "oxygen"],
    radius: float,
    residue_ignore: tuple[int, int] = (-2, 2),
    switch: bool = False,
    n_jobs: int = 10  # Added parameter for OpenMP threads
) -> Sequence[Sequence[float]]:
    """
    Calculate contacts for Best-Vendruscolo HDX prediction using vectorized 
    operations and MDAnalysis OpenMP backend.

    Args:
        universe: MDAnalysis Universe object
        target_atoms: AtomGroup containing target atoms
        contact_selection: "heavy" or "oxygen"
        radius: Cutoff distance in Angstroms
        residue_ignore: Relative residue range to ignore (start, end)
        switch: Use rational switching function
        n_jobs: Number of OpenMP threads to use for distance calculation

    Returns:
        List of contact counts per frame (N_targets, N_frames)
    """
    ic.configureOutput(prefix="DEBUG | ")
    
    # --- 1. PRE-CALCULATION PHASE ---
    
    # Define selection string
    if contact_selection == "heavy":
        sel_string = "not type H"
    elif contact_selection == "oxygen":
        sel_string = "type O"
    else:
        raise ValueError("contact_selection must be either 'heavy' or 'oxygen'")

    # Select ALL potential contact atoms once (Global selection)
    # We rely on indices, assuming topology does not change during trajectory
    all_contact_atoms = universe.select_atoms(sel_string)
    
    ic(len(target_atoms), len(all_contact_atoms), contact_selection)

    # Pre-calculate Residue Masking Matrix
    # We create a boolean matrix (N_targets x N_contacts) where True = Ignore
    # This replaces the slow 'resid X:Y' query inside the loop
    target_resids = target_atoms.resids[:, np.newaxis]  # Column vector
    contact_resids = all_contact_atoms.resids[np.newaxis, :]  # Row vector
    
    # Calculate relative residue difference matrix
    resid_diff = contact_resids - target_resids
    
    # Create mask: True if atom is within the ignored residue range
    # range is inclusive in selection strings, so we use <= and >=
    ignore_mask = (resid_diff >= residue_ignore[0]) & (resid_diff <= residue_ignore[1])
    
    ic(f"Mask shape: {ignore_mask.shape}. Ignored pairs: {np.sum(ignore_mask)}")

    # Initialize results
    n_frames = universe.trajectory.n_frames
    n_targets = len(target_atoms)
    results = np.zeros((n_targets, n_frames))

    # Switching function definition (vectorized)
    def rational_switch(r: np.ndarray, r0: float) -> np.ndarray:
        # Avoid division by zero if r close to r0 (though logic below handles it)
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (r / r0) ** 6
            val = (1 - x) / (1 - x * x)
        # Fix singularity at r = r0 (limit is 0.5)
        val[np.isnan(val)] = 0.5 
        return val

    # --- 2. TRAJECTORY LOOP ---
    
    # Setup OpenMP backend check
    # Note: 'backend' argument requires MDAnalysis >= 2.0.0
    calc_backend = 'OpenMP' if n_jobs > 1 else 'serial'
    ic(f"Using backend: {calc_backend} with {n_jobs} threads (if OpenMP)")

    for ts in tqdm(universe.trajectory, total=n_frames, desc="Processing Frames"):
        
        # Calculate full Distance Matrix (N_targets x N_all_contacts)
        # This is where OpenMP provides speedup
        dists = distance_array(
            target_atoms.positions,
            all_contact_atoms.positions,
            box=universe.dimensions,
            backend=calc_backend
        )

        # Apply the pre-calculated residue mask
        # We set distances of ignored residues to Infinity so they fail the cutoff check
        dists[ignore_mask] = np.inf

        if switch:
            # 1. Identify valid contacts within radius (boolean mask)
            within_cutoff = dists <= radius
            
            # 2. Calculate switch values for ALL pairs (vectorized)
            # We filter by within_cutoff to match original logic (and save computation time)
            # However, for pure vectorization, we calc all, then zero out failures
            switch_vals = rational_switch(dists, radius)
            
            # 3. Zero out values outside radius or masked residues
            switch_vals[~within_cutoff] = 0.0
            
            # 4. Sum across rows (axis 1 = sum contacts per target)
            results[:, ts.frame] = np.sum(switch_vals, axis=1)
            
        else:
            # Hard cutoff mode
            # Count how many items in each row are <= radius
            contact_counts = np.sum(dists <= radius, axis=1)
            results[:, ts.frame] = contact_counts

    ic(f"Final results shape: {results.shape}")
    
    return results.tolist()
