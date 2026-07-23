from collections.abc import Sequence
from typing import Literal

from MDAnalysis import Universe
import numpy as np
from MDAnalysis.core.groups import AtomGroup  # type: ignore
from MDAnalysis.lib.distances import distance_array  # type: ignore
from tqdm import tqdm

from jaxent.src.interfaces.topology.mda_adapter import mda_TopologyAdapter


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
    n_jobs: int = 10,
    environment_selection: str = "all",
) -> Sequence[Sequence[float]]:
    """Calculate Best--Vendruscolo contacts for backbone amide donors.

    The parity construction counts protein non-hydrogen atoms around the
    amide N and protein oxygen atoms around the amide H.  Sequence neighbours
    are masked by their ordinal position within the same protein chain; this
    remains correct for non-contiguous residue numbering and never masks an
    equally numbered residue in another chain.

    Args:
        universe: MDAnalysis Universe object
        target_atoms: AtomGroup containing target atoms
        contact_selection: "heavy" or "oxygen"
        radius: Cutoff distance in Angstroms
        residue_ignore: Relative residue range to ignore (start, end)
        switch: Use the legacy JAX-ENT rational switched-contact sensitivity.
        n_jobs: Number of OpenMP threads to use for distance calculation
        environment_selection: MDAnalysis selection defining atoms that may
            protect the amide. General featurisation includes all explicitly
            modelled atoms; a protein-only experiment must request ``protein``.

    Returns:
        List of contact counts per frame (N_targets, N_frames)
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if residue_ignore[0] > residue_ignore[1]:
        raise ValueError("residue_ignore must be ordered (lower, upper)")
    if len(target_atoms) == 0:
        raise ValueError("target_atoms must not be empty")

    if contact_selection == "heavy":
        atom_selection = "not type H"
    elif contact_selection == "oxygen":
        atom_selection = "type O"
    else:
        raise ValueError("contact_selection must be either 'heavy' or 'oxygen'")

    try:
        environment_atoms = universe.select_atoms(environment_selection)
    except AttributeError:
        # Minimal in-memory test Universes may omit resnames and therefore
        # cannot evaluate MDAnalysis' ``protein`` keyword.  They contain no
        # solvent/ligand classification to preserve, so all atoms are the
        # only meaningful environment in that specific case.
        if environment_selection != "protein":
            raise
        environment_atoms = universe.atoms
    all_contact_atoms = environment_atoms.select_atoms(atom_selection)
    if len(all_contact_atoms) == 0:
        return np.zeros((len(target_atoms), universe.trajectory.n_frames)).tolist()

    # Map protein residues to chain-local sequence positions.  Non-protein
    # environment atoms deliberately have no ordinal and are never removed as
    # covalent sequence neighbours.
    ordinal_by_resindex: dict[int, tuple[str, int]] = {}
    try:
        protein_residues = universe.select_atoms("protein").residues
    except AttributeError:
        protein_residues = universe.atoms.residues
    residues_by_chain: dict[str, list] = {}
    for residue in protein_residues:
        chain = str(mda_TopologyAdapter._get_chain_id(residue))
        residues_by_chain.setdefault(chain, []).append(residue)
    for chain, residues in residues_by_chain.items():
        for ordinal, residue in enumerate(residues):
            ordinal_by_resindex[int(residue.resindex)] = (chain, ordinal)

    target_positions = []
    for atom in target_atoms:
        position = ordinal_by_resindex.get(int(atom.resindex))
        if position is None:
            raise ValueError(f"target atom {atom} is not in the protein sequence")
        target_positions.append(position)

    ignore_mask = np.zeros((len(target_atoms), len(all_contact_atoms)), dtype=bool)
    for target_index, (target_chain, target_ordinal) in enumerate(target_positions):
        for contact_index, atom in enumerate(all_contact_atoms):
            contact_position = ordinal_by_resindex.get(int(atom.resindex))
            if contact_position is None:
                continue
            contact_chain, contact_ordinal = contact_position
            ordinal_delta = contact_ordinal - target_ordinal
            ignore_mask[target_index, contact_index] = (
                contact_chain == target_chain
                and residue_ignore[0] <= ordinal_delta <= residue_ignore[1]
            )

    # Initialize results
    n_frames = universe.trajectory.n_frames
    n_targets = len(target_atoms)
    results = np.zeros((n_targets, n_frames), dtype=float)

    # Switching function definition (vectorized)
    def rational_switch(r: np.ndarray, r0: float) -> np.ndarray:
        # Algebraically equivalent to (1-x)/(1-x^2), including its 0.5
        # limit at r=r0, without a removable singularity.
        return 1.0 / (1.0 + (r / r0) ** 6)

    calc_backend = "OpenMP" if n_jobs > 1 else "serial"
    for frame_index, _ in enumerate(
        tqdm(universe.trajectory, total=n_frames, desc="Processing Frames")
    ):
        dists = distance_array(
            target_atoms.positions,
            all_contact_atoms.positions,
            box=universe.dimensions,
            backend=calc_backend,
        )
        dists[ignore_mask] = np.inf

        if switch:
            within_cutoff = dists <= radius
            switch_values = np.zeros_like(dists)
            switch_values[within_cutoff] = rational_switch(dists[within_cutoff], radius)
            results[:, frame_index] = np.sum(switch_values, axis=1)
        else:
            results[:, frame_index] = np.sum(dists <= radius, axis=1)

    return results.tolist()
