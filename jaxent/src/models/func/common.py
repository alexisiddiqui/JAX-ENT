from typing import cast

import MDAnalysis as mda
from icecream import ic  # Import the icecream debugging library
from MDAnalysis.core.groups import ResidueGroup

# ####################################################################################################
# # Updated to use Partial_Topology methods for more robust residue identification
# defPartial_Topology.find_common_residues(
#     ensemble: List[Universe],
#     include_mda_selection: str = "protein",
#     ignore_mda_selection: str = "resname SOL",
# ) -> tuple[set[Partial_Topology], set[Partial_Topology]]:
#     """
#     Find the common residues across an ensemble of MDAnalysis Universe objects.

#     Args:
#         ensemble: List of MDAnalysis Universe objects to analyze
#         include_mda_selection: Selection string for atoms to include
#         ignore_mda_selection: Selection string for atoms to exclude

#     Returns:
#         Tuple containing:
#         - Set of common residues present in all universes
#         - Set of excluded residues (present in some but not all universes)

#     Raises:
#         ValueError: If no common residues are found in the ensemble
#     """
#     ic(len(ensemble), "Starting residue comparison across ensemble")
#     ic("Applying selection filters:", include_mda_selection, ignore_mda_selection)

#     # Extract residues from each universe using Partial_Topology.from_mda_universe()
#     ensemble_filtered_residues = []
#     ensemble_all_residues = []

#     for i, universe in enumerate(ensemble):
#         # Extract filtered residues (applying include/ignore selections)
#         filtered_residues = Partial_Topology.from_mda_universe(
#             universe,
#             mode="residue",  # Extract individual residues
#             include_selection=include_mda_selection,
#             exclude_selection=ignore_mda_selection,
#             exclude_termini=True,
#             renumber_residues=False,  # Keep original residue numbers for comparison
#         )
#         ensemble_filtered_residues.append(filtered_residues)

#         # Extract all residues (for identifying excluded residues)
#         all_residues = Partial_Topology.from_mda_universe(
#             universe,
#             mode="residue",
#             include_selection="all",  # Include everything
#             exclude_selection="",  # Exclude nothing
#             exclude_termini=False,  # Don't exclude termini
#             renumber_residues=False,
#         )
#         ensemble_all_residues.append(all_residues)

#         ic(i, len(filtered_residues), len(all_residues), "Filtered vs all residues in universe")

#     # Group residues by chain for each universe
#     def group_residues_by_chain(residue_list):
#         """Group Partial_Topology residues by chain"""
#         chain_groups = {}
#         for residue in residue_list:
#             chain = residue.chain
#             if chain not in chain_groups:
#                 chain_groups[chain] = {}
#             # Use residue number as key for easy lookup
#             resid = residue.residues[0]  # Single residue
#             chain_groups[chain][resid] = residue
#         return chain_groups

#     # Group filtered residues by chain for each universe
#     ensemble_filtered_by_chain = [
#         group_residues_by_chain(residues) for residues in ensemble_filtered_residues
#     ]

#     # Group all residues by chain for each universe
#     ensemble_all_by_chain = [
#         group_residues_by_chain(residues) for residues in ensemble_all_residues
#     ]

#     # Find chains common to all universes in the filtered sets
#     filtered_chains_per_universe = [
#         set(chain_dict.keys()) for chain_dict in ensemble_filtered_by_chain
#     ]
#     common_chains = (
#         set.intersection(*filtered_chains_per_universe) if filtered_chains_per_universe else set()
#     )

#     ic(len(common_chains), "Common chains found:", sorted(common_chains))

#     if not common_chains:
#         raise ValueError("No common chains found across all universes")

#     # Find union of all chains across all universes (for excluded analysis)
#     all_chains_per_universe = [set(chain_dict.keys()) for chain_dict in ensemble_all_by_chain]
#     all_unique_chains = set.union(*all_chains_per_universe) if all_chains_per_universe else set()

#     ic(len(all_unique_chains), "Total unique chains found:", sorted(all_unique_chains))

#     # For each common chain, identify common and excluded residues
#     common_residues = set()
#     excluded_residues = set()

#     for chain in common_chains:
#         ic(f"Analyzing chain {chain}")

#         # Get residue sets for this chain from each universe (filtered)
#         chain_filtered_residue_sets = []
#         for universe_chains in ensemble_filtered_by_chain:
#             if chain in universe_chains:
#                 chain_filtered_residue_sets.append(set(universe_chains[chain].keys()))
#             else:
#                 chain_filtered_residue_sets.append(set())

#         # Find residues common to all universes for this chain
#         if chain_filtered_residue_sets:
#             chain_common_resids = set.intersection(*chain_filtered_residue_sets)
#             ic(f"Chain {chain}: {len(chain_common_resids)} common residues")

#             # Add common residues to result set
#             for resid in chain_common_resids:
#                 # Get the residue object from the first universe (they should be equivalent)
#                 residue_obj = ensemble_filtered_by_chain[0][chain][resid]
#                 common_residues.add(residue_obj)

#     # For excluded residues, look at all chains and find residues not common to all
#     for chain in all_unique_chains:
#         # Get all residue sets for this chain from each universe
#         chain_all_residue_sets = []
#         chain_filtered_residue_sets = []

#         for universe_chains_all, universe_chains_filtered in zip(
#             ensemble_all_by_chain, ensemble_filtered_by_chain
#         ):
#             # All residues for this chain in this universe
#             if chain in universe_chains_all:
#                 chain_all_residue_sets.append(set(universe_chains_all[chain].keys()))
#             else:
#                 chain_all_residue_sets.append(set())

#             # Filtered residues for this chain in this universe
#             if chain in universe_chains_filtered:
#                 chain_filtered_residue_sets.append(set(universe_chains_filtered[chain].keys()))
#             else:
#                 chain_filtered_residue_sets.append(set())

#         # Find all unique residues for this chain
#         if chain_all_residue_sets:
#             all_chain_resids = set.union(*chain_all_residue_sets)

#             # Find residues common to all filtered sets for this chain
#             if chain_filtered_residue_sets and chain in common_chains:
#                 common_chain_resids = set.intersection(*chain_filtered_residue_sets)
#             else:
#                 common_chain_resids = set()

#             # Excluded residues are those in the full set but not common to all filtered
#             excluded_chain_resids = all_chain_resids - common_chain_resids

#             ic(f"Chain {chain}: {len(excluded_chain_resids)} excluded residues")

#             # Add excluded residues to result set
#             for resid in excluded_chain_resids:
#                 # Find the residue object from any universe that has it
#                 residue_obj = None
#                 for universe_chains in ensemble_all_by_chain:
#                     if chain in universe_chains and resid in universe_chains[chain]:
#                         residue_obj = universe_chains[chain][resid]
#                         break

#                 if residue_obj:
#                     excluded_residues.add(residue_obj)

#     ic(len(common_residues), "Total common residues found across all chains")
#     ic(len(excluded_residues), "Total excluded residues found across all chains")

#     # Validate results
#     if len(common_residues) == 0:
#         ic("ERROR: No common residues found")
#         raise ValueError("No common residues found in the ensemble.")

#     if len(excluded_residues) > 0:
#         ic("WARNING", len(excluded_residues), "residues excluded")
#         warnings.warn(
#             f"Excluded {len(excluded_residues)} residues that are not common across all universes."
#         )

#     # Log summary by chain
#     common_by_chain = {}
#     excluded_by_chain = {}

#     for residue in common_residues:
#         chain = residue.chain
#         if chain not in common_by_chain:
#             common_by_chain[chain] = 0
#         common_by_chain[chain] += 1

#     for residue in excluded_residues:
#         chain = residue.chain
#         if chain not in excluded_by_chain:
#             excluded_by_chain[chain] = 0
#         excluded_by_chain[chain] += 1

#     ic("Common residues by chain:", dict(sorted(common_by_chain.items())))
#     ic("Excluded residues by chain:", dict(sorted(excluded_by_chain.items())))

#     return common_residues, excluded_residues


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
