### MDA Universe methods
from typing import Mapping, Optional, Union

import MDAnalysis as mda
import numpy as np
from MDAnalysis.core.groups import Residue
from tqdm import tqdm

from jaxent.src.interfaces.topology.core import Partial_Topology
from jaxent.src.interfaces.topology.factory import TopologyFactory
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons
from jaxent.src.interfaces.topology.utils import group_set_by_chain, rank_and_index
from jaxent.src.models.func.common import compute_trajectory_average_com_distances


class mda_TopologyAdapter:
    """Adapter class for converting MDAnalysis Universe objects to Partial_Topology objects."""

    ### No need to refactor
    @staticmethod
    def _build_chain_selection_string(
        universe: mda.Universe, chain_id: str, base_selection: Optional[str] = None
    ) -> tuple[str, mda.AtomGroup]:
        """
        Utility method to build a selection string for a specific chain,
        accounting for available attributes in the universe.

        Args:
            universe: MDAnalysis Universe object
            chain_id: Chain ID to select
            base_selection: Optional base selection to combine with chain selection

        Returns:
            tuple: (selection_string, fallback_atomgroup)
            - selection_string: MDAnalysis selection string for the chain
            - fallback_atomgroup: AtomGroup to use if selection fails
        """
        # Check available attributes for chain selection
        has_segid = hasattr(universe.atoms[0], "segid") if len(universe.atoms) > 0 else False
        has_chainid = hasattr(universe.atoms[0], "chainid") if len(universe.atoms) > 0 else False

        # Build appropriate selection string based on available attributes
        chain_selection_parts = []
        if has_segid:
            chain_selection_parts.append(f"segid {chain_id}")
        if has_chainid:
            chain_selection_parts.append(f"chainid {chain_id}")

        # Initialize fallback_atoms as empty
        fallback_atoms = mda.AtomGroup([], universe)

        if not chain_selection_parts:
            # If neither attribute is available, we can't create a selection string
            selection_string = ""
        else:
            # Join selection parts with OR
            chain_selection = " or ".join(chain_selection_parts)

            # Combine with base selection if provided
            if base_selection:
                selection_string = f"({base_selection}) and ({chain_selection})"
            else:
                selection_string = chain_selection

        return selection_string, fallback_atoms

    @staticmethod
    def _get_mda_group_sort_key(
        group: Union[mda.ResidueGroup, mda.AtomGroup, Residue],
    ) -> tuple[int, tuple[int, ...], float, int]:
        """Generate a sort key for an MDAnalysis group that matches Partial_Topology ranking."""
        if isinstance(group, Residue):
            residues = [group]
            chain_id = getattr(group.atoms[0], "segid", None) or getattr(
                group.atoms[0], "chainid", "A"
            )
        elif isinstance(group, mda.ResidueGroup):
            residues = [res for res in group.residues]  # Iterate explicitly
            if len(residues) == 0:
                raise ValueError("ResidueGroup contains no residues")

            # Check that all residues are from the same chain
            chain_ids = set()
            for residue in residues:
                residue_chain_id = getattr(residue.atoms[0], "segid", None) or getattr(
                    residue.atoms[0], "chainid", "A"
                )
                chain_ids.add(residue_chain_id)

            if len(chain_ids) > 1:
                raise ValueError(
                    f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                )

            chain_id = list(chain_ids)[0]
        elif isinstance(group, mda.AtomGroup):
            residues = list(group.residues)
            if len(residues) == 0:
                raise ValueError("AtomGroup contains no residues")

            # Check that all atoms are from the same chain
            chain_ids = set()
            for atom in group:
                atom_chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
                chain_ids.add(atom_chain_id)

            if len(chain_ids) > 1:
                raise ValueError(f"AtomGroup contains atoms from multiple chains: {chain_ids}")

            chain_id = list(chain_ids)[0]
        else:
            raise TypeError("group must be a Residue, ResidueGroup, or AtomGroup")

        # Ensure chain is string and uppercase
        if isinstance(chain_id, str):
            chain_id = chain_id.upper()
        else:
            chain_id = str(chain_id)

        # Get residue numbers
        resids = [r.resid for r in residues]

        # Calculate average residue position
        avg_residue = sum(resids) / len(resids)

        # Calculate length
        length = len(resids)

        # Generate chain score (same logic as Partial_Topology._get_chain_score)
        # # TODO - this should be replaced with a common function for both this and Partial_Topology.
        # This allows proper alphanumeric sorting: A < B < ... < 1 < 2 < ...
        chain_score = tuple(ord(c) for c in chain_id)

        # Return sort key: (chain_len, chain_score, avg_residue, -length)
        # Negative length to sort longer fragments first (descending)
        return (len(chain_id), chain_score, avg_residue, -length)

    @staticmethod
    def _remove_duplicates_by_chain(
        residue_topologies: list[Partial_Topology] | set[Partial_Topology],
    ) -> set[Partial_Topology]:
        """This exists as the __hash__ method may not consider chains"""
        by_chain = group_set_by_chain(set(residue_topologies))

        unique = set()

        for chain_id, chain_topos in tqdm(
            by_chain.items(), desc="Processing excluded residues by chain"
        ):
            chain_unique = set()
            for excluded_topo in chain_topos:
                # Check if this topology is already represented in the same chain
                is_duplicate = False
                for existing_topo in chain_unique:
                    if PairwiseTopologyComparisons.contains_topology(excluded_topo, existing_topo):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    chain_unique.add(excluded_topo)

            unique.update(chain_unique)

        return unique

    @staticmethod
    def _find_excluded_residues_by_chain(
        ensemble: list[mda.Universe],
        topologies_by_universe: list[list[Partial_Topology]],
        include_selection: list[str] = ["protein"],
        termini_chain_selection: list[str] = ["protein"],
        renumber_residues: bool = True,
    ) -> set[Partial_Topology]:
        excluded_residue_topologies = set()

        # Get all possible residues (without ignore selection, without termini exclusion)
        all_chain_topologies_by_universe: list[list[Partial_Topology]] = []
        for i, universe in enumerate(ensemble):
            try:
                all_chain_topos = mda_TopologyAdapter.from_mda_universe(
                    universe,
                    mode="chain",
                    include_selection=include_selection[i],
                    exclude_selection="",  # No exclusions as
                    exclude_termini=False,  # Include termini
                    termini_chain_selection=termini_chain_selection[i],
                    renumber_residues=renumber_residues,  # Use parameter
                )
                all_chain_topologies_by_universe.append(all_chain_topos)
            except Exception as e:
                raise ValueError(f"Failed to extract all topologies from universe {i}: {e}")

        # Group chain topologies by chain using the new method
        included_by_chain = group_set_by_chain(set(*topologies_by_universe))

        # For each universe, find residues that are excluded
        for i, (all_chain_topos, topos) in enumerate(
            zip(all_chain_topologies_by_universe, topologies_by_universe)
        ):
            for all_chain_topo in all_chain_topos:
                chain_id = all_chain_topo.chain

                if chain_id in included_by_chain:
                    # Remove included residues from this chain
                    try:
                        excluded_chain_topo = TopologyFactory.remove_residues_by_topologies(
                            all_chain_topo, list(included_by_chain[chain_id])
                        )
                        excluded_residues = TopologyFactory.extract_residues(
                            excluded_chain_topo, use_peptide_trim=False
                        )
                        excluded_residue_topologies.update(excluded_residues)
                    except ValueError:
                        # No residues remaining after removal
                        continue
                else:
                    # Entire chain was excluded
                    excluded_residues = TopologyFactory.extract_residues(
                        all_chain_topo, use_peptide_trim=False
                    )
                    excluded_residue_topologies.update(excluded_residues)

        return excluded_residue_topologies

    @staticmethod
    def _find_common_residues_by_chain(
        all_topos_by_chain: Mapping[str | int, set[Partial_Topology]],
    ) -> set[Partial_Topology]:
        """Find common residues across all topologies by chain."""
        common_residue_topologies = set()
        for chain_id, topos_for_chain in all_topos_by_chain.items():
            if not topos_for_chain:
                continue

            try:
                # Find intersection of residues across all universes for this chain
                common_chain_topo = TopologyFactory.merge(
                    topos_for_chain,
                    trim=False,
                    intersection=True,
                    merged_name=f"common_chain_{chain_id}",
                )

                # Extract individual residues from the common chain
                residue_topos = TopologyFactory.extract_residues(
                    common_chain_topo, use_peptide_trim=False
                )
                common_residue_topologies.update(residue_topos)

            except ValueError as e:
                # If merge fails because there are no common residues, continue
                if "No common residues found" in str(e):
                    continue
                # Re-raise other ValueErrors
                raise e
        return common_residue_topologies

    @staticmethod
    def _find_included_residues_by_chain(
        ensemble: list[mda.Universe],
        include_selection: list[str] = ["protein"],
        exclude_selection: list[str] = ["resname SOL"],
        termini_chain_selection: list[str] = ["protein"],
        exclude_termini: list[bool] = [True],
        renumber_residues: bool = True,
    ) -> list[list[Partial_Topology]]:
        """Extract included chain topologies from each universe in the ensemble.
        Args:
            ensemble: List of MDAnalysis Universe objects
            include_selection: Selection string or list of strings for atoms to include.
                            If a list, it must have the same length as the ensemble.
            exclude_selection: Selection string or list of strings for atoms to exclude.
                            If a list, it must have the same length as the ensemble.
            termini_chain_selection: Selection string or list of strings for terminal residues.
            renumber_residues: If True, renumber residues starting from 1 for each chain.
                            If False, preserve original residue numbers.


        Returns:
            List of lists containing Partial_Topology objects for each universe.
            Each sublist corresponds to the topologies extracted from that universe.

        Raises:
            ValueError: If no topologies could be extracted from a universe
            ImportError: If MDAnalysis is not available
        """
        included_chain_topologies_by_universe: list[list[Partial_Topology]] = []
        for i, universe in enumerate(ensemble):
            try:
                # Extract chain topologies with filtering
                chain_topos: list[Partial_Topology] = mda_TopologyAdapter.from_mda_universe(
                    universe,
                    mode="chain",
                    include_selection=include_selection[i],
                    exclude_selection=exclude_selection[i],
                    exclude_termini=exclude_termini[i],
                    termini_chain_selection=termini_chain_selection[i],
                    renumber_residues=renumber_residues,  # Use parameter
                )
                included_chain_topologies_by_universe.append(chain_topos)
            except Exception as e:
                raise ValueError(f"Failed to extract topologies from universe {i}: {e}")

        return included_chain_topologies_by_universe

    @staticmethod
    def find_common_residues(
        ensemble: list[mda.Universe],
        include_selection: Union[str, list[str]] = "protein",
        exclude_selection: Union[str, list[str]] = "resname SOL",
        termini_chain_selection: Union[str, list[str]] = "protein",
        exclude_termini: Union[bool, list[bool]] = True,
        renumber_residues: bool = True,  # Changed default to False
    ) -> tuple[set[Partial_Topology], set[Partial_Topology]]:
        """Find common residues across an ensemble of MDAnalysis Universe objects.

        Args:
            ensemble: List of MDAnalysis Universe objects to analyze
            include_selection: Selection string or list of strings for atoms to include.
                            If a list, it must have the same length as the ensemble.
            exclude_selection: Selection string or list of strings for atoms to exclude.
                            If a list, it must have the same length as the ensemble.
            renumber_residues: If True, renumber residues starting from 1 for each chain.
                            If False, preserve original residue numbers.

        Returns:
            Tuple containing:
            - Set of common residues present in all universes
            - Set of excluded residues (present in some but not all universes)

        Raises:
            ValueError: If no common residues are found or if selection list lengths are incorrect.
            ImportError: If MDAnalysis is not available
        """
        if not ensemble:
            raise ValueError("Empty ensemble provided")

        # Normalize selection strings to lists
        if isinstance(include_selection, str):
            include_selection = [include_selection] * len(ensemble)
        if isinstance(exclude_selection, str):
            exclude_selection = [exclude_selection] * len(ensemble)
        if isinstance(termini_chain_selection, str):
            termini_chain_selection = [termini_chain_selection] * len(ensemble)
        if isinstance(exclude_termini, bool):
            exclude_termini = [exclude_termini] * len(ensemble)

        if len(include_selection) != len(ensemble):
            raise ValueError("include_selection list must have the same length as the ensemble")
        if len(exclude_selection) != len(ensemble):
            raise ValueError("exclude_selection list must have the same length as the ensemble")
        if len(termini_chain_selection) != len(ensemble):
            raise ValueError(
                "termini_chain_selection list must have the same length as the ensemble"
            )
        if len(exclude_termini) != len(ensemble):
            raise ValueError("exclude_termini list must have the same length as the ensemble")
        # Extract included chain topologies from each universe
        ############ requires: ensemble, include_selection, exclude_selection, renumber_residues,
        included_chain_topologies_by_universe = (
            mda_TopologyAdapter._find_included_residues_by_chain(
                ensemble=ensemble,
                include_selection=include_selection,
                exclude_selection=exclude_selection,
                termini_chain_selection=termini_chain_selection,
                renumber_residues=renumber_residues,
                exclude_termini=exclude_termini,
            )
        )
        # Group all chain topologies by chain ID across all universes
        all_topos_by_chain = group_set_by_chain(set(*included_chain_topologies_by_universe))

        # Find common residues by chain
        # using intersection merge
        common_residue_topologies = mda_TopologyAdapter._find_common_residues_by_chain(
            all_topos_by_chain=all_topos_by_chain
        )
        # Extract excluded residues
        excluded_residue_topologies = mda_TopologyAdapter._find_excluded_residues_by_chain(
            ensemble=ensemble,
            topologies_by_universe=included_chain_topologies_by_universe,
            include_selection=include_selection,
            termini_chain_selection=termini_chain_selection,
            renumber_residues=renumber_residues,
        )

        excluded_residue_topologies = mda_TopologyAdapter._remove_duplicates_by_chain(
            excluded_residue_topologies
        )

        if not common_residue_topologies:
            raise ValueError("No common residues found in the ensemble")

        return common_residue_topologies, excluded_residue_topologies

    @staticmethod
    def get_residuegroup_reordering_indices(
        residue_group: Union[mda.ResidueGroup, mda.AtomGroup],
    ) -> list[int]:
        """Get indices to reorder individual residues in a ResidueGroup/AtomGroup by topology ranking.

        This method takes a single ResidueGroup or AtomGroup containing multiple residues
        and returns indices to sort the individual residues according to Partial_Topology
        ranking logic.

        Args:
            residue_group: MDAnalysis ResidueGroup or AtomGroup containing multiple residues

        Returns:
            List of indices for reordering individual residues within the group.
            Use as: sorted_residues = [residue_group.residues[i] for i in indices]

        Raises:
            ValueError: If group contains no residues
            TypeError: If group is not a ResidueGroup or AtomGroup

        Example:
            >>> indices = Partial_Topology.get_residuegroup_reordering_indices(protein_residues)
            >>> sorted_residues = [protein_residues.residues[i] for i in indices]
        """
        if isinstance(residue_group, mda.AtomGroup):
            residues = list(residue_group.residues)
        elif isinstance(residue_group, mda.ResidueGroup):
            residues = [res for res in residue_group]  # Iterate explicitly
        else:
            raise TypeError("residue_group must be a ResidueGroup or AtomGroup")

        if not residues:
            raise ValueError("Group contains no residues")

        # Create (sort_key, original_index) pairs
        keyed_residues = []
        for i, residue in enumerate(residues):
            sort_key = mda_TopologyAdapter._get_mda_group_sort_key(residue)
            keyed_residues.append((sort_key, i))

        # Sort by sort_key and extract original indices
        keyed_residues.sort(key=lambda x: x[0])
        reorder_indices = [original_idx for _, original_idx in keyed_residues]

        return reorder_indices

    @staticmethod
    def partial_topology_pairwise_distances(
        topologies: list[Partial_Topology],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        compound: str = "group",
        pbc: bool = True,
        backend: str = "OpenMP",
        verbose: bool = True,
        check_trim: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute trajectory-averaged pairwise center-of-mass distances between Partial_Topology objects.

        Parameters
        ----------
        topologies : List[Partial_Topology]
            List of Partial_Topology objects to compute distances between
        universe : mda.Universe
            MDAnalysis Universe containing the trajectory
        include_selection : str, default "protein"
            MDAnalysis selection string for atoms to include
        exclude_selection : str, optional
            MDAnalysis selection string for atoms to exclude
        exclude_termini : bool, default True
            If True, account for N and C terminal residue exclusion
        termini_chain_selection : str, default "protein"
            Selection string used for terminal identification
        renumber_residues : bool, default True
            If True, assume residue numbers were renumbered from 1
        start : int, optional
            Start frame for analysis
        stop : int, optional
            Stop frame for analysis
        step : int, optional
            Step size for frame iteration
        compound : str, default 'group'
            How to compute center of mass for each topology
        pbc : bool, default True
            Whether to account for periodic boundary conditions
        backend : str, default 'OpenMP'
            Backend for distance calculations
        verbose : bool, default True
            Whether to show progress bar
        check_trim : bool, default False
            If True, use peptide_residues for peptides

        Returns
        -------
        distance_matrix : np.ndarray
            Trajectory-averaged distance matrix of shape (n_topologies, n_topologies)
        distance_std : np.ndarray
            Standard deviation of distances over trajectory
        """
        if not topologies:
            raise ValueError("topologies list cannot be empty")

        # Convert Partial_Topology objects to MDAnalysis groups
        mda_groups = []
        for topo in topologies:
            try:
                # Use existing to_mda_group method from Partial_Topology
                group = mda_TopologyAdapter.to_mda_group(
                    topologies={topo},  # Convert single topology to set
                    universe=universe,
                    include_selection=include_selection,
                    exclude_selection=exclude_selection,
                    exclude_termini=exclude_termini,
                    termini_chain_selection=termini_chain_selection,
                    renumber_residues=renumber_residues,
                    mda_atom_filtering=None,  # Get residues, not atoms
                )

                # Convert ResidueGroup to AtomGroup for COM calculations
                if isinstance(group, mda.ResidueGroup):
                    group = group.atoms

                mda_groups.append(group)

            except Exception as e:
                raise ValueError(f"Failed to convert topology {topo} to MDAnalysis group: {e}")

        if verbose:
            print(f"Converted {len(topologies)} topologies to MDAnalysis groups")

        # Compute distances
        distance_matrix, distance_std = compute_trajectory_average_com_distances(
            universe=universe,
            group_list=mda_groups,
            start=start,
            stop=stop,
            step=step,
            compound=compound,
            pbc=pbc,
            backend=backend,
            verbose=verbose,
        )

        return distance_matrix, distance_std

    @staticmethod
    def to_mda_residue_dict(
        topologies: Union[set[Partial_Topology], list[Partial_Topology]],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
    ) -> dict[Union[str, int], list[int]]:
        """
        Extract residue information as a dictionary of chain:[residue_indices].

        This method uses to_mda_group to get the residues and then formats them
        into a dictionary.

        Args:
            topologies: Set or list of Partial_Topology objects to convert.
            universe: MDAnalysis Universe object to select from.
            include_selection: MDAnalysis selection string for atoms to include.
            exclude_selection: Optional MDAnalysis selection string for atoms to exclude.
            exclude_termini: If True, account for N and C terminal residue exclusion.
            termini_chain_selection: Selection string used for terminal identification.
            renumber_residues: If True, assume residue numbers were renumbered from 1.
                              If False, use original residue numbers.

        Returns:
            A dictionary mapping chain ID to a list of residue indices.
        """
        residue_group = mda_TopologyAdapter.to_mda_group(
            topologies=topologies,
            universe=universe,
            include_selection=include_selection,
            exclude_selection=exclude_selection,
            exclude_termini=exclude_termini,
            termini_chain_selection=termini_chain_selection,
            renumber_residues=renumber_residues,
            mda_atom_filtering=None,  # We want residues, not atoms
        )

        if isinstance(residue_group, mda.AtomGroup):
            residues = residue_group.residues
        else:
            residues = residue_group

        residue_dict = {}
        for res in residues:
            chain_id = getattr(res, "segid", None) or getattr(res, "chainid", "A")
            if chain_id not in residue_dict:
                residue_dict[chain_id] = []
            residue_dict[chain_id].append(res.resid)

        # Sort residue indices for each chain
        for chain_id in residue_dict:
            residue_dict[chain_id].sort()

        return residue_dict

    ### No need to refactor

    ### require some attention
    @staticmethod
    def get_atomgroup_reordering_indices(
        mda_groups: list[Union[mda.ResidueGroup, mda.AtomGroup]],
        universe: mda.Universe,
        target_topologies: Optional[list[Partial_Topology]] = None,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = True,
    ) -> list[int]:
        """Get indices to reorder a list of MDAnalysis groups to match topology order.

        This method is essential for the BV_model.featurise() method to ensure that
        feature calculation results are ordered consistently with Partial_Topology ranking.

        Args:
            mda_groups: List of MDAnalysis ResidueGroup or AtomGroup objects to be reordered
            universe: MDAnalysis Universe object
            target_topologies: Optional list of Partial_Topology objects in desired order.
                             If None, will extract topologies from mda_groups and use default ranking.
            include_selection: Selection string for atoms to include
            exclude_selection: Selection string for atoms to exclude
            exclude_termini: Whether to exclude terminal residues
            termini_chain_selection: Selection for terminal identification
            renumber_residues: Whether residues were renumbered from 1

        Returns:
            List of indices for reordering mda_groups to match target topology order.
            Use as: reordered_features = [original_features[i] for i in indices]

        Raises:
            ValueError: If any topology cannot be mapped to mda_groups, or if
                       topology residues are not contained within the chain bounds
        """
        if not mda_groups:
            return []

        # If no target topologies provided, extract them from mda_groups and rank them
        if target_topologies is None:
            target_topologies = []
            for mda_group in mda_groups:
                # Convert each MDA group to a Partial_Topology
                if isinstance(mda_group, mda.ResidueGroup):
                    residues = [res for res in mda_group]  # Iterate explicitly
                    if not residues:
                        continue

                    if len(residues) == 1:
                        # Single residue
                        res = residues[0]
                        chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                            res.atoms[0], "chainid", "A"
                        )
                        topo = TopologyFactory.from_single(
                            chain=chain_id,
                            residue=res.resid,
                            fragment_name=f"{chain_id}_{res.resname}{res.resid}",
                        )
                    else:
                        # Multiple residues
                        chain_ids = set()
                        resids = []
                        for res in residues:
                            chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                                res.atoms[0], "chainid", "A"
                            )
                            chain_ids.add(chain_id)
                            resids.append(res.resid)

                        if len(chain_ids) > 1:
                            raise ValueError(
                                f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                            )

                        chain_id = list(chain_ids)[0]
                        topo = TopologyFactory.from_residues(
                            chain=chain_id,
                            residues=resids,
                            fragment_name=f"{chain_id}_multi",
                        )
                elif isinstance(mda_group, mda.AtomGroup):
                    residues = list(mda_group.residues)
                    if len(residues) == 1:
                        # Single residue
                        res = residues[0]
                        chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                            res.atoms[0], "chainid", "A"
                        )
                        topo = TopologyFactory.from_single(
                            chain=chain_id,
                            residue=res.resid,
                            fragment_name=f"{chain_id}_{res.resname}{res.resid}",
                        )
                    else:
                        # Multiple residues
                        chain_ids = set()
                        resids = []
                        for res in residues:
                            chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                                res.atoms[0], "chainid", "A"
                            )
                            chain_ids.add(chain_id)
                            resids.append(res.resid)

                        if len(chain_ids) > 1:
                            raise ValueError(
                                f"AtomGroup contains residues from multiple chains: {chain_ids}"
                            )

                        chain_id = list(chain_ids)[0]
                        topo = TopologyFactory.from_residues(
                            chain=chain_id,
                            residues=resids,
                            fragment_name=f"{chain_id}_multi",
                        )
                else:
                    raise TypeError(f"Unsupported group type: {type(mda_group)}")

                target_topologies.append(topo)

            # Rank and index the topologies using default ordering
            target_topologies = rank_and_index(target_topologies, check_trim=False)

        # Build a mapping from topology characteristics to mda_groups index
        mda_group_lookup = {}

        # Build renumbering mapping if needed
        if renumber_residues:
            renumber_mapping = mda_TopologyAdapter._build_renumbering_mapping(
                universe, exclude_termini, termini_chain_selection
            )

        for i, mda_group in enumerate(mda_groups):
            if isinstance(mda_group, mda.ResidueGroup):
                residues = [res for res in mda_group]  # Iterate explicitly
                if not residues:
                    continue

                chain_ids = set()
                resids = []
                for res in residues:
                    chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                        res.atoms[0], "chainid", "A"
                    )
                    chain_ids.add(chain_id)

                    if renumber_residues:
                        original_resid = res.resid
                        renumbered_resid = None
                        for (ch, new_id), orig_id in renumber_mapping.items():
                            if ch == chain_id and orig_id == original_resid:
                                renumbered_resid = new_id
                                break
                        if renumbered_resid is not None:
                            resids.append(renumbered_resid)
                    else:
                        resids.append(res.resid)

                if len(chain_ids) > 1:
                    raise ValueError(
                        f"ResidueGroup contains residues from multiple chains: {chain_ids}"
                    )

                if not resids:
                    continue  # Skip if no valid resids

                chain_id = list(chain_ids)[0]
                lookup_key = (chain_id, frozenset(resids))

            elif isinstance(mda_group, mda.AtomGroup):
                residues = list(mda_group.residues)
                if not residues:
                    continue

                chain_ids = set()
                resids = []
                for res in residues:
                    chain_id = getattr(res.atoms[0], "segid", None) or getattr(
                        res.atoms[0], "chainid", "A"
                    )
                    chain_ids.add(chain_id)

                    if renumber_residues:
                        original_resid = res.resid
                        renumbered_resid = None
                        for (ch, new_id), orig_id in renumber_mapping.items():
                            if ch == chain_id and orig_id == original_resid:
                                renumbered_resid = new_id
                                break
                        if renumbered_resid is not None:
                            resids.append(renumbered_resid)
                    else:
                        resids.append(res.resid)

                if len(chain_ids) > 1:
                    raise ValueError(
                        f"AtomGroup contains residues from multiple chains: {chain_ids}"
                    )

                if not resids:
                    continue  # Skip if no valid resids

                chain_id = list(chain_ids)[0]
                lookup_key = (chain_id, frozenset(resids))

            mda_group_lookup[lookup_key] = i

        # Find indices for each target topology
        result_indices = []

        for target_topo in target_topologies:
            # Validate topology containment
            mda_TopologyAdapter._validate_topology_containment(
                target_topo, universe, exclude_termini, termini_chain_selection, renumber_residues
            )

            chain_id = target_topo.chain
            active_residues = target_topo._get_active_residues(check_trim=False)
            lookup_key = (chain_id, frozenset(active_residues))

            if lookup_key not in mda_group_lookup:
                available_keys = list(mda_group_lookup.keys())
                raise ValueError(
                    f"Cannot find MDA group for topology {target_topo} "
                    f"(chain={chain_id}, resids={sorted(active_residues)}). "
                    f"Available groups: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}"
                )

            result_indices.append(mda_group_lookup[lookup_key])

        return result_indices

    @staticmethod
    def _build_renumbering_mapping(
        universe: mda.Universe,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
    ) -> dict[tuple[str, int], int]:
        """Build mapping from (chain, renumbered_resid) to original_resid."""
        renumber_mapping = {}

        # Apply the same selection logic as from_mda_universe
        try:
            selected_atoms = universe.select_atoms(termini_chain_selection)
        except Exception:
            # Fallback to all atoms if selection fails
            selected_atoms = universe.atoms

        if len(selected_atoms) == 0:
            return renumber_mapping

        # Group atoms by chain
        chains = {}
        for atom in selected_atoms:
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        for chain_id, chain_atoms in chains.items():
            # Get chain selection for terminal exclusion
            chain_selection_string, _ = mda_TopologyAdapter._build_chain_selection_string(
                universe, chain_id, termini_chain_selection
            )

            if chain_selection_string:
                try:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        chain_only_selection, _ = mda_TopologyAdapter._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                except:
                    termini_atoms = mda.AtomGroup(chain_atoms)
            else:
                termini_atoms = mda.AtomGroup(chain_atoms)

            # Sort by original resid
            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion BEFORE creating the renumbering mapping
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter chain atoms to only include residues that pass terminal exclusion
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            # Create renumbering mapping from the final filtered residues
            for new_resid, residue in enumerate(sorted_residues, 1):
                renumber_mapping[(chain_id, new_resid)] = residue.resid

        return renumber_mapping

    @staticmethod
    def _validate_topology_containment(
        topology: Partial_Topology,
        universe: mda.Universe,
        exclude_termini: bool = True,
        renumber_residues: bool = True,
    ) -> None:
        """Validate that topology residues are contained within chain bounds."""
        chain_id = topology.chain

        # Get chain selection
        chain_selection_string, _ = mda_TopologyAdapter._build_chain_selection_string(
            universe, chain_id
        )

        if chain_selection_string:
            try:
                chain_residues = universe.select_atoms(chain_selection_string).residues
            except:
                chain_residues = [
                    r
                    for r in universe.residues
                    if (getattr(r.atoms[0], "segid", None) or getattr(r.atoms[0], "chainid", "A"))
                    == chain_id
                ]
        else:
            chain_residues = [
                r
                for r in universe.residues
                if (getattr(r.atoms[0], "segid", None) or getattr(r.atoms[0], "chainid", "A"))
                == chain_id
            ]

        if not chain_residues:
            raise ValueError(f"No residues found for chain {chain_id}")

        # Sort and apply terminal exclusion
        full_chain_residues = sorted(chain_residues, key=lambda r: r.resid)
        if exclude_termini and len(full_chain_residues) > 2:
            available_residues = full_chain_residues[1:-1]
        else:
            available_residues = full_chain_residues

        if renumber_residues:
            # Check that topology residues fall within the renumbered range
            expected_range = list(range(1, len(available_residues) + 1))
            available_resids = set(expected_range)
        else:
            # Check that topology residues exist in the available residues
            available_resids = {r.resid for r in available_residues}

        # Check topology residues (use active residues based on check_trim)
        active_residues = topology._get_active_residues(check_trim=False)

        missing_residues = set(active_residues) - available_resids
        if missing_residues:
            raise ValueError(
                f"Topology {topology} contains residues {missing_residues} "
                f"that are not available in chain {chain_id}. "
                f"Available residues: {sorted(available_resids)}"
            )

    ### require some attention

    ### refactor to share same interface
    @staticmethod
    def from_mda_universe(
        universe: mda.Universe,
        mode: str = "residue",
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        fragment_name_template: str = "auto",
        renumber_residues: bool = False,
    ) -> list[Partial_Topology]:
        """Extract Partial_Topology objects from an MDAnalysis Universe

        Args:
            universe: MDAnalysis Universe object
            mode: Extraction mode - "by_chain" (one topology per chain) or
                "by_residue" (one topology per residue)
            include_selection: MDAnalysis selection string for atoms to include
            exclude_selection: Optional MDAnalysis selection string for atoms to exclude
            exclude_termini: If True, exclude N and C terminal residues from each chain
            termini_chain_selection: str
            fragment_name_template: Naming template - "auto" for automatic naming,
                                or custom template with {chain}, {resid}, {resname} placeholders
            renumber_residues: If True, renumber residues starting from 1 for each chain.
                            If False, preserve original residue numbers from MDAnalysis.

        Returns:
            List of Partial_Topology objects

        Raises:
            ValueError: If mode is not "by_chain" or "by_residue"
            ImportError: If MDAnalysis is not available
        """
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for this method. Install with: pip install MDAnalysis"
            )

        if mode not in ("chain", "residue"):
            raise ValueError("Mode must be either 'by_chain' or 'by_residue'")

        # Apply include selection
        try:
            selected_atoms = universe.select_atoms(include_selection)
        except Exception as e:
            raise ValueError(f"Invalid include selection '{include_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError(f"No atoms found with include selection '{include_selection}'")

        # Apply exclude selection if provided
        if exclude_selection:
            try:
                exclude_atoms = universe.select_atoms(exclude_selection)
                # Remove excluded atoms from selected atoms
                selected_atoms = selected_atoms - exclude_atoms
            except Exception as e:
                raise ValueError(f"Invalid exclude selection '{exclude_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError("No atoms remaining after applying exclude selection")

        # Group atoms by chain (try segid first, then chainid)
        chains = {}
        for atom in selected_atoms:
            # Prefer segid, fall back to chainid, then use 'A' as default
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        # Process each chain
        partial_topologies = []

        for chain_id, chain_atoms in chains.items():
            # Use utility method to build chain selection string
            chain_selection_string, fallback_atoms = (
                mda_TopologyAdapter._build_chain_selection_string(
                    universe, chain_id, termini_chain_selection
                )
            )

            if chain_selection_string:
                try:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        # Try without base selection
                        chain_only_selection, _ = mda_TopologyAdapter._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                except Exception:
                    # Fallback if selection fails
                    termini_atoms = mda.AtomGroup(chain_atoms)
            else:
                # If we couldn't build a selection string, use the chain atoms directly
                termini_atoms = mda.AtomGroup(chain_atoms)

            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion to determine which residues to include
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]  # Remove first and last
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter selected atoms to only include residues that pass terminal exclusion
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            if not sorted_residues:
                continue  # Skip if no residues left after filtering

            # Create renumbering mapping from the list of ALL included residues for the chain,
            # not just the ones in the current selection. This ensures consistent numbering.
            if renumber_residues:
                final_residue_mapping = {res.resid: i for i, res in enumerate(included_residues, 1)}
            else:
                # When not renumbering, the mapping is identity for all residues in the chain
                final_residue_mapping = {res.resid: res.resid for res in full_chain_residues}

            # Extract sequence information
            try:
                # Try to get single-letter amino acid codes
                sequence = []
                for residue in sorted_residues:
                    resname = residue.resname
                    # Convert common 3-letter to 1-letter codes
                    aa_map = {
                        "ALA": "A",
                        "ARG": "R",
                        "ASN": "N",
                        "ASP": "D",
                        "CYS": "C",
                        "GLN": "Q",
                        "GLU": "E",
                        "GLY": "G",
                        "HIS": "H",
                        "ILE": "I",
                        "LEU": "L",
                        "LYS": "K",
                        "MET": "M",
                        "PHE": "F",
                        "PRO": "P",
                        "SER": "S",
                        "THR": "T",
                        "TRP": "W",
                        "TYR": "Y",
                        "VAL": "V",
                        "MSE": "M",  # Selenomethionine
                    }
                    sequence.append(aa_map.get(resname, "X"))  # X for unknown
                sequence_str = "".join(sequence)
            except:
                # Fallback to residue names
                sequence_str = [res.resname for res in sorted_residues]

            if mode == "chain":
                # Create one topology per chain
                # Use the corrected mapping to get residue numbers
                residue_numbers = [final_residue_mapping[res.resid] for res in sorted_residues]

                # Generate fragment name
                if fragment_name_template == "auto":
                    fragment_name = f"chain_{chain_id}"
                else:
                    fragment_name = fragment_name_template.format(
                        chain=chain_id,
                        resid=f"{min(residue_numbers)}-{max(residue_numbers)}",
                        resname="chain",
                    )

                topology = TopologyFactory.from_residues(
                    chain=chain_id,
                    residues=residue_numbers,
                    fragment_sequence=sequence_str,
                    fragment_name=fragment_name,
                    peptide=False,
                )
                partial_topologies.append(topology)

            elif mode == "residue":
                # Create one topology per residue
                for i, residue in enumerate(sorted_residues):
                    # Use the corrected mapping to get the new resid
                    new_resid = final_residue_mapping[residue.resid]

                    # Get single residue sequence
                    if isinstance(sequence_str, str):
                        res_sequence = sequence_str[i]
                    else:
                        res_sequence = sequence_str[i]

                    # Generate fragment name
                    if fragment_name_template == "auto":
                        fragment_name = f"{chain_id}_{residue.resname}{new_resid}"
                    else:
                        fragment_name = fragment_name_template.format(
                            chain=chain_id, resid=new_resid, resname=residue.resname
                        )

                    topology = TopologyFactory.from_single(
                        chain=chain_id,
                        residue=new_resid,
                        fragment_sequence=res_sequence,
                        fragment_name=fragment_name,
                        peptide=False,
                    )
                    partial_topologies.append(topology)

        if not partial_topologies:
            raise ValueError("No partial topologies could be created from the selection")

        return partial_topologies

    @staticmethod
    def to_mda_group(
        topologies: Union[set[Partial_Topology], list[Partial_Topology]],
        universe: mda.Universe,
        include_selection: str = "protein",
        exclude_selection: Optional[str] = None,
        exclude_termini: bool = True,
        termini_chain_selection: str = "protein",
        renumber_residues: bool = False,
        mda_atom_filtering: Optional[str] = None,
    ) -> Union["mda.ResidueGroup", "mda.AtomGroup"]:
        """Create MDAnalysis ResidueGroup or AtomGroup from Partial_Topology objects."""
        try:
            import MDAnalysis as mda
        except ImportError:
            raise ImportError(
                "MDAnalysis is required for this method. Install with: pip install MDAnalysis"
            )

        if not topologies:
            raise ValueError("No topologies provided")

        # Convert to list if set
        if isinstance(topologies, set):
            topologies = list(topologies)

        # Apply include selection - SAME as from_mda_universe
        try:
            selected_atoms = universe.select_atoms(include_selection)
        except Exception as e:
            raise ValueError(f"Invalid include selection '{include_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError(f"No atoms found with include selection '{include_selection}'")

        # Apply exclude selection if provided - SAME as from_mda_universe
        if exclude_selection:
            try:
                exclude_atoms = universe.select_atoms(exclude_selection)
                selected_atoms = selected_atoms - exclude_atoms
            except Exception as e:
                raise ValueError(f"Invalid exclude selection '{exclude_selection}': {e}")

        if len(selected_atoms) == 0:
            raise ValueError("No atoms remaining after applying exclude selection")

        # Group atoms by chain - SAME as from_mda_universe
        chains = {}
        for atom in selected_atoms:
            chain_id = getattr(atom, "segid", None) or getattr(atom, "chainid", "A")
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(atom)

        # Group topologies by chain
        topologies_by_chain = {}
        for topo in topologies:
            if topo.chain not in topologies_by_chain:
                topologies_by_chain[topo.chain] = []
            topologies_by_chain[topo.chain].append(topo)

        # Create selection parts for each chain
        per_chain_selection_parts = []

        for chain_id, chain_topologies in topologies_by_chain.items():
            if chain_id not in chains:
                continue  # Skip if chain not found in universe

            # Get chain_atoms (from include/exclude selections) - SAME as from_mda_universe
            chain_atoms = chains[chain_id]

            # Use utility method to build chain selection string for terminal exclusion
            chain_selection_string, fallback_atoms = (
                mda_TopologyAdapter._build_chain_selection_string(
                    universe, chain_id, termini_chain_selection
                )
            )

            try:
                if chain_selection_string:
                    termini_atoms = universe.select_atoms(chain_selection_string)
                    if len(termini_atoms) == 0:
                        chain_only_selection, _ = mda_TopologyAdapter._build_chain_selection_string(
                            universe, chain_id
                        )
                        if chain_only_selection:
                            termini_atoms = universe.select_atoms(chain_only_selection)
                        else:
                            termini_atoms = mda.AtomGroup(chain_atoms)
                else:
                    termini_atoms = mda.AtomGroup(chain_atoms)
            except Exception as e:
                raise ValueError(f"Failed to select chain {chain_id}: {e}")

            full_chain_residues = sorted(termini_atoms.residues, key=lambda r: r.resid)

            # Apply terminal exclusion to determine which residues to include - SAME as from_mda_universe
            if exclude_termini and len(full_chain_residues) > 2:
                included_residues = full_chain_residues[1:-1]  # Remove first and last
            else:
                included_residues = full_chain_residues

            # Get the set of residue IDs that should be included after terminal exclusion
            included_resids = {res.resid for res in included_residues}

            # Filter chain atoms to only include residues that pass terminal exclusion - SAME as from_mda_universe
            chain_residues = {}
            for atom in chain_atoms:
                resid = atom.resid
                if resid in included_resids and resid not in chain_residues:
                    chain_residues[resid] = atom.residue

            # Sort residues by original resid - SAME as from_mda_universe
            sorted_residues = sorted(chain_residues.values(), key=lambda r: r.resid)

            if not sorted_residues:
                continue

            # Create renumbering mapping from the FINAL list of residues after all filtering - SAME as from_mda_universe
            if renumber_residues:
                renumber_to_original = {i: res.resid for i, res in enumerate(included_residues, 1)}
            else:
                renumber_to_original = {res.resid: res.resid for res in full_chain_residues}

            # Map topology residues to original residues
            chain_target_resids = []
            for topo in chain_topologies:
                for topo_resid in topo.residues:
                    original_resid = renumber_to_original.get(topo_resid)
                    if original_resid is not None:
                        chain_target_resids.append(original_resid)

            if not chain_target_resids:
                continue

            unique_resids = sorted(set(chain_target_resids))
            resid_selection = f"resid {' '.join(map(str, unique_resids))}"

            chain_sel_str, _ = mda_TopologyAdapter._build_chain_selection_string(universe, chain_id)
            if chain_sel_str:
                per_chain_selection_parts.append(f"(({chain_sel_str}) and ({resid_selection}))")
            else:
                per_chain_selection_parts.append(f"({resid_selection})")

        if not per_chain_selection_parts:
            raise ValueError("No matching residues found in universe")

        final_resid_selection = " or ".join(per_chain_selection_parts)

        # Select residues from universe
        try:
            # Combine with original selection to ensure consistency
            combined_selection = f"({include_selection}) and ({final_resid_selection})"
            if exclude_selection:
                combined_selection = f"({combined_selection}) and not ({exclude_selection})"

            target_atoms = universe.select_atoms(combined_selection)

            if len(target_atoms) == 0:
                raise ValueError("No atoms found matching the topology residues")

        except Exception as e:
            raise ValueError(f"Failed to select target atoms: {e}")

        # Apply atom filtering if requested
        if mda_atom_filtering:
            try:
                target_atoms = target_atoms.select_atoms(mda_atom_filtering)
                if len(target_atoms) == 0:
                    raise ValueError(f"No atoms found after applying filter '{mda_atom_filtering}'")
            except Exception as e:
                raise ValueError(f"Invalid atom filtering '{mda_atom_filtering}': {e}")

        # Return atoms or residues as requested
        if mda_atom_filtering:
            return target_atoms
        else:
            return target_atoms.residues

    ### refactor to share same interface
