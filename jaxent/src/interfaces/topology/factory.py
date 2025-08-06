from typing import Optional, Union

from jaxent.src.interfaces.topology.core import Partial_Topology
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons


class TopologyFactory:
    ### Operations for creating and merging topologies
    @staticmethod
    def from_range(
        chain: Union[str, int],
        start: int,
        end: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> Partial_Topology:
        """Create from a contiguous range of residues"""
        residues = list(range(int(start), int(end) + 1))
        return Partial_Topology(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @staticmethod
    def from_residues(
        chain: Union[str, int],
        residues: list[int],
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> Partial_Topology:
        """Create from an arbitrary list of residues"""
        return Partial_Topology(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @staticmethod
    def from_single(
        chain: Union[str, int],
        residue: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> Partial_Topology:
        """Create from a single residue"""
        return Partial_Topology(
            chain=chain,
            residues=[residue],
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @staticmethod
    def merge(
        topologies: list[Partial_Topology] | set[Partial_Topology],
        trim: bool = False,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
        intersection: bool = False,
    ) -> Partial_Topology:
        """Merge multiple topologies into a single topology

        Args:
            topologies: List of Partial_Topology objects to merge
            trim: If True, respect peptide trimming of input topologies.
                    If False, use all residues and create non-peptide result.
            merged_name: Name for the merged topology (default: concatenate names)
            merged_sequence: Sequence for merged topology (default: concatenate sequences)
            merged_index: Index for merged topology (default: use first topology's index)
            intersection: If True, only include residues that are present in all topologies

        Returns:
            New Partial_Topology containing merged residues

        Raises:
            ValueError: If topologies list is empty or topologies have different chains
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")
        topologies = list(topologies)  # Ensure we can iterate multiple times
        # Validate all topologies are on the same chain
        first_chain = topologies[0].chain
        for topo in topologies:
            if topo.chain != first_chain:
                raise ValueError(
                    f"All topologies must be on the same chain. "
                    f"Found chains: {set(t.chain for t in topologies)}"
                )

        # Process residues based on intersection or union and trim setting
        if intersection:
            # Start with first topology's residues for intersection
            all_residues = set(topologies[0]._get_active_residues(check_trim=trim))

            # Intersect with each subsequent topology
            for topo in topologies[1:]:
                residues = set(topo._get_active_residues(check_trim=trim))
                all_residues.intersection_update(residues)
        else:
            # Perform union of all residues (original behavior)
            all_residues = set()
            for topo in topologies:
                # Use check_trim=trim to respect peptide trimming when requested
                residues = topo._get_active_residues(check_trim=trim)
                all_residues.update(residues)

        # If intersection is empty, raise a more informative error
        if intersection and not all_residues:
            raise ValueError("No common residues found between all topologies")

        merged_residues = sorted(all_residues)

        # Handle merged metadata
        if merged_name is None:
            # Concatenate unique names
            names = [t.fragment_name for t in topologies if t.fragment_name != "seg"]
            if names:
                merged_name = "+".join(dict.fromkeys(names))  # Remove duplicates, preserve order
            else:
                merged_name = "merged"

        if merged_sequence is None:
            # Concatenate sequences if they're strings, otherwise use first non-empty
            sequences = [t.fragment_sequence for t in topologies if t.fragment_sequence]
            if sequences:
                if all(isinstance(seq, str) for seq in sequences):
                    merged_sequence = "".join(sequences)
                else:
                    # If mixed types or lists, use first sequence
                    merged_sequence = sequences[0]
            else:
                merged_sequence = ""

        if merged_index is None:
            # Use first topology's index that isn't None
            for topo in topologies:
                if topo.fragment_index is not None:
                    merged_index = topo.fragment_index
                    break

        # Determine peptide settings for merged result
        if trim:
            # If we respected trimming during merge, result can be a peptide
            # Use the minimum trim value from peptide inputs, or 0 if no peptides
            peptide_trims = [t.peptide_trim for t in topologies if t.peptide]
            merged_peptide = any(t.peptide for t in topologies)
            merged_peptide_trim = min(peptide_trims) if peptide_trims else 0
        else:
            # If trim=False, result is not a peptide
            merged_peptide = False
            merged_peptide_trim = 0

        return TopologyFactory.from_residues(
            chain=first_chain,
            residues=merged_residues,
            fragment_sequence=merged_sequence,
            fragment_name=merged_name,
            fragment_index=merged_index,
            peptide=merged_peptide,
            peptide_trim=merged_peptide_trim,
        )

    @staticmethod
    def merge_contiguous(
        topologies: list[Partial_Topology],
        trim: bool = False,
        gap_tolerance: int = 0,
        **kwargs,
    ) -> Partial_Topology:
        """Merge topologies that are contiguous or nearly contiguous

        Args:
            topologies: List of topologies to merge (will be sorted by residue position)
            trim: Whether to respect peptide trimming
            gap_tolerance: Maximum gap size to allow between topologies (0 = must be adjacent)
            **kwargs: Additional arguments passed to merge()

        Returns:
            Merged topology

        Raises:
            ValueError: If topologies are not contiguous within gap_tolerance
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")

        # Sort topologies by their start position
        sorted_topos = sorted(topologies, key=lambda t: min(t._get_active_residues(trim)))

        # Check contiguity
        for i in range(len(sorted_topos) - 1):
            current = sorted_topos[i]
            next_topo = sorted_topos[i + 1]

            gap = PairwiseTopologyComparisons.get_gap_to(current, next_topo, check_trim=trim)
            if gap is None:  # Overlapping is OK
                continue
            elif gap > gap_tolerance:
                current_residues = current._get_active_residues(trim)
                next_residues = next_topo._get_active_residues(trim)
                raise ValueError(
                    f"Gap of {gap} residues between topologies {current.fragment_name} "
                    f"(ends at {max(current_residues)}) and {next_topo.fragment_name} "
                    f"(starts at {min(next_residues)}) exceeds tolerance of {gap_tolerance}"
                )

        return TopologyFactory.merge(sorted_topos, trim=trim, **kwargs)

    @staticmethod
    def merge_overlapping(
        topologies: list[Partial_Topology],
        trim: bool = False,
        min_overlap: int = 1,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
    ) -> Partial_Topology:
        """Merge topologies that have overlapping residues

        Args:
            topologies: List of topologies to merge
            trim: Whether to respect peptide trimming
            min_overlap: Minimum number of overlapping residues required
            merged_name: Name for the merged topology (default: concatenate names)
            merged_sequence: Sequence for merged topology (default: concatenate sequences)
            merged_index: Index for merged topology (default: use first topology's index)

        Returns:
            Merged topology

        Raises:
            ValueError: If not all topologies overlap sufficiently
        """
        if not topologies:
            raise ValueError("Cannot merge empty list of topologies")

        if len(topologies) < 2:
            return TopologyFactory.merge(
                topologies,
                trim=trim,
                merged_name=merged_name,
                merged_sequence=merged_sequence,
                merged_index=merged_index,
            )

        # Check that all topologies have sufficient overlap with at least one other
        for i, topo1 in enumerate(topologies):
            has_sufficient_overlap = False
            for j, topo2 in enumerate(topologies):
                if i != j:
                    overlap = PairwiseTopologyComparisons.get_overlap(topo1, topo2, check_trim=trim)
                    if len(overlap) >= min_overlap:
                        has_sufficient_overlap = True
                        break

            if not has_sufficient_overlap:
                topo1_residues = topo1._get_active_residues(trim)
                raise ValueError(
                    f"Topology {topo1.fragment_name} (residues {min(topo1_residues)}-{max(topo1_residues)}) "
                    f"does not have at least {min_overlap} overlapping residues with any other topology"
                )

        return TopologyFactory.merge(
            topologies,
            trim=trim,
            merged_name=merged_name,
            merged_sequence=merged_sequence,
            merged_index=merged_index,
        )

    @staticmethod
    def extract_residues(
        topology: Partial_Topology, use_peptide_trim: bool = True
    ) -> list[Partial_Topology]:
        """Extract individual residues

        Args:
            use_peptide_trim: If True and this is a peptide, use peptide_residues.
                             If False, use all residues regardless of peptide settings.
        """
        if topology.length == 1:
            return [topology]

        if use_peptide_trim and topology.peptide and topology.peptide_residues:
            residues_to_extract = topology.peptide_residues
        else:
            residues_to_extract = topology.residues

        return [
            TopologyFactory.from_single(
                chain=topology.chain,
                residue=res,
                fragment_sequence=topology.fragment_sequence,
                fragment_name=topology.fragment_name,
                fragment_index=topology.fragment_index,
                peptide=False,  # Individual residues are not peptides
                peptide_trim=topology.peptide_trim,
            )
            for res in residues_to_extract
        ]

    ###
    @staticmethod
    def remove_residues_by_topologies(
        topology: Partial_Topology,
        topologies_to_remove: list[Partial_Topology],
        check_trim: bool = False,
    ) -> Partial_Topology:
        """Remove residues from this topology based on other topologies

        Args:
            topologies_to_remove: List of Partial_Topology objects whose residues
                              should be removed from this topology

        Returns:
            A new Partial_Topology with the specified residues removed

        Raises:
            ValueError: If any topology has a different chain than this one
            ValueError: If no residues would remain after removal
        """
        # Verify all topologies are on the same chain
        for topo in topologies_to_remove:
            if topo.chain != topology.chain:
                raise ValueError(
                    f"Cannot remove residues from different chain: {topo.chain} != {topology.chain}"
                )

        # Collect all residues to remove
        residues_to_remove = set()
        for topo in topologies_to_remove:
            residues_to_remove.update(topo.residues)

        # Create new residue list with specified residues removed
        new_residues = [
            res
            for res in topology._get_active_residues(check_trim)
            if res not in residues_to_remove
        ]

        # If no residues left, raise error
        if not new_residues:
            raise ValueError("No residues remaining after removal")

        # Create new topology with remaining residues
        return TopologyFactory.from_residues(
            chain=topology.chain,
            residues=new_residues,
            fragment_sequence=topology.fragment_sequence,
            fragment_name=topology.fragment_name,
            fragment_index=topology.fragment_index,
            peptide=topology.peptide,
            peptide_trim=topology.peptide_trim,
        )

    @staticmethod
    def union(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> Partial_Topology:
        """
        Combine two topologies into a single topology containing all residues
        """
        if top.chain != other.chain:
            raise ValueError("Cannot combine topologies with different chains")

        top_active = top._get_active_residues(check_trim)
        other_active = other._get_active_residues(check_trim)

        all_active = sorted(set(top_active) | set(other_active))
        min_res, max_res = min(all_active), max(all_active)

        combined_residues = list(range(min_res, max_res + 1))

        return TopologyFactory.from_residues(
            chain=top.chain,
            residues=combined_residues,
            fragment_sequence=top.fragment_sequence,
            fragment_name=top.fragment_name,
            fragment_index=top.fragment_index,
            peptide=top.peptide,
            peptide_trim=top.peptide_trim,
        )
