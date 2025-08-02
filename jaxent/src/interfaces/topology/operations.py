    ### Basic Operations
    @classmethod
    def rank_and_index(
        cls, topologies: list["Partial_Topology"], check_trim: bool = False
    ) -> list["Partial_Topology"]:
        """
        Sorts a list of topologies and assigns fragment_index.

        The list is sorted based on the key from the `rank_order` method.
        The sort is stable, so equally ranked objects retain their
        relative order from the input list. After sorting, the
        `fragment_index` of each topology is updated to its
        position in the sorted list.

        Args:
            topologies: A list of Partial_Topology objects to sort and index.
            check_trim: If True, use peptide_residues for ranking.

        Returns:
            The sorted and indexed list of topologies.
        """
        # Sort the list using the rank_order method as the key.
        # Python's sort is stable.
        topologies.sort(key=lambda t: t.rank_order(check_trim=check_trim))

        # Assign fragment_index based on the new order
        for i, topo in enumerate(topologies):
            topo.fragment_index = i

        return topologies

    @classmethod
    def calculate_fragment_redundancy(
        cls,
        topologies: list["Partial_Topology"],
        mode: str = "mean",
        check_trim: bool = False,
    ) -> list[float]:
        """Calculate overlap redundancy between fragments in a list

        Args:
            topologies: List of Partial_Topology objects to compare
            mode: Either "max" or "mean" to determine how to calculate overlap scores
            check_trim: If True, use peptide_residues for peptides; if False, use all residues

        Returns:
            List of overlap scores for each fragment (same order as input)

        Raises:
            ValueError: If mode is not "max" or "mean"
            ValueError: If topologies list is empty
        """
        if not topologies:
            raise ValueError("Cannot calculate redundancy for empty topology list")

        if mode not in ("max", "mean"):
            raise ValueError("Mode must be either 'max' or 'mean'")

        # Group topologies by chain for more efficient comparison
        topology_set = set(topologies)
        grouped_by_chain = cls.group_set_by_chain(topology_set)

        redundancy_scores = []

        for i, fragment in enumerate(topologies):
            # Get active residues for current fragment
            current_residues = set(fragment._get_active_residues(check_trim))

            # Only compare with fragments from the same chain
            same_chain_fragments = grouped_by_chain.get(fragment.chain, set())
            fragment_overlaps = []

            # Compare with other fragments in the same chain
            for other in same_chain_fragments:
                if other is not fragment:  # Skip self-comparison
                    # Get active residues for other fragment
                    other_residues = set(other._get_active_residues(check_trim))

                    # Calculate overlap
                    overlap_size = len(current_residues & other_residues)
                    if overlap_size > 0:
                        fragment_overlaps.append(overlap_size)

            # Calculate final redundancy score based on mode
            if fragment_overlaps:
                if mode == "max":
                    redundancy_scores.append(max(fragment_overlaps))
                elif mode == "mean":
                    redundancy_scores.append(sum(fragment_overlaps) / len(fragment_overlaps))
            else:
                redundancy_scores.append(0.0)

        return redundancy_scores

    @classmethod
    def group_set_by_chain(
        cls, topologies: set["Partial_Topology"]
    ) -> dict[Union[str, int], set["Partial_Topology"]]:
        """Group a set of Partial_Topology objects by chain.

        Args:
            topologies: Set of Partial_Topology objects to group

        Returns:
            Dictionary mapping chain ID to sets of Partial_Topology objects for that chain
        """
        result = {}
        for topo in topologies:
            if topo.chain not in result:
                result[topo.chain] = set()
            result[topo.chain].add(topo)
        return result
