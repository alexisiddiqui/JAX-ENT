
    ### Pairwise Comparisons
    def intersects(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology has any residues in common with another

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return bool(set(self_residues) & set(other_residues))

    def contains_topology(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology fully contains another topology

        Args:
            other: The other topology to check
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(other_residues).issubset(set(self_residues))

    def is_subset_of(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is a subset of another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(self_residues).issubset(set(other_residues))

    def is_superset_of(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is a superset of another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        return other.is_subset_of(self, check_trim)

    def get_overlap(self, other: "Partial_Topology", check_trim: bool = False) -> list[int]:
        """Get the residues that overlap between this and another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return []

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(self_residues) & set(other_residues))

    def get_difference(self, other: "Partial_Topology", check_trim: bool = False) -> list[int]:
        """Get residues in this topology but not in the other

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return self._get_active_residues(check_trim).copy()

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(self_residues) - set(other_residues))

    def is_adjacent_to(self, other: "Partial_Topology", check_trim: bool = False) -> bool:
        """Check if this topology is adjacent to another (residues are consecutive)

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides
        """
        if self.chain != other.chain:
            return False

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        self_end = max(self_residues)
        self_start = min(self_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        return self_end + 1 == other_start or other_end + 1 == self_start

    def get_gap_to(self, other: "Partial_Topology", check_trim: bool = False) -> Optional[int]:
        """Get the gap size between this and another topology

        Args:
            other: The other topology to compare with
            check_trim: If True, use peptide_residues for peptides

        Returns:
            None if topologies overlap or are on different chains
            0 if topologies are adjacent
            Positive integer for the gap size
        """
        if self.chain != other.chain:
            return None

        if self.intersects(other, check_trim):
            return None  # They overlap

        self_residues = self._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        self_end = max(self_residues)
        self_start = min(self_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        if self_end < other_start:
            gap = other_start - self_end - 1
        elif other_end < self_start:
            gap = self_start - other_end - 1
        else:
            return None  # Shouldn't reach here if logic is correct

        return gap

    def union(self, other: "Partial_Topology", check_trim: bool = False) -> "Partial_Topology":
        """Combine this topology with another (union of residues)

        Args:
            other: The other topology to combine with
            check_trim: If True, use peptide_residues for peptides when determining union
        """
        if self.chain != other.chain:
            raise ValueError("Cannot combine topologies with different chains")

        # For union, we need to be careful about which residues to combine
        # If check_trim is True and either topology is a peptide, we use their active residues
        # But the result should contain the full residue ranges that encompass both active regions

        self_active = self._get_active_residues(check_trim)
        other_active = other._get_active_residues(check_trim)

        # Determine the full range needed to encompass both active regions
        all_active = sorted(set(self_active) | set(other_active))
        min_res, max_res = min(all_active), max(all_active)

        # Create the union with the full range
        combined_residues = list(range(min_res, max_res + 1))

        return Partial_Topology.from_residues(
            chain=self.chain,
            residues=combined_residues,
            fragment_sequence=self.fragment_sequence,
            fragment_name=self.fragment_name,
            fragment_index=self.fragment_index,
            peptide=self.peptide,
            peptide_trim=self.peptide_trim,
        )
