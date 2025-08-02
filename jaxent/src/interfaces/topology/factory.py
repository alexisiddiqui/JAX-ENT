
    ### Operations for creating and merging topologies
    @classmethod
    def from_range(
        cls,
        chain: Union[str, int],
        start: int,
        end: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> "Partial_Topology":
        """Create from a contiguous range of residues"""
        residues = list(range(int(start), int(end) + 1))
        return cls(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @classmethod
    def from_residues(
        cls,
        chain: Union[str, int],
        residues: list[int],
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> "Partial_Topology":
        """Create from an arbitrary list of residues"""
        return cls(
            chain=chain,
            residues=residues,
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @classmethod
    def from_single(
        cls,
        chain: Union[str, int],
        residue: int,
        fragment_sequence: Union[str, list[str]] = "",
        fragment_name: str = "seg",
        fragment_index: Optional[int] = None,
        peptide: bool = False,
        peptide_trim: int = 2,
    ) -> "Partial_Topology":
        """Create from a single residue"""
        return cls(
            chain=chain,
            residues=[residue],
            fragment_sequence=fragment_sequence,
            fragment_name=fragment_name,
            fragment_index=fragment_index,
            peptide=peptide,
            peptide_trim=peptide_trim,
        )

    @classmethod
    def merge(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
        intersection: bool = False,
    ) -> "Partial_Topology":
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

        return cls.from_residues(
            chain=first_chain,
            residues=merged_residues,
            fragment_sequence=merged_sequence,
            fragment_name=merged_name,
            fragment_index=merged_index,
            peptide=merged_peptide,
            peptide_trim=merged_peptide_trim,
        )

    @classmethod
    def merge_contiguous(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        gap_tolerance: int = 0,
        **kwargs,
    ) -> "Partial_Topology":
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

            gap = current.get_gap_to(next_topo, check_trim=trim)
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

        return cls.merge(sorted_topos, trim=trim, **kwargs)

    @classmethod
    def merge_overlapping(
        cls,
        topologies: list["Partial_Topology"],
        trim: bool = False,
        min_overlap: int = 1,
        merged_name: Optional[str] = None,
        merged_sequence: Optional[Union[str, list[str]]] = None,
        merged_index: Optional[int] = None,
    ) -> "Partial_Topology":
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
            return cls.merge(
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
                    overlap = topo1.get_overlap(topo2, check_trim=trim)
                    if len(overlap) >= min_overlap:
                        has_sufficient_overlap = True
                        break

            if not has_sufficient_overlap:
                topo1_residues = topo1._get_active_residues(trim)
                raise ValueError(
                    f"Topology {topo1.fragment_name} (residues {min(topo1_residues)}-{max(topo1_residues)}) "
                    f"does not have at least {min_overlap} overlapping residues with any other topology"
                )

        return cls.merge(
            topologies,
            trim=trim,
            merged_name=merged_name,
            merged_sequence=merged_sequence,
            merged_index=merged_index,
        )
    def extract_residues(self, use_peptide_trim: bool = True) -> list["Partial_Topology"]:
        """Extract individual residues

        Args:
            use_peptide_trim: If True and this is a peptide, use peptide_residues.
                             If False, use all residues regardless of peptide settings.
        """
        if self.length == 1:
            return [self]

        if use_peptide_trim and self.peptide and self.peptide_residues:
            residues_to_extract = self.peptide_residues
        else:
            residues_to_extract = self.residues

        return [
            Partial_Topology.from_single(
                chain=self.chain,
                residue=res,
                fragment_sequence=self.fragment_sequence,
                fragment_name=self.fragment_name,
                fragment_index=self.fragment_index,
                peptide=False,  # Individual residues are not peptides
                peptide_trim=self.peptide_trim,
            )
            for res in residues_to_extract
        ]

    ###

    def remove_residues_by_topologies(
        self, topologies_to_remove: list["Partial_Topology"]
    ) -> "Partial_Topology":
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
            if topo.chain != self.chain:
                raise ValueError(
                    f"Cannot remove residues from different chain: {topo.chain} != {self.chain}"
                )

        # Collect all residues to remove
        residues_to_remove = set()
        for topo in topologies_to_remove:
            residues_to_remove.update(topo.residues)

        # Create new residue list with specified residues removed
        new_residues = [res for res in self.residues if res not in residues_to_remove]

        # If no residues left, raise error
        if not new_residues:
            raise ValueError("No residues remaining after removal")

        # Create new topology with remaining residues
        return Partial_Topology.from_residues(
            chain=self.chain,
            residues=new_residues,
            fragment_sequence=self.fragment_sequence,
            fragment_name=self.fragment_name,
            fragment_index=self.fragment_index,
            peptide=self.peptide,
            peptide_trim=self.peptide_trim,
        )
