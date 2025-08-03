from typing import Optional

from jaxent.src.interfaces.topology.core import Partial_Topology


class PairwiseTopologyComparisons:
    ### Pairwise Comparisons

    @staticmethod
    def intersects(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> bool:
        if top.chain != other.chain:
            return False

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return bool(set(top_residues) & set(other_residues))

    @staticmethod
    def contains_topology(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> bool:
        if top.chain != other.chain:
            return False

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(other_residues).issubset(set(top_residues))

    @staticmethod
    def is_subset_of(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> bool:
        if top.chain != other.chain:
            return False

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return set(top_residues).issubset(set(other_residues))

    @staticmethod
    def is_superset_of(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> bool:
        return PairwiseTopologyComparisons.is_subset_of(other, top, check_trim)

    @staticmethod
    def get_overlap(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> list[int]:
        if top.chain != other.chain:
            return []

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(top_residues) & set(other_residues))

    @staticmethod
    def get_difference(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> list[int]:
        if top.chain != other.chain:
            return top._get_active_residues(check_trim).copy()

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)
        return sorted(set(top_residues) - set(other_residues))

    @staticmethod
    def is_adjacent_to(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> bool:
        if top.chain != other.chain:
            return False

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        top_end = max(top_residues)
        top_start = min(top_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        return top_end + 1 == other_start or other_end + 1 == top_start

    @staticmethod
    def get_gap_to(
        top: Partial_Topology, other: Partial_Topology, check_trim: bool = False
    ) -> Optional[int]:
        if top.chain != other.chain:
            return None

        if PairwiseTopologyComparisons.intersects(top, other, check_trim):
            return None  # They overlap

        top_residues = top._get_active_residues(check_trim)
        other_residues = other._get_active_residues(check_trim)

        top_end = max(top_residues)
        top_start = min(top_residues)
        other_end = max(other_residues)
        other_start = min(other_residues)

        if top_end < other_start:
            gap = other_start - top_end - 1
        elif other_end < top_start:
            gap = top_start - other_end - 1
        else:
            return None  # Shouldn't reach here if logic is correct

        return gap
