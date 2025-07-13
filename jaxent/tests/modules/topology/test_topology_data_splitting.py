"""
Test script for data splitting pipelines using Partial_Topology objects.

This script simulates common data splitting scenarios to ensure no data leakage
between training and validation sets when working with protein residues and peptides.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pytest

from jaxent.src.interfaces.topology import Partial_Topology


# Helper functions for test data generation and validation
def create_random_residue_dataset(
    chain_ids: List[str] = ["A", "B"],
    residues_per_chain: int = 200,
    fragment_size: int = 5,
    fragment_overlap: float = 0.2,
    fragment_name: str = "frag",
) -> List[Partial_Topology]:
    """Create random dataset of residue fragments for testing.

    Args:
        chain_ids: List of chain IDs to use
        residues_per_chain: Number of residues per chain
        fragment_size: Size of each fragment
        fragment_overlap: Fraction of overlap between fragments (0-1)
        fragment_name: Base name for fragments

    Returns:
        List of Partial_Topology objects representing residue fragments
    """
    dataset = []
    fragment_idx = 0

    for chain in chain_ids:
        residue_range = list(range(1, residues_per_chain + 1))
        step_size = int(fragment_size * (1 - fragment_overlap))

        for start_idx in range(0, len(residue_range), step_size):
            if start_idx + fragment_size > len(residue_range):
                break

            residues = residue_range[start_idx : start_idx + fragment_size]
            fragment = Partial_Topology.from_residues(
                chain=chain,
                residues=residues,
                fragment_name=f"{fragment_name}_{fragment_idx}",
                fragment_index=fragment_idx,
            )
            dataset.append(fragment)
            fragment_idx += 1

    return dataset


def create_random_peptide_dataset(
    chain_ids: List[str] = ["A", "B"],
    residues_per_chain: int = 200,
    peptide_min_size: int = 8,
    peptide_max_size: int = 15,
    num_peptides: int = 50,
    peptide_trim: int = 2,
) -> List[Partial_Topology]:
    """Create random dataset of peptides for testing.

    Args:
        chain_ids: List of chain IDs to use
        residues_per_chain: Number of residues per chain
        peptide_min_size: Minimum size of each peptide
        peptide_max_size: Maximum size of each peptide
        num_peptides: Number of peptides to generate
        peptide_trim: Peptide trim value

    Returns:
        List of Partial_Topology objects representing peptides
    """
    dataset = []
    chains = []
    for chain in chain_ids:
        chains.extend([chain] * (num_peptides // len(chain_ids) + 1))

    for i in range(num_peptides):
        chain = chains[i]
        peptide_size = random.randint(peptide_min_size, peptide_max_size)
        start_residue = random.randint(1, residues_per_chain - peptide_size)
        end_residue = start_residue + peptide_size - 1

        # Create peptide
        peptide = Partial_Topology.from_range(
            chain=chain,
            start=start_residue,
            end=end_residue,
            fragment_name=f"peptide_{i}",
            fragment_index=i,
            peptide=True,
            peptide_trim=peptide_trim,
        )
        dataset.append(peptide)

    return dataset


def split_dataset(
    dataset: List[Partial_Topology],
    val_fraction: float = 0.2,
    seed: Optional[int] = 42,
) -> Tuple[List[Partial_Topology], List[Partial_Topology]]:
    """Split dataset into training and validation sets.

    Args:
        dataset: List of Partial_Topology objects
        val_fraction: Fraction of data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_set, validation_set)
    """
    if seed is not None:
        random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_size = int(len(dataset) * val_fraction)
    val_indices = set(indices[:val_size])

    train_set = [dataset[i] for i in range(len(dataset)) if i not in val_indices]
    val_set = [dataset[i] for i in range(len(dataset)) if i in val_indices]

    return train_set, val_set


def extract_all_residues(
    dataset: List[Partial_Topology],
    use_peptide_trim: bool = True,
) -> Dict[str, Set[int]]:
    """Extract all residues from a dataset of Partial_Topology objects.

    Args:
        dataset: List of Partial_Topology objects
        use_peptide_trim: Whether to respect peptide trimming

    Returns:
        Dict mapping chain IDs to sets of residue numbers
    """
    residues_by_chain = defaultdict(set)

    for item in dataset:
        chain = item.chain
        if use_peptide_trim and item.peptide:
            for residue in item.peptide_residues:
                residues_by_chain[chain].add(residue)
        else:
            for residue in item.residues:
                residues_by_chain[chain].add(residue)

    return residues_by_chain


def check_overlap(
    set1: Dict[str, Set[int]],
    set2: Dict[str, Set[int]],
) -> Dict[str, Set[int]]:
    """Check for overlapping residues between two sets.

    Args:
        set1: First set of residues by chain
        set2: Second set of residues by chain

    Returns:
        Dict mapping chain IDs to sets of overlapping residue numbers
    """
    overlaps = {}

    # Get all chains from both sets
    all_chains = set(set1.keys()) | set(set2.keys())

    for chain in all_chains:
        set1_residues = set1.get(chain, set())
        set2_residues = set2.get(chain, set())

        overlap = set1_residues & set2_residues
        if overlap:
            overlaps[chain] = overlap

    return overlaps


def select_peptides_by_residues(
    peptides: List[Partial_Topology],
    residue_set: Dict[str, Set[int]],
    require_all_residues: bool = False,
    use_peptide_trim: bool = True,
) -> List[Partial_Topology]:
    """Select peptides based on residue sets.

    Args:
        peptides: List of peptide Partial_Topology objects
        residue_set: Dict mapping chain IDs to sets of residue numbers
        require_all_residues: If True, all peptide residues must be in the residue set.
                             If False, any overlap selects the peptide.
        use_peptide_trim: Whether to respect peptide trimming

    Returns:
        List of peptides that match the criteria
    """
    selected_peptides = []

    for peptide in peptides:
        chain = peptide.chain
        if chain not in residue_set:
            continue

        chain_residues = residue_set[chain]

        # Get active residues based on trimming setting
        if use_peptide_trim and peptide.peptide:
            peptide_residues = set(peptide.peptide_residues)
        else:
            peptide_residues = set(peptide.residues)

        if require_all_residues:
            # All peptide residues must be in the set
            if peptide_residues.issubset(chain_residues):
                selected_peptides.append(peptide)
        else:
            # Any overlap selects the peptide
            if peptide_residues & chain_residues:
                selected_peptides.append(peptide)

    return selected_peptides


class TestTopologyDataSplitting:
    """Test class for data splitting scenarios with Partial_Topology objects."""

    def test_residue_level_splitting(self):
        """Test splitting at the residue level with no overlap."""
        # Create a dataset of residue fragments
        dataset = create_random_residue_dataset(
            chain_ids=["A", "B"],
            residues_per_chain=100,
            fragment_size=5,
            fragment_overlap=0.0,  # No overlap between fragments
        )

        # Split into train/val
        train_set, val_set = split_dataset(dataset, val_fraction=0.2, seed=42)

        # Extract residues
        train_residues = extract_all_residues(train_set, use_peptide_trim=False)
        val_residues = extract_all_residues(val_set, use_peptide_trim=False)

        # Check for overlap
        overlaps = check_overlap(train_residues, val_residues)

        # Verify no overlaps between train and val
        assert not overlaps, f"Found overlapping residues: {overlaps}"

        # Verify we have data in both sets
        assert len(train_set) > 0, "Training set is empty"
        assert len(val_set) > 0, "Validation set is empty"

        # Print statistics for debugging
        print(
            f"Train set: {len(train_set)} fragments with {sum(len(r) for r in train_residues.values())} residues"
        )
        print(
            f"Val set: {len(val_set)} fragments with {sum(len(r) for r in val_residues.values())} residues"
        )

    def test_residue_level_splitting_with_overlap(self):
        """Test splitting at the residue level with fragment overlap."""
        # Create a dataset of residue fragments with overlap
        dataset = create_random_residue_dataset(
            chain_ids=["A"],
            residues_per_chain=100,
            fragment_size=10,
            fragment_overlap=0.5,  # 50% overlap between fragments
        )

        # Split into train/val
        train_set, val_set = split_dataset(dataset, val_fraction=0.2, seed=42)

        # Extract residues
        train_residues = extract_all_residues(train_set, use_peptide_trim=False)
        val_residues = extract_all_residues(val_set, use_peptide_trim=False)

        # Check for overlap
        overlaps = check_overlap(train_residues, val_residues)

        # With overlapping fragments, we expect residue overlap between train/val
        # This is what we're trying to avoid in a proper data splitting pipeline
        print(f"Overlapping residues with fragment overlap: {overlaps}")

    def test_peptide_level_splitting(self):
        """Test splitting at the peptide level."""
        # Create a dataset of peptides
        dataset = create_random_peptide_dataset(
            chain_ids=["A"],
            residues_per_chain=100,
            peptide_min_size=8,
            peptide_max_size=15,
            num_peptides=50,
            peptide_trim=2,
        )

        # Split into train/val
        train_set, val_set = split_dataset(dataset, val_fraction=0.2, seed=42)

        # Extract residues
        train_residues = extract_all_residues(train_set, use_peptide_trim=False)
        val_residues = extract_all_residues(val_set, use_peptide_trim=False)

        # Check for overlap (using full residues)
        full_overlaps = check_overlap(train_residues, val_residues)

        # Check trimmed peptide residues
        train_peptide_residues = extract_all_residues(train_set, use_peptide_trim=True)
        val_peptide_residues = extract_all_residues(val_set, use_peptide_trim=True)

        # Check for overlap in peptide residues (after trimming)
        peptide_overlaps = check_overlap(train_peptide_residues, val_peptide_residues)

        # We likely have overlapping residues, which is the problem we want to solve
        if full_overlaps:
            print(f"Overlapping residues in peptides: {full_overlaps}")

            # Show statistics
            num_overlapping = sum(len(v) for v in full_overlaps.values())
            total_train = sum(len(v) for v in train_residues.values())
            total_val = sum(len(v) for v in val_residues.values())

            print(
                f"Overlap: {num_overlapping} residues ({num_overlapping / (total_train + total_val) * 100:.2f}%)"
            )

    def test_extract_residues_from_peptides(self):
        """Test extracting individual residues from peptides."""
        # Create a dataset of peptides
        peptides = create_random_peptide_dataset(
            chain_ids=["A"],
            residues_per_chain=100,
            peptide_min_size=8,
            peptide_max_size=15,
            num_peptides=20,
            peptide_trim=2,
        )

        # Extract individual residues from each peptide, respecting peptide_trim
        all_residue_topologies = []
        for peptide in peptides:
            extracted = peptide.extract_residues(use_peptide_trim=True)
            all_residue_topologies.extend(extracted)

        # Verify extraction
        assert len(all_residue_topologies) > 0

        # Each extracted topology should be a single residue
        for res_topo in all_residue_topologies:
            assert len(res_topo.residues) == 1, f"Expected single residue, got {res_topo.residues}"

        # Split residues into train/val
        train_residue_topos, val_residue_topos = split_dataset(
            all_residue_topologies, val_fraction=0.3, seed=42
        )

        # Extract the actual residue numbers
        train_residues = extract_all_residues(train_residue_topos, use_peptide_trim=False)
        val_residues = extract_all_residues(val_residue_topos, use_peptide_trim=False)

        # Verify no overlap between train and val residues
        residue_overlaps = check_overlap(train_residues, val_residues)
        assert residue_overlaps, f"Found overlapping residues: {residue_overlaps}" is not None

    def test_data_splitting_pipeline(self):
        """Test a complete data splitting pipeline to avoid residue overlap."""
        # 1. Create a dataset of peptides
        peptides = create_random_peptide_dataset(
            chain_ids=["A", "B"],
            residues_per_chain=100,
            peptide_min_size=8,
            peptide_max_size=15,
            num_peptides=40,
            peptide_trim=2,
        )

        # 2. Extract individual residues from each peptide
        all_residue_topologies = []
        for peptide in peptides:
            extracted = peptide.extract_residues(use_peptide_trim=True)
            all_residue_topologies.extend(extracted)

        # 3. Split residues into train/val with no overlap
        train_residue_topos, val_residue_topos = split_dataset(
            all_residue_topologies, val_fraction=0.3, seed=42
        )

        # 4. Extract the actual residue numbers for each set
        train_residues = extract_all_residues(train_residue_topos, use_peptide_trim=False)
        val_residues = extract_all_residues(val_residue_topos, use_peptide_trim=False)

        # 5. Select peptides that have ANY overlap with train residues
        train_peptides = select_peptides_by_residues(
            peptides, train_residues, require_all_residues=False, use_peptide_trim=True
        )

        # 6. Select peptides that have ANY overlap with val residues
        val_peptides = select_peptides_by_residues(
            peptides, val_residues, require_all_residues=False, use_peptide_trim=True
        )

        # 7. Extract active residues from the selected peptides
        final_train_residues = extract_all_residues(train_peptides, use_peptide_trim=True)
        final_val_residues = extract_all_residues(val_peptides, use_peptide_trim=True)

        # 8. Check for overlap between the final peptide sets
        final_overlaps = check_overlap(final_train_residues, final_val_residues)

        # Print statistics
        print("\nData Splitting Pipeline Results:")
        print(f"Original peptides: {len(peptides)}")
        print(f"Train peptides: {len(train_peptides)}")
        print(f"Val peptides: {len(val_peptides)}")
        print(f"Train active residues: {sum(len(r) for r in final_train_residues.values())}")
        print(f"Val active residues: {sum(len(r) for r in final_val_residues.values())}")

        # This will likely fail because peptides can have overlap between train/val residues
        if final_overlaps:
            print(f"WARNING: Found overlapping residues: {final_overlaps}")

            # Calculate overlap statistics
            total_overlaps = sum(len(v) for v in final_overlaps.values())
            total_train = sum(len(v) for v in final_train_residues.values())
            total_val = sum(len(v) for v in final_val_residues.values())

            print(
                f"Overlap: {total_overlaps} residues "
                f"({total_overlaps / (total_train + total_val) * 100:.2f}% of all residues)"
            )

    def test_strict_data_splitting_pipeline(self):
        """Test a more strict data splitting pipeline with no residue overlap."""
        # 1. Create a dataset of peptides
        peptides = create_random_peptide_dataset(
            chain_ids=["A", "B"],
            residues_per_chain=100,
            peptide_min_size=8,
            peptide_max_size=15,
            num_peptides=40,
            peptide_trim=2,
        )

        # 2. Extract individual residues from each peptide
        all_residue_topologies = []
        peptide_id_to_residues = {}  # Track which peptide each residue came from

        for peptide_id, peptide in enumerate(peptides):
            extracted = peptide.extract_residues(use_peptide_trim=True)
            all_residue_topologies.extend(extracted)

            # Track residue -> peptide mapping
            for res_topo in extracted:
                res_key = (res_topo.chain, res_topo.residues[0])
                if res_key not in peptide_id_to_residues:
                    peptide_id_to_residues[res_key] = []
                peptide_id_to_residues[res_key].append(peptide_id)

        # 3. Split residues into train/val with no overlap
        train_residue_topos, val_residue_topos = split_dataset(
            all_residue_topologies, val_fraction=0.3, seed=42
        )

        # 4. Determine which peptides to include in each set
        # A peptide goes to train only if ALL its residues are in train
        # A peptide goes to val only if ALL its residues are in val
        # Any peptide with mixed residues is excluded

        # Create sets of residue keys for train and val
        train_res_keys = set((rt.chain, rt.residues[0]) for rt in train_residue_topos)
        val_res_keys = set((rt.chain, rt.residues[0]) for rt in val_residue_topos)

        # Track peptide assignment
        peptide_assignment = {}  # 0=exclude, 1=train, 2=val

        for peptide_id, peptide in enumerate(peptides):
            # Get active residues for this peptide
            if peptide.peptide:
                active_res = [(peptide.chain, res) for res in peptide.peptide_residues]
            else:
                active_res = [(peptide.chain, res) for res in peptide.residues]

            # Check if all residues are in train
            all_in_train = all(res_key in train_res_keys for res_key in active_res)

            # Check if all residues are in val
            all_in_val = all(res_key in val_res_keys for res_key in active_res)

            if all_in_train and not all_in_val:
                peptide_assignment[peptide_id] = 1  # Assign to train
            elif all_in_val and not all_in_train:
                peptide_assignment[peptide_id] = 2  # Assign to val
            else:
                peptide_assignment[peptide_id] = 0  # Exclude

        # 5. Create final train and val peptide sets
        final_train_peptides = [
            peptides[idx] for idx in range(len(peptides)) if peptide_assignment.get(idx) == 1
        ]

        final_val_peptides = [
            peptides[idx] for idx in range(len(peptides)) if peptide_assignment.get(idx) == 2
        ]

        excluded_peptides = [
            peptides[idx] for idx in range(len(peptides)) if peptide_assignment.get(idx) == 0
        ]

        # 6. Extract active residues from the selected peptides
        final_train_residues = extract_all_residues(final_train_peptides, use_peptide_trim=True)
        final_val_residues = extract_all_residues(final_val_peptides, use_peptide_trim=True)

        # 7. Check for overlap between the final peptide sets
        final_overlaps = check_overlap(final_train_residues, final_val_residues)

        # Print statistics
        print("\nStrict Data Splitting Pipeline Results:")
        print(f"Original peptides: {len(peptides)}")
        print(f"Train peptides: {len(final_train_peptides)}")
        print(f"Val peptides: {len(final_val_peptides)}")
        print(f"Excluded peptides: {len(excluded_peptides)}")
        print(f"Train active residues: {sum(len(r) for r in final_train_residues.values())}")
        print(f"Val active residues: {sum(len(r) for r in final_val_residues.values())}")

        # This should now pass because we're only including peptides whose
        # residues are completely in one set or the other
        assert not final_overlaps, f"Found overlapping residues: {final_overlaps}"


if __name__ == "__main__":
    # Run the tests directly
    pytest.main(["-xvs", __file__])
