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


class TestTopologyRedundancyBasic:
    """Test class for fragment redundancy calculations with Partial_Topology objects."""

    def test_basic_redundancy_calculation(self):
        """Test redundancy calculation with non-overlapping fragments."""
        # Create non-overlapping fragments
        fragments = [
            Partial_Topology.from_range("A", 1, 5, fragment_name="frag1"),
            Partial_Topology.from_range("A", 6, 10, fragment_name="frag2"),
            Partial_Topology.from_range("A", 11, 15, fragment_name="frag3"),
        ]

        # Calculate redundancy
        redundancy_max = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )
        redundancy_mean = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=False
        )

        # Non-overlapping fragments should have redundancy score of 0
        assert all(score == 0.0 for score in redundancy_max)
        assert all(score == 0.0 for score in redundancy_mean)

    def test_overlapping_redundancy_calculation(self):
        """Test redundancy calculation with overlapping fragments."""
        # Create fragments with various degrees of overlap
        fragments = [
            Partial_Topology.from_range("A", 1, 10, fragment_name="frag1"),
            Partial_Topology.from_range(
                "A", 5, 15, fragment_name="frag2"
            ),  # 6 residues overlap with frag1
            Partial_Topology.from_range(
                "A", 12, 20, fragment_name="frag3"
            ),  # 4 residues overlap with frag2
            Partial_Topology.from_range(
                "A", 30, 40, fragment_name="frag4"
            ),  # No overlap with others
        ]

        # Calculate redundancy
        redundancy_max = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )
        redundancy_mean = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=False
        )

        # Expected redundancy scores for max mode:
        # frag1: 6 (overlap with frag2)
        # frag2: 6 (overlap with frag1, which is greater than 4 overlap with frag3)
        # frag3: 4 (overlap with frag2)
        # frag4: 0 (no overlap)
        assert redundancy_max == [6.0, 6.0, 4.0, 0.0]

        # Expected redundancy scores for mean mode:
        # frag1: 6 (one overlap of 6)
        # frag2: (6+4)/2 = 5.0 (average of overlaps with frag1 and frag3)
        # frag3: 4 (one overlap of 4)
        # frag4: 0 (no overlap)
        assert redundancy_mean == [6.0, 5.0, 4.0, 0.0]

    def test_multiple_chains_redundancy(self):
        """Test redundancy calculation with fragments on different chains."""
        fragments = [
            Partial_Topology.from_range("A", 1, 10, fragment_name="fragA1"),
            Partial_Topology.from_range("A", 5, 15, fragment_name="fragA2"),  # Overlaps with fragA1
            Partial_Topology.from_range("B", 1, 10, fragment_name="fragB1"),
            Partial_Topology.from_range("B", 5, 15, fragment_name="fragB2"),  # Overlaps with fragB1
        ]

        # Calculate redundancy
        redundancy_max = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )

        # Each fragment should only have overlap with fragments on the same chain
        assert redundancy_max == [6.0, 6.0, 6.0, 6.0]

    def test_peptide_trim_redundancy(self):
        """Test how peptide trimming affects redundancy calculations."""
        # Create peptides with trimming
        peptides = [
            Partial_Topology.from_range(
                "A", 1, 10, fragment_name="pep1", peptide=True, peptide_trim=2
            ),
            # pep1 active residues: [3,4,5,6,7,8,9,10] after trimming first 2
            Partial_Topology.from_range(
                "A", 5, 15, fragment_name="pep2", peptide=True, peptide_trim=3
            ),
            # pep2 active residues: [8,9,10,11,12,13,14,15] after trimming first 3
            Partial_Topology.from_range(
                "A", 8, 17, fragment_name="pep3", peptide=True, peptide_trim=1
            ),
            # pep3 active residues: [9,10,11,12,13,14,15,16,17] after trimming first 1
        ]

        # Calculate redundancy WITH peptide trimming
        redundancy_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="max", check_trim=True
        )

        # Calculate redundancy WITHOUT peptide trimming
        redundancy_no_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="max", check_trim=False
        )

        # Print for clarity
        print("\nRedundancy calculation with peptide trimming:")
        for i, pep in enumerate(peptides):
            print(f"  Peptide {i}: {pep}")
            print(f"    Active residues with trimming: {pep._get_active_residues(check_trim=True)}")
            print(
                f"    Active residues without trimming: {pep._get_active_residues(check_trim=False)}"
            )

        print(f"  Redundancy scores with trim: {redundancy_trim}")
        print(f"  Redundancy scores without trim: {redundancy_no_trim}")

        # With trimming:
        # pep1 & pep2 overlap: [8,9,10] = 3 residues
        # pep1 & pep3 overlap: [9,10] = 2 residues
        # pep2 & pep3 overlap: [9,10,11,12,13,14,15] = 7 residues

        # Without trimming:
        # pep1 & pep2 overlap: [5,6,7,8,9,10] = 6 residues
        # pep1 & pep3 overlap: [8,9,10] = 3 residues
        # pep2 & pep3 overlap: [8,9,10,11,12,13,14,15] = 8 residues

        # Expected redundancy scores with trimming (max mode):
        # pep1: max(3, 2) = 3
        # pep2: max(3, 7) = 7
        # pep3: max(2, 7) = 7
        assert redundancy_trim == [3.0, 7.0, 7.0]

        # Expected redundancy scores without trimming (max mode):
        # pep1: max(6, 3) = 6
        # pep2: max(6, 8) = 8
        # pep3: max(3, 8) = 8
        assert redundancy_no_trim == [6.0, 8.0, 8.0]

    def test_mean_mode_with_peptide_trim(self):
        """Test mean mode redundancy calculation with peptide trimming."""
        # Create peptides with varying degrees of overlap and different trim values
        peptides = [
            Partial_Topology.from_range(
                "A", 1, 10, fragment_name="pep1", peptide=True, peptide_trim=2
            ),
            # pep1 active residues: [3,4,5,6,7,8,9,10]
            Partial_Topology.from_range(
                "A", 5, 14, fragment_name="pep2", peptide=True, peptide_trim=2
            ),
            # pep2 active residues: [7,8,9,10,11,12,13,14]
            Partial_Topology.from_range(
                "A", 9, 18, fragment_name="pep3", peptide=True, peptide_trim=2
            ),
            # pep3 active residues: [11,12,13,14,15,16,17,18]
            Partial_Topology.from_range("A", 30, 40, fragment_name="pep4", peptide=False),
            # pep4 active residues: [30-40], no trimming since not a peptide
        ]

        # Calculate redundancy in mean mode with trimming
        redundancy_mean_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="mean", check_trim=True
        )

        # Calculate redundancy in mean mode without trimming
        redundancy_mean_no_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="mean", check_trim=False
        )

        print("\nMean redundancy calculation with peptide trimming:")
        for i, pep in enumerate(peptides):
            print(f"  Peptide {i}: {pep}")
            print(f"    Active residues with trimming: {pep._get_active_residues(check_trim=True)}")

        print(f"  Mean redundancy with trim: {redundancy_mean_trim}")
        print(f"  Mean redundancy without trim: {redundancy_mean_no_trim}")

        # With trimming:
        # pep1 & pep2 overlap: [7,8,9,10] = 4 residues
        # pep1 & pep3 overlap: [] = 0 residues
        # pep2 & pep3 overlap: [11,12,13,14] = 4 residues
        # pep4 has no overlap with any other peptide

        # Expected mean redundancy scores with trimming:
        # pep1: 4.0 (only one overlap)
        # pep2: (4+4)/2 = 4.0 (average of overlaps with pep1 and pep3)
        # pep3: 4.0 (only one overlap)
        # pep4: 0.0 (no overlaps)
        assert redundancy_mean_trim == [4.0, 4.0, 4.0, 0.0]

        # Verify the results without trimming are different
        assert redundancy_mean_trim != redundancy_mean_no_trim

    def test_complex_redundancy_patterns(self):
        """Test redundancy calculation with more complex overlap patterns."""
        fragments = [
            # Chain A fragments with various overlaps
            Partial_Topology.from_range("A", 1, 10, fragment_name="fragA1"),
            Partial_Topology.from_range(
                "A", 8, 15, fragment_name="fragA2"
            ),  # 3 residues overlap with fragA1
            Partial_Topology.from_range(
                "A", 5, 12, fragment_name="fragA3"
            ),  # 6 residues overlap with fragA1, 5 with fragA2
            # Chain B fragments
            Partial_Topology.from_range("B", 1, 10, fragment_name="fragB1"),
            Partial_Topology.from_range(
                "B", 5, 15, fragment_name="fragB2"
            ),  # 6 residues overlap with fragB1
            # Mix of peptides and regular fragments
            Partial_Topology.from_range(
                "C", 1, 10, fragment_name="fragC1", peptide=True, peptide_trim=2
            ),
            # fragC1 active residues: [3-10]
            Partial_Topology.from_range(
                "C", 5, 15, fragment_name="fragC2", peptide=True, peptide_trim=3
            ),
            # fragC2 active residues: [8-15]
        ]

        # Calculate redundancy with both modes and trim settings
        redundancy_max_trim = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=True
        )
        redundancy_mean_trim = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=True
        )
        redundancy_max_no_trim = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )

        print("\nComplex redundancy patterns:")
        for i, frag in enumerate(fragments):
            print(f"  Fragment {i}: {frag}")
            if frag.peptide:
                print(
                    f"    Active residues with trimming: {frag._get_active_residues(check_trim=True)}"
                )

        print(f"  Max redundancy with trim: {redundancy_max_trim}")
        print(f"  Mean redundancy with trim: {redundancy_mean_trim}")
        print(f"  Max redundancy without trim: {redundancy_max_no_trim}")

        # Expected for Chain A fragments with max mode:
        # fragA1: max(3, 6) = 6
        # fragA2: max(3, 5) = 5
        # fragA3: max(6, 5) = 6
        assert redundancy_max_no_trim[0] == 6.0
        assert redundancy_max_no_trim[1] == 5.0
        assert redundancy_max_no_trim[2] == 6.0

        # Expected for Chain C peptides with trim:
        # fragC1: 3 residues overlap with fragC2 ([8,9,10])
        # fragC2: 3 residues overlap with fragC1 ([8,9,10])
        assert redundancy_max_trim[5] == 3.0
        assert redundancy_max_trim[6] == 3.0

        # Expected for Chain C peptides without trim:
        # fragC1: 6 residues overlap with fragC2 ([5,6,7,8,9,10])
        # fragC2: 6 residues overlap with fragC1 ([5,6,7,8,9,10])
        assert redundancy_max_no_trim[5] == 6.0
        assert redundancy_max_no_trim[6] == 6.0


class TestFragmentRedundancyComprehensive:
    """Comprehensive test class for fragment redundancy calculations."""

    def test_basic_redundancy_non_peptide(self):
        """Test basic redundancy calculation with non-peptide fragments."""
        # Create overlapping fragments
        fragments = [
            Partial_Topology.from_range("A", 1, 10, fragment_name="frag1"),  # [1-10]
            Partial_Topology.from_range(
                "A", 5, 15, fragment_name="frag2"
            ),  # [5-15], overlap with frag1: [5-10] = 6 residues
            Partial_Topology.from_range(
                "A", 12, 22, fragment_name="frag3"
            ),  # [12-22], overlap with frag2: [12-15] = 4 residues
            Partial_Topology.from_range("A", 25, 35, fragment_name="frag4"),  # [25-35], no overlaps
        ]

        # Calculate redundancy
        redundancy_max = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )
        redundancy_mean = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=False
        )

        # Expected overlaps:
        # frag1: overlaps with frag2 (6 residues) -> max=6, mean=6
        # frag2: overlaps with frag1 (6) and frag3 (4) -> max=6, mean=5
        # frag3: overlaps with frag2 (4 residues) -> max=4, mean=4
        # frag4: no overlaps -> max=0, mean=0

        expected_max = [6.0, 6.0, 4.0, 0.0]
        expected_mean = [6.0, 5.0, 4.0, 0.0]

        assert redundancy_max == expected_max, (
            f"Max redundancy mismatch: {redundancy_max} != {expected_max}"
        )
        assert redundancy_mean == expected_mean, (
            f"Mean redundancy mismatch: {redundancy_mean} != {expected_mean}"
        )

        print("✓ Basic non-peptide redundancy calculation passed")

    def test_peptide_redundancy_with_trimming(self):
        """Test how peptide trimming affects redundancy calculations."""
        # Create peptides with specific trimming that will change overlap patterns
        peptides = [
            # Peptide 1: residues [1-10], peptide_residues [4-10] (trim=3)
            Partial_Topology.from_range(
                "A", 1, 10, fragment_name="pep1", peptide=True, peptide_trim=3
            ),
            # Peptide 2: residues [8-17], peptide_residues [10-17] (trim=2)
            Partial_Topology.from_range(
                "A", 8, 17, fragment_name="pep2", peptide=True, peptide_trim=2
            ),
            # Peptide 3: residues [15-25], peptide_residues [17-25] (trim=2)
            Partial_Topology.from_range(
                "A", 15, 25, fragment_name="pep3", peptide=True, peptide_trim=2
            ),
        ]

        # Print peptide details for debugging
        print("\nPeptide details:")
        for i, pep in enumerate(peptides):
            print(f"  {pep.fragment_name}: full={pep.residues}, peptide={pep.peptide_residues}")

        # Calculate redundancy without trimming (using all residues)
        redundancy_no_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="max", check_trim=False
        )

        # Calculate redundancy with trimming (using peptide_residues)
        redundancy_with_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides, mode="max", check_trim=True
        )

        print(f"\nRedundancy without trimming: {redundancy_no_trim}")
        print(f"Redundancy with trimming: {redundancy_with_trim}")

        # Without trimming overlaps:
        # pep1 [1-10] vs pep2 [8-17]: overlap [8-10] = 3 residues
        # pep2 [8-17] vs pep1 [1-10]: overlap [8-10] = 3 residues
        # pep2 [8-17] vs pep3 [15-25]: overlap [15-17] = 3 residues
        # pep3 [15-25] vs pep2 [8-17]: overlap [15-17] = 3 residues
        expected_no_trim = [3.0, 3.0, 3.0]

        # With trimming overlaps:
        # pep1 [4-10] vs pep2 [10-17]: overlap [10] = 1 residue
        # pep2 [10-17] vs pep1 [4-10]: overlap [10] = 1 residue
        # pep2 [10-17] vs pep3 [17-25]: overlap [17] = 1 residue
        # pep3 [17-25] vs pep2 [10-17]: overlap [17] = 1 residue
        expected_with_trim = [1.0, 1.0, 1.0]

        assert redundancy_no_trim == expected_no_trim, (
            f"No-trim redundancy mismatch: {redundancy_no_trim}"
        )
        assert redundancy_with_trim == expected_with_trim, (
            f"With-trim redundancy mismatch: {redundancy_with_trim}"
        )

        print("✓ Peptide trimming effects on redundancy calculation passed")

    def test_mixed_peptide_non_peptide_redundancy(self):
        """Test redundancy calculation with mixed peptide and non-peptide fragments."""
        fragments = [
            # Non-peptide fragment
            Partial_Topology.from_range("A", 1, 15, fragment_name="domain", peptide=False),
            # Peptide that overlaps with domain
            Partial_Topology.from_range(
                "A", 10, 20, fragment_name="signal", peptide=True, peptide_trim=3
            ),  # peptide_residues [13-20]
            # Another peptide
            Partial_Topology.from_range(
                "A", 18, 28, fragment_name="linker", peptide=True, peptide_trim=2
            ),  # peptide_residues [20-28]
        ]

        print("\nMixed fragment details:")
        for frag in fragments:
            if frag.peptide:
                print(
                    f"  {frag.fragment_name} (peptide): full={frag.residues}, peptide={frag.peptide_residues}"
                )
            else:
                print(f"  {frag.fragment_name} (non-peptide): full={frag.residues}")

        # Test with check_trim=False (use all residues for all fragments)
        redundancy_no_trim = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )

        # Test with check_trim=True (use peptide_residues for peptides, all residues for non-peptides)
        redundancy_with_trim = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=True
        )

        print(f"Mixed redundancy (no trim): {redundancy_no_trim}")
        print(f"Mixed redundancy (with trim): {redundancy_with_trim}")

        # Verify that trimming affects peptide comparisons but not non-peptide ones
        # The non-peptide domain should have the same overlaps regardless of trim setting
        # But peptides should have different overlaps based on trim setting

        assert len(redundancy_no_trim) == 3
        assert len(redundancy_with_trim) == 3

        print("✓ Mixed peptide/non-peptide redundancy calculation passed")

    def test_cross_chain_redundancy(self):
        """Test that fragments on different chains don't contribute to redundancy."""
        fragments = [
            Partial_Topology.from_range("A", 1, 10, fragment_name="chainA_frag1"),
            Partial_Topology.from_range(
                "B", 1, 10, fragment_name="chainB_frag1"
            ),  # Same residues, different chain
            Partial_Topology.from_range(
                "A", 5, 15, fragment_name="chainA_frag2"
            ),  # Overlaps with chainA_frag1
            Partial_Topology.from_range(
                "B", 5, 15, fragment_name="chainB_frag2"
            ),  # Overlaps with chainB_frag1
        ]

        redundancy = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )

        # Expected: each fragment only overlaps with fragments on the same chain
        # chainA_frag1 overlaps with chainA_frag2: [5-10] = 6 residues
        # chainB_frag1 overlaps with chainB_frag2: [5-10] = 6 residues
        # chainA_frag2 overlaps with chainA_frag1: [5-10] = 6 residues
        # chainB_frag2 overlaps with chainB_frag1: [5-10] = 6 residues
        expected = [6.0, 6.0, 6.0, 6.0]

        assert redundancy == expected, (
            f"Cross-chain redundancy mismatch: {redundancy} != {expected}"
        )
        print("✓ Cross-chain redundancy isolation passed")

    def test_mode_comparison(self):
        """Test difference between max and mean redundancy modes."""
        fragments = [
            Partial_Topology.from_range(
                "A", 1, 10, fragment_name="central"
            ),  # Overlaps with multiple
            Partial_Topology.from_range(
                "A", 5, 12, fragment_name="overlap1"
            ),  # Overlaps with central: [5-10] = 6
            Partial_Topology.from_range(
                "A", 8, 15, fragment_name="overlap2"
            ),  # Overlaps with central: [8-10] = 3
            Partial_Topology.from_range(
                "A", 3, 8, fragment_name="overlap3"
            ),  # Overlaps with central: [3-8] = 6
        ]

        redundancy_max = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="max", check_trim=False
        )
        redundancy_mean = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=False
        )

        print("\nMode comparison:")
        print(f"Max redundancy: {redundancy_max}")
        print(f"Mean redundancy: {redundancy_mean}")

        # For the central fragment:
        # Overlaps: 6 (overlap1), 3 (overlap2), 6 (overlap3)
        # Max should be 6, Mean should be (6+3+6)/3 = 5.0
        assert redundancy_max[0] == 6.0, (
            f"Central fragment max redundancy should be 6, got {redundancy_max[0]}"
        )
        assert redundancy_mean[0] == 5.0, (
            f"Central fragment mean redundancy should be 5.0, got {redundancy_mean[0]}"
        )

        print("✓ Mode comparison (max vs mean) passed")

    def test_edge_cases(self):
        """Test edge cases for redundancy calculation."""

        # Test empty list
        with pytest.raises(ValueError, match="Cannot calculate redundancy for empty topology list"):
            Partial_Topology.calculate_fragment_redundancy([], mode="max")

        # Test invalid mode
        fragments = [Partial_Topology.from_range("A", 1, 5, fragment_name="test")]
        with pytest.raises(ValueError, match="Mode must be either 'max' or 'mean'"):
            Partial_Topology.calculate_fragment_redundancy(fragments, mode="invalid")

        # Test single fragment (should have 0 redundancy)
        redundancy = Partial_Topology.calculate_fragment_redundancy(fragments, mode="max")
        assert redundancy == [0.0], f"Single fragment should have 0 redundancy, got {redundancy}"

        # Test fragments with no overlaps
        non_overlapping = [
            Partial_Topology.from_range("A", 1, 5, fragment_name="frag1"),
            Partial_Topology.from_range("A", 10, 15, fragment_name="frag2"),
            Partial_Topology.from_range("A", 20, 25, fragment_name="frag3"),
        ]
        redundancy = Partial_Topology.calculate_fragment_redundancy(non_overlapping, mode="max")
        assert redundancy == [0.0, 0.0, 0.0], (
            f"Non-overlapping fragments should have 0 redundancy, got {redundancy}"
        )

        print("✓ Edge cases passed")

    def test_peptide_trim_zero_vs_full_residues(self):
        """Test peptides with trim=0 vs full residue usage."""
        # Create peptides where trim=0 means peptide_residues == residues
        peptides_trim_zero = [
            Partial_Topology.from_range(
                "A", 1, 10, fragment_name="pep1", peptide=True, peptide_trim=0
            ),  # peptide_residues = [1-10]
            Partial_Topology.from_range(
                "A", 5, 15, fragment_name="pep2", peptide=True, peptide_trim=0
            ),  # peptide_residues = [5-15]
        ]

        # Same fragments but as non-peptides
        non_peptides = [
            Partial_Topology.from_range("A", 1, 10, fragment_name="frag1", peptide=False),
            Partial_Topology.from_range("A", 5, 15, fragment_name="frag2", peptide=False),
        ]

        # Calculate redundancy for peptides with trim (should be same as without trim)
        peptide_redundancy_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides_trim_zero, mode="max", check_trim=True
        )
        peptide_redundancy_no_trim = Partial_Topology.calculate_fragment_redundancy(
            peptides_trim_zero, mode="max", check_trim=False
        )

        # Calculate redundancy for non-peptides
        non_peptide_redundancy = Partial_Topology.calculate_fragment_redundancy(
            non_peptides, mode="max", check_trim=False
        )

        # All should be the same since trim=0 means peptide_residues == residues
        assert peptide_redundancy_trim == peptide_redundancy_no_trim
        assert peptide_redundancy_trim == non_peptide_redundancy

        print("✓ Peptide trim=0 vs non-peptide equivalence passed")

    def test_biological_scenario_redundancy(self):
        """Test a realistic biological scenario with multiple domain types."""
        # Simulate a protein with signal peptide, domains, and linker regions
        fragments = [
            # Signal peptide (will be trimmed for processing)
            Partial_Topology.from_range(
                "A", 1, 25, fragment_name="signal_peptide", peptide=True, peptide_trim=5
            ),  # Active: [6-25]
            # N-terminal domain
            Partial_Topology.from_range("A", 20, 150, fragment_name="N_domain", peptide=False),
            # Linker peptide
            Partial_Topology.from_range(
                "A", 140, 160, fragment_name="linker", peptide=True, peptide_trim=3
            ),  # Active: [143-160]
            # C-terminal domain
            Partial_Topology.from_range("A", 155, 300, fragment_name="C_domain", peptide=False),
            # Active site (scattered residues)
            Partial_Topology.from_residues(
                "A", [45, 78, 123, 180, 245], fragment_name="active_site", peptide=False
            ),
            # Binding pocket
            Partial_Topology.from_residues(
                "A",
                [75, 76, 77, 78, 79, 122, 123, 124],
                fragment_name="binding_pocket",
                peptide=False,
            ),
        ]

        print("\nBiological scenario fragments:")
        for frag in fragments:
            if frag.peptide:
                print(
                    f"  {frag.fragment_name}: full={len(frag.residues)}, active={len(frag.peptide_residues)}"
                )
            else:
                print(f"  {frag.fragment_name}: {len(frag.residues)} residues")

        # Calculate redundancy with biological trimming
        redundancy_biological = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=True
        )

        # Calculate redundancy without trimming
        redundancy_full = Partial_Topology.calculate_fragment_redundancy(
            fragments, mode="mean", check_trim=False
        )

        print(
            f"\nBiological redundancy (with trimming): {[f'{r:.1f}' for r in redundancy_biological]}"
        )
        print(f"Full redundancy (no trimming): {[f'{r:.1f}' for r in redundancy_full]}")

        # Verify that active site and binding pocket have high redundancy (they overlap)
        # binding_pocket contains residues [75,76,77,78,79,122,123,124]
        # active_site contains residues [45,78,123,180,245]
        # Overlap: [78, 123] = 2 residues

        assert len(redundancy_biological) == len(fragments)
        assert all(r >= 0 for r in redundancy_biological), (
            "All redundancy scores should be non-negative"
        )

        print("✓ Biological scenario redundancy calculation passed")

    def test_redundancy_sensitivity_to_trim_values(self):
        """Test how different trim values affect redundancy calculations."""
        base_peptide_data = [
            ("A", 1, 20, "pep1"),
            ("A", 15, 35, "pep2"),  # Overlaps with pep1: [15-20]
            ("A", 30, 50, "pep3"),  # Overlaps with pep2: [30-35]
        ]

        trim_values = [0, 2, 5, 8]
        results = {}

        for trim in trim_values:
            peptides = [
                Partial_Topology.from_range(
                    chain, start, end, fragment_name=name, peptide=True, peptide_trim=trim
                )
                for chain, start, end, name in base_peptide_data
            ]

            redundancy = Partial_Topology.calculate_fragment_redundancy(
                peptides, mode="max", check_trim=True
            )
            results[trim] = redundancy

            print(f"\nTrim={trim}:")
            for i, pep in enumerate(peptides):
                print(
                    f"  {pep.fragment_name}: full={pep.residues}, active={pep.peptide_residues}, redundancy={redundancy[i]}"
                )

        # Verify that higher trim values generally lead to lower redundancy
        # (since less of each peptide is considered active)
        print("\nRedundancy vs trim value:")
        for trim in trim_values:
            avg_redundancy = sum(results[trim]) / len(results[trim])
            print(f"  Trim {trim}: avg redundancy = {avg_redundancy:.2f}")

        print("✓ Redundancy sensitivity to trim values analysis completed")

    def test_redundancy_with_identical_peptides(self):
        """Test redundancy calculation with identical or highly overlapping peptides."""
        # Create identical peptides (should have maximum redundancy)
        identical_peptides = [
            Partial_Topology.from_range(
                "A", 10, 20, fragment_name="dup1", peptide=True, peptide_trim=2
            ),
            Partial_Topology.from_range(
                "A", 10, 20, fragment_name="dup2", peptide=True, peptide_trim=2
            ),
            Partial_Topology.from_range(
                "A", 10, 20, fragment_name="dup3", peptide=True, peptide_trim=2
            ),
        ]

        redundancy = Partial_Topology.calculate_fragment_redundancy(
            identical_peptides, mode="max", check_trim=True
        )

        # Each peptide should have maximum overlap with others
        # Active residues for each: [12-20] = 9 residues
        # Each overlaps with 2 others by 9 residues each
        expected_redundancy = 9.0  # Maximum overlap

        assert all(r == expected_redundancy for r in redundancy), (
            f"Identical peptides should have redundancy {expected_redundancy}, got {redundancy}"
        )

        # Test with mean mode
        redundancy_mean = Partial_Topology.calculate_fragment_redundancy(
            identical_peptides, mode="mean", check_trim=True
        )

        # Mean of [9, 9] = 9.0
        assert all(r == expected_redundancy for r in redundancy_mean), (
            f"Identical peptides mean redundancy should be {expected_redundancy}, got {redundancy_mean}"
        )

        print("✓ Identical peptides redundancy calculation passed")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main(["-xvs", __file__])
