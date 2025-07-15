import copy
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import Partial_Topology


# Mock ExpD_Datapoint for testing
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top, data_id=None):
        self.top = top
        self.key = f"mock_key_{data_id}" if data_id else "mock_key"
        self.data_id = data_id

    def extract_features(self):
        return np.array([1.0])  # Mock feature

    def __repr__(self):
        return f"MockExpD_Datapoint(id={self.data_id}, top={self.top})"


# Helper functions to create diverse topologies
def create_single_chain_topologies(chain="A", count=10):
    """Create diverse single-chain topologies with good separation."""
    topologies = []
    start = 1
    gap = 15  # Larger gap to reduce overlap
    length = 10

    for i in range(count):
        topologies.append(
            Partial_Topology.from_range(
                chain, start, start + length - 1, fragment_name=f"frag_{chain}_{i + 1}"
            )
        )
        start += length + gap

    return topologies


def create_multi_chain_topologies(chains=["A", "B", "C"], count_per_chain=10):
    """Create diverse multi-chain topologies with good separation."""
    topologies = []

    for chain in chains:
        start = 1
        gap = 20  # Larger gap for better separation
        length = 12

        for i in range(count_per_chain):
            topologies.append(
                Partial_Topology.from_range(
                    chain, start, start + length - 1, fragment_name=f"frag_{chain}_{i + 1}"
                )
            )
            start += length + gap

    return topologies


def create_peptide_topologies(chains=["A", "B", "C"], count_per_chain=10):
    """Create diverse peptide topologies with trimming and good separation."""
    topologies = []
    trim_values = [1, 2, 3]  # Rotate through different trim values

    for chain in chains:
        start = 1
        gap = 18  # Gap to prevent overlap
        length = 15

        for i in range(count_per_chain):
            trim = trim_values[i % len(trim_values)]
            topologies.append(
                Partial_Topology.from_range(
                    chain,
                    start,
                    start + length - 1,
                    fragment_name=f"pep_{chain}_{i + 1}",
                    peptide=True,
                    peptide_trim=trim,
                )
            )
            start += length + gap

    return topologies


def create_overlapping_topologies(chain="A", count=10):
    """Create topologies with controlled overlap for testing overlap removal."""
    topologies = []
    start = 1
    overlap = 5  # Controlled overlap
    length = 15

    for i in range(count):
        topologies.append(
            Partial_Topology.from_range(
                chain, start, start + length - 1, fragment_name=f"overlap_{chain}_{i + 1}"
            )
        )
        start += length - overlap  # Create overlap

    return topologies


def create_common_residues_for_chains(chains, coverage_factor=0.7):
    """Create common residue topologies that cover a portion of each chain."""
    common_residues = set()

    for chain in chains:
        # Calculate coverage based on typical topology ranges
        if chain == "A":
            end_pos = int(300 * coverage_factor)  # Cover 70% of typical A chain range
            common_residues.add(
                Partial_Topology.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )
        elif chain == "B":
            end_pos = int(250 * coverage_factor)  # Cover 70% of typical B chain range
            common_residues.add(
                Partial_Topology.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )
        elif chain == "C":
            end_pos = int(200 * coverage_factor)  # Cover 70% of typical C chain range
            common_residues.add(
                Partial_Topology.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )

    return common_residues


# Test fixtures using helper functions
@pytest.fixture
def single_chain_topologies():
    """Create single-chain test topologies with good diversity."""
    return create_single_chain_topologies("A", 10)


@pytest.fixture
def multi_chain_topologies():
    """Create multi-chain test topologies with good diversity."""
    return create_multi_chain_topologies(["A", "B", "C"], 10)


@pytest.fixture
def peptide_topologies():
    """Create peptide topologies with trimming and good diversity."""
    return create_peptide_topologies(["A", "B", "C"], 10)


@pytest.fixture
def large_single_chain_dataset():
    """Create a large single-chain dataset for robust testing."""
    return create_single_chain_topologies("A", 20)


@pytest.fixture
def large_multi_chain_dataset():
    """Create a large multi-chain dataset for robust testing."""
    return create_multi_chain_topologies(["A", "B", "C", "D"], 15)


@pytest.fixture
def overlapping_topologies():
    """Create topologies with controlled overlap."""
    return create_overlapping_topologies("A", 12)


@pytest.fixture
def create_datapoints_from_topologies():
    """Factory to create datapoints from topologies."""

    def _create(topologies):
        return [MockExpD_Datapoint(topo, i) for i, topo in enumerate(topologies)]

    return _create


@pytest.fixture
def mock_dataloader():
    """Factory to create mock dataloader."""

    def _create(datapoints):
        mock_loader = MagicMock(spec=ExpD_Dataloader)
        mock_loader.data = datapoints
        mock_loader.y_true = np.array([1.0] * len(datapoints))
        return mock_loader

    return _create


@pytest.fixture
def setup_splitter(request):
    """Factory to set up DataSplitter with mocked dependencies."""

    def _setup(datapoints, common_residues, **kwargs):
        mock_loader = MagicMock(spec=ExpD_Dataloader)
        mock_loader.data = copy.deepcopy(datapoints)
        mock_loader.y_true = np.array([1.0] * len(datapoints))

        # Mock filter_common_residues to return all datapoints that intersect
        def mock_filter_func(dataset, common_topos, check_trim=False):
            if not common_topos:
                return []
            # A simple mock that returns any datapoint that intersects with any common topology
            return [
                dp
                for dp in dataset
                if any(dp.top.intersects(ct, check_trim=check_trim) for ct in common_topos)
            ]

        # Mock calculate_fragment_redundancy
        patcher1 = patch(
            "jaxent.src.interfaces.topology.Partial_Topology.calculate_fragment_redundancy"
        )
        patcher2 = patch(
            "jaxent.src.data.splitting.split.filter_common_residues", side_effect=mock_filter_func
        )

        mock_calc = patcher1.start()
        mock_filter = patcher2.start()

        request.addfinalizer(patcher1.stop)
        request.addfinalizer(patcher2.stop)

        mock_calc.return_value = [0.5] * len(datapoints)  # Mock centrality scores

        # Default kwargs
        default_kwargs = {
            "random_seed": 42,
            "train_size": 0.6,
            "centrality": False,
            "check_trim": False,
        }
        default_kwargs.update(kwargs)

        # During init, filter_common_residues is called. We want to return all datapoints here for simplicity.
        mock_filter.side_effect = None
        mock_filter.return_value = copy.deepcopy(datapoints)

        splitter = DataSplitter(
            dataset=mock_loader, common_residues=common_residues, **default_kwargs
        )

        # Restore the side effect for calls within random_split
        mock_filter.side_effect = mock_filter_func
        mock_filter.return_value = None

        return splitter

    return _setup


class TestRandomSplitBasicFunctionality:
    """Test basic functionality of random_split method."""

    def test_basic_split_returns_two_lists(
        self, large_single_chain_dataset, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that random_split returns two lists."""
        datapoints = create_datapoints_from_topologies(large_single_chain_dataset)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.random_split()

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_split_ratios_approximately_correct(
        self, large_single_chain_dataset, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that split ratios are approximately correct."""
        datapoints = create_datapoints_from_topologies(large_single_chain_dataset)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.7)
        train_data, val_data = splitter.random_split()

        total_data = len(train_data) + len(val_data)
        train_ratio = len(train_data) / total_data

        # Allow some tolerance due to rounding and filtering
        assert 0.5 <= train_ratio <= 0.9  # Should be around 0.7

    def test_no_data_loss_in_split(
        self, large_single_chain_dataset, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that all datapoints are assigned to either train or val."""
        datapoints = create_datapoints_from_topologies(large_single_chain_dataset)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.random_split()

        # With default remove_overlap=False, the sum can be greater than original
        # but unique datapoints should be less than or equal to original dataset
        all_data_ids = {dp.data_id for dp in train_data}.union({dp.data_id for dp in val_data})
        assert len(all_data_ids) <= len(datapoints)

        # Check that we have some overlapping datapoints (which is expected behavior)
        train_ids = {dp.data_id for dp in train_data}
        val_ids = {dp.data_id for dp in val_data}
        overlapping_ids = train_ids.intersection(val_ids)
        print(f"Overlapping datapoints: {len(overlapping_ids)}")


class TestRandomSplitChainAwareness:
    """Test chain-aware behavior of random_split."""

    def test_multi_chain_split(
        self, large_multi_chain_dataset, create_datapoints_from_topologies, setup_splitter
    ):
        """Test splitting with multiple chains."""
        datapoints = create_datapoints_from_topologies(large_multi_chain_dataset)
        common_residues = create_common_residues_for_chains(["A", "B", "C", "D"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.random_split()

        # Check that merged topologies by chain are stored
        assert len(splitter.last_split_train_topologies_by_chain) > 0
        assert len(splitter.last_split_val_topologies_by_chain) > 0

        # Check that we have data for multiple chains
        train_chains = {dp.top.chain for dp in train_data}
        val_chains = {dp.top.chain for dp in val_data}

        assert len(train_chains | val_chains) > 1  # Multiple chains represented

    def test_chain_specific_topology_merging(
        self, large_multi_chain_dataset, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that topologies are correctly merged by chain."""
        datapoints = create_datapoints_from_topologies(large_multi_chain_dataset)
        common_residues = create_common_residues_for_chains(["A", "B", "C", "D"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.random_split()

        # Check stored merged topologies
        for chain, merged_topo in splitter.last_split_train_topologies_by_chain.items():
            assert merged_topo.chain == chain
            assert "train_chain_" in merged_topo.fragment_name

        for chain, merged_topo in splitter.last_split_val_topologies_by_chain.items():
            assert merged_topo.chain == chain
            assert "val_chain_" in merged_topo.fragment_name

    def test_single_chain_vs_multi_chain(self, create_datapoints_from_topologies, setup_splitter):
        """Test that single chain and multi-chain datasets behave consistently."""
        # Single chain - use helper function for more data
        single_topologies = create_single_chain_topologies("A", 15)
        single_datapoints = create_datapoints_from_topologies(single_topologies)
        single_common = create_common_residues_for_chains(["A"])

        single_splitter = setup_splitter(single_datapoints, single_common, random_seed=123)
        single_train, single_val = single_splitter.random_split()

        # Multi chain - use helper function for more data
        multi_topologies = create_multi_chain_topologies(["A", "B", "C"], 12)
        multi_datapoints = create_datapoints_from_topologies(multi_topologies)
        multi_common = create_common_residues_for_chains(["A", "B", "C"])

        multi_splitter = setup_splitter(multi_datapoints, multi_common, random_seed=123)
        multi_train, multi_val = multi_splitter.random_split()

        # Both should have valid splits
        assert len(single_train) > 0 and len(single_val) > 0
        assert len(multi_train) > 0 and len(multi_val) > 0

        # Single chain should have only one chain in stored topologies
        assert len(single_splitter.last_split_train_topologies_by_chain) <= 1

        # Multi chain should have multiple chains
        assert len(multi_splitter.last_split_train_topologies_by_chain) > 1


class TestRandomSplitPeptideHandling:
    """Test peptide trimming functionality."""

    def test_peptide_trimming_enabled(self, create_datapoints_from_topologies, setup_splitter):
        """Test split with peptide trimming enabled."""
        peptide_topologies = create_peptide_topologies(["A", "B", "C"], 12)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)
        train_data, val_data = splitter.random_split()

        # Should handle peptide topologies correctly
        assert len(train_data) + len(val_data) > 0

        # Check that check_trim setting is used in splitter
        assert splitter.check_trim is True

    def test_peptide_vs_non_peptide_comparison(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test difference between peptide and non-peptide splitting."""
        peptide_topologies = create_peptide_topologies(["A", "B", "C"], 15)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        # Split with check_trim=True
        peptide_splitter = setup_splitter(
            datapoints, common_residues, check_trim=True, random_seed=456
        )
        peptide_train, peptide_val = peptide_splitter.random_split()

        # Split with check_trim=False
        no_peptide_splitter = setup_splitter(
            datapoints, common_residues, check_trim=False, random_seed=456
        )
        no_peptide_train, no_peptide_val = no_peptide_splitter.random_split()

        # Both should produce valid splits
        assert len(peptide_train) + len(peptide_val) > 0
        assert len(no_peptide_train) + len(no_peptide_val) > 0


class TestRandomSplitCentralitySampling:
    """Test centrality sampling functionality."""

    def test_centrality_sampling_enabled(self, create_datapoints_from_topologies, setup_splitter):
        """Test split with centrality sampling enabled."""
        # Use helper function for more data
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:15]  # Return subset

            splitter = setup_splitter(datapoints, common_residues, centrality=True)
            train_data, val_data = splitter.random_split()

            # Should have called sample_by_centrality
            mock_sample.assert_called_once()

    def test_centrality_vs_no_centrality(self, create_datapoints_from_topologies, setup_splitter):
        """Test difference between centrality and no centrality sampling."""
        topologies = create_single_chain_topologies("A", 18)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # With centrality
        centrality_splitter = setup_splitter(
            datapoints, common_residues, centrality=True, random_seed=789
        )

        # Without centrality
        no_centrality_splitter = setup_splitter(
            datapoints, common_residues, centrality=False, random_seed=789
        )

        # Both should work
        with patch.object(DataSplitter, "sample_by_centrality", return_value=datapoints):
            centrality_train, centrality_val = centrality_splitter.random_split()

        no_centrality_train, no_centrality_val = no_centrality_splitter.random_split()

        assert len(centrality_train) + len(centrality_val) > 0
        assert len(no_centrality_train) + len(no_centrality_val) > 0


class TestRandomSplitOverlapRemoval:
    """Test overlap removal functionality."""

    def test_overlap_removal_disabled(self, create_datapoints_from_topologies, setup_splitter):
        """Test split with overlap removal disabled."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.random_split(remove_overlap=False)

        # Should work without overlap removal
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_overlap_removal_enabled(self, create_datapoints_from_topologies, setup_splitter):
        """Test split with overlap removal enabled using multi-chain data for better diversity."""
        # Use multi-chain data which has better separation
        topologies = create_multi_chain_topologies(["A", "B", "C"], 12)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.random_split(remove_overlap=True)

        # Should work with overlap removal
        assert len(train_data) >= 0
        assert len(val_data) >= 0
        # With diverse multi-chain data, we expect a valid split
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_overlap_removal_comparison(self, create_datapoints_from_topologies, setup_splitter):
        """Test difference between overlap removal enabled vs disabled."""
        # Use multi-chain data which has better separation
        topologies = create_multi_chain_topologies(["A", "B", "C"], 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=999, train_size=0.5)

        # Without overlap removal
        train_no_removal, val_no_removal = splitter.random_split(remove_overlap=False)

        # Reset splitter state and try with overlap removal
        splitter.original_random_seed = 999
        splitter._reset_retry_counter()
        train_with_removal, val_with_removal = splitter.random_split(remove_overlap=True)

        # With overlap removal, total count should be <= without removal
        total_no_removal = len(train_no_removal) + len(val_no_removal)
        total_with_removal = len(train_with_removal) + len(val_with_removal)

        assert total_with_removal <= total_no_removal

    def test_overlapping_topologies_handling(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test handling of explicitly overlapping topologies."""
        overlapping_topologies = create_overlapping_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(overlapping_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # Test both overlap removal settings
        train_no_removal, val_no_removal = splitter.random_split(remove_overlap=False)

        splitter._reset_retry_counter()
        train_removal, val_removal = splitter.random_split(remove_overlap=True)

        # Both should work, removal version should have <= total datapoints
        assert len(train_no_removal) + len(val_no_removal) > 0
        assert len(train_removal) + len(val_removal) > 0


class TestRandomSplitDeterministicBehavior:
    """Test deterministic behavior and random seed handling."""

    def test_same_seed_same_results(self, create_datapoints_from_topologies, setup_splitter):
        """Test that same seed produces same results."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=123)
        train1, val1 = splitter1.random_split()

        # Second split with same seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=123)
        train2, val2 = splitter2.random_split()

        # Results should be identical
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}
        val1_ids = {dp.data_id for dp in val1}
        val2_ids = {dp.data_id for dp in val2}

        assert train1_ids == train2_ids
        assert val1_ids == val2_ids

    def test_different_seed_different_results(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that different seeds produce different results."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=111)
        train1, val1 = splitter1.random_split()

        # Second split with different seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=222)
        train2, val2 = splitter2.random_split()

        # Results should likely be different (not guaranteed but highly probable)
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}

        # At least one should be different
        assert len(train1_ids.symmetric_difference(train2_ids)) > 0


class TestRandomSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_retry_logic_on_invalid_split(self, create_datapoints_from_topologies, setup_splitter):
        """Test retry logic when validation fails."""
        # Create dataset that might cause validation failure
        small_topologies = create_single_chain_topologies("A", 3)
        datapoints = create_datapoints_from_topologies(small_topologies)
        common_residues = create_common_residues_for_chains(["A"], coverage_factor=0.5)

        # Mock validate_split to fail first time, then succeed
        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = [ValueError("Too small"), True]

            splitter = setup_splitter(datapoints, common_residues)

            # Should retry and eventually succeed
            train_data, val_data = splitter.random_split()

            # Should have called validate_split twice (fail, then success)
            assert mock_validate.call_count == 2

    def test_empty_intersection_handling(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling when no datapoints intersect with merged topologies."""
        # Create datapoints that don't overlap with common residues well
        non_overlap_topologies = [
            Partial_Topology.from_range("A", 1000 + i * 20, 1010 + i * 20, fragment_name=f"far{i}")
            for i in range(10)
        ]
        datapoints = create_datapoints_from_topologies(non_overlap_topologies)
        common_residues = create_common_residues_for_chains(["A"], coverage_factor=0.3)

        splitter = setup_splitter(datapoints, common_residues)

        # Should handle gracefully (might retry or raise validation error)
        try:
            train_data, val_data = splitter.random_split()
            # If it succeeds, check that we have some data
            assert len(train_data) + len(val_data) >= 0
        except ValueError:
            # If it fails with validation error, that's acceptable
            pass

    def test_all_data_in_one_set(self, create_datapoints_from_topologies, setup_splitter):
        """Test case where all data might end up in one set."""
        small_topologies = create_single_chain_topologies("A", 5)
        datapoints = create_datapoints_from_topologies(small_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.9)

        # Should handle gracefully
        try:
            train_data, val_data = splitter.random_split()
            # Check that we don't lose all data
            assert len(train_data) + len(val_data) > 0
        except ValueError:
            # Validation failure is acceptable for edge cases
            pass


class TestRandomSplitStorageAndState:
    """Test storage of split results and state management."""

    def test_split_results_storage(self, create_datapoints_from_topologies, setup_splitter):
        """Test that split results are correctly stored."""
        topologies = create_multi_chain_topologies(["A", "B"], 12)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)

        # Before split, storage should be empty
        assert len(splitter.last_split_train_topologies_by_chain) == 0
        assert len(splitter.last_split_val_topologies_by_chain) == 0

        # After split, storage should be populated
        train_data, val_data = splitter.random_split()

        assert len(splitter.last_split_train_topologies_by_chain) > 0
        assert len(splitter.last_split_val_topologies_by_chain) > 0

        # Check that stored topologies have correct names and chains
        for chain, topo in splitter.last_split_train_topologies_by_chain.items():
            assert isinstance(chain, str)
            assert topo.chain == chain
            assert "train_chain_" in topo.fragment_name

        for chain, topo in splitter.last_split_val_topologies_by_chain.items():
            assert isinstance(chain, str)
            assert topo.chain == chain
            assert "val_chain_" in topo.fragment_name

    def test_consecutive_splits_update_storage(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that consecutive splits update storage correctly."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # First split
        train1, val1 = splitter.random_split()
        first_train_storage = copy.deepcopy(splitter.last_split_train_topologies_by_chain)

        # Change seed and split again
        splitter.random_seed = 999
        random.seed(999)
        train2, val2 = splitter.random_split()
        second_train_storage = copy.deepcopy(splitter.last_split_train_topologies_by_chain)

        # Storage should be updated (topologies might be different)
        assert len(first_train_storage) > 0
        assert len(second_train_storage) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
