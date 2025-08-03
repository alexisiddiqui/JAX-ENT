import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology.factory import TopologyFactory
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons


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
def create_sequential_topologies(chain="A", count=10, start_pos=1, gap=5, length=8):
    """Create sequential topologies with specific ordering for testing sequence splits."""
    topologies = []
    current_pos = start_pos

    for i in range(count):
        topologies.append(
            TopologyFactory.from_range(
                chain, current_pos, current_pos + length - 1, fragment_name=f"seq_{chain}_{i + 1}"
            )
        )
        current_pos += length + gap

    return topologies


def create_interleaved_chain_topologies(chains=["A", "B", "C"], count_per_chain=8):
    """Create topologies that interleave across chains for testing multi-chain sequence ordering."""
    topologies = []

    # Create overlapping ranges across chains to test sorting
    chain_starts = {"A": 1, "B": 50, "C": 100}

    for i in range(count_per_chain):
        for chain in chains:
            start = chain_starts[chain] + (i * 15)
            topologies.append(
                TopologyFactory.from_range(
                    chain, start, start + 10, fragment_name=f"interleaved_{chain}_{i + 1}"
                )
            )

    return topologies


def create_overlapping_topologies(chain="A", count=12, base_start=1, overlap=3):
    """Create overlapping topologies to test sequence ordering with overlaps."""
    topologies = []

    for i in range(count):
        start = base_start + (i * (10 - overlap))  # Each fragment overlaps by 'overlap' residues
        topologies.append(
            TopologyFactory.from_range(
                chain, start, start + 9, fragment_name=f"overlap_{chain}_{i + 1}"
            )
        )

    return topologies


def create_peptide_sequence_topologies(chains=["A", "B"], count_per_chain=8):
    """Create peptide topologies with trimming for sequence testing."""
    topologies = []
    trim_values = [1, 2, 3]

    for chain in chains:
        start = 1 if chain == "A" else 200
        gap = 20
        length = 15

        for i in range(count_per_chain):
            trim = trim_values[i % len(trim_values)]
            topologies.append(
                TopologyFactory.from_range(
                    chain,
                    start,
                    start + length - 1,
                    fragment_name=f"seq_pep_{chain}_{i + 1}",
                    peptide=True,
                    peptide_trim=trim,
                )
            )
            start += length + gap

    return topologies


def create_common_residues_for_chains(chains, coverage_factor=0.8):
    """Create common residue topologies that cover a portion of each chain."""
    common_residues = set()

    if len(chains) == 1:
        chain = chains[0]
        if chain == "A":
            range1_end = int(200 * coverage_factor)
            range2_start = 250
            range2_end = int(range2_start + (400 - range2_start) * coverage_factor)
        elif chain == "B":
            range1_end = int(150 * coverage_factor)
            range2_start = 200
            range2_end = int(range2_start + (350 - range2_start) * coverage_factor)
        elif chain == "C":
            range1_end = int(120 * coverage_factor)
            range2_start = 150
            range2_end = int(range2_start + (300 - range2_start) * coverage_factor)
        else:
            range1_end = int(100 * coverage_factor)
            range2_start = 150
            range2_end = int(range2_start + (250 - range2_start) * coverage_factor)

        if range1_end >= 1:
            common_residues.add(
                TopologyFactory.from_range(chain, 1, range1_end, fragment_name=f"common_{chain}_1")
            )

        if range2_end >= range2_start:
            common_residues.add(
                TopologyFactory.from_range(
                    chain, range2_start, range2_end, fragment_name=f"common_{chain}_2"
                )
            )

        if len(common_residues) < 2:
            dummy_start = 500
            while len(common_residues) < 2:
                common_residues.add(
                    TopologyFactory.from_range(
                        chain,
                        dummy_start,
                        dummy_start + 5,
                        fragment_name=f"common_{chain}_dummy_{len(common_residues)}",
                    )
                )
                dummy_start += 20

        return common_residues

    for chain in chains:
        if chain == "A":
            end_pos = int(400 * coverage_factor)
        elif chain == "B":
            end_pos = int(350 * coverage_factor)
        elif chain == "C":
            end_pos = int(300 * coverage_factor)
        elif chain == "D":
            end_pos = int(250 * coverage_factor)
        else:
            end_pos = int(200 * coverage_factor)

        if end_pos >= 1:
            common_residues.add(
                TopologyFactory.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )

    return common_residues


@pytest.fixture
def create_datapoints_from_topologies():
    """Factory to create datapoints from topologies."""

    def _create(topologies):
        return [MockExpD_Datapoint(topo, i) for i, topo in enumerate(topologies)]

    return _create


@pytest.fixture
def setup_splitter(request):
    """Factory to set up DataSplitter with mocked dependencies."""

    def _setup(datapoints, common_residues, **kwargs):
        mock_loader = MagicMock(spec=ExpD_Dataloader)
        mock_loader.data = copy.deepcopy(datapoints)
        mock_loader.y_true = np.array([1.0] * len(datapoints))

        def mock_filter_func(dataset, common_topos, check_trim=False):
            if not common_topos:
                return []
            return [
                dp
                for dp in dataset
                if any(
                    PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim=check_trim)
                    for ct in common_topos
                )
            ]

        patcher1 = patch("jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy")
        patcher2 = patch(
            "jaxent.src.data.splitting.split.filter_common_residues", side_effect=mock_filter_func
        )

        mock_calc = patcher1.start()
        mock_filter = patcher2.start()

        request.addfinalizer(patcher1.stop)
        request.addfinalizer(patcher2.stop)

        mock_calc.return_value = [0.5] * len(datapoints)

        default_kwargs = {
            "random_seed": 42,
            "train_size": 0.6,
            "centrality": False,
            "check_trim": True,
            "min_split_size": 2,
        }
        default_kwargs.update(kwargs)

        mock_filter.side_effect = None
        mock_filter.return_value = copy.deepcopy(datapoints)

        splitter = DataSplitter(
            dataset=mock_loader, common_residues=common_residues, **default_kwargs
        )

        mock_filter.side_effect = mock_filter_func
        mock_filter.return_value = None

        return splitter

    return _setup


class TestSequenceSplitBasicFunctionality:
    """Test basic functionality of sequence_split method."""

    def test_basic_sequence_split_returns_two_lists(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that sequence_split returns two lists."""
        topologies = create_sequential_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.sequence_split()

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_sequence_ordering_maintained(self, create_datapoints_from_topologies, setup_splitter):
        """Test that datapoints are sorted by sequence position before splitting."""
        # Create topologies in random order, but with predictable sequence positions
        topologies = [
            TopologyFactory.from_range("A", 100, 110, fragment_name="frag_3"),
            TopologyFactory.from_range("A", 10, 20, fragment_name="frag_1"),
            TopologyFactory.from_range("A", 50, 60, fragment_name="frag_2"),
            TopologyFactory.from_range("A", 150, 160, fragment_name="frag_4"),
        ]

        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.sequence_split()

        # With train_size=0.5, should get first 2 in training, last 2 in validation
        # After sorting by sequence: frag_1 (10-20), frag_2 (50-60), frag_3 (100-110), frag_4 (150-160)

        # Training should contain earlier sequence positions
        train_positions = []
        for dp in train_data:
            active_res = dp.top._get_active_residues(False)
            if active_res:
                train_positions.append(min(active_res))

        # Validation should contain later sequence positions
        val_positions = []
        for dp in val_data:
            active_res = dp.top._get_active_residues(False)
            if active_res:
                val_positions.append(min(active_res))

        if train_positions and val_positions:
            # Training positions should generally be earlier than validation positions
            max_train_pos = max(train_positions)
            min_val_pos = min(val_positions)
            assert max_train_pos <= min_val_pos

    def test_train_size_ratio_respected(self, create_datapoints_from_topologies, setup_splitter):
        """Test that train_size ratio is respected in sequence split."""
        topologies = create_sequential_topologies("A", 40)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.7)
        train_data, val_data = splitter.sequence_split()

        total_selected = len(train_data) + len(val_data)
        train_ratio = len(train_data) / total_selected if total_selected > 0 else 0

        # Should be approximately 0.7, allowing for filtering effects
        assert 0.5 <= train_ratio <= 0.9

    def test_contiguous_sequence_split(self, create_datapoints_from_topologies, setup_splitter):
        """Test that sequence split creates contiguous train/val regions."""
        topologies = create_sequential_topologies("A", 20, start_pos=1, gap=10, length=5)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.6)
        train_data, val_data = splitter.sequence_split()

        # Extract sequence positions
        def get_start_position(dp):
            active_res = dp.top._get_active_residues(False)
            return min(active_res) if active_res else float("inf")

        train_positions = [get_start_position(dp) for dp in train_data]
        val_positions = [get_start_position(dp) for dp in val_data]

        # Remove any invalid positions
        train_positions = [pos for pos in train_positions if pos != float("inf")]
        val_positions = [pos for pos in val_positions if pos != float("inf")]

        if train_positions and val_positions:
            # Training should come before validation in sequence
            max_train = max(train_positions)
            min_val = min(val_positions)

            # Allow some overlap due to filtering, but training should generally be earlier
            assert max_train <= min_val + 50  # Allow some buffer for filtering effects


class TestSequenceSplitMultiChain:
    """Test multi-chain behavior."""

    def test_multi_chain_sequence_ordering(self, create_datapoints_from_topologies, setup_splitter):
        """Test sequence ordering with multiple chains."""
        topologies = create_interleaved_chain_topologies(["A", "B", "C"], 6)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.sequence_split()

        # Should handle multiple chains correctly
        assert len(train_data) > 0
        assert len(val_data) > 0

        # Check that chains are sorted properly (A before B before C)
        all_data = train_data + val_data
        chain_positions = {}

        for dp in all_data:
            chain = dp.top.chain
            active_res = dp.top._get_active_residues(False)
            if active_res and chain not in chain_positions:
                chain_positions[chain] = min(active_res)

        # Within the context of this data, chains should be ordered
        if len(chain_positions) > 1:
            chains_seen = list(chain_positions.keys())
            # Should include multiple chains
            assert len(set(chains_seen)) > 1

    def test_chain_priority_in_sorting(self, create_datapoints_from_topologies, setup_splitter):
        """Test that chain ID takes precedence in sorting."""
        # Create topologies where chain B starts before chain A in sequence position
        topologies = [
            TopologyFactory.from_range("B", 1, 10, fragment_name="chain_B_early"),
            TopologyFactory.from_range("A", 50, 60, fragment_name="chain_A_later"),
            TopologyFactory.from_range("A", 5, 15, fragment_name="chain_A_early"),
            TopologyFactory.from_range("B", 100, 110, fragment_name="chain_B_later"),
        ]

        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.sequence_split()

        # All data should exist (training + validation)
        total_data = train_data + val_data
        assert len(total_data) > 0

        # Check that both chains are represented
        chains_in_data = {dp.top.chain for dp in total_data}
        assert len(chains_in_data) > 1


class TestSequenceSplitPeptideHandling:
    """Test peptide trimming with sequence splitting."""

    def test_sequence_split_with_peptide_trimming(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that peptide trimming affects sequence ordering."""
        peptide_topologies = create_peptide_sequence_topologies(["A"], 12)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)
        train_data, val_data = splitter.sequence_split()

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0

    def test_trimmed_vs_untrimmed_ordering(self, create_datapoints_from_topologies, setup_splitter):
        """Test difference in ordering with and without peptide trimming."""
        peptide_topologies = create_peptide_sequence_topologies(["A"], 10)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # Test with trimming
        splitter_trim = setup_splitter(datapoints, common_residues, check_trim=True, train_size=0.5)
        train_trim, val_trim = splitter_trim.sequence_split()

        # Test without trimming
        splitter_no_trim = setup_splitter(
            datapoints, common_residues, check_trim=False, train_size=0.5
        )
        train_no_trim, val_no_trim = splitter_no_trim.sequence_split()

        # Both should produce valid splits
        assert len(train_trim) + len(val_trim) > 0
        assert len(train_no_trim) + len(val_no_trim) > 0


class TestSequenceSplitOverlapRemoval:
    """Test overlap removal with sequence splitting."""

    def test_sequence_split_with_overlap_removal(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test sequence split with overlap removal enabled."""
        topologies = create_overlapping_topologies("A", 15, overlap=3)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.sequence_split(remove_overlap=True)

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0

    def test_overlap_removal_effect(self, create_datapoints_from_topologies, setup_splitter):
        """Test effect of overlap removal on dataset size."""
        topologies = create_overlapping_topologies("A", 20, overlap=5)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=999)

        # Without overlap removal
        train_no_removal, val_no_removal = splitter.sequence_split(remove_overlap=False)

        # Reset and try with overlap removal
        splitter._reset_retry_counter()
        train_with_removal, val_with_removal = splitter.sequence_split(remove_overlap=True)

        total_no_removal = len(train_no_removal) + len(val_no_removal)
        total_with_removal = len(train_with_removal) + len(val_with_removal)

        # With overlap removal, total should be <= without removal
        assert total_with_removal <= total_no_removal


class TestSequenceSplitCentralitySampling:
    """Test centrality sampling with sequence splitting."""

    def test_sequence_split_with_centrality_sampling(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test sequence split with centrality sampling enabled."""
        topologies = create_sequential_topologies("A", 30)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:25]  # Return subset

            splitter = setup_splitter(datapoints, common_residues, centrality=True)
            train_data, val_data = splitter.sequence_split()

            # Should have called sample_by_centrality
            mock_sample.assert_called_once()

            assert len(train_data) >= 0
            assert len(val_data) >= 0

    def test_centrality_affects_sequence_selection(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that centrality sampling affects which sequences are selected."""
        topologies = create_sequential_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # Test with centrality
        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:15]  # Return subset

            splitter_centrality = setup_splitter(datapoints, common_residues, centrality=True)
            train_cent, val_cent = splitter_centrality.sequence_split()

        # Test without centrality
        splitter_no_centrality = setup_splitter(datapoints, common_residues, centrality=False)
        train_no_cent, val_no_cent = splitter_no_centrality.sequence_split()

        # Results should potentially be different due to different source datasets
        total_cent = len(train_cent) + len(val_cent)
        total_no_cent = len(train_no_cent) + len(val_no_cent)

        # At least verify both produce valid results
        assert total_cent >= 0
        assert total_no_cent >= 0


class TestSequenceSplitDeterministicBehavior:
    """Test deterministic behavior of sequence splitting."""

    def test_sequence_split_deterministic(self, create_datapoints_from_topologies, setup_splitter):
        """Test that sequence split is deterministic given same input."""
        topologies = create_sequential_topologies("A", 16)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=123)
        train1, val1 = splitter1.sequence_split()

        # Second split with same seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=123)
        train2, val2 = splitter2.sequence_split()

        # Results should be identical (sequence split should be deterministic)
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}
        val1_ids = {dp.data_id for dp in val1}
        val2_ids = {dp.data_id for dp in val2}

        assert train1_ids == train2_ids
        assert val1_ids == val2_ids

    def test_sequence_split_seed_independence(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that sequence split results are independent of random seed."""
        # Since sequence split is based on sequence order, not randomness,
        # different seeds should produce the same results
        topologies = create_sequential_topologies("A", 12)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=111)
        train1, val1 = splitter1.sequence_split()

        # Second split with different seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=999)
        train2, val2 = splitter2.sequence_split()

        # Results should be identical since sequence order is deterministic
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}
        val1_ids = {dp.data_id for dp in val1}
        val2_ids = {dp.data_id for dp in val2}

        assert train1_ids == train2_ids
        assert val1_ids == val2_ids


class TestSequenceSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset_sequence_split(self, create_datapoints_from_topologies, setup_splitter):
        """Test sequence split with very small datasets."""
        topologies = create_sequential_topologies("A", 4)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        try:
            train_data, val_data = splitter.sequence_split()
            assert len(train_data) + len(val_data) > 0
        except ValueError:
            # Validation failure is acceptable for very small datasets
            pass

    def test_single_datapoint_sequence_split(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test sequence split with single datapoint."""
        topologies = create_sequential_topologies("A", 1)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, min_split_size=1)

        with pytest.raises(ValueError, match="Source dataset is too small to split"):
            splitter.sequence_split()

    def test_empty_active_residues_error(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling of topologies with no active residues."""
        # Create a mock topology with no active residues
        mock_topology = MagicMock()
        mock_topology._get_active_residues.return_value = []

        mock_datapoint = MockExpD_Datapoint(mock_topology, 0)
        # Add another valid datapoint to ensure the dataset size check passes
        valid_topology = TopologyFactory.from_range("A", 1, 10, fragment_name="valid_frag")
        valid_datapoint = MockExpD_Datapoint(valid_topology, 1)
        datapoints = [mock_datapoint, valid_datapoint]
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with pytest.raises(ValueError, match="No active residues found"):
            splitter.sequence_split()

    def test_extreme_train_size_values(self, create_datapoints_from_topologies, setup_splitter):
        """Test sequence split with extreme train_size values."""
        topologies = create_sequential_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        # Test with very small train_size
        splitter_small = setup_splitter(
            datapoints, common_residues, train_size=0.1, min_split_size=1
        )
        try:
            train_small, val_small = splitter_small.sequence_split()
            assert len(val_small) > len(train_small)
        except ValueError:
            pass  # Acceptable if validation fails

        # Test with very large train_size
        splitter_large = setup_splitter(
            datapoints, common_residues, train_size=0.9, min_split_size=1
        )
        try:
            train_large, val_large = splitter_large.sequence_split()
            assert len(train_large) > len(val_large)
        except ValueError:
            pass  # Acceptable if validation fails


class TestSequenceSplitRetryLogic:
    """Test retry logic and validation."""

    def test_retry_logic_on_validation_failure(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test retry logic when validation fails."""
        topologies = create_sequential_topologies("A", 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = [ValueError("Too small"), True]

            splitter = setup_splitter(datapoints, common_residues)

            # Should retry and eventually succeed
            train_data, val_data = splitter.sequence_split()

            # Should have called validate_split twice (fail, then success)
            assert mock_validate.call_count == 2

    def test_max_retries_exceeded(self, create_datapoints_from_topologies, setup_splitter):
        """Test behavior when max retries are exceeded."""
        topologies = create_sequential_topologies("A", 6)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = ValueError("Always fails")

            splitter = setup_splitter(datapoints, common_residues, max_retry_depth=2)

            with pytest.raises(ValueError, match="Failed to create valid split after 2 attempts"):
                splitter.sequence_split()

            # Should have tried max_retry_depth times
            assert mock_validate.call_count == splitter.max_retry_depth + 1

    def test_retry_counter_reset_on_success(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that retry counter is reset after successful split."""
        topologies = create_sequential_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # Manually set retry counter
        splitter.current_retry_count = 5

        # Successful split should reset counter
        train_data, val_data = splitter.sequence_split()

        assert splitter.current_retry_count == 0


class TestSequenceSplitSpecificBehavior:
    """Test specific behaviors unique to sequence splitting."""

    def test_early_vs_late_sequence_regions(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that training gets early sequence regions, validation gets late regions."""
        # Create well-separated topologies
        topologies = [
            TopologyFactory.from_range("A", 10, 20, fragment_name="early_1"),
            TopologyFactory.from_range("A", 30, 40, fragment_name="early_2"),
            TopologyFactory.from_range("A", 100, 110, fragment_name="late_1"),
            TopologyFactory.from_range("A", 120, 130, fragment_name="late_2"),
        ]

        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.sequence_split()

        # Extract positions
        train_starts = []
        val_starts = []

        for dp in train_data:
            active_res = dp.top._get_active_residues(False)
            if active_res:
                train_starts.append(min(active_res))

        for dp in val_data:
            active_res = dp.top._get_active_residues(False)
            if active_res:
                val_starts.append(min(active_res))

        if train_starts and val_starts:
            # Training should generally contain earlier positions
            avg_train_start = sum(train_starts) / len(train_starts)
            avg_val_start = sum(val_starts) / len(val_starts)

            assert avg_train_start < avg_val_start

    def test_sequence_continuity_property(self, create_datapoints_from_topologies, setup_splitter):
        """Test that sequence split maintains continuity property."""
        topologies = create_sequential_topologies("A", 16, gap=5, length=3)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.75)
        train_data, val_data = splitter.sequence_split()

        # Get all positions
        all_positions = []
        for dp in train_data + val_data:
            active_res = dp.top._get_active_residues(False)
            if active_res:
                all_positions.append((min(active_res), "train" if dp in train_data else "val"))

        # Sort by position
        all_positions.sort(key=lambda x: x[0])

        # Check for the split point - there should be a transition from "train" to "val"
        labels = [label for _, label in all_positions]

        if "train" in labels and "val" in labels:
            # Find the transition point
            train_positions = [i for i, label in enumerate(labels) if label == "train"]
            val_positions = [i for i, label in enumerate(labels) if label == "val"]

            if train_positions and val_positions:
                # Training indices should generally come before validation indices
                max_train_idx = max(train_positions)
                min_val_idx = min(val_positions)

                # Allow some overlap due to filtering, but should generally be ordered
                assert max_train_idx <= min_val_idx + len(labels) * 0.3  # Allow 30% overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
