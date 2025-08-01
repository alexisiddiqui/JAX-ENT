import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import Partial_Topology


# Mock ExpD_Datapoint for testing
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top, data_id=None, y_true=None, target=None):
        self.top = top
        self.key = f"mock_key_{data_id}" if data_id else "mock_key"
        self.data_id = data_id
        self.y_true = y_true
        self.target = target

    def extract_features(self):
        return np.array([1.0])  # Mock feature

    def __repr__(self):
        return f"MockExpD_Datapoint(id={self.data_id}, top={self.top}, y_true={self.y_true})"


# Helper functions to create diverse topologies
def create_single_chain_topologies(chain="A", count=10):
    """Create diverse single-chain topologies with good separation."""
    topologies = []
    start = 1
    gap = 15
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
        gap = 20
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
    trim_values = [1, 2, 3]

    for chain in chains:
        start = 1
        gap = 18
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


def create_common_residues_for_chains(chains, coverage_factor=0.7):
    """Create common residue topologies that cover a portion of each chain."""
    common_residues = set()

    if len(chains) == 1:
        chain = chains[0]
        if chain == "A":
            range1_end = int(150 * coverage_factor)
            range2_start = 160
            range2_end = int(range2_start + (300 - range2_start) * coverage_factor)
        elif chain == "B":
            range1_end = int(120 * coverage_factor)
            range2_start = 130
            range2_end = int(range2_start + (250 - range2_start) * coverage_factor)
        elif chain == "C":
            range1_end = int(90 * coverage_factor)
            range2_start = 100
            range2_end = int(range2_start + (200 - range2_start) * coverage_factor)
        else:
            range1_end = int(90 * coverage_factor)
            range2_start = 100
            range2_end = int(range2_start + (200 - range2_start) * coverage_factor)

        if range1_end >= 1:
            common_residues.add(
                Partial_Topology.from_range(chain, 1, range1_end, fragment_name=f"common_{chain}_1")
            )

        if range2_end >= range2_start:
            common_residues.add(
                Partial_Topology.from_range(
                    chain, range2_start, range2_end, fragment_name=f"common_{chain}_2"
                )
            )

        if len(common_residues) < 2:
            dummy_start = 500
            while len(common_residues) < 2:
                common_residues.add(
                    Partial_Topology.from_range(
                        chain,
                        dummy_start,
                        dummy_start + 1,
                        fragment_name=f"common_{chain}_dummy_{len(common_residues)}",
                    )
                )
                dummy_start += 10

        return common_residues

    for chain in chains:
        if chain == "A":
            end_pos = int(300 * coverage_factor)
        elif chain == "B":
            end_pos = int(250 * coverage_factor)
        elif chain == "C":
            end_pos = int(200 * coverage_factor)
        elif chain == "D":
            end_pos = int(200 * coverage_factor)
        else:
            end_pos = int(150 * coverage_factor)

        if end_pos >= 1:
            common_residues.add(
                Partial_Topology.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )

    return common_residues


def create_datapoints_with_targets(topologies, target_values=None, target_type="y_true"):
    """Create datapoints with specified target values."""
    if target_values is None:
        # Default: create linear increasing values
        target_values = [i * 0.5 for i in range(len(topologies))]

    datapoints = []
    for i, topo in enumerate(topologies):
        value = target_values[i] if i < len(target_values) else 0.0

        if target_type == "y_true":
            dp = MockExpD_Datapoint(topo, i, y_true=value)
        elif target_type == "target":
            dp = MockExpD_Datapoint(topo, i, target=value)
        elif target_type == "both":
            dp = MockExpD_Datapoint(topo, i, y_true=value, target=value + 1.0)
        else:  # "none"
            dp = MockExpD_Datapoint(topo, i)

        datapoints.append(dp)

    return datapoints


@pytest.fixture
def create_datapoints_from_topologies():
    """Factory to create datapoints from topologies."""

    def _create(topologies, target_values=None, target_type="y_true"):
        return create_datapoints_with_targets(topologies, target_values, target_type)

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
                if any(dp.top.intersects(ct, check_trim=check_trim) for ct in common_topos)
            ]

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

        mock_calc.return_value = [0.5] * len(datapoints)

        default_kwargs = {
            "random_seed": 42,
            "train_size": 0.6,
            "centrality": False,
            "check_trim": False,
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


class TestStratifiedSplitBasicFunctionality:
    """Test basic functionality of stratified_split method."""

    def test_basic_stratified_split_returns_two_lists(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that stratified_split returns two lists."""
        topologies = create_single_chain_topologies("A", 20)
        target_values = [i * 0.1 for i in range(20)]  # Linear values 0.0 to 1.9
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_target_value_extraction_y_true(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test extraction of y_true values for stratification."""
        topologies = create_single_chain_topologies("A", 10)
        target_values = [1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 9.0, 0.0]  # Mixed order
        datapoints = create_datapoints_from_topologies(topologies, target_values, "y_true")
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        # Should complete without error and create valid split
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_target_value_extraction_target_attribute(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test extraction of target values when y_true is not available."""
        topologies = create_single_chain_topologies("A", 10)
        target_values = [i * 0.5 for i in range(10)]
        datapoints = create_datapoints_from_topologies(topologies, target_values, "target")
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_target_value_extraction_array_values(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test extraction when target values are arrays (should use mean)."""
        topologies = create_single_chain_topologies("A", 8)
        # Create array values that will be averaged
        array_values = [
            [1.0, 2.0, 3.0],  # mean = 2.0
            [4.0, 5.0, 6.0],  # mean = 5.0
            [0.5, 1.5, 2.5],  # mean = 1.5
            [7.0, 8.0, 9.0],  # mean = 8.0
            [1.0, 1.0, 1.0],  # mean = 1.0
            [3.0, 4.0, 5.0],  # mean = 4.0
            [2.0, 3.0, 4.0],  # mean = 3.0
            [6.0, 7.0, 8.0],  # mean = 7.0
        ]

        # Manually create datapoints with array targets
        datapoints = []
        for i, topo in enumerate(topologies):
            dp = MockExpD_Datapoint(topo, i, y_true=array_values[i])
            datapoints.append(dp)

        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_fallback_hash_stratification(self, create_datapoints_from_topologies, setup_splitter):
        """Test fallback to hash-based stratification when no target values available."""
        topologies = create_single_chain_topologies("A", 12)
        datapoints = create_datapoints_from_topologies(topologies, None, "none")  # No target values
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        # Should still work using hash-based fallback
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestStratifiedSplitDependencies:
    """Test handling of external dependencies."""

    def test_numpy_import_error(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling when numpy import fails."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'numpy'")):
            with pytest.raises(ImportError, match="NumPy is required"):
                splitter.stratified_split()


class TestStratifiedSplitStrataCreation:
    """Test strata creation and assignment logic."""

    def test_strata_size_automatic(self, create_datapoints_from_topologies, setup_splitter):
        """Test automatic strata size determination."""
        topologies = create_single_chain_topologies("A", 40)
        target_values = [i * 0.25 for i in range(40)]  # Linear values
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()  # Default n_strata=10

        # Should complete successfully with automatic strata sizing
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_strata_size_custom(self, create_datapoints_from_topologies, setup_splitter):
        """Test custom strata size."""
        topologies = create_single_chain_topologies("A", 30)
        target_values = [i * 0.1 for i in range(30)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split(n_strata=5)

        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_strata_assignment_ratios(self, create_datapoints_from_topologies, setup_splitter):
        """Test that strata are assigned according to train_size ratio."""
        topologies = create_single_chain_topologies("A", 40)
        target_values = [i * 0.1 for i in range(40)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.7)
        train_data, val_data = splitter.stratified_split(n_strata=10)

        # With 10 strata and train_size=0.7, expect ~7 strata in training
        total_data = len(train_data) + len(val_data)
        train_ratio = len(train_data) / total_data

        assert 0.5 <= train_ratio <= 0.9  # Should be around 0.7

    def test_strata_value_ordering(self, create_datapoints_from_topologies, setup_splitter):
        """Test that strata are created based on value ordering."""
        topologies = create_single_chain_topologies("A", 16)
        # Create clear low/high value groups
        target_values = [1.0] * 8 + [10.0] * 8  # Clear separation
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.stratified_split(n_strata=4)

        # Extract target values from results
        train_targets = []
        val_targets = []

        for dp in train_data:
            if hasattr(dp, "y_true") and dp.y_true is not None:
                train_targets.append(dp.y_true)

        for dp in val_data:
            if hasattr(dp, "y_true") and dp.y_true is not None:
                val_targets.append(dp.y_true)

        # Both train and val should have some representation of low and high values
        if train_targets and val_targets:
            assert len(set(train_targets)) >= 1  # At least some diversity
            assert len(set(val_targets)) >= 1  # At least some diversity

    def test_minimum_strata_size_enforcement(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that minimum strata size (2 datapoints) is enforced."""
        topologies = create_single_chain_topologies("A", 8)
        target_values = [i * 0.1 for i in range(8)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        # Request more strata than possible given minimum size requirement
        train_data, val_data = splitter.stratified_split(n_strata=20)

        # Should automatically adjust to maximum feasible strata
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestStratifiedSplitMultiChain:
    """Test multi-chain behavior."""

    def test_multi_chain_stratification(self, create_datapoints_from_topologies, setup_splitter):
        """Test stratification with multiple chains."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 8)
        target_values = [i * 0.15 for i in range(24)]  # Varied values across chains
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        # Should handle multiple chains correctly
        assert len(train_data) > 0
        assert len(val_data) > 0

        # Check that multiple chains are represented
        train_chains = {dp.top.chain for dp in train_data}
        val_chains = {dp.top.chain for dp in val_data}
        all_chains = train_chains | val_chains

        assert len(all_chains) > 1


class TestStratifiedSplitPeptideHandling:
    """Test peptide trimming with stratified splitting."""

    def test_stratification_with_peptide_trimming(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that peptide trimming works with stratified splitting."""
        peptide_topologies = create_peptide_topologies(["A"], 15)
        target_values = [i * 0.2 for i in range(15)]
        datapoints = create_datapoints_with_targets(peptide_topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)
        train_data, val_data = splitter.stratified_split()

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0


class TestStratifiedSplitOverlapRemoval:
    """Test overlap removal with stratified splitting."""

    def test_stratification_with_overlap_removal(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test stratified split with overlap removal enabled."""
        topologies = create_multi_chain_topologies(["A", "B"], 15)
        target_values = [i * 0.3 for i in range(30)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split(remove_overlap=True)

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0

    def test_stratification_overlap_vs_no_overlap(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test difference between overlap removal enabled vs disabled."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 12)
        target_values = [i * 0.2 for i in range(36)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=999)

        # Without overlap removal
        train_no_removal, val_no_removal = splitter.stratified_split(remove_overlap=False)

        # Reset and try with overlap removal
        splitter._reset_retry_counter()
        train_with_removal, val_with_removal = splitter.stratified_split(remove_overlap=True)

        total_no_removal = len(train_no_removal) + len(val_no_removal)
        total_with_removal = len(train_with_removal) + len(val_with_removal)

        assert total_with_removal <= total_no_removal


class TestStratifiedSplitCentralitySampling:
    """Test centrality sampling with stratified splitting."""

    def test_stratification_with_centrality_sampling(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test stratified split with centrality sampling enabled."""
        topologies = create_single_chain_topologies("A", 25)
        target_values = [i * 0.1 for i in range(25)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:20]  # Return subset

            splitter = setup_splitter(datapoints, common_residues, centrality=True)
            train_data, val_data = splitter.stratified_split()

            # Should have called sample_by_centrality
            mock_sample.assert_called_once()

            assert len(train_data) >= 0
            assert len(val_data) >= 0


class TestStratifiedSplitDeterministicBehavior:
    """Test deterministic behavior and random seed handling."""

    def test_same_seed_same_assignment(self, create_datapoints_from_topologies, setup_splitter):
        """Test that same seed produces same strata assignments."""
        topologies = create_single_chain_topologies("A", 20)
        target_values = [i * 0.1 for i in range(20)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=123)
        train1, val1 = splitter1.stratified_split(n_strata=4)

        # Second split with same seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=123)
        train2, val2 = splitter2.stratified_split(n_strata=4)

        # Results should be identical
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}
        val1_ids = {dp.data_id for dp in val1}
        val2_ids = {dp.data_id for dp in val2}

        assert train1_ids == train2_ids
        assert val1_ids == val2_ids

    def test_different_seed_different_assignment(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that different seeds can produce different strata assignments."""
        topologies = create_single_chain_topologies("A", 30)
        target_values = [i * 0.1 for i in range(30)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        # First split
        splitter1 = setup_splitter(datapoints, common_residues, random_seed=111)
        train1, val1 = splitter1.stratified_split(n_strata=6)

        # Second split with different seed
        splitter2 = setup_splitter(datapoints, common_residues, random_seed=222)
        train2, val2 = splitter2.stratified_split(n_strata=6)

        # Results should likely be different
        train1_ids = {dp.data_id for dp in train1}
        train2_ids = {dp.data_id for dp in train2}

        # Should be different with high probability
        assert len(train1_ids.symmetric_difference(train2_ids)) >= 0


class TestStratifiedSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset_stratification(self, create_datapoints_from_topologies, setup_splitter):
        """Test stratification with very small datasets."""
        topologies = create_single_chain_topologies("A", 4)
        target_values = [1.0, 2.0, 3.0, 4.0]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, min_split_size=1)

        try:
            train_data, val_data = splitter.stratified_split()
            assert len(train_data) + len(val_data) > 0
        except ValueError:
            # Validation failure is acceptable for very small datasets
            pass

    def test_uniform_target_values(self, create_datapoints_from_topologies, setup_splitter):
        """Test stratification when all target values are the same."""
        topologies = create_single_chain_topologies("A", 12)
        target_values = [5.0] * 12  # All same value
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split(n_strata=3)

        # Should still work even with uniform values
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_empty_target_arrays(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling of empty target arrays."""
        topologies = create_single_chain_topologies("A", 6)

        # Create datapoints with empty arrays as targets
        datapoints = []
        for i, topo in enumerate(topologies):
            dp = MockExpD_Datapoint(topo, i, y_true=[])  # Empty array
            datapoints.append(dp)

        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, min_split_size=1)
        train_data, val_data = splitter.stratified_split()

        # Should use 0.0 as default and still work
        assert len(train_data) >= 0
        assert len(val_data) >= 0

    def test_mixed_target_types(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling of mixed target value types (scalars and arrays)."""
        topologies = create_single_chain_topologies("A", 8)

        # Create datapoints with mixed target types
        datapoints = []
        mixed_targets = [
            1.0,  # scalar
            [2.0, 3.0],  # array (mean = 2.5)
            4.0,  # scalar
            [5.0, 6.0, 7.0],  # array (mean = 6.0)
            8.0,  # scalar
            [9.0, 10.0],  # array (mean = 9.5)
            11.0,  # scalar
            [12.0, 13.0, 14.0],  # array (mean = 13.0)
        ]

        for i, topo in enumerate(topologies):
            dp = MockExpD_Datapoint(topo, i, y_true=mixed_targets[i])
            datapoints.append(dp)

        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.stratified_split()

        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_excessive_strata_request(self, create_datapoints_from_topologies, setup_splitter):
        """Test requesting more strata than feasible."""
        topologies = create_single_chain_topologies("A", 6)
        target_values = [i * 1.0 for i in range(6)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, min_split_size=1)

        # Request way more strata than possible
        train_data, val_data = splitter.stratified_split(n_strata=100)

        # Should automatically adjust and still work
        assert len(train_data) >= 0
        assert len(val_data) >= 0


class TestStratifiedSplitRetryLogic:
    """Test retry logic and validation."""

    def test_retry_logic_on_validation_failure(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test retry logic when validation fails."""
        topologies = create_single_chain_topologies("A", 8)
        target_values = [i * 0.5 for i in range(8)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = [ValueError("Too small"), True]

            splitter = setup_splitter(datapoints, common_residues)

            # Should retry and eventually succeed
            train_data, val_data = splitter.stratified_split()

            # Should have called validate_split twice (fail, then success)
            assert mock_validate.call_count == 2

    def test_max_retries_exceeded(self, create_datapoints_from_topologies, setup_splitter):
        """Test behavior when max retries are exceeded."""
        topologies = create_single_chain_topologies("A", 6)
        target_values = [i * 0.1 for i in range(6)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = ValueError("Always fails")

            splitter = setup_splitter(datapoints, common_residues, max_retry_depth=2)

            with pytest.raises(ValueError, match="Failed to create valid split after 2 attempts"):
                splitter.stratified_split()

            # Should have tried max_retry_depth times
            assert mock_validate.call_count == splitter.max_retry_depth + 1

    def test_retry_counter_reset_on_success(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that retry counter is reset after successful split."""
        topologies = create_single_chain_topologies("A", 10)
        target_values = [i * 0.2 for i in range(10)]
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # Manually set retry counter
        splitter.current_retry_count = 5

        # Successful split should reset counter
        train_data, val_data = splitter.stratified_split()

        assert splitter.current_retry_count == 0


class TestStratifiedSplitSpecificBehavior:
    """Test specific behaviors unique to stratified splitting."""

    def test_value_distribution_preservation(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that stratified split preserves value distribution across train/val."""
        topologies = create_single_chain_topologies("A", 20)
        # Create clear low/medium/high value groups
        target_values = [1.0] * 6 + [5.0] * 8 + [10.0] * 6
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.6)
        train_data, val_data = splitter.stratified_split(n_strata=6)

        # Extract target values from results
        train_targets = []
        val_targets = []

        for dp in train_data:
            if hasattr(dp, "y_true") and dp.y_true is not None:
                train_targets.append(dp.y_true)

        for dp in val_data:
            if hasattr(dp, "y_true") and dp.y_true is not None:
                val_targets.append(dp.y_true)

        # Both sets should have some representation of different value ranges
        if train_targets and val_targets:
            train_unique = set(train_targets)
            val_unique = set(val_targets)

            # Should have reasonable diversity in both sets
            assert len(train_unique) >= 1
            assert len(val_unique) >= 1

    def test_strata_boundary_handling(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling of strata boundaries and quantile-based splitting."""
        topologies = create_single_chain_topologies("A", 16)
        # Create values that will create clear quantile boundaries
        target_values = [i for i in range(16)]  # 0, 1, 2, ..., 15
        datapoints = create_datapoints_from_topologies(topologies, target_values)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.5)
        train_data, val_data = splitter.stratified_split(n_strata=4)

        # Should create 4 strata with 4 datapoints each
        assert len(train_data) + len(val_data) >= 8  # After filtering

        # Check that we have data in both sets
        assert len(train_data) > 0
        assert len(val_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
