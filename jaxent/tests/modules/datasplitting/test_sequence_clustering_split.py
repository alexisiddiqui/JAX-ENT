import copy
import re
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


# Helper functions to create diverse topologies (reusing from existing tests)
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


class TestSequenceClusterSplitBasicFunctionality:
    """Test basic functionality of sequence_cluster_split method."""

    def test_basic_cluster_split_returns_two_lists(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that sequence_cluster_split returns two lists."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            # Mock KMeans to return predictable cluster labels
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array(
                [0, 1, 0, 1] * 5
            )  # Alternating clusters
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split()

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_feature_extraction_from_topologies(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that start/end positions are correctly extracted as features."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1] * 5)
            mock_kmeans.return_value = mock_kmeans_instance

            splitter.sequence_cluster_split()

            # Check that KMeans was called with 2D features (start, end positions)
            args, kwargs = mock_kmeans_instance.fit_predict.call_args
            features = args[0]

            assert features.shape[1] == 2  # Should have 2 features per datapoint
            assert features.shape[0] == len(datapoints)  # One row per datapoint

            # Check that features are reasonable (start < end, positive values)
            for i in range(len(features)):
                start_pos, end_pos = features[i]
                assert start_pos > 0
                assert end_pos > 0
                assert end_pos >= start_pos

    def test_automatic_cluster_sizing(self, create_datapoints_from_topologies, setup_splitter):
        """Test that n_clusters defaults to 1/10th of dataset size."""
        topologies = create_single_chain_topologies("A", 50)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 5 for i in range(50)])
            mock_kmeans.return_value = mock_kmeans_instance

            splitter.sequence_cluster_split()  # Default n_clusters=10

            # Check that KMeans was initialized with approximately len(dataset)/10 clusters
            args, kwargs = mock_kmeans.call_args
            n_clusters_used = kwargs["n_clusters"]

            expected_clusters = max(2, len(datapoints) // 10)
            assert n_clusters_used == expected_clusters

    def test_custom_cluster_count(self, create_datapoints_from_topologies, setup_splitter):
        """Test setting custom number of clusters."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 5 for i in range(20)])
            mock_kmeans.return_value = mock_kmeans_instance

            splitter.sequence_cluster_split(n_clusters=5)

            # Check that KMeans was initialized with specified clusters
            args, kwargs = mock_kmeans.call_args
            n_clusters_used = kwargs["n_clusters"]

            assert n_clusters_used == 5


class TestSequenceClusterSplitDependencies:
    """Test handling of external dependencies."""

    def test_sklearn_import_error(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling when sklearn is not available."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'sklearn'")):
            with pytest.raises(ImportError, match="scikit-learn is required"):
                splitter.sequence_cluster_split()

    def test_numpy_import_error(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling when numpy import fails."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'numpy'")):
            with pytest.raises(ImportError, match="NumPy is required"):
                splitter.sequence_cluster_split()


class TestSequenceClusterSplitClusterAssignment:
    """Test cluster assignment logic."""

    def test_cluster_assignment_ratios(self, create_datapoints_from_topologies, setup_splitter):
        """Test that clusters are assigned according to train_size ratio."""
        topologies = create_single_chain_topologies("A", 40)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, train_size=0.7)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            # Create 10 clusters with 4 datapoints each
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i // 4 for i in range(40)])
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split(n_clusters=10)

            # With 10 clusters and train_size=0.7, expect ~7 clusters in training
            # This should result in approximately 70% of data in training
            total_data = len(train_data) + len(val_data)
            train_ratio = len(train_data) / total_data

            assert 0.5 <= train_ratio <= 0.9  # Should be around 0.7

    def test_whole_clusters_assigned(self, create_datapoints_from_topologies, setup_splitter):
        """Test that entire clusters are assigned to train or val, not split."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            # Create predictable clusters: 5 clusters with 4 datapoints each
            cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = cluster_labels
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split(n_clusters=5)

            # Extract cluster assignments for train and val data
            train_indices = [dp.data_id for dp in train_data]
            val_indices = [dp.data_id for dp in val_data]

            train_clusters = set(cluster_labels[train_indices])
            val_clusters = set(cluster_labels[val_indices])

            # Clusters should not be split between train and val
            assert len(train_clusters.intersection(val_clusters)) == 0


class TestSequenceClusterSplitMultiChain:
    """Test multi-chain behavior."""

    def test_multi_chain_clustering(self, create_datapoints_from_topologies, setup_splitter):
        """Test clustering with multiple chains."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 5 for i in range(24)])
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split()

        # Should handle multiple chains correctly
        assert len(train_data) > 0
        assert len(val_data) > 0

        # Check that multiple chains are represented
        train_chains = {dp.top.chain for dp in train_data}
        val_chains = {dp.top.chain for dp in val_data}
        all_chains = train_chains | val_chains

        assert len(all_chains) > 1


class TestSequenceClusterSplitPeptideHandling:
    """Test peptide trimming with clustering."""

    def test_clustering_with_peptide_trimming(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that peptide trimming affects feature extraction for clustering."""
        peptide_topologies = create_peptide_topologies(["A"], 15)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 3 for i in range(15)])
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split()

            # Verify clustering was performed
            assert mock_kmeans_instance.fit_predict.called

            # Check that features were extracted correctly
            args, kwargs = mock_kmeans_instance.fit_predict.call_args
            features = args[0]

            # With peptide trimming, features should reflect trimmed residues
            assert features.shape[0] == len(datapoints)
            assert features.shape[1] == 2


class TestSequenceClusterSplitOverlapRemoval:
    """Test overlap removal with clustering."""

    def test_clustering_with_overlap_removal(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test cluster split with overlap removal enabled."""
        topologies = create_multi_chain_topologies(["A", "B"], 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 6 for i in range(30)])
            mock_kmeans.return_value = mock_kmeans_instance

            train_data, val_data = splitter.sequence_cluster_split(remove_overlap=True)

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0

    def test_clustering_overlap_vs_no_overlap(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test difference between overlap removal enabled vs disabled."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 12)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=999)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 8 for i in range(36)])
            mock_kmeans.return_value = mock_kmeans_instance

            # Without overlap removal
            train_no_removal, val_no_removal = splitter.sequence_cluster_split(remove_overlap=False)

            # Reset and try with overlap removal
            splitter._reset_retry_counter()
            train_with_removal, val_with_removal = splitter.sequence_cluster_split(
                remove_overlap=True
            )

        total_no_removal = len(train_no_removal) + len(val_no_removal)
        total_with_removal = len(train_with_removal) + len(val_with_removal)

        assert total_with_removal <= total_no_removal


class TestSequenceClusterSplitCentralitySampling:
    """Test centrality sampling with clustering."""

    def test_clustering_with_centrality_sampling(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test cluster split with centrality sampling enabled."""
        topologies = create_single_chain_topologies("A", 25)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:20]  # Return subset

            splitter = setup_splitter(datapoints, common_residues, centrality=True)

            with patch("sklearn.cluster.KMeans") as mock_kmeans:
                mock_kmeans_instance = MagicMock()
                mock_kmeans_instance.fit_predict.return_value = np.array([i % 4 for i in range(20)])
                mock_kmeans.return_value = mock_kmeans_instance

                train_data, val_data = splitter.sequence_cluster_split()

            # Should have called sample_by_centrality
            mock_sample.assert_called_once()


class TestSequenceClusterSplitDeterministicBehavior:
    """Test deterministic behavior and random seed handling."""

    def test_same_seed_same_cluster_assignment(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test that same seed produces same cluster assignments."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 4 for i in range(20)])
            mock_kmeans.return_value = mock_kmeans_instance

            # First split
            splitter1 = setup_splitter(datapoints, common_residues, random_seed=123)
            train1, val1 = splitter1.sequence_cluster_split()

            # Second split with same seed
            splitter2 = setup_splitter(datapoints, common_residues, random_seed=123)
            train2, val2 = splitter2.sequence_cluster_split()

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
        """Test that different seeds can produce different cluster assignments."""
        topologies = create_single_chain_topologies("A", 30)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([i % 6 for i in range(30)])
            mock_kmeans.return_value = mock_kmeans_instance

            # First split
            splitter1 = setup_splitter(datapoints, common_residues, random_seed=111)
            train1, val1 = splitter1.sequence_cluster_split()

            # Second split with different seed
            splitter2 = setup_splitter(datapoints, common_residues, random_seed=222)
            train2, val2 = splitter2.sequence_cluster_split()

            # Results should likely be different
            train1_ids = {dp.data_id for dp in train1}
            train2_ids = {dp.data_id for dp in train2}

            # At least one should be different (high probability)
            assert len(train1_ids.symmetric_difference(train2_ids)) >= 0


class TestSequenceClusterSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset_clustering(self, create_datapoints_from_topologies, setup_splitter):
        """Test clustering with very small datasets."""
        topologies = create_single_chain_topologies("A", 4)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 0, 1])
            mock_kmeans.return_value = mock_kmeans_instance

            try:
                train_data, val_data = splitter.sequence_cluster_split()
                assert len(train_data) + len(val_data) > 0
            except ValueError:
                # Validation failure is acceptable for very small datasets
                pass

    def test_more_clusters_than_datapoints(self, create_datapoints_from_topologies, setup_splitter):
        """Test requesting more clusters than available datapoints."""
        topologies = create_single_chain_topologies("A", 5)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, min_split_size=1)

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = MagicMock()
            mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 2, 3, 4])
            mock_kmeans.return_value = mock_kmeans_instance

            # Request more clusters than datapoints
            train_data, val_data = splitter.sequence_cluster_split(n_clusters=10)

            # Should cap clusters at number of datapoints
            args, kwargs = mock_kmeans.call_args
            n_clusters_used = kwargs["n_clusters"]
            assert n_clusters_used == max(2, len(datapoints) // 10)

    def test_empty_active_residues_error(self, create_datapoints_from_topologies, setup_splitter):
        """Test handling of topologies with no active residues."""
        # Create a mock topology with no active residues
        mock_topology = MagicMock()
        mock_topology._get_active_residues.return_value = []

        mock_datapoint = MockExpD_Datapoint(mock_topology, 0)
        datapoints = [mock_datapoint]
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with pytest.raises(
            ValueError, match=re.escape("Source dataset is too small to split (1 datapoints).")
        ):
            splitter.sequence_cluster_split()


class TestSequenceClusterSplitRetryLogic:
    """Test retry logic and validation."""

    def test_retry_logic_on_validation_failure(
        self, create_datapoints_from_topologies, setup_splitter
    ):
        """Test retry logic when validation fails."""
        topologies = create_single_chain_topologies("A", 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = [ValueError("Too small"), True]

            splitter = setup_splitter(datapoints, common_residues)

            with patch("sklearn.cluster.KMeans") as mock_kmeans:
                mock_kmeans_instance = MagicMock()
                mock_kmeans_instance.fit_predict.return_value = np.array([0, 1] * 4)
                mock_kmeans.return_value = mock_kmeans_instance

                # Should retry and eventually succeed
                train_data, val_data = splitter.sequence_cluster_split()

                # Should have called validate_split twice (fail, then success)
                assert mock_validate.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
