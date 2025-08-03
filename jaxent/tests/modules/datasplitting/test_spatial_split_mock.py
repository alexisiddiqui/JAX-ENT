import copy
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology.factory import TopologyFactory
from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons


# Mock ExpD_Datapoint for testing (reusing from existing tests)
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top, data_id=None):
        self.top = top
        self.key = f"mock_key_{data_id}" if data_id else "mock_key"
        self.data_id = data_id

    def extract_features(self):
        return np.array([1.0])  # Mock feature

    def __repr__(self):
        return f"MockExpD_Datapoint(id={self.data_id}, top={self.top})"


# Helper functions (reusing from existing tests)
def create_single_chain_topologies(chain="A", count=10):
    """Create diverse single-chain topologies with good separation."""
    topologies = []
    start = 1
    gap = 15
    length = 10

    for i in range(count):
        topologies.append(
            TopologyFactory.from_range(
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
                TopologyFactory.from_range(
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
                TopologyFactory.from_range(
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
                TopologyFactory.from_range(chain, 1, end_pos, fragment_name=f"common_{chain}")
            )

    return common_residues


@pytest.fixture
def create_datapoints_from_topologies():
    """Factory to create datapoints from topologies."""

    def _create(topologies):
        return [MockExpD_Datapoint(topo, i) for i, topo in enumerate(topologies)]

    return _create


@pytest.fixture(autouse=True)
def mock_distance_calculation(request):
    """Automatically mock the distance calculation for all tests."""

    def make_distance_matrix(topologies, **kwargs):
        """Create a realistic distance matrix for the given topologies."""
        n = len(topologies)
        if n == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))

        # Create a symmetric distance matrix
        distance_matrix = np.random.rand(n, n)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Standard deviation matrix (zeros for single-frame data)
        distance_std = np.zeros_like(distance_matrix)

        return distance_matrix, distance_std

    # Patch where the method is used (in split.py), not where it's defined
    patcher = patch(
        "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances",
        side_effect=make_distance_matrix,
    )
    patcher.start()
    request.addfinalizer(patcher.stop)


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
                    PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim)
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
            "exclude_selection": "",  # Override default for tests
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


@pytest.fixture
def mock_universe():
    """Create a mock MDAnalysis Universe for testing."""
    mock_universe = MagicMock()
    mock_universe.__class__.__name__ = "Universe"
    return mock_universe


class TestSpatialSplitBasicFunctionality:
    """Test basic functionality of spatial_split method."""

    def test_basic_spatial_split_returns_two_lists(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test that spatial_split returns two lists."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)
        train_data, val_data = splitter.spatial_split(mock_universe)

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_distance_calculation_called_correctly(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test that distance calculation is called with correct parameters."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # Create a new mock to check call parameters
        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            mock_distances.return_value = (distance_matrix, distance_std)

            splitter.spatial_split(
                mock_universe,
                include_selection="custom_protein",
                exclude_selection="",
                start=10,
                stop=100,
                step=5,
            )

            # Verify the distance calculation was called with correct parameters
            assert mock_distances.called, "Distance calculation should have been called"
            call_args = mock_distances.call_args
            kwargs = call_args[1]

            assert kwargs["universe"] == mock_universe
            assert kwargs["include_selection"] == "custom_protein"
            assert kwargs["exclude_selection"] == ""
            assert kwargs["start"] == 10
            assert kwargs["stop"] == 100
            assert kwargs["step"] == 5
            assert kwargs["check_trim"] == splitter.check_trim

    def test_center_selection_and_proximity_ordering(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test that center is selected and ordering is by proximity."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=123)

        # Use a custom mock to control the distance matrix
        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create predictable distance matrix
            n_topos = len(datapoints)
            distance_matrix = np.zeros((n_topos, n_topos))

            for i in range(n_topos):
                for j in range(n_topos):
                    if i != j:
                        distance_matrix[i, j] = abs(i - j)

            distance_std = np.zeros_like(distance_matrix)
            mock_distances.return_value = (distance_matrix, distance_std)

            with patch("random.randint") as mock_randint:
                mock_randint.return_value = 0  # Always select first topology as center

                train_data, val_data = splitter.spatial_split(mock_universe)

                # With train_size=0.6 and 10 datapoints, expect 6 training, 4 validation
                assert len(train_data) == 6
                assert len(val_data) == 4

                # Training data should contain the closest topologies to center (index 0)
                train_ids = sorted([dp.data_id for dp in train_data])
                expected_train_ids = [0, 1, 2, 3, 4, 5]
                assert train_ids == expected_train_ids

    def test_spatial_split_with_different_train_sizes(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test spatial split with different training set sizes."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        for train_size in [0.3, 0.5, 0.7, 0.8]:
            splitter = setup_splitter(datapoints, common_residues, train_size=train_size)
            train_data, val_data = splitter.spatial_split(mock_universe)

            expected_train_size = int(train_size * len(datapoints))
            expected_val_size = len(datapoints) - expected_train_size

            assert len(train_data) == expected_train_size
            assert len(val_data) == expected_val_size

    def test_numpy_import_error(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test handling when numpy import fails."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch("builtins.__import__", side_effect=ImportError("No module named 'numpy'")):
            with pytest.raises(ImportError, match="NumPy is required"):
                splitter.spatial_split(mock_universe)


class TestSpatialSplitDistanceCalculation:
    """Test distance calculation aspects of spatial split."""

    def test_distance_calculation_failure_handling(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test handling of distance calculation failures."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            mock_distances.side_effect = ValueError("Distance calculation failed")

            with pytest.raises(ValueError, match="Failed to compute spatial distances"):
                splitter.spatial_split(mock_universe)

    def test_distance_matrix_validation(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test that distance matrix is properly validated and used."""
        topologies = create_single_chain_topologies("A", 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create realistic distance matrix
            distance_matrix = np.array(
                [
                    [0.0, 5.2, 8.1, 12.3, 15.6, 20.1, 25.4, 30.2],
                    [5.2, 0.0, 3.9, 7.8, 11.1, 16.3, 21.2, 26.8],
                    [8.1, 3.9, 0.0, 4.5, 7.9, 12.8, 18.5, 23.1],
                    [12.3, 7.8, 4.5, 0.0, 5.2, 9.1, 14.6, 19.4],
                    [15.6, 11.1, 7.9, 5.2, 0.0, 6.3, 11.8, 16.2],
                    [20.1, 16.3, 12.8, 9.1, 6.3, 0.0, 7.5, 12.1],
                    [25.4, 21.2, 18.5, 14.6, 11.8, 7.5, 0.0, 8.9],
                    [30.2, 26.8, 23.1, 19.4, 16.2, 12.1, 8.9, 0.0],
                ]
            )
            distance_std = np.ones_like(distance_matrix) * 0.5

            mock_distances.return_value = (distance_matrix, distance_std)

            with patch("random.randint") as mock_randint:
                mock_randint.return_value = 2  # Select index 2 as center

                train_data, val_data = splitter.spatial_split(mock_universe)

                # With center at index 2, training should get the closest points
                train_ids = sorted([dp.data_id for dp in train_data])
                expected_train_ids = [1, 2, 3, 4]  # Based on distances from center 2
                assert train_ids == expected_train_ids

    def test_trajectory_parameters_passed_correctly(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test that trajectory analysis parameters are passed correctly."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            mock_distances.return_value = (distance_matrix, distance_std)

            splitter.spatial_split(
                mock_universe,
                include_selection="protein and name CA",
                exclude_selection="resname HOH",
                start=50,
                stop=200,
                step=10,
            )

            call_kwargs = mock_distances.call_args[1]
            assert call_kwargs["start"] == 50
            assert call_kwargs["stop"] == 200
            assert call_kwargs["step"] == 10
            assert call_kwargs["include_selection"] == "protein and name CA"
            assert call_kwargs["exclude_selection"] == "resname HOH"


class TestSpatialSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset_spatial_splitting(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
    ):
        """Test spatial splitting with very small datasets."""
        topologies = create_single_chain_topologies("A", 4)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        try:
            train_data, val_data = splitter.spatial_split(mock_universe)
            assert len(train_data) + len(val_data) > 0
        except ValueError:
            # Validation failure is acceptable for very small datasets
            pass

    def test_single_datapoint_error(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test error handling with single datapoint."""
        topologies = create_single_chain_topologies("A", 1)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with pytest.raises(
            ValueError, match=re.escape("Source dataset is too small to split (1 datapoints).")
        ):
            splitter.spatial_split(mock_universe)

    def test_invalid_distance_matrix_shape(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test handling of invalid distance matrix shapes."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Return wrong shape matrix
            wrong_shape_matrix = np.random.rand(5, 5)  # Wrong size for 10 topologies
            distance_std = np.zeros_like(wrong_shape_matrix)

            mock_distances.return_value = (wrong_shape_matrix, distance_std)

            # Should fail when trying to access center distances
            with pytest.raises(IndexError):
                splitter.spatial_split(mock_universe)

    def test_distance_calculation_with_nan_values(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe
    ):
        """Test handling of NaN values in distance matrix."""
        topologies = create_single_chain_topologies("A", 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.data.splitting.split.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix[0, 1] = np.nan  # Insert NaN value
            distance_matrix[1, 0] = np.nan  # Keep symmetry
            distance_std = np.zeros_like(distance_matrix)

            mock_distances.return_value = (distance_matrix, distance_std)

            # Should handle NaN values gracefully
            train_data, val_data = splitter.spatial_split(mock_universe)

            assert len(train_data) > 0
            assert len(val_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
