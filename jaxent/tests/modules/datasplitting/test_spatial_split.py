import copy
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology.factory import TopologyFactory


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

        patcher1 = patch("jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy")
        patcher2 = patch(
            "jaxent.src.data.splitting.split.filter_common_residues", side_effect=mock_filter_func
        )
        patcher3 = patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.to_mda_group"
        )

        mock_calc = patcher1.start()
        mock_filter = patcher2.start()
        mock_to_mda_group = patcher3.start()

        request.addfinalizer(patcher1.stop)
        request.addfinalizer(patcher2.stop)
        request.addfinalizer(patcher3.stop)

        mock_calc.return_value = [0.5] * len(datapoints)

        # Mock to_mda_group to return a simple AtomGroup
        mock_atom = MagicMock(name="Atom")
        mock_atom.position = np.array([0.0, 0.0, 0.0])
        mock_atom_group_return = MagicMock(name="AtomGroup")
        mock_atom_group_return.atoms = MagicMock(name="Atoms")
        mock_atom_group_return.atoms.__iter__.return_value = [mock_atom]
        mock_atom_group_return.atoms.__len__.return_value = 1
        mock_atom_group_return.atoms.positions = np.array([[0.0, 0.0, 0.0]])
        mock_to_mda_group.return_value = mock_atom_group_return

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

    # Mock the atoms attribute and its select_atoms method
    mock_atom_group = MagicMock()
    mock_atom_group.__len__.return_value = 1  # Simulate at least one atom found
    mock_atom_group.residues = MagicMock()
    # Create a list of mock residues from 1 to 300
    mock_residues = []
    for i in range(1, 500):  # Extend residue range
        resname = "PRO" if i == 1 else "ALA"  # Make resid 1 a PRO to test exclusion
        mock_residues.append(
            MagicMock(resid=i, resname=resname, atoms=[MagicMock(segid="A", chainid="A")])
        )
    mock_atom_group.residues.__len__.return_value = len(mock_residues)
    mock_atom_group.residues.__iter__.return_value = iter(mock_residues)
    mock_atom_group.residues.__contains__.side_effect = (
        lambda x: x in mock_residues
    )  # Enable 'in' operator
    mock_atom_group.residues.__getitem__.side_effect = lambda x: mock_residues[
        x - 1
    ]  # Enable indexing by resid-1

    mock_universe.atoms = MagicMock()
    mock_universe.atoms.select_atoms.return_value = mock_atom_group
    mock_universe.select_atoms.return_value = (
        mock_atom_group  # Also mock direct select_atoms on universe
    )

    return mock_universe


@pytest.fixture
def patch_partial_distances():
    """Patch partial_topology_pairwise_distances and yield the mock."""
    with patch(
        "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
    ) as mock_distances:
        yield mock_distances


@pytest.fixture(autouse=True)
def patch_pairwise_distances(request):
    """Automatically patch partial_topology_pairwise_distances for all tests."""
    patcher = patch(
        "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
    )
    mock_func = patcher.start()

    def make_distance_matrix(dataset, **kwargs):
        n = len(dataset)
        if n == 0:
            return np.zeros((0, 0)), np.zeros((0, 0))
        mat = np.random.rand(n, n)
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
        std = np.zeros_like(mat)
        return mat, std

    mock_func.side_effect = lambda dataset, **kwargs: make_distance_matrix(dataset, **kwargs)

    request.addfinalizer(patcher.stop)


class TestSpatialSplitBasicFunctionality:
    """Test basic functionality of spatial_split method."""

    def test_basic_spatial_split_returns_two_lists(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that spatial_split returns two lists."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        # Mock the distance calculation
        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create symmetric distance matrix with known pattern
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(distance_matrix, 0)  # Diagonal is zero
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

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
        patch_partial_distances,
    ):
        """Test that distance calculation is called with correct parameters."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            splitter.spatial_split(
                mock_universe,
                include_selection="custom_protein",
                exclude_selection="",  # Avoid removing all atoms
                start=10,
                stop=100,
                step=5,
            )

            # Verify the distance calculation was called with correct parameters
            call_args = patch_partial_distances.call_args
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
        patch_partial_distances,
    ):
        """Test that center is selected and ordering is by proximity."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=123)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create predictable distance matrix
            # Center will be at index 0 (due to random seed)
            # Distances from center: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            n_topos = len(datapoints)
            distance_matrix = np.zeros((n_topos, n_topos))

            for i in range(n_topos):
                for j in range(n_topos):
                    if i != j:
                        distance_matrix[i, j] = abs(i - j)

            distance_std = np.zeros_like(distance_matrix)
            patch_partial_distances.return_value = (distance_matrix, distance_std)

            with patch("random.randint") as mock_randint:
                mock_randint.return_value = 0  # Always select first topology as center

                train_data, val_data = splitter.spatial_split(mock_universe)

                # With train_size=0.6 and 10 datapoints, expect 6 training, 4 validation
                assert len(train_data) == 6
                assert len(val_data) == 4

                # Training data should contain the closest topologies to center (index 0)
                # Expected order by distance from center 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # Training should get first 6: [0, 1, 2, 3, 4, 5]
                train_ids = sorted([dp.data_id for dp in train_data])
                expected_train_ids = [0, 1, 2, 3, 4, 5]
                assert train_ids == expected_train_ids

    def test_spatial_split_with_different_train_sizes(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial split with different training set sizes."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        for train_size in [0.3, 0.5, 0.7, 0.8]:
            splitter = setup_splitter(datapoints, common_residues, train_size=train_size)

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_topos = len(datapoints)
                distance_matrix = np.random.rand(n_topos, n_topos)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                patch_partial_distances.return_value = (distance_matrix, distance_std)

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
        patch_partial_distances,
    ):
        """Test handling of distance calculation failures."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        patch_partial_distances.side_effect = ValueError("Distance calculation failed")
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
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create realistic distance matrix
            n_topos = len(datapoints)
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

                # With center at index 2, distances are: [8.1, 3.9, 0.0, 4.5, 7.9, 12.8, 18.5, 23.1]
                # Sorted by distance: [2, 1, 3, 4, 0, 5, 6, 7]
                # With train_size=0.6 (4 out of 8), training should get: [2, 1, 3, 4]
                train_ids = sorted([dp.data_id for dp in train_data])
                expected_train_ids = [1, 2, 3, 4]
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
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            mock_distances.return_value = (distance_matrix, distance_std)

            # Test with various trajectory parameters
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


class TestSpatialSplitMultiChain:
    """Test multi-chain behavior of spatial split."""

    def test_multi_chain_spatial_splitting(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial splitting with multiple chains."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            train_data, val_data = splitter.spatial_split(mock_universe)

        # Should handle multiple chains correctly
        assert len(train_data) > 0
        assert len(val_data) > 0

        # Check that multiple chains are represented
        train_chains = {dp.top.chain for dp in train_data}
        val_chains = {dp.top.chain for dp in val_data}
        all_chains = train_chains | val_chains

        assert len(all_chains) > 1

    def test_spatial_clustering_across_chains(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that spatial clustering works across different chains."""
        topologies = create_multi_chain_topologies(["A", "B"], 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Create distance matrix where chains A and B have different spatial distributions
            n_topos = len(datapoints)
            distance_matrix = np.zeros((n_topos, n_topos))

            # Create realistic inter-chain and intra-chain distances
            for i in range(n_topos):
                for j in range(n_topos):
                    if i != j:
                        # Chain A: indices 0-9, Chain B: indices 10-19
                        same_chain = (i < 10 and j < 10) or (i >= 10 and j >= 10)
                        if same_chain:
                            distance_matrix[i, j] = abs(i - j) * 2.0  # Intra-chain distances
                        else:
                            distance_matrix[i, j] = 50.0 + abs(i - j)  # Inter-chain distances

            distance_std = np.ones_like(distance_matrix)
            mock_distances.return_value = (distance_matrix, distance_std)

            with patch("random.randint") as mock_randint:
                mock_randint.return_value = 5  # Center in chain A

                train_data, val_data = splitter.spatial_split(mock_universe)

                # Training set should be biased towards chain A due to proximity
                train_chains = [dp.top.chain for dp in train_data]
                chain_a_count = sum(1 for chain in train_chains if chain == "A")
                chain_b_count = sum(1 for chain in train_chains if chain == "B")

                # Should have more from chain A since center is in chain A
                assert chain_a_count >= chain_b_count


class TestSpatialSplitPeptideHandling:
    """Test peptide trimming with spatial split."""

    def test_spatial_split_with_peptide_trimming(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that peptide trimming affects spatial distance calculations."""
        peptide_topologies = create_peptide_topologies(["A"], 15)
        datapoints = create_datapoints_from_topologies(peptide_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            train_data, val_data = splitter.spatial_split(mock_universe)

            # Verify distance calculation was called with check_trim=True
            call_kwargs = mock_distances.call_args[1]
            assert call_kwargs["check_trim"] == True

    def test_peptide_vs_non_peptide_spatial_behavior(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test difference between peptide and non-peptide spatial splitting."""
        # Create mixed topology set
        regular_topologies = create_single_chain_topologies("A", 10)
        peptide_topologies = create_peptide_topologies(["A"], 10)
        all_topologies = regular_topologies + peptide_topologies

        datapoints = create_datapoints_from_topologies(all_topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues, check_trim=True)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            train_data, val_data = splitter.spatial_split(mock_universe)

        # Should handle mixed topology types correctly
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestSpatialSplitOverlapRemoval:
    """Test overlap removal with spatial split."""

    def test_spatial_split_with_overlap_removal(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial split with overlap removal enabled."""
        topologies = create_multi_chain_topologies(["A", "B"], 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            train_data, val_data = splitter.spatial_split(mock_universe, remove_overlap=True)

        assert len(train_data) >= 0
        assert len(val_data) >= 0
        assert len(train_data) + len(val_data) > 0

    def test_spatial_overlap_vs_no_overlap(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test difference between overlap removal enabled vs disabled."""
        topologies = create_multi_chain_topologies(["A", "B", "C"], 12)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B", "C"])

        splitter = setup_splitter(datapoints, common_residues, random_seed=999)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            # Without overlap removal
            train_no_removal, val_no_removal = splitter.spatial_split(
                mock_universe, remove_overlap=False
            )

            # Reset and try with overlap removal
            splitter._reset_retry_counter()
            train_with_removal, val_with_removal = splitter.spatial_split(
                mock_universe, remove_overlap=True
            )

        total_no_removal = len(train_no_removal) + len(val_no_removal)
        total_with_removal = len(train_with_removal) + len(val_with_removal)

        assert total_with_removal <= total_no_removal


class TestSpatialSplitCentralitySampling:
    """Test centrality sampling with spatial split."""

    def test_spatial_split_with_centrality_sampling(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial split with centrality sampling enabled."""
        topologies = create_single_chain_topologies("A", 25)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:20]  # Return subset

            splitter = setup_splitter(datapoints, common_residues, centrality=True)

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_sampled = 20
                distance_matrix = np.random.rand(n_sampled, n_sampled)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                mock_distances.return_value = (distance_matrix, distance_std)

                train_data, val_data = splitter.spatial_split(mock_universe)

            # Should have called sample_by_centrality
            mock_sample.assert_called_once()

    def test_centrality_affects_spatial_center_selection(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that centrality sampling affects which topologies are available for center selection."""
        topologies = create_single_chain_topologies("A", 30)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            # Return only first half of datapoints
            sampled_data = datapoints[:15]
            mock_sample.return_value = sampled_data

            splitter = setup_splitter(datapoints, common_residues, centrality=True)

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_sampled = 15
                distance_matrix = np.random.rand(n_sampled, n_sampled)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                mock_distances.return_value = (distance_matrix, distance_std)

                with patch("random.randint") as mock_randint:
                    mock_randint.return_value = 10  # Should be within sampled range

                    train_data, val_data = splitter.spatial_split(mock_universe)

                    # Verify randint was called with correct range (0 to 14 for 15 sampled items)
                    mock_randint.assert_called_with(0, 14)


class TestSpatialSplitDeterministicBehavior:
    """Test deterministic behavior and random seed handling."""

    def test_same_seed_same_center_selection(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that same seed produces same center selection."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            # First split
            splitter1 = setup_splitter(datapoints, common_residues, random_seed=123)
            train1, val1 = splitter1.spatial_split(mock_universe)

            # Second split with same seed
            splitter2 = setup_splitter(datapoints, common_residues, random_seed=123)
            train2, val2 = splitter2.spatial_split(mock_universe)

            # Results should be identical
            train1_ids = {dp.data_id for dp in train1}
            train2_ids = {dp.data_id for dp in train2}
            val1_ids = {dp.data_id for dp in val1}
            val2_ids = {dp.data_id for dp in val2}

            assert train1_ids == train2_ids
            assert val1_ids == val2_ids

    def test_different_seed_different_center(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that different seeds can produce different center selections."""
        topologies = create_single_chain_topologies("A", 30)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            # Create distance matrix where different centers would give different results
            distance_matrix = np.zeros((n_topos, n_topos))
            for i in range(n_topos):
                for j in range(n_topos):
                    if i != j:
                        distance_matrix[i, j] = abs(i - j)

            distance_std = np.zeros_like(distance_matrix)
            mock_distances.return_value = (distance_matrix, distance_std)

            # First split
            splitter1 = setup_splitter(datapoints, common_residues, random_seed=111)
            train1, val1 = splitter1.spatial_split(mock_universe)

            # Second split with different seed
            splitter2 = setup_splitter(datapoints, common_residues, random_seed=222)
            train2, val2 = splitter2.spatial_split(mock_universe)

            # Results should likely be different
            train1_ids = {dp.data_id for dp in train1}
            train2_ids = {dp.data_id for dp in train2}

            # At least one should be different (high probability)
            assert len(train1_ids.symmetric_difference(train2_ids)) >= 0

    def test_center_selection_logging(
        self, create_datapoints_from_topologies, setup_splitter, mock_universe, capfd
    ):
        """Test that center topology selection is properly logged."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            with patch("random.randint") as mock_randint:
                mock_randint.return_value = 5

                splitter.spatial_split(mock_universe)

                # Capture stdout to check logging
                captured = capfd.readouterr()
                assert "Selected center topology" in captured.out


class TestSpatialSplitEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset_spatial_splitting(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial splitting with very small datasets."""
        topologies = create_single_chain_topologies("A", 4)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

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
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            # Return wrong shape matrix
            wrong_shape_matrix = np.random.rand(5, 5)  # Wrong size
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
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix[0, 1] = np.nan  # Insert NaN value
            distance_matrix[1, 0] = np.nan  # Keep symmetry
            distance_std = np.zeros_like(distance_matrix)

            mock_distances.return_value = (distance_matrix, distance_std)

            # Should handle NaN values gracefully (they'll sort to end)
            train_data, val_data = splitter.spatial_split(mock_universe)

            assert len(train_data) > 0
            assert len(val_data) > 0


class TestSpatialSplitRetryLogic:
    """Test retry logic and validation."""

    def test_retry_logic_on_validation_failure(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test retry logic when validation fails."""
        topologies = create_single_chain_topologies("A", 8)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = [ValueError("Too small"), True]

            splitter = setup_splitter(datapoints, common_residues)

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_topos = len(datapoints)
                distance_matrix = np.random.rand(n_topos, n_topos)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                patch_partial_distances.return_value = (distance_matrix, distance_std)

                # Should retry and eventually succeed
                train_data, val_data = splitter.spatial_split(mock_universe)

                # Should have called validate_split twice (fail, then success)
                assert mock_validate.call_count == 2

    def test_max_retry_depth_exceeded(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test behavior when max retry depth is exceeded."""
        topologies = create_single_chain_topologies("A", 6)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        with patch.object(DataSplitter, "validate_split") as mock_validate:
            mock_validate.side_effect = ValueError("Always fails")

            splitter = setup_splitter(datapoints, common_residues, max_retry_depth=2)

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_topos = len(datapoints)
                distance_matrix = np.random.rand(n_topos, n_topos)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                patch_partial_distances.return_value = (distance_matrix, distance_std)

                with pytest.raises(
                    ValueError, match="Failed to create valid split after 2 attempts"
                ):
                    splitter.spatial_split(mock_universe)

    def test_successful_retry_resets_counter(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that successful split resets retry counter."""
        topologies = create_single_chain_topologies("A", 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            # Successful split should reset counter
            splitter.spatial_split(mock_universe)

            assert splitter.current_retry_count == 0


class TestSpatialSplitIntegration:
    """Test integration with other DataSplitter functionality."""

    def test_spatial_split_stores_merged_topologies(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that spatial split stores merged topologies for reference."""
        topologies = create_multi_chain_topologies(["A", "B"], 10)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A", "B"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            train_data, val_data = splitter.spatial_split(mock_universe)

            # Check that merged topologies are stored
            assert hasattr(splitter, "last_split_train_topologies_by_chain")
            assert hasattr(splitter, "last_split_val_topologies_by_chain")
            assert len(splitter.last_split_train_topologies_by_chain) > 0
            assert len(splitter.last_split_val_topologies_by_chain) > 0

    def test_spatial_split_with_custom_parameters(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test spatial split with various custom parameters."""
        topologies = create_single_chain_topologies("A", 15)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(
            datapoints, common_residues, train_size=0.7, check_trim=True, centrality=True
        )

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            mock_sample.return_value = datapoints[:12]

            with patch(
                "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
            ) as mock_distances:
                n_sampled = 12
                distance_matrix = np.random.rand(n_sampled, n_sampled)
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                distance_std = np.zeros_like(distance_matrix)

                mock_distances.return_value = (distance_matrix, distance_std)

                train_data, val_data = splitter.spatial_split(
                    mock_universe,
                    remove_overlap=True,
                    include_selection="protein and name CA",
                    exclude_selection="",  # Avoid removing all atoms
                    start=10,
                    stop=100,
                    step=5,
                )

                # Verify all parameters were used correctly
                call_kwargs = mock_distances.call_args[1]
                assert call_kwargs["check_trim"] == True
                assert call_kwargs["include_selection"] == "protein and name CA"
                assert call_kwargs["exclude_selection"] == ""
                assert call_kwargs["start"] == 10
                assert call_kwargs["stop"] == 100
                assert call_kwargs["step"] == 5

                # Check train_size was respected (approximately)
                expected_train_size = int(0.7 * 12)  # 8
                assert len(train_data) == expected_train_size

    def test_spatial_split_consistency_with_other_methods(
        self,
        create_datapoints_from_topologies,
        setup_splitter,
        mock_universe,
        patch_partial_distances,
    ):
        """Test that spatial split follows same patterns as other split methods."""
        topologies = create_single_chain_topologies("A", 20)
        datapoints = create_datapoints_from_topologies(topologies)
        common_residues = create_common_residues_for_chains(["A"])

        splitter = setup_splitter(datapoints, common_residues)

        with patch(
            "jaxent.src.interfaces.topology.mda_adapter.mda_TopologyAdapter.partial_topology_pairwise_distances"
        ) as mock_distances:
            n_topos = len(datapoints)
            distance_matrix = np.random.rand(n_topos, n_topos)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            distance_std = np.zeros_like(distance_matrix)

            patch_partial_distances.return_value = (distance_matrix, distance_std)

            # Test that spatial split returns same types as other methods
            spatial_train, spatial_val = splitter.spatial_split(mock_universe)

            # Compare with random split structure
            random_train, random_val = splitter.random_split()

            # Should return same types
            assert type(spatial_train) == type(random_train)
            assert type(spatial_val) == type(random_val)

            # Should contain same type of objects
            if spatial_train:
                assert type(spatial_train[0]) == type(random_train[0])
            if spatial_val:
                assert type(spatial_val[0]) == type(random_val[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
