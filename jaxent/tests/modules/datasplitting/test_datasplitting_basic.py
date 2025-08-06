from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter, filter_common_residues
from jaxent.src.interfaces.topology import (
    TopologyFactory,
)


# Mock ExpD_Datapoint for testing
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top):
        self.top = top
        self.key = "mock_key"

    def extract_features(self):
        return np.array([1.0])  # Mock feature


# Test fixtures to create test data
@pytest.fixture
def create_test_topologies():
    """Create test Partial_Topology objects."""
    topo1 = TopologyFactory.from_range("A", 1, 10, fragment_name="topo1")
    topo2 = TopologyFactory.from_range("A", 5, 15, fragment_name="topo2")
    topo3 = TopologyFactory.from_range("A", 15, 25, fragment_name="topo3")
    topo4 = TopologyFactory.from_range("B", 1, 10, fragment_name="topo4")  # Different chain

    # With peptide trimming
    topo5 = TopologyFactory.from_range(
        "A", 1, 10, fragment_name="topo5", peptide=True, peptide_trim=2
    )
    topo6 = TopologyFactory.from_range(
        "A", 5, 15, fragment_name="topo6", peptide=True, peptide_trim=3
    )

    return [topo1, topo2, topo3, topo4, topo5, topo6]


@pytest.fixture
def create_test_datapoints(create_test_topologies):
    """Create test ExpD_Datapoint objects."""
    topos = create_test_topologies
    dp1 = MockExpD_Datapoint(topos[0])  # Residues 1-10, Chain A
    dp2 = MockExpD_Datapoint(topos[1])  # Residues 5-15, Chain A
    dp3 = MockExpD_Datapoint(topos[2])  # Residues 15-25, Chain A
    dp4 = MockExpD_Datapoint(topos[3])  # Residues 1-10, Chain B
    dp5 = MockExpD_Datapoint(topos[4])  # Residues 1-10, Chain A, peptide with trim=2
    dp6 = MockExpD_Datapoint(topos[5])  # Residues 5-15, Chain A, peptide with trim=3

    return [dp1, dp2, dp3, dp4, dp5, dp6]


@pytest.fixture
def mock_dataloader(create_test_datapoints):
    """Create a mock ExpD_Dataloader."""
    mock_loader = MagicMock(spec=ExpD_Dataloader)
    mock_loader.data = create_test_datapoints
    mock_loader.y_true = np.array([1.0] * len(create_test_datapoints))

    return mock_loader


# Tests for filter_common_residues
class TestFilterCommonResidues:
    def test_filter_with_matching_residues(self, create_test_datapoints):
        """Test filtering with matching residues."""
        # Common residues in chain A, residues 5-10 (at least 2 residues)
        common_residues = {TopologyFactory.from_range("A", 5, 10, fragment_name="common")}

        # Only datapoints with topologies that intersect with residues 5-10 on chain A should remain
        # These are dp1 (1-10), dp2 (5-15), and dp5 (1-10 peptide)
        filtered = filter_common_residues(create_test_datapoints[:5], common_residues)

        # Expected to contain dp1, dp2, dp5
        assert len(filtered) == 3
        assert create_test_datapoints[0] in filtered
        assert create_test_datapoints[1] in filtered
        assert create_test_datapoints[4] in filtered

    def test_filter_with_no_matches(self, create_test_datapoints):
        """Test filtering with no matching residues (should raise ValueError)."""
        # Common residues in chain C - no match with any datapoint (at least 2 residues)
        common_residues = {TopologyFactory.from_range("C", 1, 10, fragment_name="common")}

        with pytest.raises(ValueError, match="Filtered dataset is empty"):
            filter_common_residues(create_test_datapoints, common_residues)

    def test_filter_with_peptide_trimming(self, create_test_datapoints):
        """Test filtering with peptide trimming."""
        # Common residues matching only trimmed regions
        # topo5 has residues 1-10 but with trim=2, effective residues are 3-10
        # topo6 has residues 5-15 but with trim=3, effective residues are 8-15
        common_residues = {TopologyFactory.from_range("A", 8, 10, fragment_name="common")}

        # With check_trim=True, both peptide datapoints should match
        filtered_with_trim = filter_common_residues(
            [create_test_datapoints[4], create_test_datapoints[5]], common_residues, check_trim=True
        )
        assert len(filtered_with_trim) == 2

        # With check_trim=False, both should still match as we check full residues
        filtered_without_trim = filter_common_residues(
            [create_test_datapoints[4], create_test_datapoints[5]],
            common_residues,
            check_trim=False,
        )
        assert len(filtered_without_trim) == 2


# Tests for DataSplitter class
class TestDataSplitter:
    @pytest.fixture
    def setup_datasplitter(self, mock_dataloader, create_test_datapoints):
        """Set up a DataSplitter for testing."""
        # Patch the filter_common_residues function to return test datapoints
        with patch("jaxent.src.data.splitting.split.filter_common_residues") as mock_filter:
            mock_filter.return_value = create_test_datapoints

            # Patch the calculate_fragment_redundancy method
            with patch(
                "jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy"
            ) as mock_calc:
                mock_calc.return_value = [0.0] * len(create_test_datapoints)

                # Create common residues with multiple topologies to ensure at least 2 residues
                common_residues = {
                    TopologyFactory.from_range("A", 1, 10, fragment_name="common1"),
                    TopologyFactory.from_range("A", 15, 25, fragment_name="common2"),
                }

                # Mock the validation by patching the __init__ method or the specific validation
                # Let's try mocking the total residue count calculation
                with patch.object(DataSplitter, "__init__", return_value=None) as mock_init:
                    splitter = DataSplitter.__new__(DataSplitter)
                    splitter.dataset = mock_dataloader
                    splitter.common_residues = common_residues
                    splitter.filtered_data = create_test_datapoints
                    splitter.redundancy_scores = [0.0] * len(create_test_datapoints)
                    splitter.min_split_size = 5  # Minimum size for train/val splits
                    # Mock any other attributes that might be needed
                    splitter.train_data = None
                    splitter.val_data = None

                yield splitter

    def test_validate_valid_split(self, setup_datasplitter, create_test_datapoints):
        """Test validation with a valid split."""
        splitter = setup_datasplitter

        # Create train/val split with sufficient datapoints
        # Minimum requirement is 5 datapoints per split
        train_data = create_test_datapoints * 3  # 18 datapoints
        val_data = create_test_datapoints * 2  # 12 datapoints

        assert splitter.validate_split(train_data, val_data) is True

    def test_validate_too_small_split(self, setup_datasplitter, create_test_datapoints):
        """Test validation with too few samples in one split."""
        splitter = setup_datasplitter

        # Create invalid split - too few in train (minimum is 5)
        train_data = create_test_datapoints[:4]  # Only 4 datapoints
        val_data = create_test_datapoints * 2  # 12 datapoints

        with pytest.raises(ValueError, match="set is too small"):
            splitter.validate_split(train_data, val_data)

    def test_validate_empty_split(self, setup_datasplitter):
        """Test validation with an empty split."""
        splitter = setup_datasplitter

        # Create invalid split - empty train
        train_data = []
        val_data = []

        with pytest.raises(AssertionError, match="No data found"):
            splitter.validate_split(train_data, val_data)

    def test_validate_small_ratio(self, setup_datasplitter, create_test_datapoints):
        """Test validation with sufficient samples but small ratio."""
        splitter = setup_datasplitter

        # Set the dataset size to be large
        splitter.dataset.data = create_test_datapoints * 10  # 60 datapoints

        # Create invalid split - train ratio too small
        train_data = create_test_datapoints[:5]  # 5 datapoints = 8.3% of 60 (< 10% minimum)
        val_data = create_test_datapoints * 8  # 48 datapoints

        with pytest.raises(ValueError, match="too small"):
            splitter.validate_split(train_data, val_data)


if __name__ == "__main__":
    pytest.main()
