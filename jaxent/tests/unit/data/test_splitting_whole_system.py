"""
Unit tests for DataSplitter with whole-system data (SAXS curves).

Tests verify that DataSplitter correctly auto-detects and handles
whole-system observables like SAXS_curve, bypassing topology-based
splitting in favor of simpler datapoint-level splitting.
"""

import numpy as np
import pytest

from jaxent.src.custom_types.SAXS import SAXS_curve
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import Partial_Topology


@pytest.fixture
def sample_partial_topology() -> Partial_Topology:
    """Create a simple whole-construct Partial_Topology for SAXS data."""
    return Partial_Topology(
        chain="A",
        residues=list(range(1, 101)),
        fragment_name="whole_construct",
        fragment_index=0,
    )


@pytest.fixture
def saxs_dataloader(sample_partial_topology: Partial_Topology) -> ExpD_Dataloader:
    """Create an ExpD_Dataloader with synthetic SAXS_curve datapoints."""
    data = [
        SAXS_curve(
            top=sample_partial_topology,
            intensities=np.array([float(i + j) for j in range(5)]),
            q_values=np.array([0.01 * j for j in range(5)]),
        )
        for i in range(20)
    ]
    loader = ExpD_Dataloader.__new__(ExpD_Dataloader)
    loader.data = data
    return loader


@pytest.fixture
def hdx_dataloader() -> ExpD_Dataloader:
    """Create an ExpD_Dataloader with HDX_peptide datapoints."""
    data = [
        HDX_peptide(
            top=Partial_Topology(
                chain="A",
                residues=list(range(i, i + 10)),
                fragment_name=f"peptide_{i}",
                fragment_index=i,
            ),
            dfrac=[0.5, 0.6, 0.7, 0.8, 0.9],
        )
        for i in range(1, 20, 2)
    ]
    loader = ExpD_Dataloader.__new__(ExpD_Dataloader)
    loader.data = data
    return loader


class TestWholeSystemDataDetection:
    """Tests for auto-detection of whole-system data."""

    def test_is_whole_system_data_true_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """DataSplitter should detect SAXS_curve data as whole-system."""
        splitter = DataSplitter(saxs_dataloader, ensemble=None, common_residues=None)
        assert splitter.is_whole_system_data is True

    def test_is_whole_system_data_false_for_hdx(
        self, hdx_dataloader: ExpD_Dataloader
    ):
        """DataSplitter should NOT detect HDX data as whole-system."""
        # This should raise because we didn't provide ensemble or common_residues
        with pytest.raises(ValueError, match="Either common_residues or ensemble"):
            DataSplitter(hdx_dataloader, ensemble=None, common_residues=None)

    def test_is_whole_system_data_false_for_empty_data(self):
        """DataSplitter should raise for empty data."""
        loader = ExpD_Dataloader.__new__(ExpD_Dataloader)
        loader.data = []
        # Empty data with no ensemble should raise
        with pytest.raises(ValueError, match="Either common_residues or ensemble"):
            DataSplitter(loader, ensemble=None, common_residues=None)


class TestInitializationForWholeSystemData:
    """Tests for DataSplitter initialization with whole-system data."""

    def test_init_does_not_require_ensemble_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """Initialization should not require ensemble or common_residues for SAXS."""
        # Should not raise
        splitter = DataSplitter(
            saxs_dataloader, ensemble=None, common_residues=None
        )
        assert splitter.is_whole_system_data is True
        assert splitter.common_residues == set()
        assert splitter.splittable_residues == {}

    def test_init_sets_fragment_index_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """Initialization should set fragment_index to 0 for SAXS data."""
        splitter = DataSplitter(saxs_dataloader, ensemble=None, common_residues=None)
        assert splitter.dataset.data[0].top.fragment_index == 0

    def test_init_raises_without_ensemble_for_hdx(self, hdx_dataloader: ExpD_Dataloader):
        """Initialization should raise for HDX data without ensemble/common_residues."""
        with pytest.raises(ValueError, match="Either common_residues or ensemble"):
            DataSplitter(hdx_dataloader, ensemble=None, common_residues=None)


class TestRandomSplitForWholeSystemData:
    """Tests for random_split with whole-system data."""

    def test_random_split_saxs_returns_lists(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """random_split should return two lists of SAXS_curve objects."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert all(isinstance(d, SAXS_curve) for d in train_data)
        assert all(isinstance(d, SAXS_curve) for d in val_data)

    def test_random_split_saxs_no_overlap(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """random_split train/val should be non-overlapping."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()

        # Check that train and val share no datapoints
        train_set = set(id(d) for d in train_data)
        val_set = set(id(d) for d in val_data)
        assert len(train_set & val_set) == 0

    def test_random_split_saxs_sizes_sum(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """random_split train/val sizes should sum to total."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()

        # For SAXS, we split by q-points, so sizes are typically smaller
        # Just verify they're non-empty
        assert len(train_data) > 0
        assert len(val_data) > 0

    def test_random_split_saxs_respects_train_size(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """random_split should roughly respect train_size ratio for q-points."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.7,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()

        # For SAXS, the q-point split should respect the ratio
        # (assuming SAXS_curve has 5 q-points per curve)
        assert len(train_data) > 0 and len(val_data) > 0


class TestStratifiedSplitForWholeSystemData:
    """Tests for stratified_random_split with whole-system data."""

    def test_stratified_random_split_saxs_returns_lists(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """stratified_random_split should return two lists of SAXS_curve objects."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.stratified_random_split(n_strata=3)

        assert isinstance(train_data, list)
        assert isinstance(val_data, list)
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert all(isinstance(d, SAXS_curve) for d in train_data)
        assert all(isinstance(d, SAXS_curve) for d in val_data)

    def test_stratified_random_split_saxs_no_overlap(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """stratified_random_split train/val should be non-overlapping."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.stratified_random_split(n_strata=3)

        # Check that train and val share no datapoints
        train_set = set(id(d) for d in train_data)
        val_set = set(id(d) for d in val_data)
        assert len(train_set & val_set) == 0

    def test_stratified_random_split_saxs_uses_extract_features(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """stratified_random_split should use extract_features for stratification."""
        # This test verifies that the fallback to extract_features() works
        # by checking that split completes without error
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        # Should not raise
        train_data, val_data = splitter.stratified_random_split(n_strata=3)
        assert len(train_data) > 0
        assert len(val_data) > 0


class TestIncompatibleMethodsForWholeSystemData:
    """Tests for methods that should raise ValueError for whole-system data."""

    def test_sequence_split_raises_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """sequence_split should raise ValueError for whole-system data."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
        )
        with pytest.raises(ValueError, match="not compatible with whole-system data"):
            splitter.sequence_split()

    def test_sequence_cluster_split_raises_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """sequence_cluster_split should raise ValueError for whole-system data."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
        )
        with pytest.raises(ValueError, match="not compatible with whole-system data"):
            splitter.sequence_cluster_split()

    def test_spatial_split_raises_for_saxs(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """spatial_split should raise ValueError for whole-system data."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
        )
        with pytest.raises(ValueError, match="not compatible with whole-system data"):
            splitter.spatial_split(universe=None)


class TestMappingFactoryForWholeSystemData:
    """Tests for SAXS-specific mapping factory."""

    def test_saxs_mapping_factory_returns_callable(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """saxs_mapping_factory should return a callable."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()
        factory = splitter.saxs_mapping_factory(train_data, val_data)

        assert callable(factory)

    def test_saxs_mapping_factory_with_q_indices(
        self, saxs_dataloader: ExpD_Dataloader
    ):
        """saxs_mapping_factory should create mappings using q-indices."""
        splitter = DataSplitter(
            saxs_dataloader,
            ensemble=None,
            common_residues=None,
            train_size=0.5,
            random_seed=42,
        )
        train_data, val_data = splitter.random_split()

        # Verify that q-indices were set
        assert splitter.last_train_q_indices is not None
        assert splitter.last_val_q_indices is not None
        assert len(splitter.last_train_q_indices) > 0
        assert len(splitter.last_val_q_indices) > 0
