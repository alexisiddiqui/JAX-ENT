"""
Unit tests for Phase 7: SAXS/XLMS data splitting compatibility.

Tests cover:
  - SAXS whole-system detection via is_whole_system_data
  - q-point random split
  - q-point stratified split
  - Guards on incompatible split methods
  - QSubsetMapping creation and apply()
  - saxs_mapping_factory()
  - create_datasets() does NOT call create_sparse_map when mapping_factory is provided
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from jaxent.src.custom_types.SAXS import SAXS_curve
from jaxent.src.data.splitting.mapping import QSubsetMapping, create_q_subset_mapping
from jaxent.src.interfaces.topology import Partial_Topology


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_topology(fragment_index: int = 0) -> Partial_Topology:
    """Minimal single-residue topology with a valid fragment_index."""
    top = Partial_Topology(
        chain="A",
        residues=frozenset({1}),
        fragment_name="system",
        fragment_index=fragment_index,
    )
    return top


def _make_saxs_curve(n_q: int = 50, seed: int = 0) -> SAXS_curve:
    rng = np.random.default_rng(seed)
    intensities = rng.random(n_q).astype(np.float32)
    q_values = np.linspace(0.01, 0.5, n_q, dtype=np.float32)
    errors = rng.random(n_q).astype(np.float32) * 0.01
    return SAXS_curve(
        top=_make_topology(fragment_index=0),
        intensities=intensities,
        q_values=q_values,
        errors=errors,
    )


def _make_splitter(saxs_curve: SAXS_curve, train_size: float = 0.6, seed: int = 42):
    """Build a DataSplitter around a single SAXS_curve without MDAnalysis."""
    from jaxent.src.data.loader import ExpD_Dataloader
    from jaxent.src.data.splitting.split import DataSplitter

    loader = ExpD_Dataloader([saxs_curve])
    # The SAXS early-return in __init__ bypasses common_residues/ensemble requirement
    splitter = DataSplitter(
        dataset=loader,
        random_seed=seed,
        train_size=train_size,
    )
    return splitter


# ── TestWholeSytemDetection ───────────────────────────────────────────────────

class TestWholeSytemDetection:
    def test_true_for_saxs_curve(self):
        curve = _make_saxs_curve()
        splitter = _make_splitter(curve)
        assert splitter.is_whole_system_data is True

    def test_false_for_non_saxs(self):
        """DataSplitter backed by a mock non-SAXS dataloader should return False."""
        from jaxent.src.data.splitting.split import DataSplitter

        mock_loader = MagicMock()
        mock_loader.data = [MagicMock(spec=[])]  # not a SAXS_curve instance

        # We can check the property logic directly via a manually-constructed instance
        # without calling __init__ (which requires ensemble/common_residues for non-SAXS)
        splitter = object.__new__(DataSplitter)
        splitter.dataset = mock_loader
        assert splitter.is_whole_system_data is False


# ── TestSAXSRandomSplit ───────────────────────────────────────────────────────

class TestSAXSRandomSplit:
    def setup_method(self):
        self.n_q = 50
        self.curve = _make_saxs_curve(n_q=self.n_q)
        self.splitter = _make_splitter(self.curve, train_size=0.6, seed=42)

    def test_returns_two_lists(self):
        train, val = self.splitter.random_split()
        assert isinstance(train, list) and isinstance(val, list)

    def test_single_curve_per_split(self):
        train, val = self.splitter.random_split()
        assert len(train) == 1 and len(val) == 1
        assert isinstance(train[0], SAXS_curve)
        assert isinstance(val[0], SAXS_curve)

    def test_total_equals_n_q(self):
        train, val = self.splitter.random_split()
        n_train = len(train[0].intensities)
        n_val = len(val[0].intensities)
        assert n_train + n_val == self.n_q

    def test_no_index_overlap(self):
        self.splitter.random_split()
        train_idx = set(self.splitter.last_train_q_indices.tolist())
        val_idx = set(self.splitter.last_val_q_indices.tolist())
        assert train_idx.isdisjoint(val_idx)

    def test_q_values_subsetted(self):
        train, val = self.splitter.random_split()
        # q_values length should match intensities length
        assert len(train[0].q_values) == len(train[0].intensities)
        assert len(val[0].q_values) == len(val[0].intensities)

    def test_errors_subsetted(self):
        train, val = self.splitter.random_split()
        assert train[0].errors is not None
        assert len(train[0].errors) == len(train[0].intensities)

    def test_reproducible_with_same_seed(self):
        splitter2 = _make_splitter(self.curve, train_size=0.6, seed=42)
        train1, _ = self.splitter.random_split()
        train2, _ = splitter2.random_split()
        np.testing.assert_array_equal(train1[0].intensities, train2[0].intensities)

    def test_different_seeds_give_different_splits(self):
        splitter2 = _make_splitter(self.curve, train_size=0.6, seed=99)
        _, val1 = self.splitter.random_split()
        _, val2 = splitter2.random_split()
        # Very unlikely to be equal
        assert not np.array_equal(val1[0].intensities, val2[0].intensities)

    def test_indices_stored(self):
        self.splitter.random_split()
        assert self.splitter.last_train_q_indices is not None
        assert self.splitter.last_val_q_indices is not None


# ── TestSAXSStratifiedSplit ───────────────────────────────────────────────────

class TestSAXSStratifiedSplit:
    def setup_method(self):
        self.n_q = 50
        self.curve = _make_saxs_curve(n_q=self.n_q)
        self.splitter = _make_splitter(self.curve, train_size=0.6, seed=7)

    def test_returns_two_lists(self):
        train, val = self.splitter.stratified_split()
        assert isinstance(train, list) and isinstance(val, list)

    def test_total_equals_n_q(self):
        train, val = self.splitter.stratified_split()
        assert len(train[0].intensities) + len(val[0].intensities) == self.n_q

    def test_no_index_overlap(self):
        self.splitter.stratified_split()
        train_idx = set(self.splitter.last_train_q_indices.tolist())
        val_idx = set(self.splitter.last_val_q_indices.tolist())
        assert train_idx.isdisjoint(val_idx)

    def test_indices_stored(self):
        self.splitter.stratified_split()
        assert self.splitter.last_train_q_indices is not None
        assert self.splitter.last_val_q_indices is not None

    def test_custom_n_strata(self):
        train, val = self.splitter.stratified_split(n_strata=5)
        assert len(train[0].intensities) + len(val[0].intensities) == self.n_q


# ── TestIncompatibleMethodGuards ──────────────────────────────────────────────

class TestIncompatibleMethodGuards:
    def setup_method(self):
        self.splitter = _make_splitter(_make_saxs_curve())

    def test_sequence_split_raises(self):
        with pytest.raises(ValueError, match="sequence_split"):
            self.splitter.sequence_split()

    def test_sequence_cluster_split_raises(self):
        with pytest.raises(ValueError, match="sequence_cluster_split"):
            self.splitter.sequence_cluster_split()

    def test_spatial_split_raises(self):
        with pytest.raises(ValueError, match="spatial_split"):
            self.splitter.spatial_split(universe=MagicMock())


# ── TestCreateQSubsetMapping ──────────────────────────────────────────────────

class TestCreateQSubsetMapping:
    def test_returns_q_subset_mapping(self):
        mapping = create_q_subset_mapping([0, 2, 4])
        assert isinstance(mapping, QSubsetMapping)

    def test_dtype_int32(self):
        import jax.numpy as jnp
        mapping = create_q_subset_mapping([1, 3, 5])
        assert mapping.indices.dtype == jnp.int32

    def test_apply_selects_correct_elements(self):
        import jax.numpy as jnp
        arr = jnp.arange(10, dtype=jnp.float32)
        indices = [0, 2, 4, 6]
        mapping = create_q_subset_mapping(indices)
        result = mapping.apply(arr)
        expected = jnp.array([0.0, 2.0, 4.0, 6.0])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_apply_with_numpy_array(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        mapping = create_q_subset_mapping([1, 3])
        result = mapping.apply(arr)
        np.testing.assert_array_equal(np.asarray(result), np.array([20.0, 40.0]))


# ── TestSAXSMappingFactory ────────────────────────────────────────────────────

class TestSAXSMappingFactory:
    def setup_method(self):
        self.n_q = 30
        self.curve = _make_saxs_curve(n_q=self.n_q)
        self.splitter = _make_splitter(self.curve, train_size=0.7, seed=0)
        self.train_data, self.val_data = self.splitter.random_split()
        self.factory = self.splitter.saxs_mapping_factory(self.train_data, self.val_data)

    def test_factory_is_callable(self):
        assert callable(self.factory)

    def test_train_factory_returns_q_subset_mapping(self):
        result = self.factory(None, None, self.train_data)
        assert isinstance(result, QSubsetMapping)

    def test_val_factory_returns_q_subset_mapping(self):
        result = self.factory(None, None, self.val_data)
        assert isinstance(result, QSubsetMapping)

    def test_train_indices_match(self):
        result = self.factory(None, None, self.train_data)
        np.testing.assert_array_equal(
            np.asarray(result.indices),
            np.asarray(self.splitter.last_train_q_indices),
        )

    def test_val_indices_match(self):
        result = self.factory(None, None, self.val_data)
        np.testing.assert_array_equal(
            np.asarray(result.indices),
            np.asarray(self.splitter.last_val_q_indices),
        )

    def test_test_fallback_uses_full_range(self):
        # A data_split that is neither train nor val should get the full range
        other_data = [_make_saxs_curve(n_q=self.n_q)]
        result = self.factory(None, None, other_data)
        assert isinstance(result, QSubsetMapping)
        np.testing.assert_array_equal(
            np.asarray(result.indices),
            np.arange(self.n_q),
        )


# ── TestCreateDatasetsWithMappingFactory ──────────────────────────────────────

class TestCreateDatasetsWithMappingFactory:
    def test_create_sparse_map_not_called_when_factory_provided(self):
        """When mapping_factory is given, create_sparse_map must NOT be called."""
        from jaxent.src.data.loader import ExpD_Dataloader

        curve = _make_saxs_curve(n_q=20)
        loader = ExpD_Dataloader([curve])

        splitter = _make_splitter(curve, train_size=0.6, seed=1)
        train_data, val_data = splitter.random_split()
        factory = splitter.saxs_mapping_factory(train_data, val_data)

        mock_features = MagicMock()
        mock_features.extract_features = MagicMock(return_value=np.ones(20))
        mock_topology = MagicMock()

        with patch(
            "jaxent.src.data.loader.create_sparse_map"
        ) as mock_sparse_map:
            loader.create_datasets(
                features=mock_features,
                feature_topology=mock_topology,
                train_data=train_data,
                val_data=val_data,
                test_data=train_data,  # reuse train as test for this unit test
                mapping_factory=factory,
            )
            mock_sparse_map.assert_not_called()

    def test_create_sparse_map_called_when_no_factory(self):
        """Without mapping_factory, create_sparse_map must be called (existing behaviour)."""
        from jaxent.src.data.loader import ExpD_Dataloader
        from jaxent.src.data.splitting.mapping import SparseFragmentMapping

        from jax.experimental import sparse as jax_sparse
        import jax.numpy as jnp

        curve = _make_saxs_curve(n_q=10)
        loader = ExpD_Dataloader([curve])

        splitter = _make_splitter(curve, train_size=0.6, seed=2)
        train_data, val_data = splitter.random_split()

        # Provide a fake sparse map return value
        fake_bcoo = jax_sparse.BCOO.fromdense(jnp.eye(10))

        mock_features = MagicMock()
        mock_topology = MagicMock()

        with patch(
            "jaxent.src.data.loader.create_sparse_map", return_value=fake_bcoo
        ) as mock_sparse_map:
            loader.create_datasets(
                features=mock_features,
                feature_topology=mock_topology,
                train_data=train_data,
                val_data=val_data,
                test_data=train_data,
            )
            assert mock_sparse_map.call_count == 3  # train + val + test
