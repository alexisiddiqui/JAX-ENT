"""
Unit tests for Phase 7: SAXS/XLMS data splitting compatibility.

Tests cover:
  - SAXS whole-system detection via is_whole_system_data
  - q-point random split
  - q-point stratified split
  - q-point stratified_deterministic split
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
        """DataSplitter backed by a non-whole-system datapoint should return False."""
        from jaxent.src.data.splitting.split import DataSplitter

        mock_datapoint = MagicMock()
        mock_datapoint.is_whole_system = MagicMock(return_value=False)

        mock_loader = MagicMock()
        mock_loader.data = [mock_datapoint]

        # Construct instance directly to bypass __init__ (which needs ensemble/common_residues)
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
        train, val = self.splitter.stratified_random_split()
        assert isinstance(train, list) and isinstance(val, list)

    def test_total_equals_n_q(self):
        train, val = self.splitter.stratified_random_split()
        assert len(train[0].intensities) + len(val[0].intensities) == self.n_q

    def test_no_index_overlap(self):
        self.splitter.stratified_random_split()
        train_idx = set(self.splitter.last_train_q_indices.tolist())
        val_idx = set(self.splitter.last_val_q_indices.tolist())
        assert train_idx.isdisjoint(val_idx)

    def test_indices_stored(self):
        self.splitter.stratified_random_split()
        assert self.splitter.last_train_q_indices is not None
        assert self.splitter.last_val_q_indices is not None

    def test_custom_n_strata(self):
        train, val = self.splitter.stratified_random_split(n_strata=5)
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

    def test_create_datasets_raises_without_factory_for_saxs(self):
        """SAXS data must raise ValueError when mapping_factory is not provided."""
        from jaxent.src.data.loader import ExpD_Dataloader

        curve = _make_saxs_curve(n_q=20)
        loader = ExpD_Dataloader([curve])

        mock_features = MagicMock()
        mock_topology = MagicMock()

        with pytest.raises(ValueError, match="mapping_factory"):
            loader.create_datasets(
                features=mock_features,
                feature_topology=mock_topology,
            )


# ── TestSAXSStratifiedDeterministicSplit ──────────────────────────────────────

class TestSAXSStratifiedDeterministicSplit:
    """
    Tests for DataSplitter.stratified_deterministic_split() on a SAXS_curve
    (whole-system data path).

    The method:
      - sorts q-points by intensity value
      - randomly picks n_strata in [2, n_q // 5]
      - alternates strata between train and val from a random starting assignment
    """

    def setup_method(self):
        self.n_q = 50
        self.curve = _make_saxs_curve(n_q=self.n_q, seed=3)
        self.splitter = _make_splitter(self.curve, train_size=0.6, seed=11)

    # ── basic contract ────────────────────────────────────────────────────────

    def test_returns_two_lists(self):
        train, val = self.splitter.stratified_deterministic_split()
        assert isinstance(train, list) and isinstance(val, list)

    def test_single_curve_per_split(self):
        train, val = self.splitter.stratified_deterministic_split()
        assert len(train) == 1 and len(val) == 1
        assert isinstance(train[0], SAXS_curve)
        assert isinstance(val[0], SAXS_curve)

    def test_total_equals_n_q(self):
        """All q-points should appear in exactly one of train or val."""
        train, val = self.splitter.stratified_deterministic_split()
        n_train = len(train[0].intensities)
        n_val = len(val[0].intensities)
        assert n_train + n_val == self.n_q

    def test_no_index_overlap(self):
        """Train and val q-point index sets must be disjoint."""
        self.splitter.stratified_deterministic_split()
        train_idx = set(self.splitter.last_train_q_indices.tolist())
        val_idx = set(self.splitter.last_val_q_indices.tolist())
        assert train_idx.isdisjoint(val_idx)

    def test_both_splits_non_empty(self):
        train, val = self.splitter.stratified_deterministic_split()
        assert len(train[0].intensities) > 0
        assert len(val[0].intensities) > 0

    def test_indices_stored(self):
        self.splitter.stratified_deterministic_split()
        assert self.splitter.last_train_q_indices is not None
        assert self.splitter.last_val_q_indices is not None

    # ── strata count ──────────────────────────────────────────────────────────

    def test_n_strata_in_valid_range(self):
        """
        The implementation picks n_strata ~ U[2, n_q//5].
        We cannot inspect n_strata directly, but the split sizes must reflect
        an alternating pattern over that range, so the total is always n_q.
        Indirectly verify the range constraint by running with a tiny dataset.
        """
        tiny_curve = _make_saxs_curve(n_q=10, seed=99)
        splitter = _make_splitter(tiny_curve, train_size=0.6, seed=5)
        train, val = splitter.stratified_deterministic_split()
        # n_q//5 = 2 for n_q=10, so n_strata is forced to 2
        assert len(train[0].intensities) + len(val[0].intensities) == 10

    # ── reproducibility ───────────────────────────────────────────────────────

    def test_reproducible_with_same_seed(self):
        splitter2 = _make_splitter(self.curve, train_size=0.6, seed=11)
        train1, _ = self.splitter.stratified_deterministic_split()
        train2, _ = splitter2.stratified_deterministic_split()
        np.testing.assert_array_equal(train1[0].intensities, train2[0].intensities)



# ── TestIsWholeSystem ─────────────────────────────────────────────────────────

class TestIsWholeSystem:
    """Tests for the is_whole_system() classmethod on datapoint types."""

    def test_saxs_curve_is_whole_system(self):
        from jaxent.src.custom_types.SAXS import SAXS_curve
        assert SAXS_curve.is_whole_system() is True

    def test_saxs_curve_instance_is_whole_system(self):
        curve = _make_saxs_curve()
        assert curve.is_whole_system() is True

    def test_hdx_peptide_not_whole_system(self):
        from jaxent.src.custom_types.HDX import HDX_peptide
        assert HDX_peptide.is_whole_system() is False

    def test_hdx_protection_factor_not_whole_system(self):
        from jaxent.src.custom_types.HDX import HDX_protection_factor
        assert HDX_protection_factor.is_whole_system() is False

    def test_xlms_distance_restraint_not_whole_system(self):
        from jaxent.src.custom_types.XLMS import XLMS_distance_restraint
        assert XLMS_distance_restraint.is_whole_system() is False


# ── TestExpDDataloaderWithSAXS ────────────────────────────────────────────────

class TestExpDDataloaderWithSAXS:
    """Tests that ExpD_Dataloader can be constructed directly with SAXS data."""

    def _make_loader(self, n_q: int = 30) -> tuple:
        from jaxent.src.data.loader import ExpD_Dataloader
        curve = _make_saxs_curve(n_q=n_q)
        loader = ExpD_Dataloader([curve])
        return loader, curve

    def test_init_succeeds_with_single_saxs_curve(self):
        """ExpD_Dataloader([saxs_curve]) must not raise."""
        loader, _ = self._make_loader()
        assert loader is not None

    def test_key_is_set_correctly(self):
        from jaxent.src.custom_types.SAXS import SAXS_curve
        loader, _ = self._make_loader()
        assert loader.key == SAXS_curve.key

    def test_top_is_set(self):
        loader, curve = self._make_loader()
        assert len(loader.top) == 1
        assert loader.top[0] is curve.top

    def test_fragment_index_assigned(self):
        """When fragment_index is None, __init__ assigns 0 for whole-system data."""
        from jaxent.src.data.loader import ExpD_Dataloader
        curve = _make_saxs_curve()
        curve.top.fragment_index = None
        loader = ExpD_Dataloader([curve])
        assert loader.data[0].top.fragment_index == 0

    def test_create_datasets_with_factory_succeeds(self):
        """create_datasets must succeed when mapping_factory is provided."""
        loader, curve = self._make_loader(n_q=20)
        splitter = _make_splitter(curve, train_size=0.6, seed=1)
        train_data, val_data = splitter.random_split()
        factory = splitter.saxs_mapping_factory(train_data, val_data)

        mock_features = MagicMock()
        mock_topology = MagicMock()

        with patch("jaxent.src.data.loader.create_sparse_map") as mock_sm:
            loader.create_datasets(
                features=mock_features,
                feature_topology=mock_topology,
                train_data=train_data,
                val_data=val_data,
                test_data=train_data,
                mapping_factory=factory,
            )
            mock_sm.assert_not_called()

        assert hasattr(loader, "train")
        assert hasattr(loader, "val")
        assert hasattr(loader, "test")

    def test_pytree_flatten_unflatten_roundtrip(self):
        """tree_flatten / tree_unflatten must round-trip without errors."""
        import jax
        loader, curve = self._make_loader(n_q=15)
        splitter = _make_splitter(curve, train_size=0.6, seed=3)
        train_data, val_data = splitter.random_split()
        factory = splitter.saxs_mapping_factory(train_data, val_data)

        mock_features = MagicMock()
        mock_topology = MagicMock()

        with patch("jaxent.src.data.loader.create_sparse_map"):
            loader.create_datasets(
                features=mock_features,
                feature_topology=mock_topology,
                train_data=train_data,
                val_data=val_data,
                test_data=train_data,
                mapping_factory=factory,
            )

        leaves, aux = jax.tree_util.tree_flatten(loader)
        reconstructed = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(loader), leaves
        )
        assert reconstructed.key == loader.key
