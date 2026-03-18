"""
Tests for the generalised create_functional_loss in opt/loss/base.py.

Verifies:
- Backward compat with BV PF and uptake paths via SparseFragmentMapping
- Generic path via QSubsetMapping + _DummyOutput
- Known loss values
- String (m_key) vs int prediction_index dispatch
- Builder regression after the refactor
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.typing import ClassVar
from jax import Array
from jax.experimental import sparse

from jaxent.src.custom_types.features import Output_Features
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import Dataset
from jaxent.src.data.splitting.mapping import QSubsetMapping, SparseFragmentMapping
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.loss.base import create_functional_loss
from jaxent.src.opt.loss.functional import (
    hdx_pf_l2_builder,
    hdx_uptake_l2_builder,
)


# ---------------------------------------------------------------------------
# _DummyOutput — generic Output_Features for non-HDX tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DummyOutput(Output_Features):
    values: Array

    __features__: ClassVar[set[str]] = {"values"}
    key: ClassVar[m_key] = m_key("test_dummy")

    @property
    def output_shape(self):
        return self.values.shape

    def y_pred(self) -> Array:
        return jnp.asarray(self.values)


jax.tree_util.register_pytree_node(
    _DummyOutput,
    _DummyOutput.tree_flatten,
    _DummyOutput.tree_unflatten,
)


# ---------------------------------------------------------------------------
# Shared mock dataloader
# ---------------------------------------------------------------------------


class _MockDataloader:
    def __init__(self, y_true, data_mapping_train, data_mapping_val):
        self.train = Dataset(data=[], y_true=y_true, data_mapping=data_mapping_train)
        self.val = Dataset(data=[], y_true=y_true, data_mapping=data_mapping_val)

    def tree_flatten(self):
        return (self.train, self.val), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.train, obj.val = children
        return obj


jax.tree_util.register_pytree_node(
    _MockDataloader,
    _MockDataloader.tree_flatten,
    _MockDataloader.tree_unflatten,
)


# ---------------------------------------------------------------------------
# Helpers to build Simulation objects
# ---------------------------------------------------------------------------


def _make_sim_pf(log_pf: Array) -> Simulation:
    num_residues = log_pf.shape[0]
    params = Simulation_Parameters(
        frame_weights=jnp.array([1.0]),
        frame_mask=jnp.array([1.0]),
        model_parameters=[BV_Model_Parameters()],
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )
    sim = Simulation(
        input_features=[BV_input_features(
            heavy_contacts=jnp.ones(num_residues),
            acceptor_contacts=jnp.ones(num_residues),
        )],
        forward_models=[],
        params=params,
    )
    sim.outputs = (BV_output_features(log_Pf=log_pf),)
    return sim


def _make_sim_uptake(uptake: Array) -> Simulation:
    num_residues = uptake.shape[1]
    params = Simulation_Parameters(
        frame_weights=jnp.array([1.0]),
        frame_mask=jnp.array([1.0]),
        model_parameters=[BV_Model_Parameters(timepoints=jnp.array([0.1, 1.0]))],
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )
    sim = Simulation(
        input_features=[BV_input_features(
            heavy_contacts=jnp.ones(num_residues),
            acceptor_contacts=jnp.ones(num_residues),
        )],
        forward_models=[],
        params=params,
    )
    # outputs[0] = BV_output_features (PF placeholder), outputs[1] = uptake
    sim.outputs = (
        BV_output_features(log_Pf=jnp.zeros(num_residues)),
        uptake_BV_output_features(uptake=uptake),
    )
    return sim


# ---------------------------------------------------------------------------
# Test 1: PF path — SparseFragmentMapping, flatten=True, backward compat
# ---------------------------------------------------------------------------


class TestPFRegressionSparseFragmentMapping:
    def test_pred_equals_target_loss_zero(self):
        """BV_output_features.y_pred() via SparseFragmentMapping → loss == 0."""
        log_pf = jnp.array([1.0, 2.0, 3.0])
        sim = _make_sim_pf(log_pf)

        y_true = log_pf
        identity_map = sparse.bcoo_fromdense(jnp.eye(3))
        dl = _MockDataloader(y_true, SparseFragmentMapping(identity_map), SparseFragmentMapping(identity_map))

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2), flatten=True
        )
        train_loss, val_loss = loss_fn(sim, dl, 0)
        np.testing.assert_allclose(float(train_loss), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(val_loss), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: Uptake path — SparseFragmentMapping, flatten=False
# ---------------------------------------------------------------------------


class TestUptakeRegressionSparseFragmentMapping:
    def test_pred_equals_target_loss_zero(self):
        """uptake_BV_output_features.y_pred() via SparseFragmentMapping → loss == 0."""
        uptake = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
        sim = _make_sim_uptake(uptake)

        # Identity map over residues
        identity_map = sparse.bcoo_fromdense(jnp.eye(3))
        y_true = uptake  # shape (2, 3)
        dl = _MockDataloader(y_true, SparseFragmentMapping(identity_map), SparseFragmentMapping(identity_map))

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2), flatten=False
        )
        train_loss, _ = loss_fn(sim, dl, 1)
        np.testing.assert_allclose(float(train_loss), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Identity mapping with QSubsetMapping, flatten=True → loss == 0
# ---------------------------------------------------------------------------


class TestIdentityMappingFlattenTrue:
    def test_identity_pred_eq_target(self):
        """QSubsetMapping identity + _DummyOutput.y_pred() → loss == 0."""
        values = jnp.array([1.0, 2.0, 3.0])
        output = _DummyOutput(values=values)

        params = Simulation_Parameters(
            frame_weights=jnp.array([1.0]),
            frame_mask=jnp.array([1.0]),
            model_parameters=[BV_Model_Parameters()],
            forward_model_weights=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
        )
        sim = Simulation(
            input_features=[BV_input_features(
                heavy_contacts=jnp.ones(3),
                acceptor_contacts=jnp.ones(3),
            )],
            forward_models=[],
            params=params,
        )
        sim.outputs = (output,)

        identity_map = QSubsetMapping(indices=jnp.arange(3))
        y_true = values
        dl = _MockDataloader(y_true, identity_map, identity_map)

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2), flatten=True
        )
        train_loss, _ = loss_fn(sim, dl, 0)
        np.testing.assert_allclose(float(train_loss), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Known loss value with QSubsetMapping
# ---------------------------------------------------------------------------


class TestIdentityMappingKnownLossValue:
    def test_known_l2_loss(self):
        """pred=[1,2,3], target=[1,1,1], L2 → mean((0,1,4)) = 5/3."""
        values = jnp.array([1.0, 2.0, 3.0])
        output = _DummyOutput(values=values)

        params = Simulation_Parameters(
            frame_weights=jnp.array([1.0]),
            frame_mask=jnp.array([1.0]),
            model_parameters=[BV_Model_Parameters()],
            forward_model_weights=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
        )
        sim = Simulation(
            input_features=[BV_input_features(
                heavy_contacts=jnp.ones(3),
                acceptor_contacts=jnp.ones(3),
            )],
            forward_models=[],
            params=params,
        )
        sim.outputs = (output,)

        identity_map = QSubsetMapping(indices=jnp.arange(3))
        y_true = jnp.array([1.0, 1.0, 1.0])
        dl = _MockDataloader(y_true, identity_map, identity_map)

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2),
            flatten=True,
            post_mean=False,  # raw sum/mean from loss_fn directly
        )
        train_loss, _ = loss_fn(sim, dl, 0)
        # mean((1-1)^2, (2-1)^2, (3-1)^2) = mean(0, 1, 4) = 5/3
        np.testing.assert_allclose(float(train_loss), 5.0 / 3.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: m_key string indexing — same result as int index 0
# ---------------------------------------------------------------------------


class TestMkeyStringIndexing:
    def test_string_index_eq_int_index(self):
        """prediction_index='HDX_resPF' gives same result as prediction_index=0."""
        log_pf = jnp.array([1.0, 2.0, 3.0])
        sim = _make_sim_pf(log_pf)

        y_true = jnp.array([1.0, 1.5, 2.5])
        identity_map = sparse.bcoo_fromdense(jnp.eye(3))
        dl = _MockDataloader(y_true, SparseFragmentMapping(identity_map), SparseFragmentMapping(identity_map))

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2), flatten=True
        )
        loss_int, _ = loss_fn(sim, dl, 0)
        loss_str, _ = loss_fn(sim, dl, "HDX_resPF")

        np.testing.assert_allclose(float(loss_int), float(loss_str), rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: m_key dispatches to correct output among multiple outputs
# ---------------------------------------------------------------------------


class TestMkeyDispatchesCorrectOutput:
    def test_different_keys_select_different_outputs(self):
        """Two outputs with different keys; string dispatch selects the right one."""
        log_pf = jnp.array([1.0, 2.0, 3.0])
        dummy_values = jnp.array([10.0, 20.0, 30.0])

        params = Simulation_Parameters(
            frame_weights=jnp.array([1.0]),
            frame_mask=jnp.array([1.0]),
            model_parameters=[BV_Model_Parameters()],
            forward_model_weights=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
        )
        sim = Simulation(
            input_features=[BV_input_features(
                heavy_contacts=jnp.ones(3),
                acceptor_contacts=jnp.ones(3),
            )],
            forward_models=[],
            params=params,
        )
        sim.outputs = (
            BV_output_features(log_Pf=log_pf),        # key = m_key("HDX_resPF"), index 0
            _DummyOutput(values=dummy_values),          # key = m_key("test_dummy"), index 1
        )

        y_true = jnp.array([1.0, 1.0, 1.0])
        identity_map = QSubsetMapping(indices=jnp.arange(3))
        dl = _MockDataloader(y_true, identity_map, identity_map)

        loss_fn = create_functional_loss(
            loss_fn=lambda p, t: jnp.mean((p - t) ** 2),
            flatten=True,
            post_mean=False,
        )

        loss_pf, _ = loss_fn(sim, dl, "HDX_resPF")
        loss_dummy, _ = loss_fn(sim, dl, "test_dummy")

        # PF: mean((1-1)^2, (2-1)^2, (3-1)^2) = 5/3
        np.testing.assert_allclose(float(loss_pf), 5.0 / 3.0, rtol=1e-5)
        # Dummy: mean((10-1)^2, (20-1)^2, (30-1)^2) = mean(81, 361, 841) = 1283/3
        np.testing.assert_allclose(float(loss_dummy), 1283.0 / 3.0, rtol=1e-4)
        assert float(loss_pf) != float(loss_dummy)


# ---------------------------------------------------------------------------
# Test 7: Builder regression after refactor
# ---------------------------------------------------------------------------


class TestBuilderRegressionAfterChange:
    def test_hdx_pf_l2_builder(self):
        """hdx_pf_l2_builder() still returns 0 when pred == target."""
        log_pf = jnp.array([1.0, 2.0, 3.0])
        sim = _make_sim_pf(log_pf)

        y_true = log_pf
        identity_map = sparse.bcoo_fromdense(jnp.eye(3))

        class _CompatDL:
            def __init__(self):
                self.train = Dataset(
                    data=[], y_true=y_true,
                    data_mapping=SparseFragmentMapping(identity_map),
                )
                self.val = Dataset(
                    data=[], y_true=y_true,
                    data_mapping=SparseFragmentMapping(identity_map),
                )
            def tree_flatten(self):
                return (self.train, self.val), {}
            @classmethod
            def tree_unflatten(cls, aux, children):
                obj = object.__new__(cls)
                obj.train, obj.val = children
                return obj

        jax.tree_util.register_pytree_node(
            _CompatDL,
            _CompatDL.tree_flatten,
            _CompatDL.tree_unflatten,
        )
        dl = _CompatDL()
        loss_fn = hdx_pf_l2_builder()
        train_loss, val_loss = loss_fn(sim, dl, 0)
        np.testing.assert_allclose(float(train_loss), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(val_loss), 0.0, atol=1e-5)

    def test_hdx_uptake_l2_builder(self):
        """hdx_uptake_l2_builder() still returns 0 when pred == target."""
        uptake = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
        sim = _make_sim_uptake(uptake)

        identity_map = sparse.bcoo_fromdense(jnp.eye(3))
        y_true = uptake  # shape (2, 3)
        dl = _MockDataloader(
            y_true,
            SparseFragmentMapping(identity_map),
            SparseFragmentMapping(identity_map),
        )

        loss_fn = hdx_uptake_l2_builder()
        train_loss, val_loss = loss_fn(sim, dl, 1)
        np.testing.assert_allclose(float(train_loss), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(val_loss), 0.0, atol=1e-5)
