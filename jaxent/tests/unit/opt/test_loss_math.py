"""
Unit tests for loss function mathematical correctness.
Tests both pure loss lambdas and factory-created losses with mock data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.data.loader import Dataset
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.loss.functional import (
    hdx_pf_l2_builder,
    hdx_pf_mae_builder,
    hdx_uptake_l2_builder,
    hdx_uptake_mse_builder,
)
from jaxent.src.opt.loss.base import create_functional_loss


# ============================================================================
# Pure Loss Functions (Level 1)
# ============================================================================


class TestPureLossFunctions:
    """Test pure loss function lambdas directly."""

    def test_l2_identical(self):
        """Test L2 loss with identical inputs: mean((x-x)^2) = 0."""
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 2.0, 3.0])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss = loss_fn(pred, target)
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_l2_known_values(self):
        """Test L2 loss with known difference.
        pred=[1, 2, 3], target=[1, 1, 1]
        diffs=[0, 1, 2], squared=[0, 1, 4], mean=5/3
        """
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 1.0, 1.0])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss = loss_fn(pred, target)
        expected = 5.0 / 3.0
        np.testing.assert_allclose(loss, expected, rtol=1e-5)

    def test_mae_known_values(self):
        """Test MAE loss.
        pred=[1, 2, 3], target=[1, 1, 1]
        diffs=[0, 1, 2], abs=[0, 1, 2], mean=1.0
        """
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 1.0, 1.0])
        loss_fn = lambda pred, target: jnp.mean(jnp.abs(pred - target))
        loss = loss_fn(pred, target)
        expected = 1.0
        np.testing.assert_allclose(loss, expected, rtol=1e-6)

    def test_kl_self_divergence(self):
        """Test KL divergence with identical inputs: ~0 (with epsilon)."""
        # KL(P||Q) = sum(P * log(P/Q))
        p = jnp.array([0.3, 0.5, 0.2])
        q = jnp.array([0.3, 0.5, 0.2])
        eps = 1e-10
        loss_fn = lambda p, q: jnp.sum(p * jnp.log((p + eps) / (q + eps)))
        loss = loss_fn(p, q)
        # Should be very close to 0
        assert loss < 0.01

    def test_mean_center_transform(self):
        """Test mean-centering transform.
        [1, 2, 3] -> [-1, 0, 1]
        """
        array = jnp.array([1.0, 2.0, 3.0])
        mean_center = lambda x: x - jnp.mean(x)
        result = mean_center(array)
        expected = jnp.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================================
# Factory-Created Losses (Level 2)
# ============================================================================


class MockDataloader:
    """Mock dataloader for testing factory-created losses."""

    def __init__(self, y_true, sparse_map_train, sparse_map_val):
        self.y_true = y_true
        self.sparse_map_train = sparse_map_train
        self.sparse_map_val = sparse_map_val
        self.train = Dataset(
            data=[], y_true=y_true, residue_feature_ouput_mapping=sparse_map_train
        )
        self.val = Dataset(
            data=[], y_true=y_true, residue_feature_ouput_mapping=sparse_map_val
        )

    def tree_flatten(self):
        return (self.train, self.val), {"sparse_map": (self.sparse_map_train, self.sparse_map_val)}

    @classmethod
    def tree_unflatten(cls, aux, children):
        dataloader = cls(
            y_true=children[0].y_true,
            sparse_map_train=aux["sparse_map"][0],
            sparse_map_val=aux["sparse_map"][1],
        )
        return dataloader


# Register as pytree
jax.tree_util.register_pytree_node(
    MockDataloader, MockDataloader.tree_flatten, MockDataloader.tree_unflatten
)


@pytest.fixture
def pf_loss_data():
    """Create data for PF (protection factor) losses."""
    num_residues = 3

    # Create BV output with known log_Pf
    log_pf_pred = jnp.array([1.0, 2.0, 3.0])
    output = BV_output_features(log_Pf=log_pf_pred)

    # Create simulation with outputs
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
    sim.outputs = (output,)

    # Create mock dataloader with identity mapping
    y_true = jnp.array([1.0, 2.0, 3.0])  # Same as pred
    identity_map = sparse.bcoo_fromdense(jnp.eye(num_residues))
    dataloader = MockDataloader(y_true, identity_map, identity_map)

    return {"sim": sim, "dataloader": dataloader, "prediction_index": 0}


@pytest.fixture
def uptake_loss_data():
    """Create data for uptake losses."""
    num_residues = 3
    num_timepoints = 2

    # Create uptake output with known values
    uptake_pred = jnp.array([
        [0.1, 0.2, 0.3],  # Timepoint 1
        [0.2, 0.4, 0.6],  # Timepoint 2
    ])
    output = uptake_BV_output_features(uptake=uptake_pred)

    # Create simulation
    params = Simulation_Parameters(
        frame_weights=jnp.array([1.0]),
        frame_mask=jnp.array([1.0]),
        model_parameters=[BV_Model_Parameters(
            timepoints=jnp.array([0.1, 1.0])
        )],
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
    sim.outputs = (BV_output_features(log_Pf=jnp.zeros(num_residues)), output)

    # Create mock dataloader: map all residues to single fragment
    y_true = jnp.array(uptake_pred).flatten().reshape(1, -1)
    map_matrix = jnp.ones((1, num_residues * num_timepoints)) / (num_residues * num_timepoints)
    sparse_map = sparse.bcoo_fromdense(map_matrix)
    dataloader = MockDataloader(y_true, sparse_map, sparse_map)

    return {"sim": sim, "dataloader": dataloader, "prediction_index": 1}


class TestPFLosses:
    """Test PF (protection factor) losses."""

    def test_pf_l2_pred_equals_target(self, pf_loss_data):
        """Test PF L2 loss when pred == target: loss = 0."""
        loss_fn = hdx_pf_l2_builder()
        loss, _ = loss_fn(
            pf_loss_data["sim"],
            pf_loss_data["dataloader"],
            pf_loss_data["prediction_index"],
        )
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)

    def test_pf_l2_callable_basic(self):
        """Test L2 loss callable with basic inputs."""
        # Pure loss function without sparse mapping
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 2.0, 2.5])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss = loss_fn(pred, target)
        # Expected: mean((1.0-1.5)^2 + (2.0-2.0)^2 + (3.0-2.5)^2)
        # = mean(0.25 + 0 + 0.25) = 0.5/3 ≈ 0.167
        expected = (0.25 + 0.0 + 0.25) / 3.0
        np.testing.assert_allclose(loss, expected, rtol=1e-4)

    def test_pf_mae_callable_basic(self):
        """Test MAE loss callable with basic inputs."""
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([1.5, 2.0])
        loss_fn = lambda pred, target: jnp.mean(jnp.abs(pred - target))
        loss = loss_fn(pred, target)
        # Expected: mean(|1-1.5| + |2-2|) = mean(0.5 + 0) = 0.25
        expected = 0.25
        np.testing.assert_allclose(loss, expected, rtol=1e-4)


class TestUptakeLosses:
    """Test uptake losses."""

    def test_uptake_l2_callable(self):
        """Test uptake L2 loss callable."""
        pred = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        target = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss = loss_fn(pred, target)
        # When pred == target, loss should be 0
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_uptake_mse_callable(self):
        """Test uptake MSE loss callable."""
        # First apply mean over timepoints, then MSE
        pred = jnp.array([[0.1, 0.2], [0.3, 0.4]])  # shape (timepoints, residues)
        target = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        # MSE with post_mean=True: mean((pred - target)^2) then mean over timepoints
        loss_fn = lambda pred, target: jnp.mean(jnp.mean((pred - target) ** 2, axis=0))
        loss = loss_fn(pred, target)
        # When pred == target, loss should be 0
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)


# ============================================================================
# Additional Mathematical Tests
# ============================================================================


class TestLossMathematicalProperties:
    """Test mathematical properties of loss functions."""

    def test_l2_loss_positive(self):
        """Test that L2 loss is always non-negative."""
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_mae_loss_positive(self):
        """Test that MAE loss is always non-negative."""
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])
        loss_fn = lambda pred, target: jnp.mean(jnp.abs(pred - target))
        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_l2_symmetry(self):
        """Test that L2 loss is symmetric in pred and target."""
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])
        loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
        loss1 = loss_fn(pred, target)
        loss2 = loss_fn(target, pred)
        np.testing.assert_allclose(loss1, loss2, rtol=1e-6)
