import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.parameters import (
    BV_Model_Parameters,
    linear_BV_Model_Parameters,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.gradients import create_gradient_masks


def _step(model_parameters, model_grads):
    """Apply one optimizer update with zero frame gradients and the given model gradients."""
    n = len(model_parameters)
    params = Simulation_Parameters.from_frame_weights(
        jnp.ones(3) / 3,
        model_parameters=model_parameters,
        forward_model_weights=jnp.ones(n),
        normalise_loss_functions=jnp.zeros(n),
        forward_model_scaling=jnp.ones(n),
    )
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    grads = Simulation_Parameters(
        frame_weight_logits=grads.frame_weight_logits,
        model_parameters=model_grads,
        forward_model_weights=grads.forward_model_weights,
        normalise_loss_functions=grads.normalise_loss_functions,
        forward_model_scaling=grads.forward_model_scaling,
    )
    tx = OptaxOptimizer(optimizer="sgd", clip_value=None).optimizer
    updates, _ = tx.update(grads, tx.init(params), params)
    return optax.apply_updates(params, updates).model_parameters


def test_zero_gradient_leaves_negative_raw_linear_bv_parameters_unchanged():
    params = linear_BV_Model_Parameters(interval_offsets=jnp.asarray([-0.5, 0.0, 0.5]))
    assert params.raw_bv_bc < 0  # softplus^-1(0.35)
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)

    (result,) = _step([params], [zero])

    np.testing.assert_array_equal(result.raw_bv_bc, params.raw_bv_bc)
    np.testing.assert_array_equal(result.raw_bv_bh, params.raw_bv_bh)
    np.testing.assert_array_equal(result.interval_offsets, params.interval_offsets)
    np.testing.assert_allclose(result.bv_bc, 0.35, rtol=1e-6)


def test_linear_bv_raw_parameters_may_move_negative():
    params = linear_BV_Model_Parameters(bv_bc=0.35, bv_bh=2.0)
    grad = jax.tree_util.tree_map(jnp.ones_like, params)

    (result,) = _step([params], [grad])

    assert result.raw_bv_bh < params.raw_bv_bh
    assert jnp.all(result.interval_offsets < 0)


def test_physical_bv_parameters_are_projected_nonnegative():
    params = BV_Model_Parameters(bv_bc=jnp.asarray([0.35]), bv_bh=jnp.asarray([2.0]))
    grad = BV_Model_Parameters(bv_bc=jnp.asarray([1.0]), bv_bh=jnp.asarray([-1.0]))

    (result,) = _step([params], [grad])

    np.testing.assert_array_equal(result.bv_bc, [0.0])
    np.testing.assert_allclose(result.bv_bh, [3.0])


def test_linear_bv_mask_can_fit_only_contact_scalings():
    params = linear_BV_Model_Parameters(
        interval_offsets=jnp.asarray([-0.5, 0.0, 0.5])
    )
    simulation_params = Simulation_Parameters.from_frame_weights(
        jnp.ones(3) / 3,
        model_parameters=(params,),
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    mask = create_gradient_masks(
        {Optimisable_Parameters.frame_weights, Optimisable_Parameters.model_parameters},
        simulation_params,
        None,
        frozenset({"raw_bv_bc", "raw_bv_bh"}),
    ).model_parameters[0]

    np.testing.assert_array_equal(mask.raw_bv_bc, 1.0)
    np.testing.assert_array_equal(mask.raw_bv_bh, 1.0)
    np.testing.assert_array_equal(mask.interval_offsets, jnp.zeros(3))
