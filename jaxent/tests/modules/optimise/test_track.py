import jax.numpy as jnp

from jaxent.src.opt.base import ConvergenceCarry
from jaxent.src.opt.track import check_and_advance_threshold, create_convergence_thresholds


def _carry(index: int) -> ConvergenceCarry:
    return ConvergenceCarry(
        ema_loss_delta=jnp.array(0.0, dtype=jnp.float32),
        steps_since_threshold_start=jnp.array(10, dtype=jnp.int32),
        current_threshold_idx=jnp.array(index, dtype=jnp.int32),
        converged=jnp.array(False),
    )


def test_crossed_threshold_is_pre_advance_value():
    thresholds = create_convergence_thresholds([1e-2, 1e-3, 1e-4], 1.0)
    carry = _carry(0)._replace(ema_loss_delta=jnp.array(1e-3, dtype=jnp.float32))
    new_carry, crossed = check_and_advance_threshold(carry, 1.0, thresholds, 2)
    assert int(new_carry.current_threshold_idx) == 1
    assert float(crossed) == float(thresholds[0])


def test_crossed_threshold_is_returned_on_final_step():
    thresholds = create_convergence_thresholds([1e-2, 1e-3], 1.0)
    carry = _carry(1)._replace(ema_loss_delta=jnp.array(1e-5, dtype=jnp.float32))
    new_carry, crossed = check_and_advance_threshold(carry, 1.0, thresholds, 2)
    assert bool(new_carry.converged)
    assert float(crossed) == float(thresholds[1])
