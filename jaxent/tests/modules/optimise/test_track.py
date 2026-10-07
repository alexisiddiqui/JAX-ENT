import jax.numpy as jnp

from jaxent.src.opt.base import ConvergenceCarry
from jaxent.src.opt.track import (
    check_and_advance_threshold,
    create_convergence_thresholds,
    reset_threshold_cooldown,
    update_convergence,
)


def _carry(index: int) -> ConvergenceCarry:
    return ConvergenceCarry(
        ema_loss_delta=jnp.array(0.0, dtype=jnp.float32),
        ema_initialized=jnp.array(True),
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


def test_oscillation_reset_preserves_ema_and_threshold_index():
    carry = _carry(1)._replace(ema_loss_delta=jnp.array(0.25, dtype=jnp.float32))
    reset = reset_threshold_cooldown(carry, True)

    assert int(reset.steps_since_threshold_start) == 0
    assert int(reset.current_threshold_idx) == 1
    assert float(reset.ema_loss_delta) == float(carry.ema_loss_delta)

    updated, _ = update_convergence(reset, 2.0, 1.8, 0.5)
    assert jnp.isclose(updated.ema_loss_delta, 0.225)


def test_first_step_only_starts_cooldown_without_initialising_ema():
    carry = _carry(0)._replace(
        ema_loss_delta=jnp.array(0.0),
        ema_initialized=jnp.array(False),
        steps_since_threshold_start=jnp.array(0),
    )

    first, _ = update_convergence(carry, 1.0, 1.0, 0.5, update_ema=False)
    second, _ = update_convergence(first, 1.0, 0.8, 0.5)

    assert not bool(first.ema_initialized)
    assert int(first.steps_since_threshold_start) == 1
    assert bool(second.ema_initialized)
    assert jnp.isclose(second.ema_loss_delta, 0.2)
