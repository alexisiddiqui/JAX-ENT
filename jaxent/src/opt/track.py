from collections.abc import Sequence
import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.opt.base import ConvergenceCarry


def create_convergence_thresholds(
    convergence: float | Sequence[float] | Array,
    learning_rate: float | Array,
) -> Array:
    """Create descending, LR-scaled convergence thresholds."""
    if isinstance(convergence, (float, int)):
        thresholds = jnp.array([convergence], dtype=jnp.float32)
    else:
        thresholds = jnp.asarray(convergence, dtype=jnp.float32)
    return jnp.sort(thresholds)[::-1] * jnp.asarray(learning_rate, dtype=jnp.float32)


def initialise_convergence_carry() -> ConvergenceCarry:
    """Initialise pure convergence carry."""
    return ConvergenceCarry(
        ema_loss_delta=jnp.array(0.0, dtype=jnp.float32),
        steps_since_threshold_start=jnp.array(0, dtype=jnp.int32),
        current_threshold_idx=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(False),
    )


def get_relative_convergence(carry: ConvergenceCarry, current_loss: Array) -> Array:
    current_loss = jnp.asarray(current_loss, dtype=jnp.float32)
    return jnp.where(current_loss > 0, carry.ema_loss_delta / current_loss, 0.0)


def update_convergence(
    carry: ConvergenceCarry,
    previous_loss: Array,
    current_loss: Array,
    ema_alpha: Array | float,
) -> tuple[ConvergenceCarry, Array]:
    """Update convergence EMA state and return raw loss delta."""
    previous_loss = jnp.asarray(previous_loss, dtype=jnp.float32)
    current_loss = jnp.asarray(current_loss, dtype=jnp.float32)
    raw_loss_delta = jnp.abs(previous_loss - current_loss)

    first_update = carry.steps_since_threshold_start == 0
    ema_alpha = jnp.asarray(ema_alpha, dtype=jnp.float32)

    new_ema_loss_delta = jax.lax.select(
        first_update,
        raw_loss_delta,
        ema_alpha * raw_loss_delta + (1.0 - ema_alpha) * carry.ema_loss_delta,
    )
    new_carry = ConvergenceCarry(
        ema_loss_delta=new_ema_loss_delta,
        steps_since_threshold_start=carry.steps_since_threshold_start + 1,
        current_threshold_idx=carry.current_threshold_idx,
        converged=carry.converged,
    )
    return new_carry, raw_loss_delta


def check_and_advance_threshold(
    carry: ConvergenceCarry,
    current_loss: Array,
    thresholds: Array,
    min_steps: int,
) -> tuple[ConvergenceCarry, Array]:
    """Advance thresholds and return the pre-advance threshold value."""
    threshold_idx = jnp.minimum(
        carry.current_threshold_idx,
        jnp.array(thresholds.shape[0] - 1, dtype=jnp.int32),
    )
    current_threshold = thresholds[threshold_idx]
    relative_convergence = get_relative_convergence(carry, current_loss)

    threshold_met = (
        (carry.steps_since_threshold_start >= jnp.array(min_steps, dtype=jnp.int32))
        & (relative_convergence < current_threshold)
    )

    next_idx = carry.current_threshold_idx + jnp.array(1, dtype=jnp.int32)
    has_more_thresholds = next_idx < jnp.array(thresholds.shape[0], dtype=jnp.int32)

    advance_threshold = threshold_met & has_more_thresholds
    converged = carry.converged | (threshold_met & ~has_more_thresholds)

    updated_threshold_idx = jax.lax.select(
        advance_threshold,
        next_idx,
        carry.current_threshold_idx,
    )
    updated_steps = jax.lax.select(
        advance_threshold,
        jnp.array(0, dtype=jnp.int32),
        carry.steps_since_threshold_start,
    )

    new_carry = ConvergenceCarry(
        ema_loss_delta=carry.ema_loss_delta,
        steps_since_threshold_start=updated_steps,
        current_threshold_idx=updated_threshold_idx,
        converged=converged,
    )
    return new_carry, current_threshold
