from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.interfaces.simulation import Simulation_Parameters
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


def initialise_convergence_carry(initial_params: Simulation_Parameters) -> ConvergenceCarry:
    """Initialise pure convergence carry."""
    return ConvergenceCarry(
        ema_loss_delta=jnp.array(0.0, dtype=jnp.float32),
        ema_params=initial_params,
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
    current_params: Simulation_Parameters,
    ema_alpha: float,
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
    new_ema_params = jax.tree_util.tree_map(
        lambda ema, current: jax.lax.select(
            first_update,
            current,
            ema_alpha * current + (1.0 - ema_alpha) * ema,
        ),
        carry.ema_params,
        current_params,
    )

    new_carry = ConvergenceCarry(
        ema_loss_delta=new_ema_loss_delta,
        ema_params=new_ema_params,
        steps_since_threshold_start=carry.steps_since_threshold_start + 1,
        current_threshold_idx=carry.current_threshold_idx,
        converged=carry.converged,
    )
    return new_carry, raw_loss_delta


def check_and_advance_threshold(
    carry: ConvergenceCarry,
    current_loss: Array,
    step: Array,
    thresholds: Array,
    min_steps: int,
    initial_steps: int,
) -> ConvergenceCarry:
    """Advance convergence thresholds with JAX control flow only."""
    threshold_idx = jnp.minimum(
        carry.current_threshold_idx,
        jnp.array(thresholds.shape[0] - 1, dtype=jnp.int32),
    )
    current_threshold = thresholds[threshold_idx]
    relative_convergence = get_relative_convergence(carry, current_loss)

    threshold_met = (
        (carry.steps_since_threshold_start >= jnp.array(min_steps, dtype=jnp.int32))
        & (relative_convergence < current_threshold)
        & (step > jnp.array(initial_steps, dtype=jnp.int32))
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

    return ConvergenceCarry(
        ema_loss_delta=carry.ema_loss_delta,
        ema_params=carry.ema_params,
        steps_since_threshold_start=updated_steps,
        current_threshold_idx=updated_threshold_idx,
        converged=converged,
    )


class ConvergenceTracker:
    """Compatibility wrapper used by the Python-loop optimisation path."""

    def __init__(
        self,
        convergence: list[float],
        learning_rate: float,
        ema_alpha: float,
        min_steps_per_threshold: int,
    ):
        self.ema_alpha = ema_alpha
        self.min_steps_per_threshold = min_steps_per_threshold
        self.convergence_thresholds = list(
            create_convergence_thresholds(convergence, learning_rate).tolist()
        )
        self.current_threshold_idx = 0
        self.current_threshold = self.convergence_thresholds[self.current_threshold_idx]

        self.ema_loss_delta: Optional[Any] = None
        self.ema_params: Optional[Any] = None
        self.steps_since_threshold_start = 0

    def update(self, previous_loss: Optional[Any], current_loss: Any, current_params: Any) -> Any:
        if previous_loss is not None:
            raw_loss_delta = jnp.abs(previous_loss - current_loss)

            if self.ema_loss_delta is None or self.ema_params is None:
                self.ema_loss_delta = raw_loss_delta
                self.ema_params = current_params
            else:
                self.ema_loss_delta = (
                    self.ema_alpha * raw_loss_delta + (1 - self.ema_alpha) * self.ema_loss_delta
                )
                self.ema_params = (
                    self.ema_alpha * current_params + (1 - self.ema_alpha) * self.ema_params
                )
        else:
            raw_loss_delta = 0.0

        self.steps_since_threshold_start += 1
        return raw_loss_delta

    def get_relative_convergence(self, current_loss: Any) -> float:
        if self.ema_loss_delta is not None and current_loss > 0:
            return float(self.ema_loss_delta / current_loss)
        return 0.0

    def is_threshold_met(self, current_loss: Any, step: int, initial_steps: int) -> bool:
        if (
            self.steps_since_threshold_start >= self.min_steps_per_threshold
            and self.ema_loss_delta is not None
            and self.get_relative_convergence(current_loss) < self.current_threshold
            and step > initial_steps
        ):
            return True
        return False

    def advance_threshold(self) -> bool:
        """Advance to next threshold. Returns False when all thresholds are complete."""
        self.current_threshold_idx += 1
        self.steps_since_threshold_start = 0

        if self.current_threshold_idx >= len(self.convergence_thresholds):
            return False

        self.current_threshold = self.convergence_thresholds[self.current_threshold_idx]
        return True

    def reset_threshold_steps(self) -> None:
        self.steps_since_threshold_start = 0
