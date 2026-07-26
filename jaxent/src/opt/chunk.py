"""Pure-JAX chunked optimisation runner."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from beartype.typing import NamedTuple

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import (
    ConvergenceCarry,
    InitialisedSimulation,
    JaxEnt_Loss,
    LossComponents,
    OptimizationState,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.track import check_and_advance_threshold, update_convergence


class StateSnapshot(NamedTuple):
    params: Simulation_Parameters
    losses: LossComponents
    step: Any


class ChunkInputs(NamedTuple):
    data_targets: Any
    convergence_thresholds: Any
    tolerance: Any
    ema_alpha: Any


class ChunkCarry(NamedTuple):
    opt_state: OptimizationState
    sim: InitialisedSimulation
    convergence: ConvergenceCarry
    lr: Any
    model_lr: Any
    executed_steps: Any
    active: Any
    best: StateSnapshot


class StepMetrics(NamedTuple):
    total_train_loss: Any
    total_val_loss: Any
    lr: Any
    grad_dot_product: Any
    executed: Any


class ChunkRecord(NamedTuple):
    step: Any
    params: Simulation_Parameters
    losses: LossComponents
    lr: Any
    threshold_idx: Any
    threshold_event: Any
    active: Any


class ChunkResult(NamedTuple):
    carry: ChunkCarry
    records: ChunkRecord
    metrics: StepMetrics


def _select_snapshot(
    old: StateSnapshot,
    new: StateSnapshot,
    use_new: Array,
) -> StateSnapshot:
    return StateSnapshot(
        params=jax.tree_util.tree_map(
            lambda previous, current: jax.lax.select(use_new, current, previous),
            old.params,
            new.params,
        ),
        losses=jax.tree_util.tree_map(
            lambda previous, current: jax.lax.select(use_new, current, previous),
            old.losses,
            new.losses,
        ),
        step=jax.lax.select(use_new, new.step, old.step),
    )


def _losses_or_default(state: OptimizationState, fallback: Array) -> LossComponents:
    if state.losses is not None:
        return state.losses
    return LossComponents(
        train_losses=jnp.asarray([fallback]),
        val_losses=jnp.asarray([fallback]),
        scaled_train_losses=jnp.asarray([fallback]),
        scaled_val_losses=jnp.asarray([fallback]),
        total_train_loss=fallback,
        total_val_loss=fallback,
    )


def _no_op_step(carry: ChunkCarry) -> tuple[ChunkCarry, StepMetrics]:
    losses = _losses_or_default(carry.opt_state, jnp.asarray(jnp.nan, dtype=jnp.float32))
    return carry, StepMetrics(
        total_train_loss=jnp.asarray(losses.total_train_loss),
        total_val_loss=jnp.asarray(losses.total_val_loss),
        lr=carry.lr,
        grad_dot_product=jnp.asarray(0.0, dtype=jnp.float32),
        executed=jnp.asarray(False),
    )


def optimisation_step(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    min_steps_per_threshold: int,
) -> tuple[ChunkCarry, StepMetrics]:
    """Execute one side-effect-free optimisation step."""

    def step(active_carry: ChunkCarry) -> tuple[ChunkCarry, StepMetrics]:
        (
            new_state,
            loss_value,
            save_state,
            _updated_sim,
            new_lr,
            new_model_lr,
            grad_dot_product,
        ) = OptaxOptimizer._step_with_rates(
            optimizer=optimizer,
            state=active_carry.opt_state,
            simulation=active_carry.sim,
            data_targets=inputs.data_targets,
            loss_functions=loss_functions,
            indexes=indexes,
            lr=active_carry.lr,
            model_lr=active_carry.model_lr,
        )

        losses = save_state.losses
        if losses is None:
            raise ValueError("save_state.losses cannot be None")

        previous_loss = (
            active_carry.opt_state.losses.total_train_loss
            if active_carry.opt_state.losses is not None
            else loss_value
        )
        new_convergence, _ = update_convergence(
            carry=active_carry.convergence,
            previous_loss=previous_loss,
            current_loss=loss_value,
            ema_alpha=inputs.ema_alpha,
        )

        finite = jnp.isfinite(loss_value)
        next_active = (
            active_carry.active
            & finite
            & (loss_value >= inputs.tolerance)
            & ~active_carry.convergence.converged
        )
        candidate = StateSnapshot(
            params=save_state.params,
            losses=losses,
            step=jnp.asarray(new_state.step, dtype=jnp.int32),
        )
        better = losses.val_losses[0] < active_carry.best.losses.val_losses[0]
        next_best = _select_snapshot(active_carry.best, candidate, better & finite)

        finite_carry = ChunkCarry(
            opt_state=new_state,
            sim=active_carry.sim,
            convergence=new_convergence,
            lr=new_lr,
            model_lr=new_model_lr,
            executed_steps=active_carry.executed_steps + 1,
            active=next_active,
            best=next_best,
        )
        frozen_carry = ChunkCarry(
            opt_state=active_carry.opt_state,
            sim=active_carry.sim,
            convergence=active_carry.convergence,
            lr=active_carry.lr,
            model_lr=active_carry.model_lr,
            executed_steps=active_carry.executed_steps + 1,
            active=jnp.asarray(False),
            best=active_carry.best,
        )
        next_carry = jax.tree_util.tree_map(
            lambda good, frozen: jax.lax.select(finite, good, frozen),
            finite_carry,
            frozen_carry,
        )
        return next_carry, StepMetrics(
            total_train_loss=jnp.asarray(losses.total_train_loss),
            total_val_loss=jnp.asarray(losses.total_val_loss),
            lr=new_lr,
            grad_dot_product=grad_dot_product,
            executed=jnp.asarray(True),
        )

    return jax.lax.cond(carry.active, step, _no_op_step, carry)


def evaluate_convergence(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    min_steps_per_threshold: int,
) -> tuple[ChunkCarry, Array]:
    old_convergence = carry.convergence
    current_loss = carry.opt_state.losses.total_train_loss
    new_convergence = check_and_advance_threshold(
        carry=old_convergence,
        current_loss=current_loss,
        thresholds=inputs.convergence_thresholds,
        min_steps=min_steps_per_threshold,
    )
    threshold_event = (
        (new_convergence.current_threshold_idx != old_convergence.current_threshold_idx)
        | (~old_convergence.converged & new_convergence.converged)
    )
    return carry._replace(
        convergence=new_convergence,
        active=carry.active & ~new_convergence.converged,
    ), threshold_event


def _make_record(carry: ChunkCarry, threshold_event: Array, boundary_active: Array) -> ChunkRecord:
    losses = carry.opt_state.losses
    if losses is None:
        raise ValueError("Chunk carry must contain losses")
    return ChunkRecord(
        step=jnp.asarray(carry.opt_state.step, dtype=jnp.int32),
        params=carry.opt_state.params,
        losses=losses,
        lr=carry.lr,
        threshold_idx=carry.convergence.current_threshold_idx,
        threshold_event=threshold_event,
        active=boundary_active,
    )


@partial(
    jax.jit,
    static_argnames=(
        "optimizer",
        "loss_functions",
        "indexes",
        "chunk_size",
        "min_steps_per_threshold",
    ),
)
def run_step_chunk(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    chunk_size: int,
    min_steps_per_threshold: int,
) -> tuple[ChunkCarry, StepMetrics, ChunkRecord]:
    def scan_step(current: ChunkCarry, _unused: None) -> tuple[ChunkCarry, StepMetrics]:
        return optimisation_step(
            current,
            inputs,
            optimizer,
            loss_functions,
            indexes,
            min_steps_per_threshold,
        )

    carry, metrics = jax.lax.scan(scan_step, carry, None, length=chunk_size)
    carry, threshold_event = evaluate_convergence(
        carry, inputs, min_steps_per_threshold
    )
    boundary_active = jnp.any(metrics.executed)
    record = _make_record(carry, threshold_event, boundary_active)
    return carry, metrics, record


def _stack_records(records: list[ChunkRecord]) -> ChunkRecord:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def _concat_metrics(metrics: list[StepMetrics]) -> StepMetrics:
    return jax.tree_util.tree_map(lambda *values: jnp.concatenate(values), *metrics)


def run_chunks(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    n_steps: int,
    chunk_size: int,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    min_steps_per_threshold: int,
) -> ChunkResult:
    records: list[ChunkRecord] = []
    metrics: list[StepMetrics] = []
    remaining = n_steps
    while remaining > 0:
        current_size = min(chunk_size, remaining)
        carry, chunk_metrics, record = run_step_chunk(
            carry,
            inputs,
            optimizer,
            loss_functions,
            indexes,
            current_size,
            min_steps_per_threshold,
        )
        records.append(record)
        metrics.append(chunk_metrics)
        remaining -= current_size
    return ChunkResult(carry, _stack_records(records), _concat_metrics(metrics))


def run_sequential(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    n_steps: int,
    chunk_size: int,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    min_steps_per_threshold: int,
) -> ChunkResult:
    return run_chunks(
        carry,
        inputs,
        n_steps,
        chunk_size,
        optimizer,
        loss_functions,
        indexes,
        min_steps_per_threshold,
    )


def run_batch(
    carries: ChunkCarry,
    inputs: ChunkInputs,
    data_targets: tuple[Any, ...],
    n_steps: int,
    chunk_size: int,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    min_steps_per_threshold: int,
) -> ChunkResult:
    input_axes = ChunkInputs(
        data_targets=None,
        convergence_thresholds=0,
        tolerance=0,
        ema_alpha=0,
    )

    def one(carry: ChunkCarry, lane_inputs: ChunkInputs) -> ChunkResult:
        lane_inputs = lane_inputs._replace(data_targets=data_targets)
        return run_chunks(
            carry,
            lane_inputs,
            n_steps,
            chunk_size,
            optimizer,
            loss_functions,
            indexes,
            min_steps_per_threshold,
        )

    return jax.vmap(one, in_axes=(0, input_axes))(carries, inputs)
