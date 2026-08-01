"""Pure-JAX chunked optimisation runner."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from beartype.typing import NamedTuple

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.opt.base import (
    ConvergenceCarry,
    InitialisedSimulation,
    JaxEnt_Loss,
    LossComponents,
    OptimizationState,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.track import (
    check_and_advance_threshold,
    reset_threshold_cooldown,
    update_convergence,
)


class StateSnapshot(NamedTuple):
    params: Simulation_Parameters
    losses: LossComponents
    step: Any


class ConvergenceSnapshots(NamedTuple):
    params: Simulation_Parameters
    losses: LossComponents
    steps: Any
    thresholds: Any
    valid: Any


class ChunkInputs(NamedTuple):
    data_targets: Any
    convergence_thresholds: Any
    tolerance: Any
    ema_alpha: Any
    base_lr: Any
    base_model_lr: Any


class ChunkCarry(NamedTuple):
    opt_state: OptimizationState
    sim: InitialisedSimulation
    convergence: ConvergenceCarry
    lr: Any
    model_lr: Any
    executed_steps: Any
    active: Any
    best: StateSnapshot
    convergence_snapshots: ConvergenceSnapshots


class StepMetrics(NamedTuple):
    total_train_loss: Any
    total_val_loss: Any
    lr: Any
    grad_dot_product: Any
    executed: Any
    threshold_event: Any
    crossed_threshold: Any
    threshold_slot: Any


class ChunkRecord(NamedTuple):
    step: Any
    params: Simulation_Parameters
    losses: LossComponents
    lr: Any
    threshold_idx: Any
    threshold_event: Any
    crossed_threshold: Any
    active: Any


class ChunkResult(NamedTuple):
    carry: ChunkCarry
    records: ChunkRecord
    metrics: StepMetrics


_EMPTY = jnp.zeros((0,), dtype=jnp.float32)


def _select_partitions(
    params: Simulation_Parameters,
    partitions: frozenset[Optimisable_Parameters] | None,
) -> Simulation_Parameters:
    """Replace unselected optimisable partitions with zero-size arrays before stacking."""
    if partitions is None:
        return params

    def select(value, partition):
        return value if partition in partitions else _EMPTY

    model_parameters = (
        params.model_parameters
        if Optimisable_Parameters.model_parameters in partitions
        else [jax.tree_util.tree_map(lambda _: _EMPTY, model) for model in params.model_parameters]
    )
    return Simulation_Parameters(
        frame_weight_logits=select(
            params.frame_weight_logits, Optimisable_Parameters.frame_weights
        ),
        model_parameters=model_parameters,
        forward_model_weights=select(
            params.forward_model_weights, Optimisable_Parameters.forward_model_weights
        ),
        # These are not optimisation partitions and are always retained.
        normalise_loss_functions=params.normalise_loss_functions,
        forward_model_scaling=params.forward_model_scaling,
    )


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


def _select_carry(predicate: Array, stepped: ChunkCarry, frozen: ChunkCarry) -> ChunkCarry:
    """Select dynamic carry fields; pass the immutable simulation through by reference.

    ``sim`` holds the input feature arrays and never changes during a step. Including
    it in the tree-map would allocate a full copy of those arrays on every step,
    which is the dominant source of eager-loop memory growth.
    """

    def sel(a, b):
        return jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(predicate, x, y), a, b
        )

    return ChunkCarry(
        opt_state=sel(stepped.opt_state, frozen.opt_state),
        sim=stepped.sim,
        convergence=sel(stepped.convergence, frozen.convergence),
        lr=jax.lax.select(predicate, stepped.lr, frozen.lr),
        model_lr=jax.lax.select(predicate, stepped.model_lr, frozen.model_lr),
        executed_steps=jax.lax.select(
            predicate, stepped.executed_steps, frozen.executed_steps
        ),
        active=jax.lax.select(predicate, stepped.active, frozen.active),
        best=sel(stepped.best, frozen.best),
        convergence_snapshots=sel(
            stepped.convergence_snapshots, frozen.convergence_snapshots
        ),
    )


def initialise_convergence_snapshots(
    params: Simulation_Parameters,
    losses: LossComponents,
    n_thresholds: int,
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_params: bool = True,
) -> ConvergenceSnapshots:
    """Allocate a fixed-size snapshot bank, one slot per convergence threshold."""
    selected_params = _select_partitions(
        params,
        parameter_partitions if retain_params else frozenset(),
    )
    return ConvergenceSnapshots(
        params=jax.tree_util.tree_map(
            lambda value: jnp.zeros(
                (n_thresholds,) + value.shape,
                dtype=value.dtype,
            ),
            selected_params,
        ),
        losses=jax.tree_util.tree_map(
            lambda value: jnp.zeros(
                (n_thresholds,) + value.shape,
                dtype=value.dtype,
            ),
            losses,
        ),
        steps=jnp.full((n_thresholds,), -1, dtype=jnp.int32),
        thresholds=jnp.zeros((n_thresholds,), dtype=jnp.float32),
        valid=jnp.zeros((n_thresholds,), dtype=jnp.bool_),
    )


def _store_convergence_snapshot(
    snapshots: ConvergenceSnapshots,
    params: Simulation_Parameters,
    losses: LossComponents,
    step: Any,
    threshold: Array,
    slot: Array,
    event: Array,
) -> ConvergenceSnapshots:
    """Conditionally store a threshold state without materialising per-step history."""
    max_slot = snapshots.valid.shape[0] - 1
    slot = jnp.clip(jnp.asarray(slot, dtype=jnp.int32), 0, max_slot)

    def store(buffer, value):
        updated = jax.lax.dynamic_update_index_in_dim(
            buffer, value, slot, axis=0
        )
        return jax.lax.select(event, updated, buffer)

    return ConvergenceSnapshots(
        params=jax.tree_util.tree_map(store, snapshots.params, params),
        losses=jax.tree_util.tree_map(store, snapshots.losses, losses),
        steps=store(snapshots.steps, jnp.asarray(step, dtype=jnp.int32)),
        thresholds=store(
            snapshots.thresholds, jnp.asarray(threshold, dtype=jnp.float32)
        ),
        valid=store(snapshots.valid, jnp.asarray(True)),
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
        threshold_event=jnp.asarray(False),
        crossed_threshold=jnp.asarray(jnp.nan, dtype=jnp.float32),
        threshold_slot=jnp.asarray(-1, dtype=jnp.int32),
    )


def optimisation_step(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    optimizer: OptaxOptimizer,
    loss_functions: tuple[JaxEnt_Loss, ...],
    indexes: tuple[int, ...],
    min_steps_per_threshold: int,
    reset_threshold_cooldown_on_oscillation: bool = True,
) -> tuple[ChunkCarry, StepMetrics]:
    """Execute one side-effect-free optimisation step."""

    def step(active_carry: ChunkCarry) -> tuple[ChunkCarry, StepMetrics]:
        (
            new_state,
            loss_value,
            save_state,
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
            base_lr=inputs.base_lr,
            base_model_lr=inputs.base_model_lr,
        )

        losses = save_state.losses
        if losses is None:
            raise ValueError("save_state.losses cannot be None")

        previous_loss = (
            active_carry.opt_state.losses.total_train_loss
            if active_carry.opt_state.losses is not None
            else loss_value
        )
        updated_convergence, _ = update_convergence(
            carry=active_carry.convergence,
            previous_loss=previous_loss,
            current_loss=loss_value,
            ema_alpha=inputs.ema_alpha,
            update_ema=jnp.asarray(active_carry.opt_state.step) > 0,
        )

        finite = jnp.isfinite(loss_value)
        eligible = finite & (loss_value >= inputs.tolerance)
        if reset_threshold_cooldown_on_oscillation:
            updated_convergence = reset_threshold_cooldown(
                updated_convergence,
                grad_dot_product < 0,
            )

        threshold_slot = updated_convergence.current_threshold_idx
        checked_convergence, crossed_threshold = check_and_advance_threshold(
            carry=updated_convergence,
            current_loss=loss_value,
            thresholds=inputs.convergence_thresholds,
            min_steps=min_steps_per_threshold,
        )
        threshold_event = eligible & (
            (
                checked_convergence.current_threshold_idx
                != updated_convergence.current_threshold_idx
            )
            | (~updated_convergence.converged & checked_convergence.converged)
        )
        new_convergence = jax.tree_util.tree_map(
            lambda checked, updated: jax.lax.select(
                eligible, checked, updated
            ),
            checked_convergence,
            updated_convergence,
        )
        next_active = (
            active_carry.active
            & eligible
        )
        candidate = StateSnapshot(
            params=save_state.params,
            losses=losses,
            step=jnp.asarray(new_state.step, dtype=jnp.int32),
        )
        better = losses.val_losses[0] < active_carry.best.losses.val_losses[0]
        next_best = _select_snapshot(active_carry.best, candidate, better & finite)
        # The bank was created with the requested parameter partitions. Match its
        # leaf shapes here rather than carrying the static partition policy through
        # every scan invocation.
        snapshot_params = jax.tree_util.tree_map(
            lambda bank_value, current_value: (
                current_value
                if bank_value.shape[1:] == current_value.shape
                else jnp.zeros(bank_value.shape[1:], dtype=bank_value.dtype)
            ),
            active_carry.convergence_snapshots.params,
            save_state.params,
        )
        convergence_snapshots = _store_convergence_snapshot(
            active_carry.convergence_snapshots,
            snapshot_params,
            losses,
            new_state.step,
            crossed_threshold,
            threshold_slot,
            threshold_event,
        )

        finite_carry = ChunkCarry(
            opt_state=new_state,
            sim=active_carry.sim,
            convergence=new_convergence,
            lr=new_lr,
            model_lr=new_model_lr,
            executed_steps=active_carry.executed_steps + 1,
            active=next_active,
            best=next_best,
            convergence_snapshots=convergence_snapshots,
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
            convergence_snapshots=active_carry.convergence_snapshots,
        )
        next_carry = _select_carry(finite, finite_carry, frozen_carry)
        return next_carry, StepMetrics(
            total_train_loss=jnp.asarray(losses.total_train_loss),
            total_val_loss=jnp.asarray(losses.total_val_loss),
            lr=new_lr,
            grad_dot_product=grad_dot_product,
            executed=jnp.asarray(True),
            threshold_event=threshold_event,
            crossed_threshold=crossed_threshold,
            threshold_slot=threshold_slot,
        )

    # Keep both paths in the step jaxpr and select their results afterwards.
    # A cond over the complete carry prevents XLA from optimising the expensive
    # step body for large feature arrays.  The inner finite-state select above
    # already establishes the freeze semantics needed when the step is invalid.
    stepped_carry, stepped_metrics = step(carry)
    frozen_carry, frozen_metrics = _no_op_step(carry)
    next_carry = _select_carry(carry.active, stepped_carry, frozen_carry)
    next_metrics = jax.tree_util.tree_map(
        lambda stepped, frozen: jax.lax.select(carry.active, stepped, frozen),
        stepped_metrics,
        frozen_metrics,
    )
    return next_carry, next_metrics


def evaluate_convergence(
    carry: ChunkCarry,
    inputs: ChunkInputs,
    min_steps_per_threshold: int,
) -> tuple[ChunkCarry, Array, Array]:
    old_convergence = carry.convergence
    current_loss = carry.opt_state.losses.total_train_loss
    new_convergence, crossed_threshold = check_and_advance_threshold(
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
        active=carry.active,
    ), threshold_event, crossed_threshold


def _make_record(
    carry: ChunkCarry,
    threshold_event: Array,
    crossed_threshold: Array,
    boundary_active: Array,
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_record_params: bool = True,
) -> ChunkRecord:
    losses = carry.opt_state.losses
    if losses is None:
        raise ValueError("Chunk carry must contain losses")
    return ChunkRecord(
        step=jnp.asarray(carry.opt_state.step, dtype=jnp.int32),
        params=_select_partitions(
            carry.opt_state.params,
            parameter_partitions if retain_record_params else frozenset(),
        ),
        losses=losses,
        lr=carry.lr,
        threshold_idx=carry.convergence.current_threshold_idx,
        threshold_event=threshold_event,
        crossed_threshold=crossed_threshold,
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
        "parameter_partitions",
        "retain_record_params",
        "reset_threshold_cooldown_on_oscillation",
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
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_record_params: bool = True,
    reset_threshold_cooldown_on_oscillation: bool = True,
) -> tuple[ChunkCarry, StepMetrics, ChunkRecord]:
    def scan_step(current: ChunkCarry, _unused: None) -> tuple[ChunkCarry, StepMetrics]:
        return optimisation_step(
            current,
            inputs,
            optimizer,
            loss_functions,
            indexes,
            min_steps_per_threshold,
            reset_threshold_cooldown_on_oscillation,
        )

    carry, metrics = jax.lax.scan(scan_step, carry, None, length=chunk_size)
    threshold_event = jnp.any(metrics.threshold_event)
    event_indices = jnp.where(
        metrics.threshold_event,
        jnp.arange(chunk_size, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    last_event_index = jnp.maximum(jnp.max(event_indices), 0)
    crossed_threshold = jax.lax.select(
        threshold_event,
        metrics.crossed_threshold[last_event_index],
        jnp.asarray(jnp.nan, dtype=jnp.float32),
    )
    boundary_active = jnp.any(metrics.executed)
    record = _make_record(
        carry,
        threshold_event,
        crossed_threshold,
        boundary_active,
        parameter_partitions,
        retain_record_params,
    )
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
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_record_params: bool = True,
    reset_threshold_cooldown_on_oscillation: bool = True,
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
            parameter_partitions,
            retain_record_params,
            reset_threshold_cooldown_on_oscillation,
        )
        records.append(record)
        metrics.append(chunk_metrics)
        remaining -= current_size
        # The production compiled runner calls this function from Python, so a
        # boundary sync can avoid launching further full-cost chunks after
        # convergence.  When run_chunks is traced (the profiling wrapper and
        # vmap path), the active value is a tracer and the fixed-shape loop is
        # retained instead.
        if not isinstance(carry.active, jax.core.Tracer) and not bool(
            jax.device_get(carry.active)
        ):
            break
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
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_record_params: bool = True,
    reset_threshold_cooldown_on_oscillation: bool = True,
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
        parameter_partitions,
        retain_record_params,
        reset_threshold_cooldown_on_oscillation,
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
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None,
    retain_record_params: bool = True,
    reset_threshold_cooldown_on_oscillation: bool = True,
) -> ChunkResult:
    input_axes = ChunkInputs(
        data_targets=None,
        convergence_thresholds=0,
        tolerance=0,
        ema_alpha=0,
        base_lr=0,
        base_model_lr=0,
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
            parameter_partitions,
            retain_record_params,
            reset_threshold_cooldown_on_oscillation,
        )

    return jax.vmap(one, in_axes=(0, input_axes))(carries, inputs)
