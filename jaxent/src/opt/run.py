import logging
from collections.abc import Sequence
from typing import Optional, TypedDict, cast

import numpy as np

import jax
import jax.numpy as jnp
from beartype.typing import Callable
from jax import Array

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import (
    InitialisedSimulation,
    JaxEnt_Loss,
    OptimizationHistory,
    OptimizationState,
)
from jaxent.src.opt.chunk import (
    ChunkCarry,
    ChunkInputs,
    ChunkResult,
    StateSnapshot,
    _make_record,
    evaluate_convergence,
    optimisation_step,
    run_sequential,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, compute_loss
from jaxent.src.opt.track import (
    create_convergence_thresholds,
    initialise_convergence_carry,
)

LOGGER = logging.getLogger("jaxent.opt")


def optimise(
    ensemble_paths: list[tuple[str, str]],
    features_dir: list[str],
    output_path: str,
    config_paths: list[str],
    name: str,
    batch_size: Optional[int],
    forward_models: list[str],
    loss_functions: list[str],
    log_path: Optional[str],
    overwrite: bool,
):
    # this function will be the input for the cli
    # this will take in paths and configurations and create the individual objects for analysis
    # TODO create reusable builder methods to generate objects from configuration
    _ = (
        ensemble_paths,
        features_dir,
        output_path,
        config_paths,
        name,
        batch_size,
        forward_models,
        loss_functions,
        log_path,
        overwrite,
    )
    pass


class OptimiseFnInputs(TypedDict):
    _simulation: InitialisedSimulation
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ]
    n_steps: int
    tolerance: float
    convergence: float
    indexes: tuple[int]
    loss_functions: tuple[JaxEnt_Loss]
    optimisable_funcs: list[bool] | Array | None
    optimizer: OptaxOptimizer
    opt_state: OptimizationState


def _ensure_dense_state(
    simulation: InitialisedSimulation,
    opt_state: OptimizationState,
    data_to_fit: tuple[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
        ...,
    ],
    indexes: tuple[int, ...],
    loss_functions: tuple[JaxEnt_Loss, ...],
) -> OptimizationState:
    losses = opt_state.losses
    if losses is None:
        losses = compute_loss(
            simulation,
            opt_state.params,
            data_to_fit,
            indexes,
            loss_functions,
        )
    gradients = opt_state.gradients
    if gradients is None:
        gradients = jax.tree_util.tree_map(jnp.zeros_like, opt_state.params)
    return OptimizationState(
        params=opt_state.params,
        opt_state=opt_state.opt_state,
        step=opt_state.step,
        losses=losses,
        gradients=gradients,
    )


def _build_chunk_state(
    _simulation: InitialisedSimulation,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    tolerance: float,
    convergence: float | list[float],
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
    learning_rate: float | Array | None = None,
) -> tuple[ChunkCarry, ChunkInputs, tuple[JaxEnt_Loss, ...], tuple[int, ...]]:
    tuple_data_to_fit = tuple(data_to_fit)
    tuple_indexes = tuple(indexes)
    tuple_loss_functions = tuple(loss_functions)

    dense_state = _ensure_dense_state(
        simulation=_simulation,
        opt_state=opt_state,
        data_to_fit=tuple_data_to_fit,
        indexes=tuple_indexes,
        loss_functions=tuple_loss_functions,
    )
    if dense_state.losses is None:
        raise ValueError("Dense state must include losses for chunked optimisation.")

    base_learning_rate = (
        jnp.asarray(optimizer.learning_rate, dtype=jnp.float32)
        if learning_rate is None
        else jnp.asarray(learning_rate, dtype=jnp.float32)
    )
    init_carry = ChunkCarry(
        opt_state=dense_state,
        sim=_simulation,
        convergence=initialise_convergence_carry(),
        lr=base_learning_rate,
        model_lr=jnp.asarray(
            base_learning_rate * optimizer.model_parameters_lr_scale,
            dtype=jnp.float32,
        ),
        executed_steps=jnp.array(0, dtype=jnp.int32),
        active=(
            jnp.isfinite(dense_state.losses.total_train_loss)
            & (dense_state.losses.total_train_loss >= jnp.asarray(tolerance, dtype=jnp.float32))
        ),
        best=StateSnapshot(
            params=dense_state.params,
            losses=dense_state.losses,
            step=jnp.asarray(dense_state.step, dtype=jnp.int32),
        ),
    )
    inputs = ChunkInputs(
        data_targets=tuple_data_to_fit,
        convergence_thresholds=create_convergence_thresholds(convergence, base_learning_rate),
        tolerance=jnp.asarray(tolerance, dtype=jnp.float32),
        ema_alpha=jnp.asarray(0.5, dtype=jnp.float32),
    )
    return init_carry, inputs, tuple_loss_functions, tuple_indexes


def _optimise_pure(
    _simulation: InitialisedSimulation,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    n_steps: int,
    tolerance: float,
    convergence: float | list[float],
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
    ema_alpha: float = 0.5,
    min_steps_per_threshold: int = 2,
    learning_rate: float | Array | None = None,
    chunk_size: int = 100,
    save_states: bool = True,
    save_convergence: bool = True,
    save_best: bool = True,
    state_parameter_partitions=None,
) -> tuple[InitialisedSimulation, OptaxOptimizer]:
    carry, inputs, tuple_loss_functions, tuple_indexes = _build_chunk_state(
        _simulation,
        data_to_fit,
        tolerance,
        convergence,
        indexes,
        loss_functions,
        opt_state,
        optimizer,
        learning_rate,
    )
    inputs = inputs._replace(ema_alpha=jnp.asarray(ema_alpha, dtype=jnp.float32))
    initial_state = carry.opt_state
    result = run_sequential(
        carry,
        inputs,
        n_steps,
        chunk_size,
        optimizer,
        tuple_loss_functions,
        tuple_indexes,
        min_steps_per_threshold,
        parameter_partitions=state_parameter_partitions,
        retain_record_params=save_states or save_convergence,

    )
    history = result_to_history(
        result,
        optimizer,
        save_states=save_states,
        save_convergence=save_convergence,
        save_best=save_best,
        state_parameter_partitions=state_parameter_partitions,
    )
    _prepend_short_run_initial_state(history, initial_state, result.carry.opt_state, n_steps, chunk_size)
    _simulation.params = result.carry.best.params
    return _simulation, optimizer


def _snapshot_to_state(snapshot: StateSnapshot, final_state: OptimizationState) -> OptimizationState:
    return OptimizationState(
        params=snapshot.params,
        opt_state=final_state.opt_state,
        step=snapshot.step,
        losses=snapshot.losses,
        gradients=final_state.gradients,
    )


def result_to_history(
    result: ChunkResult,
    optimizer: OptaxOptimizer,
    *,
    save_states: bool = True,
    save_convergence: bool = True,
    save_best: bool = True,
    state_parameter_partitions=None,
) -> OptimizationHistory:
    """Convert final chunk outputs into the public Python history contract."""
    records = result.records
    active = (
        np.asarray(jax.device_get(records.active))
        if (save_states or save_convergence)
        else None
    )
    threshold_event = (
        np.asarray(jax.device_get(records.threshold_event)) if save_convergence else None
    )
    active_indices = np.flatnonzero(active) if active is not None else []

    final_state = result.carry.opt_state
    history = OptimizationHistory(state_parameter_partitions=state_parameter_partitions)

    def state_from_record(index: int) -> OptimizationState:
        params = jax.tree_util.tree_map(lambda value: value[index], records.params)
        losses = jax.tree_util.tree_map(lambda value: value[index], records.losses)
        return OptimizationState(
            params=params,
            opt_state=final_state.opt_state,
            step=records.step[index],
            losses=losses,
            gradients=final_state.gradients,
        )

    convergence_indices = (
        {int(index) for index in np.flatnonzero(active & threshold_event)}
        if save_convergence
        else set()
    )
    for index in active_indices:
        state = state_from_record(int(index)) if save_states else None
        if save_states:
            history.states.append(state)
        if int(index) in convergence_indices:
            history.convergence_states.append(
                state if state is not None else state_from_record(int(index))
            )

    if save_best:
        history.best_state = _snapshot_to_state(result.carry.best, final_state)
    optimizer.history = history
    return history


def _prepend_short_run_initial_state(
    history: OptimizationHistory,
    initial_state: OptimizationState,
    final_opt_state: OptimizationState,
    n_steps: int,
    chunk_size: int,
) -> None:
    """Keep the legacy two-point diagnostic history for sub-chunk runs."""
    if n_steps < chunk_size and history.states:
        history.states.insert(
            0,
            OptimizationState(
                params=initial_state.params,
                opt_state=final_opt_state.opt_state,
                step=initial_state.step,
                losses=initial_state.losses,
                gradients=final_opt_state.gradients,
            ),
        )


def _optimise(
    _simulation: InitialisedSimulation,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    n_steps: int,
    tolerance: float,
    convergence: float | list[float],
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
    ema_alpha: float = 0.5,  # EMA smoothing factor
    min_steps_per_threshold: int = 2,  # Minimum steps before checking convergence
    chunk_size: int = 100,
    logger: logging.Logger | None = None,
    silent: bool = False,
    terminal_state_callback: Callable[[OptimizationState], None] | None = None,
    save_states: bool = True,
    save_convergence: bool = True,
    save_best: bool = True,
    state_parameter_partitions=None,
) -> tuple[InitialisedSimulation, OptaxOptimizer]:
    """Python diagnostic orchestration over the shared step primitives."""
    carry, inputs, tuple_loss_functions, tuple_indexes = _build_chunk_state(
        _simulation,
        data_to_fit,
        tolerance,
        convergence,
        indexes,
        loss_functions,
        opt_state,
        optimizer,
    )
    inputs = inputs._replace(ema_alpha=jnp.asarray(ema_alpha, dtype=jnp.float32))
    initial_state = carry.opt_state

    records = []
    metrics = []
    remaining = n_steps
    while remaining > 0:
        current_size = min(chunk_size, remaining)
        boundary_metrics = []
        old_active = carry.active
        for _ in range(current_size):
            carry, step_metrics = optimisation_step(
                carry,
                inputs,
                optimizer,
                tuple_loss_functions,
                tuple_indexes,
                min_steps_per_threshold,
            )
            boundary_metrics.append(step_metrics)
        carry, threshold_event = evaluate_convergence(
            carry,
            inputs,
            min_steps_per_threshold,
        )
        metrics.extend(boundary_metrics)
        records.append(
            _make_record(
                carry,
                threshold_event,
                old_active & jnp.any(boundary_metrics[0].executed),
                state_parameter_partitions,
                save_states or save_convergence,
            )
        )
        remaining -= current_size
        # The Python diagnostic path is intentionally allowed one convergence
        # check per chunk.  With the unconditional optimisation step, skipping
        # later chunks is important: inactive steps now evaluate and discard a
        # full step body.
        if not bool(jax.device_get(carry.active)):
            break

    result = ChunkResult(
        carry=carry,
        records=jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records),
        metrics=jax.tree_util.tree_map(lambda *values: jnp.stack(values), *metrics),
    )
    history = result_to_history(
        result,
        optimizer,
        save_states=save_states,
        save_convergence=save_convergence,
        save_best=save_best,
        state_parameter_partitions=state_parameter_partitions,
    )
    _prepend_short_run_initial_state(history, initial_state, result.carry.opt_state, n_steps, chunk_size)
    _simulation.params = result.carry.best.params
    if terminal_state_callback is not None:
        terminal_state_callback(result.carry.opt_state)
    if logger is not None and not silent:
        logger.info("Optimisation completed after %s executed steps", result.carry.executed_steps)
    return cast(InitialisedSimulation, _simulation), optimizer


def run_optimise(
    simulation: Simulation,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    config: OptimiserSettings,
    forward_models: Sequence[ForwardModel],
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    optimizer: Optional[OptaxOptimizer] = None,
    optimisable_funcs: list[bool] | Array | None = None,
    initialise: Optional[bool] = False,
    jit_update_step: bool | None = None,
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> tuple[InitialisedSimulation, OptimizationHistory]:
    """Run the optimization process."""
    _logger = logger if logger is not None else LOGGER

    if forward_models:
        Warning("forward_models arg not yet implemented in run_optimise")

    if initialise:
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation")

    _simulation = cast(InitialisedSimulation, simulation)

    if len(data_to_fit) != len(loss_functions):
        raise ValueError("Number of data targets and loss functions must match")

    if optimizer is None:
        optimizer = OptaxOptimizer(
            learning_rate=config.learning_rate,
            optimizer=config.optimiser_type,
        )
    opt_state = optimizer.initialise(
        _simulation,
        optimisable_funcs,
    )
    _optimizer: OptaxOptimizer = cast(OptaxOptimizer, optimizer)

    execution_mode = config.execution_mode
    if jit_update_step is not None:
        import warnings

        warnings.warn(
            "jit_update_step is deprecated; use OptimiserSettings.execution_mode instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        execution_mode = "compiled" if jit_update_step else "python"

    if execution_mode == "compiled":
        _simulation, optimizer = _optimise_pure(
            _simulation,
            data_to_fit,
            config.n_steps,
            config.tolerance,
            config.convergence,
            indexes,
            loss_functions,
            opt_state,
            _optimizer,
            ema_alpha=config.ema_alpha,
            min_steps_per_threshold=config.min_steps_per_threshold,
            chunk_size=config.step_chunk_size,
            save_states=config.save_states,
            save_convergence=config.save_convergence,
            save_best=config.save_best,
            state_parameter_partitions=config.parameter_partitions,
        )
    elif execution_mode == "python":
        _simulation, optimizer = _optimise(
            _simulation,
            data_to_fit,
            config.n_steps,
            config.tolerance,
            config.convergence,
            indexes,
            loss_functions,
            opt_state,
            _optimizer,
            ema_alpha=config.ema_alpha,
            min_steps_per_threshold=config.min_steps_per_threshold,
            chunk_size=config.step_chunk_size,
            logger=_logger,
            silent=silent,
            save_states=config.save_states,
            save_convergence=config.save_convergence,
            save_best=config.save_best,
            state_parameter_partitions=config.parameter_partitions,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")

    if optimizer.history.best_state is not None:
        _logger.info("Best parameters: %s", optimizer.history.best_state.params)

    if optimizer:
        return _simulation, optimizer.history
    raise ValueError("Optimisation failed")
