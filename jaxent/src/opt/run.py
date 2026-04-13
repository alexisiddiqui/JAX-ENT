import logging
import time
from collections.abc import Sequence
from typing import Optional, TypedDict, cast

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
    LossComponents,
    OptimisationCarry,
    OptimizationHistory,
    OptimizationState,
)
from jaxent.src.opt.gradients import check_gradient_oscillation, get_previous_grads
from jaxent.src.opt.log import (
    format_optimization_error,
    log_final_states,
    log_optimization_step,
    log_oscillation_warning,
    log_threshold_met,
    print_optimization_summary,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, compute_loss
from jaxent.src.opt.track import (
    ConvergenceTracker,
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
) -> tuple[OptimizationState, InitialisedSimulation]:
    updated_sim = simulation
    losses = opt_state.losses
    if losses is None:
        losses, updated_sim = compute_loss(
            simulation,
            opt_state.params,
            data_to_fit,
            indexes,
            loss_functions,
        )
    gradients = opt_state.gradients
    if gradients is None:
        gradients = jax.tree_util.tree_map(jnp.zeros_like, opt_state.params)
    return (
        OptimizationState(
            params=opt_state.params,
            opt_state=opt_state.opt_state,
            step=opt_state.step,
            losses=losses,
            gradients=gradients,
        ),
        updated_sim,
    )


def _build_history_buffers(
    n_steps: int,
    params: Simulation_Parameters,
    losses: LossComponents,
) -> tuple[Simulation_Parameters, LossComponents]:
    history_params = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_steps,) + x.shape, dtype=x.dtype),
        params,
    )
    history_losses = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_steps,) + x.shape, dtype=x.dtype),
        losses,
    )
    return history_params, history_losses


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
) -> OptimisationCarry:
    """Pure optimisation loop based on ``jax.lax.while_loop``."""
    tuple_data_to_fit = tuple(data_to_fit)
    tuple_indexes = tuple(indexes)
    tuple_loss_functions = tuple(loss_functions)

    dense_state, _simulation = _ensure_dense_state(
        simulation=_simulation,
        opt_state=opt_state,
        data_to_fit=tuple_data_to_fit,
        indexes=tuple_indexes,
        loss_functions=tuple_loss_functions,
    )
    if dense_state.losses is None:
        raise ValueError("Dense state must include losses for pure optimisation.")

    base_learning_rate = (
        jnp.asarray(optimizer.learning_rate, dtype=jnp.float32)
        if learning_rate is None
        else jnp.asarray(learning_rate, dtype=jnp.float32)
    )
    convergence_thresholds = create_convergence_thresholds(convergence, base_learning_rate)

    history_params, history_losses = _build_history_buffers(
        n_steps=n_steps,
        params=dense_state.params,
        losses=dense_state.losses,
    )
    init_carry = OptimisationCarry(
        opt_state=dense_state,
        sim=_simulation,
        convergence=initialise_convergence_carry(dense_state.params),
        lr=jnp.asarray(optimizer.initial_learning_rate, dtype=jnp.float32),
        model_lr=jnp.asarray(
            optimizer.initial_learning_rate * optimizer.model_parameters_lr_scale,
            dtype=jnp.float32,
        ),
        gradient_mask_idx=jnp.array(0, dtype=jnp.int32),
        history_params=history_params,
        history_losses=history_losses,
        write_idx=jnp.array(0, dtype=jnp.int32),
    )

    n_steps_arr = jnp.asarray(n_steps, dtype=jnp.int32)
    tolerance_arr = jnp.asarray(tolerance, dtype=jnp.float32)

    def cond_fn(carry: OptimisationCarry) -> Array:
        if carry.opt_state.losses is None:
            return jnp.array(False)
        current_loss = carry.opt_state.losses.total_train_loss
        return (
            (~carry.convergence.converged)
            & (jnp.asarray(carry.opt_state.step, dtype=jnp.int32) < n_steps_arr)
            & jnp.isfinite(current_loss)
            & (current_loss >= tolerance_arr)
        )

    def body_fn(carry: OptimisationCarry) -> OptimisationCarry:
        return jax.lax.cond(
            carry.convergence.converged,
            lambda c: c,
            lambda c: OptaxOptimizer._pure_step(
                optimizer=optimizer,
                carry=c,
                data_targets=tuple_data_to_fit,
                loss_functions=tuple_loss_functions,
                indexes=tuple_indexes,
                convergence_thresholds=convergence_thresholds,
                ema_alpha=ema_alpha,
                min_steps_per_threshold=min_steps_per_threshold,
                target_lr=base_learning_rate,
                target_model_lr=base_learning_rate * optimizer.model_parameters_lr_scale,
            ),
            carry,
        )

    return jax.lax.while_loop(cond_fn, body_fn, init_carry)


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
    logger: logging.Logger | None = None,
    silent: bool = False,
) -> tuple[InitialisedSimulation, OptaxOptimizer]:
    """Python-loop optimisation path."""
    _logger = logger if logger is not None else LOGGER

    if isinstance(convergence, float):
        convergence = [convergence]

    tracker = ConvergenceTracker(
        convergence=convergence,
        learning_rate=optimizer.learning_rate,
        ema_alpha=ema_alpha,
        min_steps_per_threshold=min_steps_per_threshold,
    )

    previous_loss = None
    prev_params = None
    save_state = None

    optimizer.history = OptimizationHistory()
    loop_start_time = time.time()
    step = 0
    try:
        for step in range(n_steps):
            prev_grads = get_previous_grads(opt_state)

            opt_state, current_loss, save_state, _simulation = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
            )

            raw_loss_delta = tracker.update(previous_loss, current_loss, save_state.params)
            previous_loss = current_loss

            grad_dot_product = check_gradient_oscillation(prev_grads, opt_state.gradients)

            log_optimization_step(
                step=step,
                n_steps=n_steps,
                current_loss=current_loss,
                raw_delta=raw_loss_delta,
                prev_params=prev_params,
                opt_state=opt_state,
                grad_dot_product=grad_dot_product,
                tracker=tracker,
                optimizer=optimizer,
                logger=_logger,
                silent=silent,
            )
            prev_params = opt_state.params

            if grad_dot_product < 0:
                log_oscillation_warning(step, logger=_logger, silent=silent)
                tracker.reset_threshold_steps()

            if hasattr(current_loss, "item"):
                _loss_val = current_loss.item()
            else:
                _loss_val = current_loss

            if (_loss_val < tolerance) or jnp.isnan(current_loss).item() or jnp.isinf(current_loss).item():
                _logger.info("Stopping optimisation at step %s due to tolerance/non-finite loss.", step)
                break

            if step == 0 or tracker.is_threshold_met(current_loss, step, optimizer.initial_steps):
                optimizer = optimizer.update_history_compute_ema_loss(
                    optimizer=optimizer,
                    simulation=_simulation,
                    data_targets=tuple(data_to_fit),
                    indexes=tuple(indexes),
                    loss_functions=tuple(loss_functions),
                    state=save_state,
                    ema_params=tracker.ema_params,
                )

                if step > 0:
                    log_threshold_met(
                        step,
                        current_loss,
                        tracker,
                        optimizer,
                        logger=_logger,
                        silent=silent,
                    )

                    if tracker.advance_threshold():
                        _logger.info(
                            "Moving to threshold %s/%s: %.2e",
                            tracker.current_threshold_idx + 1,
                            len(tracker.convergence_thresholds),
                            tracker.current_threshold,
                        )
                    else:
                        _logger.info("All relative thresholds completed at step %s", step)
                        break

    except Exception as e:
        raise RuntimeError(
            format_optimization_error(e, _simulation, save_state, tracker.ema_params, opt_state)
        )

    log_final_states(
        simulation=_simulation,
        save_state=save_state,
        ema_params=tracker.ema_params,
        opt_state=opt_state,
        logger=_logger,
        silent=silent,
    )

    if optimizer.history.states:
        best_state = optimizer.history.get_best_state()
        if best_state is not None:
            _simulation.params = optimizer.history.best_state.params

    total_time = time.time() - loop_start_time
    print_optimization_summary(step, total_time, logger=_logger, silent=silent)

    _simulation = cast(InitialisedSimulation, _simulation)
    return _simulation, optimizer


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
    _opt_fn: Callable = _optimise,  # we separate this so users can add diagnostics in the loop
    jit_update_step: bool = False,
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
    if not jit_update_step:
        jit_test_args = None
    else:
        jit_test_args = (data_to_fit, loss_functions, indexes)

    opt_state = optimizer.initialise(
        _simulation,
        optimisable_funcs,
        _jit_test_args=jit_test_args,
    )
    _optimizer: OptaxOptimizer = cast(OptaxOptimizer, optimizer)

    if _opt_fn is _optimise:
        _simulation, optimizer = _opt_fn(
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
            logger=_logger,
            silent=silent,
        )
    else:
        _simulation, optimizer = _opt_fn(
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
        )

    if optimizer.history.best_state is not None:
        _logger.info("Best parameters: %s", optimizer.history.best_state.params)

    if optimizer:
        return _simulation, optimizer.history
    raise ValueError("Optimisation failed")
