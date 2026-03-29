from collections.abc import Sequence
import time
from beartype.typing import Callable, Optional, TypedDict, cast

import chex

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss, OptimizationHistory
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState
from jaxent.src.opt.track import ConvergenceTracker
from jaxent.src.opt.gradients import get_previous_grads, check_gradient_oscillation
from jaxent.src.opt.log import (
    log_optimization_step,
    log_oscillation_warning,
    print_optimization_summary,
    format_optimization_error,
    log_final_states,
    log_threshold_met,
)

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
) -> tuple[InitialisedSimulation, OptaxOptimizer]:
    """EMA-only approach with relative convergence thresholds."""


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
            )
            prev_params = opt_state.params

            if grad_dot_product < 0:
                log_oscillation_warning(step)
                tracker.reset_threshold_steps()

            if hasattr(current_loss, "item"):
                _loss_val = current_loss.item()
            else:
                _loss_val = current_loss
                
            if (_loss_val < tolerance) or jnp.isnan(current_loss).item() or jnp.isinf(current_loss).item():
                print(f"Reached convergence tolerance/nan vals at step {step}")
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
                    log_threshold_met(step, current_loss, tracker, optimizer)

                    if tracker.advance_threshold():
                        print(f"Moving to relative threshold {tracker.current_threshold_idx + 1}/{len(tracker.convergence_thresholds)}: {tracker.current_threshold:.2e}")
                    else:
                        print(f"All relative thresholds completed at step {step}")
                        break

    except Exception as e:
        raise RuntimeError(format_optimization_error(e, _simulation, save_state, tracker.ema_params, opt_state))

    log_final_states(
        simulation=_simulation,
        save_state=save_state,
        ema_params=tracker.ema_params,
        opt_state=opt_state
    )

    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = optimizer.history.best_state.params

    total_time = time.time() - loop_start_time
    print_optimization_summary(step, total_time)

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
    _opt_fn: Callable = _optimise,  # we seperate this so that users can add diagnostics in the loss loop
    jit_update_step: bool = False,
) -> tuple[InitialisedSimulation, OptimizationHistory]:
    """Runs the optimization process"""

    if forward_models:
        Warning("forward_models arg not yet implemented in run_optimise")

    if initialise:
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation")

    _simulation = cast(InitialisedSimulation, simulation)

    if not (len(data_to_fit) == len(loss_functions)):
        raise ValueError("Number of data targets and loss functions must match")

    if optimizer is None:
        optimizer = OptaxOptimizer(
            learning_rate=config.learning_rate,
            optimizer=config.optimiser_type,
        )
    if jit_update_step is False or None:
        jit_test_args = None
    else:
        jit_test_args = (data_to_fit, loss_functions, indexes)

    opt_state = optimizer.initialise(
        _simulation,
        optimisable_funcs,
        _jit_test_args=jit_test_args,
    )
    _optimizer: OptaxOptimizer = cast(OptaxOptimizer, optimizer)

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
    # print best parameters
    print("Best parameters:")
    print(optimizer.history.best_state.params)  # type: ignore

    if optimizer:
        return _simulation, optimizer.history
    else:
        raise ValueError("Optimisation failed")
