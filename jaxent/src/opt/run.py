import logging
from collections.abc import Sequence
import time
from beartype.typing import Callable, Optional, TypedDict, cast

import chex

import jax
import jax.numpy as jnp
from jax import Array
import math

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss, OptimizationHistory
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState

logger = logging.getLogger(__name__)



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
        
    convergence_thresholds = sorted(convergence, reverse=True)
    # divide convergence thresholds by optimiser.learning_rate
    convergence_thresholds = [ct * optimizer.learning_rate for ct in convergence_thresholds]
    current_threshold_idx = 0
    current_threshold = convergence_thresholds[current_threshold_idx]

    ema_loss_delta = None
    ema_params = None
    steps_since_threshold_start = 0
    optimizer.history = OptimizationHistory()

    loop_start_time = time.time()
    try:
        previous_loss = None
        prev_opt_state = None  # replace this with opt_state
        for step in range(n_steps):
            # --- Python-level warmup LR switch ---
            # This is intentionally handled here (not inside _step) because ``step``
            # is a concrete Python integer, making it safe to use as a branch
            # condition without breaking jax.jit or jax.vmap on _step.
            if step == optimizer.initial_steps:
                optimizer.lr_schedule.update(optimizer.learning_rate)
                optimizer.model_lr_schedule.update(
                    optimizer.learning_rate * optimizer.model_parameters_lr_scale
                )
                optimisable_funcs = getattr(optimizer, "optimisable_funcs", None)
                optimizer.gradient_mask = optimizer.create_gradient_masks(
                    optimizer.parameter_partition_masks,
                    opt_state.params,
                    optimisable_funcs,
                )

            prev_grads = (
                opt_state.gradients
                if opt_state.gradients is not None
                else jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), opt_state.params)
            )  # type: ignore

            opt_state, current_loss, save_state, _simulation = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
            )

            # Calculate delta only after first step
            if previous_loss is not None:
                raw_loss_delta = jnp.abs(previous_loss - current_loss)

                # Update EMA
                if (
                    ema_loss_delta is None or ema_params is None
                ):  # First real delta calculation - initialize with first value
                    ema_loss_delta = raw_loss_delta
                    ema_params = save_state.params

                else:
                    ema_loss_delta = ema_alpha * raw_loss_delta + (1 - ema_alpha) * ema_loss_delta
                    ema_params = ema_alpha * save_state.params + (1 - ema_alpha) * ema_params

            else:
                raw_loss_delta = 0.0  # For logging purposes
                # Keep ema_loss_delta as None until we have real data - don't set to 0.0!

            # Calculate relative convergence
            relative_convergence = (
                ema_loss_delta / current_loss
                if ema_loss_delta is not None and current_loss > 0
                else 0.0
            )

            # Store current loss for next iteration (BEFORE using it in calculations!)
            previous_loss = current_loss

            steps_since_threshold_start += 1
            _opt_state = OptimizationState(
                params=opt_state.params,
                opt_state=opt_state.opt_state,
                step=opt_state.step,
            )
            if prev_opt_state is None:
                # create a pytree of zeros with the same structure as opt_state.params
                zero_params = jax.tree_util.tree_map(
                    lambda x: jnp.full_like(x, 1e0), opt_state.params
                )
                prev_opt_state = OptimizationState(
                    params=zero_params,
                    opt_state=opt_state.opt_state,
                    step=opt_state.step,
                )

            opt_state_param_frameweight_delta = jnp.linalg.norm(
                _opt_state.params.frame_weights - prev_opt_state.params.frame_weights
            )
            # compute the dot product between the previous and current gradients to check for oscillations
            grad_dot_product = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(
                    lambda a, b: jnp.vdot(a, b),
                    prev_grads,
                    opt_state.gradients,  # type: ignore
                ),
            )

            prev_opt_state = _opt_state

            jax.debug.print(
                fmt=" ".join(
                    [
                        "Step {step}/{n_steps}",
                        "Loss: {current_loss:.6e}",
                        "EMA Δ: {ema_delta:.4e}",
                        "Raw Δ: {raw_delta:.4e}",
                        "Rel Conv: {rel_conv:.6e}",
                        "Threshold {threshold_idx}/{total_thresholds} ({current_threshold:.6e})",
                        "Opt State Δ: {opt_state_delta:.4e}",
                        "Grad Dot Prod: {grad_dot_product:.4e}",
                        "LR: {learning_rate:.4e}",
                    ]
                ),
                step=step,
                n_steps=n_steps,
                current_loss=current_loss,
                ema_delta=ema_loss_delta if ema_loss_delta is not None else 0.0,
                raw_delta=raw_loss_delta,
                rel_conv=relative_convergence,
                opt_state_delta=opt_state_param_frameweight_delta,
                grad_dot_product=grad_dot_product,
                learning_rate=optimizer.lr_schedule(),
                threshold_idx=current_threshold_idx + 1,
                total_thresholds=len(convergence_thresholds),
                current_threshold=current_threshold,
            )

            # --- Python-level oscillation-based LR adaptation ---
            # grad_dot_product is a JAX scalar returned from the step.
            # Materialize it here so that using it in Python control flow is safe.
            grad_dot_product_value = float(grad_dot_product)
            # Checking it here (Python level) is safe; the same check inside _step
            # would require a concrete value under jax.jit/jax.vmap and is therefore
            # not done there.
            if grad_dot_product_value < 0 and step > 1:
                current_lr = optimizer.lr_schedule()
                new_lr = current_lr / optimizer.plateau_denominator
                optimizer.lr_schedule.update(new_lr)
                optimizer.model_lr_schedule.update(new_lr * optimizer.model_parameters_lr_scale)
                logger.warning(
                    "Gradient dot product negative at step %d (possible oscillation). "
                    "LR reduced: %.4e → %.4e",
                    step,
                    current_lr,
                    new_lr,
                )
                steps_since_threshold_start = 0

            # Convert current_loss (a JAX scalar) to a Python float for safe Python-level checks.
            current_loss_value = float(current_loss)

            if (
                current_loss_value < tolerance
                or math.isnan(current_loss_value)
                or math.isinf(current_loss_value)
            ):
                logger.info("Reached convergence tolerance/nan/inf at step %d", step)
                break

            if step == 0:
                optimizer = optimizer.update_history_compute_ema_loss(
                    optimizer=optimizer,
                    simulation=_simulation,
                    data_targets=tuple(data_to_fit),
                    indexes=tuple(indexes),
                    loss_functions=tuple(loss_functions),
                    state=save_state,
                    ema_params=ema_params,
                )

            # Relative convergence check after minimum steps (and after we have a real EMA value)
            if (
                steps_since_threshold_start >= min_steps_per_threshold
                and ema_loss_delta is not None
                and relative_convergence < current_threshold
                and step > optimizer.initial_steps
            ):
                logger.info(
                    "Relative threshold %d/%d met at step %d. "
                    "Rel conv: %.8e, threshold: %.2e",
                    current_threshold_idx + 1,
                    len(convergence_thresholds),
                    step,
                    relative_convergence,
                    current_threshold,
                )
                optimizer = optimizer.update_history_compute_ema_loss(
                    optimizer=optimizer,
                    simulation=_simulation,
                    data_targets=tuple(data_to_fit),
                    indexes=tuple(indexes),
                    loss_functions=tuple(loss_functions),
                    state=save_state,
                    ema_params=ema_params,
                )
                ema_loss = optimizer.ema_history.states[-1].losses.total_train_loss
                logger.debug(
                    "Updated history with EMA params. EMA param loss: %.6e", ema_loss
                )
                current_threshold_idx += 1
                steps_since_threshold_start = 0

                if current_threshold_idx >= len(convergence_thresholds):
                    logger.info("All relative thresholds completed at step %d", step)
                    break
                else:
                    current_threshold = convergence_thresholds[current_threshold_idx]
                    logger.info(
                        "Moving to relative threshold %d/%d: %.2e",
                        current_threshold_idx + 1,
                        len(convergence_thresholds),
                        current_threshold,
                    )
    except Exception as e:
        raise RuntimeError(
            f"Optimization failed due to an error: {e}. Returning best state from history.",
            "\\n" * 10,
            "Simulation parameters at failure: ",
            _simulation.params,
            "\\n" * 10,
            "Latest save state at failure: ",
            save_state.params,
            "\\n" * 10,
            "Latest EMA params state at failure: ",
            ema_params,
            "\\n" * 10,
            "Opt State parameters at failure: ",
            opt_state.params,
            "\\n" * 10,
        )

    logger.debug(
        "Optimization ended. Simulation params: %s", _simulation.params
    )

    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = optimizer.history.best_state.params

    loop_end_time = time.time()
    total_time = loop_end_time - loop_start_time
    avg_iteration_time = total_time / (step + 1) if step >= 0 else 0
    iterations_per_second = (step + 1) / total_time if total_time > 0 else 0

    logger.info(
        "Optimization loop performance: %d steps, %.2fs total, %.4fs/iter, %.2f iter/s",
        step + 1,
        total_time,
        avg_iteration_time,
        iterations_per_second,
    )

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

    if getattr(optimizer, "history", None) is None:
        optimizer.history = OptimizationHistory()

    # log best parameters when available
    if optimizer.history.best_state is not None:
        logger.info("Best parameters: %s", optimizer.history.best_state.params)
    else:
        logger.debug("No best_state present in optimization history.")

    if optimizer:
        return _simulation, optimizer.history
    else:
        raise ValueError("Optimisation failed")
