from typing import Callable, Optional, Sequence, Tuple, TypedDict, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss, OptimizationHistory
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState


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
    data_to_fit: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features]
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
    data_to_fit: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features],
    n_steps: int,
    tolerance: float,
    convergence: float,
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    opt_state: OptimizationState,
    optimizer: OptaxOptimizer,
) -> Tuple[InitialisedSimulation, OptaxOptimizer]:
    for step in range(n_steps):
        history = optimizer.history

        opt_state, current_loss, history = optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=_simulation,
            data_targets=tuple(data_to_fit),
            loss_functions=tuple(loss_functions),
            indexes=tuple(indexes),
            history=history,
        )
        # _simulation.params = opt_state.params
        # if step % 100 == 0:
        jax.debug.print(
            fmt=" ".join(
                [
                    "Step {step}/{n_steps}",  # type: ignore
                    "Training Loss: {train_loss:.2e}",  # type: ignore
                    "Validation Loss: {val_loss:.2e}",  # type: ignore
                ]
            ),
            step=step,
            n_steps=n_steps,
            train_loss=opt_state.losses.total_train_loss,
            val_loss=opt_state.losses.total_val_loss,
        )

        if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):
            print(f"Reached convergence tolerance/nan vals at step {step}, loss: {current_loss}")
            break
        optimizer.history = history  # update only if not nan
        # compare to the previous loss
        if (
            step > 10
            and abs(current_loss - optimizer.history.states[-2].losses.total_train_loss)  # type: ignore
            < convergence
        ):
            print(
                f"Loss converged at step {step}, loss delta: {current_loss - optimizer.history.states[-2].losses.total_train_loss}"
            )  # type: ignore
            break

        # Check convergence on training loss
    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = optimizer.history.best_state.params

    _simulation = cast(InitialisedSimulation, _simulation)
    return _simulation, optimizer


def run_optimise(
    simulation: Simulation,
    data_to_fit: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features | Array],
    config: OptimiserSettings,
    forward_models: Sequence[ForwardModel],
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    optimizer: Optional[OptaxOptimizer] = None,
    optimisable_funcs: list[bool] | Array | None = None,
    initialise: Optional[bool] = False,
    _opt_fn: Callable = _optimise,  # we seperate this so that users can add diagnostics in the loss loop
    jit_update_step: bool = False,
) -> Tuple[InitialisedSimulation, OptimizationHistory]:
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
    )
    # print best parameters
    print("Best parameters:")
    print(optimizer.history.best_state.params)  # type: ignore

    if optimizer:
        return _simulation, optimizer.history
    else:
        raise ValueError("Optimisation failed")
