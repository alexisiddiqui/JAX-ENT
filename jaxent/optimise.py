from typing import Optional, Sequence, Tuple, cast

from jax import Array

from jaxent.config.base import OptimiserSettings
from jaxent.data.loading import Experimental_Dataset
from jaxent.interfaces.features import Output_Features
from jaxent.interfaces.model import Model_Parameters
from jaxent.optimise.base import InitialisedSimulation, JaxEnt_Loss, OptimizationHistory
from jaxent.optimise.optimiser import OptaxOptimizer
from jaxent.types.base import ForwardModel, Simulation


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


def run_optimise(
    simulation: Simulation,
    data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
    config: OptimiserSettings,
    forward_models: Sequence[ForwardModel],
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    optimisable_funcs: list[bool] | Array | None = None,
    optimizer: Optional[OptaxOptimizer] = None,
    initialise: Optional[bool] = False,
) -> Tuple[Simulation, OptimizationHistory]:
    """Runs the optimization process"""

    if initialise:
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation")

    _simulation = cast(InitialisedSimulation, simulation)

    if not (len(data_to_fit) == len(loss_functions)):
        raise ValueError("Number of data targets and loss functions must match")

    if optimizer is None:
        optimizer = OptaxOptimizer()

    opt_state = optimizer.init(_simulation.params, optimisable_funcs)

    for step in range(config.n_steps):
        opt_state, current_loss = optimizer.step(
            opt_state, _simulation, data_to_fit, loss_functions, indexes
        )
        # if step % 100 == 0:
        print(f"Step {step}")
        print(f"Training Loss: {opt_state.losses.total_train_loss:.2f}")  # type: ignore
        print(f"Validation Loss: {opt_state.losses.total_val_loss:.2f}")  # type: ignore

        # print("Parameters:")
        # print(jnp.sum(opt_state.params.frame_weights))
        print(opt_state.params.model_parameters)
        # print(jnp.sum(opt_state.params.forward_model_weights))
        # simulation.params = opt_state.params
        if current_loss < config.tolerance:
            print(f"Reached convergence tolerance at step {step}")
            break
        # compare to the previous loss
        if (
            step > 10
            and abs(current_loss - optimizer.history.states[-2].losses.total_train_loss)  # type: ignore
            < config.convergence
        ):
            print(f"Loss converged at step {step}")
            break

        # Check convergence on training loss

    if optimizer.history.best_state is not None:
        simulation.params = optimizer.history.best_state.params

    # print best parameters
    print("Best parameters:")
    print(optimizer.history.best_state.params)  # type: ignore

    return simulation, optimizer.history
