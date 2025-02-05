from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxent.config.base import OptimiserSettings
from jaxent.datatypes import Experimental_Dataset, Simulation, Simulation_Parameters
from jaxent.forwardmodels.base import ForwardModel, Model_Parameters, Output_Features
from jaxent.lossfn.base import JaxEnt_Loss


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


########################################################################
# TODO: fix jax typing


def optimiser_step(
    simulation: Simulation,
    loss_value: float | np.float64,
    parameters: Simulation_Parameters,
    learning_rate: float = 0.01,
    argnums: tuple[int, ...] = (0, 1, 2),
) -> Simulation_Parameters:
    """Performs a single optimization step using gradient descent."""
    params_tuple = parameters.to_tuple()
    grads = jax.grad(lambda *args: loss_value, argnums=argnums)(*params_tuple)

    if isinstance(grads, jax.Array):
        grads = (grads,)

    grad_dict = dict(zip(argnums, grads))
    new_params = list(params_tuple)

    for idx in argnums:
        new_params[idx] = params_tuple[idx] - learning_rate * grad_dict[idx]
        if idx in (0, 2):  # Normalize weights
            new_params[idx] = new_params[idx] / jnp.sum(new_params[idx])

    return Simulation_Parameters.from_tuple(tuple(new_params))


########################################################################
# This needs a more elegant implementation


def run_optimise(
    simulation: Simulation,
    data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
    config: OptimiserSettings,
    forward_models: list[ForwardModel],
    loss_functions: list[JaxEnt_Loss],
    initialise: Optional[bool] = False,
) -> Simulation:
    """Runs the optimization process for a given simulation."""
    if initialise:
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation")

        if not (len(data_to_fit) == len(loss_functions) == len(forward_models)):
            raise ValueError(
                "Number of data targets, loss functions, and forward models must match"
            )

    # jitted_step = jit(optimiser_step)

    current_params = simulation.params

    for step in range(config.n_steps):
        losses = []
        for loss_fn, data in zip(loss_functions, data_to_fit):
            losses.append(loss_fn(simulation, data))

        average_loss = jnp.mean(jnp.array(losses))
        if average_loss < config.tolerance:
            break

        current_params = optimiser_step(
            simulation, average_loss, current_params, learning_rate=0.01, argnums=(0, 1, 2)
        )

        # create new parameters
        new_params = Simulation_Parameters.from_tuple(current_params.to_tuple())

        simulation.params = new_params

    return simulation
    ########################################################################
