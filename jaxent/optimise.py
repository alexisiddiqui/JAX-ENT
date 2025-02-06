from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from jaxent.config.base import OptimiserSettings
from jaxent.datatypes import (
    Experimental_Dataset,
    Optimisable_Parameters,
    Simulation,
    Simulation_Parameters,
)
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


# def optimiser_step(
#     simulation: Simulation,
#     loss_value: float | np.float64,
#     parameters: Simulation_Parameters,
#     learning_rate: float = 0.01,
#     argnums: tuple[int, ...] = (0, 1, 2),
# ) -> Simulation_Parameters:
#     """Performs a single optimization step using gradient descent."""
#     params_tuple = parameters.to_tuple()
#     grads = jax.grad(lambda *args: loss_value, argnums=argnums)(*params_tuple)

#     if isinstance(grads, jax.Array):
#         grads = (grads,)

#     grad_dict = dict(zip(argnums, grads))
#     new_params = list(params_tuple)

#     for idx in argnums:
#         new_params[idx] = params_tuple[idx] - learning_rate * grad_dict[idx]
#         if idx in (0, 2):  # Normalize weights
#             new_params[idx] = new_params[idx] / jnp.sum(new_params[idx])

#     return Simulation_Parameters.from_tuple(tuple(new_params))


# def optimiser_step(
#     simulation: Simulation,
#     loss_fn,  # Function that computes loss given parameters
#     parameters: Simulation_Parameters,
#     learning_rate: float = 0.01,
#     argnums: tuple[int, ...] = (0, 1, 2),
# ) -> Simulation_Parameters:
#     """Performs a single optimization step using gradient descent."""
#     params_tuple = parameters.to_tuple()

#     # Create function that returns loss for given parameters
#     def loss_from_params(*params):
#         new_params = Simulation_Parameters.from_tuple(params)
#         simulation = simulation.update(new_params, argnums)
#         return loss_fn(simulation)

#     # Calculate gradients
#     grads = jax.grad(loss_from_params, argnums=argnums)(*params_tuple)

#     if isinstance(grads, jax.Array):
#         grads = (grads,)

#     grad_dict = dict(zip(argnums, grads))
#     new_params = list(params_tuple)

#     for idx in argnums:
#         new_params[idx] = params_tuple[idx] - learning_rate * grad_dict[idx]
#         if idx in (0, 2):  # Normalize weights
#             new_params[idx] = new_params[idx] / jnp.sum(new_params[idx])


#     return Simulation_Parameters.from_tuple(tuple(new_params))


# def optimiser_step(
#     simulation: Simulation,
#     loss_fn,
#     parameters: Simulation_Parameters,
#     learning_rate: float = 0.01,
#     argnums: tuple[int, ...] = (0, 1, 2),
# ) -> Simulation_Parameters:
#     """Performs a single optimization step using gradient descent."""

#     def loss_from_params(params: Simulation_Parameters):
#         new_simulation = simulation.update(params, argnums)
#         return loss_fn(new_simulation)

#     grads = jax.grad(loss_from_params)(parameters)

#     # Convert gradients to arrays and create new parameters
#     new_frame_weights = parameters.frame_weights
#     new_model_params = parameters.model_parameters
#     new_forward_weights = parameters.forward_model_weights

#     if Optimisable_Parameters.frame_weights.value in argnums:
#         new_frame_weights = (
#             jnp.array(parameters.frame_weights) - learning_rate * grads.frame_weights
#         )
#         new_frame_weights = new_frame_weights / jnp.sum(new_frame_weights)

#     if Optimisable_Parameters.model_parameters.value in argnums:
#         new_model_params = [
#             mp.update_parameters(mp - learning_rate * g)
#             for mp, g in zip(parameters.model_parameters, grads.model_parameters)
#         ]

#     if Optimisable_Parameters.forward_model_weights.value in argnums:
#         new_forward_weights = (
#             jnp.array(parameters.forward_model_weights)
#             - learning_rate * grads.forward_model_weights
#         )
#         new_forward_weights = new_forward_weights / jnp.sum(new_forward_weights)


#     return Simulation_Parameters(
#         frame_weights=new_frame_weights,
#         model_parameters=new_model_params,
#         forward_model_weights=new_forward_weights,
#     )
def optimiser_step(
    simulation: Simulation,
    loss_fn,
    parameters: Simulation_Parameters,
    learning_rate: float = 0.000001,
    argnums: tuple[int, ...] = (0, 1, 2),
) -> Simulation_Parameters:
    """Performs a single optimization step using gradient descent."""

    def loss_from_params(params: Simulation_Parameters):
        new_simulation = simulation.update(params, argnums)
        return loss_fn(new_simulation)

    grads = jax.grad(loss_from_params)(parameters)
    grads = jax.tree_util.tree_map(
        lambda x: jnp.clip(x, -1e1, 1e1) if x is not None else None,
        grads,  #
    )

    # Convert gradients to arrays and create new parameters
    new_frame_weights = parameters.frame_weights
    new_model_params = parameters.model_parameters
    new_forward_weights = parameters.forward_model_weights

    if Optimisable_Parameters.frame_weights.value in argnums:
        new_frame_weights = (
            jnp.array(parameters.frame_weights) - learning_rate * grads.frame_weights
        )
        new_frame_weights = new_frame_weights / jnp.sum(new_frame_weights)

    if Optimisable_Parameters.model_parameters.value in argnums:
        new_model_params = [
            mp.update_parameters(mp - learning_rate * g)
            for mp, g in zip(parameters.model_parameters, grads.model_parameters)
        ]

    if Optimisable_Parameters.forward_model_weights.value in argnums:
        new_forward_weights = (
            jnp.array(parameters.forward_model_weights)
            - learning_rate * grads.forward_model_weights
        )
        new_forward_weights = new_forward_weights / jnp.sum(new_forward_weights)

    return Simulation_Parameters(
        frame_weights=new_frame_weights,
        model_parameters=new_model_params,
        forward_model_weights=new_forward_weights,
    )


def run_optimise(
    simulation: Simulation,
    data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
    config: OptimiserSettings,
    forward_models: Sequence[ForwardModel],
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

    current_params = simulation.params

    def compute_loss(simulation):
        losses = []
        for loss_fn, data in zip(loss_functions, data_to_fit):
            losses.append(loss_fn(simulation, data))
        return jnp.mean(jnp.array(losses))

    prev_loss = jnp.inf
    for step in range(config.n_steps):
        current_loss = compute_loss(simulation)
        print(f"Step {step}, Loss: {current_loss:.2f}")
        # print(f"Step {step}")

        if current_loss < config.tolerance:
            break

        if jnp.abs(prev_loss - current_loss) < config.convergence:
            break

        current_params = optimiser_step(
            simulation,
            compute_loss,
            current_params,
            learning_rate=0.0001,
            argnums=(1,),
        )

        prev_loss = current_loss

        simulation = simulation.update(
            current_params,
            (0, 1),
        )

    return simulation


# def run_optimise(
#     simulation: Simulation,
#     data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
#     config: OptimiserSettings,
#     forward_models: list[ForwardModel],
#     loss_functions: list[JaxEnt_Loss],
#     initialise: Optional[bool] = False,
# ) -> Simulation:
#     """Runs the optimization process for a given simulation."""
#     if initialise:
#         if not simulation.initialise():
#             raise ValueError("Failed to initialise simulation")

#         if not (len(data_to_fit) == len(loss_functions) == len(forward_models)):
#             raise ValueError(
#                 "Number of data targets, loss functions, and forward models must match"
#             )

#     # jitted_step = jit(optimiser_step)

#     current_params = simulation.params

#     for step in range(config.n_steps):
#         losses = []
#         for loss_fn, data in zip(loss_functions, data_to_fit):
#             losses.append(loss_fn(simulation, data))

#         average_loss = jnp.mean(jnp.array(losses))
#         if average_loss < config.tolerance:
#             break

#         current_params = optimiser_step(
#             simulation, average_loss, current_params, learning_rate=0.01, argnums=(0, 1, 2)
#         )

#         # create new parameters
#         new_params = Simulation_Parameters.from_tuple(current_params.to_tuple())

#         simulation.params = new_params

#     return simulation
########################################################################
