import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

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


@dataclass
class LossComponents:
    """Stores the various components of loss for training and validation"""

    train_losses: Array  # Individual training loss components
    val_losses: Array  # Individual validation loss components
    scaled_train_losses: Array  # Scaled training loss components
    scaled_val_losses: Array  # Scaled validation loss components
    total_train_loss: Array  # Total training loss
    total_val_loss: Array  # Total validation loss


@dataclass
class OptimizationState:
    params: Simulation_Parameters
    opt_state: optax.OptState
    parameter_masks: Simulation_Parameters
    step: int = 0
    losses: Optional[LossComponents] = None

    def update(
        self,
        new_params: Simulation_Parameters,
        new_opt_state: optax.OptState,
        new_losses: LossComponents,
    ) -> "OptimizationState":
        return OptimizationState(
            params=new_params,
            opt_state=new_opt_state,
            parameter_masks=self.parameter_masks,
            step=self.step + 1,
            losses=new_losses,
        )


@dataclass
class OptimizationHistory:
    """Tracks the history of optimization states and metrics"""

    states: list[OptimizationState] = field(default_factory=list)
    best_state: Optional[OptimizationState] = None

    def add_state(self, state: OptimizationState):
        """Add a new state to history and update best state if needed"""
        self.states.append(state)
        if (
            self.best_state is None
            or state.losses.total_train_loss < self.best_state.losses.total_train_loss
        ):
            self.best_state = state

    def get_loss_history(self) -> list[LossComponents]:
        """Get history of loss components"""
        return [state.losses for state in self.states]

    def get_parameter_history(self, param_name: str) -> list[Any]:
        """Get history of specific parameter values"""
        return [getattr(state.params, param_name) for state in self.states]


# def normalize_weights():
#     """Custom transformation to normalize weights using softmax while preserving gradient flow."""

#     def init_fn(params):
#         return optax.EmptyState()

#     def update_fn(updates, state, params):
#         # Get the updated weights
#         if hasattr(updates, "frame_weights") and hasattr(params, "frame_weights"):
#             new_weights = params.frame_weights + updates.frame_weights

#             # Apply softmax for non-negativity and normalization
#             # Using stop_gradient to prevent gradient flow through the normalization
#             logits = jnp.log(
#                 jnp.maximum(new_weights, 1e-6)
#             )  # Add small epsilon for numerical stability
#             max_logit = jax.lax.stop_gradient(jnp.max(logits))
#             exp_weights = jnp.exp(logits - max_logit)
#             normalized_weights = exp_weights / jax.lax.stop_gradient(jnp.sum(exp_weights))

#             # Calculate the effective update
#             effective_updates = normalized_weights - params.frame_weights
#             updates = updates._replace(frame_weights=effective_updates)

#         return updates, state


# #     return optax.GradientTransformation(init_fn, update_fn)
# def normalize_weights():
#     """Custom transformation to normalize weights using combined softmax and ReLU."""

#     def init_fn(params):
#         return optax.EmptyState()

#     def update_fn(updates, state, params):
#         # Get the updated weights
#         if hasattr(updates, "frame_weights") and hasattr(params, "frame_weights"):
#             new_weights = params.frame_weights + updates.frame_weights

#             # Apply softmax and ReLU in sequence using JAX primitives
#             # First apply softmax
#             max_weight = jax.lax.stop_gradient(jnp.max(new_weights))
#             exp_weights = jnp.exp(new_weights - max_weight)
#             softmaxed = exp_weights / jax.lax.stop_gradient(jnp.sum(exp_weights))

#             # Then apply ReLU and renormalize
#             rectified = jax.lax.relu(softmaxed)
#             sum_rectified = jax.lax.stop_gradient(jnp.sum(rectified))
#             normalized = jnp.where(
#                 sum_rectified > 0,
#                 rectified / sum_rectified,
#                 jnp.ones_like(rectified) / len(rectified),
#             )

#             # Calculate the effective update
#             effective_updates = normalized - params.frame_weights
#             updates = updates._replace(frame_weights=effective_updates)

#         return updates, state

#     return optax.GradientTransformation(init_fn, update_fn)


def normalize_weights():
    """Custom transformation to normalize weights while preserving gradient flow."""

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        # Only normalize the frame weights component
        if hasattr(updates, "frame_weights") and hasattr(params, "frame_weights"):
            sum_weights = jnp.sum(params.frame_weights)
            # Use stop_gradient to prevent gradient flow through the normalization
            scale_factor = jnp.where(sum_weights > 0, jax.lax.stop_gradient(1.0 / sum_weights), 0.0)

            normalized_updates = jax.tree_map(
                lambda u, p: u - jax.lax.stop_gradient(p * jnp.sum(u) / sum_weights),
                updates.frame_weights,
                params.frame_weights,
            )

            updates = updates.__replace__(frame_weights=normalized_updates)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def constrain_weights():
    """Custom transformation to constrain weights between 0 and 1."""

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if hasattr(updates, "frame_weights") and hasattr(params, "frame_weights"):
            # Calculate what the new parameters would be
            new_params = jax.tree_map(
                lambda p, u: p + u, params.frame_weights, updates.frame_weights
            )

            # Clip the new parameters between 0 and 1
            clipped_params = jax.tree_map(lambda x: jnp.clip(x, 0.0, 1.0), new_params)

            # Calculate the required adjustment to the updates
            adjusted_updates = jax.tree_map(
                lambda c, p, u: c - p,  # This gives us the actual change needed
                clipped_params,
                params.frame_weights,
                updates.frame_weights,
            )

            updates = updates.__replace__(frame_weights=adjusted_updates)

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class OptaxOptimizer:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        parameter_masks: set[Optimisable_Parameters] = {Optimisable_Parameters(0)},
        clip_value: Optional[float] = 1.0,
    ):
        self.learning_rate = learning_rate
        self.parameter_masks = parameter_masks
        self.clip_value = clip_value
        self.history = OptimizationHistory()

        optimizer_chain = []
        if clip_value is not None:
            optimizer_chain.append(optax.clip(clip_value))

        if optimizer.lower() == "adam":
            optimizer_chain.append(optax.adam(learning_rate))
        elif optimizer.lower() == "sgd":
            optimizer_chain.append(optax.sgd(learning_rate))
        elif optimizer.lower() == "adagrad":
            optimizer_chain.append(optax.adagrad(learning_rate))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        optimizer_chain.append(normalize_weights())
        # optimizer_chain.append(constrain_weights())

        optimizer_chain.append(optax.keep_params_nonnegative())

        self.optimizer = optax.chain(*optimizer_chain)

    def create_parameter_masks(self, params: Simulation_Parameters) -> Simulation_Parameters:
        """Creates gradient masks as a Simulation_Parameters instance.

        Args:
            params: The simulation parameters to create masks for

        Returns:
            A Simulation_Parameters instance containing boolean masks for each parameter
        """
        # Create masks based on which parameters are enabled for optimization
        frame_mask = Optimisable_Parameters.frame_weights in self.parameter_masks
        model_mask = Optimisable_Parameters.model_parameters in self.parameter_masks
        forward_mask = Optimisable_Parameters.forward_model_weights in self.parameter_masks
        print(self.parameter_masks)
        print(f"Masks: frame={frame_mask}, model={model_mask}, forward={forward_mask}")

        # Create frame weights mask
        frame_weights_mask = jax.tree_map(
            lambda x: jnp.full_like(x, frame_mask, dtype=bool), params.frame_weights
        )

        # Create model parameters mask - handle each Model_Parameters instance separately
        model_parameters_mask = []
        for model_param in params.model_parameters:
            # For each Model_Parameters instance, create a mask of the same structure
            masked_model_param = jax.tree_map(
                lambda x: jnp.full_like(x, model_mask, dtype=bool), model_param
            )
            model_parameters_mask.append(masked_model_param)
        # print(model_parameters_mask)
        # raise NotImplementedError
        # Create forward model weights mask
        forward_model_weights_mask = jax.tree_map(
            lambda x: jnp.full_like(x, forward_mask, dtype=bool), params.forward_model_weights
        )

        return Simulation_Parameters(
            frame_weights=frame_weights_mask,
            model_parameters=model_parameters_mask,
            forward_model_weights=forward_model_weights_mask,
            forward_model_scaling=params.forward_model_scaling,
        )

    def init(self, params: Simulation_Parameters) -> OptimizationState:
        """Initialize the optimization state"""
        parameter_masks = self.create_parameter_masks(params)
        opt_state = self.optimizer.init(params)
        return OptimizationState(
            params=params,
            opt_state=opt_state,
            parameter_masks=parameter_masks,
        )

    def apply_masks(
        self, grads: Simulation_Parameters, masks: Simulation_Parameters
    ) -> Simulation_Parameters:
        """Apply masks to gradients.

        Args:
            grads: The gradients to mask
            masks: The boolean masks to apply

        Returns:
            A new Simulation_Parameters instance with masked gradients
        """

        def bool_to_float(x):
            return jnp.where(x, 1.0, 0.0)

        # Mask frame weights
        masked_frame_weights = jax.tree_map(
            lambda g, m: g * bool_to_float(m), grads.frame_weights, masks.frame_weights
        )
        # print(masked_frame_weights)
        # Mask model parameters - handle each Model_Parameters instance separately
        masked_model_parameters = []
        for grad_param, mask_param in zip(grads.model_parameters, masks.model_parameters):
            # For each pair of Model_Parameters instances, apply the mask
            masked_param = jax.tree_map(lambda g, m: g * bool_to_float(m), grad_param, mask_param)
            masked_model_parameters.append(masked_param)

        # Mask forward model weights
        masked_forward_weights = jax.tree_map(
            lambda g, m: g * bool_to_float(m),
            grads.forward_model_weights,
            masks.forward_model_weights,
        )

        return Simulation_Parameters(
            frame_weights=masked_frame_weights,
            model_parameters=masked_model_parameters,
            forward_model_weights=masked_forward_weights,
            forward_model_scaling=grads.forward_model_scaling,
        )

    def compute_loss(
        self,
        simulation: Simulation,
        params: Simulation_Parameters,
        data_targets: Sequence[Experimental_Dataset | Model_Parameters | Output_Features],
        indexes: Sequence[int],
        loss_functions: Sequence[JaxEnt_Loss],
    ) -> LossComponents:
        """Compute training and validation losses"""
        simulation.forward(params)

        # Calculate individual loss components for training and validation
        train_losses = []
        val_losses = []
        for loss_fn, target, idx in zip(loss_functions, data_targets, indexes):
            train_loss, val_loss = loss_fn(simulation, target, idx)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # Convert to arrays
        train_losses = jnp.array(train_losses)
        val_losses = jnp.array(val_losses)

        # Apply weights and scaling
        weights = params.forward_model_weights
        scaling = jnp.array(params.forward_model_scaling)

        scaled_train = train_losses * weights * scaling
        scaled_val = val_losses * weights * scaling
        # scaled_train = train_losses
        # scaled_val = val_losses
        # Calculate total losses
        total_train = jnp.sum(scaled_train)
        total_val = jnp.sum(scaled_val)

        return LossComponents(
            train_losses=train_losses,
            val_losses=val_losses,
            scaled_train_losses=scaled_train,
            scaled_val_losses=scaled_val,
            total_train_loss=total_train,
            total_val_loss=total_val,
        )

    def step(
        self,
        state: OptimizationState,
        simulation: Simulation,
        data_targets: Sequence[Experimental_Dataset | Model_Parameters | Output_Features],
        loss_functions: Sequence[JaxEnt_Loss],
        indexes: Sequence[int],
    ) -> Tuple[OptimizationState, Array]:
        """Perform one optimization step"""
        # simulation.params = state.params
        # simulation.forward(state.params)
        # print("Forward pass done")

        def loss_fn(params: Simulation_Parameters):
            # Update simulation parameters for gradient computation
            losses = self.compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss

        # Compute gradients with value and grad to get both loss and gradients
        loss_value, grads = jax.value_and_grad(loss_fn, allow_int=False)(state.params)
        print("Loss value:", loss_value)
        # print("Raw gradients:", grads)

        # Apply masks to gradients
        masked_grads = self.apply_masks(grads, state.parameter_masks)
        # print("Masked gradients:", masked_grads)

        # Get optimizer updates
        updates, new_opt_state = self.optimizer.update(
            masked_grads, state.opt_state, params=state.params
        )
        updated_params = optax.apply_updates(state.params, updates)
        # normalized_params = Simulation_Parameters.normalize_weights(updated_params)
        # print("Normalized params:", normalized_params)
        # Compute losses for reporting
        losses = self.compute_loss(
            simulation, updated_params, data_targets, indexes, loss_functions
        )

        # Create new state
        new_state = state.update(updated_params, new_opt_state, losses)

        # switch parameters to simulation.params

        save_state = copy.deepcopy(new_state)
        save_state.params = Simulation_Parameters.normalize_weights(simulation.params)
        # Add to history
        self.history.add_state(save_state)

        return new_state, losses.total_train_loss


def run_optimise(
    simulation: Simulation,
    data_to_fit: tuple[Experimental_Dataset | Model_Parameters | Output_Features, ...],
    config: OptimiserSettings,
    forward_models: Sequence[ForwardModel],
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    optimizer: Optional[OptaxOptimizer] = None,
    initialise: Optional[bool] = False,
) -> Tuple[Simulation, OptimizationHistory]:
    """Runs the optimization process"""

    if initialise:
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation")

    if not (len(data_to_fit) == len(loss_functions) == len(forward_models)):
        raise ValueError("Number of data targets, loss functions, and forward models must match")

    if optimizer is None:
        optimizer = OptaxOptimizer(
            learning_rate=1e-2,
            optimizer="adam",
        )

    opt_state = optimizer.init(simulation.params)

    for step in range(config.n_steps):
        opt_state, current_loss = optimizer.step(
            opt_state, simulation, data_to_fit, loss_functions, indexes
        )
        # if step % 100 == 0:
        print(f"Step {step}")
        print(f"Training Loss: {opt_state.losses.total_train_loss:.2f}")
        print(f"Validation Loss: {opt_state.losses.total_val_loss:.2f}")

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
            step > 2
            and abs(current_loss - optimizer.history.states[-2].losses.total_train_loss)
            < config.convergence
        ):
            print(f"Loss converged at step {step}")
            break

        # Check convergence on training loss

    if optimizer.history.best_state is not None:
        simulation.params = optimizer.history.best_state.params

    # print best parameters
    print("Best parameters:")
    print(optimizer.history.best_state.params)

    return simulation, optimizer.history
