import copy
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from jaxent.data.loader import ExpD_Dataloader
from jaxent.interfaces.model import Model_Parameters
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.opt.base import (
    JaxEnt_Loss,
    LossComponents,
    OptimizationHistory,
    OptimizationState,
)
from jaxent.types import InitialisedSimulation
from jaxent.types.config import Optimisable_Parameters
from jaxent.types.features import Output_Features


class OptaxOptimizer:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        parameter_masks: set[Optimisable_Parameters] = {
            Optimisable_Parameters.model_parameters,
        },
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

        # optimizer_chain.append(normalize_weights())
        # optimizer_chain.append(constrain_weights())

        # optimizer_chain.append(optax.keep_params_nonnegative())

        self.optimizer = optax.chain(*optimizer_chain)

    def create_parameter_masks(
        self, params: Simulation_Parameters, optimisable_funcs: Array | None
    ) -> Simulation_Parameters:
        """Creates gradient masks as a Simulation_Parameters instance with integer values.

        Args:
            params: The simulation parameters to create masks for

        Returns:
            A Simulation_Parameters instance containing integer masks (0 or 1) for each parameter
        """
        if optimisable_funcs is None:
            optimisable_funcs = jnp.ones_like(params.forward_model_weights, dtype=jnp.float32)

        # Create masks based on which parameters are enabled for optimization
        frame_mask = 1.0 if Optimisable_Parameters.frame_weights in self.parameter_masks else 0.0
        model_mask = 1.0 if Optimisable_Parameters.model_parameters in self.parameter_masks else 0.0
        forward_mask = (
            1.0 if Optimisable_Parameters.forward_model_weights in self.parameter_masks else 0.0
        )
        mask_mask = 1.0 if Optimisable_Parameters.frame_mask in self.parameter_masks else 0.0
        scaling_mask = (
            1.0 if Optimisable_Parameters.forward_model_scaling in self.parameter_masks else 0.0
        )

        print(self.parameter_masks)
        print(
            f"Masks: frame={frame_mask}, model={model_mask}, forward={forward_mask}, frame_mask={mask_mask}, scaling={scaling_mask}"
        )

        # Create frame weights mask
        frame_weights_mask = jax.tree_map(
            lambda x: jnp.full_like(x, frame_mask, dtype=jnp.float32), params.frame_weights
        )

        # Create frame mask mask
        frame_mask_mask = jax.tree_map(
            lambda x: jnp.full_like(x, mask_mask, dtype=jnp.float32), params.frame_mask
        )
        print("Frame mask mask:", frame_mask_mask)

        # Create model parameters mask - handle each Model_Parameters instance separately
        model_parameters_mask = []
        for model_param in params.model_parameters:
            # For each Model_Parameters instance, create a mask of the same structure
            masked_model_param = jax.tree_map(
                lambda x: jnp.full_like(x, model_mask, dtype=jnp.float32), model_param
            )
            model_parameters_mask.append(masked_model_param)

        # Create forward model weights mask
        forward_model_weights_mask = jax.tree_map(
            lambda x: jnp.full_like(x, forward_mask, dtype=jnp.float32),
            params.forward_model_weights,
        )

        # mask by optimisable functions
        forward_model_weights_mask = jax.tree_map(
            lambda x, y: x * y, forward_model_weights_mask, optimisable_funcs
        )

        # Create forward model scaling mask
        forward_model_scaling_mask = jax.tree_map(
            lambda x: jnp.full_like(x, scaling_mask, dtype=jnp.float32),
            params.forward_model_scaling,
        )
        # create an empty mask for nomalisation loss
        normalise_loss_mask = jax.tree_map(
            lambda x: jnp.full_like(x, 0.0, dtype=jnp.float32),
            params.forward_model_scaling,
        )

        # In create_parameter_masks
        param_mask = Simulation_Parameters(
            frame_weights=frame_weights_mask,
            frame_mask=frame_mask_mask,
            model_parameters=model_parameters_mask,
            normalise_loss_functions=normalise_loss_mask,
            forward_model_weights=forward_model_weights_mask,
            forward_model_scaling=forward_model_scaling_mask,
        )
        print("Original params structure:", jax.tree_util.tree_structure(params))

        print(
            "Mask structure:",
            jax.tree_util.tree_structure(param_mask),
        )
        # raise ValueError("Stop here")
        return param_mask

    def init(
        self, params: Simulation_Parameters, optimisable_funcs: list[bool] | Array | None
    ) -> OptimizationState:
        """Initialize the optimization state"""
        print("Params structure:", jax.tree_util.tree_structure(params))
        if isinstance(optimisable_funcs, list):
            optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)
            optimisable_funcs = jnp.round(optimisable_funcs)
        parameter_masks = self.create_parameter_masks(params, optimisable_funcs)

        opt_state = self.optimizer.init(params)  # type: ignore
        return OptimizationState(
            params=params,
            opt_state=opt_state,
            parameter_masks=parameter_masks,
        )

    def apply_masks(
        self, grads: Simulation_Parameters, masks: Simulation_Parameters
    ) -> Simulation_Parameters:
        """Apply masks to gradients. Uses integer masks (0 or 1) instead of booleans.

        Args:
            grads: The gradients to mask
            masks: The integer masks to apply (0 or 1)

        Returns:
            A new Simulation_Parameters instance with masked gradients
        """
        # Mask frame weights - directly multiply by the integer mask
        masked_frame_weights = jax.tree_map(
            lambda g, m: g * m, grads.frame_weights, masks.frame_weights
        )

        # Mask frame mask
        masked_frame_mask = jax.tree_map(lambda g, m: g * m, grads.frame_mask, masks.frame_mask)

        # Mask model parameters - handle each Model_Parameters instance separately
        masked_model_parameters = []
        for grad_param, mask_param in zip(grads.model_parameters, masks.model_parameters):
            # For each pair of Model_Parameters instances, apply the mask
            masked_param = jax.tree_map(lambda g, m: g * m, grad_param, mask_param)
            masked_model_parameters.append(masked_param)

        # Mask forward model weights
        masked_forward_weights = jax.tree_map(
            lambda g, m: g * m,
            grads.forward_model_weights,
            masks.forward_model_weights,
        )

        # Mask forward model scaling
        masked_forward_scaling = jax.tree_map(
            lambda g, m: g * m,
            grads.forward_model_scaling,
            masks.forward_model_scaling,
        )

        # # In apply_masks
        # print("Grads structure:", jax.tree_util.tree_structure(grads))
        # print("Masks structure:", jax.tree_util.tree_structure(masks))

        masked_grads = Simulation_Parameters(
            frame_weights=masked_frame_weights,
            frame_mask=masked_frame_mask,
            model_parameters=masked_model_parameters,
            normalise_loss_functions=masks.normalise_loss_functions,
            forward_model_weights=masked_forward_weights,
            forward_model_scaling=masked_forward_scaling,
        )
        # print("Masked grads structure:", jax.tree_util.tree_structure(masked_grads))
        return masked_grads

    def compute_loss(
        self,
        simulation: InitialisedSimulation,
        params: Simulation_Parameters,
        data_targets: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features],
        indexes: Sequence[int],
        loss_functions: Sequence[JaxEnt_Loss],
    ) -> LossComponents:
        """Compute training and validation losses"""
        simulation.forward(params)

        # Calculate individual loss components more efficiently
        losses = [
            loss_fn(simulation, target, idx)
            for loss_fn, target, idx in zip(loss_functions, data_targets, indexes)
        ]

        # Unzip the results and convert to JAX arrays in one step
        train_losses, val_losses = map(jnp.array, zip(*losses))

        # Apply weights and scaling using vectorized operations
        weights = simulation.params.forward_model_weights
        scaling = jnp.array(simulation.params.forward_model_scaling)

        scaled_train = train_losses * weights * scaling
        scaled_val = val_losses * weights * scaling

        # Compute total losses with a single reduction
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
        simulation: InitialisedSimulation,
        data_targets: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features],
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
        loss_value, grads = jax.value_and_grad(loss_fn, allow_int=True)(state.params)
        # print("Loss value:", loss_value)
        # print("Raw gradients:", grads)
        # print("Frame model scaling", state.params.forward_model_scaling)
        # Apply masks to gradients
        masked_grads = self.apply_masks(grads, state.parameter_masks)
        # print("Masked gradients:", masked_grads)

        # Get optimizer updates
        updates, new_opt_state = self.optimizer.update(
            updates=masked_grads,  # type: ignore
            state=state.opt_state,
            params=state.params,  # type: ignore
        )
        # print("Updates:", updates)
        updated_params = optax.apply_updates(state.params, updates)  # type: ignore

        updated_params = optax.projections.projection_non_negative(updated_params)
        # print("Projected parameters:", updated_params)
        # Compute losses for reporting
        losses = self.compute_loss(
            simulation, updated_params, data_targets, indexes, loss_functions
        )

        # Create new state
        new_state = state.update(updated_params, new_opt_state, losses)

        # switch parameters to simulation.params

        save_state = copy.deepcopy(new_state)
        save_state.params = Simulation_Parameters.normalize_weights(save_state.params)
        # Add to history
        self.history.add_state(save_state)

        return new_state, losses.total_train_loss
