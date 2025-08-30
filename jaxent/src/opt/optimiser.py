from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.tree_util import register_pytree_node_class

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import (
    JaxEnt_Loss,
    LossComponents,
    OptimizationHistory,
    OptimizationState,
)

# jit_test_args = TypeVar(
#     "jit_test_args",
#     bound=,
#     ],
# )


@register_pytree_node_class
class OptaxOptimizer:
    learning_rate: float  # static/aux_data
    optimizer: optax.GradientTransformation  # static/aux_data
    parameter_masks: set[Optimisable_Parameters]  # static/aux_data
    clip_value: Optional[float]  # static/aux_data
    history: OptimizationHistory  # is updated during optimization - child/dynamic
    step: Callable  # function is set during optimization - child/dynamic

    def __init__(
        self,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        parameter_masks: set[Optimisable_Parameters] = {
            Optimisable_Parameters.frame_weights,
        },
        clip_value: Optional[float] = 1.0,
    ):
        self.learning_rate = learning_rate
        self.parameter_masks = parameter_masks
        self.clip_value = clip_value
        self.history = OptimizationHistory()
        self.step = self._step

        optimizer_chain = []
        if clip_value is not None:
            optimizer_chain.append(optax.clip(clip_value))

        if optimizer.lower() == "adam":
            optimizer_chain.append(optax.adam(learning_rate=learning_rate, eps=1e-8))
        elif optimizer.lower() == "sgd":
            optimizer_chain.append(optax.sgd(learning_rate))
        elif optimizer.lower() == "adagrad":
            optimizer_chain.append(optax.adagrad(learning_rate))
        elif optimizer.lower() == "adamw":
            optimizer_chain.append(optax.adamw(learning_rate=learning_rate, eps=1e-8))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # optimizer_chain.append(normalize_weights())
        # optimizer_chain.append(constrain_weights())

        # optimizer_chain.append(optax.keep_params_nonnegative())

        self.optimizer = optax.chain(*optimizer_chain)

    def tree_flatten(self):
        # Dynamic values (leaves of the pytree)
        children = (self.history,)

        # Static values (auxiliary data)
        aux_data = {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "parameter_masks": self.parameter_masks,
            "clip_value": self.clip_value,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Create a new instance without calling __init__
        self = cls.__new__(cls)

        # Set all attributes
        self.history = children[0]
        self.learning_rate = aux_data["learning_rate"]
        self.optimizer = aux_data["optimizer"]
        self.parameter_masks = aux_data["parameter_masks"]
        self.clip_value = aux_data["clip_value"]

        return self

    def initialise(
        self,
        model: InitialisedSimulation,
        optimisable_funcs: list[bool] | Array | None,
        _jit_test_args: Optional[
            tuple[
                Sequence[ExpD_Dataloader[Any] | Model_Parameters | Output_Features],
                Sequence[JaxEnt_Loss],
                Sequence[int],
            ]
        ] = None,
    ) -> OptimizationState:
        """Initialize the optimization state"""
        params = model.params
        print("Params structure:", jax.tree_util.tree_structure(params))
        if isinstance(optimisable_funcs, list):
            optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)
            optimisable_funcs = jnp.round(optimisable_funcs)
        parameter_mask = self.parameter_masks
        gradient_masks = self.create_gradient_masks(
            parameter_mask,
            params,
            optimisable_funcs,
        )

        # init seems to be an optax function, hope this isnt confusing
        opt_state = self.optimizer.init(params)  # type: ignore
        # Okay here we setup the jit step function - to do this we require the additional inputs for run_optimise...

        # test that the _step function works
        if _jit_test_args is not None:
            try:
                data_targets, loss_functions, indexes = _jit_test_args
                self.step = self._step

                _ = self.step(
                    self,
                    OptimizationState(
                        params=params,
                        opt_state=opt_state,
                        gradient_mask=gradient_masks,
                    ),
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    self.history,
                    tuple(indexes),
                )
            except Exception as e:
                del self.step
                raise ValueError(f"Error in step function: {e}")

            # delete the current step function

            # now try to jit the step function
            try:
                # raise ValueError("Stop here")
                del self.step

                self.step = jax.jit(
                    self._step,
                    donate_argnames=("state",),
                    static_argnames=(
                        # "optimizer",
                        # "simulation",
                        # "data_targets",
                        "loss_functions",
                        # "history",
                        "indexes",
                    ),
                )
                _ = self.step(
                    self,
                    OptimizationState(
                        params=params,
                        opt_state=opt_state,
                        gradient_mask=gradient_masks,
                    ),
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    self.history,
                    tuple(indexes),
                )
                print("\n\n\n\n\n\n\n\n\n Optimiser JIT compilation successful \n\n\n\n\n\n\n\n\n")
            except Exception as e:
                print(e)
                self.step = self._step
                RuntimeWarning(
                    f"JIT compilation failed: {e} \n Reverting back to non-jit step function"
                )
        else:
            print("No test args provided, skipping JIT compilation")
            self.step = self._step

        return OptimizationState(
            params=params,
            opt_state=opt_state,
            gradient_mask=gradient_masks,
        )

    @staticmethod
    def create_gradient_masks(
        parameter_masks: set[Optimisable_Parameters],
        params: Simulation_Parameters,
        optimisable_funcs: Array | None,
    ) -> Simulation_Parameters:
        """Creates gradient masks as a Simulation_Parameters instance with integer values.

        Args:
            params: The simulation parameters to create masks for

        Returns:
            A Simulation_Parameters instance containing integer masks (0 or 1) for each parameter
        """
        if optimisable_funcs is None:
            print(
                DeprecationWarning(
                    "The optimisable_funcs argument is deprecated and will be removed in future versions"
                )
            )
            optimisable_funcs = jnp.zeros_like(params.forward_model_weights, dtype=jnp.float32)

        # Create masks based on which parameters are enabled for optimization
        frame_mask = 1.0 if Optimisable_Parameters.frame_weights in parameter_masks else 0.0
        model_mask = 1.0 if Optimisable_Parameters.model_parameters in parameter_masks else 0.0

        mask_mask = 1.0 if Optimisable_Parameters.frame_mask in parameter_masks else 0.0

        print(parameter_masks)
        print(f"Masks: frame={frame_mask}, model={model_mask}, frame_mask={mask_mask}")

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

        # In create_parameter_masks
        param_mask = Simulation_Parameters(
            frame_weights=frame_weights_mask,
            frame_mask=frame_mask_mask,
            model_parameters=model_parameters_mask,
            normalise_loss_functions=jnp.zeros_like(
                params.normalise_loss_functions, dtype=jnp.float32
            ),
            forward_model_weights=jnp.zeros_like(params.forward_model_weights, dtype=jnp.float32),
            forward_model_scaling=jnp.zeros_like(params.forward_model_scaling, dtype=jnp.float32),
        )
        print("Original params structure:", jax.tree_util.tree_structure(params))

        print(
            "Mask structure:",
            jax.tree_util.tree_structure(param_mask),
        )
        # raise ValueError("Stop here")
        return param_mask

    @staticmethod
    def mask_gradients(
        grads: Simulation_Parameters, masks: Simulation_Parameters
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

        # # Mask forward model weights
        # masked_forward_weights = jax.tree_map(
        #     lambda g, m: g * m,
        #     grads.forward_model_weights,
        #     masks.forward_model_weights,
        # )

        # # Mask forward model scaling
        # masked_forward_scaling = jax.tree_map(
        #     lambda g, m: g * m,
        #     grads.forward_model_scaling,
        #     masks.forward_model_scaling,
        # )

        # # In apply_masks
        # print("Grads structure:", jax.tree_util.tree_structure(grads))
        # print("Masks structure:", jax.tree_util.tree_structure(masks))

        masked_grads = Simulation_Parameters(
            frame_weights=masked_frame_weights,
            frame_mask=masked_frame_mask,
            model_parameters=masked_model_parameters,
            normalise_loss_functions=masks.normalise_loss_functions,
            forward_model_weights=masks.forward_model_weights,
            forward_model_scaling=masks.forward_model_scaling,
        )
        # print("Masked grads structure:", jax.tree_util.tree_structure(masked_grads))
        return masked_grads

    @staticmethod
    def _step(
        optimizer: "OptaxOptimizer",
        state: OptimizationState,
        simulation: InitialisedSimulation,
        data_targets: tuple[ExpD_Dataloader | Model_Parameters | Output_Features, ...],
        loss_functions: tuple[JaxEnt_Loss, ...],
        history: OptimizationHistory,
        indexes: tuple[int, ...],
    ) -> Tuple[OptimizationState, Array, OptimizationHistory]:
        """Perform one optimization step"""

        # simulation.params = state.params
        # simulation.forward(state.params)
        # print("Forward pass done")
        def loss_fn(params: Simulation_Parameters):
            # Update simulation parameters for gradient computation
            losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss, losses

        # Compute gradients with value and grad to get both loss and gradients
        loss_val_losses, grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
            state.params
        )
        loss_value, losses = loss_val_losses

        # print("Loss value:", loss_value)
        # print("Raw gradients:", grads)
        # print("Frame model scaling", state.params.forward_model_scaling)
        # Apply masks to gradients
        masked_grads = optimizer.mask_gradients(grads, state.gradient_mask)
        # print("Masked gradients:", masked_grads)

        # Get optimizer updates
        updates, new_opt_state = optimizer.optimizer.update(
            updates=masked_grads,  # type: ignore
            state=state.opt_state,
            params=state.params,  # type: ignore
        )
        # print("Updates:", updates)
        updated_params = optax.apply_updates(state.params, updates)  # type: ignore

        # updated_params = Simulation_Parameters.normalize_weights(updated_params)
        # print("Projected parameters:", updated_params)
        # Compute losses for reporting
        # losses = compute_loss(simulation, updated_params, data_targets, indexes, loss_functions)

        # Create new state
        new_state = state.update(updated_params, new_opt_state, losses)

        # switch parameters to simulation.params

        # save_state = copy.deepcopy(new_state)
        # save_state.params = Simulation_Parameters.normalize_weights(save_state.params)
        save_state = new_state.update(
            Simulation_Parameters.normalize_weights(new_state.params),
            new_state.opt_state,
            losses,
        )
        # Add to history
        history.add_state(save_state)

        return new_state, loss_value, history


@partial(
    jax.jit,
    static_argnames=(
        "simulation",
        # "data_targets",
        "indexes",
        "loss_functions",
    ),
)
def compute_loss(
    simulation: InitialisedSimulation,
    params: Simulation_Parameters,
    data_targets: tuple[ExpD_Dataloader | Model_Parameters | Output_Features, ...],
    indexes: tuple[int, ...],
    loss_functions: tuple[JaxEnt_Loss, ...],
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
