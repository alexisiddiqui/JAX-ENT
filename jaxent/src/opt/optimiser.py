import logging
import warnings
from functools import partial
from collections.abc import Sequence
from beartype.typing import Any, Callable, Optional

import chex
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

logger = logging.getLogger(__name__)


@register_pytree_node_class
class MutableLearningRate:
    current_lr: float

    def __init__(self, initial_lr: float):
        self.current_lr = initial_lr

    def __call__(self, step=None):
        return self.current_lr

    def update(self, new_lr: float):
        self.current_lr = new_lr

    def tree_flatten(self):
        children = (self.current_lr,)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])


@register_pytree_node_class
class OptaxOptimizer:
    learning_rate: float  # static/aux_data
    optimizer: optax.GradientTransformation  # static/aux_data
    parameter_partition_masks: set[Optimisable_Parameters]  # static/aux_data
    clip_value: Optional[float]  # static/aux_data
    gradient_mask: Simulation_Parameters  # static/aux_data
    history: OptimizationHistory  # is updated during optimization - child/dynamic
    ema_history: OptimizationHistory  # New: conditionally updated - child/dynamic
    lr_schedule: MutableLearningRate  # New: dynamic - child/dynamic
    model_lr_schedule: MutableLearningRate  # New: dynamic LR for model parameters
    plateau_denominator: float  # New: static - aux_data
    force_logit_simplex: bool  # New: static - aux_data
    save_ema_history: bool  # New: static - aux_data
    step: Callable  # function is set during optimization - child/dynamic
    initial_learning_rate: float  # static/aux_data
    initial_steps: int
    model_parameters_lr_scale: float  # static/aux_data
    update_all_models: bool = False  # New: static/aux_data
    def __init__(
        self,
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        parameter_partition_masks: set[Optimisable_Parameters] = {
            Optimisable_Parameters.frame_weights,
        },
        clip_value: Optional[float] = 1.0,
        force_simplex: Optional[bool] = None,
        plateau_denominator: float = 1.005,
        save_ema_history: bool = True,
        initial_learning_rate: float = 1e0,
        initial_steps: int = 0,
        model_parameters_lr_scale: float = 1.0,
    ):
        self.parameter_partition_masks = parameter_partition_masks
        self.clip_value = clip_value
        self.history = OptimizationHistory()
        self.save_ema_history = save_ema_history
        self.model_parameters_lr_scale = model_parameters_lr_scale
        
        if save_ema_history is True:
            self.ema_history = OptimizationHistory()
        else:
            self.ema_history = None  # Initialize to None if not saving EMA history
        self.step = self._step
        self.plateau_denominator = plateau_denominator
        self.gradient_mask = None  # type: ignore
        self.initial_learning_rate = initial_learning_rate

        # Initialize lr_schedule with initial_learning_rate
        self.lr_schedule = MutableLearningRate(initial_learning_rate)
        self.model_lr_schedule = MutableLearningRate(initial_learning_rate * model_parameters_lr_scale)
        self.learning_rate = learning_rate
        self.initial_steps = initial_steps

        # Select base optimizer type
        if optimizer.lower() == "adam":
            base_optimizer_fn = optax.adam
            _force_simplex = False
        elif optimizer.lower() == "sgd":
            base_optimizer_fn = optax.sgd
            _force_simplex = True
        elif optimizer.lower() == "adagrad":
            base_optimizer_fn = optax.adagrad
            _force_simplex = False
        elif optimizer.lower() == "adamw":
            base_optimizer_fn = optax.adamw
            _force_simplex = False
        elif optimizer.lower() == "rmsprop":
            base_optimizer_fn = optax.rmsprop
            _force_simplex = False
        elif optimizer.lower() == "lbfgs":
            base_optimizer_fn = optax.lbfgs
            _force_simplex = False
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Build optimizer chains for each parameter group
        # For frame weights and frame mask (standard learning rate)
        frame_chain = []
        if clip_value is not None:
            frame_chain.append(optax.clip(clip_value))
        frame_chain.append(
            optax.inject_hyperparams(base_optimizer_fn)(learning_rate=self.lr_schedule)
        )

        # For model parameters (scaled learning rate + non-negative projection)
        model_chain = []
        if clip_value is not None:
            model_chain.append(optax.clip(clip_value))
        model_chain.append(
            optax.inject_hyperparams(base_optimizer_fn)(learning_rate=self.model_lr_schedule)
        )
        model_chain.append(optax.keep_params_nonnegative())

        # For other parameters (typically not optimized, but include for completeness)
        other_chain = []
        if clip_value is not None:
            other_chain.append(optax.clip(clip_value))
        other_chain.append(
            optax.inject_hyperparams(base_optimizer_fn)(learning_rate=self.lr_schedule)
        )

        # Combine using multi_transform with Simulation_Parameters.param_labels
        self.optimizer = optax.multi_transform(
            transforms={
                'frame': optax.chain(*frame_chain),
                'model': optax.chain(*model_chain),
                'other': optax.chain(*other_chain),
            },
            param_labels=Simulation_Parameters.param_labels,
        )

        if force_simplex is None:
            self.force_logit_simplex = _force_simplex
        else:
            self.force_logit_simplex = force_simplex

    def tree_flatten(self):
        # Dynamic values (leaves of the pytree)
        children = (
            self.history,
            self.gradient_mask,
            self.ema_history,
            self.lr_schedule,
            self.model_lr_schedule,
        )

        # Static values (auxiliary data)
        aux_data = {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "parameter_partition_masks": self.parameter_partition_masks,
            "clip_value": self.clip_value,
            "save_ema_history": self.save_ema_history,
            "plateau_denominator": self.plateau_denominator,
            "force_logit_simplex": self.force_logit_simplex,
            "initial_learning_rate": self.initial_learning_rate,
            "step": self.step,
            "initial_steps": self.initial_steps,
            "model_parameters_lr_scale": self.model_parameters_lr_scale,
            "update_all_models": self.update_all_models,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Create a new instance without calling __init__
        self = cls.__new__(cls)

        # Set all attributes
        self.history = children[0]
        self.gradient_mask = children[1]
        self.ema_history = children[2]
        self.lr_schedule = children[3]
        self.model_lr_schedule = children[4]

        self.learning_rate = aux_data["learning_rate"]
        self.optimizer = aux_data["optimizer"]
        self.parameter_partition_masks = aux_data["parameter_partition_masks"]
        self.clip_value = aux_data["clip_value"]
        self.save_ema_history = aux_data["save_ema_history"]
        self.plateau_denominator = aux_data["plateau_denominator"]
        self.force_logit_simplex = aux_data["force_logit_simplex"]
        self.step = aux_data.get("step", self._step)
        self.initial_learning_rate = aux_data.get("initial_learning_rate", 1.0)
        self.initial_steps = aux_data.get("initial_steps", 2)
        self.model_parameters_lr_scale = aux_data.get("model_parameters_lr_scale", 0.1)
        self.update_all_models = aux_data.get("update_all_models", False)

        return self

    def initialise(
        self,
        model: InitialisedSimulation,
        optimisable_funcs: Optional[list[bool] | Array] = None,
        _jit_test_args: Optional[
            tuple[
                Sequence[
                    ExpD_Dataloader[Any]
                    | Model_Parameters
                    | Output_Features
                    | Array
                    | Simulation_Parameters
                ],
                Sequence[JaxEnt_Loss],
                Sequence[int],
            ]
        ] = None,
    ) -> OptimizationState:
        """Initialize the optimization state"""
        params = model.params
        # multiply the weights by the length of the array
        params = Simulation_Parameters(
            frame_mask=params.frame_mask,
            frame_weights=params.frame_weights * len(params.frame_weights),
            model_parameters=params.model_parameters,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=params.forward_model_weights,
            forward_model_scaling=params.forward_model_scaling,
        )
        logger.debug("Params structure: %s", jax.tree_util.tree_structure(params))
        if isinstance(optimisable_funcs, list):
            optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)
            optimisable_funcs = jnp.round(optimisable_funcs)
        parameter_mask = self.parameter_partition_masks
        self.gradient_mask = self.create_gradient_masks(
            parameter_mask,
            params,
            optimisable_funcs,
        )

        # init seems to be an optax function, hope this isnt confusing
        opt_state = self.optimizer.init(params)  # type: ignore
        self.history = OptimizationHistory()  # reset history
        if self.save_ema_history is True:
            self.ema_history = OptimizationHistory()  # reset ema history

        # Reset learning rates to initial values when initializing
        self.lr_schedule.update(self.initial_learning_rate)
        self.model_lr_schedule.update(self.initial_learning_rate * self.model_parameters_lr_scale)

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
                    ),
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    indexes=tuple(indexes),
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
                    ),
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    indexes=tuple(indexes),
                )
                logger.info("Optimiser JIT compilation successful.")
            except Exception as e:
                logger.warning("JIT compilation failed: %s \n Reverting to non-jit step function", e)
                self.step = self._step
        else:
            logger.debug("No test args provided, skipping JIT compilation.")
            self.step = self._step
        self.gradient_mask = self.create_gradient_masks(
            {Optimisable_Parameters.frame_weights,},
            params,
            optimisable_funcs,
        )
        return OptimizationState(
            params=params,
            opt_state=opt_state,
        )

    @staticmethod
    def create_gradient_masks(
        parameter_partition_masks: set[Optimisable_Parameters],
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
            warnings.warn(
                "The optimisable_funcs argument is deprecated and will be removed in future versions",
                DeprecationWarning,
                stacklevel=2,
            )
            optimisable_funcs = jnp.zeros_like(params.forward_model_weights, dtype=jnp.float32)

        # Create masks based on which parameters are enabled for optimization
        frame_mask = (
            1.0 if Optimisable_Parameters.frame_weights in parameter_partition_masks else 0.0
        )
        model_mask = (
            1.0 if Optimisable_Parameters.model_parameters in parameter_partition_masks else 0.0
        )

        mask_mask = 1.0 if Optimisable_Parameters.frame_mask in parameter_partition_masks else 0.0

        if mask_mask == 1.0:
            raise NotImplementedError(
                "Frame mask optimization not fully implemented - while gradients can flow, "
                "frame masking is not applied during weights normalisation before the forward step"
            )

        logger.debug("Gradient mask partitions: %s", parameter_partition_masks)
        logger.debug("Masks: frame=%s, model=%s, frame_mask=%s", frame_mask, model_mask, mask_mask)

        # Create frame weights mask
        frame_weights_mask = jax.tree_map(
            lambda x: jnp.full_like(x, frame_mask, dtype=jnp.float32), params.frame_weights
        )

        # Create frame mask mask
        frame_mask_mask = jax.tree_map(
            lambda x: jnp.full_like(x, mask_mask, dtype=jnp.float32), params.frame_mask
        )
        logger.debug("Frame mask mask: %s", frame_mask_mask)

        # Create model parameters mask - handle each Model_Parameters instance separately
        model_parameters_mask = []
        for model_param in params.model_parameters:
            # For each Model_Parameters instance, create a mask of the same structure
            masked_model_param = jax.tree_map(
                lambda x: jnp.full_like(x, model_mask, dtype=jnp.float32), model_param
            )
            model_parameters_mask.append(masked_model_param)

        # In create_parameter_partition_masks
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
        logger.debug("Original params structure: %s", jax.tree_util.tree_structure(params))
        logger.debug("Mask structure: %s", jax.tree_util.tree_structure(param_mask))
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

        masked_grads = Simulation_Parameters(
            frame_weights=masked_frame_weights,
            frame_mask=masked_frame_mask,
            model_parameters=masked_model_parameters,
            normalise_loss_functions=masks.normalise_loss_functions,
            forward_model_weights=masks.forward_model_weights,
            forward_model_scaling=masks.forward_model_scaling,
        )
        return masked_grads

    @staticmethod
    def update_history_compute_ema_loss(
        optimizer: "OptaxOptimizer",
        simulation: InitialisedSimulation,
        data_targets: tuple[
            ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters, ...
        ],
        loss_functions: tuple[JaxEnt_Loss, ...],
        indexes: tuple[int, ...],
        state: OptimizationState,
        ema_params: Optional[Simulation_Parameters] = None,
    ) -> "OptaxOptimizer":
        """Update the optimization history and compute the EMA loss if enabled"""

        optimizer.history.add_state(state)
        if optimizer.save_ema_history is True and ema_params is not None:
            params = ema_params

            def loss_fn(params: Simulation_Parameters) -> LossComponents:
                # Update simulation parameters for gradient computation
                losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
                return losses

            losses = loss_fn(params)

            ema_state = OptimizationState(
                params=params,
                opt_state=state.opt_state,
                step=state.step,
                losses=losses,
                gradients=state.gradients,
            )
            optimizer.ema_history.add_state(ema_state)

        return optimizer

    @staticmethod
    def _step(
        optimizer: "OptaxOptimizer",
        state: OptimizationState,
        simulation: InitialisedSimulation,
        data_targets: tuple[
            ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
            ...,
        ],
        loss_functions: tuple[JaxEnt_Loss, ...],
        indexes: tuple[int, ...],
    ) -> tuple[OptimizationState, Array, OptimizationState, InitialisedSimulation]:
        """Perform one optimization step.

        This function is designed to be pure (no Python-level side effects on JAX
        traced values) so it is compatible with ``jax.jit`` and ``jax.vmap``.

        Learning-rate warmup and oscillation-based LR adaptation are intentionally
        handled at the Python level in :func:`_optimise`, where ``step`` is a
        concrete Python integer and mutations of the optimizer's learning-rate
        schedule are safe.
        """
        def loss_fn(params: Simulation_Parameters) -> tuple[Array, LossComponents]:
            # Update simulation parameters for gradient computation
            losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss, losses

        def scalar_loss_fn(params: Simulation_Parameters) -> Array:
            return loss_fn(params)[0]

        # Compute gradients with value and grad to get both loss and gradients
        loss_val_losses, grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
            state.params
        )
        loss_value, losses = loss_val_losses

        # Apply masks to gradients
        masked_grads = optimizer.mask_gradients(grads, optimizer.gradient_mask)

        # Get optimizer updates
        updates, new_opt_state = optimizer.optimizer.update(
            masked_grads,  # type: ignore
            state.opt_state,
            state.params,  # type: ignore
            value=loss_value,
            grad=masked_grads,
            value_fn=scalar_loss_fn,
        )

        # Apply updates
        updated_params = optax.apply_updates(state.params, updates)  # type: ignore

        if optimizer.force_logit_simplex:
            updated_params = Simulation_Parameters.normalize_weights(updated_params)

        # Create new state
        new_state = state.update(updated_params, new_opt_state, losses, masked_grads)

        # Create save state with normalized weights
        save_state = new_state.update(
            Simulation_Parameters.normalize_weights(new_state.params),
            new_state.opt_state,
            losses,
            masked_grads,
            step=new_state.step,
        )

        return new_state, loss_value, save_state, simulation


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
    data_targets: tuple[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
        ...,
    ],
    indexes: tuple[int, ...],
    loss_functions: tuple[JaxEnt_Loss, ...],
) -> LossComponents:
    """Compute training and validation losses"""
    simulation = simulation.forward(simulation, params)

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

    chex.assert_equal_shape([train_losses, val_losses, weights, scaling])


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
