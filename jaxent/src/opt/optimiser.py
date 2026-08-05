import logging
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional

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
from jaxent.src.opt.gradients import create_gradient_masks, mask_gradients

LOGGER = logging.getLogger("jaxent.opt")


def _clone_optimization_state(state: OptimizationState) -> OptimizationState:
    """Clone state leaves so donated buffers are not reused by later calls."""
    def _clone_leaf(x):
        if isinstance(x, jax.Array):
            return jnp.array(x, copy=True)
        return x

    return jax.tree_util.tree_map(_clone_leaf, state)


@register_pytree_node_class
class OptaxOptimizer:
    learning_rate: float
    optimizer: optax.GradientTransformation
    parameter_partition_masks: set[Optimisable_Parameters]
    clip_value: Optional[float]
    history: OptimizationHistory
    plateau_denominator: float
    step: Callable
    model_parameters_lr_scale: float
    lr_adjustment: bool
    update_all_models: bool = False
    _gradient_mask: Simulation_Parameters
    _current_lr: float
    _current_model_lr: float

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
        model_parameters_lr_scale: float = 1.0,
        lr_adjustment: bool = True,
    ):
        self.parameter_partition_masks = parameter_partition_masks
        self.clip_value = clip_value
        self.history = OptimizationHistory()
        self.model_parameters_lr_scale = model_parameters_lr_scale
        if not isinstance(lr_adjustment, bool):
            raise ValueError("lr_adjustment must be a boolean")
        self.lr_adjustment = lr_adjustment
        self.plateau_denominator = plateau_denominator
        self.learning_rate = learning_rate
        self.update_all_models = False

        self._current_lr = float(learning_rate)
        self._current_model_lr = float(learning_rate * model_parameters_lr_scale)

        if optimizer.lower() == "adam":
            base_optimizer_fn = optax.adam
        elif optimizer.lower() == "sgd":
            base_optimizer_fn = optax.sgd
        elif optimizer.lower() == "adagrad":
            base_optimizer_fn = optax.adagrad
        elif optimizer.lower() == "adamw":
            base_optimizer_fn = optax.adamw
        elif optimizer.lower() == "rmsprop":
            base_optimizer_fn = optax.rmsprop
        elif optimizer.lower() == "lbfgs":
            base_optimizer_fn = optax.lbfgs
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Unit learning rate chains. Dynamic per-step scaling is applied to gradients.
        frame_chain = []
        if clip_value is not None:
            frame_chain.append(optax.clip(clip_value))
        frame_chain.append(optax.inject_hyperparams(base_optimizer_fn)(learning_rate=1.0))

        model_chain = []
        if clip_value is not None:
            model_chain.append(optax.clip(clip_value))
        model_chain.append(optax.inject_hyperparams(base_optimizer_fn)(learning_rate=1.0))
        model_chain.append(optax.keep_params_nonnegative())

        other_chain = []
        if clip_value is not None:
            other_chain.append(optax.clip(clip_value))
        other_chain.append(optax.inject_hyperparams(base_optimizer_fn)(learning_rate=1.0))

        self.optimizer = optax.multi_transform(
            transforms={
                "frame": optax.chain(*frame_chain),
                "model": optax.chain(*model_chain),
                "other": optax.chain(*other_chain),
            },
            param_labels=Simulation_Parameters.param_labels,
        )

        if force_simplex is not None:
            warnings.warn(
                "force_simplex is deprecated and ignored; frame-weight simplex values "
                "are now always derived from logits.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._gradient_mask = None  # type: ignore[assignment]
        self.step = self._step

    @property
    def current_learning_rate(self) -> float:
        return self._current_lr

    @property
    def current_model_learning_rate(self) -> float:
        return self._current_model_lr

    def tree_flatten(self):
        children = (
            self.history,
            self._gradient_mask,
        )
        aux_data = {
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "parameter_partition_masks": self.parameter_partition_masks,
            "clip_value": self.clip_value,
            "plateau_denominator": self.plateau_denominator,
            "step": self.step,
            "model_parameters_lr_scale": self.model_parameters_lr_scale,
            "lr_adjustment": self.lr_adjustment,
            "update_all_models": self.update_all_models,
            "_current_lr": self._current_lr,
            "_current_model_lr": self._current_model_lr,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        self = cls.__new__(cls)
        self.history = children[0]
        self._gradient_mask = children[1]

        self.learning_rate = aux_data["learning_rate"]
        self.optimizer = aux_data["optimizer"]
        self.parameter_partition_masks = aux_data["parameter_partition_masks"]
        self.clip_value = aux_data["clip_value"]
        self.plateau_denominator = aux_data["plateau_denominator"]
        self.step = aux_data.get("step", self._step)
        self.model_parameters_lr_scale = aux_data.get("model_parameters_lr_scale", 1.0)
        self.lr_adjustment = aux_data.get("lr_adjustment", True)
        self.update_all_models = aux_data.get("update_all_models", False)
        self._current_lr = aux_data.get("_current_lr", self.learning_rate)
        self._current_model_lr = aux_data.get(
            "_current_model_lr",
            self.learning_rate * self.model_parameters_lr_scale,
        )
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
        """Initialize optimization state and pre-compute gradient masks."""
        params = model.params
        if isinstance(optimisable_funcs, list):
            optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)
            optimisable_funcs = jnp.round(optimisable_funcs)

        self._gradient_mask = create_gradient_masks(
            self.parameter_partition_masks,
            params,
            optimisable_funcs,
        )

        opt_state = self.optimizer.init(params)  # type: ignore[arg-type]
        self.history = OptimizationHistory()
        self._current_lr = float(self.learning_rate)
        self._current_model_lr = float(self.learning_rate * self.model_parameters_lr_scale)

        self.step = self._step
        if _jit_test_args is not None:
            data_targets, loss_functions, indexes = _jit_test_args
            smoke_state = OptimizationState(params=params, opt_state=opt_state)
            try:
                _ = self.step(
                    self,
                    smoke_state,
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    tuple(indexes),
                )
                self.step = jax.jit(
                    self._step,
                    static_argnames=("loss_functions", "indexes"),
                )
                # Smoke call with cloned state because donated buffers are invalidated.
                _ = self.step(
                    self,
                    _clone_optimization_state(smoke_state),
                    model,
                    tuple(data_targets),
                    tuple(loss_functions),
                    tuple(indexes),
                )
                LOGGER.info("Optimizer step JIT compilation successful.")
            except Exception as exc:  # pragma: no cover - defensive fallback
                LOGGER.warning("Optimizer step JIT compilation failed; using eager step: %s", exc)
                self.step = self._step

        return OptimizationState(params=params, opt_state=opt_state)

    @staticmethod
    def _scale_gradients(
        grads: Simulation_Parameters,
        lr: Array,
        model_lr: Array,
    ) -> Simulation_Parameters:
        scaled_model_parameters = [
            jax.tree_util.tree_map(lambda g: g * model_lr, model_param)
            for model_param in grads.model_parameters
        ]
        return Simulation_Parameters(
            frame_weight_logits=jax.tree_util.tree_map(
                lambda g: g * lr, grads.frame_weight_logits
            ),
            model_parameters=scaled_model_parameters,
            normalise_loss_functions=grads.normalise_loss_functions * lr,
            forward_model_weights=grads.forward_model_weights * lr,
            forward_model_scaling=grads.forward_model_scaling * lr,
        )

    @staticmethod
    def _step_with_rates(
        optimizer: "OptaxOptimizer",
        state: OptimizationState,
        simulation: InitialisedSimulation,
        data_targets: tuple[
            ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
            ...,
        ],
        loss_functions: tuple[JaxEnt_Loss, ...],
        indexes: tuple[int, ...],
        lr: Array,
        model_lr: Array,
        base_lr: Array,
        base_model_lr: Array,
    ) -> tuple[
        OptimizationState,
        Array,
        OptimizationState,
        Array,
        Array,
        Array,
    ]:
        """Pure step implementation preserving the legacy per-step LR damping."""

        step = jnp.asarray(state.step, dtype=jnp.int32)
        # Before bf528da, production runs used ``initial_steps=0``.  The base
        # rates were therefore restored to their configured targets on every
        # step, with oscillation damping applied only to that step.  Carrying a
        # damped rate into the next step compounds reductions and changes the
        # optimisation trajectory.
        base_lr = jnp.asarray(base_lr, dtype=jnp.float32)
        base_model_lr = jnp.asarray(base_model_lr, dtype=jnp.float32)
        gradient_mask = optimizer._gradient_mask

        def loss_fn(params: Simulation_Parameters) -> tuple[Array, LossComponents]:
            losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss, losses

        def scalar_loss_fn(params: Simulation_Parameters) -> Array:
            losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss

        (loss_value, _), grads = jax.value_and_grad(
            loss_fn, allow_int=True, has_aux=True
        )(state.params)

        masked_grads = mask_gradients(grads, gradient_mask)
        previous_grads = state.gradients if state.gradients is not None else masked_grads
        grad_dot_product = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(lambda a, b: jnp.vdot(a, b), previous_grads, masked_grads),
        )

        reduce_lr = optimizer.lr_adjustment & (grad_dot_product < 0) & (step > 1)
        new_lr = jax.lax.select(
            reduce_lr,
            base_lr / optimizer.plateau_denominator,
            base_lr,
        )
        new_model_lr = jax.lax.select(
            reduce_lr,
            base_model_lr / optimizer.plateau_denominator,
            base_model_lr,
        )

        scaled_grads = OptaxOptimizer._scale_gradients(masked_grads, new_lr, new_model_lr)
        updates, new_opt_state = optimizer.optimizer.update(
            scaled_grads,  # type: ignore[arg-type]
            state.opt_state,
            state.params,  # type: ignore[arg-type]
            value=loss_value,
            grad=scaled_grads,
            value_fn=scalar_loss_fn,
        )
        updated_params = optax.apply_updates(state.params, updates)  # type: ignore[arg-type]

        # The gradient is evaluated at ``state.params``, but snapshots retain
        # ``updated_params``.  Re-evaluate after applying the update so every
        # recorded loss describes the parameters stored beside it.
        updated_losses = compute_loss(
            simulation,
            updated_params,
            data_targets,
            indexes,
            loss_functions,
        )
        updated_loss_value = updated_losses.total_train_loss
        new_state = state.update(
            updated_params, new_opt_state, updated_losses, masked_grads
        )
        save_state = new_state.update(
            new_state.params,
            new_state.opt_state,
            updated_losses,
            masked_grads,
            step=new_state.step,
        )
        return (
            new_state,
            updated_loss_value,
            save_state,
            new_lr,
            new_model_lr,
            grad_dot_product,
        )

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
    ) -> tuple[OptimizationState, Array, OptimizationState]:
        """Python-loop step wrapper that stores dynamic LR state on optimizer."""
        (
            new_state,
            loss_value,
            save_state,
            new_lr,
            new_model_lr,
            _,
        ) = OptaxOptimizer._step_with_rates(
            optimizer=optimizer,
            state=state,
            simulation=simulation,
            data_targets=data_targets,
            loss_functions=loss_functions,
            indexes=indexes,
            lr=jnp.asarray(optimizer._current_lr, dtype=jnp.float32),
            model_lr=jnp.asarray(optimizer._current_model_lr, dtype=jnp.float32),
            base_lr=jnp.asarray(optimizer.learning_rate, dtype=jnp.float32),
            base_model_lr=jnp.asarray(
                optimizer.learning_rate * optimizer.model_parameters_lr_scale,
                dtype=jnp.float32,
            ),
        )

        if not isinstance(new_lr, jax.core.Tracer):
            optimizer._current_lr = float(new_lr)
        if not isinstance(new_model_lr, jax.core.Tracer):
            optimizer._current_model_lr = float(new_model_lr)
        return new_state, loss_value, save_state


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
    """Compute and return loss components."""
    simulation, _ = simulation.forward(simulation, params, mutate=False)

    losses = [
        loss_fn(simulation, target, idx)
        for loss_fn, target, idx in zip(loss_functions, data_targets, indexes)
    ]
    train_losses, val_losses = map(jnp.array, zip(*losses))

    weights = simulation.params.forward_model_weights
    scaling = jnp.array(simulation.params.forward_model_scaling)
    scaled_train = train_losses * weights * scaling
    scaled_val = val_losses * weights * scaling

    chex.assert_equal_shape([train_losses, val_losses, weights, scaling])

    total_train = jnp.sum(scaled_train)
    total_val = jnp.sum(scaled_val)
    loss_components = LossComponents(
        train_losses=train_losses,
        val_losses=val_losses,
        scaled_train_losses=scaled_train,
        scaled_val_losses=scaled_val,
        total_train_loss=total_train,
        total_val_loss=total_val,
    )
    return loss_components
