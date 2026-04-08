import logging
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
    OptimisationCarry,
    OptimizationHistory,
    OptimizationState,
)
from jaxent.src.opt.gradients import create_gradient_masks, mask_gradients
from jaxent.src.opt.track import check_and_advance_threshold, update_convergence

LOGGER = logging.getLogger("jaxent.opt")


def _clone_optimization_state(state: OptimizationState) -> OptimizationState:
    """Clone state leaves so donated buffers are not reused by later calls."""
    def _clone_leaf(x):
        if isinstance(x, jax.Array):
            return jnp.array(x, copy=True)
        return x

    return jax.tree_util.tree_map(_clone_leaf, state)


def _dynamic_write(buffer: Array, value: Array, write_idx: Array) -> Array:
    start_indices = (write_idx,) + (0,) * value.ndim
    return jax.lax.dynamic_update_slice(buffer, jnp.expand_dims(value, axis=0), start_indices)


@register_pytree_node_class
class OptaxOptimizer:
    learning_rate: float
    optimizer: optax.GradientTransformation
    parameter_partition_masks: set[Optimisable_Parameters]
    clip_value: Optional[float]
    history: OptimizationHistory
    ema_history: OptimizationHistory | None
    plateau_denominator: float
    force_logit_simplex: bool
    save_ema_history: bool
    step: Callable
    initial_learning_rate: float
    initial_steps: int
    model_parameters_lr_scale: float
    update_all_models: bool = False
    _initial_gradient_mask: Simulation_Parameters
    _final_gradient_mask: Simulation_Parameters
    _current_lr: float
    _current_model_lr: float
    _current_gradient_mask_idx: int

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
        self.plateau_denominator = plateau_denominator
        self.initial_learning_rate = initial_learning_rate
        self.initial_steps = initial_steps
        self.learning_rate = learning_rate
        self.update_all_models = False

        if save_ema_history:
            self.ema_history = OptimizationHistory()
        else:
            self.ema_history = None

        self._current_lr = float(initial_learning_rate)
        self._current_model_lr = float(initial_learning_rate * model_parameters_lr_scale)
        self._current_gradient_mask_idx = 0

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

        if force_simplex is None:
            self.force_logit_simplex = _force_simplex
        else:
            self.force_logit_simplex = force_simplex

        self._initial_gradient_mask = None  # type: ignore[assignment]
        self._final_gradient_mask = None  # type: ignore[assignment]
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
            self._initial_gradient_mask,
            self._final_gradient_mask,
            self.ema_history,
        )
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
            "_current_lr": self._current_lr,
            "_current_model_lr": self._current_model_lr,
            "_current_gradient_mask_idx": self._current_gradient_mask_idx,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        self = cls.__new__(cls)
        self.history = children[0]
        self._initial_gradient_mask = children[1]
        self._final_gradient_mask = children[2]
        self.ema_history = children[3]

        self.learning_rate = aux_data["learning_rate"]
        self.optimizer = aux_data["optimizer"]
        self.parameter_partition_masks = aux_data["parameter_partition_masks"]
        self.clip_value = aux_data["clip_value"]
        self.save_ema_history = aux_data["save_ema_history"]
        self.plateau_denominator = aux_data["plateau_denominator"]
        self.force_logit_simplex = aux_data["force_logit_simplex"]
        self.step = aux_data.get("step", self._step)
        self.initial_learning_rate = aux_data.get("initial_learning_rate", 1.0)
        self.initial_steps = aux_data.get("initial_steps", 0)
        self.model_parameters_lr_scale = aux_data.get("model_parameters_lr_scale", 1.0)
        self.update_all_models = aux_data.get("update_all_models", False)
        self._current_lr = aux_data.get("_current_lr", self.initial_learning_rate)
        self._current_model_lr = aux_data.get(
            "_current_model_lr",
            self.initial_learning_rate * self.model_parameters_lr_scale,
        )
        self._current_gradient_mask_idx = aux_data.get("_current_gradient_mask_idx", 0)
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
        params = Simulation_Parameters(
            frame_mask=params.frame_mask,
            frame_weights=params.frame_weights * len(params.frame_weights),
            model_parameters=params.model_parameters,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=params.forward_model_weights,
            forward_model_scaling=params.forward_model_scaling,
        )
        if isinstance(optimisable_funcs, list):
            optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)
            optimisable_funcs = jnp.round(optimisable_funcs)

        self._initial_gradient_mask = create_gradient_masks(
            {Optimisable_Parameters.frame_weights},
            params,
            optimisable_funcs,
        )
        self._final_gradient_mask = create_gradient_masks(
            self.parameter_partition_masks,
            params,
            optimisable_funcs,
        )

        opt_state = self.optimizer.init(params)  # type: ignore[arg-type]
        self.history = OptimizationHistory()
        if self.save_ema_history:
            self.ema_history = OptimizationHistory()

        self._current_lr = float(self.initial_learning_rate)
        self._current_model_lr = float(self.initial_learning_rate * self.model_parameters_lr_scale)
        self._current_gradient_mask_idx = 0

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
    def _select_gradient_mask(
        optimizer: "OptaxOptimizer",
        gradient_mask_idx: Array,
    ) -> Simulation_Parameters:
        idx = jnp.asarray(gradient_mask_idx, dtype=jnp.int32)
        return jax.tree_util.tree_map(
            lambda initial, final: jax.lax.select(idx == 0, initial, final),
            optimizer._initial_gradient_mask,
            optimizer._final_gradient_mask,
        )

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
            frame_weights=jax.tree_util.tree_map(lambda g: g * lr, grads.frame_weights),
            frame_mask=jax.tree_util.tree_map(lambda g: g * lr, grads.frame_mask),
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
        target_lr: Array,
        target_model_lr: Array,
        gradient_mask_idx: Array,
    ) -> tuple[
        OptimizationState,
        Array,
        OptimizationState,
        InitialisedSimulation,
        Array,
        Array,
        Array,
        Array,
    ]:
        """Pure step implementation with explicit learning-rate and mask state."""

        step = jnp.asarray(state.step, dtype=jnp.int32)
        switched_mask_idx = jax.lax.select(
            step >= jnp.asarray(optimizer.initial_steps, dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32),
            jnp.asarray(gradient_mask_idx, dtype=jnp.int32),
        )
        base_lr = jax.lax.select(
            step >= jnp.asarray(optimizer.initial_steps, dtype=jnp.int32),
            target_lr,
            lr,
        )
        base_model_lr = jax.lax.select(
            step >= jnp.asarray(optimizer.initial_steps, dtype=jnp.int32),
            target_model_lr,
            model_lr,
        )
        gradient_mask = OptaxOptimizer._select_gradient_mask(optimizer, switched_mask_idx)

        def loss_fn(params: Simulation_Parameters) -> tuple[Array, tuple[LossComponents, InitialisedSimulation]]:
            losses, updated_sim = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss, (losses, updated_sim)

        def scalar_loss_fn(params: Simulation_Parameters) -> Array:
            losses, _ = compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss

        (loss_value, aux), grads = jax.value_and_grad(
            loss_fn, allow_int=True, has_aux=True
        )(state.params)
        losses, updated_sim = aux

        masked_grads = mask_gradients(grads, gradient_mask)
        previous_grads = state.gradients if state.gradients is not None else masked_grads
        grad_dot_product = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(lambda a, b: jnp.vdot(a, b), previous_grads, masked_grads),
        )

        reduce_lr = (grad_dot_product < 0) & (step > 1)
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
        if optimizer.force_logit_simplex:
            updated_params = Simulation_Parameters.normalize_weights(updated_params)

        new_state = state.update(updated_params, new_opt_state, losses, masked_grads)
        save_state = new_state.update(
            Simulation_Parameters.normalize_weights(new_state.params),
            new_state.opt_state,
            losses,
            masked_grads,
            step=new_state.step,
        )
        return (
            new_state,
            loss_value,
            save_state,
            updated_sim,
            new_lr,
            new_model_lr,
            switched_mask_idx,
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
    ) -> tuple[OptimizationState, Array, OptimizationState, InitialisedSimulation]:
        """Python-loop step wrapper that stores dynamic LR/mask state on optimizer."""
        (
            new_state,
            loss_value,
            save_state,
            updated_sim,
            new_lr,
            new_model_lr,
            new_mask_idx,
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
            target_lr=jnp.asarray(optimizer.learning_rate, dtype=jnp.float32),
            target_model_lr=jnp.asarray(
                optimizer.learning_rate * optimizer.model_parameters_lr_scale,
                dtype=jnp.float32,
            ),
            gradient_mask_idx=jnp.asarray(optimizer._current_gradient_mask_idx, dtype=jnp.int32),
        )

        if not isinstance(new_lr, jax.core.Tracer):
            optimizer._current_lr = float(new_lr)
        if not isinstance(new_model_lr, jax.core.Tracer):
            optimizer._current_model_lr = float(new_model_lr)
        if not isinstance(new_mask_idx, jax.core.Tracer):
            optimizer._current_gradient_mask_idx = int(new_mask_idx)
        return new_state, loss_value, save_state, updated_sim

    @staticmethod
    def _pure_step(
        optimizer: "OptaxOptimizer",
        carry: OptimisationCarry,
        data_targets: tuple[
            ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
            ...,
        ],
        loss_functions: tuple[JaxEnt_Loss, ...],
        indexes: tuple[int, ...],
        convergence_thresholds: Array,
        ema_alpha: float,
        min_steps_per_threshold: int,
        target_lr: Array,
        target_model_lr: Array,
    ) -> OptimisationCarry:
        """Side-effect-free step used by ``_optimise_pure``."""
        (
            new_state,
            loss_value,
            save_state,
            updated_sim,
            new_lr,
            new_model_lr,
            new_mask_idx,
            _,
        ) = OptaxOptimizer._step_with_rates(
            optimizer=optimizer,
            state=carry.opt_state,
            simulation=carry.sim,
            data_targets=data_targets,
            loss_functions=loss_functions,
            indexes=indexes,
            lr=carry.lr,
            model_lr=carry.model_lr,
            target_lr=target_lr,
            target_model_lr=target_model_lr,
            gradient_mask_idx=carry.gradient_mask_idx,
        )

        previous_loss = (
            carry.opt_state.losses.total_train_loss
            if carry.opt_state.losses is not None
            else loss_value
        )
        new_convergence, _ = update_convergence(
            carry=carry.convergence,
            previous_loss=previous_loss,
            current_loss=loss_value,
            current_params=save_state.params,
            ema_alpha=ema_alpha,
        )
        new_convergence = check_and_advance_threshold(
            carry=new_convergence,
            current_loss=loss_value,
            step=jnp.asarray(new_state.step, dtype=jnp.int32),
            thresholds=convergence_thresholds,
            min_steps=min_steps_per_threshold,
            initial_steps=optimizer.initial_steps,
        )

        write_idx = jnp.asarray(carry.write_idx, dtype=jnp.int32)
        new_history_params = jax.tree_util.tree_map(
            lambda buf, val: _dynamic_write(buf, val, write_idx),
            carry.history_params,
            save_state.params,
        )
        if save_state.losses is None:
            raise ValueError("save_state.losses cannot be None in pure optimisation path")
        new_history_losses = jax.tree_util.tree_map(
            lambda buf, val: _dynamic_write(buf, val, write_idx),
            carry.history_losses,
            save_state.losses,
        )

        return OptimisationCarry(
            opt_state=new_state,
            sim=carry.sim,
            convergence=new_convergence,
            lr=new_lr,
            model_lr=new_model_lr,
            gradient_mask_idx=new_mask_idx,
            history_params=new_history_params,
            history_losses=new_history_losses,
            write_idx=write_idx + 1,
        )

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
        """Update optimization history and optional EMA history."""
        optimizer.history.add_state(state)
        if optimizer.save_ema_history and optimizer.ema_history is not None and ema_params is not None:
            losses, _ = compute_loss(simulation, ema_params, data_targets, indexes, loss_functions)
            ema_state = OptimizationState(
                params=ema_params,
                opt_state=state.opt_state,
                step=state.step,
                losses=losses,
                gradients=state.gradients,
            )
            optimizer.ema_history.add_state(ema_state)
        return optimizer


def compute_loss(
    simulation: InitialisedSimulation,
    params: Simulation_Parameters,
    data_targets: tuple[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters,
        ...,
    ],
    indexes: tuple[int, ...],
    loss_functions: tuple[JaxEnt_Loss, ...],
) -> tuple[LossComponents, InitialisedSimulation]:
    """Compute losses and return the updated functional simulation."""
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
    return loss_components, simulation
