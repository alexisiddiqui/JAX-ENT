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


class OptaxOptimizer:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        parameter_masks: Optional[Sequence[Optimisable_Parameters]] = [Optimisable_Parameters(1)],
        clip_value: Optional[float] = 1.0,
    ):
        self.learning_rate = learning_rate
        self.parameter_masks = parameter_masks or list(Optimisable_Parameters)
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

        self.optimizer = optax.chain(*optimizer_chain)

    def create_parameter_masks(self, params: Simulation_Parameters) -> Simulation_Parameters:
        """Creates gradient masks as a Simulation_Parameters instance."""
        frame_mask = Optimisable_Parameters.frame_weights in self.parameter_masks
        model_mask = Optimisable_Parameters.model_parameters in self.parameter_masks
        forward_mask = Optimisable_Parameters.forward_model_weights in self.parameter_masks

        frame_weights_mask = jax.tree_map(lambda _: frame_mask, params.frame_weights)
        model_parameters_mask = [
            jax.tree_map(lambda _: model_mask, mp) for mp in params.model_parameters
        ]
        forward_model_weights_mask = jax.tree_map(
            lambda _: forward_mask, params.forward_model_weights
        )

        return Simulation_Parameters(
            frame_weights=frame_weights_mask,
            model_parameters=model_parameters_mask,
            forward_model_weights=forward_model_weights_mask,
            forward_model_scaling=params.forward_model_scaling,
        )

    def init(self, params: Simulation_Parameters) -> OptimizationState:
        """Initialize the optimization state with parameter masks"""
        parameter_masks = self.create_parameter_masks(params)
        init_state = OptimizationState(
            params=params, opt_state=self.optimizer.init(params), parameter_masks=parameter_masks
        )
        return init_state

    def compute_loss(
        self,
        simulation: Simulation,
        params: Simulation_Parameters,
        data_targets: Sequence[Experimental_Dataset | Model_Parameters | Output_Features],
        indexes: Sequence[int],
        loss_functions: Sequence[JaxEnt_Loss],
    ) -> LossComponents:
        """Compute training and validation losses"""

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
        simulation.forward()
        print("Forward pass done")

        def loss_fn(params: Simulation_Parameters):
            # Only compute training loss for gradient calculation
            losses = self.compute_loss(simulation, params, data_targets, indexes, loss_functions)
            return losses.total_train_loss

        # Compute gradients using only training loss
        grads = jax.grad(loss_fn)(state.params)
        print("Gradients computed", grads)
        # Apply masks to gradients
        masked_grads = jax.tree_map(
            lambda g, m: g * m if m is not None else g, grads, state.parameter_masks
        )

        # Get optimizer updates
        updates, new_opt_state = self.optimizer.update(masked_grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        # Compute losses again for reporting/tracking (outside of gradient computation)
        losses = self.compute_loss(simulation, new_params, data_targets, indexes, loss_functions)

        # Create new state with updated losses
        new_state = state.update(new_params, new_opt_state, losses)

        # Add to history
        self.history.add_state(new_state)

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
            learning_rate=config.learning_rate,
            optimizer="adam",
            parameter_masks=[Optimisable_Parameters.model_parameters],
        )

    opt_state = optimizer.init(simulation.params)

    for step in range(config.n_steps):
        opt_state, current_loss = optimizer.step(
            opt_state, simulation, data_to_fit, loss_functions, indexes
        )
        # if step % 100 == 0:
        print(f"Step {step}")
        print(f"Training Loss: {opt_state.losses.total_train_loss}")
        print(f"Validation Loss: {opt_state.losses.total_val_loss}")

        simulation.params = opt_state.params
        if current_loss < config.tolerance:
            print(f"Reached convergence tolerance at step {step}")
            break

        if (
            step > 10
            and abs(current_loss - optimizer.history.best_state.losses.total_train_loss)
            < config.convergence
        ):
            print(f"Loss converged at step {step}")
            break

        # Check convergence on training loss

    if optimizer.history.best_state is not None:
        simulation.params = optimizer.history.best_state.params

    return simulation, optimizer.history
