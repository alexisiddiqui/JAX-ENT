# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, TypeVar

import optax
from jax import Array

from jaxent.data.loading import Experimental_Dataset
from jaxent.interfaces.simulation import Model_Parameters, Simulation_Parameters
from jaxent.types.base import ForwardModel, ForwardPass, Input_Features, Output_Features


class InitialisedSimulation(Protocol):
    """
    Protocol representing an initialized simulation with non-optional parameters.
    """

    input_features: list[Input_Features]
    forward_models: Sequence[ForwardModel]  # Replace with actual ForwardModel type
    params: Simulation_Parameters  # Non-optional
    forwardpass: Sequence[ForwardPass]  # Replace with actual ForwardPass type
    outputs: Sequence[Array]

    def forward(self, params: Simulation_Parameters) -> None: ...


M = TypeVar(
    "M",
    # Simulation,
    # Sequence[Simulation],
    InitialisedSimulation,
    Sequence[InitialisedSimulation],
    contravariant=True,
)
D = TypeVar(
    "D",
    Output_Features,
    Experimental_Dataset,
    Model_Parameters,
    Simulation_Parameters,
    contravariant=True,
)


class JaxEnt_Loss(Protocol[M, D]):
    def __call__(
        self, model: M, dataset: D, prediction_index: int | str | None
    ) -> tuple[Array, Array]: ...


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
