# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Optional, Protocol, Sequence, TypeVar

import jax
import optax
from jax import Array

from jaxent.data.loader import ExpD_Dataloader
from jaxent.interfaces.simulation import Model_Parameters, Simulation_Parameters
from jaxent.types import InitialisedSimulation
from jaxent.types.features import Output_Features

# @runtime_checkable
# class InitialisedSimulation(Protocol):
#     """
#     Protocol representing an initialized simulation with non-optional parameters.
#     """

#     # input_features: list[Input_Features]
#     # forward_models: Sequence[ForwardModel]  # Replace with actual ForwardModel type
#     params: Simulation_Parameters  # Non-optional
#     # forwardpass: Sequence[ForwardPass]  # Replace with actual ForwardPass type
#     outputs: Sequence[Output_Features]

#     def forward(self, params: Simulation_Parameters) -> None: ...


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
    ExpD_Dataloader,
    Model_Parameters,
    Simulation_Parameters,
    contravariant=True,
)


class JaxEnt_Loss(Protocol[M, D]):
    def __call__(
        self, model: M, dataset: D, prediction_index: int | str | None
    ) -> tuple[Array, Array]: ...


# @dataclass(frozen=True)
class LossComponents(NamedTuple):
    """Stores the various components of loss for training and validation"""

    train_losses: Array  # Individual training loss components
    val_losses: Array  # Individual validation loss components
    scaled_train_losses: Array  # Scaled training loss components
    scaled_val_losses: Array  # Scaled validation loss components
    total_train_loss: Array  # Total training loss
    total_val_loss: Array  # Total validation loss


# @dataclass(frozen=True)
class OptimizationState(NamedTuple):
    params: Simulation_Parameters
    opt_state: optax.OptState
    gradient_mask: Simulation_Parameters
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
            gradient_mask=self.gradient_mask,
            step=self.step + 1,
            losses=new_losses,
        )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["states", "best_state"],
    meta_fields=[],
)
@dataclass
class OptimizationHistory:
    """Tracks the history of optimization states and metrics"""

    states: list[OptimizationState] = field(default_factory=list)
    best_state: Optional[OptimizationState] = None

    def add_state(self, state: OptimizationState):
        """Add a new state to history and update best state if needed"""
        self.states.append(state)
        # Handle None case separately since 'is None' checks aren't JIT-compatible

    @staticmethod
    def _pick_best_state(states: list[OptimizationState]) -> OptimizationState:
        """Pick the best state based on validation loss"""

        best_state: OptimizationState = states[-1]
        for state in states:
            if state.losses.total_val_loss < best_state.losses.total_val_loss:
                best_state = state

        return best_state

    def get_best_state(self) -> OptimizationState:
        """Get the best state based on validation loss"""

        self.best_state = self._pick_best_state(self.states)

        return self.best_state
