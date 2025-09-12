# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from dataclasses import dataclass, field
from functools import partial
from typing import NamedTuple, Optional, Protocol, Sequence, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax import Array

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Model_Parameters, Simulation_Parameters

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
    Array,
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


class OptimizationState(NamedTuple):
    """Represents the state of the optimization at a given step.
    Basic math operations (addition, subtraction, multiplecation, division, etc.) are supported
    between two OptimizationState instances, as all elements in this are jax pytrees and tree map can be used to apply the operations.
    For all math operations the step and opt_state are taken from the right hand side instance.
    """

    params: Simulation_Parameters
    opt_state: optax.OptState
    step: int = 0
    losses: Optional[LossComponents] = None
    gradients: Optional[Simulation_Parameters] = None

    def update(
        self,
        new_params: Simulation_Parameters,
        new_opt_state: optax.OptState,
        new_losses: LossComponents,
        new_gradients: Optional[Simulation_Parameters] = None,
        step: Optional[int] = None,
    ) -> "OptimizationState":
        return OptimizationState(
            params=new_params,
            opt_state=new_opt_state,
            step=(self.step + 1) if step is None else step,
            losses=new_losses if new_losses is not None else self.losses,
            gradients=new_gradients if new_gradients is not None else self.gradients,
        )

    def _apply_op(self, op, other: "OptimizationState") -> "OptimizationState":
        raise NotImplementedError(
            "Not implemented yet - please use the operations in the Simulation_Parameters class"
        )
        if not isinstance(other, OptimizationState):
            raise TypeError("Operand must be an instance of OptimizationState")
        new_params = jax.tree_util.tree_map(op, self.params, other.params)

        new_losses = None
        if self.losses is not None and other.losses is not None:
            new_losses = jax.tree_util.tree_map(op, self.losses, other.losses)

        new_gradients = None
        if self.gradients is not None and other.gradients is not None:
            new_gradients = jax.tree_util.tree_map(op, self.gradients, other.gradients)

        return OptimizationState(
            params=new_params,
            opt_state=other.opt_state,
            step=other.step,
            losses=new_losses,
            gradients=new_gradients,
        )

    def __add__(self, other) -> "OptimizationState":
        return self._apply_op(jnp.add, other)

    def __sub__(self, other) -> "OptimizationState":
        return self._apply_op(jnp.subtract, other)

    def __mul__(self, other) -> "OptimizationState":
        return self._apply_op(jnp.multiply, other)

    def __truediv__(self, other) -> "OptimizationState":
        return self._apply_op(jnp.divide, other)


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
        """Pick the best state based on unscaled validation loss"""

        best_state: OptimizationState = states[-1]
        for state in states:
            # Use the sum of unscaled validation losses instead of total_val_loss
            if jnp.sum(state.losses.val_losses) < jnp.sum(best_state.losses.val_losses):
                best_state = state

        return best_state

    def get_best_state(self) -> OptimizationState:
        """Get the best state based on unscaled validation loss"""

        self.best_state = self._pick_best_state(self.states)

        return self.best_state
