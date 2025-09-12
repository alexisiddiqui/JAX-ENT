"""

This module builds on the jaxENT loss protocol to define a standard interface for loss functions.


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



"""

from jax import Array

from jaxent.src.opt.base import (
    D,
    JaxEnt_Loss,
    M,
)


class Loss_Fn(