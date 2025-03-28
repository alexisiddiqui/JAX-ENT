from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from jax import Array  # noqa: F401

    from jaxent.data.loader import ExpD_Datapoint  # noqa: F401
    from jaxent.interfaces.model import Model_Parameters  # noqa: F401
    from jaxent.models.config import Model_Config  # noqa: F401
    from jaxent.models.core import (
        Simulation,  # noqa: F401
        Simulation_Parameters,  # noqa: F401
    )
    from jaxent.opt.base import JaxEnt_Loss  # noqa: F401
    from jaxent.opt.optimiser import OptaxOptimizer  # noqa: F401
    from jaxent.types.config import OptimiserSettings  # noqa: F401
    from jaxent.types.features import Input_Features, Output_Features  # noqa: F401


T_In = TypeVar("T_In", bound="Input_Features", contravariant=True)
T_Out = TypeVar("T_Out", bound="Output_Features", covariant=True)
T_Params = TypeVar("T_Params", bound="Model_Parameters", contravariant=True)
T_Config = TypeVar("T_Config", bound="Model_Config")
T_Feat_In = TypeVar("T_Feat_In", bound="Input_Features", covariant=True)

T_ExpD = TypeVar("T_ExpD", bound="ExpD_Datapoint")


@runtime_checkable
class InitialisedSimulation(Protocol):
    """
    Protocol representing an initialized simulation with non-optional parameters.
    """

    # input_features: list[Input_Features]
    # forward_models: Sequence[ForwardModel]  # Replace with actual ForwardModel type
    params: "Simulation_Parameters"  # Non-optional
    # forwardpass: Sequence[ForwardPass]  # Replace with actual ForwardPass type
    outputs: Sequence["Output_Features"]

    def forward(self, params: "Simulation_Parameters") -> None: ...
