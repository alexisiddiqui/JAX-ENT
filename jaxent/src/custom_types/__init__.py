from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

from beartype.typing import runtime_checkable

from jaxent.src.custom_types.config import Model_Config  # noqa: F401
from jaxent.src.custom_types.SAXS import SAXS_curve  # noqa: F401
from jaxent.src.custom_types.XLMS import XLMS_distance_restraint  # noqa: F401
if TYPE_CHECKING:
    from jaxent.src.custom_types.datapoint import ExpD_Datapoint  # noqa: F401
    from jaxent.src.custom_types.features import Input_Features, Output_Features  # noqa: F401
    from jaxent.src.interfaces.model import Model_Parameters  # noqa: F401
    from jaxent.src.models.core import Simulation_Parameters  # noqa: F401


T_In = TypeVar("T_In", bound="jaxent.src.custom_types.features.Input_Features", contravariant=True)
T_Out = TypeVar("T_Out", bound="jaxent.src.custom_types.features.Output_Features", covariant=True)
T_Params = TypeVar("T_Params", bound="jaxent.src.interfaces.model.Model_Parameters", contravariant=True)
T_Config = TypeVar("T_Config", bound=Model_Config)
T_Feat_In = TypeVar("T_Feat_In", bound="jaxent.src.custom_types.features.Input_Features", covariant=True)

T_ExpD = TypeVar("T_ExpD", bound="jaxent.src.custom_types.datapoint.ExpD_Datapoint")


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
