from abc import ABC, abstractmethod
from beartype.typing import Generic, Protocol, Union
from beartype.typing import runtime_checkable   

import MDAnalysis as mda
from MDAnalysis import Universe

from jaxent.src.custom_types import T_Config, T_Feat_In, T_In, T_Out, T_Params
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology

@runtime_checkable
class ForwardPass(Protocol[T_In, T_Out, T_Params]):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    """

    def __call__(self, input_features: T_In, parameters: T_Params) -> T_Out: ...

@runtime_checkable
class Featuriser(Protocol[T_Feat_In]):
    """
    A featuriser is a callable object that takes in a list of Universes and then returns a list of features.
    """

    def __call__(
        self, ensemble: list[Universe]
    ) -> tuple[T_Feat_In, list[Partial_Topology]]: ...


class ForwardModel(ABC, Generic[T_Params, T_In, T_Config]):
    def __init__(self, config: T_Config) -> None:
        self.config: T_Config = config
        self.compatability: dict[m_key, ExpD_Datapoint]
        self.forward: dict[m_key, ForwardPass]
        self.params = config.forward_parameters

    def __post_init__(self):
        self.key = self.config.key

    @abstractmethod
    def initialise(self, ensemble: Union[list[Universe], list[Partial_Topology]]) -> bool:
        """
        Validate compatibility of the forward model with an ensemble.

        Accepts either a list of MDAnalysis Universe objects (for full trajectory-based
        initialisation) or a list of pre-computed Partial_Topology objects (for
        topology-only initialisation when trajectory data is unavailable or unnecessary).

        When initialised from Partial_Topology objects the common topology is set
        directly from the provided topologies. Universe-dependent quantities such as
        intrinsic exchange rates (k_ints) and frame counts are not computed and must be
        supplied through other means before featurisation is called.
        """
        pass

    @abstractmethod
    def featurise(self, ensemble: list[Universe]) -> tuple[T_In, list[Partial_Topology]]:
        pass

    @property
    def forwardpass(self) -> ForwardPass:
        _fp: ForwardPass = self.forward[self.config.key]
        return _fp  # i hope this fixes the typing
