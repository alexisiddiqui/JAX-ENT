from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, Union

from MDAnalysis import Universe


class Output_Features(Protocol):
    @property
    def output_shape(self) -> tuple[float, ...]: ...


class Input_Features(Protocol):
    @property
    def features_shape(self) -> tuple[float | int, ...]: ...


class Model_Parameters(Protocol):
    @property
    def forward_parameters(self) -> tuple[float, ...]: ...


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(self, ensemble: list[Universe]) -> Input_Features: ...


T_In = TypeVar("T_In", bound=Input_Features, contravariant=True)
T_Out = TypeVar("T_Out", bound=Output_Features, covariant=True)
T_Params = TypeVar("T_Params", bound=Model_Parameters, contravariant=True)


class ForwardPass(Protocol[T_In, T_Out, T_Params]):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    """

    def __call__(self, input_features: T_In, parameters: T_Params) -> T_Out: ...


class ForwardModel(ABC, Generic[T_Params]):
    def __init__(self, config: T_Params) -> None:
        self.config: T_Params = config
        self.compatability: Union[Any, Any]
        self.forward: ForwardPass

    @abstractmethod
    def initialise(self, ensemble: list[Universe]) -> bool:
        """
        This should be some form of validation to ensure that the data is compatible with the forward model.
        """
        pass

    @abstractmethod
    def featurise(self, ensemble: list[Universe]) -> Input_Features:
        pass

    @property
    def forwardpass(self) -> ForwardPass:
        return self.forward
