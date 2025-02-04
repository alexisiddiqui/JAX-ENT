from abc import ABC, abstractmethod
from typing import Any, Protocol, Union

from MDAnalysis import Universe


class Output_Features(Protocol):
    @property
    def output_shape(self) -> tuple[int, ...]: ...


class Input_Features(Protocol):
    @property
    def features_shape(self) -> tuple[int, ...]: ...


class Model_Parameters(Protocol):
    @property
    def forward_parameters(self) -> tuple[float, ...]: ...


# T_In = TypeVar("T_In", bound=Input_Features)
# T_Out = TypeVar("T_Out", bound=Output_Features)
# T_Params = TypeVar("T_Params", bound=Model_Parameters)


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(self, ensemble: list[Universe]) -> Input_Features: ...


class ForwardPass(Protocol):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    Returns a list of features.
    """

    def __call__(
        self, input_features: Input_Features, parameters: Model_Parameters
    ) -> Output_Features: ...


class ForwardModel(ABC):
    def __init__(self) -> None:  # TODO use a generic to use BaseConfig as a generic type
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
