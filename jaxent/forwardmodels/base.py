from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Union

from MDAnalysis import Universe


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(self, ensemble: list[Universe]) -> list[Any]: ...


@dataclass(frozen=True)
class Output_Features(Protocol):
    @property
    def output_shape(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class Input_Features(Protocol):
    @property
    def features_shape(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class Model_Parameters(Protocol):
    @property
    def forward_parameters(self) -> tuple[float, ...]: ...


class ForwardPass(Protocol):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    Returns a list of features.
    """

    def __call__(
        self, input_features: Input_Features, parameters: Model_Parameters.forward_parameters
    ) -> Output_Features: ...


class ForwardModel(ABC):
    def __init__(self) -> None:
        self.compatability: Union[type, type]
        self.ensemble: list[Universe]
        self.forward: ForwardPass

    @abstractmethod
    def initialise(self, ensemble: list[Universe]) -> bool:
        """
        This should be some form of validation to ensure that the data is compatible with the forward model.
        """
        pass

    @abstractmethod
    def featurise(self, ensemble: list[Universe]) -> list[Any]:
        pass

    @property
    def forwardpass(self) -> ForwardPass:
        return self.forward
