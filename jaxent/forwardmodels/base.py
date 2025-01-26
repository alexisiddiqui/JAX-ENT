from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Protocol, Union

from MDAnalysis import Universe


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


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(self, ensemble: list[Universe]) -> list[Any]: ...


class Output_Features(NamedTuple):
    pass


class Input_Features(NamedTuple):
    pass


class Model_Parameters(NamedTuple):
    pass


class ForwardPass(Protocol):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    Returns a list of features.
    """

    def __call__(
        self, input_features: Input_Features, parameters: Model_Parameters
    ) -> Output_Features: ...
