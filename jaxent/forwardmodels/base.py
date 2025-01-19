from abc import ABC, abstractmethod
from typing import Any, Protocol

from MDAnalysis import Universe


class ForwardModel(ABC):
    def __init__(self) -> None:
        self.compatability: set[type]
        self.ensemble: list[Universe]

    @abstractmethod
    def initialise(self, ensemble: list[Universe]) -> bool:
        pass


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(self, ensemble: list[Universe]) -> list[Any]: ...
