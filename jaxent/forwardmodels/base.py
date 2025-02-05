from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeVar, Union

from MDAnalysis import Universe


class Output_Features(Protocol):
    @property
    def output_shape(self) -> tuple[float, ...]: ...


class Input_Features(Protocol):
    @property
    def features_shape(self) -> tuple[float | int, ...]: ...


# @dataclass(frozen=True)
# class Model_Parameters:
# temperature: float = 300
#     ph: float = 7


T = TypeVar("T", bound="Model_Parameters")


class Model_Parameters:
    """Base class providing generic PyTree methods for slots-enabled model parameters"""

    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including parent classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    def tree_flatten(self) -> tuple[tuple[float, ...], tuple]:
        arrays = tuple(float(getattr(self, slot)) for slot in self._get_ordered_slots())
        static = ()
        return arrays, static

    @classmethod
    def tree_unflatten(cls: type[T], static: tuple, arrays: tuple[float, ...]) -> T:
        kwargs = {slot: array for slot, array in zip(cls._get_ordered_slots(), arrays)}
        return cls(**kwargs)


class Model_Config(Protocol):
    @property
    def forward_parameters(self) -> Model_Parameters: ...


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
    def __init__(self, config: Model_Config) -> None:
        self.config: Model_Config = config
        self.compatability: Union[Any, Any]
        self.forward: ForwardPass
        self.params: T_Params = self.config.forward_parameters

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
