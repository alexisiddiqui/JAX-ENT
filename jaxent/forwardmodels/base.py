from abc import ABC, abstractmethod
from typing import ClassVar, Generic, Protocol, TypeVar

import jax.numpy as jnp
from jax import Array
from MDAnalysis import Universe

from jaxent.core_types import Experimental_Fragment, Topology_Fragment, m_key


class Output_Features(Protocol):
    key: m_key

    @property
    def output_shape(self) -> tuple[float, ...]: ...

    # grabs the output shape of the features

    @abstractmethod
    def data(self) -> Array:
        # default to grabbing the first slot
        for c in type(self).__mro__:
            if hasattr(c, "__slots__"):
                return jnp.asarray(getattr(self, c.__slots__[0]))
        raise AttributeError("No slots found in class")

    # grabs selected data from features - this is used to build the output features


class Input_Features(Protocol):
    key: set[m_key]

    @property
    def features_shape(self) -> tuple[float | int, ...]: ...

    @abstractmethod
    def cast_to_jax(self) -> None:
        # casts all features in slots to jax arrays
        for c in type(self).__mro__:
            if hasattr(c, "__slots__"):
                for slot in c.__slots__:
                    # try to cast slot to jax array - if it fails, print a warning
                    # since this wont be compiled we can use try/except to cover improper slots
                    try:
                        setattr(self, slot, jnp.asarray(getattr(self, slot)))
                    # warn if slot is not castable
                    except Exception as e:
                        print(
                            f"Warning: slot {slot} in {self} is not castable to jax array: {e}"
                            f"\n\n\nContinuiing but this may cause issues with the model"
                        )

        # raises an error if no slots are found
        raise AttributeError("No slots found in class")


T = TypeVar("T", bound="Model_Parameters")


class Model_Parameters:
    """Base class providing generic PyTree methods for slots-enabled model parameters"""

    key: frozenset[m_key]
    static_params: ClassVar[set[str]] = {"key"}
    dynamic_params: ClassVar[set[str]] = set()

    ####################################################################################################
    # TODO update this to use static and dynamic parameters instead of trying to get ordered slots
    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including parent classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @abstractmethod
    def tree_flatten(self) -> tuple[tuple[float, ...], tuple]:
        arrays = tuple(
            jnp.asarray(getattr(self, slot)).astype(jnp.float32)
            for slot in self._get_ordered_slots()
        )
        static = ()
        return arrays, static

    @classmethod
    @abstractmethod
    def tree_unflatten(cls: type[T], static: tuple, arrays: tuple[float, ...]) -> T:
        kwargs = {slot: array for slot, array in zip(cls._get_ordered_slots(), arrays)}
        return cls(**kwargs)

    @abstractmethod
    def update_parameters(self, new_params: "Model_Parameters") -> "Model_Parameters":
        """Creates new instance with updated non-static parameters.
        You should override this method in your subclass for faster updates.
        """
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(new_params, slot)
        return self.__class__(**param_dict)


####################################################################################################
class Model_Config(Protocol):
    key: m_key

    @property
    def forward_parameters(self) -> Model_Parameters: ...


class Featuriser(Protocol):
    """
    A featuriser is a callable object that takes in a list of universes and then returns a list of features.
    """

    def __call__(
        self, ensemble: list[Universe]
    ) -> tuple[Input_Features, list[Topology_Fragment]]: ...


T_In = TypeVar("T_In", bound=Input_Features, contravariant=True)
T_Out = TypeVar("T_Out", bound=Output_Features, covariant=True)
T_Params = TypeVar("T_Params", bound=Model_Parameters, contravariant=True)


class ForwardPass(Protocol[T_In, T_Out, T_Params]):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    """

    def __call__(self, input_features: T_In, parameters: T_Params) -> T_Out: ...


T_Config = TypeVar("T_Config", bound=Model_Config)


class ForwardModel(ABC, Generic[T_Params]):
    def __init__(self, config: T_Config) -> None:
        self.config: T_Config = config
        self.compatability: dict[m_key, Experimental_Fragment]
        self.forward: dict[m_key, ForwardPass]
        self.params: T_Params = config.forward_parameters

    def __post_init__(self):
        self.key = self.config.key

    @abstractmethod
    def initialise(self, ensemble: list[Universe]) -> bool:
        """
        This should be some form of validation to ensure that the data is compatible with the forward model.
        """
        pass

    @abstractmethod
    def featurise(self, ensemble: list[Universe]) -> tuple[Input_Features, list[Topology_Fragment]]:
        pass

    @property
    def forwardpass(self) -> ForwardPass:
        _fp: ForwardPass = self.forward[self.config.key]
        return _fp  # i hope this fixes the typing
