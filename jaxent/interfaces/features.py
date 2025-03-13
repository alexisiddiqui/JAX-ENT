from abc import abstractmethod
from typing import ClassVar, Protocol

import jax.numpy as jnp
from jax import Array

from jaxent.types.base import m_key


class Output_Features(Protocol):
    key: ClassVar[m_key]

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
    __slots__: ClassVar[tuple[str]]
    __features__: ClassVar[set[str]]
    key: ClassVar[set[m_key]]

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
