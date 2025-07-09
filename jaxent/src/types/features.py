from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, Sequence

import jax.numpy as jnp
from jax import Array

from jaxent.src.types import T_Feat_In, T_Out
from jaxent.src.types.key import m_key


@dataclass()
class Output_Features(Protocol):
    key: ClassVar[m_key]
    __features__: ClassVar[set[str]]

    @property
    def output_shape(self) -> tuple[float, ...]: ...

    # grabs the output shape of the features

    @abstractmethod
    def y_pred(self) -> Array:
        # default to grabbing the first __features__ in the class

        for c in type(self).__mro__:
            if hasattr(c, "__features__"):
                for feature in c.__features__:
                    return jnp.asarray(getattr(self, feature))

        raise AttributeError("No __features__ found in class, check implementation")

    # grabs selected data from features - this is used to build the output features

    @classmethod
    def _get_ordered_slots(cls: type[T_Out]) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls: type[T_Out]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.

        Dynamic slots are all slots not listed in cls.static_params.

        Returns:
            tuple: (dynamic_slots, static_slots)
        """
        dynamic_slots = []
        static_slots = []
        for slot in cls._get_ordered_slots():
            if slot in cls.__features__:
                dynamic_slots.append(slot)
            else:
                static_slots.append(slot)
        return tuple(dynamic_slots), tuple(static_slots)

    def tree_flatten(self) -> tuple[tuple[Array, ...], tuple[Any, ...]]:
        dynamic_slots, static_slots = self._get_grouped_slots()

        # Dynamic parameters become leaves
        arrays = tuple(
            jnp.asarray(getattr(self, slot)).astype(jnp.float32) for slot in dynamic_slots
        )

        # Static parameters go in aux data
        static_data = tuple(getattr(self, slot) for slot in static_slots)

        return arrays, static_data

    @classmethod
    def tree_unflatten(
        cls: type[T_Out], static_data: tuple[Any, ...], arrays: tuple[Array, ...]
    ) -> T_Out:
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)


@dataclass()
class Input_Features(Protocol, Generic[T_Feat_In]):
    __slots__: ClassVar[tuple[str]]
    __features__: ClassVar[set[str]]
    key: ClassVar[set[m_key]]

    @property
    def features_shape(self) -> tuple[float | int, ...]: ...

    # @abstractmethod
    def cast_to_jax(self: "Input_Features") -> "Input_Features":
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
            raise AttributeError("No slots found in class")
        return self
        # raises an error if no slots are found

    @property
    def feat_pred(self) -> Sequence[Array]:
        # default to grabbing the all__features__ in the class
        features = []
        for c in type(self).__mro__:
            if hasattr(c, "__features__"):
                for feature in c.__features__:
                    features.append(jnp.asarray(getattr(self, feature)))
            raise AttributeError("No __features__ found in class, check implementation")
        return features

    @classmethod
    def _get_ordered_slots(cls: type[T_Feat_In]) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls: type[T_Feat_In]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.

        Dynamic slots are all slots not listed in cls.static_params.

        Returns:
            tuple: (dynamic_slots, static_slots)
        """
        dynamic_slots = []
        static_slots = []
        for slot in cls._get_ordered_slots():
            if slot in cls.__features__:
                dynamic_slots.append(slot)
            else:
                static_slots.append(slot)
        return tuple(dynamic_slots), tuple(static_slots)

    def tree_flatten(self) -> tuple[tuple[Array, ...], tuple[Any, ...]]:
        dynamic_slots, static_slots = self._get_grouped_slots()

        # Dynamic parameters become leaves
        arrays = tuple(
            jnp.asarray(getattr(self, slot)).astype(jnp.float32) for slot in dynamic_slots
        )

        # Static parameters go in aux data
        static_data = tuple(getattr(self, slot) for slot in static_slots)

        return arrays, static_data

    @classmethod
    def tree_unflatten(
        cls: type[T_Feat_In], static_data: tuple[Any, ...], arrays: tuple[Array, ...]
    ) -> T_Feat_In:
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)
