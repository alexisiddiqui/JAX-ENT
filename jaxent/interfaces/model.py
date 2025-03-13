########################################################################
# TODO need to simplify code using _create_modified_instance and lambda functions - using this we can then use lax to speed up the optimisation
from abc import abstractmethod
from typing import Any, Callable, ClassVar, TypeVar, cast

import jax.numpy as jnp
from jax import Array

from jaxent.types.base import m_key

T_mp = TypeVar("T_mp", bound="Model_Parameters")


class Model_Parameters:
    """Base class providing generic PyTree methods for slots-enabled model parameters"""

    key: frozenset[m_key]
    static_params: ClassVar[set[str]] = {"key"}
    # dynamic_params: ClassVar[set[str]] | None

    @classmethod
    def _get_ordered_slots(cls: type[T_mp]) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls: type[T_mp]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get dynamic and static slots.

        Dynamic slots are all slots not listed in cls.static_params.
        If cls.dynamic_params is defined, it should be a subset of these dynamic slots.

        Returns:
            tuple: (dynamic_slots, static_slots)
        """
        dynamic_slots = []
        static_slots = []
        for slot in cls._get_ordered_slots():
            if slot in cls.static_params:
                static_slots.append(slot)
            else:
                dynamic_slots.append(slot)
        return tuple(dynamic_slots), tuple(static_slots)

    def _create_modified_instance(self: T_mp, modifier: Callable[[str, Any], Any]) -> T_mp:
        """Create a new instance with modified dynamic parameters.

        Args:
            modifier: Function taking (slot_name, slot_value) and returning new value
        """
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = modifier(slot, getattr(self, slot))
        return cast(T_mp, self.__class__(**param_dict))

    # # these are currently used during the optimisation process - we suggest that you implement these methods to speed up these operations
    # @abstractmethod
    # def tree_flatten(self) -> tuple[tuple[Array, ...], tuple]:
    #     arrays = tuple(
    #         jnp.asarray(getattr(self, slot)).astype(jnp.float32)
    #         for slot in self._get_ordered_slots()
    #     )
    #     static = ()
    #     return arrays, static

    # # these are currently used during the optimisation process - we suggest that you implement these methods to speed up these operations
    # @classmethod
    # @abstractmethod
    # def tree_unflatten(cls: type[T_mp], static: tuple, arrays: tuple[Array, ...]) -> T_mp:
    #     kwargs = {slot: array for slot, array in zip(cls._get_ordered_slots(), arrays)}
    #     return cls(**kwargs)

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
        cls: type[T_mp], static_data: tuple[Any, ...], arrays: tuple[Array, ...]
    ) -> T_mp:
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)

    # these are currently used during the optimisation process - we suggest that you implement these methods to speed up these operations
    @abstractmethod
    def update_parameters(self: T_mp, new_params: T_mp) -> T_mp:
        """Creates new instance with updated non-static parameters.
        You should override this method in your subclass for faster updates.
        """
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(new_params, slot)
        return cast(T_mp, self.__class__(**param_dict))

    def __add__(self: T_mp, other: T_mp) -> T_mp:
        """Add two model parameters together"""
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(self, slot) + getattr(other, slot)
        return cast(T_mp, self.__class__(**param_dict))

    # these are currently used during the optimisation process - we suggest that you implement these methods to speed up these operations
    @abstractmethod
    def __sub__(self: T_mp, other: T_mp) -> T_mp:
        """Subtract other model parameters from self"""
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(self, slot) - getattr(other, slot)
        return cast(T_mp, self.__class__(**param_dict))

    # these are currently used during the optimisation process - we suggest that you implement these methods to speed up these operations
    @abstractmethod
    def __mul__(self: T_mp, scalar: float | Array) -> T_mp:
        """Multiply model parameters by a scalar"""
        scalar = jnp.asarray(scalar)
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(self, slot) * scalar
        return cast(T_mp, self.__class__(**param_dict))

    __rmul__ = __mul__

    def __truediv__(self: T_mp, scalar: float | Array) -> T_mp:
        """Divide model parameters by a scalar"""
        scalar = jnp.asarray(scalar)
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = getattr(self, slot) / scalar
        return cast(T_mp, self.__class__(**param_dict))

    def __neg__(self: T_mp) -> T_mp:
        """Negate model parameters"""
        param_dict = {}
        for slot in self._get_ordered_slots():
            if slot in self.static_params:
                param_dict[slot] = getattr(self, slot)
            else:
                param_dict[slot] = -getattr(self, slot)
        return cast(T_mp, self.__class__(**param_dict))

    def __pos__(self: T_mp) -> T_mp:
        """Unary positive of model parameters (no-op)"""
        return cast(
            T_mp,
            self.__class__(**{slot: getattr(self, slot) for slot in self._get_ordered_slots()}),
        )
