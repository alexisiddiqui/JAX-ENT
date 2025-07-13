from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.key import m_key


@dataclass(slots=True, frozen=True)
class BV_input_features:
    """
    Concrete implementation of Input_Features for BV input features.
    """

    heavy_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    acceptor_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    k_ints: Optional[list] | Optional[Array] = None

    __features__: ClassVar[set[str]] = {"heavy_contacts", "acceptor_contacts"}
    key: ClassVar[set[m_key]] = {m_key("HDX_resPF"), m_key("HDX_peptide")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        if not (
            isinstance(self.heavy_contacts, Array) and isinstance(self.acceptor_contacts, Array)
        ):
            length = len(self.heavy_contacts[0])
            heavy_shape = len(self.heavy_contacts)
            acceptor_shape = len(self.acceptor_contacts)
        elif isinstance(self.heavy_contacts, Array) and isinstance(self.acceptor_contacts, Array):
            length = self.heavy_contacts.shape[1]
            heavy_shape = self.heavy_contacts.shape[0]
            acceptor_shape = self.acceptor_contacts.shape[0]
        else:
            raise TypeError("heavy_contacts and acceptor_contacts must be of the same type")
        return (heavy_shape, acceptor_shape, length)

    def cast_to_jax(self) -> "BV_input_features":
        """Cast features to JAX arrays and return a new instance with JAX arrays"""
        # Create new arrays only if not already JAX arrays
        heavy_contacts = (
            jnp.asarray(self.heavy_contacts)
            if not isinstance(self.heavy_contacts, Array)
            else self.heavy_contacts
        )
        acceptor_contacts = (
            jnp.asarray(self.acceptor_contacts)
            if not isinstance(self.acceptor_contacts, Array)
            else self.acceptor_contacts
        )
        k_ints = (
            jnp.asarray(self.k_ints)
            if self.k_ints is not None and not isinstance(self.k_ints, Array)
            else self.k_ints
        )
        # Return a new instance with the JAX arrays
        return BV_input_features(heavy_contacts, acceptor_contacts, k_ints)

    @property
    def feat_pred(self) -> Sequence[Array]:
        return (jnp.asarray(self.heavy_contacts), jnp.asarray(self.acceptor_contacts))

    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.
        Dynamic slots are all slots in cls.__features__.
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
    def tree_unflatten(cls, static_data: tuple[Any, ...], arrays: tuple[Array, ...]):
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)


@dataclass(frozen=True, slots=True)
class BV_output_features:
    """Concrete implementation of Output_Features for BV output features."""

    log_Pf: list | Sequence[float] | Array  # (1, residues)
    k_ints: Optional[list] | Optional[Array] = None

    __features__: ClassVar[set[str]] = {"log_Pf", "k_ints"}
    key: ClassVar[m_key] = m_key("HDX_resPF")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.log_Pf))

    def y_pred(self) -> Array:
        return jnp.asarray(self.log_Pf)

    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.
        Dynamic slots are all slots in cls.__features__.
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
            jnp.asarray(getattr(self, slot)).astype(jnp.float32)
            for slot in dynamic_slots
            if getattr(self, slot) is not None
        )

        # Static parameters go in aux data
        static_data = tuple(getattr(self, slot) for slot in static_slots)

        return arrays, static_data

    @classmethod
    def tree_unflatten(cls, static_data: tuple[Any, ...], arrays: tuple[Array, ...]):
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)


@dataclass(frozen=True, slots=True)
class uptake_BV_output_features:
    """Concrete implementation of Output_Features for uptake BV output features."""

    uptake: list[list[float]] | Sequence[Sequence[float]] | Array  # (1, residues, timepoints)

    __features__: ClassVar[set[str]] = {"uptake"}
    key: ClassVar[m_key] = m_key("HDX_peptide")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.uptake), len(self.uptake[0]))

    def y_pred(self) -> Array:
        return jnp.asarray(self.uptake)

    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.
        Dynamic slots are all slots in cls.__features__.
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
    def tree_unflatten(cls, static_data: tuple[Any, ...], arrays: tuple[Array, ...]):
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)


# Register classes as PyTree nodes
register_pytree_node(
    BV_input_features, BV_input_features.tree_flatten, BV_input_features.tree_unflatten
)
register_pytree_node(
    BV_output_features, BV_output_features.tree_flatten, BV_output_features.tree_unflatten
)
register_pytree_node(
    uptake_BV_output_features,
    uptake_BV_output_features.tree_flatten,
    uptake_BV_output_features.tree_unflatten,
)
