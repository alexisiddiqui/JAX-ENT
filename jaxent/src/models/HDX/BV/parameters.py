from dataclasses import dataclass, field
from typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Model_Parameters


@dataclass(frozen=True, slots=True)
class BV_Model_Parameters(Model_Parameters):
    bv_bc: Array = field(default_factory=lambda: jnp.array([0.35]))
    bv_bh: Array = field(default_factory=lambda: jnp.array([2.0]))
    key = frozenset({m_key("HDX_resPF"), m_key("HDX_peptide")})
    temperature: float = 300
    timepoints: Array = field(default_factory=lambda: jnp.array([0.167, 1.0, 10.0]))
    static_params: ClassVar[set[str]] = {"temperature", "key", "timepoints"}

    def __mul__(self, scalar: float | Array) -> "BV_Model_Parameters":
        scalar = jnp.asarray(scalar)

        return BV_Model_Parameters(
            bv_bc=self.bv_bc * scalar,
            bv_bh=self.bv_bh * scalar,
            timepoints=self.timepoints,
            temperature=self.temperature,
        )

    __rmul__ = __mul__

    def __sub__(self, other: "BV_Model_Parameters") -> "BV_Model_Parameters":
        return BV_Model_Parameters(
            bv_bc=self.bv_bc - other.bv_bc,
            bv_bh=self.bv_bh - other.bv_bh,
            timepoints=self.timepoints,
            temperature=self.temperature,
        )

    def update_parameters(self, new_params: "BV_Model_Parameters") -> "BV_Model_Parameters":
        """
        Creates a new instance with updated parameters, preserving static parameters.

        Args:
            new_params: Tuple of new parameter values in the order bv_bc, bv_bh
        Returns:
            A new BV_Model_Parameters instance with updated non-static parameters
        """

        return BV_Model_Parameters(
            bv_bc=new_params.bv_bc,
            bv_bh=new_params.bv_bh,
            temperature=self.temperature,
            timepoints=self.timepoints,
        )


register_pytree_node(
    BV_Model_Parameters, BV_Model_Parameters.tree_flatten, BV_Model_Parameters.tree_unflatten
)


@dataclass(frozen=True, slots=True)
class linear_BV_Model_Parameters(Model_Parameters):
    bv_bc: Array = field(default_factory=lambda: jnp.array([0.35]))
    bv_bh: Array = field(default_factory=lambda: jnp.array([2.0]))
    key = frozenset({m_key("HDX_resPF"), m_key("HDX_peptide")})
    temperature: float = 300
    timepoints: Array = field(default_factory=lambda: jnp.array([0.167, 1.0, 10.0]))
    static_params: ClassVar[set[str]] = {"temperature", "timepoints", "key"}

    def __mul__(self, scalar: float | Array) -> "linear_BV_Model_Parameters":
        return linear_BV_Model_Parameters(
            bv_bc=self.bv_bc * jnp.asarray(scalar),
            bv_bh=self.bv_bh * scalar,
            temperature=self.temperature,
            num_timepoints=self.num_timepoints,
        )

    __rmul__ = __mul__

    def __sub__(self, other: "linear_BV_Model_Parameters") -> "linear_BV_Model_Parameters":
        return linear_BV_Model_Parameters(
            bv_bc=self.bv_bc - other.bv_bc,
            bv_bh=self.bv_bh - other.bv_bh,
            temperature=self.temperature,
            num_timepoints=self.num_timepoints,
        )

    def update_parameters(
        self, new_params: "linear_BV_Model_Parameters"
    ) -> "linear_BV_Model_Parameters":
        return linear_BV_Model_Parameters(
            bv_bc=new_params.bv_bc,
            bv_bh=new_params.bv_bh,
            temperature=self.temperature,
            num_timepoints=self.num_timepoints,
        )

    # def tree_flatten(self):
    #     # Override the base class tree_flatten to handle array parameters
    #     dynamic_fields = []
    #     static_fields = []

    #     for slot in self._get_ordered_slots():
    #         value = getattr(self, slot)
    #         if slot in self.static_params:
    #             static_fields.append(value)
    #         else:
    #             dynamic_fields.append(value)

    #     return (tuple(dynamic_fields), tuple(static_fields))

    # @classmethod
    # def tree_unflatten(cls, static, arrays):
    #     # Reconstruct the parameters from flattened data
    #     all_values = list(arrays) + list(static)
    #     slots = cls._get_ordered_slots()

    #     return cls(**dict(zip(slots, all_values)))


register_pytree_node(
    linear_BV_Model_Parameters,
    linear_BV_Model_Parameters.tree_flatten,
    linear_BV_Model_Parameters.tree_unflatten,
)
