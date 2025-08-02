from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key


@dataclass(slots=True, frozen=True)
class BV_input_features(Input_Features):
    """
    Concrete implementation of Input_Features for BV input features.
    """

    heavy_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    acceptor_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    k_ints: Optional[list] | Optional[Array] = None  # (residues,)

    __features__: ClassVar[set[str]] = {"heavy_contacts", "acceptor_contacts"}
    key: ClassVar[set[m_key]] = {m_key("HDX_resPF"), m_key("HDX_peptide")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        if type(self.heavy_contacts) != type(self.acceptor_contacts):
            raise TypeError("heavy_contacts and acceptor_contacts must be of the same type")

        if isinstance(self.heavy_contacts, Array):
            # For JAX arrays: (residues, frames)
            n_residues, n_frames = self.heavy_contacts.shape
        else:
            # For nested sequences: (residues, frames)
            n_residues = len(self.heavy_contacts)
            n_frames = len(self.heavy_contacts[0]) if n_residues > 0 else 0
        return (n_residues, n_frames)

    @property
    def feat_pred(self) -> Sequence[Array]:
        return (jnp.asarray(self.heavy_contacts), jnp.asarray(self.acceptor_contacts))


@dataclass(frozen=True, slots=True)
class BV_output_features(Output_Features):
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


@dataclass(frozen=True, slots=True)
class uptake_BV_output_features(Output_Features):
    """Concrete implementation of Output_Features for uptake BV output features."""

    uptake: list[list[float]] | Sequence[Sequence[float]] | Array  # (1, residues, timepoints)

    __features__: ClassVar[set[str]] = {"uptake"}
    key: ClassVar[m_key] = m_key("HDX_peptide")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.uptake[0]), len(self.uptake[0][0]))

    def y_pred(self) -> Array:
        return jnp.asarray(self.uptake)


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
