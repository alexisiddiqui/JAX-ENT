from dataclasses import dataclass
from collections.abc import Sequence
from beartype.typing import ClassVar, Optional

import jax.numpy as jnp
from numpy import ndarray
from jax import Array
from jax.tree_util import register_pytree_node
from jaxtyping import Float

from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key


@dataclass(slots=True, frozen=True)
class BV_input_features(Input_Features):
    """
    Concrete implementation of Input_Features for BV input features.
    """

    # Shape can be 2D (n_residues, n_frames) before averaging, or 1D (n_residues,) after
    # Also accepts numpy arrays when loading from .npz files
    heavy_contacts: Float[Array, "n_residues n_frames"] | Float[Array, " n_residues"] | Float[ndarray, "n_residues n_frames"] | Float[ndarray, " n_residues"]
    acceptor_contacts: Float[Array, "n_residues n_frames"] | Float[Array, " n_residues"] | Float[ndarray, "n_residues n_frames"] | Float[ndarray, " n_residues"]
    k_ints: Float[Array, " n_residues"] | Float[ndarray, " n_residues"] | None = None

    __features__: ClassVar[set[str]] = {"heavy_contacts", "acceptor_contacts", "k_ints"}
    key: ClassVar[set[m_key]] = {m_key("HDX_resPF"), m_key("HDX_peptide")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        if type(self.heavy_contacts) is not type(self.acceptor_contacts):
            raise TypeError("heavy_contacts and acceptor_contacts must be of the same type")

        if isinstance(self.heavy_contacts, (Array, ndarray)):
            # Handle both 1D (after averaging) and 2D (before averaging) arrays
            if self.heavy_contacts.ndim == 1:
                n_residues = self.heavy_contacts.shape[0]
                n_frames = 1  # Already averaged
            else:
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

    # log_Pf can be 1-D (n_residues,) from forward or 2-D (n_residues, n_frames) from predict
    log_Pf: Float[Array, " n_residues"] | Float[Array, "n_residues n_frames"]
    # k_ints can be 1-D array, 0-D scalar (NaN placeholder during JIT), or None
    k_ints: Float[Array, " n_residues"] | Float[Array, ""] | None = None

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

    # uptake can be 2-D (n_timepoints, n_residues) or 3-D (batch, n_timepoints, n_residues) when stacked
    uptake: Float[Array, "n_timepoints n_residues"] | Float[Array, "batch n_timepoints n_residues"]

    __features__: ClassVar[set[str]] = {"uptake"}
    key: ClassVar[m_key] = m_key("HDX_peptide")

    @property
    def output_shape(self) -> tuple[int, ...]:
        if isinstance(self.uptake, (Array, ndarray)):
            return self.uptake.shape
        else:
            uptake = jnp.asarray(self.uptake)
            return uptake.shape

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
