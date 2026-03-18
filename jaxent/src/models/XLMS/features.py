from dataclasses import dataclass
from collections.abc import Sequence
from beartype.typing import ClassVar
import jax.numpy as jnp
from numpy import ndarray
from jax import Array
from jax.tree_util import register_pytree_node
from jaxtyping import Float
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key


@dataclass(slots=True, frozen=True)
class XLMS_input_features(Input_Features):
    """Pairwise residue distances per frame, shape (n_residues, n_residues, n_frames).
    Frames are the last axis for frame_average_features compatibility."""
    distances: Float[Array, "n_residues n_residues n_frames"] | Float[ndarray, "n_residues n_residues n_frames"]
    __features__: ClassVar[set[str]] = {"distances"}
    key: ClassVar[set[m_key]] = {m_key("XLMS_distance")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.distances).shape

    @property
    def feat_pred(self) -> Sequence[Array]:
        return (jnp.asarray(self.distances),)


@dataclass(frozen=True, slots=True)
class XLMS_output_features(Output_Features):
    """Ensemble-averaged pairwise distance matrix, shape (n_residues, n_residues).
    PairIndexMapping extracts specific observed cross-link pairs."""
    distances: Float[Array, "n_residues n_residues"]
    __features__: ClassVar[set[str]] = {"distances"}
    key: ClassVar[m_key] = m_key("XLMS_distance")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.distances).shape

    def y_pred(self) -> Array:
        return jnp.asarray(self.distances)


register_pytree_node(
    XLMS_input_features, XLMS_input_features.tree_flatten, XLMS_input_features.tree_unflatten
)
register_pytree_node(
    XLMS_output_features, XLMS_output_features.tree_flatten, XLMS_output_features.tree_unflatten
)
