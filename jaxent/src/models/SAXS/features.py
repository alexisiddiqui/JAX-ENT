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
class SAXS_curve_input_features(Input_Features):
    """Pre-computed I(q) curves per structure, shape (n_q, n_frames).
    Frames are the last axis for compatibility with frame_average_features."""
    intensities: Float[Array, "n_q n_frames"] | Float[ndarray, "n_q n_frames"]
    __features__: ClassVar[set[str]] = {"intensities"}
    key: ClassVar[set[m_key]] = {m_key("SAXS_Iq")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.intensities).shape

    @property
    def feat_pred(self) -> Sequence[Array]:
        return (jnp.asarray(self.intensities),)


@dataclass(slots=True, frozen=True)
class SAXS_basis_input_features(Input_Features):
    """Six Debye basis profiles per structure, shape (6, n_q, n_frames).
    Frames are the last axis. After frame averaging: (6, n_q)."""
    basis_profiles: Float[Array, "6 n_q n_frames"] | Float[ndarray, "6 n_q n_frames"]
    __features__: ClassVar[set[str]] = {"basis_profiles"}
    key: ClassVar[set[m_key]] = {m_key("SAXS_Iq")}

    @property
    def features_shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.basis_profiles).shape

    @property
    def feat_pred(self) -> Sequence[Array]:
        return (jnp.asarray(self.basis_profiles),)


@dataclass(frozen=True, slots=True)
class SAXS_output_features(Output_Features):
    """Ensemble-averaged I(q) curve, shape (n_q,)."""
    intensity: Float[Array, " n_q"]
    __features__: ClassVar[set[str]] = {"intensity"}
    key: ClassVar[m_key] = m_key("SAXS_Iq")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return jnp.asarray(self.intensity).shape

    def y_pred(self) -> Array:
        return jnp.asarray(self.intensity)


register_pytree_node(
    SAXS_curve_input_features,
    SAXS_curve_input_features.tree_flatten,
    SAXS_curve_input_features.tree_unflatten,
)
register_pytree_node(
    SAXS_basis_input_features,
    SAXS_basis_input_features.tree_flatten,
    SAXS_basis_input_features.tree_unflatten,
)
register_pytree_node(
    SAXS_output_features,
    SAXS_output_features.tree_flatten,
    SAXS_output_features.tree_unflatten,
)
