from dataclasses import dataclass
from beartype.typing import ClassVar
from jax.tree_util import register_pytree_node
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Model_Parameters


@dataclass(frozen=True, slots=True)
class XLMS_Model_Parameters(Model_Parameters):
    """No optimizable model parameters — only frame weights are optimized."""
    key = frozenset({m_key("XLMS_distance")})
    static_params: ClassVar[set[str]] = {"key"}


register_pytree_node(
    XLMS_Model_Parameters, XLMS_Model_Parameters.tree_flatten, XLMS_Model_Parameters.tree_unflatten
)
