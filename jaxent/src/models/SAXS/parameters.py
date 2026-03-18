from dataclasses import dataclass, field
from beartype.typing import ClassVar
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node
from jaxtyping import Float
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Model_Parameters


@dataclass(frozen=True, slots=True)
class SAXS_direct_Model_Parameters(Model_Parameters):
    """No optimizable model parameters — only frame weights are optimized."""
    key = frozenset({m_key("SAXS_Iq")})
    static_params: ClassVar[set[str]] = {"key"}


# Backwards-compat alias — remove once all call sites are updated
SAXS_Reweighted_Parameters = SAXS_direct_Model_Parameters


@dataclass(frozen=True, slots=True)
class SAXS_Debye_Parameters(Model_Parameters):
    """Debye 6-term cross-term parameters: c1, c2, c, b."""
    c1: Float[Array, ""] = field(default_factory=lambda: jnp.array(1.0))
    c2: Float[Array, ""] = field(default_factory=lambda: jnp.array(0.0))
    c: Float[Array, ""] = field(default_factory=lambda: jnp.array(1.0))
    b: Float[Array, ""] = field(default_factory=lambda: jnp.array(0.0))
    key = frozenset({m_key("SAXS_Iq")})
    static_params: ClassVar[set[str]] = {"key"}


register_pytree_node(
    SAXS_direct_Model_Parameters,
    SAXS_direct_Model_Parameters.tree_flatten,
    SAXS_direct_Model_Parameters.tree_unflatten,
)
register_pytree_node(
    SAXS_Debye_Parameters,
    SAXS_Debye_Parameters.tree_flatten,
    SAXS_Debye_Parameters.tree_unflatten,
)
