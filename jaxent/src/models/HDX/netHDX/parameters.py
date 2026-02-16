from dataclasses import dataclass, field
from beartype.typing import ClassVar

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Model_Parameters


@dataclass(frozen=True, slots=True)
class NetHDX_Model_Parameters(Model_Parameters):
    shell_energy_scaling: float = 0.84  # Energy scaling factor for each shell contact (-0.5 kcal/mol per shell (-2.1 kj/mol)), using R=8.31/1000 and T=300K
    temperature: float = 300.0
    timepoints: Float[Array, " n_timepoints"] = field(default_factory=lambda: jnp.array([0.167, 1.0, 10.0]))
    static_params: ClassVar[set[str]] = {"temperature", "key", "timepoints"}
    key = frozenset({m_key("HDX_resPF"), m_key("HDX_peptide")})
