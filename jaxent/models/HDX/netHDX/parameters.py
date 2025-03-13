from dataclasses import dataclass, field
from typing import ClassVar

import jax.numpy as jnp
from jax import Array

from jaxent.interfaces.simulation import Model_Parameters
from jaxent.types.base import m_key


@dataclass(frozen=True, slots=True)
class NetHDX_Model_Parameters(Model_Parameters):
    shell_energy_scaling: float = 0.84  # Energy scaling factor for each shell contact (-0.5 kcal/mol per shell (-2.1 kj/mol)), using R=8.31/1000 and T=300K
    temperature: float = 300
    timepoints: Array = field(default_factory=lambda: jnp.array([0.167, 1.0, 10.0]))
    static_params: ClassVar[set[str]] = {"temperature", "key", "timepoints"}
    key = frozenset({m_key("HDX_resPF"), m_key("HDX_peptide")})
