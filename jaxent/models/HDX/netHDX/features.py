from dataclasses import dataclass
from typing import ClassVar, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxent.types.key import m_key


@dataclass(frozen=True)
class NetworkMetrics:
    """Per-residue network metrics for a single frame"""

    degrees: Mapping[int, float]
    clustering_coeffs: Mapping[int, float]
    betweenness: Mapping[int, float]
    kcore_numbers: Mapping[int, float]
    min_path_lengths: Mapping[int, float]
    mean_path_lengths: Mapping[int, float]
    max_path_lengths: Mapping[int, float]

    def cast_to_jax(self) -> None:
        # TODO check that this is actually setting the values to jax arrays - do we need to convert the keys to jax arrays?
        for attr in self.__annotations__:
            setattr(self, attr, jnp.asarray(getattr(self, attr)))


@dataclass(frozen=True)
class NetHDX_input_features:
    """Features representing the hydrogen bond network for each frame"""

    contact_matrices: list[np.ndarray]  # Shape: (n_frames, n_residues, n_residues)
    residue_ids: Sequence[int]  # Shape: (n_residues,)
    network_metrics: Optional[list[NetworkMetrics]] = None  # Shape: (n_frames,)
    __features__: ClassVar[set[str]] = {"contact_matrices"}
    key: ClassVar[set[m_key]] = {m_key("HDX_resPF"), m_key("HDX_peptide")}

    @property
    def features_shape(self) -> Tuple[int, ...]:
        return (len(self.contact_matrices), len(self.residue_ids), len(self.residue_ids))

    def cast_to_jax(self) -> None:
        setattr(self, "contact_matrices", jnp.asarray(self.contact_matrices))
        if self.network_metrics is not None:
            for nm in self.network_metrics:
                nm.cast_to_jax()


@dataclass(frozen=True)
class NetHDX_output_features:
    """Output features for netHDX model"""

    log_Pf: list  # (1, residues)
    k_ints: Optional[list]

    key = m_key("HDX_resPF")

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (1, len(self.log_Pf))

    def data(self) -> Array:
        return jnp.asarray(self.log_Pf)


@dataclass(frozen=True)
class uptake_NetHDX_output_features:
    """Output features for netHDX model"""

    uptake: list[list[float]] | Sequence[Sequence[float]] | Array  # (1, residues, timepoints)]
    k_ints: Optional[list]

    key = m_key("HDX_peptide")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.uptake), len(self.uptake[0]))

    def data(self) -> Array:
        return jnp.asarray(self.uptake)
