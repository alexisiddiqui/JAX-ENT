from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import jax.numpy as jnp
from jax import Array

from jaxent.types.base import m_key


@dataclass(frozen=True, slots=True)
class BV_input_features:
    heavy_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    acceptor_contacts: Sequence[Sequence[float]] | Array  # (frames, residues)
    k_ints: Optional[list] | Optional[Array]

    __features__: ClassVar[set[str]] = {"heavy_contacts", "acceptor_contacts"}
    key: ClassVar[set[m_key]] = {m_key("HDX_resPF"), m_key("HDX_peptide")}

    ########################################################################
    # update the features shape to have a fixed/more consistent structure
    @property
    def features_shape(self) -> tuple[int, ...]:
        length = len(self.heavy_contacts[0])
        heavy_shape = len(self.heavy_contacts)
        acceptor_shape = len(self.acceptor_contacts)
        return (heavy_shape, acceptor_shape, length)

    def cast_to_jax(self) -> None:
        setattr(self, "heavy_contacts", jnp.asarray(self.heavy_contacts))
        setattr(self, "acceptor_contacts", jnp.asarray(self.acceptor_contacts))


########################################################################
# fix the typing to use numpy arrays
@dataclass(frozen=True)
class BV_output_features:
    __slots__ = ["log_Pf", "k_ints"]

    log_Pf: list | Sequence[float] | Array  # (1, residues)]
    k_ints: Optional[list] | Optional[Array]
    key: ClassVar[m_key] = m_key("HDX_resPF")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.log_Pf))

    def data(self) -> Array:
        return jnp.asarray(self.log_Pf)


@dataclass(frozen=True, slots=True)
class uptake_BV_output_features:
    uptake: list[list[float]] | Sequence[Sequence[float]] | Array  # (1, residues, timepoints)]
    key: ClassVar[m_key] = m_key("HDX_peptide")

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, len(self.uptake), len(self.uptake[0]))

    def data(self) -> Array:
        return jnp.asarray(self.uptake)
