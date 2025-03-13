from dataclasses import dataclass

import numpy as np

from jaxent.data.loading import Experimental_Fragment
from jaxent.types.base import m_key


@dataclass()
class HDX_peptide(Experimental_Fragment):
    """
    dataclass that holds the information of a single peptide produced by HDX-MS experiments
    """

    dfrac: list[float]
    charge: int | None = None
    retention_time: float | None = None
    intensity: float | None = None
    key = m_key("HDX_peptide")

    def extract_features(self) -> np.ndarray:
        # reshape to be 1, n_timepoints
        return np.array(self.dfrac).reshape(-1, 1)


@dataclass()
class HDX_protection_factor(Experimental_Fragment):
    """
    Dataclass that holds the information of a single protection factor produced by REX experiments
    or NMR derived protection factors. May also be used to describe protection factors output from forward models.
    """

    protection_factor: float
    key = m_key("HDX_resPf")

    def extract_features(self) -> np.ndarray:
        return np.array([self.protection_factor])
