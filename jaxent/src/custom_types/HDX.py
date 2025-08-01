from dataclasses import dataclass

import numpy as np

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology


@dataclass()
class HDX_peptide(ExpD_Datapoint):
    """
    Dataclass that holds the information of a single peptide produced by HDX-MS experiments
    """

    dfrac: list[float]
    charge: int | None = None
    retention_time: float | None = None
    intensity: float | None = None
    key = m_key("HDX_peptide")

    def extract_features(self) -> np.ndarray:
        # reshape to be 1, n_timepoints
        return np.array(self.dfrac).reshape(-1, 1)

    @classmethod
    def _create_from_features(
        cls, topology: Partial_Topology, features: np.ndarray
    ) -> "HDX_peptide":
        """Custom creation method for HDX_peptide"""
        return cls(top=topology, dfrac=[float(x) for x in features.flatten()])


@dataclass()
class HDX_protection_factor(ExpD_Datapoint):
    """
    Dataclass that holds the information of a single protection factor produced by REX experiments
    or NMR derived protection factors. May also be used to describe protection factors output from forward models.
    """

    protection_factor: float
    key = m_key("HDX_resPf")

    def extract_features(self) -> np.ndarray:
        return np.array([self.protection_factor])

    @classmethod
    def _create_from_features(
        cls, topology: Partial_Topology, features: np.ndarray
    ) -> "HDX_protection_factor":
        """Custom creation method for HDX_protection_factor"""
        if features.shape != (1,):
            raise ValueError(f"HDX_protection_factor expects a single feature with shape (1,), got {features.shape}")
        return cls(top=topology, protection_factor=float(features[0]))
