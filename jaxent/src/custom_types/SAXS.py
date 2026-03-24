from dataclasses import dataclass, field
from beartype.typing import ClassVar
import numpy as np
from numpy import ndarray

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology


@dataclass()
class SAXS_curve(ExpD_Datapoint):
    """SAXS I(q) scattering profile for a whole construct.

    Stores a single SAXS I(q) curve (intensities at given q-values)
    as one datapoint. Topology covers the entire construct.
    """
    key: ClassVar[m_key] = m_key("SAXS_Iq")
    intensities: ndarray = field(default_factory=lambda: np.array([]))
    q_values: ndarray = field(default_factory=lambda: np.array([]))
    errors: ndarray | None = None

    @classmethod
    def is_whole_system(cls) -> bool:
        """SAXS covers the entire construct — splitting is over q-points, not fragments."""
        return True

    def extract_features(self) -> np.ndarray:
        """Return intensities as a 1D feature vector."""
        return np.asarray(self.intensities).flatten()

    @classmethod
    def _create_from_features(
        cls, topology: Partial_Topology, features: np.ndarray
    ) -> "SAXS_curve":
        """Create from a flat features array (intensities only).

        q_values and errors are not round-tripped through feature
        serialisation - they must be set separately after loading.
        """
        return cls(top=topology, intensities=features.flatten())
