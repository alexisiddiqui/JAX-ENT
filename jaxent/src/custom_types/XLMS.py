from dataclasses import dataclass
from beartype.typing import ClassVar
import numpy as np

from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology


@dataclass()
class XLMS_distance_restraint(ExpD_Datapoint):
    """XL-MS cross-link distance restraint between two residues.

    Each datapoint represents one observed cross-link with a measured
    or maximum distance. top (inherited) is the first residue, top_j
    is the second.
    """
    key: ClassVar[m_key] = m_key("XLMS_distance")
    top_j: Partial_Topology | None = None
    distance: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None

    def extract_features(self) -> np.ndarray:
        """Return distance as a single-element feature vector."""
        return np.array([self.distance])

    @classmethod
    def _create_from_features(
        cls, topology: Partial_Topology, features: np.ndarray
    ) -> "XLMS_distance_restraint":
        """Create from features array (distance only).

        top_j is not round-tripped through the standard feature path -
        it must be set from supplementary topology data.
        """
        if features.shape != (1,):
            raise ValueError(
                f"XLMS_distance_restraint expects a single feature "
                f"with shape (1,), got {features.shape}"
            )
        return cls(top=topology, distance=float(features[0]))
