from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology


@dataclass()
class ExpD_Datapoint:
    """
    Base class for experimental data - grouped into subdomain fragments
    Limtation is that it only covers a single chain - which should be fine in most cases.
    """

    top: Partial_Topology
    key: ClassVar[m_key]

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the child class.")
