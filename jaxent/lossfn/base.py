# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from typing import Protocol

from jaxent.datatypes import Experimental_Dataset, Simulation
from jaxent.forwardmodels.base import Model_Parameters, Output_Features


class JaxEnt_Loss(Protocol):
    def __call__(
        self,
        model: Simulation | list[Simulation],
        dataset: Output_Features | Experimental_Dataset | Model_Parameters,
    ) -> float: ...
