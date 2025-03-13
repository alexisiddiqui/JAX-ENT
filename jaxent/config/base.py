from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol

from jaxent.interfaces.model import Model_Parameters
from jaxent.types.base import m_key


class BaseConfig:
    name: str

    def from_json(self, json_path: Path):
        pass

    def to_json(self, json_path: Path):
        pass


@dataclass
class FeaturiserSettings:
    name: str
    batch_size: int | None


class LossConstants(NamedTuple):
    GAMMA: float
    LAMBDA: float
    PHI: float
    PSI: float


@dataclass
class OptimiserSettings:
    name: str
    n_steps: int = 1000
    tolerance: float = 1e-1
    convergence: float = 1e-4
    learning_rate: float = 1e-4
    optimiser_type: str = "adam"
    loss_constants: LossConstants = LossConstants(GAMMA=0.1, LAMBDA=0.1, PHI=0.1, PSI=0.1)


class Optimisable_Parameters(Enum):
    frame_weights = 0
    model_parameters = 1
    forward_model_weights = 2
    frame_mask = 3
    forward_model_scaling = 4


@dataclass
class Settings(BaseConfig):
    protein_name: str
    condition: str
    experiment_name: str
    experiment_type: str

    optimiser_config: OptimiserSettings
    featuriser_config: FeaturiserSettings
    forward_model_config: BaseConfig

    n_replicates: int = 3

    n_workers: int = 4
    # set name for child classes?


####################################################################################################
class Model_Config(Protocol):
    key: m_key

    @property
    def forward_parameters(self) -> Model_Parameters: ...
