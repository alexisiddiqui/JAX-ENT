from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from beartype.typing import NamedTuple, Protocol, runtime_checkable, TYPE_CHECKING
from typing import Literal
from jaxent.src.custom_types.key import m_key

if TYPE_CHECKING:
    from jaxent.src.interfaces.model import Model_Parameters


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


class LossConstants(NamedTuple):  # TODO we need to change these to have more meaningful names
    GAMMA: float
    LAMBDA: float
    PHI: float
    PSI: float


class Optimisable_Parameters(Enum):
    frame_weights = 0
    model_parameters = 1
    forward_model_weights = 2


@dataclass
class OptimiserSettings:
    """Configuration for optimisation and history retention.

    ``parameter_partitions=None`` retains all selectable parameter partitions.
    ``forward_model_scaling`` and ``normalise_loss_functions`` are always retained;
    they are small scalar fields and have no ``Optimisable_Parameters`` enum member.
    """
    name: str
    n_steps: int = 100
    tolerance: float = 1e-2
    convergence: float | list[float] = 1e-5
    learning_rate: float = 1e-4
    optimiser_type: str = "adam"
    loss_constants: LossConstants = LossConstants(GAMMA=0.1, LAMBDA=0.1, PHI=0.1, PSI=0.1)
    ema_alpha: float = 0.5
    min_steps_per_threshold: int = 2
    step_chunk_size: int = 100
    execution_mode: Literal["compiled", "python"] = "compiled"
    parameter_partitions: frozenset[Optimisable_Parameters] | None = None
    save_states: bool = True
    save_convergence: bool = True
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.step_chunk_size < 1:
            raise ValueError("step_chunk_size must be >= 1")
        if self.execution_mode not in ("compiled", "python"):
            raise ValueError("execution_mode must be 'compiled' or 'python'")
        if not (self.save_states or self.save_convergence or self.save_best):
            raise ValueError("At least one of save_states, save_convergence, or save_best must be true")
        if self.parameter_partitions is not None:
            try:
                partitions = frozenset(self.parameter_partitions)
            except TypeError as exc:
                raise ValueError("parameter_partitions must be a collection of Optimisable_Parameters") from exc
            object.__setattr__(self, "parameter_partitions", partitions)
            if not partitions:
                raise ValueError("parameter_partitions must not be empty")
            if not all(isinstance(partition, Optimisable_Parameters) for partition in partitions):
                raise ValueError("parameter_partitions must contain only Optimisable_Parameters members")


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
    peptide_trim: int = 2

    n_workers: int = 4
    # set name for child classes?


####################################################################################################


@runtime_checkable
class Model_Config(Protocol):
    key: m_key

    @property
    def forward_parameters(self) -> "Model_Parameters": ...
