import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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


@dataclass
class OptimiserSettings:
    name: str
    n_steps: int = 100000
    tolerance: float = 1e-1
    convergence: float = 1e-4
    learning_rate: float = 0.01  # not implemented


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


class PathManager:
    def __init__(
        self,
        settings: Settings,
        overwrite: bool = False,
        exist_ok: bool = False,
        experimental_data_path: Optional[Path] = None,
        simulation_data_path: Optional[Path] = None,
        base_dir: Path = Path("./data"),
        logs_dir: Optional[Path] = None,  # Add logs_dir parameter
        preserve_logs: bool = True,  # Add preserve_logs flag
    ):
        self.settings = settings
        self.overwrite = overwrite
        self.exist_ok = exist_ok
        self.base_dir = Path(base_dir)
        self._experimental_data_path = experimental_data_path
        self._simulation_data_path = simulation_data_path
        self._logs_dir = logs_dir  # Store custom logs directory
        self.preserve_logs = preserve_logs  # Store preserve_logs flag

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize experiment directory
        self.experiment_dir = self._initialize_experiment_directory()

    @property
    def logs_dir(self) -> Path:
        """Path to logs directory."""
        # Return custom logs directory if specified, otherwise use default
        return self._logs_dir if self._logs_dir is not None else self.experiment_dir / "logs"

    @logs_dir.setter
    def logs_dir(self, path: Path):
        """Set logs directory path."""
        self._logs_dir = Path(path)

    def clean_all_directories(self):
        """Delete all created directories except logs if preserve_logs is True."""
        if self.experiment_dir.exists():
            if self.preserve_logs:
                # Delete everything except logs directory
                for path in self.experiment_dir.iterdir():
                    if path != self.logs_dir:
                        if path.is_file():
                            path.unlink()
                        else:
                            shutil.rmtree(path)
            else:
                shutil.rmtree(self.experiment_dir)

    def _initialize_experiment_directory(self) -> Path:
        """Initialize the main experiment directory with collision handling."""
        base_path = (
            self.base_dir
            / self.settings.experiment_type
            / self.settings.protein_name
            / self.settings.condition
            / self.settings.experiment_name
        )

        if base_path.exists():
            if self.overwrite:
                shutil.rmtree(base_path)
            elif not self.exist_ok:
                base_path = self._handle_name_collision(base_path)

        return base_path

    def _handle_name_collision(self, path: Path) -> Path:
        """Handle naming collisions by adding numerical suffixes."""
        counter = 1
        while True:
            new_path = path.parent / f"{path.name}_{counter}"
            if not new_path.exists():
                self.logger.info(f"Changed experiment path to: {new_path}")
                return new_path
            counter += 1

    @property
    def checkpoints_dir(self) -> Path:
        """Path to model checkpoints directory."""
        return self.experiment_dir / "checkpoints"

    @property
    def optimization_dir(self) -> Path:
        """Path to optimization results directory."""
        return self.experiment_dir / "optimization"

    @property
    def analysis_dir(self) -> Path:
        """Path to analysis outputs directory."""
        return self.experiment_dir / "analysis"

    @property
    def experimental_data_path(self) -> Optional[Path]:
        """Path to experimental data."""
        return self._experimental_data_path

    @experimental_data_path.setter
    def experimental_data_path(self, path: Path):
        """Set experimental data path."""
        self._experimental_data_path = Path(path)
        self.validate_external_path(self._experimental_data_path)

    @property
    def simulation_data_path(self) -> Optional[Path]:
        """Path to simulation data."""
        return self._simulation_data_path

    @simulation_data_path.setter
    def simulation_data_path(self, path: Path):
        """Set simulation data path."""
        self._simulation_data_path = Path(path)
        self.validate_external_path(self._simulation_data_path)

    def create_checkpoints_dir(self):
        """Create checkpoints directory."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=self.exist_ok)

    def create_optimization_dir(self):
        """Create optimization results directory."""
        self.optimization_dir.mkdir(parents=True, exist_ok=self.exist_ok)

    def create_logs_dir(self):
        """Create logs directory."""
        self.logs_dir.mkdir(parents=True, exist_ok=self.exist_ok)

    def create_analysis_dir(self):
        """Create analysis outputs directory."""
        self.analysis_dir.mkdir(parents=True, exist_ok=self.exist_ok)

    def create_all_directories(self):
        """Create all required directories."""
        self.create_checkpoints_dir()
        self.create_optimization_dir()
        self.create_logs_dir()
        self.create_analysis_dir()

    def delete_checkpoints_dir(self):
        """Delete checkpoints directory."""
        if self.checkpoints_dir.exists():
            shutil.rmtree(self.checkpoints_dir)

    def delete_optimization_dir(self):
        """Delete optimization results directory."""
        if self.optimization_dir.exists():
            shutil.rmtree(self.optimization_dir)

    def delete_logs_dir(self):
        """Delete logs directory."""
        if self.logs_dir.exists():
            shutil.rmtree(self.logs_dir)

    def delete_analysis_dir(self):
        """Delete analysis outputs directory."""
        if self.analysis_dir.exists():
            shutil.rmtree(self.analysis_dir)

    def validate_external_path(self, path: Path):
        """Validate that an external path exists."""
        if not path.exists():
            raise FileNotFoundError(f"External path does not exist: {path}")

    def validate_all_paths(self) -> bool:
        """Validate that all required directories exist."""
        required_dirs = [
            self.checkpoints_dir,
            self.optimization_dir,
            self.logs_dir,
            self.analysis_dir,
        ]

        all_exist = all(path.exists() for path in required_dirs)

        if self._experimental_data_path:
            all_exist &= self._experimental_data_path.exists()
        if self._simulation_data_path:
            all_exist &= self._simulation_data_path.exists()

        return all_exist

    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """Get path for a specific checkpoint."""
        return self.checkpoints_dir / checkpoint_name

    def get_optimization_result_path(self, result_name: str) -> Path:
        """Get path for a specific optimization result."""
        return self.optimization_dir / result_name

    def get_analysis_output_path(self, output_name: str) -> Path:
        """Get path for a specific analysis output."""
        return self.analysis_dir / output_name
