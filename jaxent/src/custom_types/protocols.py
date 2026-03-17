from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable
from jax import Array
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.interfaces.simulation import Simulation_Parameters

@runtime_checkable
class SimulationLike(Protocol):
    """Protocol for objects that can be used in loss functions."""
    params: Simulation_Parameters
    outputs: Sequence[Output_Features]

@runtime_checkable
class InputFeaturesLike(Protocol):
    """Protocol for input features."""
    features_shape: tuple[int, ...]
    __features__: Sequence[str]
    def cast_to_jax(self) -> "InputFeaturesLike": ...

@runtime_checkable
class ModelParametersLike(Protocol):
    """Protocol for model parameters."""
    pass # Model parameters are often just PyTrees, so a simple Protocol is fine for now

@runtime_checkable  
class DatasetLike(Protocol):
    """Protocol for dataset objects with required attributes."""
    y_true: Array
    data_mapping: Any  # DataMapping protocol

@runtime_checkable
class DataloaderLike(Protocol):
    """Protocol for objects providing train/val datasets."""
    train: DatasetLike
    val: DatasetLike

@runtime_checkable
class ExpDDatapointLike(Protocol):
    """Protocol for experimental data point objects."""
    top: Any  # Partial_Topology
    key: Any  # m_key or ClassVar
    
    def extract_features(self) -> Any: ...
