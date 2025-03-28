import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import importlib
from ast import literal_eval
from typing import Optional, Type, TypeVar

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from jaxent.interfaces.model import Model_Parameters
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.opt.base import LossComponents, OptimizationHistory, OptimizationState

T_mp = TypeVar("T_mp", bound=Model_Parameters)


def save_array_to_hdf5(h5file, path: str, array, **kwargs) -> None:
    """
    Save a JAX array to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        array: The JAX array to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    # Convert JAX array to numpy for HDF5 compatibility
    np_array = np.asarray(array)

    # Skip compression for scalar datasets because they don't support chunking
    if np_array.shape == () or np_array.size == 1:
        # Create a copy of kwargs without compression options
        scalar_kwargs = kwargs.copy()
        for key in ["compression", "compression_opts", "chunks"]:
            scalar_kwargs.pop(key, None)
        h5file.create_dataset(path, data=np_array, **scalar_kwargs)
    else:
        h5file.create_dataset(path, data=np_array, **kwargs)


def load_array_from_hdf5(h5file, path: str):
    """
    Load a numpy array from HDF5 and convert to JAX array.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.

    Returns:
        The loaded JAX array.
    """
    return jnp.asarray(h5file[path][()])


def save_model_parameters_to_hdf5(
    h5file, path: str, model_params: Model_Parameters, **kwargs
) -> None:
    """
    Save Model_Parameters to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        model_params: The Model_Parameters object to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    group = h5file.create_group(path)

    # Save class info for loading
    class_info = f"{model_params.__class__.__module__}.{model_params.__class__.__name__}"
    group.attrs["class_info"] = class_info

    # Save dynamic parameters
    dynamic_slots, static_slots = model_params._get_grouped_slots()
    for slot in dynamic_slots:
        value = getattr(model_params, slot)
        save_array_to_hdf5(group, slot, value, **kwargs)

    # Save static parameters (just the key for now)
    key_list = list(model_params.key)
    group.attrs["key"] = str(key_list)


def load_model_parameters_from_hdf5(
    h5file, path: str, default_cls: Optional[Type[T_mp]] = None
) -> Model_Parameters:
    """
    Load Model_Parameters from HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        default_cls: The default Model_Parameters class to use if class info is not available.

    Returns:
        The loaded Model_Parameters object.
    """
    group = h5file[path]

    # Get class info or use provided class
    if default_cls is None:
        class_info = group.attrs["class_info"]
        module_name, class_name = class_info.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    else:
        cls = default_cls

    # Load dynamic parameters
    dynamic_slots, static_slots = cls._get_grouped_slots()
    param_dict = {}
    for slot in dynamic_slots:
        param_dict[slot] = load_array_from_hdf5(group, slot)

    # Load static parameters
    key_str = group.attrs["key"]
    key_list = literal_eval(key_str)
    # param_dict["key"] = frozenset(key_list)

    return cls(**param_dict)


def save_simulation_parameters_to_hdf5(
    h5file, path: str, sim_params: Simulation_Parameters, **kwargs
) -> None:
    """
    Save Simulation_Parameters to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        sim_params: The Simulation_Parameters object to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    group = h5file.create_group(path)

    # Save arrays
    save_array_to_hdf5(group, "frame_weights", sim_params.frame_weights, **kwargs)
    save_array_to_hdf5(group, "frame_mask", sim_params.frame_mask, **kwargs)
    save_array_to_hdf5(group, "forward_model_weights", sim_params.forward_model_weights, **kwargs)
    save_array_to_hdf5(
        group, "normalise_loss_functions", sim_params.normalise_loss_functions, **kwargs
    )
    save_array_to_hdf5(group, "forward_model_scaling", sim_params.forward_model_scaling, **kwargs)

    # Save model_parameters sequence
    model_params_group = group.create_group("model_parameters")
    for i, model_param in enumerate(sim_params.model_parameters):
        save_model_parameters_to_hdf5(model_params_group, f"{i}", model_param, **kwargs)


def load_simulation_parameters_from_hdf5(
    h5file, path: str, default_model_params_cls: Optional[Type[T_mp]] = None
) -> Simulation_Parameters:
    """
    Load Simulation_Parameters from HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        default_model_params_cls: The default Model_Parameters class to use if class info is not available.

    Returns:
        The loaded Simulation_Parameters object.
    """
    group = h5file[path]

    # Load arrays
    frame_weights = load_array_from_hdf5(group, "frame_weights")
    frame_mask = load_array_from_hdf5(group, "frame_mask")
    forward_model_weights = load_array_from_hdf5(group, "forward_model_weights")
    normalise_loss_functions = load_array_from_hdf5(group, "normalise_loss_functions")
    forward_model_scaling = load_array_from_hdf5(group, "forward_model_scaling")

    # Load model_parameters sequence
    model_params_group = group["model_parameters"]
    model_params = []
    for i in range(len(model_params_group)):
        model_param = load_model_parameters_from_hdf5(
            model_params_group, f"{i}", default_model_params_cls
        )
        model_params.append(model_param)

    return Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=frame_mask,
        model_parameters=model_params,
        forward_model_weights=forward_model_weights,
        normalise_loss_functions=normalise_loss_functions,
        forward_model_scaling=forward_model_scaling,
    )


def save_loss_components_to_hdf5(h5file, path: str, loss_comps: LossComponents, **kwargs) -> None:
    """
    Save LossComponents to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        loss_comps: The LossComponents object to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    group = h5file.create_group(path)

    # Save arrays
    save_array_to_hdf5(group, "train_losses", loss_comps.train_losses, **kwargs)
    save_array_to_hdf5(group, "val_losses", loss_comps.val_losses, **kwargs)
    save_array_to_hdf5(group, "scaled_train_losses", loss_comps.scaled_train_losses, **kwargs)
    save_array_to_hdf5(group, "scaled_val_losses", loss_comps.scaled_val_losses, **kwargs)
    save_array_to_hdf5(group, "total_train_loss", loss_comps.total_train_loss, **kwargs)
    save_array_to_hdf5(group, "total_val_loss", loss_comps.total_val_loss, **kwargs)


def load_loss_components_from_hdf5(h5file, path: str) -> LossComponents:
    """
    Load LossComponents from HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.

    Returns:
        The loaded LossComponents object.
    """
    group = h5file[path]

    # Load arrays
    train_losses = load_array_from_hdf5(group, "train_losses")
    val_losses = load_array_from_hdf5(group, "val_losses")
    scaled_train_losses = load_array_from_hdf5(group, "scaled_train_losses")
    scaled_val_losses = load_array_from_hdf5(group, "scaled_val_losses")
    total_train_loss = load_array_from_hdf5(group, "total_train_loss")
    total_val_loss = load_array_from_hdf5(group, "total_val_loss")

    return LossComponents(
        train_losses=train_losses,
        val_losses=val_losses,
        scaled_train_losses=scaled_train_losses,
        scaled_val_losses=scaled_val_losses,
        total_train_loss=total_train_loss,
        total_val_loss=total_val_loss,
    )


def save_optimization_state_to_hdf5(h5file, path: str, state: OptimizationState, **kwargs) -> None:
    """
    Save OptimizationState to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        state: The OptimizationState object to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    group = h5file.create_group(path)

    # Save params
    save_simulation_parameters_to_hdf5(group, "params", state.params, **kwargs)

    # Save gradient_mask
    save_simulation_parameters_to_hdf5(group, "gradient_mask", state.gradient_mask, **kwargs)

    # Save step
    group.attrs["step"] = state.step

    # Save losses if present
    if state.losses is not None:
        save_loss_components_to_hdf5(group, "losses", state.losses, **kwargs)
        group.attrs["has_losses"] = True
    else:
        group.attrs["has_losses"] = False


def load_optimization_state_from_hdf5(
    h5file, path: str, default_model_params_cls: Optional[Type[T_mp]] = None
) -> OptimizationState:
    """
    Load OptimizationState from HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        default_model_params_cls: The default Model_Parameters class to use if class info is not available.

    Returns:
        The loaded OptimizationState object.
    """
    group = h5file[path]

    # Load params
    params = load_simulation_parameters_from_hdf5(group, "params", default_model_params_cls)

    # Load gradient_mask
    gradient_mask = load_simulation_parameters_from_hdf5(
        group, "gradient_mask", default_model_params_cls
    )

    # Load step
    step = group.attrs["step"]

    # Load losses if present
    losses = None
    if group.attrs["has_losses"]:
        losses = load_loss_components_from_hdf5(group, "losses")

    return OptimizationState(
        params=params,
        opt_state=None,  # As per requirement, opt_state doesn't need to be saved
        gradient_mask=gradient_mask,
        step=step,
        losses=losses,
    )


def save_optimization_history_to_hdf5(
    h5file, path: str, history: OptimizationHistory, **kwargs
) -> None:
    """
    Save OptimizationHistory to HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        history: The OptimizationHistory object to save.
        **kwargs: Additional arguments to pass to h5py.create_dataset.
    """
    group = h5file.create_group(path)

    # Save states
    states_group = group.create_group("states")
    for i, state in enumerate(history.states):
        save_optimization_state_to_hdf5(states_group, f"{i}", state, **kwargs)

    # Save best_state if present
    if history.best_state is not None:
        save_optimization_state_to_hdf5(group, "best_state", history.best_state, **kwargs)
        group.attrs["has_best_state"] = True
    else:
        group.attrs["has_best_state"] = False


def load_optimization_history_from_hdf5(
    h5file, path: str, default_model_params_cls: Optional[Type[T_mp]] = None
) -> OptimizationHistory:
    """
    Load OptimizationHistory from HDF5.

    Args:
        h5file: The HDF5 file or group.
        path: The path within the HDF5 file.
        default_model_params_cls: The default Model_Parameters class to use if class info is not available.

    Returns:
        The loaded OptimizationHistory object.
    """
    group = h5file[path]

    # Load states
    states_group = group["states"]
    states = []
    for i in range(len(states_group)):
        state = load_optimization_state_from_hdf5(states_group, f"{i}", default_model_params_cls)
        states.append(state)

    # Load best_state if present
    best_state = None
    if group.attrs["has_best_state"]:
        best_state = load_optimization_state_from_hdf5(
            group, "best_state", default_model_params_cls
        )

    return OptimizationHistory(states=states, best_state=best_state)


def save_optimization_history_to_file(
    filename: str, history: OptimizationHistory, compress: bool = True
) -> None:
    """
    Save OptimizationHistory to an HDF5 file.

    Args:
        filename: The path to the HDF5 file.
        history: The OptimizationHistory object to save.
        compress: Whether to compress the datasets in the HDF5 file.
    """
    mode = "w"
    kwargs = {}
    if compress:
        kwargs["compression"] = "gzip"
        kwargs["compression_opts"] = 9
        # Add chunks=True to let h5py determine appropriate chunk sizes for non-scalar data
        kwargs["chunks"] = True

    with h5py.File(filename, mode) as f:
        save_optimization_history_to_hdf5(f, "optimization_history", history, **kwargs)


def load_optimization_history_from_file(
    filename: str, default_model_params_cls: Optional[Type[T_mp]] = None
) -> OptimizationHistory:
    """
    Load OptimizationHistory from an HDF5 file.

    Args:
        filename: The path to the HDF5 file.
        default_model_params_cls: The default Model_Parameters class to use if class info is not available.

    Returns:
        The loaded OptimizationHistory object.
    """
    with h5py.File(filename, "r") as f:
        return load_optimization_history_from_hdf5(
            f, "optimization_history", default_model_params_cls
        )
