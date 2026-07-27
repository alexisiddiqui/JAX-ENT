import h5py
import jax.numpy as jnp
import numpy as np

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.utils.hdf import (
    load_simulation_parameters_from_hdf5,
    save_simulation_parameters_to_hdf5,
)


def _params(weights):
    return Simulation_Parameters.from_frame_weights(
        weights,
        model_parameters=[],
        forward_model_weights=jnp.asarray([1.0]),
        normalise_loss_functions=jnp.asarray([1]),
        forward_model_scaling=jnp.asarray([1.0]),
    )


def _write_non_frame_fields(group):
    group.create_dataset("forward_model_weights", data=np.asarray([1.0]))
    group.create_dataset("normalise_loss_functions", data=np.asarray([1]))
    group.create_dataset("forward_model_scaling", data=np.asarray([1.0]))
    group.create_group("model_parameters")


def test_round_trip_writes_both_datasets_and_prefers_simplex(tmp_path):
    path = tmp_path / "weights.h5"
    weights = np.asarray([0.7, 0.2, 0.1])
    with h5py.File(path, "w") as handle:
        save_simulation_parameters_to_hdf5(handle, "params", _params(weights))
        assert "frame_weight_logits" in handle["params"]
        assert "frame_weight_simplex" in handle["params"]
        handle["params/frame_weight_logits"][...] = np.asarray([100.0, -100.0, 0.0])

    with h5py.File(path, "r") as handle:
        loaded = load_simulation_parameters_from_hdf5(handle, "params")
    np.testing.assert_allclose(loaded.frame_weight_simplex, weights, rtol=1e-6)


def test_legacy_frame_weights_load_under_both_role_hints(tmp_path):
    weights = np.asarray([0.7, 0.2, 0.1])
    for role in ("simplex", None):
        path = tmp_path / f"legacy-{role}.h5"
        with h5py.File(path, "w") as handle:
            group = handle.create_group("params")
            group.create_dataset("frame_weights", data=weights)
            _write_non_frame_fields(group)
        with h5py.File(path, "r") as handle:
            loaded = load_simulation_parameters_from_hdf5(
                handle, "params", legacy_role=role
            )
        np.testing.assert_allclose(loaded.frame_weight_simplex, weights, rtol=1e-6)


def test_legacy_logits_role_is_preserved(tmp_path):
    logits = np.asarray([2.0, -1.0, 0.5])
    path = tmp_path / "legacy-logits.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("params")
        group.create_dataset("frame_weights", data=logits)
        _write_non_frame_fields(group)
    with h5py.File(path, "r") as handle:
        loaded = load_simulation_parameters_from_hdf5(
            handle, "params", legacy_role="logits"
        )
    np.testing.assert_allclose(loaded.frame_weight_logits, logits)
