import numpy as np
import h5py
import jax
import jax.numpy as jnp
import pytest

from jaxent.src.custom_types.config import Optimisable_Parameters, OptimiserSettings
from jaxent.src.opt.base import HParamBatch, OptimizationHistory, OptimizationState
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.chunk import ChunkResult, run_sequential
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _build_chunk_state, result_to_history, run_optimise
from jaxent.src.utils.hdf import load_optimization_history_from_file, save_optimization_history_to_file
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    synthetic_output_l2_loss,
)


def _run(config: OptimiserSettings):
    simulation, models = _create_synthetic_simulation()
    return run_optimise(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        config,
        models,
        [0],
        [synthetic_output_l2_loss],
    )[1]


def _run_result(
    config: OptimiserSettings,
    *,
    retain_record_params: bool = True,
    parameter_partitions=None,
) -> tuple[ChunkResult, OptaxOptimizer]:
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=config.learning_rate)
    opt_state = optimizer.initialise(simulation)
    carry, inputs, loss_functions, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        config.tolerance,
        config.convergence,
        [0],
        [synthetic_output_l2_loss],
        opt_state,
        optimizer,
    )
    inputs = inputs._replace(ema_alpha=jnp.asarray(config.ema_alpha, dtype=jnp.float32))
    result = run_sequential(
        carry,
        inputs,
        config.n_steps,
        config.step_chunk_size,
        optimizer,
        loss_functions,
        indexes,
        config.min_steps_per_threshold,
        parameter_partitions=parameter_partitions,
        retain_record_params=retain_record_params,
    )
    return result, optimizer


def _legacy_result_to_history(result: ChunkResult) -> OptimizationHistory:
    """Reference conversion matching the pre-item-7 result-to-history behavior."""
    records = result.records
    active = np.asarray(jax.device_get(records.active))
    threshold_event = np.asarray(jax.device_get(records.threshold_event))
    final_state = result.carry.opt_state
    history = OptimizationHistory()

    def state_from_record(index: int) -> OptimizationState:
        return OptimizationState(
            params=jax.tree_util.tree_map(lambda value: value[index], records.params),
            opt_state=final_state.opt_state,
            step=records.step[index],
            losses=jax.tree_util.tree_map(lambda value: value[index], records.losses),
            gradients=final_state.gradients,
        )

    convergence_indices = {
        int(index) for index in np.flatnonzero(active & threshold_event)
    }
    for index in np.flatnonzero(active):
        state = state_from_record(int(index))
        history.states.append(state)
        if int(index) in convergence_indices:
            history.convergence_states.append(state)

    best = result.carry.best
    history.best_state = OptimizationState(
        params=best.params,
        opt_state=final_state.opt_state,
        step=best.step,
        losses=best.losses,
        gradients=final_state.gradients,
    )
    return history


def _assert_state_equal(actual: OptimizationState, expected: OptimizationState) -> None:
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        assert jnp.array_equal(actual_leaf, expected_leaf)


def _config(**kwargs) -> OptimiserSettings:
    values = dict(
        name="history_config",
        n_steps=4,
        step_chunk_size=2,
        tolerance=-jnp.inf,
        convergence=1e6,
        learning_rate=0.1,
    )
    values.update(kwargs)
    return OptimiserSettings(**values)


def test_default_matches_legacy_conversion_on_same_chunk_result() -> None:
    result, optimizer = _run_result(_config())
    actual = result_to_history(result, optimizer)
    expected = _legacy_result_to_history(result)
    assert len(actual.states) == len(expected.states)
    assert len(actual.convergence_states) == len(expected.convergence_states)
    assert [int(state.step) for state in actual.states] == [int(state.step) for state in expected.states]
    for actual_state, expected_state in zip(actual.states, expected.states):
        _assert_state_equal(actual_state, expected_state)
    for actual_state, expected_state in zip(actual.convergence_states, expected.convergence_states):
        _assert_state_equal(actual_state, expected_state)
    _assert_state_equal(actual.best_state, expected.best_state)
    assert all(
        all(actual_state is not state for state in actual.states)
        for actual_state in actual.convergence_states
    )


def test_partition_selection_and_best_state() -> None:
    history = _run(
        _config(parameter_partitions={Optimisable_Parameters.frame_weights})
    )
    assert history.states[-1].params.frame_weight_simplex.size > 0
    assert history.states[-1].params.model_parameters[0].bias.size == 0
    assert history.best_state is not None
    assert history.best_state.params.model_parameters[0].bias.size > 0


def test_retention_toggles_and_clear_best_state_error() -> None:
    no_states = _run(_config(save_states=False, save_convergence=True))
    assert no_states.states == []
    assert no_states.convergence_states
    no_best = _run(_config(save_best=False))
    assert no_best.best_state is None
    no_convergence = _run(_config(save_convergence=False))
    assert no_convergence.states
    assert no_convergence.convergence_states == []
    empty = _run(_config(save_states=False, save_convergence=True, save_best=False))
    with pytest.raises(ValueError, match="save_best=False and save_states=False"):
        empty.get_best_state()


def test_retain_record_params_false_emits_empty_parameter_leaves() -> None:
    result, _ = _run_result(
        _config(save_states=False, save_convergence=False),
        parameter_partitions=frozenset(),
        retain_record_params=False,
    )
    params = result.records.params
    assert params.frame_weight_simplex.size == 0
    assert params.forward_model_weights.size == 0
    assert params.model_parameters[0].bias.size == 0
    assert params.forward_model_scaling.size > 0
    assert params.normalise_loss_functions.size > 0


def test_config_normalises_and_validates_partition_policy() -> None:
    config = _config(parameter_partitions=[Optimisable_Parameters.frame_weights])
    assert config.parameter_partitions == frozenset({Optimisable_Parameters.frame_weights})
    with pytest.raises(ValueError, match="At least one"):
        _config(save_states=False, save_convergence=False, save_best=False)
    with pytest.raises(ValueError, match="must not be empty"):
        _config(parameter_partitions=[])


def test_hdf_round_trip_preserves_convergence_and_metadata(tmp_path) -> None:
    history = _run(_config(parameter_partitions={Optimisable_Parameters.frame_weights}))
    path = tmp_path / "history.h5"
    save_optimization_history_to_file(str(path), history)
    loaded = load_optimization_history_from_file(str(path))
    assert len(loaded.convergence_states) == len(history.convergence_states)
    assert loaded.state_parameter_partitions == history.state_parameter_partitions
    if loaded.convergence_states:
        assert loaded.convergence_states[0] is not loaded.states[0]


def test_legacy_python_history_migrates_states_after_initial_diagnostic(tmp_path) -> None:
    current = _run(_config())
    threshold_states = current.convergence_states
    assert threshold_states
    initial_diagnostic = current.states[0]._replace(step=jnp.asarray(1))
    legacy = OptimizationHistory(
        states=[initial_diagnostic, *threshold_states],
        best_state=current.best_state,
    )
    path = tmp_path / "legacy_history.h5"
    save_optimization_history_to_file(str(path), legacy)
    with h5py.File(path, "r+") as h5file:
        group = h5file["optimization_history"]
        group.attrs["history_format_version"] = 1
        del group["convergence_thresholds"]

    loaded = load_optimization_history_from_file(
        str(path),
        legacy_convergence_recovery=lambda history: tuple(
            10.0 ** -(index + 1)
            for index in range(len(history.convergence_states))
        ),
    )

    assert [int(state.step) for state in loaded.convergence_states] == [
        int(state.step) for state in threshold_states
    ]
    assert loaded.convergence_thresholds == tuple(
        10.0 ** -(index + 1) for index in range(len(threshold_states))
    )


def test_batch_best_states_remain_complete_when_history_best_is_disabled() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _config(save_best=False, parameter_partitions={Optimisable_Parameters.frame_weights})
    result = batch_optimise(
        simulation,
        HParamBatch(
            forward_model_weights=jnp.ones((2, 1)),
            forward_model_scaling=jnp.ones((2, 1)),
        ),
        batch_size=2,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )
    assert all(state is not None for state in result.best_states)
    assert all(history.best_state is None for history in result.histories)
    assert all(state.params.model_parameters[0].bias.size > 0 for state in result.best_states)
