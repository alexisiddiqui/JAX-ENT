import jax.numpy as jnp
import pytest

from jaxent.src.custom_types.config import Optimisable_Parameters, OptimiserSettings
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.run import run_optimise
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


def test_default_and_explicit_all_partitions_match() -> None:
    default = _run(_config())
    explicit = _run(
        _config(
            parameter_partitions=frozenset(Optimisable_Parameters),
        )
    )
    assert [int(state.step) for state in default.states] == [int(state.step) for state in explicit.states]
    assert jnp.array_equal(default.states[-1].params.frame_weights, explicit.states[-1].params.frame_weights)
    assert len(default.convergence_states) == len(explicit.convergence_states)
    assert default.convergence_states[0] is default.states[0]


def test_partition_selection_and_best_state() -> None:
    history = _run(
        _config(parameter_partitions={Optimisable_Parameters.frame_weights})
    )
    assert history.states[-1].params.frame_weights.size > 0
    assert history.states[-1].params.model_parameters[0].bias.size == 0
    assert history.best_state is not None
    assert history.best_state.params.model_parameters[0].bias.size > 0


def test_retention_toggles_and_clear_best_state_error() -> None:
    no_states = _run(_config(save_states=False, save_convergence=True))
    assert no_states.states == []
    assert no_states.convergence_states
    no_best = _run(_config(save_best=False))
    assert no_best.best_state is None
    empty = _run(_config(save_states=False, save_convergence=True, save_best=False))
    with pytest.raises(ValueError, match="save_best=False and save_states=False"):
        empty.get_best_state()


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
