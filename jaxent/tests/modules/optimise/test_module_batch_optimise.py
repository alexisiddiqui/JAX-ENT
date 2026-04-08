import jax.numpy as jnp

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.run import run_optimise
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    synthetic_output_l2_loss,
)


def _make_config() -> OptimiserSettings:
    return OptimiserSettings(
        name="module_batch",
        n_steps=8,
        tolerance=1e-8,
        convergence=200.0,
        learning_rate=0.1,
    )


def _with_hparams(
    base_params: Simulation_Parameters,
    forward_model_weights,
    forward_model_scaling,
) -> Simulation_Parameters:
    return Simulation_Parameters(
        frame_weights=base_params.frame_weights,
        frame_mask=base_params.frame_mask,
        model_parameters=base_params.model_parameters,
        forward_model_weights=forward_model_weights,
        forward_model_scaling=forward_model_scaling,
        normalise_loss_functions=base_params.normalise_loss_functions,
    )


def _run_sequential_reference(
    hparam_batch: HParamBatch,
    config: OptimiserSettings,
) -> list[tuple]:
    references = []
    n_hparams = int(hparam_batch.forward_model_weights.shape[0])
    learning_rates = (
        hparam_batch.learning_rate
        if hparam_batch.learning_rate is not None
        else jnp.full((n_hparams,), config.learning_rate, dtype=jnp.float32)
    )

    for i in range(n_hparams):
        simulation, models = _create_synthetic_simulation()
        simulation.params = _with_hparams(
            simulation.params,
            forward_model_weights=hparam_batch.forward_model_weights[i],
            forward_model_scaling=hparam_batch.forward_model_scaling[i],
        )
        simulation.forward(simulation, simulation.params)
        config_i = OptimiserSettings(
            name=f"{config.name}_seq_{i}",
            n_steps=config.n_steps,
            tolerance=config.tolerance,
            convergence=config.convergence,
            learning_rate=float(learning_rates[i]),
            optimiser_type=config.optimiser_type,
            loss_constants=config.loss_constants,
            ema_alpha=config.ema_alpha,
            min_steps_per_threshold=config.min_steps_per_threshold,
        )
        _, history = run_optimise(
            simulation,
            data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
            config=config_i,
            forward_models=models,
            indexes=[0],
            loss_functions=[synthetic_output_l2_loss],
        )
        references.append(
            (
                history.best_state.losses.total_train_loss,
                history.best_state.params.frame_weights,
            )
        )

    return references


def test_module_batch_optimise_padded_sweep_trims_results() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparam_batch = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.asarray([[1.0], [0.8], [1.2]], dtype=jnp.float32),
        learning_rate=jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float32),
    )

    result = batch_optimise(
        simulation=simulation,
        hparam_batch=hparam_batch,
        batch_size=2,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )

    assert len(result.histories) == 3
    assert len(result.best_states) == 3
    assert result.convergence_steps.shape == (3,)
    assert bool(jnp.all(result.convergence_steps > 0))
    assert bool(jnp.all(result.convergence_steps <= config.n_steps))


def test_module_batch_optimise_uses_config_lr_when_hparam_lr_none() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparam_batch = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.asarray([[1.0], [0.8], [1.2]], dtype=jnp.float32),
        learning_rate=None,
    )
    explicit_lr_batch = HParamBatch(
        forward_model_weights=hparam_batch.forward_model_weights,
        forward_model_scaling=hparam_batch.forward_model_scaling,
        learning_rate=jnp.full((3,), config.learning_rate, dtype=jnp.float32),
    )

    result_none = batch_optimise(
        simulation=simulation,
        hparam_batch=hparam_batch,
        batch_size=2,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )

    simulation_explicit, _ = _create_synthetic_simulation()
    result_explicit = batch_optimise(
        simulation=simulation_explicit,
        hparam_batch=explicit_lr_batch,
        batch_size=2,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )

    for best_none, best_explicit in zip(result_none.best_states, result_explicit.best_states):
        assert jnp.isfinite(best_none.losses.total_train_loss)
        assert jnp.allclose(
            best_none.losses.total_train_loss,
            best_explicit.losses.total_train_loss,
            rtol=1e-4,
        )
        assert jnp.allclose(
            best_none.params.frame_weights,
            best_explicit.params.frame_weights,
            atol=1e-4,
        )


def test_module_batch_optimise_matches_sequential_final_loss_and_weights() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparam_batch = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.asarray([[1.0], [0.8], [1.2]], dtype=jnp.float32),
        learning_rate=jnp.asarray([0.1, 0.1, 0.1], dtype=jnp.float32),
    )

    batch_result = batch_optimise(
        simulation=simulation,
        hparam_batch=hparam_batch,
        batch_size=2,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
    )
    sequential_reference = _run_sequential_reference(hparam_batch, config)

    for batch_best, (seq_loss, seq_weights) in zip(batch_result.best_states, sequential_reference):
        assert jnp.allclose(
            batch_best.losses.total_train_loss,
            seq_loss,
            rtol=1e-4,
        )
        assert jnp.allclose(
            batch_best.params.frame_weights,
            seq_weights,
            atol=1e-4,
        )


def test_module_jit_update_step_case_matches_eager_reference() -> None:
    config = _make_config()
    simulation_eager, models_eager = _create_synthetic_simulation()
    _, eager_history = run_optimise(
        simulation_eager,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        forward_models=models_eager,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
        jit_update_step=False,
    )

    simulation_jit, models_jit = _create_synthetic_simulation()
    _, jit_history = run_optimise(
        simulation_jit,
        data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
        config=config,
        forward_models=models_jit,
        indexes=[0],
        loss_functions=[synthetic_output_l2_loss],
        jit_update_step=True,
    )

    assert jnp.allclose(
        eager_history.best_state.losses.total_train_loss,
        jit_history.best_state.losses.total_train_loss,
        rtol=1e-4,
    )
    assert jnp.allclose(
        eager_history.best_state.params.frame_weights,
        jit_history.best_state.params.frame_weights,
        atol=1e-4,
    )
