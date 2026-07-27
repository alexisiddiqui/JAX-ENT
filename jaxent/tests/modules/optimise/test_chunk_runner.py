import jax
import jax.numpy as jnp

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.opt.chunk import ChunkInputs, run_batch, run_sequential
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _build_chunk_state, result_to_history, run_optimise
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    synthetic_output_l2_loss,
)


def _run_chunk(
    chunk_size: int,
    *,
    n_steps: int = 10,
    tolerance: float = -jnp.inf,
    convergence: list[float] | None = None,
):
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=0.1)
    initial_state = optimizer.initialise(simulation)
    carry, inputs, loss_functions, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        tolerance,
        [-jnp.inf] if convergence is None else convergence,
        [0],
        [synthetic_output_l2_loss],
        initial_state,
        optimizer,
    )
    result = run_sequential(
        carry,
        inputs,
        n_steps,
        chunk_size,
        optimizer,
        loss_functions,
        indexes,
        2,
    )
    return result, optimizer, initial_state


def test_chunk_remainders_execute_exactly_requested_steps_and_match() -> None:
    results = [_run_chunk(size)[0] for size in (1, 3, 5, 10, 16)]
    assert all(int(result.carry.executed_steps) == 10 for result in results)
    reference = results[0].carry.opt_state.params.frame_weights
    for result in results[1:]:
        assert jnp.allclose(result.carry.opt_state.params.frame_weights, reference, atol=1e-6)


def test_tolerance_termination_stops_before_n_steps() -> None:
    result, _, _ = _run_chunk(3, n_steps=100, tolerance=1e-2)
    assert int(result.carry.executed_steps) == 5
    assert result.records.step.shape == (2,)
    assert not bool(result.carry.active)


def test_convergence_carry_contains_only_scalar_leaves() -> None:
    result, _, _ = _run_chunk(3, n_steps=1)
    leaves = jax.tree_util.tree_leaves(result.carry.convergence)
    assert leaves
    assert all(jnp.ndim(leaf) == 0 for leaf in leaves)


def test_static_gradient_mask_is_active_from_first_step() -> None:
    def run_one_step(partitions):
        simulation, _ = _create_synthetic_simulation()
        optimizer = OptaxOptimizer(
            learning_rate=0.1,
            optimizer="sgd",
            parameter_partition_masks=partitions,
        )
        initial_state = optimizer.initialise(simulation)
        carry, inputs, loss_functions, indexes = _build_chunk_state(
            simulation,
            (jnp.asarray([10.0], dtype=jnp.float32),),
            -jnp.inf,
            [-jnp.inf],
            [0],
            [synthetic_output_l2_loss],
            initial_state,
            optimizer,
        )
        result = run_sequential(
            carry, inputs, 1, 1, optimizer, loss_functions, indexes, 2
        )
        return initial_state.params.model_parameters[0].bias, result.carry.opt_state.params.model_parameters[0].bias

    initial_bias, enabled_bias = run_one_step(
        {Optimisable_Parameters.frame_weights, Optimisable_Parameters.model_parameters}
    )
    _, disabled_bias = run_one_step({Optimisable_Parameters.frame_weights})

    assert not jnp.isclose(enabled_bias, initial_bias)
    assert jnp.isclose(disabled_bias, initial_bias)


def test_nonfinite_loss_freezes_last_finite_state() -> None:
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=0.1)
    initial_state = optimizer.initialise(simulation)
    carry, inputs, loss_functions, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([jnp.nan], dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [synthetic_output_l2_loss],
        initial_state,
        optimizer,
    )
    result = run_sequential(carry, inputs, 10, 3, optimizer, loss_functions, indexes, 2)
    assert not bool(result.carry.active)
    assert jnp.array_equal(
        result.carry.opt_state.params.frame_weights,
        initial_state.params.frame_weights,
    )


def test_single_terminal_threshold_is_recorded_as_convergence_event() -> None:
    result, optimizer, _ = _run_chunk(3, convergence=[1e6])
    history = result_to_history(result, optimizer)
    assert bool(jnp.any(result.records.threshold_event))
    assert history.convergence_states


def test_fixed_step_compiled_python_parity() -> None:
    simulation_compiled, models_compiled = _create_synthetic_simulation()
    compiled_config = OptimiserSettings(
        name="compiled_parity",
        n_steps=10,
        tolerance=-jnp.inf,
        convergence=-jnp.inf,
        learning_rate=0.1,
        step_chunk_size=1,
        execution_mode="compiled",
    )
    _, compiled_history = run_optimise(
        simulation_compiled,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        compiled_config,
        models_compiled,
        [0],
        [synthetic_output_l2_loss],
    )

    simulation_python, models_python = _create_synthetic_simulation()
    python_config = OptimiserSettings(
        name="python_parity",
        n_steps=10,
        tolerance=-jnp.inf,
        convergence=-jnp.inf,
        learning_rate=0.1,
        step_chunk_size=1,
        execution_mode="python",
    )
    _, python_history = run_optimise(
        simulation_python,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        python_config,
        models_python,
        [0],
        [synthetic_output_l2_loss],
    )
    assert jnp.allclose(
        compiled_history.best_state.losses.total_train_loss,
        python_history.best_state.losses.total_train_loss,
        rtol=1e-4,
    )
    assert jnp.allclose(
        compiled_history.best_state.params.frame_weights,
        python_history.best_state.params.frame_weights,
        atol=1e-4,
    )


def test_history_is_boundary_granular() -> None:
    simulation, models = _create_synthetic_simulation()
    config = OptimiserSettings(
        name="history_granularity",
        n_steps=1000,
        tolerance=-jnp.inf,
        convergence=-jnp.inf,
        learning_rate=0.1,
        step_chunk_size=100,
    )
    _, history = run_optimise(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        config,
        models,
        [0],
        [synthetic_output_l2_loss],
    )
    assert len(history.states) <= 10
    assert all(state in history.states for state in history.convergence_states)


def _oscillating_parameter_loss(model, _target, _index):
    loss = jnp.square(model.params.frame_weights[0] - 0.8)
    return loss, loss


def _non_monotonic_parameter_loss(model, _target, _index):
    loss = jnp.sin(20.0 * model.params.frame_weights[0]) + 1.0
    return loss, loss


def test_learning_rate_state_persists_across_chunks() -> None:
    rates = []
    for chunk_size in (1, 4, 12):
        simulation, _ = _create_synthetic_simulation()
        optimizer = OptaxOptimizer(learning_rate=5.0, optimizer="sgd")
        initial_state = optimizer.initialise(simulation)
        carry, inputs, loss_functions, indexes = _build_chunk_state(
            simulation,
            (jnp.asarray([0.0], dtype=jnp.float32),),
            -jnp.inf,
            [-jnp.inf],
            [0],
            [_non_monotonic_parameter_loss],
            initial_state,
            optimizer,
        )
        result = run_sequential(
            carry,
            inputs,
            12,
            chunk_size,
            optimizer,
            loss_functions,
            indexes,
            2,
        )
        rate = result.metrics.lr
        assert bool(jnp.all(rate[1:] <= rate[:-1]))
        reductions = int(jnp.sum(rate[1:] < rate[:-1]))
        assert reductions >= 2
        assert jnp.isclose(
            rate[-1],
            5.0 / optimizer.plateau_denominator**reductions,
            rtol=1e-5,
        )
        rates.append(rate)
    assert jnp.allclose(rates[0], rates[1], atol=1e-6)
    assert jnp.allclose(rates[0], rates[2], atol=1e-6)


def test_running_best_is_not_replaced_by_final_state() -> None:
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=0.5, optimizer="sgd")
    initial_state = optimizer.initialise(simulation)
    carry, inputs, loss_functions, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([0.0], dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [_non_monotonic_parameter_loss],
        initial_state,
        optimizer,
    )
    result = run_sequential(
        carry,
        inputs,
        15,
        3,
        optimizer,
        loss_functions,
        indexes,
        2,
    )
    assert result.carry.best.step < result.carry.opt_state.step
    assert result.carry.best.losses.total_val_loss < result.carry.opt_state.losses.total_val_loss


def test_stopped_batch_lane_remains_frozen() -> None:
    simulation, _ = _create_synthetic_simulation()
    simulations = [simulation, simulation]
    optimizer = OptaxOptimizer(learning_rate=0.1)
    carries = []
    inputs = []
    initial_params = []
    for simulation, tolerance in zip(simulations, (1e9, -jnp.inf)):
        initial_state = optimizer.initialise(simulation)
        carry, lane_inputs, loss_functions, indexes = _build_chunk_state(
            simulation,
            (jnp.asarray([10.0], dtype=jnp.float32),),
            tolerance,
            [-jnp.inf],
            [0],
            [synthetic_output_l2_loss],
            initial_state,
            optimizer,
        )
        carries.append(carry)
        inputs.append(lane_inputs)
        initial_params.append(initial_state.params)

    batched_carries = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *carries)
    batched_inputs = ChunkInputs(
        data_targets=(),
        convergence_thresholds=jnp.stack([item.convergence_thresholds for item in inputs]),
        tolerance=jnp.stack([item.tolerance for item in inputs]),
        ema_alpha=jnp.stack([item.ema_alpha for item in inputs]),
    )
    result = run_batch(
        batched_carries,
        batched_inputs,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        8,
        3,
        optimizer,
        (synthetic_output_l2_loss,),
        (0,),
        2,
    )
    assert jnp.array_equal(
        result.carry.opt_state.params.frame_weights[0],
        initial_params[0].frame_weights,
    )
    assert int(result.carry.executed_steps[0]) == 0


def test_chunk_runner_has_at_most_one_boundary_sync_per_chunk() -> None:
    from profiling.profile_hdx_cpu import count_host_materialisation

    with count_host_materialisation() as counts:
        result, _, _ = _run_chunk(3, n_steps=10)
        result.carry.executed_steps.block_until_ready()
    # Boundary convergence checks are allowed; there must be no per-step sync.
    assert sum(counts.values()) <= 4
