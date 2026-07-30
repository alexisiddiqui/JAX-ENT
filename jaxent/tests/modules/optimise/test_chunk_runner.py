import gc

import jax
import jax.numpy as jnp

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.config import Optimisable_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt import chunk as chunk_module
from jaxent.src.opt.chunk import (
    ChunkCarry,
    ChunkInputs,
    _select_carry,
    optimisation_step,
    run_batch,
    run_sequential,
    run_step_chunk,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _build_chunk_state, result_to_history, run_optimise
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    SyntheticForwardModel,
    SyntheticForwardModelConfig,
    SyntheticInputFeatures,
    SyntheticModelParameters,
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
    reference = results[0].carry.opt_state.params.frame_weight_simplex
    for result in results[1:]:
        assert jnp.allclose(result.carry.opt_state.params.frame_weight_simplex, reference, atol=1e-6)


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


def test_optimisation_step_preserves_sim_identity_for_all_eager_selections() -> None:
    def make_step(target):
        simulation, _ = _create_synthetic_simulation()
        optimizer = OptaxOptimizer(learning_rate=0.1)
        initial_state = optimizer.initialise(simulation)
        carry, inputs, losses, indexes = _build_chunk_state(
            simulation,
            (jnp.asarray([target], dtype=jnp.float32),),
            -jnp.inf,
            [-jnp.inf],
            [0],
            [synthetic_output_l2_loss],
            initial_state,
            optimizer,
        )
        return carry, inputs, optimizer, losses, indexes

    active = make_step(10.0)
    active_carry, active_inputs, active_optimizer, active_losses, active_indexes = active
    active_result, _ = optimisation_step(
        active_carry,
        active_inputs,
        active_optimizer,
        active_losses,
        active_indexes,
        2,
    )

    frozen_carry = active_carry._replace(active=jnp.asarray(False))
    frozen_result, _ = optimisation_step(
        frozen_carry,
        active_inputs,
        active_optimizer,
        active_losses,
        active_indexes,
        2,
    )

    invalid = make_step(jnp.nan)
    invalid_result, _ = optimisation_step(
        invalid[0], invalid[1], invalid[2], invalid[3], invalid[4], 2
    )

    assert active_result.sim is active_carry.sim
    assert frozen_result.sim is frozen_carry.sim
    assert invalid_result.sim is invalid[0].sim


# This is an end-to-end numerical and compiled-vs-traced consistency check, not
# the direct guard for accidentally dropping a dynamic carry field.
def test_jit_scan_matches_pure_scan_for_full_step_state(monkeypatch) -> None:
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=0.1)
    initial_state = optimizer.initialise(simulation)
    carry, inputs, losses, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [synthetic_output_l2_loss],
        initial_state,
        optimizer,
    )

    def run_scan():
        def scan_step(current, _):
            return optimisation_step(current, inputs, optimizer, losses, indexes, 2)

        return jax.lax.scan(scan_step, carry, None, length=10)

    def _reference_select_carry(predicate, stepped, frozen):
        """Pre-fix whole-carry select, kept as a parity reference only."""
        return jax.tree_util.tree_map(
            lambda a, b: jax.lax.select(predicate, a, b), stepped, frozen
        )

    production_select_carry = chunk_module._select_carry
    monkeypatch.setattr(chunk_module, "_select_carry", _reference_select_carry)
    reference_carry, reference_metrics = run_scan()
    monkeypatch.setattr(chunk_module, "_select_carry", production_select_carry)
    scan_carry, scan_metrics = run_scan()

    assert jnp.allclose(
        reference_metrics.total_train_loss, scan_metrics.total_train_loss
    )
    assert jnp.allclose(
        reference_carry.opt_state.losses.total_train_loss,
        scan_carry.opt_state.losses.total_train_loss,
    )
    assert jnp.allclose(
        reference_carry.opt_state.losses.total_train_loss,
        scan_carry.opt_state.losses.total_train_loss,
    )
    for reference_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(reference_carry.opt_state.params),
        jax.tree_util.tree_leaves(scan_carry.opt_state.params),
        strict=True,
    ):
        assert jnp.allclose(reference_leaf, scan_leaf)
    assert jnp.array_equal(reference_carry.executed_steps, scan_carry.executed_steps)
    assert jnp.array_equal(reference_carry.active, scan_carry.active)
    assert jnp.allclose(reference_carry.lr, scan_carry.lr)
    assert jnp.allclose(reference_carry.model_lr, scan_carry.model_lr)
    for reference_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(reference_carry.convergence),
        jax.tree_util.tree_leaves(scan_carry.convergence),
        strict=True,
    ):
        assert jnp.allclose(reference_leaf, scan_leaf)
    for reference_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(reference_carry._replace(sim=None)),
        jax.tree_util.tree_leaves(scan_carry._replace(sim=None)),
        strict=True,
    ):
        if reference_leaf is None:
            assert scan_leaf is None
        else:
            assert jnp.allclose(reference_leaf, scan_leaf, equal_nan=True)

    compiled_carry, compiled_metrics, _ = run_step_chunk(
        carry, inputs, optimizer, losses, indexes, 10, 2
    )
    assert jnp.allclose(compiled_metrics.total_train_loss, scan_metrics.total_train_loss)
    assert jnp.allclose(
        compiled_carry.opt_state.losses.total_train_loss,
        scan_carry.opt_state.losses.total_train_loss,
    )
    for compiled_leaf, scan_leaf in zip(
        jax.tree_util.tree_leaves(compiled_carry.opt_state.params),
        jax.tree_util.tree_leaves(scan_carry.opt_state.params),
        strict=True,
    ):
        assert jnp.allclose(compiled_leaf, scan_leaf)
    assert jnp.array_equal(compiled_carry.executed_steps, scan_carry.executed_steps)
    assert jnp.array_equal(compiled_carry.active, scan_carry.active)


def test_select_carry_takes_frozen_values_for_every_dynamic_field() -> None:
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(learning_rate=0.1)
    initial_state = optimizer.initialise(simulation)
    carry, inputs, losses, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([10.0], dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [synthetic_output_l2_loss],
        initial_state,
        optimizer,
    )
    stepped, _ = optimisation_step(carry, inputs, optimizer, losses, indexes, 2)
    stepped = stepped._replace(
        lr=jnp.asarray(9.0),
        model_lr=jnp.asarray(8.0),
        executed_steps=stepped.executed_steps + 7,
        active=jnp.asarray(True),
    )
    frozen = carry._replace(
        lr=jnp.asarray(1.0),
        model_lr=jnp.asarray(2.0),
        active=jnp.asarray(False),
    )

    chosen_frozen = _select_carry(jnp.asarray(False), stepped, frozen)
    chosen_stepped = _select_carry(jnp.asarray(True), stepped, frozen)

    expected_fields = {
        "opt_state",
        "sim",
        "convergence",
        "lr",
        "model_lr",
        "executed_steps",
        "active",
        "best",
        "convergence_snapshots",
    }
    assert set(ChunkCarry._fields) == expected_fields

    for field in ChunkCarry._fields:
        if field == "sim":
            assert chosen_frozen.sim is stepped.sim
            assert chosen_stepped.sim is stepped.sim
        elif field in {
            "opt_state",
            "convergence",
            "best",
            "convergence_snapshots",
        }:
            for chosen, expected in (
                (getattr(chosen_frozen, field), getattr(frozen, field)),
                (getattr(chosen_stepped, field), getattr(stepped, field)),
            ):
                for chosen_leaf, expected_leaf in zip(
                    jax.tree_util.tree_leaves(chosen),
                    jax.tree_util.tree_leaves(expected),
                    strict=True,
                ):
                    assert jnp.allclose(chosen_leaf, expected_leaf)
        else:
            assert jnp.allclose(
                getattr(chosen_frozen, field), getattr(frozen, field)
            )
            assert jnp.allclose(
                getattr(chosen_stepped, field), getattr(stepped, field)
            )


def test_eager_step_live_array_growth_stays_below_feature_tree_size() -> None:
    threshold_bytes = 100_000
    models = [SyntheticForwardModel(SyntheticForwardModelConfig())]
    parameters = Simulation_Parameters.from_frame_weights(
        jnp.ones(5, dtype=jnp.float32),
        model_parameters=[SyntheticModelParameters()],
        forward_model_weights=jnp.ones(1, dtype=jnp.float32),
        forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.float32),
    )
    simulation = Simulation(
        input_features=[
            SyntheticInputFeatures(jnp.ones((20_000, 5), dtype=jnp.float32))
        ],
        forward_models=models,
        params=parameters,
    )
    simulation.initialise()
    optimizer = OptaxOptimizer(learning_rate=0.1)
    initial_state = optimizer.initialise(simulation)
    carry, inputs, losses, indexes = _build_chunk_state(
        simulation,
        (jnp.ones(20_000, dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [synthetic_output_l2_loss],
        initial_state,
        optimizer,
    )

    sim_bytes = sum(
        int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(carry.sim)
    )
    assert sim_bytes >= 4 * threshold_bytes

    samples = []
    for _ in range(12):
        carry, metrics = optimisation_step(
            carry, inputs, optimizer, losses, indexes, 2
        )
        jax.tree_util.tree_map(
            lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
            (carry, metrics),
        )
        gc.collect()
        samples.append(sum(int(array.nbytes) for array in jax.live_arrays()))

    samples = samples[2:]
    slope = (samples[-1] - samples[0]) / (len(samples) - 1)
    assert slope <= threshold_bytes


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
        result.carry.opt_state.params.frame_weight_simplex,
        initial_state.params.frame_weight_simplex,
    )


def test_single_terminal_threshold_is_recorded_as_convergence_event() -> None:
    result, optimizer, _ = _run_chunk(3, convergence=[1e6])
    history = result_to_history(result, optimizer)
    assert bool(jnp.any(result.records.threshold_event))
    assert history.convergence_states


def test_multiple_thresholds_are_captured_at_distinct_steps_within_one_chunk() -> None:
    thresholds = [1e6, 1e5, 1e4]
    chunked, optimizer, _ = _run_chunk(
        40,
        n_steps=40,
        convergence=thresholds,
    )
    per_step, _, _ = _run_chunk(
        1,
        n_steps=40,
        convergence=thresholds,
    )

    chunked_history = result_to_history(chunked, optimizer)
    chunked_steps = [int(state.step) for state in chunked_history.convergence_states]
    per_step_steps = [
        int(state.step)
        for state in result_to_history(per_step, optimizer).convergence_states
    ]

    assert len(chunked_steps) == len(thresholds)
    assert chunked_steps == sorted(set(chunked_steps))
    assert chunked_steps == per_step_steps
    assert len(chunked.records.step) == 1
    for chunked_state, per_step_state in zip(
        chunked_history.convergence_states,
        result_to_history(per_step, optimizer).convergence_states,
        strict=True,
    ):
        assert jnp.allclose(
            chunked_state.params.frame_weight_simplex,
            per_step_state.params.frame_weight_simplex,
            atol=1e-7,
        )


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
        compiled_history.best_state.params.frame_weight_simplex,
        python_history.best_state.params.frame_weight_simplex,
        atol=1e-4,
    )


def test_history_is_boundary_granular_and_convergence_states_are_independent() -> None:
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
    assert all(
        all(convergence_state is not state for state in history.states)
        for convergence_state in history.convergence_states
    )


def _oscillating_parameter_loss(model, _target, _index):
    loss = jnp.square(model.params.frame_weight_simplex[0] - 0.8)
    return loss, loss


def _non_monotonic_parameter_loss(model, _target, _index):
    loss = jnp.sin(20.0 * model.params.frame_weight_simplex[0]) + 1.0
    return loss, loss


def test_oscillation_damping_is_non_compounding_across_chunks() -> None:
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
        base_rate = jnp.asarray(5.0)
        damped_rate = base_rate / optimizer.plateau_denominator
        assert bool(
            jnp.all(
                jnp.isclose(rate, base_rate, rtol=1e-5)
                | jnp.isclose(rate, damped_rate, rtol=1e-5)
            )
        )
        assert int(jnp.sum(jnp.isclose(rate, damped_rate, rtol=1e-5))) >= 2
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
        base_lr=jnp.stack([item.base_lr for item in inputs]),
        base_model_lr=jnp.stack([item.base_model_lr for item in inputs]),
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
        result.carry.opt_state.params.frame_weight_simplex[0],
        initial_params[0].frame_weight_simplex,
    )
    assert int(result.carry.executed_steps[0]) == 0


def test_chunk_runner_has_at_most_one_boundary_sync_per_chunk() -> None:
    from profiling.profile_hdx_cpu import count_host_materialisation

    with count_host_materialisation() as counts:
        result, _, _ = _run_chunk(3, n_steps=10)
        result.carry.executed_steps.block_until_ready()
    # Boundary convergence checks are allowed; there must be no per-step sync.
    assert sum(counts.values()) <= 4
