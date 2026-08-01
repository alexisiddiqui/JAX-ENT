import jax.numpy as jnp
import pytest

from jaxent.examples.common.config import OptimizationConfig
from jaxent.src.opt.chunk import run_sequential
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _build_chunk_state, result_to_history
from jaxent.src.utils.jax_fn import frame_average_features
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    synthetic_output_l2_loss,
)


def _run_rates(lr_adjustment: bool):
    simulation, _ = _create_synthetic_simulation()
    optimizer = OptaxOptimizer(
        learning_rate=1.0,
        optimizer="sgd",
        lr_adjustment=lr_adjustment,
    )
    initial_state = optimizer.initialise(simulation)
    carry, inputs, losses, indexes = _build_chunk_state(
        simulation,
        (jnp.asarray([0.0], dtype=jnp.float32),),
        -jnp.inf,
        [-jnp.inf],
        [0],
        [_oscillating_loss],
        initial_state,
        optimizer,
    )
    result = run_sequential(
        carry, inputs, 12, 1, optimizer, losses, indexes, 2
    )
    return result, optimizer


def _oscillating_loss(model, _target, _index):
    loss = jnp.sin(20.0 * model.params.frame_weight_simplex[0]) + 1.0
    return loss, loss


def test_lr_adjustment_off_stays_at_base_lr_during_oscillation():
    result, _ = _run_rates(False)
    assert jnp.allclose(result.metrics.lr, 1.0)
    assert jnp.any(result.metrics.grad_dot_product[2:] < 0)


def test_lr_adjustment_on_is_noncompounding_and_restores_base_rate():
    result, optimizer = _run_rates(True)
    damped = 1.0 / optimizer.plateau_denominator
    rates = result.metrics.lr
    assert jnp.all(jnp.isclose(rates, 1.0) | jnp.isclose(rates, damped))
    # The synthetic trajectory has both oscillating and non-oscillating steps;
    # a positive-dot-product step must immediately return to base LR.
    for index in range(2, len(rates)):
        if float(result.metrics.grad_dot_product[index]) >= 0:
            assert float(rates[index]) == pytest.approx(1.0, rel=1e-6)
            break
    else:
        pytest.fail("synthetic forced-oscillation trajectory had no recovery step")


def test_frame_average_legacy_expression_and_float32_tolerance():
    values = jnp.arange(3 * 7, dtype=jnp.float32).reshape(3, 7) / 11.0
    weights = jnp.asarray([0.01, 0.02, 0.03, 0.04, 0.1, 0.3, 0.5], dtype=jnp.float32)
    expected = jnp.sum(values * weights.reshape(1, -1), axis=-1)
    legacy = frame_average_features(values, weights, "legacy_sum")
    tensordot = frame_average_features(values, weights, "tensordot")
    assert jnp.array_equal(legacy, expected)
    assert jnp.allclose(legacy, tensordot, rtol=0.0, atol=3.8146973e-6)


def test_convergence_ladder_is_observational_and_chunk_equivalent():
    def run(chunk_size):
        simulation, _ = _create_synthetic_simulation()
        optimizer = OptaxOptimizer(learning_rate=0.1)
        initial = optimizer.initialise(simulation)
        carry, inputs, losses, indexes = _build_chunk_state(
            simulation,
            (jnp.asarray([10.0], dtype=jnp.float32),),
            -jnp.inf,
            [1e6, 1e5, 1e4],
            [0],
            [synthetic_output_l2_loss],
            initial,
            optimizer,
        )
        result = run_sequential(carry, inputs, 40, chunk_size, optimizer, losses, indexes, 2)
        return result_to_history(result, optimizer), result

    one, one_result = run(1)
    hundred, hundred_result = run(100)
    assert bool(hundred_result.carry.active)
    assert int(hundred_result.carry.executed_steps) == 40
    assert [float(x) for x in one.convergence_thresholds] == [float(x) for x in hundred.convergence_thresholds]
    assert [int(x.step) for x in one.convergence_states] == [int(x.step) for x in hundred.convergence_states]
    assert [float(x.losses.total_train_loss) for x in one.convergence_states] == pytest.approx(
        [float(x.losses.total_train_loss) for x in hundred.convergence_states], rel=1e-6, abs=1e-6
    )


def test_ablation_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        OptimizationConfig(frame_average_impl="bad")
    with pytest.raises(ValueError):
        OptimizationConfig(step_chunk_size=0)
    with pytest.raises(ValueError):
        OptimizationConfig(lr_adjustment="on")
