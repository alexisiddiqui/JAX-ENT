import jax.numpy as jnp
import pytest

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import (
    _pad_array,
    _replace_hparams,
    _reshape_for_batches,
    batch_optimise,
)
from jaxent.tests.modules.optimise.test_module_optimise_convergence import (
    _create_synthetic_simulation,
    synthetic_output_l2_loss,
)


def _make_config() -> OptimiserSettings:
    return OptimiserSettings(
        name="unit_batch",
        n_steps=4,
        tolerance=1e-8,
        convergence=200.0,
        learning_rate=0.1,
    )


def test_pad_array_noop_when_n_pad_is_zero() -> None:
    values = jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    padded = _pad_array(values, 0)
    assert padded.shape == values.shape
    assert jnp.array_equal(padded, values)


def test_pad_array_repeats_last_row() -> None:
    values = jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    padded = _pad_array(values, 2)
    expected = jnp.asarray([[1.0], [2.0], [3.0], [3.0], [3.0]], dtype=jnp.float32)
    assert padded.shape == (5, 1)
    assert jnp.array_equal(padded, expected)


def test_reshape_for_batches_preserves_row_order() -> None:
    values = jnp.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=jnp.float32)
    reshaped = _reshape_for_batches(values, n_batches=2, batch_size=2)
    expected = jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=jnp.float32)
    assert reshaped.shape == (2, 2, 1)
    assert jnp.array_equal(reshaped, expected)


def test_replace_hparams_only_changes_target_fields() -> None:
    simulation, _ = _create_synthetic_simulation()
    original = simulation.params
    new_weights = jnp.asarray([0.25], dtype=jnp.float32)
    new_scaling = jnp.asarray([0.5], dtype=jnp.float32)

    updated = _replace_hparams(original, new_weights, new_scaling)

    assert jnp.array_equal(updated.forward_model_weights, new_weights)
    assert jnp.array_equal(updated.forward_model_scaling, new_scaling)
    assert jnp.array_equal(updated.frame_weights, original.frame_weights)
    assert jnp.array_equal(updated.frame_mask, original.frame_mask)
    assert updated.model_parameters == original.model_parameters
    assert jnp.array_equal(
        updated.normalise_loss_functions,
        original.normalise_loss_functions,
    )


def test_batch_optimise_rejects_non_positive_batch_size() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparams = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.ones((3, 1), dtype=jnp.float32),
        learning_rate=jnp.ones((3,), dtype=jnp.float32) * config.learning_rate,
    )

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        batch_optimise(
            simulation=simulation,
            hparam_batch=hparams,
            batch_size=0,
            data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
            config=config,
            indexes=[0],
            loss_functions=[synthetic_output_l2_loss],
        )


def test_batch_optimise_rejects_mismatched_hparam_lengths() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparams = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.ones((2, 1), dtype=jnp.float32),
        learning_rate=jnp.ones((3,), dtype=jnp.float32) * config.learning_rate,
    )

    with pytest.raises(
        ValueError,
        match="forward_model_weights and forward_model_scaling must have the same n_hparams",
    ):
        batch_optimise(
            simulation=simulation,
            hparam_batch=hparams,
            batch_size=2,
            data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
            config=config,
            indexes=[0],
            loss_functions=[synthetic_output_l2_loss],
        )


def test_batch_optimise_rejects_learning_rate_length_mismatch() -> None:
    simulation, _ = _create_synthetic_simulation()
    config = _make_config()
    hparams = HParamBatch(
        forward_model_weights=jnp.ones((3, 1), dtype=jnp.float32),
        forward_model_scaling=jnp.ones((3, 1), dtype=jnp.float32),
        learning_rate=jnp.ones((2,), dtype=jnp.float32) * config.learning_rate,
    )

    with pytest.raises(
        ValueError,
        match="learning_rate must match n_hparams when provided",
    ):
        batch_optimise(
            simulation=simulation,
            hparam_batch=hparams,
            batch_size=2,
            data_to_fit=(jnp.asarray([10.0], dtype=jnp.float32),),
            config=config,
            indexes=[0],
            loss_functions=[synthetic_output_l2_loss],
        )
