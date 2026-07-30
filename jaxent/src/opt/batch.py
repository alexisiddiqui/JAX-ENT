from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import (
    BatchOptimisationResult,
    HParamBatch,
    JaxEnt_Loss,
    OptimizationState,
)
from jaxent.src.opt.chunk import ChunkInputs, ChunkResult, run_batch as run_chunk_batch
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _build_chunk_state, _snapshot_to_state, result_to_history


def _pad_array(values: Array, n_pad: int) -> Array:
    if n_pad == 0:
        return values
    pad_values = jnp.repeat(values[-1:], n_pad, axis=0)
    return jnp.concatenate([values, pad_values], axis=0)


def _reshape_for_batches(values: Array, n_batches: int, batch_size: int) -> Array:
    return values.reshape((n_batches, batch_size) + values.shape[1:])


def _replace_hparams(
    params: Simulation_Parameters,
    forward_model_weights: Array,
    forward_model_scaling: Array,
) -> Simulation_Parameters:
    return Simulation_Parameters(
        frame_weight_logits=params.frame_weight_logits,
        model_parameters=params.model_parameters,
        normalise_loss_functions=params.normalise_loss_functions,
        forward_model_weights=forward_model_weights,
        forward_model_scaling=forward_model_scaling,
    )


def batch_optimise(
    simulation: Simulation,
    hparam_batch: HParamBatch,
    batch_size: int,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    config: OptimiserSettings,
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    optimizer: Optional[OptaxOptimizer] = None,
    optimisable_funcs: list[bool] | Array | None = None,
) -> BatchOptimisationResult:
    """Run hyperparameter sweeps in vmapped batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if hparam_batch.forward_model_weights.shape[0] != hparam_batch.forward_model_scaling.shape[0]:
        raise ValueError("forward_model_weights and forward_model_scaling must have the same n_hparams")
    if hparam_batch.learning_rate is not None and (
        hparam_batch.learning_rate.shape[0] != hparam_batch.forward_model_weights.shape[0]
    ):
        raise ValueError("learning_rate must match n_hparams when provided")

    if not hasattr(simulation, "_input_features"):
        if not simulation.initialise():
            raise ValueError("Failed to initialise simulation before batch optimisation")

    if optimizer is None:
        optimizer = OptaxOptimizer(
            learning_rate=config.learning_rate,
            optimizer=config.optimiser_type,
        )

    base_opt_state = optimizer.initialise(
        simulation,
        optimisable_funcs=optimisable_funcs,
    )
    base_dynamic, base_aux = simulation.tree_flatten()
    _, _, base_input_features = base_dynamic

    n_hparams = hparam_batch.forward_model_weights.shape[0]
    n_pad = (-n_hparams) % batch_size
    n_total = n_hparams + n_pad
    n_batches = n_total // batch_size

    padded_weights = _pad_array(hparam_batch.forward_model_weights, n_pad)
    padded_scaling = _pad_array(hparam_batch.forward_model_scaling, n_pad)
    if hparam_batch.learning_rate is None:
        padded_lr = jnp.full((n_total,), config.learning_rate, dtype=jnp.float32)
    else:
        padded_lr = _pad_array(hparam_batch.learning_rate, n_pad)

    batched_weights = _reshape_for_batches(padded_weights, n_batches, batch_size)
    batched_scaling = _reshape_for_batches(padded_scaling, n_batches, batch_size)
    batched_lr = _reshape_for_batches(padded_lr, n_batches, batch_size)

    def make_single(
        forward_model_weights: Array,
        forward_model_scaling: Array,
        learning_rate: Array,
    ):
        run_params = _replace_hparams(
            base_opt_state.params,
            forward_model_weights=forward_model_weights,
            forward_model_scaling=forward_model_scaling,
        )
        run_sim = Simulation.tree_unflatten(base_aux, (run_params, tuple(), base_input_features))
        run_state = OptimizationState(
            params=run_params,
            opt_state=base_opt_state.opt_state,
            step=0,
            losses=None,
            gradients=base_opt_state.gradients,
        )
        carry, inputs, tuple_loss_functions, tuple_indexes = _build_chunk_state(
            run_sim,
            data_to_fit=data_to_fit,
            tolerance=config.tolerance,
            convergence=config.convergence,
            indexes=indexes,
            loss_functions=loss_functions,
            opt_state=run_state,
            optimizer=optimizer,
            learning_rate=learning_rate,
            convergence_parameter_partitions=config.parameter_partitions,
            retain_convergence_params=config.save_convergence,
        )
        return (
            carry,
            inputs.convergence_thresholds,
            inputs.tolerance,
            jnp.asarray(config.ema_alpha, dtype=jnp.float32),
        )

    def run_group(batch_inputs):
        batch_weights, batch_scaling, batch_lr = batch_inputs
        (
            carries,
            thresholds,
            tolerances,
            ema_alphas,
        ) = jax.vmap(make_single, in_axes=(0, 0, 0))(
            batch_weights,
            batch_scaling,
            batch_lr,
        )
        inputs = ChunkInputs(
            data_targets=(),
            convergence_thresholds=thresholds,
            tolerance=tolerances,
            ema_alpha=ema_alphas,
        )
        return run_chunk_batch(
            carries,
            inputs,
            tuple(data_to_fit),
            config.n_steps,
            config.step_chunk_size,
            optimizer,
            tuple(loss_functions),
            tuple(indexes),
            config.min_steps_per_threshold,
            config.parameter_partitions,
            config.save_states,
            config.reset_threshold_cooldown_on_oscillation,
        )

    batched_results = jax.lax.map(run_group, (batched_weights, batched_scaling, batched_lr))
    flat_results = jax.tree_util.tree_map(
        lambda x: x.reshape((n_total,) + x.shape[2:])[:n_hparams],
        batched_results,
    )

    histories = []
    best_states: list[OptimizationState] = []
    convergence_steps = []
    for run_idx in range(n_hparams):
        run_result = ChunkResult(
            carry=jax.tree_util.tree_map(lambda x: x[run_idx], flat_results.carry),
            records=jax.tree_util.tree_map(lambda x: x[run_idx], flat_results.records),
            metrics=jax.tree_util.tree_map(lambda x: x[run_idx], flat_results.metrics),
        )
        history = result_to_history(
            run_result,
            optimizer,
            save_states=config.save_states,
            save_convergence=config.save_convergence,
            save_best=config.save_best,
            state_parameter_partitions=config.parameter_partitions,
        )
        convergence_steps.append(run_result.carry.executed_steps)
        best_states.append(_snapshot_to_state(run_result.carry.best, run_result.carry.opt_state))
        histories.append(history)

    result_hparam_batch = HParamBatch(
        forward_model_weights=hparam_batch.forward_model_weights,
        forward_model_scaling=hparam_batch.forward_model_scaling,
        learning_rate=hparam_batch.learning_rate,
    )
    return BatchOptimisationResult(
        histories=tuple(histories),
        best_states=tuple(best_states),
        convergence_steps=jnp.stack(convergence_steps).astype(jnp.int32),
        hparam_batch=result_hparam_batch,
    )
