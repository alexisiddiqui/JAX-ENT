"""
Functional loss implementations using the builder pattern.
These losses operate on model outputs (predictions vs targets).
"""

import jax
import jax.numpy as jnp
import optax
from jax import Array

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.opt.loss.base import apply_transforms, create_functional_loss, register_loss


# HDX Protection Factor Losses
@register_loss("hdx_pf_l2")
def hdx_pf_l2_builder():
    """HDX Protection Factor L2 loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2), flatten=True
    )


@register_loss("hdx_pf_mae")
def hdx_pf_mae_builder():
    """HDX Protection Factor MAE loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean(jnp.abs(pred - target)), flatten=True
    )


# HDX Uptake Standard Losses
@register_loss("hdx_uptake_l1")
def hdx_uptake_l1_builder():
    """HDX Uptake L1 loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean(jnp.abs(pred - target)), flatten=False
    )


@register_loss("hdx_uptake_l2")
def hdx_uptake_l2_builder():
    """HDX Uptake L2 loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2), flatten=False
    )


@register_loss("hdx_uptake_mae")
def hdx_uptake_mae_builder():
    """HDX Uptake MAE loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean(jnp.abs(pred - target)),
        flatten=False,
        post_mean=True,
    )


@register_loss("hdx_uptake_mse")
def hdx_uptake_mse_builder():
    """HDX Uptake MSE loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2),
        flatten=False,
        post_mean=True,
    )


# Mean-Centered Losses
def _mean_center_transform(array: jnp.ndarray) -> jnp.ndarray:
    """Transform that centers array around its mean"""
    return array - jnp.mean(array)


@register_loss("hdx_uptake_mean_centred_l1")
def hdx_uptake_mean_centred_l1_builder():
    """HDX Uptake Mean-Centered L1 loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean(jnp.abs(pred - target)),
        transform_chain=[_mean_center_transform],
        flatten=False,
        post_mean=False,  # Already handled in original implementation
    )


@register_loss("hdx_uptake_mean_centred_l2")
def hdx_uptake_mean_centred_l2_builder():
    """HDX Uptake Mean-Centered L2 loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2),
        transform_chain=[_mean_center_transform],
        flatten=False,
        post_mean=False,
    )


@register_loss("hdx_uptake_mean_centred_mse")
def hdx_uptake_mean_centred_mse_builder():
    """HDX Uptake Mean-Centered MSE loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2),
        transform_chain=[_mean_center_transform],
        flatten=False,
        post_mean=True,
    )


@register_loss("hdx_uptake_mean_centred_mae")
def hdx_uptake_mean_centred_mae_builder():
    """HDX Uptake Mean-Centered MAE loss builder"""
    return create_functional_loss(
        loss_fn=lambda pred, target: jnp.mean(jnp.abs(pred - target)),
        transform_chain=[_mean_center_transform],
        flatten=False,
        post_mean=True,
    )


# Special HDX Losses
def _hdx_absolute_loss_fn_jax_compatible(
    model: InitialisedSimulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    predictions = model.outputs[prediction_index]
    sparse_mapping = dataset.train.residue_feature_ouput_mapping
    y_true = dataset.train.y_true

    def scan_body(total_loss, timepoint_idx):
        true_uptake_timepoint = y_true[timepoint_idx, :]
        pred_uptake_timepoint = predictions.uptake[timepoint_idx]

        pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
        true_mapped = true_uptake_timepoint

        pred_mean = jnp.mean(pred_mapped)
        true_mean = jnp.mean(true_mapped)

        timepoint_loss = jnp.abs(pred_mean - true_mean)
        # Ensure division by a JAX array
        divisor = jnp.where(
            true_uptake_timepoint.shape[0] > 0,
            jnp.asarray(true_uptake_timepoint.shape[0]),
            jnp.asarray(1.0),
        )
        total_loss += timepoint_loss / divisor
        return total_loss, None

    # Initialize total_loss as a JAX array
    initial_total_loss = jnp.asarray(0.0)
    final_total_loss, _ = jax.lax.scan(scan_body, initial_total_loss, jnp.arange(y_true.shape[0]))

    # For validation, we can use the same logic or a simplified version
    # For now, let's assume train and val losses are the same for this specific loss type
    return final_total_loss, final_total_loss


@register_loss("hdx_uptake_abs")
def hdx_uptake_abs_builder():
    """HDX Uptake Absolute difference of means loss builder"""
    return _hdx_absolute_loss_fn_jax_compatible


# KL Divergence Losses
def _kl_divergence_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """KL divergence loss for probability distributions"""
    epsilon = 1e-5

    # Normalize to create proper probability distributions
    target_norm = target + epsilon
    target_probs = target_norm / jnp.sum(target_norm)

    pred_norm = pred + epsilon
    pred_probs = pred_norm / jnp.sum(pred_norm)
    log_pred = jnp.log(pred_probs)

    return jnp.sum(optax.losses.kl_divergence(log_predictions=log_pred, targets=target_probs))


def _convex_kl_divergence_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Convex KL divergence loss for probability distributions"""
    epsilon = 1e-5

    # Normalize to create proper probability distributions
    target_norm = target + epsilon
    target_probs = target_norm / jnp.sum(target_norm)

    pred_norm = pred + epsilon
    pred_probs = pred_norm / jnp.sum(pred_norm)
    log_pred = jnp.log(pred_probs)

    return jnp.sum(
        optax.losses.convex_kl_divergence(log_predictions=log_pred, targets=target_probs)
    )


@register_loss("hdx_uptake_kl")
def hdx_uptake_kl_builder():
    """HDX Uptake KL divergence loss builder"""
    return create_functional_loss(loss_fn=_kl_divergence_loss, flatten=False, post_mean=False)


@register_loss("hdx_uptake_convex_kl")
def hdx_uptake_convex_kl_builder():
    """HDX Uptake Convex KL divergence loss builder"""
    return create_functional_loss(
        loss_fn=_convex_kl_divergence_loss, flatten=False, post_mean=False
    )


# HDXer-style Losses (with exponential transforms)
def _hdxer_transform(loss_value: jnp.ndarray) -> jnp.ndarray:
    """Transform used in HDXer losses: 1 - exp(-loss/2)"""
    return 1 - jnp.exp(-loss_value / 2)


def _hdxer_loss_fn_jax_compatible(
    model: InitialisedSimulation,
    dataset: ExpD_Dataloader,
    prediction_index: int,
    transform_chain: list,
) -> tuple[Array, Array]:
    predictions = model.outputs[prediction_index]
    sparse_mapping = dataset.train.residue_feature_ouput_mapping
    y_true = dataset.train.y_true

    # Ensure y_true is at least 2D for consistent indexing
    if y_true.ndim == 1:
        y_true = jnp.expand_dims(y_true, axis=0)

    def scan_body(total_loss, timepoint_idx):
        true_dfracs_timepoint = y_true[timepoint_idx, :]
        pred_uptake_timepoint = predictions.uptake[timepoint_idx]

        pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
        true_mapped = true_dfracs_timepoint

        # Apply transforms if any
        pred_transformed = apply_transforms(pred_mapped, transform_chain)
        true_transformed = apply_transforms(true_mapped, transform_chain)

        # Compute squared differences and sum (not mean)
        squared_diff = (pred_transformed - true_transformed) ** 2
        timepoint_loss = jnp.sum(squared_diff) / 2
        total_loss += timepoint_loss
        return total_loss, None

    initial_total_loss = jnp.asarray(0.0)
    final_total_loss, _ = jax.lax.scan(scan_body, initial_total_loss, jnp.arange(y_true.shape[0]))

    # Apply HDXer exponential transform
    final_total_loss = _hdxer_transform(final_total_loss)

    return final_total_loss, final_total_loss


@register_loss("hdxer_mse")
def hdxer_mse_builder():
    """HDXer MSE loss builder"""
    return lambda model, dataset, prediction_index: _hdxer_loss_fn_jax_compatible(
        model, dataset, prediction_index, []
    )


@register_loss("hdxer_mc_mse")
def hdxer_mc_mse_builder():
    """HDXer Mean-Centered MSE loss builder"""
    return lambda model, dataset, prediction_index: _hdxer_loss_fn_jax_compatible(
        model, dataset, prediction_index, [_mean_center_transform]
    )


# Monotonicity Loss
def _monotonicity_loss_fn(
    model: InitialisedSimulation, dataset, prediction_index: int
) -> tuple[Array, Array]:
    predictions = model.outputs[prediction_index]

    # Get uptake predictions
    deut = jnp.array(predictions.uptake)

    # Calculate differences between adjacent timepoints
    time_diffs = deut[:, 1:] - deut[:, :-1]

    # Penalize negative differences (violations of monotonicity)
    violations = jnp.maximum(-time_diffs, 0)

    # Square violations and take mean
    loss = jnp.where(time_diffs.size > 0, jnp.mean(violations**2), jnp.array(0.0))

    return loss, loss


@register_loss("hdx_uptake_monotonicity")
def hdx_uptake_monotonicity_builder():
    """HDX Uptake Monotonicity loss builder"""
    return _monotonicity_loss_fn


# Vectorized implementations for efficiency
@register_loss("hdx_uptake_mae_vectorized")
def hdx_uptake_mae_vectorized_builder():
    """Vectorized HDX Uptake MAE loss builder"""

    def _vectorized_mae_loss_fn(
        model: InitialisedSimulation, dataset: ExpD_Dataloader, prediction_index: int
    ) -> tuple[Array, Array]:
        predictions = model.outputs[prediction_index]

        def compute_loss_vectorized(sparse_mapping, y_true):
            def loop_body(timepoint_idx, carry):
                total_loss = carry

                true_uptake_timepoint = y_true[timepoint_idx, :]
                pred_uptake_timepoint = predictions.uptake[timepoint_idx]

                pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)

                if len(true_uptake_timepoint.shape) > 1:
                    true_uptake_timepoint = jnp.squeeze(true_uptake_timepoint)

                min_len = min(pred_mapped.shape[0], true_uptake_timepoint.shape[0])
                pred_mapped = pred_mapped[:min_len]
                true_uptake_timepoint = true_uptake_timepoint[:min_len]

                timepoint_loss = jnp.mean(jnp.abs(pred_mapped - true_uptake_timepoint))
                return total_loss + timepoint_loss

            n_timepoints = min(predictions.uptake.shape[0], y_true.shape[0])
            total_loss = jax.lax.fori_loop(0, n_timepoints, loop_body, 0.0)

            return total_loss / n_timepoints

        train_loss = compute_loss_vectorized(
            dataset.train.residue_feature_ouput_mapping, dataset.train.y_true
        )
        val_loss = compute_loss_vectorized(
            dataset.val.residue_feature_ouput_mapping, dataset.val.y_true
        )

        return train_loss, val_loss

    return _vectorized_mae_loss_fn
