"""
Consistency loss implementations using the builder pattern.
These losses compare relationships between structure similarities and weight similarities.
"""

import jax.numpy as jnp
import optax

from jaxent.src.opt.loss.base import create_consistency_loss, register_loss


# Basic L2 Consistency Loss
@register_loss("frame_weight_consistency")
def frame_weight_consistency_builder():
    """Basic L2 consistency loss between weight and structure similarities"""
    return create_consistency_loss(
        loss_fn=lambda w, s: jnp.mean((jnp.abs(s - w)) ** 2),
        normalize_upper_tri=False,
    )


# L1 Consistency Loss
@register_loss("l1_frame_weight_consistency")
def l1_frame_weight_consistency_builder():
    """L1 consistency loss between weight and structure similarities"""
    return create_consistency_loss(
        loss_fn=lambda w, s: jnp.mean(jnp.abs(s - w)),
        normalize_upper_tri=False,
    )


# Exponential Consistency Loss
def _exponential_l2_loss(weight_sim: jnp.ndarray, struct_sim: jnp.ndarray) -> jnp.ndarray:
    """Exponential transformation of L2 distance: -1 + exp((abs(diff))^2)"""
    l2_distance = jnp.abs(struct_sim - weight_sim)
    return jnp.mean(-1 + jnp.exp(l2_distance**2))


@register_loss("exp_frame_weight_consistency")
def exp_frame_weight_consistency_builder():
    """Exponential consistency loss with L2 base distance"""
    return create_consistency_loss(
        loss_fn=_exponential_l2_loss,
        normalize_upper_tri=False,
    )


# Normalized (Mean-Centered) Consistency Loss
def _mean_center_transform_consistency(array: jnp.ndarray) -> jnp.ndarray:
    """Mean center the array for consistency losses"""
    return array - jnp.mean(array)


def _normalized_exponential_loss(weight_sim: jnp.ndarray, struct_sim: jnp.ndarray) -> jnp.ndarray:
    """Mean-centered exponential consistency loss"""
    # Mean center both arrays
    weight_centered = weight_sim - jnp.mean(weight_sim)
    struct_centered = struct_sim - jnp.mean(struct_sim)

    # Compute exponential loss
    l2_distance = (jnp.abs(struct_centered - weight_centered)) ** 2
    return jnp.mean(-1 + jnp.exp(l2_distance))


@register_loss("normalised_frame_weight_consistency")
def normalised_frame_weight_consistency_builder():
    """Normalized (mean-centered) exponential consistency loss"""
    return create_consistency_loss(
        loss_fn=_normalized_exponential_loss,
        normalize_upper_tri=False,
    )


# Convex KL Consistency Loss
def _convex_kl_consistency_loss(weight_sim: jnp.ndarray, struct_sim: jnp.ndarray) -> jnp.ndarray:
    """Convex KL divergence consistency loss with probability normalization"""
    epsilon = 1e-5

    # Normalize weights to probability distributions
    weight_norm = weight_sim + epsilon
    weight_probs = weight_norm / jnp.sum(weight_norm)

    struct_norm = struct_sim + epsilon
    struct_probs = struct_norm / jnp.sum(struct_norm)

    # Compute KL divergence
    kl_loss = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(weight_probs),
        targets=struct_probs,
    )

    # Apply exponential transformation as in original
    return -1 + jnp.exp(jnp.sum(kl_loss))


@register_loss("convex_kl_frame_weight_consistency")
def convex_kl_frame_weight_consistency_builder():
    """Convex KL divergence consistency loss"""
    return create_consistency_loss(
        loss_fn=_convex_kl_consistency_loss,
        normalize_upper_tri=False,
    )


# Correlation-Based Consistency Loss
def _correlation_consistency_loss(weight_sim: jnp.ndarray, struct_sim: jnp.ndarray) -> jnp.ndarray:
    """Correlation-based consistency loss (1 - correlation coefficient)"""
    # Compute means
    weight_mean = jnp.mean(weight_sim)
    struct_mean = jnp.mean(struct_sim)

    # Center the data
    weight_centered = weight_sim - weight_mean
    struct_centered = struct_sim - struct_mean

    # Compute correlation coefficient
    numerator = jnp.sum(weight_centered * struct_centered)
    denominator = jnp.sqrt(jnp.sum(weight_centered**2) * jnp.sum(struct_centered**2) + 1e-12)

    correlation = numerator / denominator
    correlation = jnp.clip(correlation, -1.0, 1.0)

    # Convert to distance (1 - correlation)
    return 1.0 - correlation


@register_loss("corr_frame_weight_consistency")
def corr_frame_weight_consistency_builder():
    """Correlation-based consistency loss"""
    return create_consistency_loss(
        loss_fn=_correlation_consistency_loss,
        normalize_upper_tri=False,
    )


# Cosine-Based Consistency Loss
def jax_pairwise_cosine_similarity(array1: jnp.ndarray, array2: jnp.ndarray) -> jnp.ndarray:
    """Calculates the pairwise cosine similarity between two arrays."""
    # Center both vectors
    array1_mean = jnp.mean(array1)
    array2_mean = jnp.mean(array2)
    array1_centered = array1 - array1_mean
    array2_centered = array2 - array2_mean
    numerator = jnp.sum(array1_centered * array2_centered)
    denominator = jnp.sqrt(jnp.sum(array1_centered**2) * jnp.sum(array2_centered**2) + 1e-12)
    cos_sim = numerator / denominator
    return jnp.clip(cos_sim, -1.0, 1.0)


def _cosine_consistency_loss(weight_sim: jnp.ndarray, struct_sim: jnp.ndarray) -> jnp.ndarray:
    """Cosine-based consistency loss (1 - cosine similarity)"""
    cos_sim = jax_pairwise_cosine_similarity(weight_sim, struct_sim)
    return 1.0 - cos_sim


@register_loss("cosine_frame_weight_consistency")
def cosine_frame_weight_consistency_builder():
    """Cosine-based consistency loss"""
    return create_consistency_loss(
        loss_fn=_cosine_consistency_loss,
        normalize_upper_tri=False,
    )
