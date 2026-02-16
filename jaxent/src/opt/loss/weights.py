"""
Parameter loss implementations using factory functions.
These losses operate on model parameters (weights) and priors.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import JaxEnt_Loss
from jaxent.src.opt.loss.base import LossRegistry, create_parameter_loss, register_loss


# Maximum Entropy Losses
def _softmax_cross_entropy_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Softmax cross entropy loss for weight comparison"""
    return jnp.asarray(
        optax.losses.safe_softmax_cross_entropy(jnp.log(model_weights), prior_weights)
    )


def _scaled_convex_kl_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Convex KL divergence loss scaled by number of frames"""
    num_frames = model_weights.shape[0]
    kl_loss = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(model_weights),
        targets=prior_weights,
    )
    return kl_loss / num_frames


def _jsd_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Jensen-Shannon divergence loss"""
    num_frames = model_weights.shape[0]

    # Calculate midpoint distribution M
    M = (model_weights + prior_weights) / 2

    # Compute KL divergences against M
    kl_sim = optax.losses.convex_kl_divergence(log_predictions=jnp.log(M), targets=model_weights)
    kl_prior = optax.losses.convex_kl_divergence(log_predictions=jnp.log(M), targets=prior_weights)

    # JSD is the average of the two KL divergences
    jsd = (kl_sim + kl_prior) / 2
    return jsd / num_frames


def _w1_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Wasserstein-1 (Earth Mover's) distance loss - simplified as L1 from uniform"""
    # Original implementation compares to zero (uniform distribution)
    return jnp.mean(jnp.abs(model_weights - jnp.zeros_like(model_weights)))


def _l1_regularization_loss(model_weights: Array, prior_weights: Array) -> Array:
    """L1 regularization loss between model and prior weights"""
    return jnp.mean(jnp.abs(model_weights - prior_weights))


def _l2_regularization_loss(model_weights: Array, prior_weights: Array) -> Array:
    """L2 regularization loss between model and prior weights"""
    return jnp.mean((model_weights - prior_weights) ** 2)


def _ess_maximization_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Effective Sample Size maximization loss (1 - normalized ESS)"""
    # Calculate ESS
    ess = 1 / jnp.sum(model_weights**2)
    n_frames = model_weights.shape[0]
    ess_scaled = ess / n_frames
    ess_scaled = jnp.clip(ess_scaled, 1e-8, 1.0)

    # Loss is 1 - ESS (we want to minimize this to maximize ESS)
    return 1 - ess_scaled


def _ess_minimization_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Effective Sample Size minimization loss (normalized ESS)"""
    # Calculate ESS
    ess = 1 / jnp.sum(model_weights**2)
    n_frames = model_weights.shape[0]
    ess_scaled = ess / n_frames
    ess_scaled = jnp.clip(ess_scaled, 1e-8, 1.0)

    # Loss is ESS itself (we want to minimize ESS)
    return ess_scaled


# Factory functions for basic parameter losses
@register_loss("max_entropy")
def max_entropy_builder():
    """Maximum entropy loss using softmax cross entropy"""
    return create_parameter_loss(
        loss_fn=_softmax_cross_entropy_loss,
        normalise=True,
        eps=1e-8,
        scale_eps=False,
    )


@register_loss("maxent_convex_kl")
def maxent_convex_kl_builder():
    """Maximum entropy loss using convex KL divergence"""
    return create_parameter_loss(
        loss_fn=_scaled_convex_kl_loss,
        normalise=True,
        eps=1e-10,
        scale_eps=True,
    )


@register_loss("maxent_jsd")
def maxent_jsd_builder():
    """Maximum entropy loss using Jensen-Shannon divergence"""
    return create_parameter_loss(
        loss_fn=_jsd_loss,
        normalise=True,
        eps=1e-3,
        scale_eps=True,
    )


@register_loss("maxent_w1")
def maxent_w1_builder():
    """Maximum entropy loss using Wasserstein-1 distance"""
    return create_parameter_loss(
        loss_fn=_w1_loss,
        normalise=True,
        eps=1e-10,
        scale_eps=False,
    )


@register_loss("maxent_l1")
def maxent_l1_builder():
    """Maximum entropy loss using L1 regularization"""
    return create_parameter_loss(
        loss_fn=_l1_regularization_loss,
        normalise=True,
        eps=1e-10,
        scale_eps=False,
    )


@register_loss("maxent_l2")
def maxent_l2_builder():
    """Maximum entropy loss using L2 regularization"""
    return create_parameter_loss(
        loss_fn=_l2_regularization_loss,
        normalise=True,
        eps=1e-10,
        scale_eps=False,
    )


@register_loss("maxent_ess")
def maxent_ess_builder():
    """Maximum entropy loss using ESS maximization"""
    return create_parameter_loss(
        loss_fn=_ess_maximization_loss,
        normalise=True,
        eps=1e-8,
        scale_eps=False,
    )


@register_loss("minent_ess")
def minent_ess_builder():
    """Minimum entropy loss using ESS minimization"""
    return create_parameter_loss(
        loss_fn=_ess_minimization_loss,
        normalise=True,
        eps=1e-8,
        scale_eps=False,
    )


# Sparse Weight Handling Factory
def create_sparse_parameter_loss(
    loss_fn: Callable,
    normalise: bool = True,
    eps: float = 1e-8,
    scale_eps: bool = False,
) -> JaxEnt_Loss:
    """Create a sparse parameter loss using factory approach"""

    def sparse_parameter_loss(
        model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
    ) -> tuple[Array, Array]:
        # Apply mask to weights
        active_mask = model.params.frame_mask > 0.5

        # Get masked weights
        model_weights = jnp.abs(model.params.frame_weights) * active_mask
        prior_weights = jnp.abs(dataset.frame_weights) * active_mask

        # Normalize with epsilon
        current_eps = jnp.where(scale_eps, eps * jnp.sum(active_mask), eps)

        def _normalise_weights_sparse(weights: jnp.ndarray) -> jnp.ndarray:
            weights = jnp.abs(weights) + current_eps
            normalized = weights / jnp.sum(weights)
            return jnp.where(jnp.asarray(normalise), normalized, weights)

        model_weights = _normalise_weights_sparse(model_weights)
        prior_weights = _normalise_weights_sparse(prior_weights)

        # Apply loss function
        loss = loss_fn(model_weights, prior_weights)
        return loss, loss

    return sparse_parameter_loss


@register_loss("sparse_max_entropy")
def sparse_max_entropy_builder():
    """Sparse maximum entropy loss using masked weights"""
    return create_sparse_parameter_loss(
        loss_fn=_softmax_cross_entropy_loss,
        normalise=True,
        eps=1e-8,
        scale_eps=False,
    )


# Mask Regularization Loss Factory
def create_mask_l0_loss() -> JaxEnt_Loss:
    """Create mask L0 loss function using factory approach"""

    def mask_l0_loss(
        model: InitialisedSimulation,
        dataset: Simulation_Parameters,
        prediction_index: None,
    ) -> tuple[Array, Array]:
        # L0 penalty - sum of mask values (encourages sparsity)
        frame_masks = model.params.frame_mask
        loss = jnp.sum(frame_masks)
        return loss, loss

    return mask_l0_loss


@register_loss("mask_l0")
def create_mask_l0_loss_factory():
    """Mask L0 regularization loss factory"""
    return create_mask_l0_loss()


# Custom Normalization Factory
def create_custom_normalized_loss(
    loss_fn: Callable,
    normalization_strategy: str = "standard",
    normalise: bool = True,
    eps: float = 1e-8,
    scale_eps: bool = False,
) -> JaxEnt_Loss:
    """Create parameter loss with custom normalization strategy"""

    def custom_normalized_loss(
        model: InitialisedSimulation,
        dataset: Simulation_Parameters,
        prediction_index: None,
    ) -> tuple[Array, Array]:
        def _normalise_weights_custom(weights: jnp.ndarray) -> jnp.ndarray:
            """Custom weight normalization based on strategy"""
            if not normalise:
                return weights

            if normalization_strategy == "standard":
                # Standard normalization: abs + eps, then divide by sum
                weights = jnp.abs(weights) + eps
                return weights / jnp.sum(weights)
            elif normalization_strategy == "softmax":
                # Softmax normalization
                return jax.nn.softmax(weights)
            elif normalization_strategy == "l2":
                # L2 normalization
                weights = jnp.abs(weights) + eps
                return weights / jnp.sqrt(jnp.sum(weights**2))
            else:
                # Fallback to standard
                weights = jnp.abs(weights) + eps
                return weights / jnp.sum(weights)

        model_weights = _normalise_weights_custom(model.params.frame_weights)
        prior_weights = _normalise_weights_custom(dataset.frame_weights)

        loss = loss_fn(model_weights, prior_weights)
        return loss, loss

    return custom_normalized_loss


@register_loss("maxent_softmax_kl")
def create_maxent_softmax_kl_loss():
    """Maximum entropy loss with softmax normalization and KL divergence"""
    return create_custom_normalized_loss(
        loss_fn=_scaled_convex_kl_loss,
        normalization_strategy="softmax",
        normalise=True,
        eps=1e-8,
    )


# Composite Parameter Loss Factory
def create_composite_parameter_loss(
    loss_components: list[tuple[Callable, float]],
    normalise: bool = True,
    eps: float = 1e-8,
    scale_eps: bool = False,
) -> JaxEnt_Loss:
    """Create composite parameter loss combining multiple loss functions"""

    def composite_loss_fn(model_weights: Array, prior_weights: Array) -> Array:
        """Combine multiple loss components with weights"""
        total_loss = 0.0
        for loss_fn, weight in loss_components:
            component_loss = loss_fn(model_weights, prior_weights)
            total_loss += weight * component_loss
        return total_loss

    return create_parameter_loss(
        loss_fn=composite_loss_fn,
        normalise=normalise,
        eps=eps,
        scale_eps=scale_eps,
    )


def create_composite_entropy_loss(kl_weight: float = 1.0, l2_weight: float = 0.1):
    """Factory for creating composite entropy loss with KL + L2 regularization"""
    return create_composite_parameter_loss(
        loss_components=[(_scaled_convex_kl_loss, kl_weight), (_l2_regularization_loss, l2_weight)],
        normalise=True,
        eps=1e-8,
    )


@register_loss("composite_kl_l2")
def create_composite_kl_l2_loss():
    """Composite loss combining KL divergence and L2 regularization"""
    return create_composite_entropy_loss(1.0, 0.1)


# Additional composite loss variants
@register_loss("composite_kl_l1")
def create_composite_kl_l1_loss():
    """Composite loss combining KL divergence and L1 regularization"""
    return create_composite_parameter_loss(
        loss_components=[(_scaled_convex_kl_loss, 1.0), (_l1_regularization_loss, 0.1)],
        normalise=True,
        eps=1e-8,
    )


@register_loss("composite_entropy_ess")
def create_composite_entropy_ess_loss():
    """Composite loss combining entropy and ESS maximization"""
    return create_composite_parameter_loss(
        loss_components=[(_softmax_cross_entropy_loss, 1.0), (_ess_maximization_loss, 0.5)],
        normalise=True,
        eps=1e-8,
    )


# Convenience function to register custom losses
def register_parameter_loss(
    name: str,
    loss_fn: Callable,
    normalise: bool = True,
    eps: float = 1e-8,
    scale_eps: bool = False,
):
    """Register a custom parameter loss with the registry"""

    def factory():
        return create_parameter_loss(
            loss_fn=loss_fn,
            normalise=normalise,
            eps=eps,
            scale_eps=scale_eps,
        )

    LossRegistry.register(name, factory)


# Example of registering a custom loss
def _custom_squared_difference_loss(model_weights: Array, prior_weights: Array) -> Array:
    """Custom squared difference loss"""
    return jnp.sum((model_weights - prior_weights) ** 2)


register_parameter_loss(
    "custom_squared_diff",
    _custom_squared_difference_loss,
    normalise=True,
    eps=1e-10,
)


if __name__ == "__main__":
    print("Available parameter losses:", LossRegistry.list_losses())
