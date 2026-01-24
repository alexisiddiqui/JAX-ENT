from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import JaxEnt_Loss

# Type aliases
PureLossFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
TransformFunction = Callable[[jnp.ndarray], jnp.ndarray]


def apply_transforms(array: jnp.ndarray, transform_chain: list) -> jnp.ndarray:
    """Apply transforms using JAX-compatible operations."""
    for transform in transform_chain:
        array = transform(array)
    return array


def apply_post_processing(loss: jnp.ndarray, length: int, post_mean: bool) -> jnp.ndarray:
    """Apply post-processing using JAX-compatible operations."""
    # Convert to JAX arrays for safe boolean operations
    length_scalar = jnp.asarray(length)
    post_mean_array = jnp.asarray(post_mean)
    cond = jnp.logical_and(post_mean_array, length_scalar > 0)
    return jnp.where(cond, loss / length_scalar, loss)


def normalize_weights(
    weights: jnp.ndarray, normalise: bool, eps: float, scale_eps: bool
) -> jnp.ndarray:
    """Normalize weights using JAX-compatible operations."""
    # Calculate epsilon value using JAX-compatible operations
    scale_eps_array = jnp.asarray(scale_eps)
    eps_value = jnp.where(scale_eps_array, eps * weights.shape[0], eps)

    # Use JAX-compatible conditional logic
    normalise_array = jnp.asarray(normalise)
    abs_weights = jnp.abs(weights)

    # Use jnp.where to conditionally add eps_value
    weights_with_eps = jnp.where(normalise_array, abs_weights + eps_value, abs_weights)

    # Use jnp.where for normalization logic
    sum_weights = jnp.sum(weights_with_eps)
    normalized = weights_with_eps / sum_weights

    return jnp.where(normalise_array, normalized, weights)


def pairwise_cosine_similarity(array: Array) -> Array:
    """Calculate pairwise cosine similarity using JAX operations."""
    # Handle empty arrays first
    if array.size == 0:
        return jnp.empty((0, 0))

    # Reshape if needed
    array = jnp.reshape(array, (array.shape[0], -1)) if array.ndim == 1 else array

    # Compute similarity
    dot_products = jnp.matmul(array, array.T)
    norms = jnp.linalg.norm(array, axis=1)
    norm_products = jnp.outer(norms, norms)
    norm_products = jnp.maximum(norm_products, 1e-8)
    similarity_matrix = dot_products / norm_products
    return 1 + jnp.clip(similarity_matrix, -1.0, 1.0)


def extract_upper_triangle(matrix: Array) -> Array:
    """Extract upper triangular elements using JAX operations."""
    n = matrix.shape[0]
    if n == 0:
        return jnp.array([])
    rows, cols = jnp.triu_indices(n, k=1)
    return matrix[rows, cols]


def normalize_upper_triangle(upper_tri: Array, normalize: bool) -> Array:
    """Normalize upper triangle using JAX operations."""
    normalize_array = jnp.asarray(normalize)
    # Check if we should normalize and array is not empty
    should_normalize = jnp.logical_and(normalize_array, jnp.asarray(upper_tri.size) > 0)
    normalized = jax.nn.softmax(upper_tri)
    return jnp.where(should_normalize, normalized, upper_tri)


# Also ensure the _as_python_scalar_if_scalar function handles edge cases properly
def _as_python_scalar_if_scalar(x):
    """Convert scalar JAX arrays to Python floats for legacy compatibility"""
    arr = jnp.asarray(x)
    # Only convert to Python scalar if we're not in a traced context
    try:
        # Use JAX-compatible check for scalar arrays
        if arr.ndim == 0 and hasattr(arr, "item"):
            return arr.item()
        else:
            return x
    except Exception:
        # If we're in a traced context, just return the JAX array
        return x


def create_functional_loss(
    loss_fn: PureLossFunction,
    transform_chain: list = None,
    post_mean: bool = True,
    flatten: bool = True,
) -> JaxEnt_Loss:
    """Create a functional loss without builders."""

    transform_chain = transform_chain or []

    # @functools.partial(jax.jit, static_argnames=["prediction_index"])
    def functional_loss(
        model: InitialisedSimulation, dataset: ExpD_Dataloader, prediction_index: int
    ) -> tuple[Array, Array]:
        # Direct indexing with static prediction_index - no need for jax.lax.switch
        predictions = model.outputs[prediction_index]

        def compute_loss(sparse_mapping, y_true):
            if flatten:
                # Protection factors
                pred_values = jnp.array(predictions.log_Pf).reshape(-1)
                pred_mapped = apply_sparse_mapping(sparse_mapping, pred_values)
                true_values = y_true.reshape(-1)

                pred_transformed = apply_transforms(pred_mapped, transform_chain)
                true_transformed = apply_transforms(true_values, transform_chain)

                loss = loss_fn(pred_transformed, true_transformed)
                return apply_post_processing(loss, true_values.shape[0], post_mean)
            else:
                # Uptake data
                total_loss = 0.0
                for timepoint_idx in range(y_true.shape[0]):
                    true_uptake = y_true[timepoint_idx, :]
                    pred_uptake = predictions.uptake[timepoint_idx]
                    pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake)

                    pred_transformed = apply_transforms(pred_mapped, transform_chain)
                    true_transformed = apply_transforms(true_uptake, transform_chain)

                    total_loss += loss_fn(pred_transformed, true_transformed)

                return apply_post_processing(total_loss, y_true.shape[0], post_mean)

        train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
        val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)
        # Return JAX arrays directly - let caller handle conversion if needed
        return train_loss, val_loss

    return functional_loss


def create_parameter_loss(
    loss_fn: PureLossFunction,
    transform_chain: list = None,
    post_mean: bool = True,
    normalise: bool = True,
    eps: float = 1e-8,
    scale_eps: bool = False,
) -> JaxEnt_Loss:
    """Create a parameter loss without builders."""

    transform_chain = transform_chain or []

    # @functools.partial(jax.jit, static_argnames=["prediction_index"])
    def parameter_loss(
        model: InitialisedSimulation,
        dataset: Simulation_Parameters,
        prediction_index: None,
    ) -> tuple[Array, Array]:
        model_weights = normalize_weights(model.params.frame_weights, normalise, eps, scale_eps)
        prior_weights = normalize_weights(dataset.frame_weights, normalise, eps, scale_eps)

        model_transformed = apply_transforms(model_weights, transform_chain)
        prior_transformed = apply_transforms(prior_weights, transform_chain)

        loss = loss_fn(model_transformed, prior_transformed)
        final_loss = apply_post_processing(loss, model_weights.shape[0], post_mean)

        # Return JAX arrays directly - let caller handle conversion if needed
        return final_loss, final_loss

    return parameter_loss


def create_consistency_loss(
    loss_fn: PureLossFunction,
    transform_chain: list = None,
    post_mean: bool = True,
    normalize_upper_tri: bool = False,
) -> JaxEnt_Loss:
    """Create a consistency loss without builders."""

    transform_chain = transform_chain or []

    # @functools.partial(jax.jit, static_argnames=["prediction_index"])
    def consistency_loss(
        model: InitialisedSimulation,
        dataset: Array,
        prediction_index: int,
    ) -> tuple[Array, Array]:
        weights = model.params.frame_weights
        weight_similarity = pairwise_cosine_similarity(weights)

        weight_upper = extract_upper_triangle(weight_similarity)
        dataset_upper = extract_upper_triangle(dataset)

        # Handle empty arrays
        is_empty = jnp.logical_or(weight_upper.size == 0, dataset_upper.size == 0)
        zero_loss = jnp.array(0.0)

        # Apply normalization
        weight_normalized = normalize_upper_triangle(weight_upper, normalize_upper_tri)
        dataset_normalized = normalize_upper_triangle(dataset_upper, normalize_upper_tri)

        # Apply transforms
        weight_transformed = apply_transforms(weight_normalized, transform_chain)
        dataset_transformed = apply_transforms(dataset_normalized, transform_chain)

        # Compute loss
        loss = loss_fn(weight_transformed, dataset_transformed)
        final_loss = apply_post_processing(loss, weight_upper.shape[0], post_mean)

        result = jnp.where(is_empty, zero_loss, final_loss)
        # Return JAX arrays directly - let caller handle conversion if needed
        return result, result

    return consistency_loss


# Registry using functional approach
class LossRegistry:
    """Registry using functional composition instead of builders."""

    _loss_factories: dict[str, Callable[[], JaxEnt_Loss]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], JaxEnt_Loss]):
        """Register a loss factory function."""
        cls._loss_factories[name] = factory

    @classmethod
    def get(cls, name: str) -> JaxEnt_Loss:
        """Get a Loss function."""
        if name not in cls._loss_factories:
            raise KeyError(f"Loss '{name}' not found in registry")

        # Build fresh each time to avoid stale closures
        loss_fn = cls._loss_factories[name]()
        return loss_fn

    @classmethod
    def list_losses(cls) -> list[str]:
        """List all registered losses."""
        return list(cls._loss_factories.keys())


# register loss decorator
def register_loss(name: str, factory: Callable[[], JaxEnt_Loss] = None):
    """Decorator or direct register for a loss factory."""
    if factory is None:

        def decorator(fn: Callable[[], JaxEnt_Loss]):
            LossRegistry.register(name, fn)
            return fn

        return decorator
    else:
        LossRegistry.register(name, factory)
        return factory


# Legacy adapter for compatibility
class LegacyLossAdapter:
    """Simple adapter that wraps legacy functions."""

    @staticmethod
    def wrap_existing_function(legacy_fn: Callable, *args, **kwargs):
        """Wrap a legacy function for the registry (ignore extra args)."""
        return lambda: legacy_fn


# Example usage
if __name__ == "__main__":
    print("Available losses:", LossRegistry.list_losses())
