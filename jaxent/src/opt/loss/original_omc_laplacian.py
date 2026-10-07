"""Original weight-weighted OMC graph-Laplacian regularisation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.opt.base import JaxEnt_Loss
from jaxent.src.opt.loss.base import register_loss


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["similarity"],
    meta_fields=["n_nodes", "metric", "bandwidth"],
)
@dataclass(frozen=True, slots=True)
class OMCKernel:
    """Dense fixed structural similarity used by the original OMC objective."""

    similarity: Array
    n_nodes: int
    metric: str
    bandwidth: float


def build_omc_kernel(
    distance_matrix: np.ndarray,
    *,
    bandwidth: float,
    metric: str,
) -> OMCKernel:
    """Build a dense Gaussian similarity from a validated distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    n_nodes = matrix.shape[0]
    if n_nodes < 2:
        raise ValueError("OMC kernel requires at least two frames")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("distance matrix must be finite and non-negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
        raise ValueError("distance matrix must be symmetric")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")

    exponent = np.minimum(np.square(matrix) / (2.0 * float(bandwidth) ** 2), 80.0)
    similarity = np.exp(-exponent)
    return OMCKernel(
        similarity=jnp.asarray(similarity),
        n_nodes=n_nodes,
        metric=metric,
        bandwidth=float(bandwidth),
    )


def omc_graph_energy(
    frame_weights: Array | np.ndarray,
    similarity: Array | np.ndarray,
    *,
    normalise: bool = False,
) -> Array:
    """Evaluate the original OMC energy on simplex frame weights."""
    weights = jnp.asarray(frame_weights)
    kernel = jnp.asarray(similarity)
    weighted_similarity = kernel * weights[:, None] * weights[None, :]
    differences = weights[:, None] - weights[None, :]
    raw = 0.5 * jnp.sum(weighted_similarity * jnp.square(differences))
    scaled = weights.size**2 * raw
    if not normalise:
        return scaled
    denominator = jnp.sum(weighted_similarity)
    return jnp.where(
        denominator > 0,
        scaled / denominator,
        jnp.asarray(0.0, dtype=weights.dtype),
    )


def create_original_omc_loss(*, normalise: bool = False) -> JaxEnt_Loss:
    """Create a registry-compatible original OMC loss."""

    def original_omc_loss(
        model: InitialisedSimulation,
        dataset: OMCKernel,
        prediction_index: int | str | None,
    ) -> tuple[Array, Array]:
        del prediction_index
        if model.params.frame_weight_simplex.shape[0] != dataset.n_nodes:
            raise ValueError("model weights and OMC kernel have different node counts")
        loss = omc_graph_energy(
            model.params.frame_weight_simplex,
            dataset.similarity,
            normalise=normalise,
        )
        return loss, loss

    return original_omc_loss


@register_loss("original_omc_laplacian")
def original_omc_laplacian_builder() -> JaxEnt_Loss:
    """Register the unnormalised original OMC objective."""
    return create_original_omc_loss(normalise=False)


@register_loss("original_omc_laplacian_norm")
def original_omc_laplacian_norm_builder() -> JaxEnt_Loss:
    """Register the fitted-weight-normalised comparison objective."""
    return create_original_omc_loss(normalise=True)
