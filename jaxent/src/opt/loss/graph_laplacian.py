"""Prior-relative graph-Laplacian regularisation for frame weights.

The loss smooths changes from an explicit frame-weight prior rather than assigning
absolute Boltzmann weights to individual MD samples.  A graph is constructed once
from fixed frame features and is treated as data during optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.neighbors import NearestNeighbors

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.opt.base import JaxEnt_Loss
from jaxent.src.opt.loss.base import register_loss


GraphMetric = Literal["pf_l2", "work_scale"]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["edge_sources", "edge_targets", "edge_weights"],
    meta_fields=["n_nodes", "metric", "k"],
)
@dataclass(frozen=True, slots=True)
class FrameGraph:
    """Sparse undirected frame graph stored as one canonical row per edge."""

    edge_sources: Array
    edge_targets: Array
    edge_weights: Array
    n_nodes: int
    metric: str
    k: int


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["prior_logits", "edge_sources", "edge_targets", "edge_weights"],
    meta_fields=["n_nodes", "metric", "k"],
)
@dataclass(frozen=True, slots=True)
class FrameGraphPrior:
    """Prior logits and a fixed graph supplied to the Laplacian loss."""

    prior_logits: Array
    edge_sources: Array
    edge_targets: Array
    edge_weights: Array
    n_nodes: int
    metric: str
    k: int

    @classmethod
    def from_frame_weights(
        cls, frame_weights: Array | np.ndarray, graph: FrameGraph
    ) -> "FrameGraphPrior":
        """Create a loss target from normalized, strictly positive weights."""
        weights = np.asarray(frame_weights, dtype=float)
        if weights.shape != (graph.n_nodes,):
            raise ValueError(
                f"prior weights have shape {weights.shape}; "
                f"expected {(graph.n_nodes,)}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("prior weights must be finite and strictly positive")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("prior weights must have positive finite mass")
        return cls(
            prior_logits=jnp.log(jnp.asarray(weights / total)),
            edge_sources=graph.edge_sources,
            edge_targets=graph.edge_targets,
            edge_weights=graph.edge_weights,
            n_nodes=graph.n_nodes,
            metric=graph.metric,
            k=graph.k,
        )


def _graph_coordinates(
    input_features: BV_input_features,
    bv_bc: float,
    bv_bh: float,
    metric: GraphMetric,
) -> np.ndarray:
    heavy = np.asarray(input_features.heavy_contacts, dtype=float)
    acceptor = np.asarray(input_features.acceptor_contacts, dtype=float)
    if heavy.shape != acceptor.shape or heavy.ndim != 2:
        raise ValueError(
            "BV graph features must be matching (n_residues, n_frames) arrays"
        )
    if not np.all(np.isfinite(heavy)) or not np.all(np.isfinite(acceptor)):
        raise ValueError("BV graph features must be finite")
    if heavy.shape[0] == 0 or heavy.shape[1] < 2:
        raise ValueError("BV graph construction requires residues and at least two frames")

    log_pf = float(bv_bc) * heavy + float(bv_bh) * acceptor
    if metric == "pf_l2":
        # RMS scaling preserves within-system neighbour order while making recorded
        # distances comparable across proteins with different residue counts.
        return log_pf.T / np.sqrt(log_pf.shape[0])
    if metric == "work_scale":
        return np.mean(log_pf, axis=0, keepdims=True).T
    raise ValueError(f"unknown BV frame-graph metric: {metric!r}")


def build_frame_graph(
    coordinates: np.ndarray,
    *,
    k: int,
    metric: str,
    weighted: bool = True,
    symmetrization: Literal["union", "mutual"] = "union",
) -> FrameGraph:
    """Build a symmetric self-tuning Gaussian k-nearest-neighbour graph."""
    values = np.asarray(coordinates, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("graph coordinates must have shape (n_frames, n_features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("graph coordinates must be finite")
    n_nodes = values.shape[0]
    if not 1 <= k < n_nodes:
        raise ValueError(f"k must satisfy 1 <= k < n_frames; got k={k}, n={n_nodes}")

    search = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(values)
    distances, indices = search.kneighbors(values)
    return _build_graph_from_neighbour_search(
        distances,
        indices,
        k=k,
        metric=metric,
        weighted=weighted,
        symmetrization=symmetrization,
    )


def build_frame_graph_from_distances(
    distance_matrix: np.ndarray,
    *,
    k: int,
    metric: str,
    weighted: bool = True,
    symmetrization: Literal["union", "mutual"] = "union",
) -> FrameGraph:
    """Build the same graph from a validated precomputed distance matrix."""
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    n_nodes = matrix.shape[0]
    if n_nodes < 2 or not 1 <= k < n_nodes:
        raise ValueError(f"k must satisfy 1 <= k < n_frames; got k={k}, n={n_nodes}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("distance matrix must be finite and non-negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
        raise ValueError("distance matrix must be symmetric")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    indices = np.argsort(matrix, axis=1, kind="stable")[:, : k + 1]
    distances = np.take_along_axis(matrix, indices, axis=1)
    return _build_graph_from_neighbour_search(
        distances,
        indices,
        k=k,
        metric=metric,
        weighted=weighted,
        symmetrization=symmetrization,
    )


def _build_graph_from_neighbour_search(
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    k: int,
    metric: str,
    weighted: bool = True,
    symmetrization: Literal["union", "mutual"] = "union",
) -> FrameGraph:
    """Convert a nearest-neighbour result into canonical weighted edges."""
    if symmetrization not in ("union", "mutual"):
        raise ValueError("symmetrization must be 'union' or 'mutual'")
    n_nodes = indices.shape[0]
    neighbour_distances = np.empty((n_nodes, k), dtype=float)
    neighbours = np.empty((n_nodes, k), dtype=np.int64)
    for node in range(n_nodes):
        keep = indices[node] != node
        neighbours[node] = indices[node, keep][:k]
        neighbour_distances[node] = distances[node, keep][:k]

    local_scale = neighbour_distances[:, -1]
    positive = local_scale[local_scale > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    local_scale = np.where(local_scale > 0, local_scale, fallback)

    edge_values: dict[tuple[int, int], float] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    tiny = np.finfo(float).tiny
    for source in range(n_nodes):
        for target, distance in zip(
            neighbours[source], neighbour_distances[source], strict=True
        ):
            left, right = sorted((source, int(target)))
            denominator = max(local_scale[source] * local_scale[target], tiny)
            exponent = min((distance * distance) / denominator, 80.0)
            weight = float(np.exp(-exponent)) if weighted else 1.0
            # The Gaussian expression is symmetric. max also makes duplicate
            # directed kNN proposals deterministic under floating-point ties.
            edge_values[(left, right)] = max(edge_values.get((left, right), 0.0), weight)
            edge_counts[(left, right)] = edge_counts.get((left, right), 0) + 1

    ordered = sorted(
        edge
        for edge in edge_values
        if symmetrization == "union" or edge_counts[edge] == 2
    )
    if not ordered:
        raise ValueError("frame graph contains no edges")
    sources = np.fromiter((edge[0] for edge in ordered), dtype=np.int32)
    targets = np.fromiter((edge[1] for edge in ordered), dtype=np.int32)
    weights = np.fromiter((edge_values[edge] for edge in ordered), dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("frame graph produced invalid edge weights")
    return FrameGraph(
        edge_sources=jnp.asarray(sources),
        edge_targets=jnp.asarray(targets),
        edge_weights=jnp.asarray(weights),
        n_nodes=n_nodes,
        metric=metric,
        k=k,
    )


def build_all_pairs_graph_from_distances(
    distance_matrix: np.ndarray,
    *,
    metric: str,
    bandwidth: float | None,
) -> FrameGraph:
    """Build a complete graph with uniform or Gaussian distance weights.

    ``bandwidth=None`` gives the geometry-free uniform complete graph. A positive
    bandwidth retains every edge and assigns an RBF weight. The exponent is clipped
    only to prevent floating-point underflow; no edge is pruned.
    """
    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    n_nodes = matrix.shape[0]
    if n_nodes < 2:
        raise ValueError("all-pairs graph requires at least two frames")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("distance matrix must be finite and non-negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
        raise ValueError("distance matrix must be symmetric")
    if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("bandwidth must be positive and finite, or None")

    source, target = np.triu_indices(n_nodes, k=1)
    if bandwidth is None:
        weights = np.ones(len(source), dtype=float)
    else:
        exponent = np.minimum(
            np.square(matrix[source, target] / float(bandwidth)) / 2.0, 80.0
        )
        weights = np.exp(-exponent)
    return FrameGraph(
        edge_sources=jnp.asarray(source, dtype=jnp.int32),
        edge_targets=jnp.asarray(target, dtype=jnp.int32),
        edge_weights=jnp.asarray(weights),
        n_nodes=n_nodes,
        metric=metric,
        k=n_nodes - 1,
    )


def build_bv_frame_graph(
    input_features: BV_input_features,
    *,
    bv_bc: float = 0.35,
    bv_bh: float = 2.0,
    metric: GraphMetric = "pf_l2",
    k: int = 10,
) -> FrameGraph:
    """Build a frozen frame graph from fixed-BV log-protection-factor features."""
    coordinates = _graph_coordinates(input_features, bv_bc, bv_bh, metric)
    return build_frame_graph(coordinates, k=k, metric=metric)


def prior_relative_graph_energy(
    frame_weight_logits: Array,
    prior_logits: Array,
    edge_sources: Array,
    edge_targets: Array,
    edge_weights: Array,
) -> Array:
    """Return mean weighted squared edge differences of log weight ratios."""
    residual = jnp.asarray(frame_weight_logits) - jnp.asarray(prior_logits)
    source = jnp.asarray(edge_sources, dtype=jnp.int32)
    target = jnp.asarray(edge_targets, dtype=jnp.int32)
    weights = jnp.asarray(edge_weights)
    edge_delta = residual[source] - residual[target]
    total_weight = jnp.sum(weights)
    return jnp.where(
        total_weight > 0,
        jnp.sum(weights * jnp.square(edge_delta)) / total_weight,
        jnp.asarray(0.0, dtype=residual.dtype),
    )


def create_prior_graph_laplacian_loss() -> JaxEnt_Loss:
    """Create the registry-compatible prior-relative graph loss."""

    def prior_graph_laplacian_loss(
        model: InitialisedSimulation,
        dataset: FrameGraphPrior,
        prediction_index: int | str | None,
    ) -> tuple[Array, Array]:
        del prediction_index
        if model.params.frame_weight_logits.shape[0] != dataset.n_nodes:
            raise ValueError("model weights and frame graph have different node counts")
        loss = prior_relative_graph_energy(
            model.params.frame_weight_logits,
            dataset.prior_logits,
            dataset.edge_sources,
            dataset.edge_targets,
            dataset.edge_weights,
        )
        return loss, loss

    return prior_graph_laplacian_loss


@register_loss("prior_graph_laplacian")
def prior_graph_laplacian_builder() -> JaxEnt_Loss:
    """Register the prior-relative graph-Laplacian loss."""
    return create_prior_graph_laplacian_loss()


def graph_prior_from_parameters(
    parameters: Simulation_Parameters, graph: FrameGraph
) -> FrameGraphPrior:
    """Create a graph target while preserving the parameters' current simplex."""
    return FrameGraphPrior.from_frame_weights(parameters.frame_weight_simplex, graph)
