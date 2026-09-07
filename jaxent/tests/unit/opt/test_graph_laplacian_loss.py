"""Tests for prior-relative frame-graph regularisation."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.opt.loss.base import LossRegistry
from jaxent.src.opt.loss.graph_laplacian import (
    FrameGraph,
    FrameGraphPrior,
    build_bv_frame_graph,
    build_all_pairs_graph_from_distances,
    build_frame_graph,
    build_frame_graph_from_distances,
    create_prior_graph_laplacian_loss,
    prior_relative_graph_energy,
)


def _graph() -> FrameGraph:
    return FrameGraph(
        edge_sources=jnp.array([0, 1, 2]),
        edge_targets=jnp.array([1, 2, 3]),
        edge_weights=jnp.array([1.0, 2.0, 0.5]),
        n_nodes=4,
        metric="test",
        k=1,
    )


def _parameters(logits: jax.Array) -> Simulation_Parameters:
    return Simulation_Parameters(
        frame_weight_logits=logits,
        model_parameters=(),
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )


class _SimulationStub:
    outputs = ()

    def __init__(self, params):
        self.params = params

    @staticmethod
    def forward(sim, params, mutate=True):
        del params, mutate
        return sim


def test_loss_is_zero_and_has_zero_gradient_at_prior():
    prior = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
    graph = _graph()

    def objective(logits):
        return prior_relative_graph_energy(
            logits,
            prior,
            graph.edge_sources,
            graph.edge_targets,
            graph.edge_weights,
        )

    np.testing.assert_allclose(objective(prior), 0.0, atol=1e-7)
    np.testing.assert_allclose(jax.grad(objective)(prior), 0.0, atol=1e-7)


def test_loss_is_invariant_to_additive_logit_constants():
    graph = _graph()
    logits = jnp.array([-2.0, -1.0, 0.5, 3.0])
    prior = jnp.array([0.5, -0.5, 0.25, -1.0])
    args = (graph.edge_sources, graph.edge_targets, graph.edge_weights)
    expected = prior_relative_graph_energy(logits, prior, *args)
    actual = prior_relative_graph_energy(logits + 17.0, prior - 4.0, *args)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_sparse_energy_matches_dense_combinatorial_laplacian():
    graph = _graph()
    residual = np.array([0.2, -0.4, 1.3, 0.7])
    adjacency = np.zeros((4, 4))
    source = np.asarray(graph.edge_sources)
    target = np.asarray(graph.edge_targets)
    weight = np.asarray(graph.edge_weights)
    adjacency[source, target] = weight
    adjacency[target, source] = weight
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    expected = residual @ laplacian @ residual / weight.sum()
    actual = prior_relative_graph_energy(
        jnp.asarray(residual),
        jnp.zeros(4),
        graph.edge_sources,
        graph.edge_targets,
        graph.edge_weights,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_disconnected_graph_leaves_component_offsets_unpenalized():
    source = jnp.array([0, 2])
    target = jnp.array([1, 3])
    weights = jnp.ones(2)
    logits = jnp.array([3.0, 3.0, -8.0, -8.0])
    loss = prior_relative_graph_energy(logits, jnp.zeros(4), source, target, weights)
    np.testing.assert_allclose(loss, 0.0, atol=1e-7)


def test_energy_is_equivariant_to_node_permutation():
    graph = _graph()
    logits = jnp.array([-2.0, 0.1, 0.7, 1.2])
    prior = jnp.array([0.0, -0.5, 0.5, 0.2])
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted_source = inverse[np.asarray(graph.edge_sources)]
    permuted_target = inverse[np.asarray(graph.edge_targets)]
    expected = prior_relative_graph_energy(
        logits,
        prior,
        graph.edge_sources,
        graph.edge_targets,
        graph.edge_weights,
    )
    actual = prior_relative_graph_energy(
        logits[permutation],
        prior[permutation],
        jnp.asarray(permuted_source),
        jnp.asarray(permuted_target),
        graph.edge_weights,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_graph_construction_is_deterministic_symmetric_and_valid():
    coordinates = np.array([[0.0], [0.1], [1.0], [1.2], [4.0]])
    first = build_frame_graph(coordinates, k=2, metric="test")
    second = build_frame_graph(coordinates, k=2, metric="test")
    np.testing.assert_array_equal(first.edge_sources, second.edge_sources)
    np.testing.assert_array_equal(first.edge_targets, second.edge_targets)
    np.testing.assert_allclose(first.edge_weights, second.edge_weights)
    assert np.all(np.asarray(first.edge_sources) < np.asarray(first.edge_targets))
    assert np.all(np.asarray(first.edge_weights) > 0)
    assert len(set(zip(first.edge_sources.tolist(), first.edge_targets.tolist()))) == len(
        first.edge_weights
    )


def test_coordinate_and_precomputed_distance_graphs_match():
    coordinates = np.array([[0.0], [0.1], [1.0], [1.2], [4.0]])
    distance = np.abs(coordinates - coordinates.T)
    direct = build_frame_graph(coordinates, k=2, metric="test")
    precomputed = build_frame_graph_from_distances(distance, k=2, metric="test")
    np.testing.assert_array_equal(direct.edge_sources, precomputed.edge_sources)
    np.testing.assert_array_equal(direct.edge_targets, precomputed.edge_targets)
    np.testing.assert_allclose(direct.edge_weights, precomputed.edge_weights)


def test_uniform_knn_preserves_edges_and_removes_distance_weighting():
    coordinates = np.array([[0.0], [0.1], [1.0], [1.2], [4.0]])
    weighted = build_frame_graph(coordinates, k=2, metric="weighted")
    uniform = build_frame_graph(
        coordinates, k=2, metric="uniform", weighted=False
    )
    np.testing.assert_array_equal(weighted.edge_sources, uniform.edge_sources)
    np.testing.assert_array_equal(weighted.edge_targets, uniform.edge_targets)
    np.testing.assert_allclose(uniform.edge_weights, 1.0)


def test_mutual_knn_keeps_only_reciprocal_neighbours():
    coordinates = np.array([[0.0], [0.1], [1.0], [3.0]])
    union = build_frame_graph(coordinates, k=1, metric="union")
    mutual = build_frame_graph(
        coordinates, k=1, metric="mutual", symmetrization="mutual"
    )
    union_edges = set(zip(union.edge_sources.tolist(), union.edge_targets.tolist()))
    mutual_edges = set(
        zip(mutual.edge_sources.tolist(), mutual.edge_targets.tolist())
    )
    assert mutual_edges == {(0, 1)}
    assert mutual_edges < union_edges


def test_weighted_all_pairs_contains_every_edge_and_uses_rbf_weights():
    coordinates = np.array([0.0, 1.0, 3.0])
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    graph = build_all_pairs_graph_from_distances(
        distance, metric="dense_rbf", bandwidth=2.0
    )
    assert graph.edge_weights.size == 3
    np.testing.assert_array_equal(graph.edge_sources, [0, 0, 1])
    np.testing.assert_array_equal(graph.edge_targets, [1, 2, 2])
    np.testing.assert_allclose(
        graph.edge_weights,
        np.exp(-np.square(np.array([1.0, 3.0, 2.0]) / 2.0) / 2.0),
    )


def test_uniform_all_pairs_matches_closed_form_variance():
    n_frames = 7
    distance = np.abs(np.arange(n_frames)[:, None] - np.arange(n_frames)[None, :])
    graph = build_all_pairs_graph_from_distances(
        distance, metric="complete_uniform", bandwidth=None
    )
    residual = jnp.array([-1.0, 0.2, 0.7, 2.1, -0.4, 0.3, 1.2])
    actual = prior_relative_graph_energy(
        residual,
        jnp.zeros(n_frames),
        graph.edge_sources,
        graph.edge_targets,
        graph.edge_weights,
    )
    expected = (2.0 * n_frames / (n_frames - 1)) * jnp.var(residual)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_all_pairs_rejects_invalid_bandwidth():
    distance = np.array([[0.0, 1.0], [1.0, 0.0]])
    for bandwidth in (0.0, -1.0, np.nan):
        with np.testing.assert_raises_regex(ValueError, "bandwidth"):
            build_all_pairs_graph_from_distances(
                distance, metric="dense", bandwidth=bandwidth
            )


def test_bv_graph_supports_profile_and_work_scale_metrics():
    features = BV_input_features(
        heavy_contacts=np.array([[0.0, 1.0, 2.0], [1.0, 1.5, 3.0]]),
        acceptor_contacts=np.array([[0.2, 0.1, 0.8], [0.3, 0.7, 0.9]]),
    )
    for metric in ("pf_l2", "work_scale"):
        graph = build_bv_frame_graph(features, metric=metric, k=1)
        assert graph.n_nodes == 3
        assert graph.metric == metric
        assert graph.edge_weights.size >= 2


def test_registry_loss_accepts_jittable_graph_target():
    graph = _graph()
    target = FrameGraphPrior.from_frame_weights(
        np.array([0.1, 0.2, 0.3, 0.4]), graph
    )
    params = _parameters(target.prior_logits + jnp.array([0.0, 0.2, -0.1, 0.3]))
    model = _SimulationStub(params)
    loss_fn = create_prior_graph_laplacian_loss()
    eager = loss_fn(model, target, None)[0]

    def pure(logits, graph_target):
        local_model = _SimulationStub(_parameters(logits))
        return loss_fn(local_model, graph_target, None)[0]

    compiled = jax.jit(pure)(params.frame_weight_logits, target)
    assert np.isfinite(float(compiled))
    np.testing.assert_allclose(compiled, eager, rtol=1e-6)
    assert "prior_graph_laplacian" in LossRegistry.list_losses()


def test_loss_gradient_does_not_touch_model_parameters():
    graph = _graph()
    target = FrameGraphPrior.from_frame_weights(np.full(4, 0.25), graph)
    params = _parameters(jnp.array([0.0, 0.2, -0.1, 0.3]))

    def objective(candidate):
        model = _SimulationStub(candidate)
        return create_prior_graph_laplacian_loss()(model, target, None)[0]

    gradient = jax.grad(objective)(params)
    assert not gradient.model_parameters
    assert np.all(np.isfinite(np.asarray(gradient.frame_weight_logits)))
