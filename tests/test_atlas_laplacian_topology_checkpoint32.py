import itertools

import numpy as np

from jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 import (
    BANDWIDTH_QUANTILES,
    K_VALUES,
    build_candidate_graph,
    bandwidth_from_quantile,
    candidate_specs,
    confirmation_rows,
    metric_distances,
    normalized_energy_numpy,
    optimise_graph_batch,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    load_rows as cp31_load_rows,
    edge_diagnostics,
    normalized_energy,
    optimise_weights,
    pseudo_uptake,
    relabel_graph,
)
from jaxent.src.opt.loss.graph_laplacian import build_frame_graph


def test_candidate_grid_contains_every_metric_topology_and_parameter():
    specs = candidate_specs()
    assert len(specs) == 4 * (2 * len(K_VALUES) + len(BANDWIDTH_QUANTILES))
    assert len({spec.key for spec in specs}) == len(specs)


def test_confirmation_cohort_excludes_every_development_system():
    development = {row["system_id"] for row in cp31_load_rows(False, None)}
    confirmation = {row["system_id"] for row in confirmation_rows(None)}
    assert len(development) == 24
    assert len(confirmation) == 87
    assert development.isdisjoint(confirmation)


def test_bandwidth_uses_positive_upper_triangle_quantile():
    distances = np.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [2.0, 4.0, 0.0]]
    )
    np.testing.assert_allclose(bandwidth_from_quantile(distances, 0.5), 3.0)


def test_legacy_distance_is_mean_l1_and_pf_distance_is_euclidean():
    values = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 0.0]])
    legacy = metric_distances(values, np.arange(3), "work_density_legacy_zq")
    pf = metric_distances(values, np.arange(3), "pf_l2")
    np.testing.assert_allclose(legacy[0, 1], 2.0)
    np.testing.assert_allclose(pf[0, 1], np.sqrt(10.0))


def test_all_pairs_candidate_retains_every_pair():
    distances = np.abs(np.arange(5.0)[:, None] - np.arange(5.0)[None, :])
    spec = next(
        item
        for item in candidate_specs()
        if item.metric == "work_scale" and item.topology == "all_pairs_rbf"
    )
    graph = build_candidate_graph(distances, spec)
    assert graph.edge_weights.size == 10
    assert np.all(np.asarray(graph.edge_weights) > 0)


def test_uniform_and_weighted_knn_share_topology():
    distances = np.abs(np.arange(8.0)[:, None] - np.arange(8.0)[None, :])
    weighted = next(
        item
        for item in candidate_specs()
        if item.metric == "work_scale"
        and item.topology == "self_tuned_knn"
        and item.parameter == 3
    )
    uniform = next(
        item
        for item in candidate_specs()
        if item.metric == "work_scale"
        and item.topology == "uniform_knn"
        and item.parameter == 3
    )
    weighted_graph = build_candidate_graph(distances, weighted)
    uniform_graph = build_candidate_graph(distances, uniform)
    np.testing.assert_array_equal(
        weighted_graph.edge_sources, uniform_graph.edge_sources
    )
    np.testing.assert_array_equal(
        weighted_graph.edge_targets, uniform_graph.edge_targets
    )
    np.testing.assert_allclose(uniform_graph.edge_weights, 1.0)


def test_knn_is_capped_for_small_smoke_ensembles():
    distances = np.abs(np.arange(8.0)[:, None] - np.arange(8.0)[None, :])
    spec = next(
        item
        for item in candidate_specs()
        if item.metric == "pf_l2"
        and item.topology == "self_tuned_knn"
        and item.parameter == 40
    )
    graph = build_candidate_graph(distances, spec)
    assert graph.k == 7


def test_batched_graph_optimization_matches_single_graph_path():
    log_pf = np.array(
        [[-1.0, -0.5, 0.2, 0.9, 1.2], [0.5, 0.1, -0.2, 0.3, 0.8]]
    )
    observables = pseudo_uptake(log_pf)
    graph = build_frame_graph(log_pf.T, k=2, metric="test")
    truth = np.full(5, 0.2)
    prior = np.array([0.35, 0.25, 0.2, 0.15, 0.05])
    train = np.array([0, 1, 3])
    validation = np.array([2, 4])
    strengths = (0.0, 0.1)
    expected = optimise_weights(
        observables,
        truth,
        prior,
        train,
        validation,
        graph,
        "laplacian",
        strengths,
        10,
    )
    actual = optimise_graph_batch(
        observables,
        truth,
        prior,
        train,
        validation,
        {"test": graph},
        strengths,
        10,
    )["test"]
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(actual[1:], expected[1:], rtol=1e-6)


def test_numpy_audit_energy_matches_jax_loss():
    graph = build_frame_graph(np.arange(8.0)[:, None], k=3, metric="test")
    values = np.array([-1.0, 0.2, 0.8, -0.4, 1.1, 2.0, 0.5, -0.2])
    np.testing.assert_allclose(
        normalized_energy_numpy(values, graph),
        normalized_energy(values, graph),
        rtol=1e-6,
    )


def test_exact_all_pairs_null_matches_mean_over_every_node_relabelling():
    coordinates = np.array([0.0, 0.2, 1.0, 3.0])
    structural = np.abs(coordinates[:, None] - coordinates[None, :])
    density = np.array([-0.8, 0.1, 0.6, 1.7])
    labels = np.array([0, 0, 1, 1])
    graph = build_frame_graph(coordinates[:, None], k=1, metric="test")
    energy = []
    distance = []
    purity = []
    for permutation in itertools.permutations(range(4)):
        control = relabel_graph(graph, np.asarray(permutation), "null")
        energy.append(normalized_energy_numpy(density, control))
        edge_distance, edge_purity = edge_diagnostics(control, structural, labels)
        distance.append(edge_distance)
        purity.append(edge_purity)
    source, target = np.triu_indices(4, k=1)
    expected_energy = np.mean(np.square(density[source] - density[target])) / np.var(
        density
    )
    np.testing.assert_allclose(np.mean(energy), expected_energy)
    np.testing.assert_allclose(np.mean(distance), np.mean(structural[source, target]))
    np.testing.assert_allclose(np.mean(purity), np.mean(labels[source] == labels[target]))
