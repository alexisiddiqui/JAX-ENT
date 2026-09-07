import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    effective_sample_size,
    graph_components,
    optimise_weights,
    prior_for_ess,
    pseudo_uptake,
    relabel_graph,
    select_graph,
)
from jaxent.src.opt.loss.graph_laplacian import build_frame_graph


def test_relabel_control_preserves_topology_and_degree_sequence():
    graph = build_frame_graph(np.arange(8.0)[:, None], k=2, metric="test")
    permutation = np.array([3, 6, 0, 7, 2, 5, 1, 4])
    control = relabel_graph(graph, permutation, "rewired")

    def degrees(candidate):
        values = np.zeros(candidate.n_nodes, dtype=int)
        np.add.at(values, np.asarray(candidate.edge_sources), 1)
        np.add.at(values, np.asarray(candidate.edge_targets), 1)
        return np.sort(values)

    np.testing.assert_array_equal(degrees(graph), degrees(control))
    np.testing.assert_allclose(
        np.sort(np.asarray(graph.edge_weights)),
        np.sort(np.asarray(control.edge_weights)),
    )
    assert graph_components(graph) == graph_components(control)


def test_prior_for_ess_reaches_requested_fraction():
    score = np.linspace(-2.0, 2.0, 100)
    weights = prior_for_ess(score, 0.4)
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    np.testing.assert_allclose(effective_sample_size(weights) / len(weights), 0.4)


def test_small_reweighting_optimization_is_finite():
    log_pf = np.array(
        [
            [-1.0, -0.5, 0.2, 0.9, 1.2],
            [0.5, 0.1, -0.2, 0.3, 0.8],
            [1.0, 0.7, 0.0, -0.4, -0.8],
        ]
    )
    observables = pseudo_uptake(log_pf)
    graph = build_frame_graph(log_pf.T, k=2, metric="test")
    truth = np.full(5, 0.2)
    prior = prior_for_ess(np.arange(5.0), 0.6)
    weights, strength, validation = optimise_weights(
        observables,
        truth,
        prior,
        np.array([0, 1, 3, 4]),
        np.array([2, 5]),
        graph,
        "laplacian",
        (0.0, 0.1),
        20,
    )
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
    assert strength in (0.0, 0.1)
    assert np.isfinite(validation)


def test_selection_uses_replica_b_and_evaluates_replica_c():
    rows = []
    for replica in (1, 2, 3):
        for system in ("a", "b", "c"):
            for metric, gain in (("pf_l2", 0.4), ("work_scale", 0.2)):
                rows.append(
                    {
                        "system_id": system,
                        "replica": replica,
                        "metric": metric,
                        "k": 5,
                        "density_energy_gain": gain,
                        "structural_edge_gain": 0.3,
                        "density_energy_p": 0.01,
                    }
                )
    selected, gate = select_graph(pd.DataFrame(rows))
    assert selected == {"metric": "pf_l2", "k": 5}
    assert gate["passed"]
