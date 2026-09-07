import numpy as np
import pandas as pd

import jaxent.examples.ATLAS_BV.analysis.laplacian_metric_comparison_checkpoint31 as cp31
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_frame_features,
)


def _data():
    z = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 1.5, 2.5, 4.0], [0.2, 0.4, 1.0, 1.8]]
    )
    return {"z": z, "system": "test", "config": {}}


def test_metric_frame_values_match_preregistered_definitions(monkeypatch):
    data = _data()
    monkeypatch.setattr(
        cp31,
        "load_score_frames",
        lambda system, config: {"ref2015__total": np.array([4.0, 3.0, 2.0, 1.0])},
    )
    np.testing.assert_allclose(
        cp31.metric_frame_values(data, "work_scale")[:, 0], data["z"].mean(axis=0)
    )
    expected = thermodynamic_frame_features(data["z"])["work_density_legacy_zq"].T
    np.testing.assert_allclose(
        cp31.metric_frame_values(data, "work_density_legacy_zq"), expected
    )
    np.testing.assert_allclose(
        cp31.metric_frame_values(data, "pyro_ref2015")[:, 0], [4.0, 3.0, 2.0, 1.0]
    )


def test_pyrosetta_frame_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(
        cp31,
        "load_score_frames",
        lambda system, config: {"ref2015__total": np.ones(3)},
    )
    with np.testing.assert_raises_regex(ValueError, "frame mismatch"):
        cp31.metric_frame_values(_data(), "pyro_ref2015")


def test_legacy_graph_uses_mean_l1_distance():
    values = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 0.0], [5.0, 5.0]])
    graph = cp31.metric_graph(values, np.arange(4), "work_density_legacy_zq", 1)
    assert graph.metric == "work_density_legacy_zq"
    assert graph.n_nodes == 4
    assert np.all(np.asarray(graph.edge_weights) > 0)


def test_basin_mass_prior_transfers_exact_mass_and_ess_is_incidental():
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    prior, target, basin = cp31.basin_mass_prior(labels, 0.25)
    np.testing.assert_allclose(prior.sum(), 1.0)
    np.testing.assert_allclose(prior[labels == basin].sum(), 0.45)
    np.testing.assert_allclose(target, 0.45)
    assert not np.isclose(cp31.effective_sample_size(prior) / len(prior), 0.25)


def test_candidate_selection_is_independent_per_metric():
    rows = []
    choices = {"pf_l2": 5, "work_scale": 10, "work_density_legacy_zq": 20, "pyro_ref2015": 50}
    for metric, best_k in choices.items():
        for replica in (1, 2, 3):
            for k in (5, 10, 20, 50):
                gain = 0.8 if k == best_k else 0.2
                for system in ("a", "b", "c"):
                    rows.append(
                        {
                            "system_id": system,
                            "replica": replica,
                            "metric": metric,
                            "k": k,
                            "density_energy_gain": gain,
                            "structural_edge_gain": gain,
                            "density_energy_p": 0.01,
                            "cluster_edge_purity": 0.9,
                        }
                    )
    selected, gates = cp31.select_candidate_graphs(pd.DataFrame(rows))
    assert selected == choices
    assert all(gate["passed"] for gate in gates.values())
