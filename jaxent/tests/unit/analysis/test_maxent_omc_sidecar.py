from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxent.src.opt.loss.original_omc_laplacian import omc_graph_energy

SCRIPT = (Path(__file__).resolve().parents[3] / "examples" / "1_IsoValidation_OMass"
          / "fitting" / "jaxENT" / "run_maxent_omc_sidecar.py")
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("maxent_omc_sidecar", SCRIPT)
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)


def grid(tmp_path, values=sidecar.DEFAULT_MAXENT):
    args = argparse.Namespace(maxent_values=values, n_splits=3, n_steps=5000,
                              output_dir=tmp_path, features_dir=tmp_path,
                              datasplit_dir=tmp_path, clustering_dir=tmp_path)
    paths = {("ISO_TRI", name, .1): tmp_path / f"{name}.npz"
             for name in sidecar.COMPARISONS[1:]}
    return sidecar.build_specs(args, paths)


def test_grid_reciprocal_axis_and_fixed_sources(tmp_path):
    specs = grid(tmp_path)
    assert len(specs) == len({s.run_id for s in specs}) == 252
    assert {s.ensemble for s in specs} == {"ISO_TRI"}
    assert {s.alpha for s in specs} == {.1}
    assert {s.omc_strength for s in specs if s.method == "omc"} == {1., .1, .01}
    for spec in specs:
        assert np.isclose(spec.bandwidth * spec.sweep_maxent, 1.)
        assert bool(spec.sigma_path) == (spec.sigma_source != "mse")


def test_work_scale_kernel_and_omc_slot_are_exact(tmp_path):
    features = SimpleNamespace(heavy_contacts=np.array([[1., 2., 4.], [3., 2., 0.]]),
                               acceptor_contacts=np.array([[0., 1., 1.], [0., 2., 3.]]))
    model = SimpleNamespace(params=SimpleNamespace(bv_bc=np.array([.35]), bv_bh=np.array([2.])))
    scalar = (.35 * features.heavy_contacts + 2. * features.acceptor_contacts).mean(axis=0)
    distance = np.abs(scalar[:, None] - scalar)
    np.testing.assert_allclose(sidecar.work_distances(features, model), distance)
    specs = grid(tmp_path, (2.,))
    for spec in specs:
        slot = sidecar.regularizer(spec, features, model)
        if spec.method == "maxent":
            assert slot is None
        else:
            name, _, kernel, strength = slot
            assert name == "original_omc_laplacian"
            assert strength == spec.omc_strength
            assert kernel.n_nodes == 3
            np.testing.assert_allclose(kernel.similarity,
                                       np.exp(-np.minimum(distance**2 / (2 * .5**2), 80)),
                                       rtol=1e-6, atol=1e-8)


def test_four_panel_plot(tmp_path):
    rows = [dict(panel=s.panel, sigma_source=s.sigma_source, split_idx=s.split_idx,
                 sweep_x=s.bandwidth, recovery_percent=45.) for s in grid(tmp_path, (1., 10.))]
    sidecar.plot_metric(pd.DataFrame(rows), "recovery_percent", tmp_path / "plot")
    assert (tmp_path / "plot.png").is_file()
    assert (tmp_path / "plot.svg").is_file()


def test_analysis_keeps_final_step_without_convergence_checkpoint(tmp_path, monkeypatch):
    spec = grid(tmp_path, (1.,))[0]
    monkeypatch.setattr(sidecar.source, "run_is_complete", lambda _: True)
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file",
                        lambda _: SimpleNamespace(states=[object()]))
    monkeypatch.setattr(sidecar.source, "iter_labeled_convergence_states", lambda _: [])
    monkeypatch.setattr(sidecar, "score_context", lambda *args: None)
    monkeypatch.setattr(sidecar, "score", lambda *args: (
        dict(run_id=spec.run_id, panel=spec.panel, sigma_source=spec.sigma_source,
             sweep_x=1., recovery_percent=45., ess_percent=80., val_mse=.01), np.ones(2)/2))
    monkeypatch.setattr(sidecar, "plot_metric", lambda *args: None)
    sidecar.analyze([spec], tmp_path)
    assert len(pd.read_csv(tmp_path / "final_step_results.csv")) == 1
    assert pd.read_csv(tmp_path / "selected_results.csv").empty
    assert len(pd.read_csv(tmp_path / "missing_selection.csv")) == 1
    assert pd.read_csv(tmp_path / "incomplete_runs.csv").empty


def test_linear_scoring_uses_linear_model_and_separate_cache(tmp_path, monkeypatch):
    uptake = grid(tmp_path, (1.,))[0]
    linear = dataclasses.replace(uptake, uptake_mode="linear")
    assert linear.run_id != uptake.run_id
    assert linear.averaging_label == "linear_uptake"
    assignments = np.array([0, 1])
    monkeypatch.setattr(sidecar.source, "score_context",
                        lambda *args: ("features", assignments, "old model", "loader", "truth"))
    monkeypatch.setattr(sidecar.shrinkage, "configure_model", lambda mode, _: mode)
    cache = {}
    assert sidecar.score_context(linear, cache)[2] == "linear"
    assert sidecar.score_context(uptake, cache)[2] == "uptake"
    assert len(cache) == 2


def test_linear_uses_same_work_graph_as_standard_bv(tmp_path):
    features = SimpleNamespace(heavy_contacts=np.array([[1., 2., 4.], [3., 2., 0.]]),
                               acceptor_contacts=np.array([[0., 1., 1.], [0., 2., 3.]]))
    spec = next(s for s in grid(tmp_path, (2.,)) if s.method == "omc")
    standard = sidecar.shrinkage.configure_model("uptake", np.array([0, 1, -1]))
    linear = sidecar.shrinkage.configure_model("linear", np.array([0, 1, -1]))
    a = sidecar.regularizer(spec, features, standard)[2]
    b = sidecar.regularizer(dataclasses.replace(spec, uptake_mode="linear"), features, linear)[2]
    np.testing.assert_array_equal(a.similarity, b.similarity)


def test_sparse_omc_matches_dense_masked_energy_and_gradient():
    graph = sidecar.FrameGraph(edge_sources=jnp.array([0, 0, 1]),
                               edge_targets=jnp.array([1, 2, 3]),
                               edge_weights=jnp.array([.3, .7, .8]),
                               n_nodes=4, metric="test", k=2)
    kernel = jnp.zeros((4, 4)).at[graph.edge_sources, graph.edge_targets].set(graph.edge_weights)
    kernel = kernel + kernel.T
    def sparse(w):
        model = SimpleNamespace(params=SimpleNamespace(frame_weight_simplex=w))
        return sidecar.sparse_omc_loss(model, graph, None)[0]
    w = jnp.array([.1, .2, .3, .4])
    np.testing.assert_allclose(sparse(w), omc_graph_energy(w, kernel), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(sparse)(w),
                               jax.grad(lambda x: omc_graph_energy(x, kernel))(w), rtol=1e-6)


def test_precomputed_knn_excludes_self_and_matches_l2_neighbours(tmp_path, monkeypatch):
    features = SimpleNamespace(heavy_contacts=np.array([[0., 1., 3., 8.]]),
                               acceptor_contacts=np.zeros((1, 4)))
    monkeypatch.setattr(sidecar.shrinkage, "load_features", lambda *args: (features, None))
    args = argparse.Namespace(graph_k=1, output_dir=tmp_path, features_dir=tmp_path)
    path = sidecar.prepare_neighbours(args)
    with np.load(path) as a:
        edges=set(zip(a['edge_sources'].tolist(),a['edge_targets'].tolist()))
        assert edges == {(0, 1), (1, 2), (2, 3)}
        np.testing.assert_allclose(a['distances'],
                                   np.abs(features.heavy_contacts.T-features.heavy_contacts)*.35,
                                   rtol=1e-6)
        assert all(i!=j for i,j in edges)
    assert sidecar.prepare_neighbours(args) == path
    features.heavy_contacts[0, 0] = 10.
    import pytest
    with pytest.raises(ValueError, match="input changed"):
        sidecar.prepare_neighbours(args)
