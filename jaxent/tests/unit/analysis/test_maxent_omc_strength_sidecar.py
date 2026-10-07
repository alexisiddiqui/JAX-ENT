from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/1_IsoValidation_OMass/fitting/jaxENT"))
import run_maxent_omc_strength_sidecar as sidecar


def grid(tmp_path, strengths=sidecar.STRENGTHS, bandwidths=sidecar.BANDWIDTHS):
    return sidecar.build_specs(SimpleNamespace(output_dir=tmp_path, features_dir=tmp_path,
        datasplit_dir=tmp_path, clustering_dir=tmp_path, n_steps=5000, n_splits=3,
        strengths=strengths, bandwidths=bandwidths))


def test_grid_mse_only_allpairs_and_shared_strength(tmp_path):
    specs = grid(tmp_path)
    assert len(specs) == len({s.run_id for s in specs}) == 288
    assert sum(s.method == "maxent" for s in specs) == 36
    assert {s.ensemble for s in specs} == {"ISO_TRI"}
    assert {s.sigma_source for s in specs} == {"mse"}
    assert {s.uptake_mode for s in specs} == {"linear", "uptake"}
    assert all(s.graph_k == 0 and not s.graph_path and not s.sigma_path for s in specs)
    for s in specs:
        if s.method == "omc":
            assert s.bandwidth == s.kernel_bandwidth
            assert s.omc_strength == 1/s.sweep_maxent
        else:
            assert s.bandwidth == 1/s.sweep_maxent


def test_kernel_bandwidth_independent_of_strength_and_mode(tmp_path):
    features = SimpleNamespace(heavy_contacts=np.array([[1., 2., 4.], [3., 2., 0.]]),
                               acceptor_contacts=np.array([[0., 1., 1.], [0., 2., 3.]]))
    model = SimpleNamespace(params=SimpleNamespace(bv_bc=np.array([.35]), bv_bh=np.array([2.])))
    kernels = []
    for spec in grid(tmp_path, strengths=[10., 1e6], bandwidths=[.1]):
        slot = sidecar.base.regularizer(spec, features, model)
        if spec.method == "maxent":
            assert slot is None
        else:
            _, _, kernel, strength = slot
            assert strength == 1/spec.sweep_maxent
            assert kernel.bandwidth == .1
            kernels.append(np.asarray(kernel.similarity))
    for kernel in kernels[1:]:
        np.testing.assert_array_equal(kernel, kernels[0])


def test_plots_two_forward_panels(tmp_path):
    rows = [dict(uptake_mode=s.uptake_mode, method=s.method, bandwidth=s.bandwidth if s.method=="omc" else np.nan,
                 split_idx=s.split_idx, data_strength=s.sweep_maxent, recovery_percent=45.,
                 intermediate_percent=20., val_closed_sigma_mse=.001)
            for s in grid(tmp_path, strengths=[10., 100.])]
    for metric in ("recovery_percent", "intermediate_percent", "val_closed_sigma_mse"):
        output = tmp_path/metric
        sidecar.plot_metric(pd.DataFrame(rows), metric, output)
        assert output.with_suffix(".png").is_file()
        assert output.with_suffix(".svg").is_file()


def test_no_convergence_still_exports_final_state(tmp_path, monkeypatch):
    spec = grid(tmp_path, strengths=[10.], bandwidths=[1.])[0]
    monkeypatch.setattr(sidecar, "run_is_complete", lambda _: True)
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file",
                        lambda _: SimpleNamespace(states=[object()]))
    monkeypatch.setattr(sidecar.source, "iter_labeled_convergence_states", lambda _: [])
    monkeypatch.setattr(sidecar.base, "score_context",
                        lambda *args: (None, np.array([0, 1]), None, None, None))
    monkeypatch.setattr(sidecar.base, "score", lambda *args: (
        dict(run_id=spec.run_id, uptake_mode="linear", recovery_percent=45., ess_percent=80.,
             val_mse=.01, val_closed_sigma_mse=.005), np.array([.4, .6])))
    monkeypatch.setattr(sidecar, "plot_metric", lambda *args: None)
    sidecar.analyze([spec], tmp_path)
    assert len(pd.read_csv(tmp_path/"final_step_results.csv")) == 1
    assert pd.read_csv(tmp_path/"selected_results.csv").empty
    assert len(pd.read_csv(tmp_path/"missing_selection.csv")) == 1


def test_intermediate_percent_is_scored_from_assignment_minus_one(tmp_path, monkeypatch):
    spec = grid(tmp_path, strengths=[10.], bandwidths=[1.])[0]
    state = SimpleNamespace(step=5)
    history = SimpleNamespace(states=[state])
    monkeypatch.setattr(sidecar, "run_is_complete", lambda _: True)
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file", lambda _: history)
    monkeypatch.setattr(sidecar.source, "iter_labeled_convergence_states", lambda _: [])
    monkeypatch.setattr(sidecar.base, "score_context",
                        lambda *args: (None, np.array([0, -1, 1, -1]), None, None, None))
    monkeypatch.setattr(sidecar.base, "score", lambda *args: (
        dict(run_id=spec.run_id, uptake_mode="linear", recovery_percent=45., ess_percent=80.,
             val_mse=.01, val_closed_sigma_mse=.005), np.array([.1, .2, .3, .4])))
    monkeypatch.setattr(sidecar, "plot_metric", lambda *args: None)
    sidecar.analyze([spec], tmp_path)
    row = pd.read_csv(tmp_path/"final_step_results.csv").iloc[0]
    assert row.intermediate_percent == pytest.approx(60.)


def test_completion_count_is_json_serializable(tmp_path, monkeypatch):
    spec = grid(tmp_path, strengths=[10.], bandwidths=[1.])[0]
    spec.run_dir.mkdir(parents=True)
    spec.history_path.touch()
    spec.config_path.write_text("{}")
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file", lambda _: SimpleNamespace(
        states=[SimpleNamespace(losses=SimpleNamespace(total_train_loss=np.float32(.1)))]))
    assert type(sidecar.run_is_complete(spec)) is bool
    assert json.dumps(dict(completed=sum(sidecar.run_is_complete(s) for s in [spec]))) == '{"completed": 1}'
