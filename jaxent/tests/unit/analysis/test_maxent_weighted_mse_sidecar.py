from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "examples/1_IsoValidation_OMass/fitting/jaxENT"
sys.path.insert(0, str(SCRIPT_DIR))
import run_maxent_weighted_mse_sidecar as sidecar


def test_grid(tmp_path):
    args = argparse.Namespace(maxent_values=sidecar.DEFAULT_MAXENT, n_splits=3,
        n_steps=5000, output_dir=tmp_path, features_dir=tmp_path,
        datasplit_dir=tmp_path, clustering_dir=tmp_path)
    sigma = {(e, l, 0.): tmp_path / f"{e}_{l}.npz"
             for e in sidecar.ENSEMBLES for l in sidecar.LOSSES[1:3]}
    weights = {(e, s): tmp_path / f"{e}_{s}.npz" for e in sidecar.ENSEMBLES for s in range(3)}
    specs = sidecar.build_specs(args, sigma, weights)
    assert len(specs) == len({s.run_id for s in specs}) == 336
    assert {s.alpha for s in specs} == {0.}
    assert {s.uptake_mode for s in specs} == {"linear", "uptake"}
    assert all(bool(s.sigma_path) == s.sigma_source.endswith("coordinate") for s in specs)


def test_combine_extension_rejects_duplicate_points():
    row = dict(ensemble="ISO_TRI", uptake_mode="uptake", sigma_source="mse",
               maxent=1e4, split_idx=0, recovery_percent=50.)
    previous = pd.DataFrame([row])
    current = pd.DataFrame([row | dict(maxent=1e5), row | dict(maxent=1e6)])
    combined = sidecar.combine_results(current, previous)
    assert combined.maxent.tolist() == [1e4, 1e5, 1e6]
    with pytest.raises(ValueError, match="overlap"):
        sidecar.combine_results(previous, previous)


def test_tri_only_extension_grid(tmp_path):
    args = argparse.Namespace(ensembles=["ISO_TRI"], maxent_values=[1e7], n_splits=3,
        n_steps=5000, output_dir=tmp_path, features_dir=tmp_path,
        datasplit_dir=tmp_path, clustering_dir=tmp_path, precision_normalization="per_peptide")
    weights = {("ISO_TRI", s): tmp_path / str(s) for s in range(3)}
    specs = sidecar.build_specs(args, {}, weights)
    assert len(specs) == len({s.run_id for s in specs}) == 24
    assert {s.ensemble for s in specs} == {"ISO_TRI"}
    assert {s.sweep_maxent for s in specs} == {1e7}


def test_oracle_variance_is_population_and_time_resolved():
    curves = np.array([[[0., 1., 100.], [0., .5, 100.]],
                       [[.2, .4, 100.], [1., 1., 100.]]])
    mean, variance, precision = sidecar.marginal_precision(curves, [.4, .6, 0.], 1e-4)
    np.testing.assert_allclose(mean, [[.6, .3], [.32, 1.]])
    np.testing.assert_allclose(variance, [[.24, .06], [.0096, 0.]], atol=1e-15)
    assert np.isclose(precision.mean(), 1.)
    assert np.isclose(precision[0, 1] / precision[0, 0], 4.)
    assert np.isfinite(precision).all()


def test_per_peptide_precision_equalizes_peptide_weight_not_time_weight():
    curves = np.array([[[0., 1.], [0., .5]], [[0., .1], [0., .1]]])
    mean, variance, w = sidecar.marginal_precision(curves, [.4, .6], normalization="per_peptide")
    np.testing.assert_allclose(w.sum(axis=1), [2., 2.])
    np.testing.assert_allclose(w[0, 1] / w[0, 0], 4.)
    np.testing.assert_allclose(w[1], [1., 1.])
    np.testing.assert_allclose(w.mean(), 1.)
    # A peptide has identical weights regardless of other peptides in its split.
    _, _, isolated = sidecar.marginal_precision(curves[:1], [.4, .6], normalization="per_peptide")
    np.testing.assert_allclose(w[:1], isolated)
    mg, vg, _ = sidecar.marginal_precision(curves, [.4, .6])
    np.testing.assert_array_equal(mean, mg)
    np.testing.assert_array_equal(variance, vg)
    with pytest.raises(ValueError, match="Unknown precision"):
        sidecar.marginal_precision(curves, [.4, .6], normalization="unknown")


def test_reuse_baselines_only_replaces_unmodified_losses(tmp_path, monkeypatch):
    args = argparse.Namespace(maxent_values=[1000.], n_splits=1, n_steps=5000,
        output_dir=tmp_path / "old", features_dir=tmp_path, datasplit_dir=tmp_path,
        clustering_dir=tmp_path, precision_normalization="global")
    weights = {(e, 0): tmp_path / e for e in sidecar.ENSEMBLES}
    old = sidecar.build_specs(args, {}, weights)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text(json.dumps(dict(
        settings=dict(code_sha256=sidecar.COMPATIBLE_BASELINE_HASH),
        specs=[dataclasses.asdict(s) for s in old])))
    args.output_dir = tmp_path / "new"
    args.precision_normalization = "per_peptide"
    new = sidecar.build_specs(args, {}, weights)
    monkeypatch.setattr(sidecar, "run_is_complete", lambda _: True)
    merged = sidecar.reuse_baselines(new, baseline)
    assert sum(s.output_dir == str(args.output_dir) for s in merged) == 4
    assert all(s.run_id.endswith("_per_peptide") for s in merged if s.sigma_source == "gt_uptake_weighted")
    assert all(s.precision_normalization == "global" for s in merged if s.sigma_source != "gt_uptake_weighted")
    args.n_steps = 50
    with pytest.raises(ValueError, match="n_steps"):
        sidecar.reuse_baselines(sidecar.build_specs(args, {}, weights), baseline)


def test_weighted_loss_matches_flat_diagonal_quadratic_and_gradient():
    mapping = jnp.array([[.5, .5, 0.], [0., 0., 1.]])
    target = jnp.array([[.2, .8], [.1, .9]])
    data = SimpleNamespace(residue_feature_ouput_mapping=sparse.BCOO.fromdense(mapping),
                           y_true=target[..., None])
    dataset = SimpleNamespace(train=data, val=data)
    precision = jnp.array([[.5, 1.], [1.5, 1.]])
    loss = sidecar.make_weighted_loss(precision, precision)
    uptake = jnp.array([[.3, .5, .6], [.6, .7, .8]])

    def actual(u):
        return loss(SimpleNamespace(outputs=[SimpleNamespace(uptake=u)]), dataset, 0)[0]

    def expected(u):
        residual = (mapping @ u.T - target).reshape(-1)
        return .5 * residual @ jnp.diag(precision.reshape(-1)) @ residual / residual.size

    np.testing.assert_allclose(actual(uptake), expected(uptake), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(actual)(uptake), jax.grad(expected)(uptake), rtol=1e-6)
    uniform = sidecar.make_weighted_loss(jnp.ones((2, 2)), jnp.ones((2, 2)))
    np.testing.assert_allclose(uniform(SimpleNamespace(outputs=[SimpleNamespace(uptake=uptake)]), dataset, 0)[0],
                               .5 * jnp.mean((mapping @ uptake.T - target)**2))


def test_map_frames_before_variance():
    # Perfectly anticorrelated residues have a constant peptide average.
    residue_curves = np.array([[[0., 1.], [1., 0.]]])
    peptide = np.einsum("pr,trf->ptf", [[.5, .5]], residue_curves)
    _, variance, _ = sidecar.marginal_precision(peptide, [.4, .6])
    np.testing.assert_allclose(variance, 0.)


def test_plot_eight_hues_two_panels(tmp_path):
    rows = [dict(ensemble=e, uptake_mode=m, sigma_source=l, split_idx=s, maxent=x,
                 recovery_percent=40+x, ess_percent=60-x)
            for e in sidecar.ENSEMBLES for m in sidecar.MODES for l in sidecar.LOSSES
            for s in range(3) for x in (1., 10.)]
    sidecar.plot_metric(pd.DataFrame(rows), "recovery_percent", tmp_path / "final_step")
    assert (tmp_path / "final_step.png").is_file()
    assert (tmp_path / "final_step.svg").is_file()


def test_final_state_retained_without_selection(tmp_path, monkeypatch):
    spec = sidecar.RunSpec(ensemble="ISO_TRI", sigma_source="gt_uptake_weighted", alpha=0.,
        split_type="sequence_cluster", split_idx=0, sigma_path="", output_dir=str(tmp_path),
        features_dir="", datasplit_dir="", clustering_dir="", n_steps=5, learning_rate=1.,
        ema_alpha=.5, forward_model_scaling=1000., execution_mode="compiled",
        uptake_mode="linear", sweep_maxent=1000., weights_path="")
    monkeypatch.setattr(sidecar.source, "run_is_complete", lambda _: True)
    monkeypatch.setattr(sidecar.source, "score_context", lambda *args: (None, None, None, None, None))
    monkeypatch.setattr(sidecar.shrinkage, "configure_model", lambda *args: None)
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file",
                        lambda _: SimpleNamespace(states=[object()]))
    monkeypatch.setattr(sidecar.source, "iter_labeled_convergence_states", lambda _: [])
    monkeypatch.setattr(sidecar.source, "score_state", lambda *args: (
        dict(ensemble="ISO_TRI", split_idx=0, recovery_percent=40., ess_percent=60., val_mse=.01),
        np.array([.4, .6])))
    monkeypatch.setattr(sidecar, "plot_metric", lambda *args: None)
    sidecar.analyze([spec], tmp_path)
    final = pd.read_csv(tmp_path / "final_step_results.csv")
    assert len(final) == 1
    assert final.maxent.iloc[0] == 1000.
    assert final.sigma_source.iloc[0] == "gt_uptake_weighted"
    assert pd.read_csv(tmp_path / "selected_results.csv").empty
    assert len(pd.read_csv(tmp_path / "missing_selection.csv")) == 1


def test_empty_history_not_complete_and_analysis_reports_it(tmp_path, monkeypatch):
    spec = SimpleNamespace(ensemble="ISO_TRI", split_idx=0, uptake_mode="linear",
                           history_path=tmp_path / "empty.hdf5", run_id="empty")
    monkeypatch.setattr(sidecar.source, "run_is_complete", lambda _: True)
    monkeypatch.setattr(sidecar.source, "load_optimization_history_from_file",
                        lambda _: SimpleNamespace(states=[]))
    monkeypatch.setattr(sidecar.source, "score_context", lambda *args: (None, None, None))
    monkeypatch.setattr(sidecar.shrinkage, "configure_model", lambda *args: None)
    assert not sidecar.run_is_complete(spec)
    with pytest.raises(RuntimeError, match="No completed fits"):
        sidecar.analyze([spec], tmp_path)
    rows = pd.read_csv(tmp_path / "incomplete_runs.csv")
    assert rows.reason.tolist() == ["empty_optimization_history"]
