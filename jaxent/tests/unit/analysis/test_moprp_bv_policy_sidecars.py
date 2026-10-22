from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[3]/"examples"
sys.path.insert(0, str(EXAMPLES/"3_CrossValidationBV/fitting/jaxENT"))
sys.path.insert(0, str(EXAMPLES/"3_CrossValidationBV/analysis"))
import run_moprp_bv_policy_sidecars as bv
from plot_moprp_bv_policy_sidecars import profile_bv_penalties
from plot_moprp_policy_model_selection_bars import select_across_parameters, summarize
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters


def campaign_args(tmp_path):
    source = tmp_path/"source/sequence_cluster"
    source.mkdir(parents=True)
    for ensemble in bv.fixed.ENSEMBLES:
        for penalty in bv.BV_PENALTIES:
            loss = dict(primary_loss="hdx_uptake_eye_MSE_loss", optimize_bv_params=True,
                regularization_losses=[dict(name="model_params_L1_loss")],
                bv_reg_scaling=penalty, normalize_bv_reg=False, maxent_scaling=100.)
            (source/f"{ensemble}_MSE_sequence_cluster_split000_maxent100_bvreg{penalty}_config.json").write_text(
                json.dumps(dict(loss_config=loss, opt_config={})))
    return SimpleNamespace(source_campaign=source.parent, n_splits=3, n_steps=5000,
        output_dir=tmp_path/"output", geometry_dir=tmp_path/"geometry", features_dir=tmp_path,
        datasplit_dir=tmp_path, clustering_dir=tmp_path, strengths=list(bv.fixed.STRENGTHS),
        bandwidths=list(bv.BANDWIDTHS), bv_reg_values=list(bv.BV_PENALTIES))


def test_production_and_smoke_cover_bv_penalties_and_both_graphs(tmp_path):
    args = campaign_args(tmp_path)
    specs = bv.build_specs(args)
    assert len(specs) == len({s.run_id for s in specs}) == 2520
    assert len(bv.fixed.experiment_specs(specs, "maxent")) == 168
    assert sum(s.method == "omc" and s.graph_k == 0 for s in specs) == 1176
    assert sum(s.method == "omc" and s.graph_k == 20 for s in specs) == 1176
    assert {s.bv_reg_scaling for s in specs} == {.5, 1.}
    assert all(s.geometry_dir == str(args.geometry_dir) for s in specs)
    args.n_splits, args.n_steps, args.strengths, args.bandwidths = 1, 20, [100.], [.001, 1., 1000.]
    smoke = bv.build_specs(args)
    assert len(smoke) == 56
    assert len(bv.fixed.experiment_specs(smoke, "maxent")) == 8
    assert len(bv.fixed.experiment_specs(smoke, "all_pairs")) == 32


@pytest.mark.parametrize("mode", ["rate", "linear_uptake"])
def test_requested_forward_grid_preserves_native_bv_settings_and_graph_geometry(tmp_path, mode):
    args = campaign_args(tmp_path)
    args.frame_averaging_mode = mode
    args.bandwidths = [10., 100., 1000.]
    specs = bv.build_specs(args)
    assert len(specs) == 1176
    assert sum(s.method == "maxent" for s in specs) == 168
    assert all(s.frame_averaging_mode == mode and f"MoPrP{bv.fixed.MODE_SUFFIXES[mode]}_" in s.run_id for s in specs)
    assert {s.bv_reg_scaling for s in specs} == {.5, 1.}
    assert all(s.geometry_dir == str(args.geometry_dir) for s in specs)


def test_additive_linear_bv_fits_only_slopes_with_original_l1_penalty(tmp_path):
    spec = replace(bv.build_specs(campaign_args(tmp_path))[0], frame_averaging_mode="linear_uptake")
    template = json.loads(Path(spec.template_config).read_text())
    template["loss_config"]["trainable_model_parameters"] = "frozenset({'raw_bv_bc', 'raw_bv_bh'})"
    loss = bv.loss_configuration(spec, template)
    assert loss.trainable_model_parameters == frozenset({"raw_bv_bc", "raw_bv_bh"})
    assert loss.optimize_bv_params and not loss.normalize_bv_reg
    assert loss.regularization_losses == [dict(name="model_params_L1_loss")]
    assert loss.bv_reg_scaling == spec.bv_reg_scaling


def test_revised_grid_preserves_retained_bv_jobs_and_adds_only_s_point_one(tmp_path):
    from dataclasses import asdict
    args = campaign_args(tmp_path)
    specs = [replace(s, reuse_history=f"/reference/{s.run_id}.hdf5", reuse_config=f"/reference/{s.run_id}.json")
        if s.method == "maxent" else s for s in bv.build_specs(args)]
    settings = dict(strengths=args.strengths[:], bandwidths=args.bandwidths[:], bv_reg_values=[.5, 1.])
    previous = dict(settings=settings, specs=[asdict(s) for s in specs], expected_fits=len(specs), verified_reuses=168)
    args.strengths = [.1]+args.strengths
    args.bandwidths = [10., 100., 1000.]
    revised = bv.fixed.revise_grid_manifest(previous, bv.build_specs(args),
        dict(settings, strengths=args.strengths, bandwidths=args.bandwidths))
    assert revised["expected_fits"] == 1344 and revised["verified_reuses"] == 168
    assert revised["grid_revisions"][-1]["new_fits"] == 168
    assert revised["grid_revisions"][-1]["inactive_fits"] == 1344
    old = {s.run_id: s for s in specs}
    current = [bv.RunSpec(**row) for row in revised["specs"]]
    assert all(s == old[s.run_id] for s in current if s.run_id in old)
    assert sum(s.method == "maxent" and s.strength == .1 for s in current) == 24
    assert all(s.bandwidth in (10., 100., 1000.) for s in current if s.method == "omc")
    with pytest.raises(ValueError, match="other campaign settings"):
        bv.fixed.revise_grid_manifest(previous, bv.build_specs(args),
            dict(settings, strengths=args.strengths, bandwidths=args.bandwidths, bv_reg_values=[1.]))


def state(bc, step=20):
    params = BV_Model_Parameters(bv_bc=jnp.array([bc]), bv_bh=jnp.array([2.]), timepoints=(1.,))
    return SimpleNamespace(step=step, params=SimpleNamespace(frame_weight_simplex=jnp.array([.5, .5]),
        model_parameters=(params,)), losses=SimpleNamespace(train_losses=[.5*(bc-2.)**2], val_losses=[.5*(bc-2.)**2]))


def test_candidate_identity_includes_fitted_bv_parameters():
    first, second = state(1.), state(2.)
    history = SimpleNamespace(states=[first], convergence_states=[first, second], best_state=second)
    entries = list(bv.saved_candidates(history))
    assert len(entries) == 2
    assert [kind for kind, _, _ in entries] == ["trajectory", "convergence"]
    assert [float(s.params.model_parameters[0].bv_bc.item()) for _, s, _ in entries] == [1., 2.]


def test_scoring_replays_each_checkpoint_with_its_own_bv_coefficients(tmp_path, monkeypatch):
    first, second = state(1.), state(2.)
    history = SimpleNamespace(states=[first, second], convergence_states=[], best_state=None)
    path = tmp_path/"history.hdf5"
    path.write_bytes(b"fixture")
    spec = bv.build_specs(campaign_args(tmp_path))[0]
    spec = replace(spec, n_steps=20, reuse_history=str(path))
    mapping = SimpleNamespace(todense=lambda: np.ones((1, 1)))
    dataset = SimpleNamespace(y_true=np.array([[[2.]]]), residue_feature_ouput_mapping=mapping)
    loader = SimpleNamespace(train=dataset, val=dataset)
    seen = []
    def predict(weights, params):
        seen.append(float(params.bv_bc.item()))
        return np.array([[params.bv_bc.item()]])
    native = SimpleNamespace(params=first.params.model_parameters[0])
    config = SimpleNamespace(scoring=SimpleNamespace(state_mapping={0: "Folded", 1: "PUF1"}))
    monkeypatch.setattr(bv, "context", lambda spec, cache: (
        None, None, native, None, None, loader, np.array([0, 1]), config, dict(Folded=.9, PUF1=.1), predict))
    monkeypatch.setattr(bv, "aligned_file_weights", lambda data: np.array([[3.]]))
    monkeypatch.setattr(bv, "load_optimization_history_from_file", lambda path: history)
    rows, audit = bv.score_history(spec, {})
    assert seen == [1., 2.]
    assert [row["val_weighted_mse"] for row in rows] == [3., 0.]
    assert [row["bv_bc"] for row in rows] == [1., 2.]
    assert audit["checkpoint_specific_bv_parameters_verified"]
    assert audit["native_loss_max_absolute_delta"] == 0


def test_protocol_remains_frozen_while_bv_coefficients_can_change():
    original = state(1.).params.model_parameters[0]
    changed = state(2.).params.model_parameters[0]
    bv.validate_parameters(changed, original)
    with pytest.raises(AssertionError, match="timepoints"):
        bv.validate_parameters(replace(changed, timepoints=(2.,)), original)


def test_native_bv_loss_and_optimizer_settings_are_preserved(tmp_path):
    spec = bv.build_specs(campaign_args(tmp_path))[0]
    template = json.loads(Path(spec.template_config).read_text())
    loss = bv.loss_configuration(spec, template)
    assert loss.optimize_bv_params and not loss.normalize_bv_reg
    assert loss.primary_loss == "hdx_uptake_eye_MSE_loss"
    assert loss.regularization_losses == [dict(name="model_params_L1_loss")]
    assert loss.bv_reg_scaling == spec.bv_reg_scaling
    assert loss.maxent_scaling == spec.strength
    assert bv.loss_configuration(replace(spec, method="omc"), template).maxent_scaling == 1.


def test_bv_penalties_are_selected_before_sweep_and_bar_replication():
    rows = []
    for split in bv.fixed.SPLITS:
        for index in range(3):
            for method, edge_type, graph_k in (("maxent", "none", 0),
                    ("omc", "work_scale", 0), ("omc", "dihedral_rmsd", 0),
                    ("omc", "work_scale", 20), ("omc", "dihedral_rmsd", 20)):
                for penalty, score in ((.5, 2.), (1., 1.)):
                    rows.append(dict(ensemble="AF2_MSAss", method=method, edge_type=edge_type,
                        graph_k=graph_k, split_type=split, split_idx=index, strength=100., bandwidth=1.,
                        bv_reg_scaling=penalty, val_weighted_mse=score, step=20, candidate_index=0,
                        run_id=f"{method}_{edge_type}_{graph_k}_{split}_{index}_{penalty}",
                        recovery_percent=40.+index, ess_percent=10.+index))
    candidates = pd.DataFrame(rows)
    profiled = profile_bv_penalties(candidates)
    assert len(profiled) == 30 and profiled.bv_reg_scaling.eq(1.).all()
    selected = select_across_parameters(candidates)
    assert len(selected) == 36 and selected.bv_reg_scaling.eq(1.).all()
    assert summarize(selected).n.eq(6).all()
