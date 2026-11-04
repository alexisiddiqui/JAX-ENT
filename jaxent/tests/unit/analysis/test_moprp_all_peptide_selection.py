from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parents[3]/"examples"
sys.path.insert(0, str(EXAMPLES/"2_CrossValidation/analysis"))
import rescore_moprp_all_peptide_selection as all_peptide
from plot_moprp_policy_sidecars import selection_label
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters


@pytest.mark.parametrize("fitted", [False, True])
def test_all_peptide_scores_include_train_peptides_and_use_saved_parameters(monkeypatch, fitted):
    runner = all_peptide.bv if fitted else all_peptide.fixed
    parameters = [BV_Model_Parameters(bv_bc=jnp.array([bc]), timepoints=(1.,))
        for bc in (.35, .45 if fitted else .35)]
    states = [SimpleNamespace(step=step, params=SimpleNamespace(model_parameters=(p,)),
        losses=SimpleNamespace(val_losses=[loss]))
        for step, p, loss in zip((100, 200), parameters, (0., .125), strict=True)]
    seen = []
    def predict(weights, params):
        seen.append(float(params.bv_bc.item()))
        return np.array([[1., 0.]]) if weights[0] else np.array([[0., .5]])
    context = (None, None, SimpleNamespace(params=parameters[0]), None, None, None, None, None, None, predict)
    dataset = np.eye(2), np.zeros((2, 1)), np.array([[10.], [1.]]), [1], [0, 1]
    monkeypatch.setattr(all_peptide, "full_dataset", lambda *args: (context, dataset))
    monkeypatch.setattr(all_peptide.fixed, "load_optimization_history_from_file", lambda path: None)
    entries = [("trajectory", s, w) for s, w in zip(states, (np.array([1., 0.]), np.array([0., 1.])), strict=True)]
    monkeypatch.setattr(runner, "saved_candidates", lambda history: iter(entries))
    source = pd.DataFrame(dict(candidate_index=[0, 1], candidate_kind=["trajectory"]*2,
        step=[100, 200], val_mse=[0., .25], val_weighted_mse=[0., .25], ess_percent=[50., 50.],
        bv_bc=[float(p.bv_bc.item()) for p in parameters], bv_bh=[2., 2.]))
    spec = SimpleNamespace(saved_history="history.hdf5", run_id="test")
    scores = pd.DataFrame(all_peptide.score_run(spec, source, runner, {}, {}))
    np.testing.assert_allclose(scores.all_peptide_weighted_mse, [5., .125])
    np.testing.assert_allclose(scores.all_peptide_mse, [.5, .125])
    assert scores.validation_replay_absolute_error.max() == 0
    assert seen == [float(p.bv_bc.item()) for p in parameters]
    candidates = source.assign(run_id="test").merge(scores, on=all_peptide.IDENTITY)
    assert all_peptide.select_trajectories(candidates).candidate_index.item() == 1
    assert source.loc[source.val_weighted_mse.idxmin(), "candidate_index"] == 0


def test_all_peptide_selector_changes_bars_and_bv_penalty_without_changing_val_scores():
    rows = []
    for method, edge, graph in (("maxent", "none", 0), ("omc", "work_scale", 0),
            ("omc", "dihedral_rmsd", 0), ("omc", "work_scale", 20), ("omc", "dihedral_rmsd", 20)):
        for i in (0, 1):
            rows.append(dict(run_id=f"{method}_{edge}_{graph}_{i}", ensemble="AF2_MSAss", method=method,
                edge_type=edge, graph_k=graph, split_type="spatial", split_idx=0, strength=1., bandwidth=10.,
                bv_reg_scaling=.5+i*.5, step=100, candidate_index=i, val_weighted_mse=1.+i,
                all_peptide_weighted_mse=2.-i, selection_metric=all_peptide.SCORE))
    frame = pd.DataFrame(rows)
    original = frame.copy(deep=True)
    assert all_peptide.select_across_parameters(frame).candidate_index.eq(0).all()
    selected = all_peptide.select_across_parameters(frame, all_peptide.SCORE)
    assert selected.candidate_index.eq(1).all()
    assert set(selected.panel_graph_k) == {0, 20}
    assert all_peptide.profile_bv_penalties(frame, all_peptide.SCORE).bv_reg_scaling.eq(1.).all()
    assert selection_label(selected) == "all-peptide"
    pd.testing.assert_frame_equal(frame, original)


def test_full_dataset_counts_overlapping_split_peptides_once(monkeypatch):
    full = [SimpleNamespace(top=SimpleNamespace(fragment_index=i)) for i in range(2)]
    train, val = full, [full[1]]
    matrix = np.eye(2)
    observed = np.array([[[.1]], [[.2]]])
    validation = SimpleNamespace(residue_feature_ouput_mapping=SimpleNamespace(todense=lambda: matrix[[1]]),
        y_true=observed[[1]])
    all_data = SimpleNamespace(residue_feature_ouput_mapping=SimpleNamespace(todense=lambda: matrix), y_true=observed)
    context = (None, None, None, train, val, SimpleNamespace(val=validation))
    runner = SimpleNamespace(context=lambda *args: context)
    monkeypatch.setattr(all_peptide.fixed.loading, "load_experimental_data", lambda *args: (train, val, full, None))
    monkeypatch.setattr(all_peptide.fixed, "create_data_loaders", lambda *args: SimpleNamespace(test=all_data))
    monkeypatch.setattr(all_peptide.fixed, "aligned_file_weights", lambda data: np.array([[float(d.top.fragment_index+1)] for d in data]))
    spec = SimpleNamespace(ensemble="AF2_MSAss", split_type="spatial", split_idx=0,
        frame_averaging_mode="rate", datasplit_dir="splits")
    _, (mapping, truth, weights, val_indices, ids) = all_peptide.full_dataset(spec, runner, {}, {})
    assert ids == [0, 1] and val_indices == [1]
    assert mapping.shape == (2, 2) and truth.shape == weights.shape == (2, 1)


def test_read_only_worker_shards_preserve_every_history_and_split():
    specs = [SimpleNamespace(ensemble=ens, split_type=split, split_idx=i, run_id=f"{ens}_{split}_{i}_{j}")
        for ens in ("AF2_MSAss", "AF2_filtered") for split in ("spatial", "sequence_cluster")
        for i in range(3) for j in range(1+i)]
    shards = all_peptide.worker_shards(specs, 10)
    actual = [s.run_id for shard in shards for s in shard]
    assert len(shards) <= 10 and all(shards)
    assert len(actual) == len(set(actual)) == len(specs)
    assert set(actual) == {s.run_id for s in specs}
