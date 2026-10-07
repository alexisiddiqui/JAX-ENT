from pathlib import Path
from types import SimpleNamespace
import sys
import json

import numpy as np
import pandas as pd
import pytest
from MDAnalysis.analysis.rms import rmsd

HERE = Path(__file__).resolve().parents[3] / "examples/1_IsoValidation_OMass/fitting/jaxENT"
sys.path.insert(0, str(HERE))
from iso_sidecar_geometry import median_scale, pairwise_ca_rmsd
from sidecar_selection import select_best_rows
import run_iso_policy_sidecars as sidecar


@pytest.mark.parametrize("effective", [
    {"execution_mode": "native_strength_batch", "frozen_forward_cache": False},
    {"execution_mode": "compiled", "frozen_forward_cache": True},
    {"execution_mode": "native_strength_batch", "omc_diagonal_policy": "excluded_exact_zero_terms"},
])
def test_accelerated_histories_are_not_reused_even_if_replay_is_finite(tmp_path, monkeypatch, effective):
    history_path = tmp_path/"history.hdf5"
    history_path.touch()
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps({"effective_settings": effective}))
    spec = SimpleNamespace(saved_history=history_path, saved_config=config_path, method="maxent")
    monkeypatch.setattr(sidecar.base.source, "load_optimization_history_from_file",
        lambda path: SimpleNamespace(states=[SimpleNamespace(losses=SimpleNamespace(total_train_loss=1.))]))
    assert not sidecar.completed(spec)


def test_original_finite_history_can_be_reused(tmp_path, monkeypatch):
    history_path = tmp_path/"history.hdf5"
    history_path.touch()
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps({"effective_settings": {"execution_mode": "compiled"}}))
    spec = SimpleNamespace(saved_history=history_path, saved_config=config_path, method="maxent")
    monkeypatch.setattr(sidecar.base.source, "load_optimization_history_from_file",
        lambda path: SimpleNamespace(states=[SimpleNamespace(losses=SimpleNamespace(total_train_loss=1.))]))
    assert sidecar.completed(spec)


def test_median_positive_unique_pairs_excludes_zero_and_diagonal():
    raw = np.array([[0, 0, 2, 4], [0, 0, 6, 8], [2, 6, 0, 10], [4, 8, 10, 0]], dtype=float)
    scaled, median = median_scale(raw)
    assert median == 6
    np.testing.assert_allclose(scaled, raw/6)
    doubled, scale = median_scale(13*raw)
    np.testing.assert_allclose(doubled, scaled)
    assert scale == 78


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.ones((3, 3)),
    np.array([[0, -1], [-1, 0]]), np.array([[0, 1], [2, 0]])])
def test_invalid_distance_matrices_fail(matrix):
    with pytest.raises(ValueError):
        median_scale(matrix)


def test_pairwise_optimal_rmsd_matches_independent_qcp():
    rng = np.random.default_rng(20261004)
    xyz = rng.normal(size=(5, 9, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    xyz[1] = xyz[0] @ rotation + [3, -4, 2]
    distances = pairwise_ca_rmsd(xyz)
    for i in range(5):
        for j in range(i+1, 5):
            assert distances[i, j] == pytest.approx(rmsd(xyz[i], xyz[j], center=True, superposition=True), abs=1e-7)
    assert distances[0, 1] < 1e-7
    np.testing.assert_array_equal(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), np.zeros(5))


def test_selector_changes_checkpoint_and_keeps_each_trajectory_separate():
    frame = pd.DataFrame(dict(run_id=["a", "a", "b", "b"],
        val_mse=[1., 2., 2., 1.], val_closed_sigma_mse=[2., 1., 1., 2.],
        convergence_rank=[0, 1, 0, 1]))
    assert select_best_rows(frame, "val_mse").convergence_rank.tolist() == [0, 1]
    assert select_best_rows(frame).convergence_rank.tolist() == [1, 0]
    tie = frame.copy()
    tie["val_mse"] = 1.
    assert select_best_rows(tie, "val_mse").convergence_rank.tolist() == [0, 0]


def test_existing_candidate_pool_selects_fit_without_convergence_checkpoints():
    def state(step, weights):
        return SimpleNamespace(step=step, params=SimpleNamespace(frame_weight_simplex=np.asarray(weights)))
    history = SimpleNamespace(states=[state(100, [.4, .6])], convergence_states=[],
        best_state=state(20, [.5, .5]))
    candidates = list(sidecar.original.candidates(history))
    assert [kind for kind, _, _ in candidates] == ["trajectory", "running_best"]
    scores = pd.DataFrame([dict(run_id="a", candidate_index=i, candidate_kind=kind,
        step=s.step, val_mse=float(s.step), val_closed_sigma_mse=float(s.step))
        for i, (kind, s, _) in enumerate(candidates)])
    for selector in sidecar.SELECTORS:
        selected = select_best_rows(scores, selector)
        assert selected.step.tolist() == [20]
        assert selected.candidate_kind.tolist() == ["running_best"]


def test_partial_replicates_remain_in_connected_mean():
    from plot_iso_policy_sidecars import trace, plt
    rows = pd.DataFrame(dict(data_strength=[10., 10., 10., 100.], split_idx=[0, 1, 2, 0],
        recovery_percent=[20., 30., 40., 50.]))
    fig, ax = plt.subplots()
    try:
        trace(ax, rows, "recovery_percent", "black", "-")
        mean = ax.lines[-1]
        np.testing.assert_array_equal(mean.get_xdata(), [10., 100.])
        np.testing.assert_array_equal(mean.get_ydata(), [30., 50.])
    finally:
        plt.close(fig)


def test_grid_axes_independent_and_ids_include_split(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, features_dir=tmp_path, datasplit_dir=tmp_path,
        clustering_dir=tmp_path, n_splits=3, n_steps=5000, smoke=False,
        strengths=[10.**i for i in range(1, 7)], bandwidths=[10.**i for i in range(-3, 4)])
    paths = {m: tmp_path/f"{m}.npz" for m in sidecar.METRICS}
    specs = sidecar.build_specs(args, paths)
    assert len(specs) == len({s.run_id for s in specs}) == 882
    assert sum(s.method == "maxent" for s in specs) == 126
    assert sum(s.method == "omc" for s in specs) == 756
    assert {s.sigma_source for s in specs} == {"mse"}
    assert {s.uptake_mode for s in specs} == {"uptake"}
    assert {s.split_type for s in specs} == {"sequence_cluster", "spatial"}
    omc = [s for s in specs if s.method == "omc"]
    assert {s.ensemble for s in omc} == {"ISO_TRI"}
    assert all(s.graph_k == 0 and s.bandwidth == s.kernel_bandwidth and s.omc_strength == 1/s.sweep_maxent for s in omc)


def test_cached_kernel_uses_normalized_distances_and_rejects_mismatch(tmp_path):
    path = tmp_path/"rmsd.npz"
    matrix = np.array([[0., 1., 3.], [1., 0., 2.], [3., 2., 0.]])
    np.savez(path, normalized_distances=matrix, metric="rmsd")
    spec = SimpleNamespace(method="omc", distance_path=str(path), graph_metric="rmsd", bandwidth=2., omc_strength=.1)
    features = SimpleNamespace(features_shape=(10, 3))
    _, _, kernel, coefficient = sidecar.base.regularizer(spec, features, None)
    np.testing.assert_allclose(kernel.similarity, np.exp(-matrix**2/8), rtol=1e-6)
    assert coefficient == .1
    spec.graph_metric = "pyrosetta"
    with pytest.raises(ValueError, match="metric"):
        sidecar.base.regularizer(spec, features, None)

