from pathlib import Path
from dataclasses import asdict, replace
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[3]/"examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(HERE))
from moprp_sidecar_support import raw_weighted_mse, regularizer
from run_moprp_policy_sidecars import BANDWIDTHS, STRENGTHS, build_specs, experiment_specs, extend_bandwidth_manifest


def test_revised_active_grid_keeps_original_fits_and_adds_84_new_jobs(tmp_path):
    from run_moprp_policy_sidecars import revise_grid_manifest, RunSpec
    source = tmp_path/"source/sequence_cluster"
    source.mkdir(parents=True)
    for ensemble in ("AF2_MSAss", "AF2_filtered"):
        (source/f"{ensemble}_MSE_sequence_cluster_split000_maxent100_config.json").write_text("{}")
    args = SimpleNamespace(source_campaign=source.parent, n_splits=3, n_steps=5000, output_dir=tmp_path,
        features_dir=tmp_path, datasplit_dir=tmp_path, clustering_dir=tmp_path,
        strengths=list(STRENGTHS), bandwidths=[.001, .01, .1, 1., 10., 100., 1000.])
    specs = [replace(s, reuse_history=f"/original/{s.run_id}.hdf5", reuse_config=f"/original/{s.run_id}.json")
        if s.method == "maxent" else s for s in build_specs(args)]
    settings = dict(strengths=args.strengths[:], bandwidths=args.bandwidths[:], n_steps=5000)
    previous = dict(settings=settings, specs=[asdict(s) for s in specs], expected_fits=1260)
    args.strengths = [.1]+args.strengths
    args.bandwidths = [10., 100., 1000.]
    revised = revise_grid_manifest(previous, build_specs(args), dict(settings,
        strengths=args.strengths, bandwidths=args.bandwidths))
    assert revised["expected_fits"] == 672 and revised["verified_reuses"] == 84
    assert revised["grid_revisions"][-1]["new_fits"] == 84
    assert revised["grid_revisions"][-1]["inactive_fits"] == 672
    old = {s.run_id: s for s in specs}
    current = [RunSpec(**row) for row in revised["specs"]]
    assert all(s == old[s.run_id] for s in current if s.run_id in old)
    assert sum(s.method == "maxent" and s.strength == .1 for s in current) == 12


def test_raw_file_weights_are_not_squared_inverted_or_sum_normalized():
    prediction = np.array([[1., 2.], [3., 4.]])
    weights = np.array([[1., 2.], [3., 4.]])
    assert raw_weighted_mse(prediction, np.zeros_like(prediction), weights) == 25.
    assert raw_weighted_mse(prediction, np.zeros_like(prediction), 10*weights) == 250.
    with pytest.raises(ValueError, match="align"):
        raw_weighted_mse(prediction, prediction, weights[:, :1])


def test_rate_fits_have_separate_identity_and_old_full_configs_remain_complete(tmp_path):
    import json
    from run_moprp_policy_sidecars import RunSpec, completed
    spec = RunSpec(ensemble="AF2_MSAss", method="omc", edge_type="work_scale", graph_k=20,
        split_type="spatial", split_idx=0, strength=1., bandwidth=10., n_steps=5000,
        output_dir=str(tmp_path), features_dir="features", datasplit_dir="splits", clustering_dir="clusters",
        geometry_dir="geometry", template_config="template.json")
    spec.run_dir.mkdir(parents=True)
    spec.history_path.write_bytes(b"fixture")
    legacy = asdict(spec)
    legacy.pop("frame_averaging_mode")
    spec.config_path.write_text(json.dumps(dict(moprp_sidecar=dict(spec=legacy))))
    assert completed(spec)
    rate = replace(spec, frame_averaging_mode="rate")
    assert rate.run_id != spec.run_id and rate.history_path != spec.history_path
    assert not completed(rate)
    linear = replace(spec, frame_averaging_mode="linear_uptake")
    assert len({spec.run_id, rate.run_id, linear.run_id}) == 3
    assert not completed(linear)


def test_additive_linear_model_uses_mean_contacts_and_matching_minute_units(monkeypatch):
    import run_moprp_policy_sidecars as runner
    from jaxent.src.custom_types.key import m_key
    from jaxent.src.models.HDX.BV.features import BV_input_features
    monkeypatch.setattr(runner, "model", lambda: SimpleNamespace(params=SimpleNamespace(timepoints=(1., 2.))))
    native = runner.configured_model("linear_uptake")
    assert native.params.kint_unit == "min^-1" and native.params.time_unit == "min"
    features = BV_input_features(heavy_contacts=jnp.array([[1., 3.]]),
        acceptor_contacts=jnp.array([[0., 1.]]), k_ints=jnp.array([.4]))
    weights = jnp.array([.25, .75])
    actual = native.forward[m_key("HDX_peptide")].average_frames(features, native.params, weights).uptake
    rate = .4*np.exp(-.35*2.5-2.*.75)
    np.testing.assert_allclose(actual[:, 0], 1-np.exp(-np.array([1., 2.])*rate), rtol=2e-6)
    runner.validate_protocol(native.params, native.params)
    with pytest.raises(ValueError, match="rate/time units"):
        runner.validate_protocol(replace(native.params, kint_unit="s^-1"), native.params)
    with pytest.raises(AssertionError, match="interval offsets"):
        runner.validate_protocol(replace(native.params, interval_offsets=jnp.array([.1, 0.])), native.params)


def test_full_and_smoke_grids_cover_requested_models_splits_and_bandwidths(tmp_path):
    source = tmp_path/"source/sequence_cluster"
    source.mkdir(parents=True)
    for ensemble in ("AF2_MSAss", "AF2_filtered"):
        (source/f"{ensemble}_MSE_sequence_cluster_split000_maxent100_config.json").write_text("{}")
    args = SimpleNamespace(source_campaign=source.parent, n_splits=3, n_steps=5000, output_dir=tmp_path,
        features_dir=tmp_path, datasplit_dir=tmp_path, clustering_dir=tmp_path,
        strengths=list(STRENGTHS), bandwidths=list(BANDWIDTHS))
    specs = build_specs(args)
    assert len(specs) == len({s.run_id for s in specs}) == 756
    assert sum(s.method == "maxent" for s in specs) == 84
    assert sum(s.method == "omc" for s in specs) == 672
    assert all(s.ensemble == "AF2_MSAss" for s in specs if s.method == "omc")
    assert {s.graph_k for s in specs if s.method == "omc"} == {0, 20}
    assert len(experiment_specs(specs, "maxent")) == 84
    assert len(experiment_specs(specs, "all_pairs")) == 420
    assert len(experiment_specs(specs, "knn20")) == 420
    assert len(experiment_specs(specs, "all")) == 756
    args.n_splits, args.n_steps, args.strengths = 1, 20, [100.]
    smoke = build_specs(args)
    assert len(smoke) == 36
    assert len({(s.graph_k, s.edge_type, s.bandwidth, s.split_type) for s in smoke if s.method == "omc"}) == 32


@pytest.mark.parametrize("edge_type", ["work_scale", "dihedral_rmsd"])
def test_both_edge_types_share_logpf_neighbours_and_preserve_original_pairwise_energy(tmp_path, edge_type):
    from jaxent.src.opt.loss.original_omc_laplacian import omc_graph_energy
    distances = np.array([[0., 1., 4.], [1., 0., 2.], [4., 2., 0.]])
    np.savez(tmp_path/f"{edge_type}.npz", normalized_distances=distances)
    np.savez(tmp_path/"logpf_knn20.npz", edge_sources=[0, 1], edge_targets=[1, 2], n_nodes=3, k=20)
    spec = SimpleNamespace(method="omc", edge_type=edge_type, graph_k=20, geometry_dir=str(tmp_path),
        bandwidth=.5, strength=100.)
    _, loss, graph, strength = regularizer(spec)
    np.testing.assert_array_equal(graph.edge_sources, [0, 1])
    np.testing.assert_array_equal(graph.edge_targets, [1, 2])
    weights = jnp.array([.2, .3, .5])
    dense = np.zeros((3, 3))
    dense[[0, 1], [1, 2]] = np.asarray(graph.edge_weights)
    dense += dense.T
    actual, _ = loss(SimpleNamespace(params=SimpleNamespace(frame_weight_simplex=weights)), graph, 0)
    assert float(actual) == pytest.approx(float(omc_graph_energy(weights, dense)), rel=1e-6)
    assert strength == .01


def test_periodic_dihedral_distance_handles_angle_wrap():
    from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import periodic_dihedral_pair_distance
    angles = np.deg2rad(np.array([[179., -179.], [179., -179.]]))
    distance = periodic_dihedral_pair_distance(angles, np.array([0]), np.array([1]))
    np.testing.assert_allclose(distance, [2.], atol=1e-10)


def test_bandwidth_extension_preserves_every_existing_job_and_verified_reuse(tmp_path):
    source = tmp_path/"source/sequence_cluster"
    source.mkdir(parents=True)
    for ensemble in ("AF2_MSAss", "AF2_filtered"):
        (source/f"{ensemble}_MSE_sequence_cluster_split000_maxent100_config.json").write_text("{}")
    args = SimpleNamespace(source_campaign=source.parent, n_splits=3, n_steps=5000, output_dir=tmp_path,
        features_dir=tmp_path, datasplit_dir=tmp_path, clustering_dir=tmp_path,
        strengths=list(STRENGTHS), bandwidths=list(BANDWIDTHS))
    old_specs = [replace(s, reuse_history=f"/original/{s.run_id}.hdf5", reuse_config=f"/original/{s.run_id}.json")
        if s.method == "maxent" else s for s in build_specs(args)]
    settings = dict(bandwidths=list(BANDWIDTHS), n_steps=5000)
    previous = dict(settings=settings, specs=[asdict(s) for s in old_specs], expected_fits=756, verified_reuses=84)
    args.bandwidths += [10., 100., 1000.]
    generated = build_specs(args)
    new_settings = dict(settings, bandwidths=args.bandwidths)
    extended = extend_bandwidth_manifest(previous, generated, new_settings)
    assert extended["expected_fits"] == 1260
    assert extended["extensions"][-1]["new_fits"] == 504
    assert extended["extensions"][-1]["added_bandwidths"] == [10., 100., 1000.]
    from run_moprp_policy_sidecars import RunSpec
    new_specs = {s.run_id: s for s in (RunSpec(**row) for row in extended["specs"])}
    assert all(new_specs[s.run_id] == s for s in old_specs)
    assert sum(bool(s.reuse_history) for s in new_specs.values()) == 84
    with pytest.raises(ValueError, match="any other campaign setting"):
        extend_bandwidth_manifest(previous, generated, dict(new_settings, n_steps=1000))
    with pytest.raises(ValueError, match="strictly add"):
        extend_bandwidth_manifest(previous, generated, dict(new_settings, bandwidths=[.001, 10.]))
