import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.examples.ATLAS_BV.analysis import omc_decoy_control as e

jax.config.update("jax_enable_x64", True)


def informative_data():
    z = np.array([[4.0, 7.0, 9.0, 11.0], [10.0, 5.0, 7.0, 3.0], [6.0, 10.0, 3.0, 8.0]])
    rates = np.array([0.5, 2.0, 0.8])
    w = np.array([0.6, 0.15, 0.2, 0.05])
    target = e.uptake(z, rates, w)
    kernels, sigmas = e.graph(z)
    return dict(
        log_pf=z,
        rates=rates,
        target=target,
        times=e.TIMES,
        kernels=kernels,
        sigmas=sigmas,
        scale=np.array(np.var(target) + 1e-8),
    )


def test_default_forward_parity_units_and_small_uptake():
    from jaxent.src.models.HDX.forward import BV_uptake_ForwardPass
    from jaxent.src.models.HDX.BV.features import BV_input_features
    from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters

    data = informative_data()
    w = np.array([0.6, 0.15, 0.2, 0.05])
    features = BV_input_features(
        heavy_contacts=jnp.asarray(data["log_pf"] @ w),
        acceptor_contacts=jnp.zeros(3),
        k_ints=jnp.asarray(data["rates"]),
    )
    params = BV_Model_Parameters(
        bv_bc=jnp.array([1.0]), bv_bh=jnp.array([0.0]), timepoints=tuple(e.TIMES)
    )
    observed = BV_uptake_ForwardPass()(features, params).uptake
    np.testing.assert_allclose(observed, data["target"], atol=2e-15)
    np.testing.assert_allclose(
        e.uptake(data["log_pf"], data["rates"] * 60, w, e.TIMES / 60), data["target"]
    )
    tiny = e.uptake(np.full((1, 2), 200.0), np.ones(1), np.ones(2) / 2)
    assert np.all(tiny > 0)  # expm1 preserves tiny values rather than rounding to zero.
    assert np.max(tiny) < 1e-70


def test_objective_matches_pair_sum_and_finite_difference_gradient():
    data = informative_data()
    x = np.array([0.7, -0.2, 0.1, -0.8])
    kernel = data["kernels"][2]

    def loss(v):
        return e.jax_objective(
            v,
            jnp.asarray(data["log_pf"]),
            jnp.asarray(data["rates"]),
            jnp.asarray(data["target"]),
            jnp.asarray(data["times"]),
            data["scale"],
            jnp.asarray(kernel),
            0.1,
            0,
        )

    w = np.asarray(jax.nn.softmax(x))
    expected = np.mean(
        (e.uptake(data["log_pf"], data["rates"], w) - data["target"]) ** 2
    ) / data["scale"] + 0.1 * e.regularisation(w, kernel)
    np.testing.assert_allclose(loss(x), expected, atol=1e-12)
    delta = np.eye(4) * 1e-5
    finite = np.array([(loss(x + d) - loss(x - d)) / (2e-5) for d in delta])
    np.testing.assert_allclose(
        jax.grad(loss)(jnp.asarray(x)), finite, rtol=1e-5, atol=1e-9
    )
    assert e.regularisation(np.ones(4) / 4, kernel) == 0


def test_empirical_internal_removal_preserves_relative_populations():
    labels = np.repeat([0, 1, 2], [60, 30, 10])
    w, masses = e.empirical_target(labels, 0)
    np.testing.assert_allclose(masses, [0, 0.75, 0.25])
    assert not w[labels == 0].any()
    np.testing.assert_allclose(w.sum(), 1)
    assert labels.shape == (100,)  # candidate inputs are not filtered or mutated.


def test_independent_partitions_and_unique_actual_representatives():
    rng = np.random.default_rng(19)
    vectors = np.concatenate([rng.normal(i * 20, 0.5, (60, 5)) for i in range(4)])
    reps, candidate_labels = e.partition_candidates(vectors)
    table, labels = e.partition_targets(vectors)
    assert len(reps) == len(set(reps)) == 100
    assert all(candidate_labels[r] == i for i, r in enumerate(reps))
    assert table.loc[table.selected, "k"].item() == 4
    assert set(labels) == {0, 1, 2, 3}
    # Calling target clustering does not alter the already frozen candidate partition.
    reps2, labels2 = e.partition_candidates(vectors)
    np.testing.assert_array_equal(reps, reps2)
    np.testing.assert_array_equal(candidate_labels, labels2)


def test_decoys_reproducible_crop_tile_and_marginals():
    z = np.arange(35.0).reshape(5, 7)
    for size in [3, 5, 13]:
        values, origin = e.donor_decoys(z, size, 25, 10)
        repeat, same = e.donor_decoys(z, size, 25, 10)
        np.testing.assert_array_equal(values, repeat)
        for j in range(25):
            tiled = np.tile(z[:, origin["frame"][j]], origin["repeats"][j])
            np.testing.assert_array_equal(
                values[:, j], tiled[origin["offset"][j] : origin["offset"][j] + size]
            )
        assert values.shape == (size, 25)
    random, frames = e.random_decoys(z, 25, 10)
    np.testing.assert_array_equal(random, z[np.arange(5)[:, None], frames])
    assert (frames[0] != frames[1]).any()
    np.testing.assert_array_equal(random, e.random_decoys(z, 25, 10)[0])


def test_bandwidths_and_native_edges_do_not_change_on_append():
    z = informative_data()["log_pf"]
    native, sigmas = e.graph(z)
    expanded, frozen = e.graph(np.column_stack([z, z[:, :2] * 20]), sigmas)
    np.testing.assert_array_equal(sigmas, frozen)
    np.testing.assert_array_equal(expanded[:, :4, :4], native)
    assert (
        expanded[:, ~np.eye(6, dtype=bool)].mean(axis=1)
        != native[:, ~np.eye(4, dtype=bool)].mean(axis=1)
    ).any()


def test_ess_native_mass_and_variable_candidate_count():
    a = e.ess_stats(np.ones(100) / 100, 100)
    b = e.ess_stats(np.ones(125) / 125, 100)
    np.testing.assert_allclose([a["native_ess_fraction"], b["native_ess_fraction"]], 1)
    np.testing.assert_allclose([a["ess"], b["ess"]], [100, 125])
    np.testing.assert_allclose(b["native_mass"], 0.8)
    assert e.ess_stats(np.array([0.0, 0.0, 1.0]), 2)["native_ess"] == 0


def test_preflight_rejects_tiny_saturated_and_uninformative_targets():
    weights = [np.ones(4) / 4, np.array([1.0, 0.0, 0.0, 0.0])]
    tiny, _ = e.physical_check(np.full((3, 4), 200.0), np.ones(3), weights)
    assert not tiny["passed"]
    saturated, _ = e.physical_check(np.zeros((3, 4)), np.ones(3) * 1e6, weights)
    assert not saturated["passed"]
    data = informative_data()
    good, _ = e.physical_check(data["log_pf"], data["rates"], weights)
    assert good["passed"]
    bad, _ = e.physical_check(data["log_pf"], data["rates"], weights, contact_ok=False)
    assert not bad["passed"]
    equal, _ = e.physical_check(data["log_pf"], data["rates"], [weights[0], weights[0]])
    assert not equal["passed"]


def test_nonlinear_fit_reduces_data_error_and_preserves_simplex():
    data = informative_data()
    fit = e.fit_case(data, checkpoints=(100, 300), window=25)
    assert fit["weights"].shape == (13, 4)
    assert fit["initialisation_weights"].shape == (2, 13, 4)
    np.testing.assert_allclose(fit["weights"].sum(axis=1), 1, atol=1e-12)
    before = np.mean(
        (e.uptake(data["log_pf"], data["rates"], np.ones(4) / 4) - data["target"]) ** 2
    )
    after = np.mean(
        (e.uptake(data["log_pf"], data["rates"], fit["weights"][-1]) - data["target"])
        ** 2
    )
    assert after < before * 0.01
    assert np.isfinite(fit["grad_norm"]).all()


def test_failed_gate_cannot_start_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "load_manifest", lambda output: dict(fit_gate_passed=False))

    def forbidden(*args, **kwargs):
        pytest.fail("Worker must never run for failed physical preflight")

    monkeypatch.setattr(e, "run_worker", forbidden)
    e.write_json(tmp_path / "preflight.json", dict(reasons=["Tiny uptake"]))
    e.run(tmp_path, workers=10)
    state = json.loads((tmp_path / "run_status.json").read_text())
    assert state["fitted_cases"] == 0
    assert state["status"] == "blocked_physical_preflight"


def test_prepared_artifact_corruption_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "code_identity", lambda: {"code": "frozen"})
    e.save_npz(tmp_path / "input.npz", a=np.ones(2))
    e.write_json(
        tmp_path / "manifest.json",
        dict(
            code={"code": "frozen"},
            input_hashes={},
            artifacts={"input.npz": e.digest(tmp_path / "input.npz")},
        ),
    )
    e.load_manifest(tmp_path)
    e.save_npz(tmp_path / "input.npz", a=np.zeros(2))
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        e.load_manifest(tmp_path)


def test_report_renders_fitted_populations_and_diagnostics(tmp_path, monkeypatch):
    import pandas as pd
    from jaxent.examples.ATLAS_BV.analysis.omc_decoy_report import report

    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(3), 40)
    native_labels = labels[:100]
    z = informative_data()["log_pf"][:, labels]
    core = z[:, :100]
    rates = informative_data()["rates"]
    target_weights = np.ones(120) / 120
    target = e.uptake(z, rates, target_weights)
    masses = np.ones(3) / 3
    projection = masses[native_labels] / np.bincount(native_labels)[native_labels]
    kernels, sigmas = e.graph(core)
    folder = tmp_path / "cases/baseline"
    e.save_npz(
        folder / "input.npz",
        log_pf=core,
        rates=rates,
        times=e.TIMES,
        target=target,
        target_weights=target_weights,
        target_masses=masses,
        projection=projection,
        kernels=kernels,
        sigmas=sigmas,
        native_labels=native_labels,
        decoy=np.zeros(100, bool),
        scale=np.array(0.1),
    )
    candidate_labels = np.arange(120) % 100
    e.save_npz(
        tmp_path / "source.npz",
        log_pf=z,
        rates=rates,
        heavy=z / 0.35,
        acceptor=np.zeros_like(z),
        residues=np.arange(3) + 2,
        target_labels=labels,
        native_labels=native_labels,
        target_predictions=target[None],
        candidate_labels=candidate_labels,
        candidate_indices=np.arange(100),
        candidate_frames=np.arange(100),
        xyz=rng.normal(size=(120, 4, 3)),
        structural_distance=rng.uniform(size=(120, 100)),
    )
    pd.crosstab(candidate_labels, labels).to_csv(
        tmp_path / "candidate_target_mixing.csv"
    )
    pd.DataFrame(
        [dict(k=3, silhouette=0.5, min_size=40, eligible=True, selected=True)]
    ).to_csv(tmp_path / "target_clustering.csv", index=False)
    audit = dict(
        contact_passed=True,
        rates_passed=True,
        contacts=[dict(feature="heavy", passed=True)],
    )
    e.write_json(tmp_path / "feature_audit.json", dict(recipient=audit, donor=audit))
    e.write_json(
        tmp_path / "preflight.json",
        dict(
            passed=True,
            uptake_min=float(target.min()),
            uptake_max=float(target.max()),
            informative_reference_observations=15,
            target_names=["full_reference"],
            target_max_abs_change=[0.0],
            contact_method=dict(bv_bc=0.35, bv_bh=2.0),
        ),
    )
    manifest = dict(
        code={},
        input_hashes={},
        artifacts={},
        fit_gate_passed=True,
        cases=[dict(name="baseline", kind="baseline", removed=-1, informative=True)],
    )
    e.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(e, "code_identity", lambda: {})
    e.save_npz(
        folder / "fit.npz",
        weights=np.tile(projection, (13, 1)),
        converged=np.ones(13, bool),
        objective=np.zeros(13),
        grad_norm=np.zeros(13),
        relative_change=np.zeros(13),
        steps=np.ones(13, int) * 1000,
        initialisation_objective_gap=np.zeros(13),
        initialisation_weight_tv=np.zeros(13),
    )
    e.write_json(
        folder / "complete.json",
        dict(
            identity=e.digest(tmp_path / "manifest.json"),
            sha256=e.digest(folder / "fit.npz"),
        ),
    )
    report(tmp_path)
    fits = pd.read_csv(tmp_path / "fits.csv")
    np.testing.assert_allclose(fits.population_tv, 0, atol=1e-14)
    assert len(fits) == 13
    assert (tmp_path / "figures/populations_baseline.svg").exists()
    assert len(pd.read_csv(tmp_path / "residuals.csv")) == 13 * 15
    assert "Physical preflight passed" in (tmp_path / "index.html").read_text()
    # Re-rendering derived files must not invalidate prepared inputs.
    report(tmp_path)
    assert len(pd.read_csv(tmp_path / "mse_selected.csv")) == 3


def test_current_experiment_overrides_do_not_change_atlas_protocol():
    original = e.load_config()
    current = e.experiment_config()
    assert current["protocol"]["contact_mode"] == "smooth_cutoff"
    assert current["protocol"]["switch_scale_nc_angstrom"] == 0.5
    assert current["protocol"]["switch_scale_nh_angstrom"] == 0.5
    assert e.load_config() == original
    assert original["protocol"]["contact_mode"] == "bradshaw_switch"
