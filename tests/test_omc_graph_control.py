import numpy as np
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis import omc_graph_control as g
from jaxent.examples.ATLAS_BV.analysis import omc_decoy_control as e
from jaxent.examples.ATLAS_BV.analysis import omc_graph_report as report


def example():
    z = np.array([[4.0, 7.0, 9.0, 11.0], [10.0, 5.0, 7.0, 3.0], [6.0, 10.0, 3.0, 8.0]])
    rates = np.array([0.5, 2.0, 0.8])
    w = np.array([0.6, 0.15, 0.2, 0.05])
    target = e.uptake(z, rates, w)
    kernel, sigma = e.graph(z)
    return dict(
        log_pf=z,
        rates=rates,
        times=e.TIMES,
        target=target,
        kernels=kernel,
        sigmas=sigma,
        scale=np.array(np.var(target) + 1e-8),
    )


def test_profile_distance_preserves_cancelled_patterns():
    z = np.array([[2.0, 18.0], [10.0, 10.0], [18.0, 2.0]])
    assert abs(z.mean(axis=0)[0] - z.mean(axis=0)[1]) == 0
    d = g.profile_distance(z)
    np.testing.assert_allclose(d[0, 1], np.sqrt((16**2 + 16**2) / 3))
    np.testing.assert_array_equal(np.diag(d), 0)


def test_structural_distance_invariant_to_independent_rigid_transforms():
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(8, 6, 3))
    changed = xyz.copy()
    for frame in range(len(xyz)):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        changed[frame] = xyz[frame] @ q + rng.normal(size=3) * 100
    np.testing.assert_allclose(
        g.structural_distance(xyz), g.structural_distance(changed), atol=1e-13
    )


def test_energy_frame_join_and_invariances():
    archive = dict(
        frame=np.array([30, 10, 50, 20]), ref2015__total=np.array([7.0, 2.0, 9.0, 3.0])
    )
    energy = g.energy_join(archive, np.array([20, 50, 10]))
    np.testing.assert_array_equal(energy, [3.0, 9.0, 2.0])
    ref = np.ones((6, 3, 3)) * 0.2
    for k in ref:
        np.fill_diagonal(k, 1)
    distance = abs(energy[:, None] - energy[None, :])
    sigmas = g.bandwidths(distance, 3)
    original = g.matched_kernels(distance, sigmas, ref)["kernels"]
    shifted = energy + 123
    np.testing.assert_allclose(abs(shifted[:, None] - shifted[None, :]), distance)
    scaled = g.matched_kernels(distance * 7, g.bandwidths(distance * 7, 3), ref)[
        "kernels"
    ]
    np.testing.assert_allclose(original, scaled, atol=1e-14)
    with pytest.raises(ValueError, match="Missing"):
        g.energy_join(archive, np.array([99]))
    with pytest.raises(ValueError, match="unique"):
        g.energy_join({**archive, "frame": np.array([30, 10, 50, 10])}, np.array([10]))
    with pytest.raises(ValueError, match="Nonfinite"):
        g.energy_join(
            {**archive, "ref2015__total": np.array([1, np.nan, 2, 3])}, np.array([10])
        )


def test_kernel_coupling_and_native_bandwidth_freeze():
    data = example()
    distance = g.profile_distance(data["log_pf"])
    sigma = g.bandwidths(distance, 4)
    matched = g.matched_kernels(distance, sigma, data["kernels"])
    off = ~np.eye(4, dtype=bool)
    np.testing.assert_allclose(
        matched["kernels"][:, off].mean(axis=1),
        data["kernels"][:, off].mean(axis=1),
        rtol=1e-13,
    )
    np.testing.assert_array_equal(matched["kernels"][:, np.arange(4), np.arange(4)], 1)
    expanded = g.profile_distance(
        np.column_stack([data["log_pf"], np.ones((3, 2)) * 100])
    )
    np.testing.assert_array_equal(g.bandwidths(expanded, 4), sigma)
    np.testing.assert_array_equal(expanded[:4, :4], distance)
    # Every off-diagonal edge is scaled by the same factor, preserving graph shape.
    np.testing.assert_allclose(
        matched["kernels"][:, off],
        matched["raw_kernels"][:, off] * matched["scaling_factor"][:, None],
    )


def test_degenerate_graph_and_external_geometry_rejected():
    with pytest.raises(ValueError, match="positive"):
        g.bandwidths(np.zeros((4, 4)), 4)
    with pytest.raises(ValueError, match="Invalid"):
        g.matched_kernels(np.ones((4, 4)), np.ones(6), np.ones((6, 4, 4)))
    assert all(
        g.applicable("profile_logpf", kind)
        for kind in ["baseline", "internal", "random", "donor"]
    )
    for graph in ["structure", "ref2015_total"]:
        assert g.applicable(graph, "internal")
        assert not g.applicable(graph, "random")
        assert not g.applicable(graph, "donor")


def test_six_arm_adapter_matches_original_optimizer():
    data = example()
    fit = g.fit_graph(data, checkpoints=(100, 300), window=25)
    old = e.fit_case(data, checkpoints=(100, 300), window=25)
    g.validate_fit(fit, 4)
    for key in ["weights", "objective", "converged", "relative_change", "steps"]:
        np.testing.assert_allclose(fit[key], old[key][:6], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        fit["initialisation_weights"],
        old["initialisation_weights"][:, :6],
        rtol=1e-6,
        atol=1e-9,
    )


def test_pairing_excludes_unresolved_arms_and_handles_no_new_fits():
    rows = []
    for graph in ["scalar_logpf", "profile_logpf"]:
        for q in [0.02, 0.04]:
            rows.append(
                dict(
                    case="baseline",
                    kind="baseline",
                    graph=graph,
                    quantile=q,
                    converged=not (graph == "profile_logpf" and q == 0.04),
                    population_tv=0.1 if graph == "scalar_logpf" else 0.05,
                    retained_population_tv=0.1,
                    decoy_mass=0.0,
                    mse=0.01,
                    native_ess_fraction=0.5,
                    active_native_coverage_distance=0.2,
                )
            )
    frame = pd.DataFrame(rows)
    pairs = report.paired_results(frame)
    assert pairs.valid_pair.sum() == 1
    assert pairs.loc[pairs.valid_pair, "delta_population_tv"].item() == pytest.approx(
        -0.05
    )
    assert np.isnan(pairs.loc[~pairs.valid_pair, "delta_population_tv"]).all()
    empty = report.paired_results(frame[frame.graph == "scalar_logpf"])
    assert empty.empty and "kind" in empty.columns
    assert (
        empty.loc[empty.valid_pair.astype(bool)].groupby(["kind", "graph"]).size().empty
    )


def test_manifest_rejects_artifact_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "scientific_identity", lambda: {"version": "a"})
    e.save_npz(tmp_path / "graph.npz", a=np.ones(3))
    e.write_json(
        tmp_path / "manifest.json",
        dict(
            code={"version": "a"},
            inputs={},
            artifacts={"graph.npz": e.digest(tmp_path / "graph.npz")},
        ),
    )
    g.load_manifest(tmp_path)
    e.save_npz(tmp_path / "graph.npz", a=np.zeros(3))
    with pytest.raises(ValueError, match="artifact changed"):
        g.load_manifest(tmp_path)


def test_run_job_resumes_without_refitting(tmp_path, monkeypatch):
    folder = tmp_path / "cases/baseline/profile_logpf"
    e.save_npz(folder / "fit.npz", weights=np.ones((6, 4)) / 4)
    e.write_json(
        folder / "complete.json",
        dict(identity="same", sha256=e.digest(folder / "fit.npz")),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Resuming must not refit")

    monkeypatch.setattr(g, "fit_graph", forbidden)
    result = g.run_job(
        (str(tmp_path), dict(case="baseline", graph="profile_logpf"), "same")
    )
    assert result["status"] == "resumed"


def test_numerical_failure_is_recorded(tmp_path, monkeypatch):
    import json

    case = tmp_path / "cases/baseline"
    e.save_npz(case / "input.npz", log_pf=np.ones((3, 4)))
    e.save_npz(case / "profile_logpf/graph.npz", kernels=np.ones((6, 4, 4)))

    def fail(*args, **kwargs):
        raise FloatingPointError("nonfinite optimiser state")

    monkeypatch.setattr(g, "fit_graph", fail)
    outcome = g.run_job(
        (str(tmp_path), dict(case="baseline", graph="profile_logpf"), "frozen")
    )
    assert outcome["status"] == "numerical_failure"
    saved = json.loads((case / "profile_logpf/failure.json").read_text())
    assert saved["identity"] == "frozen"
    assert not (case / "profile_logpf/complete.json").exists()
