import numpy as np
from jaxent.examples.ATLAS_BV.analysis import omc_coupling_control as e
from jaxent.examples.ATLAS_BV.analysis import omc_bandwidth_diagnostics as d


def test_coupling_controls_reference_and_edge_pattern():
    x = np.linspace(0, 4, 30)
    work = abs(x[:, None] - x[None, :])
    keep = np.array([0, 2, 3, 7, 8, 13, 17, 20, 25])
    sigmas = [0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
    k, c = e.graph_controls(work, keep, sigmas)
    off = ~np.eye(len(keep), dtype=bool)
    np.testing.assert_allclose(k["fixed_coupling"][:, off].mean(axis=1), c[2])
    np.testing.assert_allclose(k["fixed_pattern"][:, off].mean(axis=1), c)
    for i in range(6):
        np.testing.assert_allclose(
            k["fixed_pattern"][i] / c[i], k["existing"][2] / c[2]
        )
    for v in e.VARIANTS:
        np.testing.assert_array_equal(k[v][2], k["existing"][2])
    # Padding frames cannot change the retained graph or its coupling.
    changed = work.copy()
    changed[1, :] = 10000
    changed[:, 1] = 10000
    kk, cc = e.graph_controls(changed, keep, sigmas)
    np.testing.assert_array_equal(c, cc)
    for v in k:
        np.testing.assert_array_equal(k[v], kk[v])
    assert len(e.fit_specs()) == 10
    assert all(s["q_index"] != 2 for s in e.fit_specs())


def test_energy_linear_scaling_and_diagonal_invariance():
    w = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    k = np.exp(-abs(np.arange(4)[:, None] - np.arange(4)[None, :]))
    energy = d.graph_stats(k, w, labels)["graph_energy"]
    for scale in (0.15, 1, 7):
        np.testing.assert_allclose(
            d.graph_stats(k * scale, w, labels)["graph_energy"], energy * scale
        )
    np.fill_diagonal(k, 123)
    np.testing.assert_allclose(d.graph_stats(k, w, labels)["graph_energy"], energy)


def test_resume_identity_corruption_and_new_fit_shape(tmp_path):
    source = {"indices": np.arange(8)}
    keep = np.array([0, 2, 4, 6])
    weights = np.zeros((10, 8))
    weights[:, keep] = 0.25
    fit = dict(
        weights=weights,
        **{
            key: np.ones(10)
            for key in (
                "objective",
                "relative_change",
                "absolute_change",
                "steps",
                "grad_norm",
            )
        },
    )
    e.old.atomic_npz(tmp_path / "fit.npz", **fit)
    e.old.atomic_json(
        tmp_path / "complete.json",
        dict(identity="a", sha256=e.old.digest(tmp_path / "fit.npz")),
    )
    assert e.load_complete(tmp_path, "a", source, keep) is not None
    assert e.load_complete(tmp_path, "b", source, keep) is None
    assert e.load_complete(tmp_path, "a", source, np.array([0, 2, 4])) is None
    (tmp_path / "fit.npz").write_bytes(b"corrupt")
    assert e.load_complete(tmp_path, "a", source, keep) is None


def test_factorial_contrasts_include_interaction():
    from jaxent.examples.ATLAS_BV.analysis.omc_coupling_report import (
        factorial_contrasts,
    )

    # An interacting response: individual effects alone do not explain the total.
    result = factorial_contrasts(existing=0.8, pattern=0.4, coupling=0.5, reference=0.2)
    np.testing.assert_allclose(result["coupling_effect"], 0.3)
    np.testing.assert_allclose(result["edge_pattern_effect"], 0.2)
    np.testing.assert_allclose(result["interaction"], 0.1)
    np.testing.assert_allclose(result["total_change"], 0.6)
    for value in factorial_contrasts(0.2, 0.2, 0.2, 0.2).values():
        assert value == 0
