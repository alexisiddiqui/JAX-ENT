"""Algebraic and mapping regressions for analysis without refitting."""

import numpy as np
import pytest
from jaxent.examples.ATLAS_BV.analysis import omc_bandwidth_diagnostics as d


def test_ess_decomposition_and_population_ceiling():
    labels = np.array([0, 0, 1, 1, 1])
    w1 = np.array([0.3, 0.2, 0.1, 0.15, 0.25])
    w2 = np.array([0.15, 0.15, 0.3, 0.3, 0.1])
    result = d.inverse_ess_change(w1, w2, labels)
    assert result["population_contribution"] + result[
        "within_contribution"
    ] == pytest.approx(w2 @ w2 - w1 @ w1)
    truth = np.array([0.8, 0.2])
    balanced, ceiling = d.balanced_weights(labels, truth)
    assert 1 / (balanced @ balanced) == pytest.approx(ceiling)
    # Redistributing within a basin can only lower ESS at fixed basin mass.
    perturbed = balanced + np.array([0.02, -0.02, 0, 0, 0])
    assert 1 / (perturbed @ perturbed) < ceiling
    with pytest.raises(ValueError):
        d.balanced_weights(np.array([0, 0]), truth)


def test_residual_terms_include_coverage_and_cross_cancellation():
    rng = np.random.default_rng(4)
    x = rng.random((9, 12))
    labels = np.repeat([0, 1], 6)
    keep = np.array([0, 2, 6, 7, 8, 10])
    w = rng.dirichlet(np.ones(len(keep)))
    residual, terms = d.residual_decomposition(x, labels, keep, w)
    np.testing.assert_allclose(residual, x[:, keep] @ w - x.mean(axis=1))
    np.testing.assert_allclose(sum(terms.values()), residual**2)
    assert np.mean(terms["coverage_sq"]) > 0
    balanced, _ = d.balanced_weights(labels[keep], np.array([0.5, 0.5]))
    _, bt = d.residual_decomposition(x, labels, keep, balanced)
    np.testing.assert_allclose(bt["population_sq"], 0, atol=1e-28)
    np.testing.assert_allclose(bt["reweight_sq"], 0, atol=1e-28)


def test_graph_energy_and_kernel_diagonal_invariance():
    labels = np.array([0, 0, 1, 1])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    distance = abs(np.arange(4)[:, None] - np.arange(4)[None, :])
    kernel = np.exp(-(distance**2) / 2)
    r = d.graph_stats(kernel, w, labels)
    assert r["graph_energy"] == pytest.approx(r["within_energy"] + r["between_energy"])
    np.fill_diagonal(kernel, 0)
    assert d.graph_stats(kernel, w, labels)["graph_energy"] == pytest.approx(
        r["graph_energy"]
    )
    assert d.graph_stats(kernel, np.ones(4) / 4, labels)[
        "graph_energy"
    ] == pytest.approx(0)


def test_svd_includes_exact_nullspace_but_not_uniform_contrast():
    labels = np.array([0, 0, 1, 1])
    # Observations respond to within-basin movement only: population is invisible.
    x = np.array([[1.0, -1, 1, -1]])
    for row in d.observability(x, labels):
        assert row["weak_population_fraction"] == pytest.approx(1)
    # A basin indicator observes the population contrast strongly.
    for row in d.observability(np.array([[0.0, 0, 1, 1]]), labels):
        assert row["weak_population_fraction"] == pytest.approx(0, abs=1e-28)


def test_alignment_removes_rigid_motion():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(8, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    result = d.aligned(np.stack([x, x @ q + 10]))
    np.testing.assert_allclose(result[0], result[1], atol=1e-13)
