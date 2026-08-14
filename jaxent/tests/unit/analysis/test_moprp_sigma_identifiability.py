"""Reduced-geometry tests for the Phase-2 identifiability runner."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(FITTING_DIR))
study = importlib.import_module("moprp_sigma_identifiability")
noise = importlib.import_module("moprp_sigma_noise_model")
sys.path.remove(str(FITTING_DIR))


def tiny_geometry():
    mapping = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    log_pf = np.array([0.2, 1.0, 2.0])
    rates = np.array([0.2, 1.0, 3.0])
    times = np.array([0.2, 1.0, 5.0])
    backend = noise.EX2Backend()
    uptake = np.asarray(backend.residue_uptake(log_pf, rates, times))
    sensitivity = np.asarray(backend.logpf_sensitivity(log_pf, rates, times))
    mean = np.asarray(noise.peptide_uptake(uptake, mapping))
    propagation = np.asarray(noise.stack_propagation_matrix(sensitivity, mapping))
    p, t = mean.shape
    zp = np.zeros((p*t, p))
    zt = np.zeros((p*t, t))
    slope = -np.asarray(noise.peptide_uptake(sensitivity, mapping))
    flat = np.asarray(noise.vectorize_time_major(mean))
    for j in range(t):
        for peptide in range(p):
            index = j*p + peptide
            zp[index, peptide] = flat[index]
            zt[index, j] = slope[peptide, j]
    corr = np.array([[1, .3, -.1], [.3, 1, .2], [-.1, .2, 1]])
    return study.Geometry(mean, mapping, log_pf, rates, times, sensitivity, propagation, np.ones(3), corr,
                          zp, zt, p, t)


def test_simulator_is_deterministic_and_covariance_matches():
    geometry = tiny_geometry()
    parameters = study.SimulationParameters()
    first, covariance = study.simulate_surface(geometry, parameters, 12)
    second, covariance2 = study.simulate_surface(geometry, parameters, 12)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(covariance, covariance2)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_conditional_nll_matches_independent_marginal_case():
    residual = np.array([0.2, -0.3, 0.4])
    covariance = np.diag([0.5, 0.7, 0.9])
    got = study.conditional_gaussian_nll(residual, covariance, np.array([0, 1]), np.array([2]))
    expected = float(noise.gaussian_nll_from_cholesky(np.array([0.4]), np.array([[np.sqrt(.9)]])))
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_parameter_transform_roundtrip_and_reduced_fit_is_finite():
    geometry = tiny_geometry()
    truth = study.SimulationParameters(sigma_exp=.04, tau_z=.2, kappa=.7)
    free = ("sigma_exp", "tau_z", "kappa", "anm_lambda")
    decoded = study._decode(study._encode(truth, free), truth, free)
    for name in free:
        expected = 1e-6 if name == "anm_lambda" else getattr(truth, name)
        np.testing.assert_allclose(getattr(decoded, name), expected)
    surface, _ = study.simulate_surface(geometry, truth, 5)
    fitted, objective, _ = study.fit_parameters(surface, geometry, truth,
                                                free=("sigma_exp",), maxiter=10)
    assert np.isfinite(objective)
    assert fitted.sigma_exp > 0


def test_time_folds_cover_every_coordinate_once():
    folds = list(study.time_folds(3, 5, block=2))
    held = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(held), np.arange(15))


def test_recordkeeping_schema_and_verbatim_normalisation(monkeypatch):
    monkeypatch.setattr(study, "_git_commit", lambda: "test-commit")
    records = [study._recordkeeping(question) for question in ("anm", "sign", "scale", "delta")]
    assert all(set(record) == set(records[0]) for record in records)
    for record in records:
        assert record["git_commit"] == "test-commit"
        assert record["numerical_floor"] == study.NUMERICAL_FLOOR == 1e-10
        assert record["uptake_backend"] == "ex2"
        assert record["pf_fit"] == {"performed": False, "start_count": None, "seed": None}
        assert record["seeds"] == study.SEED_REGISTRY
        assert record["uptake_normalisation"] == (
            "MoPrP uptake is peptide-wise maxD normalised using a completely deuterated control subjected to\n"
            "the same quench, digestion, LC and MS processing. This normalisation implicitly compensates\n"
            "peptide-dependent mean back-exchange in centroid uptake but does not reconstruct absolute\n"
            "pre-quench deuterium occupancy or explicitly model residue-specific / time-dependent\n"
            "back-exchange. Labelling experiments were conducted at approximately 95% D₂O."
        )
    assert records[0]["anm_variant"]["used_in_question"]
    assert records[1]["anm_variant"]["used_in_question"]
    assert not records[2]["anm_variant"]["used_in_question"]
    assert not records[3]["anm_variant"]["used_in_question"]


def test_generated_full_manifests_share_recordkeeping_keys():
    root = FITTING_DIR / "_moprp_sigma_identifiability"
    paths = sorted(root.glob("*_full/manifest.json"))
    assert len(paths) == 4
    manifests = [json.loads(path.read_text()) for path in paths]
    assert all(set(manifest) == set(manifests[0]) for manifest in manifests)
    for manifest in manifests:
        assert manifest["git_commit"]
        assert manifest["numerical_floor"] == 1e-10
        assert manifest["uptake_backend"] == "ex2"
        assert manifest["uptake_normalisation"] == study.UPTAKE_NORMALISATION
        assert manifest["seeds"] == study.SEED_REGISTRY


def test_phase3_delta_gate_is_fixed_at_point_one():
    decisions = (FITTING_DIR / "_moprp_sigma_identifiability/phase2_decisions.md").read_text()
    assert "at most 0.10" in decisions
    assert "operative Phase\n3 rule" in decisions
    assert "fixed provisionally" not in decisions
