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


def test_log_gamma_transform_and_kinetic_means():
    geometry = tiny_geometry()
    parameters = study.SimulationParameters(log_gamma=-2.3)
    decoded = study._decode(study._encode(parameters, ("log_gamma",)), parameters,
                            ("log_gamma",))
    assert decoded.log_gamma == parameters.log_gamma
    np.testing.assert_array_equal(study.mean_for_kinetics(geometry, None), geometry.mean)
    fast = np.log(np.median(geometry.k_int) * 1e6)
    np.testing.assert_allclose(study.mean_for_kinetics(geometry, fast), geometry.mean,
                               atol=2e-6)


def test_slow_kinetic_fit_recovers_gamma_and_beats_ex2():
    geometry = tiny_geometry()
    true_log_gamma = float(np.log(np.median(geometry.k_int) * 0.3))
    surface = study.mean_for_kinetics(geometry, true_log_gamma)
    initial = study.SimulationParameters(sigma_exp=1e-4, tau_z=0, tau_peptide=0,
                                         tau_time=0, kappa=0, log_gamma=np.log(2.0))
    fitted, k2_objective, _ = study.fit_kinetic_parameters(
        surface, geometry, initial, free=("log_gamma",), maxiter=150
    )
    k0, k0_objective, _ = study.fit_kinetic_parameters(
        surface, geometry, study.SimulationParameters(sigma_exp=1e-4, tau_z=0,
        tau_peptide=0, tau_time=0, kappa=0), free=(), maxiter=5
    )
    assert k0.log_gamma is None
    assert abs(fitted.log_gamma - true_log_gamma) < 0.05
    assert k2_objective < k0_objective


def test_reverse_null_drives_gamma_to_fast_limit():
    geometry = tiny_geometry()
    initial = study.SimulationParameters(sigma_exp=1e-4, tau_z=0, tau_peptide=0,
                                         tau_time=0, kappa=0, log_gamma=np.log(10.0))
    fitted, _, _ = study.fit_kinetic_parameters(
        geometry.mean, geometry, initial, free=("log_gamma",), maxiter=200
    )
    assert fitted.log_gamma >= np.log(np.median(geometry.k_int) * 1e4)


def test_ll_ex2_difference_scales_as_inverse_gamma():
    geometry = tiny_geometry()
    median_rate = float(np.median(geometry.k_int))
    differences = np.asarray([
        np.max(np.abs(
            study.mean_for_kinetics(geometry, np.log(c * median_rate)) - geometry.mean
        ))
        for c in study.K1_SCALING_C
    ])
    assert np.all(np.diff(differences) < 0)
    np.testing.assert_allclose(
        differences[:-1] / differences[1:], 10.0,
        rtol=study.K1_SCALING_RTOL, atol=0.0,
    )


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
    gamma = study._recordkeeping("gamma")
    assert gamma["uptake_backend"] == "ll"
    assert gamma["ll_gamma_parameterisation"]
    assert gamma["ll_log_gamma"] == list(study.GAMMA_LADDER)


def test_generated_full_manifests_share_recordkeeping_keys():
    root = FITTING_DIR / "_moprp_sigma_identifiability"
    paths = sorted(root.glob("*_full/manifest.json"))
    assert len(paths) == 5  # anm, delta, gamma, scale, sign
    manifests = [json.loads(path.read_text()) for path in paths]
    assert all(set(manifest) == set(manifests[0]) for manifest in manifests)
    for path, manifest in zip(paths, manifests):
        question = path.parent.name.removesuffix("_full")
        assert manifest["git_commit"]
        assert manifest["numerical_floor"] == 1e-10
        assert manifest["uptake_normalisation"] == study.UPTAKE_NORMALISATION
        # Phase 2.5 is the only question that activates the finite-gating backend.
        if question == "gamma":
            assert manifest["uptake_backend"] == "ll"
            assert manifest["seeds"] == {**study.SEED_REGISTRY, "gamma_surface": study.GAMMA_SEED}
            # the absolute floor is rate-source dependent; the ladder itself is dimensionless
            assert manifest["median_k_int_per_min"] > 0
            assert manifest["k1_fast_limit_c"] == study.K1_FAST_LIMIT_C
            assert manifest["k1_fast_limit_max_abs"] <= study.K1_TOLERANCE
        else:
            assert manifest["uptake_backend"] == "ex2"
            assert manifest["seeds"] == study.SEED_REGISTRY
            assert manifest["median_k_int_per_min"] is None


def test_phase3_delta_gate_is_fixed_at_point_one():
    decisions = (FITTING_DIR / "_moprp_sigma_identifiability/phase2_decisions.md").read_text()
    assert "at most 0.10" in decisions
    assert "operative Phase\n3 rule" in decisions
    assert "fixed provisionally" not in decisions
