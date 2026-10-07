"""Phase-1 correctness tests for the MoPrP joint HDX noise model."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(FITTING_DIR))
noise = importlib.import_module("moprp_sigma_noise_model")
sys.path.remove(str(FITTING_DIR))


def test_time_major_roundtrip_and_block_extraction():
    p, t = 3, 4
    surface = jnp.arange(p * t).reshape(p, t)
    vector = noise.vectorize_time_major(surface)
    np.testing.assert_array_equal(vector, surface.T.reshape(-1))
    np.testing.assert_array_equal(noise.unvectorize_time_major(vector, p, t), surface)
    blocks = np.stack([np.full((p, p), j + 1.0) for j in range(t)])
    covariance = np.zeros((p * t, p * t))
    for j in range(t):
        covariance[j*p:(j+1)*p, j*p:(j+1)*p] = blocks[j]
    np.testing.assert_array_equal(noise.extract_time_blocks(covariance, p, t), blocks)


def test_strict_ex2_mean_and_irregular_interval_recursion_match_native():
    from jaxent.src.analysis.hdx_ex2 import PeptideExchangeMap, predict_ex2_uptake
    mapping = np.array([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    peptide_map = PeptideExchangeMap(mapping, np.arange(1, 4), np.arange(1, 3),
                                     np.array([1, 2]), np.array([2, 3]), np.array([2, 2]),
                                     1, (), "test")
    z = np.array([0.2, 2.0, 5.0])
    rates = np.array([0.02, 1.0, 20.0])
    times = np.array([0.0834, 0.3336, 1.0002, 19.9998, 1440.0])
    residue = noise.EX2Backend().residue_uptake(z, rates, times)
    predicted = noise.peptide_uptake(residue, mapping)
    native = predict_ex2_uptake(z, rates, times, peptide_map)
    np.testing.assert_allclose(predicted, native, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(noise.interval_hazard_uptake(z, rates, times), residue,
                               rtol=2e-14, atol=2e-14)


def test_ex2_sensitivity_finite_difference_across_kinetic_windows():
    times = jnp.array([1.0])
    rates = jnp.ones(3)
    z = jnp.array([np.log(1000.0), 0.0, -np.log(1000.0)])
    analytic = noise.EX2Backend().logpf_sensitivity(z, rates, times)[:, 0]
    eps = 1e-5
    fd = []
    for r in range(3):
        direction = np.zeros(3)
        direction[r] = eps
        plus = noise.EX2Backend().residue_uptake(z + direction, rates, times)[r, 0]
        minus = noise.EX2Backend().residue_uptake(z - direction, rates, times)[r, 0]
        fd.append((plus - minus) / (2 * eps))
    np.testing.assert_allclose(analytic, fd, rtol=2e-7, atol=1e-12)


def test_ll_fast_limit_conservation_monotonicity_and_ad():
    ll = noise.LLBackend()
    ex2 = noise.EX2Backend()
    z = jnp.array([0.1, 1.0, 5.0])
    rates = jnp.array([0.02, 1.0, 20.0])
    times = jnp.array([0.0834, 0.3336, 1.0002, 19.9998, 1440.0])
    errors = []
    for gamma in (1e4, 1e6, 1e8):
        uptake = ll.residue_uptake(z, rates, times, jnp.log(gamma))
        p_c, p_o, exchanged = noise.ll_state_probabilities(z, rates, times, jnp.log(gamma))
        np.testing.assert_allclose(p_c + p_o + exchanged, 1.0, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(exchanged, uptake, rtol=0.0, atol=2e-15)
        errors.append(float(jnp.max(jnp.abs(uptake - ex2.residue_uptake(z, rates, times)))))
        assert np.all(np.diff(np.asarray(uptake), axis=1) >= -1e-12)
        assert np.all((np.asarray(uptake) >= 0) & (np.asarray(uptake) <= 1))
    assert errors[2] < errors[1] < errors[0]
    analytic = ll.logpf_sensitivity(z, rates, times, jnp.log(2.0))
    eps = 1e-5
    for r in range(3):
        direction = jnp.zeros_like(z).at[r].set(eps)
        fd = (ll.residue_uptake(z + direction, rates, times, jnp.log(2.0))[r]
              - ll.residue_uptake(z - direction, rates, times, jnp.log(2.0))[r]) / (2 * eps)
        np.testing.assert_allclose(analytic[r], fd, rtol=2e-5, atol=2e-8)
    compiled = jax.jit(lambda value: ll.residue_uptake(value, rates, times, jnp.log(2.0)))
    assert np.all(np.isfinite(np.asarray(compiled(z))))


def test_ll_near_degenerate_branch_has_finite_value_and_gradient():
    ll = noise.LLBackend()
    # At PF=1, k_int=gamma gives a repeated eigenvalue (-gamma).
    def fn(z):
        return jnp.sum(ll.residue_uptake(jnp.array([z]), jnp.array([1.0]),
                                        jnp.array([0.1, 1.0, 10.0]), jnp.array(0.0)))
    assert np.isfinite(float(fn(0.0)))
    assert np.isfinite(float(jax.grad(fn)(0.0)))


def test_covariance_components_schur_and_domain_flip_invariants():
    rng = np.random.default_rng(4)
    q = rng.normal(size=(6, 3))
    cov = q @ q.T + np.eye(6)
    sd = np.sqrt(np.diag(cov))
    corr = cov / sd[:, None] / sd[None, :]
    unsigned = np.asarray(noise.schur_square_correlation(corr))
    np.testing.assert_allclose(np.diag(unsigned), 1.0)
    assert np.linalg.eigvalsh(unsigned).min() > -1e-12
    signs = np.array([1, -1, 1, -1, -1, 1])
    flipped = np.asarray(noise.domain_flip_correlation(corr, signs))
    np.testing.assert_allclose(np.linalg.eigvalsh(flipped), np.linalg.eigvalsh(corr), atol=1e-12)
    a = rng.normal(size=(8, 6))
    joint = noise.build_joint_covariance(a, cov, numerical_floor=1e-9)
    np.testing.assert_allclose(joint, joint.T, atol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(joint)).min() >= 0


def test_cholesky_nll_dense_reference_homoscedastic_mse_and_mixture_stability():
    residual = jnp.array([0.2, -0.5, 1.3])
    sigma = 0.7
    covariance = sigma**2 * jnp.eye(3)
    chol = jnp.linalg.cholesky(covariance)
    got = noise.gaussian_nll_from_cholesky(residual, chol)
    sign, logdet = np.linalg.slogdet(np.asarray(covariance))
    dense = 0.5 * np.asarray(residual) @ np.linalg.solve(np.asarray(covariance), np.asarray(residual))
    dense += 0.5 * logdet + 1.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(got, dense, rtol=1e-13)
    # With only homoscedastic acquisition variance, the data-dependent term is scaled MSE.
    np.testing.assert_allclose(got - 3*np.log(sigma) - 1.5*np.log(2*np.pi),
                               0.5*np.sum(np.asarray(residual)**2)/sigma**2)
    residuals = jnp.stack([residual, residual * 2])
    chols = jnp.stack([chol, chol])
    mixture = noise.mixture_nll(residuals, chols, jnp.array([10000.0, -10000.0]))
    assert np.isfinite(float(mixture))
    np.testing.assert_allclose(mixture, got, rtol=1e-13)


def test_mask_guard_rejects_sentinel_and_excluded_mapping_columns():
    mapping = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    active = np.any(mapping != 0, axis=0)
    log_pf = np.array([1.0, -1.0, -1.0])
    assert not np.all(np.isfinite(log_pf[active])) or np.any(log_pf[active] < 0)
    with pytest.raises(ValueError, match="PF >= 1"):
        noise.LLBackend().residue_uptake(log_pf, np.ones(3), np.ones(1), 0.0)
    assert np.all(mapping[:, ~active] == 0)
