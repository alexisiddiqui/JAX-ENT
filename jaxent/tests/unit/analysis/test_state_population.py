"""Unit tests for the known-population state interfaces.

These tests lock the strict-support conventions: zero-target decoy states must contribute
to the Jensen-Shannon recovery, reference weights must place exact per-state mass uniformly
within each state, covariance matching is symmetric and zero at truth, and states that are
indistinguishable in the covariance coordinate must remain non-identifiable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from jaxent.src.analysis import state_population as sp

REPO_ROOT = Path(__file__).resolve().parents[4]
STATE_RATIOS_JSON = (
    REPO_ROOT / "jaxent/examples/2_CrossValidation/analysis/state_ratios.json"
)


# --------------------------------------------------------------------------------------
# Targets and reference weights
# --------------------------------------------------------------------------------------
def test_load_state_targets_full_support():
    support, targets = sp.load_state_targets(STATE_RATIOS_JSON)
    assert support == sp.FULL_STATE_SUPPORT
    mapping = dict(zip(support, targets))
    assert mapping["Folded"] > mapping["PUF1"] > mapping["PUF2"] > 0.0
    for decoy in ("PUF3", "unfolded", "PUF2-like"):
        assert mapping[decoy] == 0.0
    assert targets.sum() == pytest.approx(1.0)


def test_reference_weights_exact_mass_uniform_within_state():
    support = ("Folded", "PUF1", "PUF2", "PUF3")
    targets = np.array([0.9, 0.08, 0.02, 0.0])
    states = np.array(
        ["Folded"] * 5 + ["PUF1"] * 3 + ["PUF2"] * 2 + ["PUF3"] * 4, dtype=object
    )
    weights = sp.reference_weights_from_states(states, support, targets)

    assert weights.sum() == pytest.approx(1.0)
    # Uniform within state and exact per-state mass.
    np.testing.assert_allclose(weights[states == "Folded"], 0.9 / 5)
    np.testing.assert_allclose(weights[states == "PUF1"], 0.08 / 3)
    np.testing.assert_allclose(weights[states == "PUF2"], 0.02 / 2)
    # Decoy frames remain available but carry zero mass.
    np.testing.assert_allclose(weights[states == "PUF3"], 0.0)

    populations = np.asarray(sp.state_populations(weights, states, support))
    np.testing.assert_allclose(populations, targets, atol=1e-12)


def test_reference_weights_raise_when_target_state_missing():
    support = ("Folded", "PUF2")
    targets = np.array([0.5, 0.5])
    states = np.array(["Folded"] * 4, dtype=object)  # no PUF2 frames
    with pytest.raises(ValueError):
        sp.reference_weights_from_states(states, support, targets)


# --------------------------------------------------------------------------------------
# Strict full-support JSD recovery
# --------------------------------------------------------------------------------------
def test_recovery_is_perfect_at_reference_weights():
    support = ("Folded", "PUF1", "PUF2", "PUF3")
    targets = np.array([0.9, 0.08, 0.02, 0.0])
    states = np.array(
        ["Folded"] * 5 + ["PUF1"] * 3 + ["PUF2"] * 2 + ["PUF3"] * 4, dtype=object
    )
    weights = sp.reference_weights_from_states(states, support, targets)
    jsd = float(sp.strict_population_jsd(weights, states, support, targets))
    recovery = float(sp.strict_recovery_percent(weights, states, support, targets))
    assert jsd == pytest.approx(0.0, abs=1e-7)
    assert recovery == pytest.approx(100.0, abs=1e-4)


def test_zero_target_decoy_mass_hurts_recovery():
    support = ("Folded", "PUF1", "PUF2", "PUF3")
    targets = np.array([0.9, 0.08, 0.02, 0.0])
    states = np.array(
        ["Folded"] * 5 + ["PUF1"] * 3 + ["PUF2"] * 2 + ["PUF3"] * 4, dtype=object
    )
    good = sp.reference_weights_from_states(states, support, targets)

    # Move 20% of the mass into the zero-target PUF3 decoy state.
    decoy = np.zeros_like(good)
    decoy[states == "PUF3"] = 0.2 / int(np.sum(states == "PUF3"))
    perturbed = 0.8 * good + decoy
    perturbed = perturbed / perturbed.sum()

    jsd_good = float(sp.strict_population_jsd(good, states, support, targets))
    jsd_bad = float(sp.strict_population_jsd(perturbed, states, support, targets))
    # The decoy mass cannot be renormalized away: it must raise the divergence.
    assert jsd_bad > jsd_good + 1e-3
    # And it must appear in the recovered populations.
    populations = np.asarray(sp.state_populations(perturbed, states, support))
    assert populations[support.index("PUF3")] == pytest.approx(0.2, abs=1e-6)


def test_population_jsd_is_differentiable_in_weights():
    support = ("Folded", "PUF1", "PUF2")
    targets = np.array([0.7, 0.2, 0.1])
    states = np.array(["Folded"] * 3 + ["PUF1"] * 3 + ["PUF2"] * 3, dtype=object)

    def loss(logits):
        weights = jax.nn.softmax(logits)
        return sp.strict_population_jsd(weights, states, support, targets)

    grad = jax.grad(loss)(jnp.zeros(states.size))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad)) > 0.0


# --------------------------------------------------------------------------------------
# Covariance coordinates: symmetry, zero at truth, non-identifiability
# --------------------------------------------------------------------------------------
def _toy_features(seed=0, n_residues=6, n_frames=40):
    rng = np.random.default_rng(seed)
    log_pf = jnp.asarray(rng.normal(size=(n_residues, n_frames)))
    mapping = jnp.asarray(
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
        / 2.0
    )
    return log_pf, mapping


def test_covariance_matching_is_symmetric_and_zero_at_truth():
    log_pf, mapping = _toy_features()
    weights = jnp.full(log_pf.shape[1], 1.0 / log_pf.shape[1])
    cov = sp.peptide_logpf_covariance(log_pf, mapping, weights)

    profile = sp.marginal_profile(cov)
    assert float(sp.log_ratio_profile_loss(profile, profile)) == pytest.approx(0.0, abs=1e-10)
    assert float(
        sp.projected_full_covariance_loss(cov, cov, mapping)
    ) == pytest.approx(0.0, abs=1e-8)

    # Swapping predicted/target leaves the log-ratio loss unchanged (symmetry).
    other = sp.marginal_profile(
        sp.peptide_logpf_covariance(log_pf, mapping, jax.nn.softmax(jnp.asarray(log_pf[0])))
    )
    forward = float(sp.log_ratio_profile_loss(profile, other))
    backward = float(sp.log_ratio_profile_loss(other, profile))
    assert forward == pytest.approx(backward, rel=1e-6)


def test_shrunk_precision_is_finite_and_trace_normalized():
    log_pf, mapping = _toy_features()
    weights = jnp.full(log_pf.shape[1], 1.0 / log_pf.shape[1])
    cov = sp.peptide_logpf_covariance(log_pf, mapping, weights)
    precision = sp.shrunk_trace_normalized_precision(cov, alpha=0.05)
    precision_np = np.asarray(precision)
    assert np.all(np.isfinite(precision_np))
    np.testing.assert_allclose(precision_np, precision_np.T, atol=1e-6)
    assert np.trace(precision_np) == pytest.approx(precision_np.shape[0], rel=1e-5)


def test_rank_deficient_covariance_stays_invertible_after_shrinkage():
    # A single distinct frame gives a rank-0 covariance; shrinkage must keep it finite.
    log_pf = jnp.tile(jnp.asarray([[1.0], [2.0], [3.0]]), (1, 8))
    mapping = jnp.eye(3)
    weights = jnp.full(8, 1.0 / 8)
    cov = sp.peptide_logpf_covariance(log_pf, mapping, weights)
    precision = sp.shrunk_trace_normalized_precision(cov)
    assert np.all(np.isfinite(np.asarray(precision)))


def test_indistinguishable_frames_are_non_identifiable_in_covariance():
    # Two frames with identical log-PF: covariance depends only on their summed weight, so
    # transferring mass between them leaves the covariance (and any matching loss) unchanged.
    base = np.random.default_rng(1).normal(size=(4, 6))
    log_pf = np.concatenate([base, base[:, :1]], axis=1)  # frame 6 duplicates frame 0
    log_pf = jnp.asarray(log_pf)
    mapping = jnp.eye(4)

    w_a = np.full(7, 1.0 / 7)
    w_b = w_a.copy()
    transfer = 0.05
    w_b[0] += transfer  # move mass between the two identical frames (0 and 6)
    w_b[6] -= transfer
    cov_a = np.asarray(sp.peptide_logpf_covariance(log_pf, mapping, jnp.asarray(w_a)))
    cov_b = np.asarray(sp.peptide_logpf_covariance(log_pf, mapping, jnp.asarray(w_b)))
    # Identical up to float32 rounding: the covariance carries no information distinguishing
    # the two equal-valued frames, so the mass transfer is non-identifiable.
    np.testing.assert_allclose(cov_a, cov_b, atol=1e-6)


def test_correlation_of_unit_diagonal_and_scale_free():
    log_pf, mapping = _toy_features()
    weights = jnp.full(log_pf.shape[1], 1.0 / log_pf.shape[1])
    cov = sp.peptide_logpf_covariance(log_pf, mapping, weights)
    corr = np.asarray(sp.correlation_of(cov))
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)
    # scaling the covariance does not change its correlation
    np.testing.assert_allclose(np.asarray(sp.correlation_of(7.0 * cov)), corr, atol=1e-6)


def test_correlation_shape_loss_zero_at_matching_shape_and_scale_free():
    from jaxent.src.analysis.pf_variance import overlap_projection

    log_pf, mapping = _toy_features()
    weights = jnp.full(log_pf.shape[1], 1.0 / log_pf.shape[1])
    cov = sp.peptide_logpf_covariance(log_pf, mapping, weights)
    projection = overlap_projection(mapping)
    prior = sp.correlation_of(cov)
    # zero when the covariance's shape equals the prior correlation
    assert float(sp.correlation_shape_loss(cov, prior, projection)) == pytest.approx(0.0, abs=1e-8)
    # invariant to the magnitude of the covariance (shape-only)
    assert float(sp.correlation_shape_loss(100.0 * cov, prior, projection)) == pytest.approx(0.0, abs=1e-8)
    # differentiable in the frame weights
    def loss(logits):
        w = jax.nn.softmax(logits)
        return sp.correlation_shape_loss(sp.peptide_logpf_covariance(log_pf, mapping, w), prior, projection)
    grad = jax.grad(loss)(jnp.zeros(log_pf.shape[1]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_uptake_covariance_shape_and_symmetry():
    log_pf, mapping = _toy_features()
    k_ints = jnp.asarray(np.abs(np.random.default_rng(2).normal(size=log_pf.shape[0])) + 0.1)
    timepoints = jnp.asarray([0.5, 5.0, 60.0])
    weights = jnp.full(log_pf.shape[1], 1.0 / log_pf.shape[1])
    covs = sp.peptide_uptake_covariances(log_pf, k_ints, timepoints, mapping, weights)
    assert covs.shape == (3, mapping.shape[0], mapping.shape[0])
    for t in range(covs.shape[0]):
        block = np.asarray(covs[t])
        np.testing.assert_allclose(block, block.T, atol=1e-8)


# --------------------------------------------------------------------------------------
# Recovery-helper fix: full support vs legacy differ exactly by decoy mass
# --------------------------------------------------------------------------------------
def _load_recovery_module():
    path = (
        REPO_ROOT
        / "jaxent/examples/2_CrossValidation/analysis/compute_recovery%_PUF.py"
    )
    spec = importlib.util.spec_from_file_location("compute_recovery_puf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_recovery_discards_decoy_full_support_retains_it():
    module = _load_recovery_module()
    state_mapping = {0: "Folded", 1: "PUF1", 2: "PUF2", 3: "PUF3", 4: "unfolded"}
    target_ratios = {"Folded": 0.9, "PUF1": 0.08, "PUF2": 0.02}
    # 10 frames each in Folded/PUF1/PUF2/PUF3/unfolded -> 40% decoy mass at uniform weights.
    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
    assignments = pd.Series(labels)
    weights = np.full(len(labels), 1.0 / len(labels))

    legacy = module.calculate_recovery_percentage(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=True
    )
    full = module.calculate_recovery_percentage(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=False
    )

    # Legacy drops the 40% decoy mass entirely from the current proportions.
    assert set(legacy["current_proportions"]) == set(target_ratios)
    # Full support surfaces the decoy states with their mass.
    assert full["current_proportions"]["PUF3"] == pytest.approx(0.2)
    assert full["current_proportions"]["unfolded"] == pytest.approx(0.2)
    target_mass_full = sum(full["current_proportions"][s] for s in target_ratios)
    assert target_mass_full == pytest.approx(0.6)

    jsd_legacy = module.calculate_recovery_JSD(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=True
    )
    jsd_full = module.calculate_recovery_JSD(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=False
    )
    # Retaining decoy mass strictly increases the divergence.
    assert jsd_full > jsd_legacy


def test_full_support_equals_legacy_without_decoy_frames():
    module = _load_recovery_module()
    state_mapping = {0: "Folded", 1: "PUF1", 2: "PUF2"}
    target_ratios = {"Folded": 0.9, "PUF1": 0.08, "PUF2": 0.02}
    labels = [0] * 10 + [1] * 10 + [2] * 10
    assignments = pd.Series(labels)
    weights = np.full(len(labels), 1.0 / len(labels))
    jsd_legacy = module.calculate_recovery_JSD(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=True
    )
    jsd_full = module.calculate_recovery_JSD(
        assignments, weights, target_ratios, state_mapping, legacy_target_support=False
    )
    assert jsd_full == pytest.approx(jsd_legacy)
