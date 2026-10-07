"""Tests for generic joint weight/BV fitting against frozen geometry priors."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from jaxent.src.analysis.joint_covariance_geometry import (
    relative_log_variance,
    scale_free_log_spd,
)
from jaxent.src.analysis.joint_geometry_reweighting import (
    FrozenPeptidePrior,
    ReweightingCell,
    coefficients_from_theta,
    inverse_softplus,
    optimize_joint_reweighting,
    peptide_logpf_covariance,
    predict_uptake,
)
from jaxent.src.analysis.pf_variance import (
    covariance_profiles_from_covariance,
    inverse_overlap_degree_weights,
    overlap_projection,
)


def _cell(kind: str = "fixed") -> ReweightingCell:
    heavy = np.asarray(
        [[0.2, 0.4, 1.0, 1.2], [0.1, 0.5, 0.9, 1.4], [0.3, 0.6, 1.1, 1.5]],
        dtype=np.float32,
    )
    acceptor = np.asarray(
        [[0.1, 0.2, 0.5, 0.6], [0.2, 0.1, 0.4, 0.7], [0.0, 0.3, 0.6, 0.8]],
        dtype=np.float32,
    )
    mapping = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32)
    weights = jnp.asarray([0.4, 0.3, 0.2, 0.1])
    log_pf = 0.35 * heavy + 2.0 * acceptor
    times = np.asarray([0.2, 1.0, 10.0], dtype=np.float32)
    kints = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    observed = np.asarray(predict_uptake(log_pf, kints, times, mapping, weights)).T
    covariance = peptide_logpf_covariance(log_pf, mapping, weights)
    projection = overlap_projection(mapping)
    overlap_weights = inverse_overlap_degree_weights(mapping)
    prior = FrozenPeptidePrior(
        kind=kind,
        geometry=scale_free_log_spd(covariance, projection),
        relative_variance=relative_log_variance(covariance, overlap_weights),
    )
    uniform = jnp.full(4, 0.25)
    reference_prediction = np.asarray(predict_uptake(log_pf, kints, times, mapping, uniform)).T
    mean_reference = float(np.mean(np.square(reference_prediction - observed)))
    return ReweightingCell(
        name="toy",
        heavy_contacts=heavy,
        acceptor_contacts=acceptor,
        k_ints=kints,
        timepoints=times,
        mapping=mapping,
        observed_uptake=observed,
        train_time_indices=np.arange(3),
        projection=projection,
        marginal_weights=overlap_weights,
        mean_reference=mean_reference,
        prior=prior,
    )


def test_coefficient_transform_round_trip_and_positive():
    theta = jnp.asarray([inverse_softplus(0.35), inverse_softplus(2.0)])
    bc, bh = coefficients_from_theta(theta)
    np.testing.assert_allclose([bc, bh], [0.35, 2.0], rtol=1e-5)
    assert float(bc) > 0 and float(bh) > 0


def test_toy_joint_optimizer_is_finite_and_returns_normalized_weights():
    result = optimize_joint_reweighting(
        {"toy": _cell()}, steps=80, starts=2, learning_rate=0.04
    )
    assert np.isfinite(result["objective"])
    assert result["bc"] > 0 and result["bh"] > 0
    np.testing.assert_allclose(result["weights"]["toy"].sum(), 1.0, atol=1e-6)
    assert set(result["components"]) == {"mean", "geometry", "marginal", "kl", "score"}


def test_legacy_variance_oracle_comparators_remain_finite():
    base = _cell()
    log_pf = 0.35 * base.heavy_contacts + 2.0 * base.acceptor_contacts
    covariance = peptide_logpf_covariance(log_pf, base.mapping, jnp.full(4, 0.25))
    profiles = covariance_profiles_from_covariance(covariance)
    for kind, index in (("marginal_oracle", 1), ("conditional_oracle", 3)):
        prior = FrozenPeptidePrior(
            kind=kind,
            profile=profiles[index],
            profile_weights=base.marginal_weights,
        )
        result = optimize_joint_reweighting(
            {"toy": replace(base, prior=prior)}, steps=3, starts=1
        )
        assert np.isfinite(result["objective"])
