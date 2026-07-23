"""Safeguards for geometry-regularised HDX target-variance inference."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from jaxent.src.analysis.hdx_target_variance import (
    ARTIFACT_TYPE,
    build_hdx_covariance,
    build_rate_geometries,
    distance_sequence_support_kernel,
    effective_rates,
    fit_curve_moment_variance,
    fit_structured_residual_variance,
    load_frozen_settings,
    map_hdx_covariance,
    nearest_neighbor_sequence_kernel,
    positive_two_moment_uptake,
    predict_curve_moment_uptake,
    predict_fixed_mean_uptake,
    propagated_uptake_covariance,
    qualification_gate,
    wendland_distance_kernel,
    write_frozen_settings,
)


def _small_geometry_inputs():
    rates = np.asarray(
        [
            [0.10, 0.13, 0.18, 0.22, 0.16],
            [0.30, 0.26, 0.40, 0.34, 0.37],
            [0.50, 0.65, 0.57, 0.72, 0.61],
            [0.80, 0.73, 0.95, 0.86, 0.91],
        ]
    )
    coordinates = np.asarray([[0, 0, 0], [3, 0, 0], [20, 0, 0], [23, 0, 0]], float)
    residue_ids = np.asarray([10, 11, 20, 21])
    return rates, coordinates, residue_ids


def test_effective_rate_coordinates_are_intrinsic_rate_times_inverse_pf():
    log_pf = np.log(np.asarray([[2.0, 4.0], [5.0, 10.0]]))
    k_ints = np.asarray([0.8, 2.0])
    np.testing.assert_allclose(effective_rates(log_pf, k_ints), [[0.4, 0.2], [0.4, 0.2]])


def test_geometry_kernels_are_psd_and_distance_support_is_compact():
    rates, coordinates, residue_ids = _small_geometry_inputs()
    distance = wendland_distance_kernel(coordinates, cutoff_angstrom=8.0)
    sequence = nearest_neighbor_sequence_kernel(residue_ids)
    support = distance_sequence_support_kernel(coordinates, residue_ids, cutoff_angstrom=8.0)
    assert distance[0, 2] == 0.0
    assert support[0, 2] == 0.0
    assert sequence[0, 1] > 0 and sequence[0, 2] == 0
    for matrix in (distance, sequence, support, *build_rate_geometries(rates, coordinates, residue_ids).values()):
        np.testing.assert_allclose(np.diag(matrix), 1.0)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-10


def test_hdx_covariance_has_diagonal_d_is_psd_and_maps_by_congruence():
    rates, coordinates, residue_ids = _small_geometry_inputs()
    geometry = build_rate_geometries(rates, coordinates, residue_ids)[
        "covariance_distance_sequence"
    ]
    variances = np.asarray([0.01, 0.04, 0.09, 0.16])
    covariance = build_hdx_covariance(variances, geometry)
    mapping = np.asarray([[0.5, 0.5, 0, 0], [0, 0.25, 0.5, 0.25]])
    np.testing.assert_allclose(np.diag(covariance), variances)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    np.testing.assert_allclose(map_hdx_covariance(covariance, mapping), mapping @ covariance @ mapping.T)


def test_zero_variance_recovers_fixed_mean_limit():
    means = np.asarray([0.15, 0.7, 1.3])
    times = np.asarray([0.0, 0.2, 1.0, 5.0])
    mapping = np.eye(3)
    expected = 1.0 - np.exp(-times[:, None] * means[None, :])
    np.testing.assert_allclose(positive_two_moment_uptake(means, np.zeros(3), times), expected)
    np.testing.assert_allclose(predict_fixed_mean_uptake(means, times, mapping), expected.T)


def test_curve_moment_estimator_recovers_small_positive_rate_mixture_moments():
    means = np.asarray([0.15, 0.5, 1.2])
    truth = np.asarray([0.004, 0.05, 0.35])
    times = np.geomspace(0.02, 30.0, 18)
    mapping = np.eye(3)
    observed = predict_curve_moment_uptake(means, truth, times, mapping)
    fit = fit_curve_moment_variance(
        observed,
        means,
        times,
        mapping,
        np.eye(3),
        geometry_name="identity",
        initial_variance=0.2,
        maxiter=500,
    )
    assert fit.success
    np.testing.assert_allclose(fit.variances, truth, rtol=2e-3, atol=1e-6)
    np.testing.assert_allclose(fit.predicted_uptake, observed, atol=1e-7)


def test_structured_residual_uses_positive_definite_time_covariance_and_finite_nll():
    means = np.asarray([0.2, 0.6])
    times = np.asarray([0.2, 1.0, 5.0])
    mapping = np.asarray([[1.0, 0.0], [0.5, 0.5]])
    geometry = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    covariance = build_hdx_covariance(np.asarray([0.01, 0.04]), geometry)
    for time in times:
        sigma = propagated_uptake_covariance(means, covariance, time, mapping, 1e-4)
        assert np.linalg.eigvalsh(sigma).min() > 0
    observed = predict_fixed_mean_uptake(means, times, mapping) + np.asarray(
        [[0.01, -0.02, 0.01], [-0.005, 0.015, -0.01]]
    )
    fit = fit_structured_residual_variance(
        observed, means, times, mapping, geometry, noise_variance=1e-4, maxiter=200
    )
    assert np.isfinite(fit.objective)
    assert np.all(fit.variances > 0)
    assert fit.estimator == "structured_residual_model_discrepancy"


def test_estimator_interfaces_do_not_accept_population_truth():
    forbidden = {"weights", "target_weights", "nmr_weights", "state_populations"}
    for estimator in (fit_curve_moment_variance, fit_structured_residual_variance):
        assert forbidden.isdisjoint(inspect.signature(estimator).parameters)


def test_qualification_gate_and_frozen_artifact_boundary(tmp_path):
    rows = [
        {
            "ensemble": ensemble,
            "panel": panel,
            "heldout_mean_mse_ratio": 1.01,
            "log_variance_spearman": 0.7,
            "mapped_variance_log_rmse": 0.7,
            "constant_mapped_variance_log_rmse": 1.0,
            "beats_shuffled_geometry": True,
            "psd": True,
            "finite_objective": True,
        }
        for ensemble in ("ISO_BI", "ISO_TRI")
        for panel in ("equal", "random_fixed", "random_variable")
    ]
    decision = qualification_gate(rows)
    assert decision["qualified"]
    assert not qualification_gate(
        [row for row in rows if row["ensemble"] == "ISO_BI"]
    )["qualified"]
    artifact = tmp_path / "frozen.json"
    write_frozen_settings(
        artifact,
        settings={"estimator": "curve_moment", "geometry": "identity"},
        qualification=decision,
    )
    assert load_frozen_settings(artifact)["artifact_type"] == ARTIFACT_TYPE

    former_stage_j = tmp_path / "former_stage_j.json"
    former_stage_j.write_text(json.dumps({"artifact_version": 1, "iso_primary_method": "unlearned"}))
    with pytest.raises(ValueError, match="former Stage-J"):
        load_frozen_settings(former_stage_j)


def test_failed_qualification_cannot_be_frozen(tmp_path):
    with pytest.raises(ValueError, match="failed TeaA/ISO qualification"):
        write_frozen_settings(
            tmp_path / "failed.json",
            settings={},
            qualification={"qualified": False},
        )
