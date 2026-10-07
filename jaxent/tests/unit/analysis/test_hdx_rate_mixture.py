from __future__ import annotations

import numpy as np

from jaxent.src.analysis.hdx_rate_mixture import (
    chronological_folds,
    fit_shared_rate_mixture,
    hessian_uncertainty,
    peptide_score_covariances,
    prediction_jacobian_diagnostics,
    predict_uptake,
    project_curves_to_rates,
    rate_distribution_summaries,
    rate_bounds,
    uptake_basis,
)


TIMES = np.asarray([0.08, 0.33, 0.67, 1.0, 5.0, 20.0, 60.0, 240.0, 1440.0])


def _exact_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.asarray([0.015, 1.2])
    weights = np.asarray(
        [
            [0.55, 0.35, 0.10],
            [0.20, 0.65, 0.15],
            [0.70, 0.10, 0.20],
            [0.10, 0.75, 0.15],
        ]
    )
    return rates, weights, predict_uptake(TIMES, rates, weights)


def test_basis_and_prediction_are_bounded_and_monotone():
    rates, weights, curves = _exact_curves()
    basis = uptake_basis(TIMES, rates)
    assert basis.shape == (len(TIMES), len(rates))
    assert curves.shape == (len(weights), len(TIMES))
    assert np.all((curves >= 0) & (curves <= 1))
    assert np.all(np.diff(curves, axis=1) >= -1e-12)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_shared_fit_recovers_exact_curves_and_ordered_rates():
    _, _, curves = _exact_curves()
    fit = fit_shared_rate_mixture(curves, TIMES, 2, starts=3, seed=7, maxiter=600)
    lower, upper = rate_bounds(TIMES)
    assert fit.rmse < 2e-3
    assert lower < fit.rates[0] < fit.rates[1] < upper
    assert np.allclose(fit.weights.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(fit.weights >= 0)


def test_hessian_covariances_are_symmetric_and_have_expected_shapes():
    _, _, curves = _exact_curves()
    fit = fit_shared_rate_mixture(curves, TIMES, 2, shrinkage=1e-3, starts=2, seed=9)
    uncertainty = hessian_uncertainty(fit, curves, TIMES, condition_limit=np.inf)
    n_parameters = len(fit.parameters)
    n_weights = fit.weights.size
    assert uncertainty.parameter_covariance.shape == (n_parameters, n_parameters)
    assert uncertainty.joint_weight_covariance.shape == (n_weights, n_weights)
    assert uncertainty.conditional_weight_covariance.shape == (n_weights, n_weights)
    assert uncertainty.rate_covariance.shape == (2, 2)
    assert uncertainty.curve_covariance.shape == (curves.size, curves.size)
    for matrix in (
        uncertainty.parameter_covariance,
        uncertainty.joint_weight_covariance,
        uncertainty.conditional_weight_covariance,
        uncertainty.rate_covariance,
        uncertainty.curve_covariance,
    ):
        assert np.allclose(matrix, matrix.T, atol=1e-7)


def test_prediction_jacobian_reports_local_parameter_identifiability():
    _, _, curves = _exact_curves()
    fit = fit_shared_rate_mixture(curves, TIMES, 2, starts=2, seed=13)
    diagnostics = prediction_jacobian_diagnostics(fit, TIMES)
    assert diagnostics.n_observations == curves.size
    assert diagnostics.n_parameters == len(fit.parameters)
    assert 0 < diagnostics.numerical_rank <= diagnostics.n_parameters
    assert 0 < diagnostics.effective_rank <= diagnostics.numerical_rank + 1e-9
    assert diagnostics.condition_number >= 1


def test_capacity_fitter_supports_more_than_three_components():
    _, _, curves = _exact_curves()
    fit = fit_shared_rate_mixture(curves, TIMES, 4, starts=1, seed=21, maxiter=300)
    assert len(fit.rates) == 4
    assert fit.weights.shape == (len(curves), 5)
    assert np.all(np.diff(fit.rates) > 0)


def test_fixed_rate_projection_and_heterogeneity_covariance():
    rates, weights, curves = _exact_curves()
    projected, fitted = project_curves_to_rates(curves, TIMES, rates, starts=2, seed=11)
    assert np.sqrt(np.mean((fitted - curves) ** 2)) < 1e-4
    assert np.allclose(projected.sum(axis=1), 1.0)
    covariance = peptide_score_covariances(projected)
    assert set(covariance) == {"empirical_population", "ledoit_wolf"}
    assert covariance["empirical_population"].shape == (3, 3)
    assert np.linalg.eigvalsh(covariance["ledoit_wolf"]).min() >= -1e-12


def test_rate_summaries_label_slow_mass_and_do_not_require_component_matching():
    rates, weights, curves = _exact_curves()
    fit = fit_shared_rate_mixture(curves, TIMES, 2, starts=2, seed=31)
    summaries = rate_distribution_summaries(fit)

    assert set(summaries) == {
        "exchanging_amplitude",
        "unresolved_slow_fraction",
        "mean_log_rate",
        "sd_log_rate",
        "log_rate_q10",
        "log_rate_q50",
        "log_rate_q90",
    }
    np.testing.assert_allclose(
        summaries["exchanging_amplitude"] + summaries["unresolved_slow_fraction"], 1.0
    )
    assert np.all(summaries["log_rate_q10"] <= summaries["log_rate_q50"])
    assert np.all(summaries["log_rate_q50"] <= summaries["log_rate_q90"])


def test_chronological_folds_cover_each_timepoint_once():
    folds = chronological_folds(15, 5)
    assert [len(fold) for fold in folds] == [3] * 5
    assert np.array_equal(np.sort(np.concatenate(folds)), np.arange(15))
