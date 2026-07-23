from pathlib import Path

import numpy as np
import pytest

from jaxent.src.analysis.hdx_ex2 import (
    HDXExperimentProtocol,
    build_expfact_peptide_map,
    compare_trajectory_hdx,
    convolve_isotope_and_deuteron_distributions,
    fit_ex2_solution_set,
    load_expfact_dataset,
    load_intrinsic_rate_file,
    peptide_deuteron_count_distribution,
    predict_ex2_uptake,
    predict_trajectory_ex2,
    thin_deuteron_count_distribution,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MOPRP = PACKAGE_ROOT / "examples/2_CrossValidation/data/_MoPrP"
SOURCE_DATA_AVAILABLE = all(
    (MOPRP / filename).exists()
    for filename in (
        "moprp.seq",
        "moprp.list",
        "moprp.times",
        "moprp.dexp",
        "median.pfact",
        "_output/MoPrP_dfrac.dat",
    )
)


@pytest.mark.skipif(not SOURCE_DATA_AVAILABLE, reason="local exPfact validation data unavailable")
def test_source_dataset_uses_exact_clock_and_fitted_peptide_topology():
    dataset = load_expfact_dataset(MOPRP)

    assert dataset.observed_uptake.shape == (14, 15)
    assert dataset.protocol.timepoints_min[0] == pytest.approx(0.0834)
    assert dataset.protocol.timepoints_min[3] == pytest.approx(1.0002)
    assert dataset.protocol.timepoints_min[-1] == pytest.approx(1440.0)
    assert dataset.protocol.experimental_pd == 4.0
    assert dataset.protocol.intrinsic_rate_ph == 4.4
    assert dataset.peptide_map.n_terminal_residues_dropped == 1
    # Peptide 8 uses moprp.list (67--74), not the discrepant clustering moprp.ass file.
    assert dataset.peptide_map.peptide_ends[7] == 74
    assert dataset.peptide_map.active_amide_counts.tolist() == [
        5,
        12,
        14,
        4,
        11,
        6,
        14,
        7,
        4,
        7,
        9,
        20,
        7,
        6,
    ]

    derived = np.loadtxt(MOPRP / "_output/MoPrP_dfrac.dat", comments="#")
    np.testing.assert_allclose(dataset.observed_uptake, derived, rtol=0.0, atol=0.0)


def test_expfact_map_drops_one_peptide_residue_and_excludes_proline():
    peptide_map = build_expfact_peptide_map("APGA", np.asarray([[1, 1, 4]]))

    # IDs 2--4 are in the source interval after the N-terminal drop; ID 2 is Pro.
    assert peptide_map.active_amide_counts.tolist() == [2]
    np.testing.assert_array_equal(np.flatnonzero(peptide_map.matrix[0]) + 1, [3, 4])
    np.testing.assert_allclose(peptide_map.matrix[0, [2, 3]], [0.5, 0.5])


def test_source_intrinsic_rates_have_explicit_units_conditions_and_exclusions():
    rates = load_intrinsic_rate_file(
        MOPRP / "expfact_kint_pH4p4_298K_min.dat",
        provider="exPfact-3Ala",
        temperature_k=298.0,
        ph=4.4,
    )

    assert len(rates.residue_ids) == 101
    assert rates.rates_min[0] == -1
    assert rates.rates_min[13] == -1  # Pro14
    assert rates.rates_min[34] == -1  # Pro35
    assert rates.rates_min[41] == -1  # Pro42
    assert rates.rates_min[54] == pytest.approx(18.599472428894)


def test_single_amide_ex2_forward_matches_analytic_solution():
    peptide_map = build_expfact_peptide_map("AA", np.asarray([[1, 1, 2]]))
    log_pf = np.asarray([np.nan, np.log(10.0)])
    rates = np.asarray([-1.0, 2.0])
    times = np.asarray([0.0, 1.0, 3.0])

    predicted = predict_ex2_uptake(log_pf, rates, times, peptide_map)
    expected = 1.0 - np.exp(-0.2 * times)

    np.testing.assert_allclose(predicted[0], expected, rtol=1e-13, atol=1e-13)


def test_deuteron_count_distribution_contains_centroid_plus_shape_information():
    peptide_map = build_expfact_peptide_map("AAA", np.asarray([[1, 1, 3]]))
    log_pf = np.asarray([np.nan, 0.0, 4.0])
    rates = np.asarray([-1.0, 1.0, 1.0])
    time = 2.0

    distribution = peptide_deuteron_count_distribution(
        log_pf, rates, time, peptide_map, peptide_index=0
    )
    centroid = predict_ex2_uptake(
        log_pf, rates, np.asarray([time]), peptide_map
    )[0, 0]

    assert distribution.shape == (3,)
    np.testing.assert_allclose(distribution.sum(), 1.0)
    expected_count = np.arange(3) @ distribution
    np.testing.assert_allclose(expected_count / 2.0, centroid)
    # Heterogeneous residue probabilities do not give a binomial envelope.
    binomial = np.asarray([(1 - centroid) ** 2, 2 * centroid * (1 - centroid), centroid**2])
    assert not np.allclose(distribution, binomial)


def test_effective_quench_survival_scales_count_mean_and_convolves_with_control():
    pre_quench = np.asarray([0.0, 0.0, 1.0])
    retained = thin_deuteron_count_distribution(pre_quench, 0.25)

    np.testing.assert_allclose(retained, [0.75**2, 2 * 0.25 * 0.75, 0.25**2])
    assert np.arange(3) @ retained == pytest.approx(0.5)

    protonated = np.asarray([0.8, 0.2])
    envelope = convolve_isotope_and_deuteron_distributions(protonated, retained)
    assert envelope.shape == (4,)
    np.testing.assert_allclose(envelope.sum(), 1.0)
    assert np.arange(4) @ envelope == pytest.approx(0.2 + 0.5)


@pytest.mark.skipif(not SOURCE_DATA_AVAILABLE, reason="local exPfact validation data unavailable")
def test_fixed_moprp_pf_vector_matches_source_forward_values():
    dataset = load_expfact_dataset(MOPRP)
    rates = load_intrinsic_rate_file(
        MOPRP / "expfact_kint_pH4p4_298K_min.dat",
        provider="exPfact-3Ala",
        temperature_k=298.0,
        ph=4.4,
    )
    log_pf = np.loadtxt(MOPRP / "median.pfact")[:, 1]

    predicted = predict_ex2_uptake(
        log_pf,
        rates.aligned(dataset.peptide_map.residue_ids),
        dataset.protocol.timepoints_min,
        dataset.peptide_map,
    )

    # Fixed regression values from the exPfact loop and its recomputed 298 K/pH 4.4 rates.
    assert predicted[0, 0] == pytest.approx(0.03502295494673531, rel=1e-12)
    assert predicted[0, -1] == pytest.approx(1.0, abs=1e-14)
    assert predicted[7, 0] == pytest.approx(0.1407832555178447, rel=1e-12)
    assert predicted[11, -1] == pytest.approx(0.8886298786767027, rel=1e-12)


def test_average_first_and_static_frame_mixture_are_distinct_physics():
    peptide_map = build_expfact_peptide_map("AA", np.asarray([[1, 1, 2]]))
    frame_log_pf = np.asarray([[np.nan, np.nan], [0.0, 4.0]])
    rates = np.asarray([-1.0, 1.0])
    times = np.asarray([1.0, 10.0])

    average_first, frame_mixture = predict_trajectory_ex2(
        frame_log_pf, rates, times, peptide_map
    )

    assert average_first.shape == frame_mixture.shape == (1, 2)
    assert not np.allclose(average_first, frame_mixture)
    expected_average = 1.0 - np.exp(-times * np.exp(-2.0))
    expected_mixture = 0.5 * (
        1.0 - np.exp(-times)
        + 1.0
        - np.exp(-times * np.exp(-4.0))
    )
    np.testing.assert_allclose(average_first[0], expected_average)
    np.testing.assert_allclose(frame_mixture[0], expected_mixture)


def test_multistart_fit_retains_solutions_without_residuewise_averaging():
    peptide_map = build_expfact_peptide_map(
        "AAA", np.asarray([[1, 1, 2], [2, 2, 3]])
    )
    rates = np.asarray([-1.0, 1.2, 0.7])
    truth = np.asarray([np.nan, 2.0, 4.0])
    times = np.asarray([0.1, 1.0, 10.0, 100.0])
    observed = predict_ex2_uptake(truth, rates, times, peptide_map)

    fitted = fit_ex2_solution_set(
        observed, rates, times, peptide_map, starts=3, seed=4, maxiter=1000
    )

    assert len(fitted.solutions) == 3
    assert fitted.best.rmse < 1e-7
    np.testing.assert_allclose(fitted.best.log_pf[1:], truth[1:], atol=2e-5)
    lower, upper = fitted.solution_range
    assert lower.shape == upper.shape == truth.shape


def test_source_harmonic_adds_only_the_smoothing_boundary_residue():
    peptide_map = build_expfact_peptide_map("AAAA", np.asarray([[1, 2, 4]]))
    rates = np.asarray([-1.0, 1.0, 1.0, 1.0])
    truth = np.asarray([np.nan, np.nan, 2.0, 3.0])
    times = np.asarray([0.1, 1.0, 10.0])
    observed = predict_ex2_uptake(truth, rates, times, peptide_map)

    unregularized = fit_ex2_solution_set(
        observed, rates, times, peptide_map, starts=1, seed=2, maxiter=500
    )
    harmonic = fit_ex2_solution_set(
        observed,
        rates,
        times,
        peptide_map,
        starts=1,
        seed=2,
        maxiter=500,
        harmonic_strength=1e-8,
    )

    assert np.isnan(unregularized.best.log_pf[1])
    assert np.isfinite(harmonic.best.log_pf[1])
    assert np.isnan(harmonic.best.log_pf[0])


def test_trajectory_comparison_reports_curve_errors_not_covariance_claims():
    peptide_map = build_expfact_peptide_map("AA", np.asarray([[1, 1, 2]]))
    protocol = HDXExperimentProtocol(
        timepoints_min=np.asarray([1.0, 5.0]),
        temperature_k=298.0,
        experimental_pd=4.0,
        intrinsic_rate_ph=4.4,
    )
    frame_log_pf = np.asarray([[np.nan, np.nan], [2.0, 2.0]])
    rates = np.asarray([-1.0, 1.0])
    observed, _ = predict_trajectory_ex2(
        frame_log_pf, rates, protocol.timepoints_min, peptide_map
    )

    comparison = compare_trajectory_hdx(
        frame_log_pf,
        rates,
        protocol,
        peptide_map,
        observed_uptake=observed,
    )

    assert comparison.average_first_rmse == pytest.approx(0.0)
    assert comparison.frame_mixture_rmse == pytest.approx(0.0)
