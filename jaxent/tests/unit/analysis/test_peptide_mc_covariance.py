import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_peptide_mc_covariance.py"
)
SPEC = importlib.util.spec_from_file_location("peptide_mc_covariance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _analytic_covariance(dimension: int = 4) -> np.ndarray:
    rng = np.random.default_rng(11)
    root = rng.standard_normal((dimension, dimension))
    return root @ root.T + np.eye(dimension)


def test_psd_factor_reproduces_covariance_including_rank_deficient():
    covariance = _analytic_covariance()
    factor = MODULE.psd_factor(covariance)
    np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-10)

    # Curve constructions are rank-deficient; a plain Cholesky would fail here.
    values = np.arange(12, dtype=float).reshape(3, 4)
    singular = MODULE.LITMUS.weighted_covariance(values, np.full(3, 1 / 3))
    assert np.linalg.matrix_rank(singular, tol=1e-10) < 4
    singular_factor = MODULE.psd_factor(singular)
    np.testing.assert_allclose(singular_factor @ singular_factor.T, singular, atol=1e-10)


def test_seeded_sampling_is_deterministic_and_seed_dependent():
    factor = MODULE.psd_factor(_analytic_covariance())
    first = MODULE.draw_standard_normal(factor, 32, np.random.default_rng(7))
    repeat = MODULE.draw_standard_normal(factor, 32, np.random.default_rng(7))
    other = MODULE.draw_standard_normal(factor, 32, np.random.default_rng(8))
    np.testing.assert_array_equal(first, repeat)
    assert not np.allclose(first, other)


def test_condition_seed_is_stable_and_distinguishes_conditions():
    assert MODULE.condition_seed("equal", "raw", 100, 3) == MODULE.condition_seed("equal", "raw", 100, 3)
    assert MODULE.condition_seed("equal", "raw", 100, 3) != MODULE.condition_seed("equal", "raw", 100, 4)


def test_sample_covariance_recovers_analytic_covariance():
    covariance = _analytic_covariance()
    draws = MODULE.draw_standard_normal(MODULE.psd_factor(covariance), 400_000, np.random.default_rng(3))
    estimate = MODULE.sample_covariance(draws, "population")
    assert MODULE.LITMUS.relative_error(estimate, covariance) < 0.02


def test_population_and_sample_normalization_differ_by_bessel_factor():
    draws = MODULE.draw_standard_normal(MODULE.psd_factor(_analytic_covariance()), 50, np.random.default_rng(5))
    population = MODULE.sample_covariance(draws, "population")
    sample = MODULE.sample_covariance(draws, "sample")
    np.testing.assert_allclose(sample, population * 50 / 49, rtol=1e-12)


def test_scale_zero_gives_exact_mean_and_zero_covariance():
    mean = np.asarray([[0.1, 0.4], [0.3, 0.9]])
    unit = MODULE.draw_standard_normal(MODULE.psd_factor(_analytic_covariance(2)), 16, np.random.default_rng(1))
    residuals = 0.0 * unit
    np.testing.assert_array_equal(mean[0] + residuals, np.broadcast_to(mean[0], residuals.shape))
    np.testing.assert_array_equal(MODULE.sample_covariance(residuals, "population"), np.zeros((2, 2)))


def test_supplied_covariance_scales_exactly_as_s_squared():
    covariance = _analytic_covariance()
    unit = MODULE.draw_standard_normal(MODULE.psd_factor(covariance), 2_000, np.random.default_rng(2))
    base = MODULE.sample_covariance(unit, "population")
    for scale in (0.1, 1.0, 3.0):
        scaled = MODULE.sample_covariance(scale * unit, "population")
        np.testing.assert_allclose(scaled, scale**2 * base, rtol=1e-10)


def test_bound_violations_count_entries_and_do_not_clip():
    values = np.asarray([[-0.2, 0.5], [0.5, 1.4]])
    assert MODULE.bound_violation_fraction(values) == 0.5
    # The accounting must not mutate or clip its input.
    np.testing.assert_array_equal(values, np.asarray([[-0.2, 0.5], [0.5, 1.4]]))
    assert MODULE.bound_violation_fraction(np.asarray([[0.0, 1.0]])) == 0.0


def test_profiles_are_extracted_after_shrinking_the_full_matrix():
    covariance = _analytic_covariance()
    alpha = MODULE.PRIMARY_ALPHA
    computed = MODULE.LITMUS.profiles(covariance, alpha)
    shrunk = MODULE.LITMUS.shrink(covariance, alpha)
    np.testing.assert_allclose(computed["marginal"], np.diag(shrunk))
    np.testing.assert_allclose(computed["conditional"], 1.0 / np.diag(np.linalg.inv(shrunk)))
    # Shrinking the marginal profile directly is a different quantity and must not be used.
    assert not np.allclose(computed["marginal"], np.diag(covariance))


def test_evaluate_estimate_is_perfect_and_passes_at_truth():
    covariance = _analytic_covariance()
    row = MODULE.evaluate_estimate(covariance, covariance)
    assert row["full_gate_pass"]
    np.testing.assert_allclose(row["trace_ratio"], 1.0)
    assert row["normalized_frobenius_distance"] < 1e-12
    for profile in ("marginal", "conditional"):
        for alpha in MODULE.ALPHAS:
            assert row[f"{profile}_a{alpha:g}_gate_pass"]
            np.testing.assert_allclose(row[f"{profile}_a{alpha:g}_pearson"], 1.0, atol=1e-10)


def test_shrinkage_is_stable_across_alphas_at_truth():
    covariance = _analytic_covariance()
    row = MODULE.evaluate_estimate(covariance, covariance)
    for alpha in MODULE.ALPHAS:
        assert row[f"conditional_a{alpha:g}_log_rmse"] < 1e-10


def test_stacked_target_is_separable_and_keeps_cross_time_blocks():
    curve = np.linspace(0.05, 0.8, 15).reshape(5, 3)
    peptide = MODULE.LITMUS.curve_covariances(curve, MODULE.TIMEPOINTS)
    stacked = MODULE.stacked_targets(curve, MODULE.TIMEPOINTS)
    name = "curve_raw_uptake"
    assert stacked[name].shape == (15, 15)
    transforms = MODULE.observable_transforms(curve, MODULE.TIMEPOINTS)
    expected = np.kron(MODULE.time_covariance(transforms[name]), peptide[name])
    np.testing.assert_allclose(stacked[name], expected)
    # Cross-time blocks are retained rather than collapsed to a block diagonal.
    assert np.linalg.norm(stacked[name][:3, 3:6]) > 0


def test_observable_transforms_match_the_registered_litmus_constructions():
    curve = np.linspace(0.05, 0.8, 15).reshape(5, 3)
    transforms = MODULE.observable_transforms(curve, MODULE.TIMEPOINTS)
    expected = MODULE.LITMUS.curve_covariances(curve, MODULE.TIMEPOINTS)
    assert set(transforms) == set(expected)
    for name, values in transforms.items():
        recomputed = MODULE.LITMUS.weighted_covariance(values, np.full(len(values), 1 / len(values)))
        np.testing.assert_allclose(recomputed, expected[name], atol=1e-12)


def test_numpy_helpers_agree_with_jax_pf_variance_definitions():
    from jaxent.src.analysis.pf_variance import (
        covariance_profiles_from_covariance,
        shrink_covariance,
        weighted_population_covariance,
    )

    values = np.asarray([[0.0, 1.0, 2.0, 4.0], [1.0, 1.5, 3.0, 2.0], [2.0, 0.5, 1.0, 3.0]])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    numpy_covariance = MODULE.LITMUS.weighted_covariance(values.T, weights)
    jax_covariance = np.asarray(weighted_population_covariance(values, weights))
    np.testing.assert_allclose(numpy_covariance, jax_covariance, atol=1e-6)

    alpha = MODULE.PRIMARY_ALPHA
    np.testing.assert_allclose(
        MODULE.LITMUS.shrink(numpy_covariance, alpha),
        np.asarray(shrink_covariance(jax_covariance, alpha)),
        atol=1e-6,
    )
    _, marginal, _, conditional = covariance_profiles_from_covariance(jax_covariance, alpha)
    numpy_profiles = MODULE.LITMUS.profiles(numpy_covariance, alpha)
    np.testing.assert_allclose(numpy_profiles["marginal"], np.asarray(marginal), atol=1e-6)
    np.testing.assert_allclose(numpy_profiles["conditional"], np.asarray(conditional), atol=1e-6)


def test_registry_separates_the_two_sampling_models():
    registry = MODULE.construction_registry()
    assert set(registry.sampling_model) == set(MODULE.SAMPLING_MODELS)
    for name, rows in registry.groupby("construction"):
        # Each construction is registered once per model it can be expressed in, with a
        # distinct formula; the two models are never merged into one row.
        assert rows.sampling_model.nunique() == len(rows)
        assert rows.formula.nunique() == len(rows)
    slope = registry[registry.construction == "curve_adjacent_survival_slope"]
    assert set(slope.sampling_model) == {"peptide_by_peptide"}


def test_adjacent_slope_is_excluded_from_the_stacked_model():
    curve = np.linspace(0.05, 0.8, 15).reshape(5, 3)
    stacked = MODULE.stacked_targets(curve, MODULE.TIMEPOINTS)
    # It lives on T-1 intervals, so it cannot be a covariance of the T-length drawn curve.
    assert "curve_adjacent_survival_slope" not in stacked
    for matrix in stacked.values():
        assert matrix.shape == (15, 15)


def test_smoke_integration_covers_both_weight_sets_and_three_panels(tmp_path):
    MODULE.run(
        argparse.Namespace(
            results_dir=SCRIPT.parent / "_pf_peptide_moment_final",
            output_dir=tmp_path,
            smoke=True,
        )
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["fitting_performed"] is False
    assert manifest["clipping_applied"] is False
    assert manifest["cluster_recovery_used"] is False

    samples = pd.read_csv(tmp_path / "sample_metrics.csv")
    assert set(samples.weights) == {"uniform", "known_40_60"}
    assert set(samples.panel) == {"equal", "random_fixed", "random_variable"}
    assert set(samples.sampling_model) == set(MODULE.SAMPLING_MODELS)
    # The exact-mean control carries no covariance and is never evaluated.
    assert (samples.scale > 0).all()

    bounds = pd.read_csv(tmp_path / "bound_violations.csv")
    assert (bounds[bounds.scale == 0.0].bound_violation_fraction == 0).all()

    gates = json.loads((tmp_path / "gate_results.json").read_text())
    assert gates["stage2_status"] == "pending_user_review"
    for name in ("report.md", "seed_summaries.csv", "oracle_correspondence.csv", "convergence.png"):
        assert (tmp_path / name).exists()


def test_oracle_correspondence_reports_profiles_not_only_the_full_matrix(tmp_path):
    MODULE.run(
        argparse.Namespace(
            results_dir=SCRIPT.parent / "_pf_peptide_moment_final",
            output_dir=tmp_path,
            smoke=True,
        )
    )
    oracle = pd.read_csv(tmp_path / "oracle_correspondence.csv")
    # Estimability and correspondence are separate questions; the profile columns are the
    # only place the second one is visible for marginal/conditional.
    for profile in ("marginal", "conditional"):
        for alpha in MODULE.ALPHAS:
            for metric in ("pearson", "spearman", "log_rmse", "scale_ratio"):
                assert f"{profile}_a{alpha:g}_{metric}" in oracle.columns
    assert set(oracle.panel) == {"equal", "random_fixed", "random_variable"}
    # Oracle correspondence is descriptive and must never carry a gate verdict.
    assert not [column for column in oracle.columns if "gate" in column or "qualif" in column]


def test_sampling_models_are_never_pooled_in_one_estimator(tmp_path):
    MODULE.run(
        argparse.Namespace(
            results_dir=SCRIPT.parent / "_pf_peptide_moment_final",
            output_dir=tmp_path,
            smoke=True,
        )
    )
    summaries = pd.read_csv(tmp_path / "seed_summaries.csv")
    keys = ["weights", "panel", "sampling_model", "construction", "scale", "n", "normalization"]
    # sampling_model is an aggregation key, so no summary row averages across the two models.
    assert not summaries.duplicated(subset=keys).any()
    shared = summaries.groupby([c for c in keys if c != "sampling_model"]).sampling_model
    assert shared.nunique().max() == 2, "the two models must stay as separate rows, not merge"
    assert (shared.apply(lambda models: models.is_unique)).all()
