import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.distance import pdist

from jaxent.examples.ATLAS_BV.analysis import absolute_l1_stage2
from jaxent.examples.ATLAS_BV.analysis.basin_census import align_to_reference
from jaxent.examples.ATLAS_BV.analysis.absolute_l1_stage2 import (
    absolute_l1_jax,
    absolute_l1_numpy,
    fit_coefficients,
    paired_improvement,
    profiled_scale,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    distribution_distances,
    integrated_autocorrelation_frames,
    post_equilibration_indices,
)
from jaxent.examples.ATLAS_BV.analysis.evaluate_gate import evaluate_census
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    align_to_structure,
    calibration_metrics,
    make_fold_pairs,
    pair_rmsd,
    pf_pair_distance,
    transform_logpf,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    effective_endpoint_frames,
    intraframe_distance_distributions,
    quantile_signatures,
    target_bands,
    w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 import (
    boundary_predictions,
    endpoint_slopes,
    probability_distribution_errors,
    residual_intervals,
)
from jaxent.examples.ATLAS_BV.analysis.within_basin_stage1 import (
    decision,
    fixed_width_edges,
    weighted_line,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    absolute_change_vectors,
    deterministic_cap,
    feature_standardizer,
    make_inner_directions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import (
    fit_selected,
    preprocess_features,
    select_pca_ridge,
    select_ridge,
)
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import (
    conditional_distribution_errors,
    exact_neighbors,
    inverse_distance_weights,
    point_prediction,
    probability_mass_errors,
    select_k,
)
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import (
    bootstrap_median_ci,
    holm_adjust,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
    cross_fitted_scale,
    gaussian_conditional_mass,
    gaussian_nll,
    log_variance_target,
    select_variance_ridge,
    valid_checkpoint,
)
from jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_checkpoint5 import (
    calibrated_q90,
    cross_replica_novelty_residuals,
    fit_scale_calibrator,
    interval_score,
    select_bins,
)
from jaxent.examples.ATLAS_BV.analysis.vector_nearest_support_checkpoint7 import (
    calibrated_variance,
    inner_nearest_residual_sets,
)
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import (
    extrapolation_scale,
    frame_novelty,
    finite_conformal_quantile,
    fit_ridge_a_only,
    mondrian_quantiles,
    ordered_assignments,
    support_category,
)
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import (
    extended_mass_errors,
    gaussian_mass as strict_gaussian_mass,
    interval_score as strict_interval_score,
)


def test_holm_adjust_controls_family_and_preserves_order():
    adjusted = holm_adjust(np.array([0.04, 0.001, 0.02]))

    assert adjusted == pytest.approx([0.04, 0.003, 0.04])


def test_system_bootstrap_median_ci_is_deterministic():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert bootstrap_median_ci(values, 100, 7) == bootstrap_median_ci(values, 100, 7)


def test_strict_likelihood_gaussian_mass_is_normalized():
    mass = strict_gaussian_mass(
        np.array([0.25, 0.75]), np.array([0.1, 0.2]), 0.0, 1.0, 10
    )

    assert np.isfinite(mass).all()
    assert mass.sum() == pytest.approx(1.0)


def test_strict_interval_score_penalizes_misses():
    covered = strict_interval_score(np.array([0.5]), np.array([0.0]), np.array([1.0]))
    missed = strict_interval_score(np.array([2.0]), np.array([0.0]), np.array([1.0]))

    assert missed[0] > covered[0]


def test_extended_mass_errors_are_zero_for_exact_probability_mass():
    mass = np.array([0.1, 0.2, 0.7])
    result = extended_mass_errors(mass, mass, 1e-12)

    assert result["distribution_recovery"] == pytest.approx(1.0)
    assert result["distribution_cosine_distance"] == pytest.approx(0.0)
    assert result["distribution_correlation_distance"] == pytest.approx(0.0)


def test_gaussian_conditional_mass_is_normalized_and_finite():
    mass = gaussian_conditional_mass(
        np.array([-100.0, 0.5, 100.0]), np.array([0.1, 0.2, 0.1]), 0.0, 1.0, 10
    )

    assert np.isfinite(mass).all()
    assert mass.sum() == pytest.approx(1.0)


def test_gaussian_nll_prefers_matching_variance():
    residual = np.array([-1.0, 1.0])

    assert gaussian_nll(residual, np.ones(2)) < gaussian_nll(residual, np.full(2, 100.0))


def test_variance_ridge_selection_is_finite():
    x = np.arange(8, dtype=float)[:, None]
    residual_sets = [(x, np.linspace(0.1, 0.8, 8)), (x + 0.2, np.linspace(0.2, 0.9, 8))]
    selected = select_variance_ridge(residual_sets, [0.01, 1.0], 1e-6)

    assert selected.alpha in (0.01, 1.0)
    assert np.isfinite(selected.inner_nll)


def test_log_variance_target_applies_chi_square_bias_correction():
    corrected = log_variance_target(np.ones(3), 0.0)

    assert (corrected > 1.0).all()


def test_cross_fitted_scale_is_positive_and_finite():
    x = np.arange(8, dtype=float)[:, None]
    residual_sets = [(x, np.linspace(0.1, 0.8, 8)), (x + 0.2, np.linspace(0.2, 0.9, 8))]

    scale = cross_fitted_scale(residual_sets, 1.0, 1e-6)
    assert np.isfinite(scale)
    assert scale > 0


def test_likelihood_checkpoint_validation(tmp_path):
    system = "test_A"
    results = pd.DataFrame({"system_id": [system] * 3, "heldout_replica": [1, 2, 3]})
    hyperparameters = results.copy()
    atomic_parquet(results, tmp_path / f"{system}.results.parquet")
    atomic_parquet(hyperparameters, tmp_path / f"{system}.hyperparameters.parquet")

    assert valid_checkpoint(tmp_path, system)


def test_scale_calibrator_is_positive_and_clips_boundaries():
    mean = np.linspace(0, 1, 100)
    residual = 0.1 + mean
    calibrator = fit_scale_calibrator(mean, residual, 5)
    predicted = calibrated_q90(calibrator, np.array([-10.0, 0.5, 10.0]))

    assert (predicted > 0).all()
    assert predicted[0] == calibrator.quantiles[0]
    assert predicted[-1] == calibrator.quantiles[-1]


def test_interval_score_penalizes_too_narrow_intervals():
    residual = np.array([0.0, 0.5, 1.0])

    assert interval_score(residual, np.full(3, 0.1)) > interval_score(residual, np.ones(3))


def test_select_bins_includes_constant_scale_candidate():
    mean = np.linspace(0, 1, 100)
    sets = [(mean, np.ones(100)), (mean + 0.01, np.ones(100))]

    assert select_bins(sets, [1, 5]).bins == 1


def test_pf_novelty_is_residue_count_normalized():
    fit = np.array([[0.0, 0.0], [2.0, 2.0]])
    validation = np.array([[2.0, 2.0]])
    inner = [(fit, validation, {}, {})]
    sets = cross_replica_novelty_residuals(inner, [(np.array([0.0]), np.array([1.0]))], 1e-8)

    assert sets[0][0][0] == pytest.approx(1.0)


def test_monotone_scale_calibrator_never_shrinks_with_novelty():
    coordinate = np.arange(100, dtype=float)
    residual = np.tile([10.0, 0.1], 50)
    calibrator = fit_scale_calibrator(coordinate, residual, 10, monotone=True)

    assert (np.diff(calibrator.quantiles) >= 0).all()


def test_nearest_support_distance_is_directional_and_normalized():
    fit = np.array([[0.0, 0.0], [2.0, 2.0]])
    validation = np.array([[1.0, 1.0]])
    sets = inner_nearest_residual_sets(
        [(fit, validation, {}, {})], [(np.array([0.0]), np.array([0.5]))], 1e-8
    )

    assert sets[0][0][0] == pytest.approx(1.0)
    assert sets[0][1][0] == pytest.approx(0.5)


def test_calibrated_variance_converts_gaussian_q90():
    variance = calibrated_variance(np.array([1.6448536269514722]))

    assert variance[0] == pytest.approx(1.0)


def test_strict_conformal_has_all_six_ordered_assignments():
    assignments = ordered_assignments()

    assert len(assignments) == 6
    assert len(set(assignments)) == 6
    assert all(set(assignment) == {1, 2, 3} for assignment in assignments)


def test_finite_conformal_quantile_uses_corrected_rank():
    scores = np.arange(1, 10, dtype=float)

    assert finite_conformal_quantile(scores, 0.9) == 9.0


def test_mondrian_quantiles_fall_back_for_empty_region():
    quantiles, labels = mondrian_quantiles(
        np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.2]), np.array([1.0, 2.0]),
        np.array([0.1, 1.9]), 3, 0.9, 7.0,
    )

    assert len(quantiles) == len(labels) == 2
    assert np.isfinite(quantiles).all()


def test_extrapolation_scale_is_one_in_support_and_grows_outside():
    train = np.array([0.0, 1.0, 2.0, 3.0])
    scale = extrapolation_scale(np.array([1.0, 4.0, 6.0]), train)

    assert scale[0] == 1.0
    assert scale[2] > scale[1] > 1.0


def test_support_category_uses_declared_precedence():
    category = support_category(
        np.array([True, False, True, True]), np.ones(4, dtype=bool),
        np.array([False, False, False, True]), np.array([False, False, True, True]),
    )

    assert category.tolist() == [
        "common_support", "pf_extrapolation", "pf_vector_oos", "structurally_novel"
    ]


def test_ridge_tuning_is_deterministic_and_a_only():
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = x[:, 0] * 0.5
    first, alpha_first = fit_ridge_a_only(x, y, [0.01, 1.0], 0.2, 9)
    second, alpha_second = fit_ridge_a_only(x, y, [0.01, 1.0], 0.2, 9)

    assert alpha_first == alpha_second
    assert first.predict(x) == pytest.approx(second.predict(x))


def test_frame_novelty_threshold_uses_calibration_not_test():
    fit = np.array([[0.0], [1.0]])
    calibration = np.array([[0.1], [1.1]])
    test = np.array([[0.05], [3.0]])
    distances, threshold = frame_novelty(fit, calibration, test, 0.95)

    assert threshold < 0.2
    assert distances[0] < threshold
    assert distances[1] > threshold


def test_post_equilibration_indices_drop_zero_through_ten_ns():
    indices = post_equilibration_indices(1001, 10.0, 0.1)

    assert indices[0] == 101
    assert indices[-1] == 1000
    assert len(indices) == 900


def test_distribution_distances_are_zero_for_identical_samples():
    values = np.linspace(-1, 1, 100)
    ks, js = distribution_distances(values, values, 10, 30)

    assert ks == 0.0
    assert js == 0.0


def test_autocorrelation_block_is_positive_and_detects_persistence():
    independent = np.tile([-1.0, 1.0], 100)
    persistent = np.repeat([-1.0, 1.0], 100)

    assert integrated_autocorrelation_frames(independent) == 1
    assert integrated_autocorrelation_frames(persistent) > 1


def test_batched_kabsch_alignment_removes_rigid_transform():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    moved = reference @ rotation + np.array([4, 5, 6])

    aligned = align_to_reference(np.stack([reference, moved]))

    np.testing.assert_allclose(aligned[0], aligned[1], atol=1e-12)


def test_census_gate_stops_when_fewer_than_twenty_systems_are_informative():
    census = pd.DataFrame(
        {"usable_basins": [1] * 89 + [0] * 22, "basins": [1] * 111}
    )
    config = {
        "analysis": {
            "stage1": {
                "min_informative_systems": 20,
                "min_usable_basins": 3,
                "min_delta_f_range_kcal_mol": 0.5,
            }
        }
    }

    decision = evaluate_census(census, config)

    assert decision["decision"] == "redesign_required"
    assert decision["stage2_authorized"] is False


def test_fixed_width_edges_respect_configured_bounds():
    values = np.random.default_rng(7).normal(size=900)
    edges = fixed_width_edges(values, {"pc1_bins_min": 15, "pc1_bins_max": 30})

    assert 15 <= len(edges) - 1 <= 30
    assert edges[0] == values.min()
    assert edges[-1] == values.max()


def test_weighted_line_recovers_slope_and_intercept():
    x = np.arange(6, dtype=float)
    y = 2.5 * x - 1.25

    slope, intercept = weighted_line(x, y, np.arange(1, 7))

    np.testing.assert_allclose([slope, intercept], [2.5, -1.25], atol=1e-12)


def test_absolute_l1_numpy_matches_bv_forward_pass():
    rng = np.random.default_rng(11)
    heavy = rng.normal(size=(5, 12))
    acceptor = rng.normal(size=(5, 12))

    expected = absolute_l1_numpy(heavy, acceptor, 3, 0.35, 2.0)
    observed = np.asarray(absolute_l1_jax(heavy, acceptor, 3, 0.35, 2.0))

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_profiled_scale_is_invariant_to_log_density_zero():
    q = np.array([0.0, 1.0, 2.0, 4.0])
    y = -1.7 * q + 8.0
    weights = np.array([20.0, 30.0, 40.0, 50.0])

    scale, loss = profiled_scale(q, y, weights)
    shifted_scale, shifted_loss = profiled_scale(q, y - 123.0, weights)

    np.testing.assert_allclose([scale, shifted_scale], [1.7, 1.7], atol=1e-12)
    np.testing.assert_allclose([loss, shifted_loss], [0.0, 0.0], atol=1e-24)


def test_absolute_l1_fit_recovers_synthetic_coefficients():
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(6), 30)
    delta_heavy = rng.normal(size=(7, len(labels))) + labels[None, :] * rng.normal(
        size=(7, 1)
    )
    delta_acceptor = rng.normal(size=(7, len(labels))) + labels[None, :] * rng.normal(
        size=(7, 1)
    )
    true_bc, true_bh = 0.4, 1.3
    distances = np.abs(true_bc * delta_heavy + true_bh * delta_acceptor).sum(axis=0)
    means = np.array([distances[labels == label].mean() for label in range(6)])
    counts = np.bincount(labels)
    common = np.ones(6, dtype=bool)

    fitted = fit_coefficients(
        delta_heavy,
        delta_acceptor,
        labels,
        common,
        -means + 5.0,
        counts,
        grid_points=257,
    )

    np.testing.assert_allclose([fitted["bc"], fitted["bh"]], [true_bc, true_bh], rtol=2e-3)


def test_within_basin_decision_reports_all_spaces_and_gates_on_absolute_l1():
    metric = lambda rho, slope, null: {  # noqa: E731
        "rho": rho,
        "slope": slope,
        "permutation_rhos": [null] * 20,
    }
    fold = {
        "valid": True,
        "pmf_range_kcal_mol": 1.0,
        "metrics": {
            "G": metric(0.9, 1.0, 0.0),
            "d1": metric(0.8, -1.0, 0.0),
            "d2": metric(0.7, -1.0, 0.0),
            "H": {"rho": 0.2},
            "Rg": {"rho": 0.2},
            "RMSD": {"rho": 0.2},
            "native_contacts": {"rho": 0.2},
        },
    }
    results = [
        {
            "system_id": f"system_{index}",
            "eligible": True,
            "within_basin_converged": False,
            "folds": [fold, fold, fold],
        }
        for index in range(20)
    ]
    config = {
        "analysis": {
            "redesign_version": "test",
            "stage1": {
                "min_informative_systems": 20,
                "min_delta_f_range_kcal_mol": 0.5,
                "population_null_alpha": 0.05,
                "majority_fraction": 0.6,
                "sign_consistency_fraction": 0.7,
            },
        }
    }

    observed = decision(results, config)

    assert observed["primary_coordinate"] == "absolute_l1"
    assert set(observed["thermodynamic_test"]["coordinate_results"]) == {
        "signed_l1",
        "absolute_l1",
        "l2",
    }
    assert observed["stage2_authorized"] is True


def test_stage2_exploratory_override_preserves_failed_stage1(tmp_path, monkeypatch):
    decision_path = tmp_path / "outputs" / "analysis" / "stage1_decision.yaml"
    decision_path.parent.mkdir(parents=True)
    decision_path.write_text(
        yaml.safe_dump(
            {
                "stage1_pass": False,
                "stage2_authorized": False,
                "blocking_gate": "within_basin_stage1_absolute_l1",
                "thermodynamic_test": {
                    "coordinate_results": {
                        "absolute_l1": {
                            "population_permutation_p_one_sided": 0.001,
                            "fraction_beats_best_compactness_baseline": 0.523,
                        }
                    }
                },
            }
        )
    )
    monkeypatch.setattr(absolute_l1_stage2, "HERE", tmp_path)

    with pytest.raises(SystemExit, match="--exploratory-override"):
        absolute_l1_stage2.require_authorized(False)
    decision, authorization = absolute_l1_stage2.require_authorized(True)

    assert decision["stage1_pass"] is False
    assert authorization["exploratory"] is True
    assert authorization["stage1_pass_preserved"] is False


def test_paired_improvement_requires_positive_one_sided_bound():
    improved = paired_improvement([0.2] * 20, samples=1000, seed=3)
    uncertain = paired_improvement([-0.1, 0.1] * 10, samples=1000, seed=3)

    assert improved["success"] is True
    assert uncertain["success"] is False


def test_geometry_alignment_and_pair_rmsd_remove_rigid_transform():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    moved = reference @ rotation + np.array([4, 5, 6])

    aligned = align_to_structure(np.stack([reference, moved]), reference)

    np.testing.assert_allclose(pair_rmsd(aligned, np.array([0]), np.array([1])), 0.0, atol=1e-12)


def test_pf_pair_distances_are_normalized_and_symmetric():
    z = np.array([[0.0, 1.0], [0.0, 3.0]])
    left = np.array([0, 1])
    right = np.array([1, 0])

    l1 = pf_pair_distance(z, left, right, "l1")
    l2 = pf_pair_distance(z, left, right, "l2")
    cosine = pf_pair_distance(z, left, right, "cosine")
    correlation = pf_pair_distance(z, left, right, "correlation")

    np.testing.assert_allclose(l1, [2.0, 2.0])
    np.testing.assert_allclose(l2, [np.sqrt(5.0), np.sqrt(5.0)])
    np.testing.assert_allclose(cosine, [1.0, 1.0])
    np.testing.assert_allclose(correlation, [1.0, 1.0])


def test_cosine_and_correlation_pf_distances_capture_different_invariances():
    z = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    left = np.array([0])
    right = np.array([1])

    cosine = pf_pair_distance(z, left, right, "cosine")
    correlation = pf_pair_distance(z, left, right, "correlation")

    assert cosine[0] > 0.0
    np.testing.assert_allclose(correlation, 0.0, atol=1e-12)


def test_absolute_change_vectors_preserve_residue_identity():
    z = np.array([[1.0, 4.0, 1.0], [2.0, 2.0, 7.0]])
    vectors = absolute_change_vectors(z, np.array([0, 0]), np.array([1, 2]))

    np.testing.assert_allclose(vectors, [[3.0, 0.0], [0.0, 5.0]])


def test_vector_standardizer_uses_training_pairs_and_floors_constants():
    train = np.array([[1.0, 2.0], [3.0, 2.0]])
    mean, sigma, floored = feature_standardizer(train, 1e-6)

    np.testing.assert_allclose(mean, [2.0, 2.0])
    np.testing.assert_allclose(sigma, [1.0, 1e-6])
    assert floored == 1


def test_deterministic_pair_cap_is_stable_sorted_and_unique():
    first = deterministic_cap(100, 12, 41)
    second = deterministic_cap(100, 12, 41)

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 12
    assert np.all(np.diff(first) > 0)


def test_inner_vector_splits_use_only_one_replica_per_side():
    replicas = np.repeat([1, 2, 3], 5)
    frames = np.tile(np.arange(5), 3)
    coordinates = np.zeros((15, 2, 3))
    coordinates[:, 1, 0] = np.arange(15)
    directions = make_inner_directions(
        coordinates, replicas, frames, heldout=3,
        theiler={1: 0, 2: 0, 3: 0}, pair_cap=4, seed=17,
    )

    assert [(item.fit_replica, item.validation_replica) for item in directions] == [(1, 2), (2, 1)]
    for item in directions:
        assert np.all(replicas[item.fit_pairs.left] == item.fit_replica)
        assert np.all(replicas[item.fit_pairs.right] == item.fit_replica)
        assert np.all(replicas[item.validation_pairs.left] == item.validation_replica)
        assert np.all(replicas[item.validation_pairs.right] == item.validation_replica)


def test_vector_preprocessing_fits_zscore_on_training_pairs_only():
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    heldout = np.array([[100.0, 200.0]])
    transformed, transformed_heldout, mean, sigma, floored = preprocess_features(
        train, heldout, "zscore", 1e-8
    )

    np.testing.assert_allclose(mean, [1.0, 3.0])
    np.testing.assert_allclose(sigma, [1.0, 1.0])
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0)
    np.testing.assert_allclose(transformed_heldout, [[99.0, 197.0]])
    assert floored == 0


def test_replica_tuned_ridge_and_pca_ridge_return_valid_selected_models():
    x1 = np.arange(60, dtype=float).reshape(20, 3) / 10
    x2 = x1 + 0.2
    targets1 = {"rmsd": 0.4 * x1[:, 0] + 0.2, "w1": 0.1 * x1[:, 1]}
    targets2 = {"rmsd": 0.4 * x2[:, 0] + 0.2, "w1": 0.1 * x2[:, 1]}
    inner = [(x1, x2, targets1, targets2), (x2, x1, targets2, targets1)]

    ridge = select_ridge(inner, "rmsd", "zscore", [0.01, 1.0], 1e-8)
    pca_ridge = select_pca_ridge(
        inner, "rmsd", "zscore", [0.01, 1.0], [0.8, 0.99], 1e-8
    )

    assert ridge.model == "ridge"
    assert ridge.alpha in {0.01, 1.0}
    assert pca_ridge.model == "pca_ridge"
    assert pca_ridge.pca_variance in {0.8, 0.99}
    prediction, coefficient, metadata = fit_selected(
        pca_ridge, x1, targets1["rmsd"], x2, 1e-8
    )
    assert prediction.shape == (20,)
    assert coefficient.shape == (3,)
    assert 1 <= metadata["components"] <= 3
    assert np.all(prediction >= 0)


def test_inverse_distance_knn_gives_exact_matches_all_probability_mass():
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 4.0]])
    weights = inverse_distance_weights(distances)

    np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    prediction = point_prediction(
        distances, np.array([[1, 0, 2], [0, 1, 2]]),
        np.array([2.0, 5.0, 8.0]), neighbors=3,
    )
    assert prediction[0] == 5.0


def test_exact_knn_and_replica_tuning_select_from_declared_grid():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    query = np.array([[0.1], [2.9]])
    distances, indices = exact_neighbors(train, query, 3)
    targets = {"rmsd": np.array([0.0, 1.0, 2.0, 3.0])}
    validation = {"rmsd": np.array([0.1, 2.9])}
    selected = select_k(
        [(distances, indices, targets, validation), (distances, indices, targets, validation)],
        "rmsd", [1, 3],
    )

    assert selected.neighbors in {1, 3}
    np.testing.assert_array_equal(indices[:, 0], [0, 3])


def test_conditional_knn_distribution_is_exact_for_matching_neighbor_targets():
    distances = np.zeros((4, 1))
    indices = np.arange(4)[:, None]
    target = np.array([0.1, 0.3, 0.6, 0.9])
    result = conditional_distribution_errors(
        distances, indices, target, target, np.ones(4, dtype=bool),
        neighbors=1, low=0.0, high=1.0, bins=5, smoothing=1e-12,
    )

    assert result["distribution_sqrt_jsd"] < 1e-6
    assert result["distribution_recovery"] == pytest.approx(1.0, abs=1e-6)


def test_probability_mass_error_is_dimensionless_and_zero_for_equal_mass():
    result = probability_mass_errors(np.array([2.0, 1.0]), np.array([2.0, 1.0]), 1e-12)

    assert result["distribution_l1"] == 0.0
    assert result["distribution_recovery"] == 1.0


def test_geometry_transforms_use_training_statistics_only():
    z = np.array([[0.0, 2.0, 100.0], [1.0, 5.0, -100.0]])
    train = np.array([0, 1])

    standardized, floored = transform_logpf(z, train, "residue_standardized", 1e-8)
    centered, _ = transform_logpf(z, train, "frame_centered", 1e-8)

    np.testing.assert_allclose(standardized[:, train].mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized[:, train].std(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-12)
    assert floored == 0


def test_geometry_pair_sampling_is_deterministic_and_respects_replica_holdout():
    rng = np.random.default_rng(3)
    coordinates = rng.normal(size=(18, 4, 3))
    replicas = np.repeat([1, 2, 3], 6)
    frames = np.tile(np.arange(6), 3)
    windows = {1: 1, 2: 1, 3: 1}

    first = make_fold_pairs(coordinates, replicas, frames, 3, windows, 20, 10, 17)
    second = make_fold_pairs(coordinates, replicas, frames, 3, windows, 20, 10, 17)

    np.testing.assert_array_equal(first[0].left, second[0].left)
    np.testing.assert_array_equal(first[1].right, second[1].right)
    assert np.all(replicas[first[1].left] == 3)
    assert np.all(replicas[first[1].right] == 3)
    assert np.all(np.abs(frames[first[1].left] - frames[first[1].right]) > 1)


def test_geometry_calibration_recovers_monotone_heldout_relation():
    train_distance = np.linspace(0, 3, 300)
    train_rmsd = 0.5 + 2.0 * train_distance
    test_distance = np.linspace(0.01, 2.99, 200)
    test_rmsd = 0.5 + 2.0 * test_distance

    metrics, prediction, coverage = calibration_metrics(
        train_distance, train_rmsd, test_distance, test_rmsd, interval_bins=10
    )

    assert metrics["skill_vs_train_median"] > 0.99
    assert metrics["spearman_rho"] > 0.99
    assert metrics["interval_90_coverage"] >= 0.9
    assert coverage.mean() >= 0.9
    np.testing.assert_allclose(prediction, test_rmsd, atol=0.03)


def test_geometry_calibration_has_no_skill_for_constant_pf_distance():
    train_rmsd = np.linspace(0, 2, 100)
    test_rmsd = np.linspace(0, 2, 100)

    metrics, _, coverage = calibration_metrics(
        np.zeros(100), train_rmsd, np.zeros(100), test_rmsd, interval_bins=10
    )

    assert metrics["skill_vs_train_median"] <= 0.0
    assert metrics["spearman_rho"] == 0.0
    assert coverage.dtype == np.bool_


def test_intraframe_w1_uses_unique_pairs_and_is_rigid_transform_invariant():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    moved = reference @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 7
    stretched = reference * 2

    distributions = intraframe_distance_distributions(np.stack([reference, moved, stretched]))

    assert distributions.shape == (3, 3)
    np.testing.assert_allclose(w1_pair_distance(distributions, np.array([0]), np.array([1])), 0)
    expected = np.mean(np.abs(np.sort(pdist(reference)) - np.sort(pdist(stretched))))
    np.testing.assert_allclose(
        w1_pair_distance(distributions, np.array([0]), np.array([2])), expected
    )


def test_quantile_w1_signature_is_exact_when_all_quantiles_are_retained():
    rng = np.random.default_rng(31)
    distributions = np.sort(rng.normal(size=(4, 12)), axis=1)
    signatures = quantile_signatures(distributions, distributions.shape[1])

    np.testing.assert_allclose(signatures, distributions)


def test_effective_endpoint_frames_detects_repeated_rare_frame():
    balanced = effective_endpoint_frames(np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3]))
    concentrated = effective_endpoint_frames(np.array([0, 0, 0, 0]), np.array([1, 2, 3, 4]))

    assert balanced[0] == 4
    assert concentrated[0] == 5
    assert concentrated[1] < balanced[1]


def test_rmsd_support_bands_use_declared_physical_regimes():
    test = np.array([1.0, 1.25, 2.49, 2.5, 3.0])
    bands = target_bands("rmsd", np.linspace(0, 4, 10), test)

    assert [name for name, *_ in bands] == ["hyperlocal", "local", "global"]
    np.testing.assert_array_equal(bands[0][1], [True, False, False, False, False])
    np.testing.assert_array_equal(bands[1][1], [False, True, True, False, False])
    np.testing.assert_array_equal(bands[2][1], [False, False, False, True, True])


def test_endpoint_extrapolation_is_continuous_and_nonnegative():
    train_x = np.linspace(1, 3, 200)
    train_y = 2 * train_x + 1
    test_x = np.array([0.5, 1.0, 2.0, 3.0, 3.5])

    result = boundary_predictions(train_x, train_y, test_x, 0.1, 10)

    np.testing.assert_allclose(result["extrapolated_test"], 2 * test_x + 1, atol=1e-10)
    assert result["low_slope"] >= 0
    assert result["high_slope"] >= 0
    assert np.all(result["extrapolated_test"] >= 0)
    np.testing.assert_array_equal(result["in_pf_support"], [False, True, True, True, False])


def test_negative_endpoint_slopes_are_clamped_to_zero():
    x = np.linspace(0, 1, 100)
    low, high = endpoint_slopes(x, -x, 0.1, 5)

    assert low == 0.0
    assert high == 0.0


def test_probability_distribution_errors_are_dimensionless_and_zero_for_exact_fit():
    values = np.array([0.05, 0.15, 0.55, 0.95])
    result = probability_distribution_errors(values, values, 0.0, 1.0, 10, 1e-12)

    assert result["distribution_l1"] == 0.0
    assert result["distribution_l2"] == 0.0
    assert result["distribution_jsd"] == 0.0
    assert result["distribution_sqrt_jsd"] == 0.0
    assert result["distribution_kld_target_to_prediction"] == 0.0
    assert result["distribution_recovery"] == 1.0


def test_probability_distribution_errors_detect_disjoint_probability_mass():
    result = probability_distribution_errors(
        np.zeros(100), np.ones(100), 0.0, 1.0, 10, 1e-12
    )

    assert result["distribution_l1"] == pytest.approx(2.0)
    assert result["distribution_l2"] == pytest.approx(np.sqrt(2.0))
    assert result["distribution_jsd"] == pytest.approx(np.log(2.0))
    assert result["distribution_sqrt_jsd"] == pytest.approx(np.sqrt(np.log(2.0)))


def test_boundary_residual_intervals_follow_requested_test_prediction():
    train_x = np.linspace(0, 2, 200)
    train_y = train_x + np.tile([-0.1, 0.1], 100)
    train_prediction = train_x
    test_x = np.array([0.2, 1.8])
    test_prediction = np.array([5.0, 7.0])

    low, high = residual_intervals(
        train_x, train_y, train_prediction, test_x, test_prediction, bins=4
    )

    np.testing.assert_allclose((low + high) / 2, test_prediction, atol=1e-12)
    assert np.all(high > low)
