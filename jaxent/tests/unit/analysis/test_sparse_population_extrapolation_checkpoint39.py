import numpy as np
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis.sparse_population_extrapolation_checkpoint39 import (
    anchored_prediction,
    data_requirements,
    fit_alpha,
    local_targets_and_weights,
    maximin_order,
    md_distribution_recovery,
    pair_distance,
    reconstruct_value,
)


def test_pair_distance_matches_mean_l1_and_rms_l2_scaling():
    values = np.array([[0.0, 2.0, 4.0], [0.0, 4.0, 8.0]])
    l1 = pair_distance(values)
    l2 = pair_distance(values, kind="l2")
    assert l1[0, 1] == pytest.approx(3.0)
    assert l2[0, 1] == pytest.approx(np.sqrt(10.0))
    np.testing.assert_allclose(l1, l1.T)
    np.testing.assert_allclose(np.diag(l2), 0.0)


def test_two_anchor_scale_and_reconstruction_recover_scalar_line():
    coordinate = np.array([0.0, 1.0, 2.0, 3.0])
    distance = np.abs(coordinate[:, None] - coordinate[None, :])
    target = 2.5 + 3.0 * coordinate
    labels = np.array([0, 3])
    alpha, degenerate = fit_alpha(distance, target, labels)
    assert not degenerate
    assert alpha == pytest.approx(3.0)
    prediction, fitted, _ = anchored_prediction(distance, distance, target, labels)
    assert fitted == pytest.approx(3.0)
    np.testing.assert_allclose(prediction, target)


def test_reconstruction_tie_is_deterministic_and_degenerate_scale_falls_back():
    assert reconstruct_value(
        np.array([0.0, 2.0]), np.array([1.0, 1.0])
    ) == pytest.approx(1.0)
    zero = np.zeros((3, 3))
    target = np.array([1.0, 2.0, 3.0])
    prediction, alpha, degenerate = anchored_prediction(
        zero, zero, target, np.array([0, 2])
    )
    assert degenerate
    assert alpha == 0.0
    np.testing.assert_allclose(prediction, 2.0)


def test_maximin_order_is_complete_deterministic_and_spreads_points():
    x = np.array([0.0, 1.0, 5.0, 6.0])
    distance = np.abs(x[:, None] - x[None, :])
    first = maximin_order(distance)
    second = maximin_order(distance)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(4))
    assert abs(x[first[0]] - x[first[1]]) >= 5.0


def test_local_kernel_targets_exclude_a_landmarks_own_frame_and_normalize():
    positions = np.arange(6, dtype=float)
    distance = np.abs(positions[:, None] - positions[None, :])
    replicas = np.array([1, 1, 2, 2, 3, 3])
    target, weights, effective = local_targets_and_weights(
        distance, np.array([0, 2, 4]), replicas, bandwidth=1.0
    )
    for replica in (1, 2, 3):
        np.testing.assert_allclose(weights[replica].sum(axis=1), 1.0)
        assert np.isfinite(target[replica]).all()
        assert (effective[replica] >= 1.0).all()
    assert weights[1][0, 0] == 0.0
    assert weights[2][1, 0] == 0.0
    assert weights[3][2, 0] == 0.0


def test_md_distribution_recovery_is_one_for_identical_changes():
    target = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    assert md_distribution_recovery(target, target, target) == pytest.approx(1.0)
    collapsed = np.zeros_like(target)
    assert md_distribution_recovery(target, collapsed, target) < 1.0


def test_data_requirement_uses_paired_system_differences_and_saturation():
    rows = []
    for system in ("a", "b", "c"):
        for known, sparse in ((2, 0.87), (4, 0.90)):
            for method, value in (
                ("sparse_alpha", sparse),
                ("label_mean", 0.50),
                ("shuffled_sparse", 0.60),
            ):
                rows.append(
                    {
                        "system_id": system,
                        "mode": "within_C",
                        "metric": "pf_l1",
                        "model": "direct",
                        "known": known,
                        "method": method,
                        "selection": "random",
                        "stratum": "all",
                        "distribution_recovery": value,
                        "nmae": 1.0 - value,
                        "spearman": 0.5 if method == "sparse_alpha" else 0.0,
                    }
                )
    result = data_requirements(pd.DataFrame(rows))
    assert set(result.required_labels) == {2}
    assert set(result.moderate_localization_labels) == {2}
    assert (result.delta_vs_mean_ci_low > 0).all()
    assert (result.delta_vs_shuffle_ci_low > 0).all()
    assert (result.spearman_delta_vs_shuffle_ci_low > 0).all()
