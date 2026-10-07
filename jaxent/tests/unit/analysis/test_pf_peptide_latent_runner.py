import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_pf_peptides.py"
)
SPEC = importlib.util.spec_from_file_location("investigate_pf_peptides_test_module", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)
sys.modules["investigate_pf_peptides"] = RUNNER
OVERLAP_PATH = RUNNER_PATH.with_name("investigate_pf_peptide_overlap.py")
OVERLAP_SPEC = importlib.util.spec_from_file_location(
    "investigate_pf_peptide_overlap_test_module", OVERLAP_PATH
)
OVERLAP = importlib.util.module_from_spec(OVERLAP_SPEC)
sys.modules[OVERLAP_SPEC.name] = OVERLAP
OVERLAP_SPEC.loader.exec_module(OVERLAP)


def test_truth_weights_encode_cluster_populations_not_per_frame_weights():
    assignments = np.asarray([0, 0, 1, 1, 1, -1])

    weights = RUNNER._truth_weights(assignments)

    np.testing.assert_allclose(weights[assignments == 0].sum(), 0.4)
    np.testing.assert_allclose(weights[assignments == 1].sum(), 0.6)
    np.testing.assert_allclose(weights[assignments == -1].sum(), 0.0)
    np.testing.assert_allclose(weights[assignments == 0], np.asarray([0.2, 0.2]))
    np.testing.assert_allclose(weights[assignments == 1], np.asarray([0.2, 0.2, 0.2]))


def test_target_scale_changes_only_the_analytic_covariance():
    covariance = np.asarray([[2.0, 0.5], [0.5, 1.0]])

    scaled = {scale: scale**2 * covariance for scale in (0.0, 0.1, 1.0)}

    np.testing.assert_array_equal(scaled[0.0], np.zeros_like(covariance))
    np.testing.assert_allclose(scaled[0.1], 0.01 * covariance)
    np.testing.assert_array_equal(scaled[1.0], covariance)


def test_default_target_scales_exclude_nonidentifiable_zero():
    assert RUNNER.Config().target_scales == (0.1, 1.0)


def test_registered_peptide_splits_are_locked_against_clustering_ties():
    train, validation = RUNNER.LOCKED_SPLITS[("equal", 2)]

    assert train == (0, 1, 2, 3, 4, 5)
    assert validation == (6, 7, 8, 9, 10, 11, 12, 13, 14)


def test_equal_panel_bounds_are_contiguous_after_first_residue_trim():
    equal = RUNNER.generate_panel_bounds()["equal"]
    active = [(start + 1, end) for start, end in equal]

    assert len(active) == 15
    assert active[0][0] == 2
    assert active[-1][1] == 309
    assert all(left[1] + 1 == right[0] for left, right in zip(active, active[1:]))


def test_projected_covariance_optimizer_is_finite_and_reduces_structural_loss():
    residue_log_pf = np.asarray(
        [
            [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
            [0.1, 0.4, 1.1, 1.5, 2.1, 2.4],
            [0.3, 0.6, 1.4, 1.7, 2.4, 2.7],
            [0.2, 0.5, 1.2, 1.6, 2.2, 2.5],
        ],
        dtype=np.float32,
    )
    mapping = np.asarray(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5]],
        dtype=np.float32,
    )
    peptide_log_pf = np.asarray(
        RUNNER.map_frame_log_pf_to_peptides(mapping, residue_log_pf)
    )
    target_weights = np.asarray([0.35, 0.25, 0.15, 0.10, 0.10, 0.05], dtype=np.float32)
    target_covariance = np.asarray(
        RUNNER.weighted_population_covariance(peptide_log_pf, target_weights)
    )
    target_uptake = np.asarray(
        RUNNER.map_residue_uptake_to_peptides(
            mapping,
            RUNNER.average_first_uptake(
                residue_log_pf,
                np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                RUNNER.TIMEPOINTS,
                target_weights,
            ),
        )
    )
    train = np.asarray([0, 1])
    projection = RUNNER.overlap_projection(mapping[train], 1e-6)
    initial_covariance = RUNNER.weighted_population_covariance(
        peptide_log_pf[train], np.full(6, 1 / 6, dtype=np.float32)
    )
    initial_loss = float(
        RUNNER.projected_log_euclidean_covariance_loss(
            initial_covariance,
            target_covariance[np.ix_(train, train)],
            projection,
            alpha=0.05,
        )
    )

    result = RUNNER.optimize(
        predicted_residue_log_pf=residue_log_pf,
        predicted_peptide_log_pf=peptide_log_pf,
        predicted_k_ints=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        peptide_mapping=mapping,
        target_peptide_covariance=target_covariance,
        target_uptake=target_uptake,
        clean_target_uptake=target_uptake,
        assignments=np.asarray([0, 0, 1, 1, 2, 2]),
        train_indices=train,
        val_indices=np.asarray([2]),
        method="projected_covariance",
        gamma=10.0,
        maxent_value=0.0,
        alpha=0.05,
        steps=200,
        learning_rate=0.03,
        start_seed=0,
    )

    assert result["finite"]
    assert np.isfinite(result["val_pf_profile_loss"])
    assert result["train_pf_profile_loss"] < initial_loss


def test_projected_calibration_enforces_curve_gate_before_covariance_selection(tmp_path):
    rows = []
    for split in range(3):
        rows.extend(
            [
                {
                    "gamma": 1.0,
                    "maxent_value": 1.0,
                    "alpha": 0.05,
                    "split_index": split,
                    "start": 0,
                    "train_total": 0.1,
                    "val_curve_mse": 1.10,
                    "val_pf_profile_loss": 0.01,
                },
                {
                    "gamma": 10.0,
                    "maxent_value": 1.0,
                    "alpha": 0.10,
                    "split_index": split,
                    "start": 0,
                    "train_total": 0.2,
                    "val_curve_mse": 0.90,
                    "val_pf_profile_loss": 0.02,
                },
            ]
        )
    comparator = pd.DataFrame(
        {
            "layout_id": ["calibration"] * 3,
            "method": ["covariance_mse"] * 3,
            "split_index": [0, 1, 2],
            "val_curve_mse": [1.0, 1.0, 1.0],
        }
    )
    comparator.to_csv(tmp_path / "selected_results.csv", index=False)
    layout = OVERLAP.Layout("calibration", 30, "random_fixed", 0, ((0, 2),))

    frozen, decision = OVERLAP._select_projected_calibration(
        pd.DataFrame(rows), tmp_path, layout
    )

    assert decision["status"] == "passed"
    assert decision["selection_uses_recovery"] is False
    assert frozen.iloc[0].gamma == 10.0
    assert frozen.iloc[0].median_val_curve_mse_ratio == 0.9
