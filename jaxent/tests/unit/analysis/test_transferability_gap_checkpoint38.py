import numpy as np
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis.transferability_gap_checkpoint38 import (
    add_cohort_atypicality,
    aggregate_system_predictor_gaps,
    attach_alpha_mismatch,
    bh_adjust,
    paired_gap_cells,
    primary_mechanism_models,
)


def result_frame(recoveries, pairs=(10, 30)):
    return pd.DataFrame(
        {
            "system_id": ["a", "a"],
            "metric": ["work_scale", "work_scale"],
            "model": ["direct", "direct"],
            "band": ["q2", "q3"],
            "pairs": pairs,
            "distribution_recovery": recoveries,
        }
    )


def test_gap_pairing_and_equal_vs_pair_weighted_aggregation():
    cells = paired_gap_cells(result_frame([0.2, 0.8]), result_frame([0.5, 0.7]))
    np.testing.assert_allclose(cells.signed_gap, [0.3, -0.1])
    np.testing.assert_allclose(cells.absolute_gap, [0.3, 0.1])
    summary = aggregate_system_predictor_gaps(cells).iloc[0]
    assert summary.absolute_gap == pytest.approx(0.2)
    assert summary.signed_gap == pytest.approx(0.1)
    assert summary.absolute_gap_pair_weighted == pytest.approx(0.15)
    assert summary.absolute_gap_q23 == pytest.approx(0.2)
    assert summary.bands == 2


def test_gap_pairing_rejects_mismatched_keys_and_pairs():
    shifted = result_frame([0.2, 0.3])
    shifted.loc[1, "band"] = "q4"
    with pytest.raises(ValueError, match="keys do not match"):
        paired_gap_cells(result_frame([0.2, 0.3]), shifted)
    with pytest.raises(ValueError, match="pair counts differ"):
        paired_gap_cells(result_frame([0.2, 0.3]), result_frame([0.2, 0.3], pairs=(10, 31)))


def test_alpha_mismatch_is_log_ratio_and_marks_primary_rows():
    gaps = aggregate_system_predictor_gaps(
        paired_gap_cells(result_frame([0.2, 0.8]), result_frame([0.5, 0.7]))
    )
    old = pd.DataFrame(
        {"system_id": ["a"], "metric": ["work_scale"], "model": ["direct"], "alpha": [2.0]}
    )
    new = pd.DataFrame(
        {"metric": ["work_scale"], "model": ["direct"], "global_alpha": [4.0]}
    )
    result = attach_alpha_mismatch(gaps, old, new).iloc[0]
    assert result.alpha_mismatch == pytest.approx(np.log(2.0))
    assert result.primary


def test_cohort_atypicality_is_finite_and_composition_aware():
    frame = pd.DataFrame(
        {
            "system_id": ["a", "b", "c", "d"],
            "n_residues": [60, 90, 140, 220],
            "rg_mean": [10.0, 12.0, 16.0, 22.0],
            "helix_fraction": [0.8, 0.5, 0.2, 0.1],
            "sheet_fraction": [0.1, 0.2, 0.5, 0.1],
            "coil_fraction": [0.1, 0.3, 0.3, 0.8],
        }
    )
    result = add_cohort_atypicality(frame, 7)
    columns = [
        "length_atypicality",
        "compactness_atypicality",
        "secondary_structure_atypicality",
        "heterogeneity_atypicality",
    ]
    assert np.isfinite(result[columns]).all().all()
    assert (result[columns] >= 0).all().all()
    assert result.secondary_structure_atypicality.nunique() > 1


def test_bh_adjustment_is_monotonic_in_p_value_order():
    p_values = np.array([0.04, 0.001, 0.2, 0.02])
    adjusted = bh_adjust(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1)


def test_primary_mechanism_inference_is_deterministic_on_small_fixture():
    rows = []
    for system_index in range(8):
        heterogeneity = system_index / 7
        for metric_index, metric in enumerate(("work_scale", "pf_l1")):
            mismatch = heterogeneity + 0.05 * metric_index
            rows.append(
                {
                    "system_id": f"s{system_index}",
                    "metric": metric,
                    "model": "direct",
                    "absolute_gap": 0.2 + 0.4 * mismatch,
                    "alpha_mismatch": mismatch,
                    "heterogeneity_atypicality": heterogeneity,
                    "bands": 3 + system_index % 2,
                    "primary": True,
                }
            )
    frame = pd.DataFrame(rows)
    first = primary_mechanism_models(frame, bootstrap_samples=20, permutations=20, seed=11)
    second = primary_mechanism_models(frame, bootstrap_samples=20, permutations=20, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.model) == {
        "A_atypicality_to_alpha_mismatch",
        "B_atypicality_to_recovery_gap",
        "C_joint_gap_model_heterogeneity",
        "C_joint_gap_model_alpha_mismatch",
    }
