"""Tests for the observed-only MoPrP validation-score study."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def val_scores():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_val_score_correlation")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def test_mse_matches_plain_unweighted_mean(val_scores):
    predicted = np.asarray([[0.1, 0.7], [0.3, 0.8]])
    observed = np.asarray([[0.2, 0.6], [0.5, 0.9]])
    mapping = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    scores = val_scores.validation_scores(predicted, observed, mapping)
    assert scores["mse"] == pytest.approx(np.mean((predicted - observed) ** 2))
    assert set(scores) == set(val_scores.SCORE_NAMES)
    assert all(np.isfinite(value) for value in scores.values())


def test_scores_handle_saturated_and_zero_observations(val_scores):
    predicted = np.asarray([[0.1, 0.9], [0.2, 0.8]])
    observed = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    mapping = np.eye(2)
    scores = val_scores.validation_scores(predicted, observed, mapping)
    assert all(np.isfinite(value) for value in scores.values())


def test_redundancy_weights_downweight_recounted_region(val_scores):
    mapping = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    weights = val_scores.peptide_redundancy_weights(mapping)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] > weights[0]
    assert weights.mean() == pytest.approx(1.0)


def test_all_zero_sensitivity_falls_back_to_finite_mean(val_scores):
    predicted = np.asarray([[0.2, 0.8]])
    observed = np.asarray([[0.0, 0.0]])
    scores = val_scores.validation_scores(predicted, observed, np.asarray([[1.0]]))
    assert scores["sens_mse"] == pytest.approx(scores["mse"])


def test_primary_correlation_is_computed_within_fixed_ess(val_scores):
    rows = []
    for ess in (1.0, 5.0):
        for gamma, recovery in enumerate((10.0, 20.0, 30.0)):
            row = {
                "grid": "fixed_ess",
                "arm": "diag_d_scalefree",
                "gamma": float(gamma),
                "control_value": ess,
                "ensemble": "A",
                "split": 0,
                "recovery": recovery,
            }
            row.update({score: -recovery for score in val_scores.SCORE_NAMES})
            rows.append(row)
    correlations = val_scores.correlation_table(pd.DataFrame(rows))
    condition = correlations[correlations["row_type"] == "condition"]
    assert len(condition) == 2 * len(val_scores.SCORE_NAMES)
    np.testing.assert_allclose(condition["spearman"], -1.0)


def test_e1_ranking_detects_folded_reversal(val_scores):
    rows = []
    for state, value in (("Folded", 0.1), ("PUF3", 0.2)):
        row = {
            "grid": "fixed_ess",
            "ess_target": 1.0,
            "ensemble": "A",
            "split": 0,
            "gamma": 0.0 if state == "Folded" else 0.1,
            "dominant_state": state,
            "recovery": 80.0 if state == "Folded" else 0.0,
        }
        row.update({score: value for score in val_scores.SCORE_NAMES})
        rows.append(row)
    ranking = val_scores.e1_ranking_table(pd.DataFrame(rows))
    assert ranking["comparable"].all()
    assert ranking["reverses_decoy_win"].all()
    np.testing.assert_allclose(ranking["folded_pairwise_win_fraction"], 1.0)
