"""Pure validation-boundary tests for the frozen ISO -> MoPrP runner."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest


FITTING_DIR = Path(__file__).resolve().parents[3] / "examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(FITTING_DIR))
try:
    RUNNER = importlib.import_module("validate_moprp_joint_geometry_prior")
finally:
    sys.path.remove(str(FITTING_DIR))


def test_time_folds_are_five_contiguous_three_point_blocks():
    folds = RUNNER.time_folds(15)
    assert len(folds) == 5
    assert [fold.tolist() for fold in folds] == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
    ]


def _decision_frame(*, primary_gain: float, positive_folds: int = 5) -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for ensemble in ("AF2_MSAss", "AF2_filtered"):
            rows.append({"fold": str(fold), "method": "mean_only", "ensemble": ensemble,
                         "val_mean_mse": 1.0, "recovery_pct": 40.0, "decoy_mass": 0.3})
            rows.append({"fold": str(fold), "method": "shape", "ensemble": ensemble,
                         "val_mean_mse": 1.0, "recovery_pct": 50.0, "decoy_mass": 0.2})
            gain = primary_gain if fold < positive_folds else -1.0
            rows.append({"fold": str(fold), "method": "point", "ensemble": ensemble,
                         "val_mean_mse": 1.02, "recovery_pct": 50.0 + gain,
                         "decoy_mass": 0.15})
            rows.append({"fold": str(fold), "method": "unlearned", "ensemble": ensemble,
                         "val_mean_mse": 1.01, "recovery_pct": 49.0,
                         "decoy_mass": 0.18})
    return pd.DataFrame(rows)


def test_transfer_gate_requires_both_ensembles_and_four_positive_folds():
    passed = RUNNER.evaluate_transfer(_decision_frame(primary_gain=2.0), "point")
    assert passed["transfer_supported"]
    assert passed["learning_adds_value"]

    failed = RUNNER.evaluate_transfer(
        _decision_frame(primary_gain=2.0, positive_folds=3), "point"
    )
    assert not failed["transfer_supported"]


def test_transfer_gate_rejects_mean_degradation():
    frame = _decision_frame(primary_gain=2.0)
    frame.loc[frame.method == "point", "val_mean_mse"] = 1.2
    decision = RUNNER.evaluate_transfer(frame, "point")
    assert not decision["transfer_supported"]


def test_failed_stage_j_runner_is_hard_blocked():
    with pytest.raises(RuntimeError, match="failed its ISO held-out mean gate"):
        RUNNER.prohibit_failed_stage_j_moprp_launch()
