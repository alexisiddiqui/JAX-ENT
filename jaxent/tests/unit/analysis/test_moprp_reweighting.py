"""Fast unit tests for the Stage B reweighting harness (pure logic; no optimization).

The heavy optimization path is exercised by the runner's ``--smoke`` integration mode; here
we lock the timepoint-fold construction and the CV selection rule, which must never use
recovery percent.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def reweighting():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_covariance_recovery")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def test_time_folds_partition(reweighting):
    folds = reweighting._time_folds(15, block=3)
    assert len(folds) == 5
    covered = np.concatenate(folds)
    np.testing.assert_array_equal(np.sort(covered), np.arange(15))
    assert all(fold.size == 3 for fold in folds)


def _baseline_row(val_mse, recovery):
    return {
        "ensemble": "E", "coefficient": "published", "method": "baseline",
        "coordinate": "none", "gamma": 0.0, "eta": 0.0, "fold": 0,
        "train_objective": 0.1, "val_mse": val_mse, "val_mse_uniform": 0.30,
        "val_cov_loss": float("nan"), "recovery_percent": recovery, "decoy_mass": 0.0,
    }


def _sym_row(gamma, val_mse, cov, recovery):
    return {
        "ensemble": "E", "coefficient": "published", "method": "symmetric",
        "coordinate": "logpf_projected", "gamma": gamma, "eta": 0.0, "fold": 0,
        "train_objective": 0.1, "val_mse": val_mse, "val_mse_uniform": 0.30,
        "val_cov_loss": cov, "recovery_percent": recovery, "decoy_mass": 0.0,
    }


def _pick(selected, method, coordinate):
    return next(r for r in selected if r["method"] == method and r["coordinate"] == coordinate)


def test_selection_symmetric_uses_covariance_not_recovery(reweighting):
    # Both symmetric settings pass the mean gate (<=1.05x baseline 0.10); the rule must pick
    # the lowest covariance loss, never the higher recovery.
    rows = [
        _baseline_row(0.10, 50.0),
        _sym_row(1.0, 0.10, 0.5, 95.0),
        _sym_row(10.0, 0.104, 0.2, 40.0),
    ]
    selected = reweighting._select(rows)
    sym = _pick(selected, "symmetric", "logpf_projected")
    assert sym["selected_gamma"] == 10.0
    assert sym["val_cov_loss"] == pytest.approx(0.2)


def test_selection_symmetric_rejects_mean_gate_violators(reweighting):
    # gamma=10 has the lowest covariance loss and best recovery but violates the mean gate
    # (0.30 > 1.05*0.10 baseline), so the gated rule must keep gamma=1.
    rows = [
        _baseline_row(0.10, 20.0),
        _sym_row(1.0, 0.10, 0.5, 20.0),
        _sym_row(10.0, 0.30, 0.01, 99.0),
    ]
    selected = reweighting._select(rows)
    sym = _pick(selected, "symmetric", "logpf_projected")
    assert sym["selected_gamma"] == 1.0
    assert sym["mean_gate_passed"] is True


def test_selection_gates_against_baseline_not_uniform(reweighting):
    # The gate must reference the baseline METHOD (0.10), not uniform (0.30). A symmetric
    # fit at val_mse 0.32 would pass a uniform gate but must fail the baseline gate.
    rows = [
        _baseline_row(0.10, 20.0),
        _sym_row(10.0, 0.32, 0.01, 99.0),
        _sym_row(0.1, 0.105, 0.4, 25.0),
    ]
    selected = reweighting._select(rows)
    sym = _pick(selected, "symmetric", "logpf_projected")
    assert sym["selected_gamma"] == 0.1  # the only setting within 1.05x*0.10
    assert sym["promotable"] is True  # 25% > baseline 20% and mean gate passed


def test_selection_dynamic_uses_validation_mse(reweighting):
    rows = [
        _baseline_row(0.10, 50.0),
        {
            "ensemble": "E", "coefficient": "published", "method": "dynamic",
            "coordinate": "uptake_projected", "gamma": 0.0, "eta": 0.0, "fold": 0,
            "train_objective": 0.1, "val_mse": 0.20, "val_mse_uniform": 0.30,
            "val_cov_loss": 0.01, "recovery_percent": 99.0, "decoy_mass": 0.0,
        },
        {
            "ensemble": "E", "coefficient": "published", "method": "dynamic",
            "coordinate": "uptake_projected", "gamma": 0.0, "eta": 0.1, "fold": 0,
            "train_objective": 0.1, "val_mse": 0.05, "val_mse_uniform": 0.30,
            "val_cov_loss": 0.9, "recovery_percent": 10.0, "decoy_mass": 0.0,
        },
    ]
    selected = reweighting._select(rows)
    dyn = _pick(selected, "dynamic", "uptake_projected")
    assert dyn["selected_eta"] == 0.1  # lowest validation MSE, recovery ignored
