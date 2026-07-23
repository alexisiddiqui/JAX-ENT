"""Tests for the Stage H shape-prior reweighting selection (largest γ within the mean gate)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def rw():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_shape_prior_reweighting")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def _row(gamma, val_mse, recovery):
    return {"gamma": gamma, "eta": 0.0, "val_mse": val_mse, "recovery": recovery, "decoy": 0.0}


def test_selects_largest_gamma_within_gate(rw):
    baseline = 0.030  # gate = 0.0315
    agg = [_row(0.1, 0.0300, 60.0), _row(1.0, 0.0308, 75.0), _row(10.0, 0.0312, 88.0), _row(30.0, 0.0400, 95.0)]
    sel = rw._select_largest_gamma(agg, baseline)
    # γ=30 fails the gate (0.040 > 0.0315); γ=10 is the largest that passes
    assert sel["gamma"] == 10.0
    assert sel["recovery"] == 88.0


def test_falls_back_to_min_mse_when_none_pass_gate(rw):
    baseline = 0.030
    agg = [_row(0.1, 0.050, 60.0), _row(1.0, 0.040, 70.0)]  # none within 1.05×0.030
    sel = rw._select_largest_gamma(agg, baseline)
    # no eligible -> pool is all; largest γ preferred, tie-break by val_mse
    assert sel["gamma"] == 1.0


def test_tie_break_eta_by_val_mse(rw):
    baseline = 0.030
    agg = [
        {"gamma": 10.0, "eta": 0.0, "val_mse": 0.0314, "recovery": 80.0, "decoy": 0.0},
        {"gamma": 10.0, "eta": 0.01, "val_mse": 0.0310, "recovery": 85.0, "decoy": 0.0},
    ]
    sel = rw._select_largest_gamma(agg, baseline)
    assert sel["gamma"] == 10.0 and sel["eta"] == 0.01  # lower val_mse wins the tie
