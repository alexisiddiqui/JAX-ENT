"""Pure-helper tests for the ISO joint-geometry training runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_iso_joint_geometry_prior.py"
)
SPEC = importlib.util.spec_from_file_location("iso_joint_geometry_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
sys.path.insert(0, str(RUNNER_PATH.parent))
try:
    SPEC.loader.exec_module(RUNNER)
finally:
    sys.path.remove(str(RUNNER_PATH.parent))


def test_population_grid_covers_moprp_like_tail_and_keeps_decoys_zero():
    assert RUNNER.POPULATIONS[0] == 0.01
    assert RUNNER.POPULATIONS[-1] == 0.99
    assignments = np.asarray([0, 0, 1, 1, 1, -1, -1])
    weights = RUNNER.population_weights(assignments, 0.975)
    np.testing.assert_allclose(weights[assignments == 0].sum(), 0.975)
    np.testing.assert_allclose(weights[assignments == 1].sum(), 0.025)
    np.testing.assert_array_equal(weights[assignments == -1], 0.0)


def test_population_folds_are_contiguous_complete_and_disjoint():
    folds = RUNNER.population_folds(15)
    assert len(folds) == 5
    np.testing.assert_array_equal(np.concatenate(folds), np.arange(15))
    assert all(np.all(np.diff(fold) == 1) for fold in folds)


def test_geometry_selection_uses_grouped_out_of_fold_error():
    frame = pd.DataFrame(
        [
            {"method": "point", "rank": 0, "ridge": ridge, "score_penalty": 0.0,
             "geometry_mse": error, "fold": fold, "population_index": fold}
            for ridge, error in ((0.1, 0.3), (1.0, 0.1))
            for fold in range(3)
        ]
        + [
            {"method": "family", "rank": rank, "ridge": 1.0, "score_penalty": 0.1,
             "geometry_mse": error, "fold": fold, "population_index": fold}
            for rank, error in ((1, 0.2), (2, 0.05))
            for fold in range(3)
        ]
    )
    selected = RUNNER.select_geometry_settings(frame)
    assert selected["point"]["ridge"] == 1.0
    assert selected["family"]["rank"] == 2


def test_reweighting_selection_applies_mean_gate_before_recovery():
    rows = []
    for ensemble in ("ISO_BI", "ISO_TRI"):
        key = {
            "panel": "equal",
            "split": 0,
            "open_population_target": 0.5,
            "ensemble": ensemble,
        }
        rows.append({**key, "method": "mean_only", "marginal_strength": 0.0,
                     "val_mean_mse": 1.0, "recovery_pct": 40.0, "decoy_mass": 0.3})
        # Higher recovery, but it must be rejected by the 1.05 mean gate.
        rows.append({**key, "method": "point", "marginal_strength": 10.0,
                     "val_mean_mse": 1.2, "recovery_pct": 99.0, "decoy_mass": 0.0})
        rows.append({**key, "method": "point", "marginal_strength": 1.0,
                     "val_mean_mse": 1.0, "recovery_pct": 60.0, "decoy_mass": 0.2})
        rows.append({**key, "method": "family", "marginal_strength": 1.0,
                     "val_mean_mse": 1.0, "recovery_pct": 55.0, "decoy_mass": 0.2})
        rows.append({**key, "method": "unlearned", "marginal_strength": 1.0,
                     "val_mean_mse": 1.0, "recovery_pct": 50.0, "decoy_mass": 0.2})
    selected = RUNNER.select_reweighting_settings(pd.DataFrame(rows))
    assert selected["per_method"]["point"]["marginal_strength"] == 1.0
    assert selected["primary_method"] == "point"
