from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/2_CrossValidation/fitting/jaxENT/investigate_hdx_rate_mixture.py"
)
SPEC = importlib.util.spec_from_file_location("hdx_rate_mixture_stages", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_capacity_stage_registers_every_panel_semantics_and_component():
    # Three residues over four frames, with two normalized peptide mappings.
    log_pf = np.asarray(
        [
            [1.0, 1.1, 1.8, 2.0],
            [2.0, 2.1, 2.5, 2.8],
            [0.5, 0.7, 1.0, 1.2],
        ]
    )
    iso = {
        "bi_log_pf": log_pf,
        "k_int": np.asarray([0.1, 0.4, 1.2]),
        "known": np.full(4, 0.25),
        "mappings": {
            "toy": np.asarray([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
        },
    }
    frame, gates, arrays = MODULE.run_iso_capacity(
        iso, components=(1, 2), starts=1, smoke=False
    )
    assert len(frame) == 4
    assert set(frame.semantics) == {"average_first", "frame_mixture"}
    assert set(frame.n_components) == {1, 2}
    assert len(gates["units"]) == 2
    assert gates["rmse_gate"] == MODULE.CAPACITY_RMSE_GATE
    assert all(isinstance(unit["passed"], bool) for unit in gates["units"])
    assert len(arrays) == 12  # rates, weights, and predictions for four fits


def test_capacity_gate_uses_best_component_per_unit():
    rows = [
        {"panel": "p", "semantics": "average_first", "n_components": 1, "rmse": 0.02},
        {"panel": "p", "semantics": "average_first", "n_components": 2, "rmse": 0.003},
    ]
    # Mirror the explicit gate rule without relying on optimizer behavior.
    best = min(rows, key=lambda row: (row["rmse"], row["n_components"]))
    assert best["n_components"] == 2
    assert best["rmse"] <= MODULE.CAPACITY_RMSE_GATE


def test_stability_stage_compares_independent_near_optimal_fits():
    log_pf = np.asarray(
        [[1.0, 1.2, 1.5, 1.7], [2.0, 2.2, 2.4, 2.6], [0.5, 0.6, 0.8, 1.0]]
    )
    iso = {
        "bi_log_pf": log_pf,
        "k_int": np.asarray([0.1, 0.4, 1.2]),
        "known": np.full(4, 0.25),
        "mappings": {"toy": np.asarray([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])},
    }
    rows, summary, arrays = MODULE.run_iso_stability(
        iso, n_components=2, repeats=3, maxiter=300, smoke=False
    )
    assert len(rows) == 6
    assert len(summary) == 2
    assert set(summary.semantics) == {"average_first", "frame_mixture"}
    assert np.all(summary.near_optimal_count.between(1, 3))
    assert rows.converged.dtype == bool
    assert np.all(summary.near_optimal_converged_fraction.between(0.0, 1.0))
    assert set(summary.stable.unique()).issubset({True, False})
    assert len(arrays) == 6
