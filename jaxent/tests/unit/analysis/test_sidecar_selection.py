from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/1_IsoValidation_OMass/fitting/jaxENT"))
import sidecar_selection as selection
import reselect_maxent_sidecar as reselect


def test_missing_new_metric_never_falls_back_to_mse():
    with pytest.raises(ValueError, match="rescore"):
        selection.select_best_rows(pd.DataFrame([dict(run_id="a", val_mse=.1, convergence_rank=0)]))


def test_fixed_closed_validation_formula(monkeypatch):
    p = np.array([[1.2, -.3], [-.3, .8]])
    monkeypatch.setattr(selection, "_validation_precision", lambda *args: p)
    monkeypatch.setattr(selection, "_trajectory_dir", lambda *args: "unused")
    spec = SimpleNamespace(ensemble="ISO_TRI", features_dir="", clustering_dir="", datasplit_dir="",
                           split_type="sequence_cluster", split_idx=0, output_dir="")
    residual = np.array([[.2, .1], [-.1, .3]])
    value = selection.closed_validation_mse(spec, residual, np.zeros_like(residual))
    assert value == pytest.approx(.5*np.trace(residual.T @ p @ residual)/residual.size)


def test_reselection_keeps_original_candidate_set():
    frame = pd.DataFrame(dict(run_id=["a", "a"], step=[100, 200], val_mse=[.1, .2], convergence_rank=[0, 1]))
    candidates = pd.DataFrame(dict(run_id=["a"]*3, step=[100, 150, 200],
                                   mse=[.1, .05, .2], closed_coordinate=[.3, .01, .1]))
    attached = reselect.attach_scores(frame, candidates)
    result = selection.select_best_rows(attached)
    assert result.step.tolist() == [200]  # step 150 is not a convergence checkpoint
    assert len(attached) == len(frame)
    with pytest.raises(ValueError, match="Ambiguous"):
        reselect.attach_scores(frame, pd.concat([candidates, candidates]))
