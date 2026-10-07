from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/1_IsoValidation_OMass/fitting/jaxENT'))
import analyze_within_trajectory_selection as analysis


def test_scores_match_native_quadratic_forms():
    residual = np.array([[1., 2.], [-1., 3.]])
    precision = np.array([[1.5, -.2], [-.2, .5]])
    w = np.array([[.5, 1.5], [1.2, .8]])
    scores = analysis.score_errors(residual, precision, np.eye(2), w)
    assert scores['mse'] == np.mean(residual**2)
    assert scores['closed_coordinate'] == .5*scores['mse']
    assert scores['gt_coordinate'] == pytest.approx(
        sum(.5*residual[:, t] @ precision @ residual[:, t] for t in range(2))/4)
    assert scores['gt_uptake_weighted'] == pytest.approx(.5*np.mean(w*residual**2))


def test_selectors_only_select_within_run_and_share_candidate_pool():
    rows = []
    for run in ['a', 'b']:
        for i, recovery in enumerate([40., 80., 60.]):
            rows.append(dict(run_id=run, candidate_index=i, step=i*100,
                recovery_percent=recovery, ess_percent=100-i*20,
                mse=[.1, .2, .3][i], gt_coordinate=[.3, .1, .2][i],
                closed_coordinate=[.3, .2, .1][i], gt_uptake_weighted=[.1, .1, .2][i]))
    selected = analysis.select_within(pd.DataFrame(rows))
    assert len(selected) == 8
    assert selected[selected.selector=='mse'].recovery_regret_pp.tolist() == [40., 40.]
    assert selected[selected.selector=='gt_coordinate'].recovery_regret_pp.tolist() == [0., 0.]
    assert selected[selected.selector=='gt_uptake_weighted'].step.tolist() == [0, 0]
    assert selected.candidates.tolist() == [3]*8


def test_nonfinite_candidate_rejected():
    frame = pd.DataFrame([dict(run_id='a', mse=np.nan, gt_coordinate=0., closed_coordinate=0.,
                              gt_uptake_weighted=0., recovery_percent=50., ess_percent=60.)])
    with pytest.raises(ValueError, match='Nonfinite'):
        analysis.select_within(frame)
