from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[3]/"examples/1_IsoValidation_OMass/fitting/jaxENT"
sys.path.insert(0, str(HERE))
from plot_iso_policy_model_selection_bars import PANELS, select_across_parameters, summarize


def test_validation_selection_precedes_split_aggregation_and_excludes_h_above_one():
    rows = []
    for split_index, split in enumerate(("sequence_cluster", "spatial")):
        for index in range(3):
            for method, metric in [("maxent", "none"), ("omc", "work_scale"), ("omc", "rmsd"), ("omc", "pyrosetta")]:
                choices = [(10., 1., 3., .1, 40.), (100., 3., 1., .1, 50.)]
                if method == "omc":
                    choices.append((1000., 0., 0., 10., 99.))
                for strength, mse, sigma, bandwidth, recovery in choices:
                    rows.append(dict(run_id=f"{method}_{metric}_{split}_{index}_{strength}",
                        ensemble="ISO_TRI", method=method, graph_metric=metric, split_type=split,
                        split_idx=index, data_strength=strength, bandwidth=bandwidth if method == "omc" else 0.,
                        val_mse=mse, val_closed_sigma_mse=sigma, step=100, candidate_index=0,
                        recovery_percent=recovery+3*split_index+index, ess_percent=10.))
    frames = {panel: pd.DataFrame(rows) for panel in PANELS}
    selected = select_across_parameters(frames)
    assert len(selected) == 96
    assert selected.loc[selected.selection_metric == "val_mse", "data_strength"].eq(10.).all()
    assert selected.loc[selected.selection_metric == "val_closed_sigma_mse", "data_strength"].eq(100.).all()
    summary = summarize(selected)
    assert summary.n.eq(6).all()
    np.testing.assert_allclose(summary.recovery_sd, np.std(np.arange(6), ddof=1))
    np.testing.assert_allclose(summary.loc[summary.selection_metric == "val_mse", "recovery_mean"], 42.5)
    np.testing.assert_allclose(summary.loc[summary.selection_metric == "val_closed_sigma_mse", "recovery_mean"], 52.5)
    with pytest.raises(ValueError, match="six split-replicate selections"):
        summarize(selected[selected.split_type != "spatial"])
