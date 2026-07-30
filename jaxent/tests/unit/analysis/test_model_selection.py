import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.mlm import (
    compute_model_selection_performance,
)
from jaxent.examples.common.analysis.selection import (
    filter_best_convergence_by_validation_mse,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ensemble": "A",
                "split_type": "spatial",
                "split_idx": split,
                "loss_function": "MSE",
                "maxent_value": 10.0,
                "convergence_value": 0.1,
                "val_loss": 0.1,
                "val_mse": 3.0 + split,
                "recovery_percent": 10.0 + split,
            }
            for split in range(3)
        ]
        + [
            {
                "ensemble": "A",
                "split_type": "spatial",
                "split_idx": split,
                "loss_function": "MSE",
                "maxent_value": 10.0,
                "convergence_value": 0.01,
                "val_loss": 10.0,
                "val_mse": 1.0 + split,
                "recovery_percent": 20.0 + 10.0 * split,
            }
            for split in range(3)
        ]
    )


def test_convergence_filter_uses_validation_mse_not_validation_loss() -> None:
    selected = filter_best_convergence_by_validation_mse(_scores())
    assert selected["convergence_value"].tolist() == [0.01, 0.01, 0.01]
    assert selected["val_loss"].tolist() == [10.0, 10.0, 10.0]


def test_model_selection_reports_three_split_mean_and_standard_deviation() -> None:
    summary, by_split = compute_model_selection_performance(
        _scores(),
        metric_cols=["val_mse"],
        target_metric="recovery_percent",
    )
    row = summary.loc[summary["score_metric"] == "val_mse"].iloc[0]
    expected = np.asarray([20.0, 30.0, 40.0])

    assert row["recovery_percent_count"] == 3
    assert row["recovery_percent_mean"] == np.mean(expected)
    assert row["recovery_percent_std"] == np.std(expected, ddof=1)
    assert set(by_split["convergence_value"]) == {0.01}
    assert set(by_split["maxent_value"]) == {10.0}
