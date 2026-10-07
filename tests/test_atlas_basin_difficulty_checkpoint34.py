import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.basin_difficulty_reanalysis_checkpoint34 import (
    DIFFICULTY_LABELS,
    SIZE_LABELS,
    add_bins,
    summarize,
)


def synthetic_rows():
    rows = []
    for system in range(8):
        for level in (0.1, 0.25, 0.4):
            error = 0.01 + 0.01 * system + level / 100
            rows.append(
                {
                    "system_id": f"s{system}",
                    "n_residues": 60 + 20 * system,
                    "maxent_tv": error,
                    "tv_gain": error / 2,
                    "fractional_tv_recovery": 0.5,
                    "population_mae_gain": error / 4,
                }
            )
    return pd.DataFrame(rows)


def test_bins_are_assigned_at_challenge_and_system_levels():
    result = add_bins(synthetic_rows())
    assert tuple(result.difficulty_bin.cat.categories) == DIFFICULTY_LABELS
    assert tuple(result.size_quartile.cat.categories) == SIZE_LABELS
    assert result.groupby("system_id").size().eq(3).all()


def test_summary_reports_absolute_and_fractional_recovery():
    result = add_bins(synthetic_rows())
    summary = summarize(result, ["difficulty_bin"])
    assert len(summary) == 4
    assert summary.systems.min() >= 2
    assert (summary.fractional_tv_recovery_mean == 0.5).all()
