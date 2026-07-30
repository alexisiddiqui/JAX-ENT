"""Shared model-selection helpers for scored convergence candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_best_convergence_by_validation_mse(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the finite minimum-validation-MSE checkpoint for each fitted model."""
    group_cols = ["ensemble", "split_type", "split_idx", "maxent_value"]
    for col in ("loss_function", "bv_reg_value", "bv_reg_function"):
        if col in df.columns:
            group_cols.append(col)

    required = group_cols + ["val_mse"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Cannot select convergence checkpoints; missing columns: "
            + ", ".join(missing)
        )

    out = df.copy()
    out["val_mse"] = pd.to_numeric(out["val_mse"], errors="coerce")
    out = out[np.isfinite(out["val_mse"])].copy()
    out = out.sort_values("val_mse", ascending=True, kind="stable")
    out = out.drop_duplicates(subset=group_cols, keep="first")
    return out.sort_index()
