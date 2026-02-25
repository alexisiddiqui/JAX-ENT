from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ScoringConfig

def compute_model_scores(
    df: pd.DataFrame,
    scoring_config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """Compute model scores from a loss DataFrame.

    Score = ``-log10(val_loss) - train_penalty * log10(train_loss) - kl_penalty * kl + variance_bonus``

    The *train_penalty* and *kl_penalty* default to 1.0 and 0.1 respectively
    (Exp 2 formula) unless overridden via *scoring_config*.
    """
    df = df.copy()
    eps = 1e-300

    # Extract penalty weights
    train_penalty = 1.0
    kl_penalty_weight = 0.1
    if scoring_config and scoring_config.scoring_weights:
        train_penalty = scoring_config.scoring_weights.get("train_penalty", 1.0)
        kl_penalty_weight = scoring_config.scoring_weights.get("kl_penalty", 0.1)

    val_loss = df.get("val_loss", pd.Series(dtype=float)).fillna(np.nan).clip(lower=eps)
    train_loss = df.get("train_loss", pd.Series(dtype=float)).fillna(np.nan).clip(lower=eps)

    base_score = -np.log10(val_loss) - (train_penalty * np.log10(train_loss))
    df["model_score"] = base_score

    # KL divergence penalty
    if "kl_divergence" in df.columns:
        kl = df["kl_divergence"].fillna(0).clip(lower=0)
        df["model_score"] -= kl_penalty_weight * kl

    # Variance bonus (coefficient of variation)
    grouping_cols = [
        c
        for c in ["split_type", "ensemble", "loss_function", "maxent_value", "convergence_step"]
        if c in df.columns
    ]
    if grouping_cols:
        val_mean = df.groupby(grouping_cols)["val_loss"].transform("mean").fillna(np.nan).clip(lower=eps)
        val_std = df.groupby(grouping_cols)["val_loss"].transform("std").fillna(0.0)
        cv = val_std / val_mean
        df["model_score"] += -np.log10(1.0 + cv)

    return df


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """Select the best-scoring model per (split_type, ensemble, loss_function, split)."""
    df_scored = compute_model_scores(df)
    df_maxent = df_scored[df_scored["maxent_value"] > 0].copy()
    if df_maxent.empty:
        return pd.DataFrame()

    return df_maxent.loc[
        df_maxent.groupby(["split_type", "ensemble", "loss_function", "split"])[
            "model_score"
        ].idxmax()
    ]
