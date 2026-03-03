#!/usr/bin/env python3
"""Thin wrapper for selected-models ISO/TRI/BI analysis plots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import seaborn as sns

from jaxent.examples.common.analysis import (
    aggregate_df,
    calculate_concordance_maps,
    compute_minimax_df,
    load_and_process_data,
    run_fixed_effects_analysis,
    save_gt_scores,
)
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.plotting import (
    plot_aggregated_analysis,
    plot_fixed_effects,
    plot_minimax_panel,
    plot_rank_panel,
    plot_score_panel,
    setup_publication_style,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot recovery scores and selection analysis.")
    p.add_argument("--before-csv", required=True)
    p.add_argument("--after-csv", required=True)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--include-transformed", action="store_true")
    p.add_argument("--config", default=None)
    a = p.parse_args()

    setup_publication_style()
    sns.set_style("ticks")
    sns.set_context("paper", rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 12, "ytick.labelsize": 12})

    exp_dir = Path(__file__).resolve().parent.parent
    cfg = ExperimentConfig.from_yaml(Path(a.config).resolve() if a.config else exp_dir / "config.yaml")
    os.makedirs(a.output_dir, exist_ok=True)

    df_before, _, df_diff = load_and_process_data(a.before_csv, a.after_csv, a.include_transformed)
    df_before_agg, df_diff_agg = aggregate_df(df_before), aggregate_df(df_diff)

    for y, yerr, ylabel, title, fname, df in [
        ("mean", "std", "Mean Recovery Score (%)", "Mean Recovery Score (Overall)", "mean_recovery_score_overall", df_before_agg),
        ("cv", "cv_std", "Coefficient of Variation", "Coefficient of Variation (Overall)", "cv_overall", df_before_agg),
        ("mean", "std", "Filtering Regret (%)", "Filtering Regret (Overall)", "filtering_regret_overall", df_diff_agg),
    ]:
        plot_score_panel(df, y, yerr, ylabel, title, a.output_dir, fname, style=cfg.style)

    if "spearman_mean" in df_before_agg.columns:
        plot_score_panel(
            df_before_agg,
            "spearman_mean",
            "spearman_std",
            "Spearman Correlation",
            "Spearman Correlation (Overall)",
            a.output_dir,
            "spearman_correlation_overall",
            style=cfg.style,
        )

    plot_minimax_panel(df_before, a.output_dir, "minimax_recovery_score", style=cfg.style)
    df_minimax = compute_minimax_df(df_before)
    fe_results = run_fixed_effects_analysis(df_before, df_diff, df_minimax, a.output_dir)
    for score_name, (summary_df, _) in fe_results.items():
        if summary_df is not None:
            plot_fixed_effects(summary_df, score_name, a.output_dir)

    concordance_maps = calculate_concordance_maps(fe_results)
    plot_aggregated_analysis(fe_results, concordance_maps, a.output_dir)

    for df, col, title, fname, transform in [
        (df_before, "mean", "Rank of Mean Recovery", "rank_mean_recovery", None),
        (df_before, "cv", "Rank of CV (-log)", "rank_cv", lambda x: -np.log(x + 1e-10)),
        (df_diff, "mean", "Rank of Regret (-val)", "rank_regret", lambda x: -x),
        (df_minimax, "mean", "Rank of Minimax Recovery", "rank_minimax", None),
    ]:
        plot_rank_panel(df, col, title, a.output_dir, fname, False, transform, style=cfg.style)

    save_gt_scores(df_before, df_diff, df_minimax, a.output_dir)


if __name__ == "__main__":
    main()
