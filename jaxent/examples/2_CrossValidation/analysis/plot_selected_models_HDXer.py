#!/usr/bin/env python3
"""Selected-model analysis plots for HDXer reweighting results — Example 2.

Cluster populations and recovery are computed using the same jaxENT pipeline
(calculate_cluster_ratios + calculate_recovery_percentage) so that the state
definitions and cluster map are shared with the jaxENT analysis and fully
adjustable via config.yaml and state_ratios.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jaxent.examples.common.analysis import (
    aggregate_df,
    calculate_cluster_ratios,
    calculate_recovery_percentage,
    compute_minimax_df,
    kl_divergence,
    save_gt_scores,
)
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.plotting import (
    plot_cluster_populations,
    plot_cluster_populations_by_split,
    plot_kl_divergence_panel,
    plot_minimax_panel,
    plot_rank_panel,
    plot_score_panel,
    setup_publication_style,
)

# HDXer ensemble/split naming → jaxENT naming
_ENSEMBLE_MAP = {
    "MoPrP_af_clean": "AF2_filtered",
    "MoPrP_af_dirty":  "AF2_MSAss",
}
_SPLIT_MAP = {
    "R3_k_sequence":    "sequence_cluster",
    "Sp_res_neighbours": "spatial",
}
# Reverse maps for building NPZ filenames
_ENSEMBLE_MAP_R = {v: k for k, v in _ENSEMBLE_MAP.items()}
_SPLIT_MAP_R    = {v: k for k, v in _SPLIT_MAP.items()}

_DEFAULT_WEIGHTS_DIR = (
    "analysis/pymol_viz_configs/HDXer/_analysis_output_JSD4/weights"
)
_REPLICATES = [1, 2, 3]


def _load_target_ratios(
    state_ratios_json: str | Path,
    state_mapping: dict[int, str],
) -> dict[str, float]:
    """Build target_ratios dict matching state_mapping values from state_ratios.json."""
    with open(state_ratios_json) as f:
        data = json.load(f)
    fp = data["fractional_populations"]
    state_name_lower = {v.lower(): v for v in state_mapping.values()}
    target_ratios: dict[str, float] = {}
    for key, val in fp.items():
        if key == "sum":
            continue
        matched = state_name_lower.get(key.lower())
        if matched:
            frac = val["fraction"] if isinstance(val, dict) else float(val)
            target_ratios[matched] = frac
    return target_ratios


def _load_hdxer_data(
    weights_dir: str | Path,
    clustering_dir: str | Path,
    ensemble_clustering_map: dict[str, str],
    state_mapping: dict[int, str],
    target_ratios: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load per-replicate HDXer weights and compute cluster populations + recovery.

    Returns (df_before_aggregated, df_per_replicate, cluster_cols).
    """
    weights_dir   = Path(weights_dir)
    clustering_dir = Path(clustering_dir)

    # Pre-load cluster assignments for each ensemble
    cluster_assignments: dict[str, np.ndarray] = {}
    for jaxent_ens, subdir in ensemble_clustering_map.items():
        csv_path = clustering_dir / subdir / f"{subdir}_frame_to_cluster.csv"
        if not csv_path.exists():
            print(f"Warning: cluster file not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        cluster_assignments[jaxent_ens] = df["cluster_label"].values

    rows: list[dict] = []
    for jaxent_ens, hdxer_ens in _ENSEMBLE_MAP_R.items():
        if jaxent_ens not in cluster_assignments:
            continue
        ca = cluster_assignments[jaxent_ens]

        for jaxent_split, hdxer_split in _SPLIT_MAP_R.items():
            for rep in _REPLICATES:
                npz_name = f"weights_{hdxer_ens}_{hdxer_split}_{rep}.npz"
                npz_path = weights_dir / npz_name
                if not npz_path.exists():
                    print(f"Warning: {npz_path} not found, skipping")
                    continue

                raw_w = np.load(npz_path)["weights"].astype(float)
                if len(raw_w) != len(ca):
                    print(f"Warning: weight/cluster length mismatch for {npz_name} "
                          f"({len(raw_w)} vs {len(ca)}), skipping")
                    continue

                total = raw_w.sum()
                if total == 0:
                    continue
                w = raw_w / total

                # Cluster populations (sum of weight in each macro-cluster)
                ratios = calculate_cluster_ratios(ca, w)

                # JSD-based recovery (same formula as jaxENT)
                recovery = calculate_recovery_percentage(
                    ca, w, target_ratios, state_mapping
                )

                # KL divergence from uniform prior
                uniform = np.ones(len(w)) / len(w)
                kl_div = kl_divergence(w, uniform)

                row: dict = {
                    "ensemble":           jaxent_ens,
                    "split_type":         jaxent_split,
                    "replicate":          rep,
                    "open_state_recovery": recovery,
                    "kl_div_uniform":     kl_div,
                }
                row.update(ratios)
                rows.append(row)

    if not rows:
        raise RuntimeError("No HDXer weight files could be loaded. Check --weights-dir.")

    raw = pd.DataFrame(rows)
    cluster_cols = sorted(c for c in raw.columns if c.startswith("cluster_"))
    # Ensembles may have different macro-cluster sets; fill missing cluster fractions with 0
    raw[cluster_cols] = raw[cluster_cols].fillna(0.0)

    # Aggregate across replicates
    group_cols = ["ensemble", "split_type"]
    agg_dict: dict = {
        "mean":               ("open_state_recovery", "mean"),
        "std":                ("open_state_recovery", "std"),
        "count":              ("open_state_recovery", "count"),
        "min":                ("open_state_recovery", "min"),
        "kl_divergence_mean": ("kl_div_uniform",      "mean"),
        "kl_divergence_std":  ("kl_div_uniform",      "std"),
    }
    for c in cluster_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"]  = (c, "std")

    df_before = raw.groupby(group_cols).agg(**agg_dict).reset_index()
    df_before["loss_function"] = "HDXer"
    df_before["score_metric"]  = "ValDXer"
    df_before["direction"]     = "max"
    df_before["cv"]     = df_before["std"] / df_before["mean"].replace(0, np.nan)
    df_before["cv_std"] = df_before["cv"] / np.sqrt(df_before["count"])

    per_rep = raw.copy()
    per_rep["loss_function"] = "HDXer"
    per_rep["score_metric"]  = "ValDXer"

    return df_before, per_rep, cluster_cols


def _plot_06_selection(df: pd.DataFrame, output_dir: str, style=None) -> None:
    """06_selection_performance_ValDXer.png — ensemble panels × split_type bars.

    Error bars = SD across replicates (pre-computed in df_before std column).
    """
    split_colors: dict = {}
    split_name_mapping: dict = {}
    if style is not None:
        if style.split_type_colors:
            split_colors.update(style.split_type_colors)
        if style.split_name_mapping:
            split_name_mapping.update(style.split_name_mapping)

    ensembles = sorted(df["ensemble"].unique())
    split_types = sorted(df["split_type"].unique())

    fig, axes = plt.subplots(1, len(ensembles), figsize=(5 * len(ensembles), 5), sharey=True)
    if len(ensembles) == 1:
        axes = [axes]

    ens_name_map: dict = {}
    if style is not None and style.ensemble_name_mapping:
        ens_name_map.update(style.ensemble_name_mapping)

    for ax, ens in zip(axes, ensembles):
        ens_data = df[df["ensemble"] == ens]
        for i, split in enumerate(split_types):
            row = ens_data[ens_data["split_type"] == split]
            if row.empty:
                ax.bar(i, 0, color="grey")
                continue
            mean_val = row["mean"].values[0]
            std_val  = row["std"].values[0]
            color    = split_colors.get(split, "steelblue")
            label    = split_name_mapping.get(split, split)
            ax.bar(i, mean_val, yerr=std_val, label=label, color=color,
                   capsize=4, edgecolor="black", alpha=0.9, linewidth=1)

        ax.set_xticks(range(len(split_types)))
        ax.set_xticklabels(
            [split_name_mapping.get(s, s) for s in split_types],
            rotation=20, ha="right",
        )
        ax.set_ylim(0, 100)
        ax.set_title(ens_name_map.get(ens, ens))
        ax.set_ylabel("Recovery Score (%)")
        ax.legend(title="Split Type", loc="upper right")

    fig.suptitle(
        "HDXer Performance → Recovery (Mean ± SD across replicates)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "06_selection_performance_ValDXer.png"),
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot HDXer selected-model summaries for Example 2.")
    p.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing per-replicate HDXer weight NPZ files. "
             "Defaults to the Example 2 HDXer output path.",
    )
    p.add_argument("--output-dir", default=".")
    p.add_argument("--config", default=None)
    a = p.parse_args()

    setup_publication_style()
    sns.set_style("ticks")
    sns.set_context(
        "paper",
        rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 12, "ytick.labelsize": 12},
    )

    exp_dir = Path(__file__).resolve().parent.parent
    cfg = ExperimentConfig.from_yaml(Path(a.config).resolve() if a.config else exp_dir / "config.yaml")
    os.makedirs(a.output_dir, exist_ok=True)

    weights_dir    = Path(a.weights_dir) if a.weights_dir else exp_dir / _DEFAULT_WEIGHTS_DIR
    clustering_dir = exp_dir / cfg.clustering_dir
    state_ratios_path = exp_dir / cfg.state_ratios_json

    state_mapping          = cfg.scoring.state_mapping
    ensemble_clustering_map = cfg.scoring.ensemble_clustering_map
    target_ratios = _load_target_ratios(state_ratios_path, state_mapping)

    print(f"Weights dir:     {weights_dir}")
    print(f"Clustering dir:  {clustering_dir}")
    print(f"State mapping:   {state_mapping}")
    print(f"Target ratios:   {target_ratios}")

    df_before, per_rep, cluster_cols = _load_hdxer_data(
        weights_dir, clustering_dir, ensemble_clustering_map, state_mapping, target_ratios
    )
    df_before_agg = aggregate_df(df_before)

    # HDXer has no filtering step — regret is identically zero
    df_diff = df_before.copy()
    df_diff["mean"] = 0.0
    df_diff["std"]  = 0.0

    df_minimax = compute_minimax_df(df_before)

    for y, yerr, ylabel, title, fname, df in [
        ("mean",   "std",    "Mean Recovery Score (%)",  "Mean Recovery Score (Overall)",      "mean_recovery_score_overall", df_before_agg),
        ("cv",     "cv_std", "Coefficient of Variation", "Coefficient of Variation (Overall)", "cv_overall",                  df_before_agg),
    ]:
        plot_score_panel(df, y, yerr, ylabel, title, a.output_dir, fname, style=cfg.style)

    plot_minimax_panel(df_before, a.output_dir, "minimax_recovery_score", style=cfg.style)

    _plot_06_selection(df_before, a.output_dir, style=cfg.style)

    # Fixed-effects / concordance analysis is skipped: HDXer has a single score_metric
    # and loss_function, so there is no within-group variance to model.

    df_diff_for_rank = df_diff.copy()
    for df, col, title, fname, transform in [
        (df_before,       "mean", "Rank of Mean Recovery",    "rank_mean_recovery", None),
        (df_before,       "cv",   "Rank of CV (-log)",        "rank_cv",            lambda x: -np.log(x + 1e-10)),
        (df_diff_for_rank,"mean", "Rank of Regret (-val)",    "rank_regret",        lambda x: -x),
        (df_minimax,      "mean", "Rank of Minimax Recovery", "rank_minimax",       None),
    ]:
        plot_rank_panel(df, col, title, a.output_dir, fname, False, transform, style=cfg.style)

    save_gt_scores(df_before, df_diff, df_minimax, a.output_dir)

    # Cluster population plots (aggregated)
    pop_cols_agg = [f"{c}_mean" for c in cluster_cols]
    if pop_cols_agg:
        plot_cluster_populations(
            df_before_agg,
            a.output_dir,
            filename="cluster_populations_by_metric",
            title="Cluster Populations — HDXer Selected Models",
            style=cfg.style,
            pop_cols=pop_cols_agg,
        )

    # Per-replicate cluster populations by split (error bars = std across replicates)
    if cluster_cols:
        for metric in sorted(per_rep["score_metric"].dropna().unique()):
            plot_cluster_populations_by_split(
                per_rep,
                metric,
                a.output_dir,
                ensemble_colors=None,
                split_colors=None,
                split_name_mapping=None,
                style=cfg.style,
                pop_cols=cluster_cols,
                loss_filter="HDXer",
            )

    # ── Plot 09: KL divergence — one file per model selection metric ──────────
    for _metric in sorted(df_before["score_metric"].dropna().unique()):
        _safe = "".join(c for c in _metric if c.isalnum() or c in ("_", "-"))
        plot_kl_divergence_panel(
            df_before, a.output_dir, f"09_kl_divergence_{_safe}",
            score_metric=_metric, style=cfg.style,
        )


if __name__ == "__main__":
    main()
