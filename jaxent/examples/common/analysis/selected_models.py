"""Shared selected-model analysis helpers."""

from __future__ import annotations

import os
from itertools import cycle

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import statsmodels.formula.api as smf


SCORE_METRIC_COLORS = {
    "loss": "#c0392b",
    "mse": "#2980b9",
    "d_mse": "#1abc9c",
    "kl": "#8e44ad",
    "work": "#d35400",
    "recovery": "#27ae60",
    "regret": "#f1c40f",
    "spearman": "#16a085",
}
_metric_color_cache = {}
_metric_palette_cycle = cycle(sns.color_palette("tab20"))


def get_metric_color(metric):
    key = str(metric)
    color = SCORE_METRIC_COLORS.get(key) or SCORE_METRIC_COLORS.get(key.lower())
    if color:
        return color
    if key not in _metric_color_cache:
        _metric_color_cache[key] = next(_metric_palette_cycle)
    return _metric_color_cache[key]


def p_to_stars(p_val):
    """Convert p-value to significance stars."""
    if pd.isna(p_val):
        return "n.s."
    elif p_val < 0.0001:
        return "****"
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    elif p_val < 0.1:
        return "‡"
    elif p_val < 0.25:
        return "†"
    else:
        return "n.s."


def ttest_from_stats(m1, s1, n1, m2, s2, n2):
    """Calculate Welch's t-test from summary statistics."""
    vn1 = s1**2 / n1
    vn2 = s2**2 / n2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m2) / np.sqrt(vn1 + vn2)
        df = (vn1 + vn2) ** 2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    return t, p


def calculate_kendalls_w(data_matrix):
    """Return Kendall's W (0..1) for raters x subjects matrix."""
    data_matrix = np.array(data_matrix)
    m, n = data_matrix.shape
    if m <= 1 or n <= 1:
        return 0.0

    ranks = np.apply_along_axis(stats.rankdata, 1, data_matrix)
    sum_ranks = np.sum(ranks, axis=0)
    mean_sum_ranks = m * (n + 1) / 2
    S = np.sum((sum_ranks - mean_sum_ranks) ** 2)

    denom = m**2 * (n**3 - n) / 12.0
    if denom == 0:
        return 0.0
    return S / denom


def get_metric_order(metrics):
    """Sort metrics by loss, mse, kl/work, then other."""
    g1, g2, g3, g4 = [], [], [], []
    for m in metrics:
        m_lower = str(m).lower()
        if "loss" in m_lower:
            g1.append(m)
        elif "mse" in m_lower:
            g2.append(m)
        elif "kl" in m_lower or "work" in m_lower:
            g3.append(m)
        else:
            g4.append(m)
    return sorted(g1) + sorted(g2) + sorted(g3) + sorted(g4)


def load_and_process_data(before_path, after_path, include_transformed=False):
    """Load before and after CSVs and compute difference stats."""
    print(f"Loading Before: {before_path}")
    df_before = pd.read_csv(before_path)
    print(f"Loading After: {after_path}")
    df_after = pd.read_csv(after_path)

    def standardize_df(df):
        target_base = None
        if "recovery_percent_mean" in df.columns:
            target_base = "recovery_percent"
        elif "recovery_score_mean" in df.columns:
            target_base = "recovery_score"
        else:
            for c in df.columns:
                if "recovery" in c and c.endswith("_mean"):
                    target_base = c[:-5]
                    break

        if target_base:
            print(f"Identified target metric: {target_base}")
            rename_map = {
                f"{target_base}_mean": "mean",
                f"{target_base}_std": "std",
                f"{target_base}_count": "count",
                f"{target_base}_min": "min",
            }
            df = df.rename(columns=rename_map)

        spearman_cols = [c for c in df.columns if "spearman" in c.lower()]
        if spearman_cols:
            print(f"Found spearman columns: {spearman_cols}")
            s_mean = next((c for c in spearman_cols if "mean" in c.lower()), None)
            s_std = next((c for c in spearman_cols if "std" in c.lower()), None)
            s_count = next((c for c in spearman_cols if "count" in c.lower()), None)
            if not s_mean:
                candidates = [c for c in spearman_cols if c not in [s_std, s_count] and c is not None]
                if candidates:
                    s_mean = candidates[0]
            rename_map_spearman = {}
            if s_mean:
                rename_map_spearman[s_mean] = "spearman_mean"
            if s_std:
                rename_map_spearman[s_std] = "spearman_std"
            if s_count:
                rename_map_spearman[s_count] = "spearman_count"
            if rename_map_spearman:
                print(f"Renaming spearman columns: {rename_map_spearman}")
                df = df.rename(columns=rename_map_spearman)
        else:
            print("No spearman columns found in dataframe.")

        known_keys = [
            "ensemble",
            "split_type",
            "loss_function",
            "bv_reg_function",
            "bv_reg_value",
            "score_metric",
            "direction",
            "mean",
            "std",
            "count",
            "min",
            "spearman_mean",
            "spearman_std",
            "spearman_count",
        ]
        keep_cols = [c for c in df.columns if c in known_keys]
        return df[keep_cols]

    df_before = standardize_df(df_before)
    df_after = standardize_df(df_after)

    if not include_transformed:
        print("Filtering out transformed metrics...")
        if "score_metric" in df_before.columns:
            df_before = df_before[
                ~df_before["score_metric"].astype(str).str.contains("transformed", case=False, na=False)
            ]
        if "score_metric" in df_after.columns:
            df_after = df_after[
                ~df_after["score_metric"].astype(str).str.contains("transformed", case=False, na=False)
            ]

    df_before["condition"] = "Before Filtering"
    df_after["condition"] = "After Filtering"

    exclude_cols = [
        "mean",
        "std",
        "count",
        "condition",
        "direction",
        "min",
        "spearman_mean",
        "spearman_std",
        "spearman_count",
    ]
    merge_cols = [c for c in df_before.columns if c not in exclude_cols]
    print(f"Merging on: {merge_cols}")

    df_merged = pd.merge(
        df_after,
        df_before,
        on=merge_cols,
        suffixes=("_after", "_before"),
        how="inner",
    )

    df_diff = df_merged.copy()
    df_diff["mean"] = df_diff["mean_before"] - df_diff["mean_after"]
    df_diff["std"] = np.sqrt(df_diff["std_after"] ** 2 + df_diff["std_before"] ** 2)
    df_diff["count"] = np.minimum(df_diff["count_after"], df_diff["count_before"])
    df_diff["condition"] = "Difference (Before - After)"

    cols_to_keep = merge_cols + ["mean", "std", "count", "condition"]
    if "direction_after" in df_diff.columns:
        df_diff["direction"] = df_diff["direction_after"]
        cols_to_keep.append("direction")
    df_diff = df_diff[cols_to_keep]

    df_after["cv"] = df_after["std"] / df_after["mean"]
    df_after["cv_err"] = df_after["cv"] / np.sqrt(2 * df_after["count"])
    df_before["cv"] = df_before["std"] / df_before["mean"]
    df_before["cv_err"] = df_before["cv"] / np.sqrt(2 * df_before["count"])

    return df_before, df_after, df_diff


def aggregate_df(df):
    """Aggregate dataframe over loss_function if present."""
    if df is None or df.empty:
        return df
    if "loss_function" not in df.columns:
        return df.copy()

    exclude_cols = [
        "mean",
        "std",
        "count",
        "loss_function",
        "cv",
        "cv_err",
        "cv_std",
        "spearman_mean",
        "spearman_std",
        "spearman_count",
    ]
    group_cols = [c for c in df.columns if c not in exclude_cols]
    grouped = df.groupby(group_cols, as_index=False)

    agg_cols = ["mean", "std", "count"]
    if "spearman_mean" in df.columns:
        agg_cols.append("spearman_mean")
    if "spearman_std" in df.columns:
        agg_cols.append("spearman_std")
    if "spearman_count" in df.columns:
        agg_cols.append("spearman_count")

    df_agg = grouped[agg_cols].mean()
    if "cv" in df.columns:
        df_agg["cv"] = grouped["cv"].mean()["cv"]
        df_agg["cv_std"] = grouped["cv"].std()["cv"].fillna(0)
    return df_agg


def compute_minimax_df(df):
    """Compute minimax dataframe: min mean recovery across losses."""
    if "loss_function" not in df.columns:
        return df.copy()
    return df.groupby(["ensemble", "split_type", "score_metric"], as_index=False)["mean"].min()


def run_fixed_effects_for_score(df, value_col, score_name, output_dir, transform_func=None, has_loss=True):
    """Run fixed-effects model and save summary CSV."""
    print(f"\n--- Running Fixed Effects Analysis: {score_name} ---")
    local_df = df.copy()

    required_cols = ["score_metric", "ensemble", "split_type"]
    if has_loss:
        required_cols.append("loss_function")
    required_cols.append(value_col)
    local_df = local_df.dropna(subset=required_cols)

    if transform_func:
        local_df["val_transformed"] = local_df[value_col].apply(transform_func)
    else:
        local_df["val_transformed"] = local_df[value_col]

    local_df["percentile"] = local_df.groupby("ensemble")["val_transformed"].rank(pct=True) * 100
    local_df["score_metric"] = local_df["score_metric"].astype(str)
    local_df["ensemble"] = local_df["ensemble"].astype(str)
    local_df["split_type"] = local_df["split_type"].astype(str)
    if has_loss:
        local_df["loss_function"] = local_df["loss_function"].astype(str)

    formula = "percentile ~ C(score_metric) + C(split_type) + C(ensemble)"
    if has_loss:
        formula += " + C(loss_function)"

    try:
        model = smf.ols(formula, data=local_df).fit()
    except Exception as e:
        print(f"Model fitting failed for {score_name}: {e}")
        return None, None

    local_df["resid"] = model.resid
    local_df["fitted"] = model.fittedvalues

    consistency_df = (
        local_df.groupby("score_metric")["resid"]
        .apply(lambda x: np.sqrt((x**2).mean()))
        .reset_index()
    )
    consistency_df.rename(columns={"resid": "Inconsistency"}, inplace=True)

    performance_df = local_df.groupby("score_metric")["fitted"].mean().reset_index()
    performance_df.rename(columns={"fitted": "Performance_Percentile"}, inplace=True)

    results = pd.merge(performance_df, consistency_df, on="score_metric")
    results["Inconsistency_Percentile"] = (
        results["Inconsistency"].apply(lambda x: -x).rank(pct=True) * 100
    )
    results = results.sort_values("Performance_Percentile", ascending=False)

    out_csv = os.path.join(output_dir, f"fixed_effects_{score_name}.csv")
    results.to_csv(out_csv, index=False)
    return results, local_df


def run_fixed_effects_analysis(df_before, df_diff, df_minimax, output_dir):
    """Orchestrate FE analysis for Mean, CV, Regret, Minimax."""
    results_dict = {}

    res_summary, res_full = run_fixed_effects_for_score(
        df_before,
        "mean",
        "Mean_Recovery",
        output_dir,
        transform_func=None,
        has_loss=True,
    )
    if res_summary is not None:
        results_dict["Mean_Recovery"] = (res_summary, res_full)

    res_summary, res_full = run_fixed_effects_for_score(
        df_before,
        "cv",
        "CV",
        output_dir,
        transform_func=lambda x: -np.log(x + 1e-10),
        has_loss=True,
    )
    if res_summary is not None:
        results_dict["CV"] = (res_summary, res_full)

    res_summary, res_full = run_fixed_effects_for_score(
        df_diff,
        "mean",
        "Regret",
        output_dir,
        transform_func=lambda x: -x,
        has_loss=True,
    )
    if res_summary is not None:
        results_dict["Regret"] = (res_summary, res_full)

    res_summary, res_full = run_fixed_effects_for_score(
        df_minimax,
        "mean",
        "Minimax",
        output_dir,
        transform_func=None,
        has_loss=False,
    )
    if res_summary is not None:
        results_dict["Minimax"] = (res_summary, res_full)

    return results_dict


def calculate_concordance_maps(results_dict):
    """Calculate Kendall's W maps for performance/inconsistency/combined."""
    print("\n--- Calculating Concordance Maps (Performance, Inconsistency, Combined) ---")
    gt_scores = list(results_dict.keys())
    if len(gt_scores) < 2:
        return {k: {} for k in ["Performance", "Inconsistency", "Combined"]}

    full_dfs = {k: v[1] for k, v in results_dict.items()}
    all_metrics = set()
    for df in full_dfs.values():
        all_metrics.update(df["score_metric"].unique())

    perf_map, inc_map, comb_map = {}, {}, {}

    for m in all_metrics:
        perf_ranks, inc_ranks, comb_ranks = [], [], []
        valid_metric = True
        common_index = None
        subsets = {}

        for gt in gt_scores:
            df = full_dfs[gt]
            sub = df[df["score_metric"] == m].copy()
            if sub.empty:
                valid_metric = False
                break

            grp = sub.groupby(["ensemble", "split_type"])
            agg_sub = grp.agg({
                "val_transformed": "mean",
                "resid": lambda x: np.sqrt((x**2).mean()),
            }).reset_index()

            agg_sub["id"] = agg_sub["ensemble"].astype(str) + "_" + agg_sub["split_type"].astype(str)
            agg_sub = agg_sub.set_index("id")
            subsets[gt] = agg_sub

            if common_index is None:
                common_index = agg_sub.index
            else:
                common_index = common_index.intersection(agg_sub.index)

        if not valid_metric or common_index is None or len(common_index) < 2:
            perf_map[m] = 0.0
            inc_map[m] = 0.0
            comb_map[m] = 0.0
            continue

        for gt in gt_scores:
            sub = subsets[gt].loc[common_index]
            val = sub["val_transformed"].values
            r_perf = stats.rankdata(-val, method="min")
            perf_ranks.append(r_perf)

            resid = sub["resid"].values
            r_inc = stats.rankdata(resid, method="min")
            inc_ranks.append(r_inc)

            r_comb_score = (r_perf + r_inc) / 2.0
            r_comb = stats.rankdata(r_comb_score, method="min")
            comb_ranks.append(r_comb)

        perf_map[m] = calculate_kendalls_w(np.array(perf_ranks))
        inc_map[m] = calculate_kendalls_w(np.array(inc_ranks))
        comb_map[m] = calculate_kendalls_w(np.array(comb_ranks))

    return {"Performance": perf_map, "Inconsistency": inc_map, "Combined": comb_map}


def save_gt_scores(df_before, df_diff, df_minimax, output_dir):
    """Save GT scores to long-format CSV for cross-experiment analysis."""
    print("\n--- Saving GT Scores for Mixed Effects Model ---")
    data_rows = []

    if "mean" in df_before.columns:
        for _, row in df_before.iterrows():
            data_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "split_type": row["split_type"],
                    "loss_function": row.get("loss_function", "Aggr"),
                    "score_metric": row["score_metric"],
                    "gt_score_type": "Mean_Recovery",
                    "value": row["mean"],
                }
            )

    if "cv" in df_before.columns:
        for _, row in df_before.iterrows():
            data_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "split_type": row["split_type"],
                    "loss_function": row.get("loss_function", "Aggr"),
                    "score_metric": row["score_metric"],
                    "gt_score_type": "CV",
                    "value": row["cv"],
                }
            )

    if df_diff is not None and "mean" in df_diff.columns:
        for _, row in df_diff.iterrows():
            data_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "split_type": row["split_type"],
                    "loss_function": row.get("loss_function", "Aggr"),
                    "score_metric": row["score_metric"],
                    "gt_score_type": "Regret",
                    "value": row["mean"],
                }
            )

    if df_minimax is not None and "mean" in df_minimax.columns:
        for _, row in df_minimax.iterrows():
            data_rows.append(
                {
                    "ensemble": row["ensemble"],
                    "split_type": row["split_type"],
                    "loss_function": "Aggr",
                    "score_metric": row["score_metric"],
                    "gt_score_type": "Minimax",
                    "value": row["mean"],
                }
            )

    out_df = pd.DataFrame(data_rows)
    out_path = os.path.join(output_dir, "gt_scores_long.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


__all__ = [
    "SCORE_METRIC_COLORS",
    "_metric_color_cache",
    "_metric_palette_cycle",
    "get_metric_color",
    "p_to_stars",
    "ttest_from_stats",
    "calculate_kendalls_w",
    "get_metric_order",
    "load_and_process_data",
    "aggregate_df",
    "compute_minimax_df",
    "run_fixed_effects_for_score",
    "run_fixed_effects_analysis",
    "calculate_concordance_maps",
    "save_gt_scores",
]
