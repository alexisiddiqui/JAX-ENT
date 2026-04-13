"""Shared selected-model plotting helpers."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from ..analysis.selected_models import (
    calculate_kendalls_w,
    get_metric_color,
    get_metric_order,
    p_to_stars,
    ttest_from_stats,
)
from ..config import PlotStyle


_DEFAULT_ENSEMBLE_COLORS = {
    "AF2_MSAss": "RoyalBlue",
    "AF2_filtered": "Cyan",
}
_DEFAULT_SPLIT_COLORS = {
    "Random": "fuchsia",
    "Sequence": "black",
    "Non-Redundant": "green",
    "Spatial": "grey",
    "Flat": "orange",
}
_DEFAULT_SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "random": "Random",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",
    "_flat": "Flat",
}


def _resolve_style_dicts(style: PlotStyle | None):
    ensemble_colors = dict(_DEFAULT_ENSEMBLE_COLORS)
    split_name_mapping = dict(_DEFAULT_SPLIT_NAME_MAPPING)
    split_colors = dict(_DEFAULT_SPLIT_COLORS)

    if style is not None:
        if style.split_name_mapping:
            split_name_mapping.update(style.split_name_mapping)
        if style.ensemble_colors:
            ensemble_colors.update(style.ensemble_colors)
        if style.split_type_colors:
            for k, v in style.split_type_colors.items():
                split_colors[k] = v
                split_colors[split_name_mapping.get(k, k)] = v

    return ensemble_colors, split_colors, split_name_mapping


def plot_minimax_panel(df, output_dir, filename, style: PlotStyle | None = None):
    """Plot minimum mean and abs-min recovery across losses."""
    ensemble_colors, _, split_name_mapping = _resolve_style_dicts(style)
    if "loss_function" not in df.columns:
        print("Cannot plot minimax: 'loss_function' column missing.")
        return

    split_types = sorted(df["split_type"].unique())
    metrics = get_metric_order(df["score_metric"].unique())
    ensembles = sorted(df["ensemble"].unique())

    n_splits = len(split_types)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 6), sharey=False)
    if n_splits == 1:
        axes = [axes]

    width = 0.8 / len(ensembles)
    x = np.arange(len(metrics))

    for i, split in enumerate(split_types):
        ax = axes[i]
        split_data = df[df["split_type"] == split]

        for j, ens in enumerate(ensembles):
            ens_data = split_data[split_data["ensemble"] == ens]

            min_means = []
            abs_mins = []

            for m in metrics:
                rows = ens_data[ens_data["score_metric"] == m]
                if not rows.empty:
                    min_mean = rows["mean"].min()
                    min_means.append(min_mean)
                    if "min" in rows.columns:
                        abs_min = rows["min"].min()
                    else:
                        abs_min = 0
                    abs_mins.append(abs_min)
                else:
                    min_means.append(0)
                    abs_mins.append(0)

            offset = (j - len(ensembles) / 2 + 0.5) * width
            color = ensemble_colors.get(ens, "grey")

            ax.bar(
                x + offset,
                min_means,
                width,
                label=ens if i == 0 else "",
                color=color,
                capsize=4,
                edgecolor="black",
                alpha=0.6,
                linewidth=1,
            )

            if any(a > 0 for a in abs_mins):
                ax.bar(
                    x + offset,
                    abs_mins,
                    width * 0.5,
                    color=color,
                    edgecolor="black",
                    alpha=1.0,
                    linewidth=1,
                    hatch="//",
                )

        ax.set_title(f"{split_name_mapping.get(split, split)}", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_xlabel("Metric", fontweight="bold")
        if i == 0:
            ax.set_ylabel("Min Recovery Score (%)", fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(0, 110)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Ensemble",
    )

    plt.suptitle("Minimax Recovery (Min Mean & Abs Min across Losses)", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_score_panel(
    df,
    y_col,
    y_err_col,
    ylabel,
    title,
    output_dir,
    filename,
    style: PlotStyle | None = None,
    metric_col="score_metric",
):
    """Plot score bars with split types on separate panels."""
    ensemble_colors, _, split_name_mapping = _resolve_style_dicts(style)

    split_types = sorted(df["split_type"].unique())
    metrics = get_metric_order(df[metric_col].unique())
    ensembles = sorted(df["ensemble"].unique())

    n_splits = len(split_types)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 6), sharey=False)
    if n_splits == 1:
        axes = [axes]

    width = 0.8 / len(ensembles)
    x = np.arange(len(metrics))

    for i, split in enumerate(split_types):
        ax = axes[i]
        split_data = df[df["split_type"] == split]

        for j, ens in enumerate(ensembles):
            ens_data = split_data[split_data["ensemble"] == ens]

            means, stds, counts = [], [], []
            for m in metrics:
                row = ens_data[ens_data[metric_col] == m]
                if not row.empty:
                    means.append(row[y_col].values[0])
                    if y_err_col and y_err_col in row.columns:
                        stds.append(row[y_err_col].values[0])
                        counts.append(row["count"].values[0])
                    else:
                        stds.append(0)
                        counts.append(0)
                else:
                    means.append(0)
                    stds.append(0)
                    counts.append(0)

            offset = (j - len(ensembles) / 2 + 0.5) * width
            color = ensemble_colors.get(ens, "grey")

            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds if y_err_col else None,
                label=ens,
                color=color,
                capsize=4,
                edgecolor="black",
                alpha=0.9,
                linewidth=1,
            )

        if len(ensembles) == 2 and y_err_col:
            ens1, ens2 = ensembles
            for k, m in enumerate(metrics):
                d1 = split_data[(split_data["ensemble"] == ens1) & (split_data[metric_col] == m)]
                d2 = split_data[(split_data["ensemble"] == ens2) & (split_data[metric_col] == m)]

                if not d1.empty and not d2.empty:
                    _, p = ttest_from_stats(
                        d1[y_col].values[0],
                        d1[y_err_col].values[0],
                        d1["count"].values[0],
                        d2[y_col].values[0],
                        d2[y_err_col].values[0],
                        d2["count"].values[0],
                    )
                    star = p_to_stars(p)
                    h1 = d1[y_col].values[0] + d1[y_err_col].values[0]
                    h2 = d2[y_col].values[0] + d2[y_err_col].values[0]
                    h = max(h1, h2)
                    if h < 0:
                        h = 0
                    star_fontsize = 16 if star != "n.s." else 10
                    ax.text(
                        k,
                        h + (1 if "Regret" not in title else 1),
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=star_fontsize,
                        fontweight="bold",
                    )

        ax.set_title(f"{split_name_mapping.get(split, split)}", fontweight="bold", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_xlabel("Metric", fontweight="bold")
        if i == 0:
            ax.set_ylabel(ylabel, fontweight="bold")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if "Mean Recovery" in title:
            ax.set_ylim(0, 110)

    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1.0, 1.0), title="Ensemble")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_rank_panel(
    df,
    value_col,
    title,
    output_dir,
    filename,
    ascending,
    transform_func,
    style: PlotStyle | None = None,
):
    """Plot ensemble-normalized ranks aggregated over ensembles/losses."""
    _, split_colors, split_name_mapping = _resolve_style_dicts(style)
    local_df = df.copy()
    local_df["split_type"] = local_df["split_type"].map(lambda x: split_name_mapping.get(x, x))

    if transform_func:
        local_df[value_col] = local_df[value_col].apply(transform_func)

    rank_groups = ["ensemble", "split_type"]
    if "loss_function" in local_df.columns:
        rank_groups.append("loss_function")

    local_df["rank"] = local_df.groupby(rank_groups)[value_col].rank(ascending=ascending, method="min")
    metrics = get_metric_order(local_df["score_metric"].unique())

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=local_df,
        x="score_metric",
        y="rank",
        hue="split_type",
        palette=split_colors,
        order=metrics,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_title(f"{title} (Aggregated)", fontweight="bold")
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Rank (1 is Best)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.invert_yaxis()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Split Type")
    out_path = os.path.join(output_dir, f"{filename}_rank_aggregated.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_fixed_effects(summary_df, score_name, output_dir):
    """Plot scatter and bar fixed-effects outputs for one GT score."""
    metrics = get_metric_order(summary_df["score_metric"].unique())
    metric_palette = {m: get_metric_color(m) for m in metrics}

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=summary_df,
        x="Performance_Percentile",
        y="Inconsistency",
        hue="score_metric",
        s=150,
        edgecolor="black",
        palette=metric_palette,
        alpha=0.8,
    )

    for _, row in summary_df.iterrows():
        plt.text(
            row["Performance_Percentile"],
            row["Inconsistency"] + 0.05,
            row["score_metric"],
            fontsize=9,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Average Performance (Percentile Rank)", fontweight="bold", fontsize=14)
    plt.ylabel("Inconsistency (RMS Residuals)", fontweight="bold", fontsize=14)
    plt.title(
        f"{score_name}: Performance vs. Consistency\n(Fixed Effects on Ensemble-Norm Percentiles)",
        fontweight="bold",
        fontsize=16,
    )
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Metric")
    plt.tight_layout()

    out_plot = os.path.join(output_dir, f"fixed_effects_scatter_{score_name}.png")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"Saved {out_plot}")
    plt.close()

    melted = summary_df.melt(
        id_vars=["score_metric"],
        value_vars=["Performance_Percentile", "Inconsistency_Percentile"],
        var_name="Type",
        value_name="Percentile",
    )
    melted["Type"] = melted["Type"].replace(
        {
            "Performance_Percentile": "Performance",
            "Inconsistency_Percentile": "Inconsistency",
        }
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="score_metric",
        y="Percentile",
        hue="Type",
        order=metrics,
        palette={"Performance": "skyblue", "Inconsistency": "salmon"},
        edgecolor="black",
        linewidth=1,
    )

    plt.title(f"{score_name}: Performance vs Inconsistency (Percentile Ranks)", fontweight="bold")
    plt.xlabel("Metric", fontweight="bold")
    plt.ylabel("Percentile Rank", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_bar = os.path.join(output_dir, f"fixed_effects_bar_{score_name}.png")
    plt.savefig(out_bar, dpi=300, bbox_inches="tight")
    print(f"Saved {out_bar}")
    plt.close()


def plot_aggregated_analysis(results_dict, concordance_maps, output_dir):
    """Plot aggregated FE rank analyses and concordance summaries."""
    print("\n--- Running Aggregated Analysis ---")

    all_rows = []
    for gt_score, (df_summary, _) in results_dict.items():
        df_temp = df_summary.copy()
        df_temp["GT_Score"] = gt_score
        all_rows.append(df_temp)

    if not all_rows:
        print("No results to aggregate.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df["Combined_Percentile"] = (
        full_df["Performance_Percentile"] + full_df["Inconsistency_Percentile"]
    ) / 2.0

    metrics = get_metric_order(full_df["score_metric"].unique())
    metric_palette = {m: get_metric_color(m) for m in metrics}

    def plot_grouped_ranks(y_col, title_prefix, filename_suffix, concordance_key):
        plt.figure(figsize=(14, 7))

        pivot_df = full_df.pivot(index="GT_Score", columns="score_metric", values=y_col)
        pivot_df = pivot_df[metrics]
        if pivot_df.isnull().values.any():
            pivot_df = pivot_df.fillna(0)
        global_w = calculate_kendalls_w(pivot_df.values)

        ax = sns.barplot(
            data=full_df,
            x="score_metric",
            y=y_col,
            hue="GT_Score",
            order=metrics,
            palette="viridis",
            edgecolor="black",
            linewidth=1,
        )

        for i, m in enumerate(metrics):
            w = concordance_maps[concordance_key].get(m, 0.0)
            group_data = full_df[full_df["score_metric"] == m]
            max_h = group_data[y_col].max() if not group_data.empty else 0
            ax.text(
                i,
                max_h + 2,
                f"W={w:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="darkred",
            )

        plt.title(f"{title_prefix} (Global Metric Rank W = {global_w:.3f})", fontweight="bold", fontsize=16)
        plt.xlabel("Metric", fontweight="bold")
        plt.ylabel("Percentile Rank", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 115)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="GT Score")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"aggregated_{filename_suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close()

    plot_grouped_ranks("Performance_Percentile", "Aggregated Performance Ranks", "performance_ranks", "Performance")
    plot_grouped_ranks("Inconsistency_Percentile", "Aggregated Inconsistency Ranks", "inconsistency_ranks", "Inconsistency")
    plot_grouped_ranks("Combined_Percentile", "Aggregated Combined Ranks", "combined_ranks", "Combined")

    corr_rows = []
    for m in metrics:
        m_data = full_df[full_df["score_metric"] == m]
        if len(m_data) > 1:
            perf = m_data["Performance_Percentile"].values
            inc = m_data["Inconsistency_Percentile"].values
            rho, p = stats.spearmanr(perf, inc)
            if np.isnan(rho):
                rho = 0
            corr_rows.append({"score_metric": m, "Spearman_Rho": rho, "p_value": p})

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=corr_df,
            x="score_metric",
            y="Spearman_Rho",
            order=metrics,
            palette=metric_palette,
            edgecolor="black",
            linewidth=1,
        )

        plt.title("Spearman Correlation between Performance and Inconsistency (across GT Scores)", fontweight="bold")
        plt.xlabel("Metric", fontweight="bold")
        plt.ylabel("Spearman's Rho", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.axhline(0, color="black", linewidth=1)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()

        out_corr = os.path.join(output_dir, "aggregated_spearman_correlation.png")
        plt.savefig(out_corr, dpi=300, bbox_inches="tight")
        print(f"Saved {out_corr}")
        plt.close()

    w_rows = []
    for m in metrics:
        w = concordance_maps["Combined"].get(m, 0.0)
        w_rows.append({"score_metric": m, "Kendall_W": w})

    if w_rows:
        w_df = pd.DataFrame(w_rows)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=w_df,
            x="score_metric",
            y="Kendall_W",
            order=metrics,
            palette=metric_palette,
            edgecolor="black",
            linewidth=1,
        )

        plt.title("Concordance of Combined Ranks (Kendall's W across GT Scores)", fontweight="bold", fontsize=16)
        plt.xlabel("Metric", fontweight="bold")
        plt.ylabel("Kendall's W", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.1)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_w = os.path.join(output_dir, "aggregated_combined_concordance_w.png")
        plt.savefig(out_w, dpi=300, bbox_inches="tight")
        print(f"Saved {out_w}")
        plt.close()


def plot_p_values(df, name, output_dir, style: PlotStyle | None = None):
    """Plot -log10(1-p) insignificance between ensembles for one condition."""
    _, split_colors, split_name_mapping = _resolve_style_dicts(style)
    data_rows = []

    split_types = df["split_type"].unique()
    metrics = get_metric_order(df["score_metric"].unique())
    ensembles = sorted(df["ensemble"].unique())

    if len(ensembles) != 2:
        print(f"Skipping p-value plot for {name}: need exactly 2 ensembles, got {len(ensembles)}")
        return

    ens1, ens2 = ensembles
    for split in split_types:
        for m in metrics:
            d1 = df[(df["ensemble"] == ens1) & (df["split_type"] == split) & (df["score_metric"] == m)]
            d2 = df[(df["ensemble"] == ens2) & (df["split_type"] == split) & (df["score_metric"] == m)]
            if not d1.empty and not d2.empty:
                _, p = ttest_from_stats(
                    d1["mean"].values[0],
                    d1["std"].values[0],
                    d1["count"].values[0],
                    d2["mean"].values[0],
                    d2["std"].values[0],
                    d2["count"].values[0],
                )
                p_clipped = min(p, 1.0 - 1e-15)
                val = -np.log10(1 - p_clipped)
                data_rows.append(
                    {
                        "Condition": name,
                        "Split Type": split_name_mapping.get(split, split),
                        "Metric": m,
                        "Insignificance": val,
                        "p-value": p,
                    }
                )

    if not data_rows:
        return

    p_df = pd.DataFrame(data_rows)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=p_df,
        x="Metric",
        y="Insignificance",
        hue="Split Type",
        palette=split_colors,
        edgecolor="black",
        linewidth=1,
        order=metrics,
    )

    thresholds = [0.25, 0.1, 0.05, 0.01, 0.001, 0.0001]
    symbols = ["†", "‡", "*", "**", "***", "****"]
    linewidths = np.linspace(2.5, 0.5, len(thresholds))
    custom_lines = []
    for p_thresh, lw, sym in zip(thresholds, linewidths, symbols):
        y_val = -np.log10(1 - p_thresh)
        ax.axhline(y_val, color="black", linestyle="--", linewidth=lw, alpha=0.6)
        custom_lines.append(
            Line2D([0], [0], color="black", lw=lw, linestyle="--", alpha=0.6, label=f"p<{p_thresh} ({sym})")
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Metric", fontweight="bold")
    ax.set_ylabel("Insignificance (-log10(1-p))", fontweight="bold")
    ax.set_yscale("log")
    ax.set_title(f"Statistical Insignificance of Ensemble Differences ({name})", fontsize=18, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + custom_lines,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Split Type / Significance",
    )

    out_path = os.path.join(output_dir, f"insignificance_comparison_{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


def plot_cluster_populations(
    populations_df: pd.DataFrame,
    output_dir: str,
    filename: str = "cluster_populations",
    title: str = "Cluster Populations for Selected Models",
    style: "PlotStyle | None" = None,
    pop_cols: list[str] | None = None,
    loss_filter: str | None = None,
) -> None:
    """Bar chart of cluster / state populations across replicates.

    Reads a long-form DataFrame with at minimum these columns:
        * ``ensemble``, ``split_type``, ``split``
        * cluster population columns (auto-detected as ending in ``_ratio``
          or ``_current``, or supplied via ``pop_cols``).

    Bars: one group per cluster/state on the x-axis.
    Bar colour: ensemble colour.
    Grouping: split_type → one subplot per split_type.
    Error bars: std across replicates (``split`` column).
    Individual replicate populations are overlaid as scatter jitter.

    Parameters
    ----------
    populations_df:
        DataFrame as described above.  Can be the conformational_recovery_data.csv
        or any frame-weights cluster-ratio table.
    output_dir:
        Directory where the PNG is written.
    filename:
        Output file basename (without extension).
    title:
        Suptitle for the figure.
    style:
        Optional PlotStyle for colours / name mappings.
    pop_cols:
        Explicit list of population column names.  If None, columns ending in
        ``_ratio`` or ``_current`` are used.
    loss_filter:
        If given, only rows where ``loss_function == loss_filter`` are kept.
        If None, populations are averaged over all loss functions and metrics.
    """
    ensemble_colors, _, split_name_mapping = _resolve_style_dicts(style)

    df = populations_df.copy()

    # Optionally filter to a single loss function
    if loss_filter is not None and "loss_function" in df.columns:
        df = df[df["loss_function"] == loss_filter]

    # Exclude unweighted baseline rows (loss == 'Original') for variance plots
    if "loss_function" in df.columns:
        df = df[df["loss_function"] != "Original"]

    if df.empty:
        print(f"plot_cluster_populations: no data after filtering – skipping {filename}")
        return

    # Detect population columns
    if pop_cols is None:
        # Prioritize suffixes: _ratio > _proportion > _current > _mean (pre-aggregated)
        # We want to avoid plotting same thing twice (e.g. state_ratio and state_current)
        suffixes = ["_ratio", "_proportion", "_current"]
        all_matches = [c for c in df.columns if any(c.endswith(s) for s in suffixes)]

        # Also detect pre-aggregated *_mean/*_std pairs (from model selection summary)
        # Only include bare _mean cols that have a corresponding _std col
        # Exclude compound like _rank_mean, _transformed_mean, _percentile_mean
        excluded_infixes = ["_rank_", "_transformed_", "_percentile_"]
        mean_cols = [
            c for c in df.columns
            if c.endswith("_mean")
            and not any(x in c for x in excluded_infixes)
            and c.replace("_mean", "_std") in df.columns
        ]

        # Get unique prefixes from explicit suffixes
        prefixes: set[str] = set()
        for c in all_matches:
            for s in suffixes:
                if c.endswith(s):
                    prefixes.add(c[: -len(s)])

        # Pick the best column for each prefix
        pop_cols = []
        for p in sorted(prefixes):
            if f"{p}_ratio" in df.columns:
                pop_cols.append(f"{p}_ratio")
            elif f"{p}_proportion" in df.columns:
                pop_cols.append(f"{p}_proportion")
            elif f"{p}_current" in df.columns:
                pop_cols.append(f"{p}_current")

        # Fall back to pre-aggregated mean cols if no direct ratio/current/proportion found
        if not pop_cols and mean_cols:
            pop_cols = sorted(mean_cols)

    if not pop_cols:
        print("plot_cluster_populations: no population columns found – skipping")
        return

    # Friendly state labels (strip suffix and replace _ with space)
    state_labels = [
        c.replace("_ratio", "").replace("_current", "").replace("_proportion", "").replace("_", " ")
        for c in pop_cols
    ]

    split_types    = sorted(df["split_type"].unique()) if "split_type" in df.columns else ["all"]
    ensembles      = sorted(df["ensemble"].unique())   if "ensemble"   in df.columns else ["ensemble"]
    score_metrics  = sorted(df["score_metric"].unique()) if "score_metric" in df.columns else [None]

    n_cols     = len(split_types)
    n_rows     = len(score_metrics)
    n_clusters = len(pop_cols)
    n_ens      = len(ensembles)

    cell_w     = max(6, 3 * n_clusters)
    cell_h     = 4
    fig, axes  = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        sharey="row",
        squeeze=False,
    )

    bar_width = 0.75 / max(n_ens, 1)
    x_pos     = np.arange(n_clusters)

    for row_idx, metric in enumerate(score_metrics):
        mdf = df[df["score_metric"] == metric] if metric is not None else df

        for col_idx, split in enumerate(split_types):
            ax  = axes[row_idx][col_idx]
            sdf = mdf[mdf["split_type"] == split] if "split_type" in df.columns else mdf

            for ens_idx, ens in enumerate(ensembles):
                edf = sdf[sdf["ensemble"] == ens] if "ensemble" in df.columns else sdf

                means, stds, all_replicates = [], [], []
                for col in pop_cols:
                    # Support pre-aggregated layout: col is e.g. "cluster_0_mean"
                    std_col = col.replace("_mean", "_std") if col.endswith("_mean") else None
                    if std_col and std_col in edf.columns:
                        # Pre-aggregated: grab the mean/std directly
                        val = edf[col].dropna()
                        std = edf[std_col].dropna()
                        means.append(float(val.mean()) if len(val) else 0.0)
                        stds.append(float(std.mean())  if len(std) else 0.0)
                        all_replicates.append(np.array([]))  # no scatter for pre-agg data
                    else:
                        rep_vals = edf[col].dropna().values if col in edf.columns else np.array([])
                        means.append(float(np.mean(rep_vals)) if len(rep_vals) else 0.0)
                        stds.append(float(np.std(rep_vals))   if len(rep_vals) > 1 else 0.0)
                        all_replicates.append(rep_vals)

                offset = (ens_idx - n_ens / 2 + 0.5) * bar_width
                color  = ensemble_colors.get(ens, f"C{ens_idx}")

                ax.bar(
                    x_pos + offset,
                    means,
                    bar_width,
                    yerr=stds,
                    label=ens,
                    color=color,
                    capsize=4,
                    edgecolor="black",
                    alpha=0.75,
                    linewidth=1,
                    error_kw={"elinewidth": 1.5, "ecolor": "black"},
                )

                # Overlay individual replicate values as scatter
                for ci, rep_vals in enumerate(all_replicates):
                    if len(rep_vals) == 0:
                        continue
                    jitter = np.random.default_rng(seed=42 + ens_idx * 100 + ci).uniform(
                        -bar_width * 0.25, bar_width * 0.25, size=len(rep_vals)
                    )
                    ax.scatter(
                        x_pos[ci] + offset + jitter,
                        rep_vals,
                        color=color,
                        edgecolor="black",
                        s=30,
                        alpha=0.9,
                        linewidth=0.8,
                        zorder=5,
                    )

            # Column header (split type) – only on the first row
            if row_idx == 0:
                split_label = split_name_mapping.get(split, str(split))
                ax.set_title(split_label, fontweight="bold", fontsize=14)

            # Row label (score metric) – only on the leftmost column
            if col_idx == 0:
                row_label = str(metric) if metric is not None else "all metrics"
                ax.set_ylabel(f"{row_label}\n\nPopulation (fraction)", fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Cluster / State", fontweight="bold")
            ax.set_ylim(0, None)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Ensemble",
    )

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()



__all__ = [
    "plot_score_panel",
    "plot_minimax_panel",
    "plot_rank_panel",
    "plot_fixed_effects",
    "plot_aggregated_analysis",
    "plot_p_values",
    "plot_cluster_populations",
]
