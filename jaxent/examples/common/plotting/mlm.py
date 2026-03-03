from __future__ import annotations

import os
from typing import Iterable

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..config import PlotStyle

_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X", "d"]
_DEFAULT_SPLIT_COLORS = {
    "r": "fuchsia",
    "s": "black",
    "random": "fuchsia",
    "R3": "green",
    "sequence_cluster": "green",
    "Sp": "grey",
    "spatial": "grey",
    "_flat": "orange",
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


def _safe_palette(palette: dict[str, str], categories: Iterable[str]) -> dict[str, str]:
    resolved = dict(palette)
    fallback_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    )
    for i, category in enumerate(categories):
        if category not in resolved:
            resolved[category] = fallback_cycle[i % len(fallback_cycle)]
    return resolved


def _predictor_cols_from_results(regression_results: dict) -> list[str]:
    ensembles = [k for k, v in regression_results.items() if isinstance(v, dict)]
    if not ensembles:
        return []
    first = regression_results[ensembles[0]]
    return [k for k in first.keys() if k != "model"]


def _get_split_colors(style: PlotStyle | None, split_colors: dict[str, str] | None) -> dict[str, str]:
    if split_colors is not None:
        return split_colors
    if style is not None and style.split_type_colors:
        merged = dict(_DEFAULT_SPLIT_COLORS)
        merged.update(style.split_type_colors)
        return merged
    return dict(_DEFAULT_SPLIT_COLORS)


def _get_split_name_mapping(
    style: PlotStyle | None,
    split_name_mapping: dict[str, str] | None,
) -> dict[str, str]:
    if split_name_mapping is not None:
        return split_name_mapping
    if style is not None and style.split_name_mapping:
        merged = dict(_DEFAULT_SPLIT_NAME_MAPPING)
        merged.update(style.split_name_mapping)
        return merged
    return dict(_DEFAULT_SPLIT_NAME_MAPPING)


def _metric_bar_style(metric: str, style: PlotStyle | None) -> dict[str, object]:
    if style is None:
        return {"facecolor": "steelblue", "alpha": 0.7, "hatch": None, "text_color": "black"}

    metric_lower = metric.lower()
    is_work = "work" in metric_lower
    is_mse = "mse" in metric_lower

    if is_work:
        return {
            "facecolor": style.work_facecolor or "steelblue",
            "alpha": style.work_alpha,
            "hatch": style.work_hatch,
            "text_color": style.work_text_color or "black",
        }
    if is_mse:
        return {
            "facecolor": style.mse_facecolor or "steelblue",
            "alpha": style.mse_alpha,
            "hatch": style.mse_hatch,
            "text_color": style.mse_text_color or "black",
        }

    return {"facecolor": "steelblue", "alpha": 0.7, "hatch": None, "text_color": "black"}


def plot_coefficient_comparison(
    regression_results: dict,
    output_dir: str,
    ensemble_colors: dict[str, str],
    style: PlotStyle | None = None,
    predictor_cols: list[str] | None = None,
) -> None:
    """Save 01_coefficient_comparison.png."""
    ensembles = sorted([k for k, v in regression_results.items() if isinstance(v, dict)])
    if not ensembles:
        return

    predictor_cols = predictor_cols or _predictor_cols_from_results(regression_results)
    if not predictor_cols:
        return

    fig, axes = plt.subplots(1, len(ensembles), figsize=(7 * len(ensembles), 6), sharey=True)
    if len(ensembles) == 1:
        axes = [axes]

    for idx, ensemble in enumerate(ensembles):
        ax = axes[idx]
        results = regression_results.get(ensemble, {})

        betas = [results.get(m, {}).get("beta_standardized", 0) for m in predictor_cols]
        ses = [results.get(m, {}).get("se", 0) for m in predictor_cols]
        y_pos = np.arange(len(predictor_cols))

        for i, metric in enumerate(predictor_cols):
            mstyle = _metric_bar_style(metric, style)
            ax.barh(
                y_pos[i],
                betas[i],
                xerr=ses[i],
                color=mstyle["facecolor"],
                alpha=mstyle["alpha"],
                hatch=mstyle["hatch"],
                edgecolor="black",
                linewidth=1.5,
            )

        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(predictor_cols, fontsize=12)
        ax.set_xlabel("Standardized β", fontsize=14)
        ax.set_title(f"{ensemble}", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, (metric, beta, se) in enumerate(zip(predictor_cols, betas, ses)):
            mstyle = _metric_bar_style(metric, style)
            label_x = beta + se + 0.01 if beta > 0 else beta - se - 0.01
            ha = "left" if beta > 0 else "right"
            ax.text(
                label_x,
                i,
                f"{beta:.4f}",
                va="center",
                ha=ha,
                fontsize=10,
                fontweight="bold",
                color=mstyle["text_color"],
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "01_coefficient_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_partial_r2_comparison(
    regression_results: dict,
    output_dir: str,
    ensemble_colors: dict[str, str],
    style: PlotStyle | None = None,
    predictor_cols: list[str] | None = None,
) -> None:
    """Save 02_partial_r2_comparison.png."""
    ensembles = sorted([k for k, v in regression_results.items() if isinstance(v, dict)])
    if not ensembles:
        return

    predictor_cols = predictor_cols or _predictor_cols_from_results(regression_results)
    if not predictor_cols:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ensembles))
    width = 0.8 / max(1, len(predictor_cols))

    for i, metric in enumerate(predictor_cols):
        r2_vals = [
            regression_results.get(ens, {}).get(metric, {}).get("partial_r2", 0)
            for ens in ensembles
        ]
        mstyle = _metric_bar_style(metric, style)
        ax.bar(
            x + i * width,
            r2_vals,
            width,
            label=metric,
            alpha=mstyle["alpha"],
            color=mstyle["facecolor"],
            hatch=mstyle["hatch"],
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_ylabel("Partial R²", fontsize=14)
    ax.set_xlabel("Ensemble", fontsize=14)
    ax.set_title("Partial R²: Metric Predictive Utility", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax.set_xticklabels(ensembles, fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "02_partial_r2_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stability_comparison(
    stability_df: pd.DataFrame,
    output_dir: str,
    ensemble_colors: dict[str, str],
    split_colors: dict[str, str],
    split_name_mapping: dict[str, str],
    suffix: str = "",
    style: PlotStyle | None = None,
) -> None:
    """Save 03_stability_comparison{suffix}.png."""
    if stability_df.empty:
        return

    groups = list(stability_df["group"].dropna().unique())
    predictor_cols = list(stability_df["metric"].dropna().unique())
    if not groups or not predictor_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(predictor_cols))

    ax1 = axes[0]
    for i, metric in enumerate(predictor_cols):
        metric_df = stability_df[stability_df["metric"] == metric]
        vals = [
            metric_df.loc[metric_df["group"] == g, "stability_index"].iloc[0]
            if (metric_df["group"] == g).any()
            else 0
            for g in groups
        ]
        ax1.bar(x + i * width, vals, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1.0)

    ax1.set_ylabel("Stability Index", fontsize=14)
    ax1.set_xlabel("Group", fontsize=14)
    ax1.set_title("Stability Across Groups", fontsize=16, fontweight="bold")
    ax1.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax1.set_xticklabels([split_name_mapping.get(g, g) for g in groups], fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, metric in enumerate(predictor_cols):
        metric_df = stability_df[stability_df["metric"] == metric]
        vals = [
            metric_df.loc[metric_df["group"] == g, "cv_across_splits"].iloc[0]
            if (metric_df["group"] == g).any()
            else 0
            for g in groups
        ]
        vals = [min(v, 1.0) if np.isfinite(v) else 0 for v in vals]
        ax2.bar(x + i * width, vals, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1.0)

    ax2.set_ylabel("Coefficient of Variation", fontsize=14)
    ax2.set_xlabel("Group", fontsize=14)
    ax2.set_title("Metric Variability Across Groups", fontsize=16, fontweight="bold")
    ax2.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax2.set_xticklabels([split_name_mapping.get(g, g) for g in groups], fontsize=12)
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"03_stability_comparison{suffix}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eta_and_ftest(
    stability_df: pd.DataFrame,
    output_dir: str,
    ensemble_colors: dict[str, str],
    split_colors: dict[str, str],
    split_name_mapping: dict[str, str],
    suffix: str = "",
    style: PlotStyle | None = None,
) -> None:
    """Save 04_eta_ftest_comparison{suffix}.png."""
    if stability_df.empty:
        return

    groups = list(stability_df["group"].dropna().unique())
    predictor_cols = list(stability_df["metric"].dropna().unique())
    if not groups or not predictor_cols:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(predictor_cols))

    ax1 = axes[0]
    for i, metric in enumerate(predictor_cols):
        metric_df = stability_df[stability_df["metric"] == metric]
        vals = [
            metric_df.loc[metric_df["group"] == g, "eta_squared"].iloc[0]
            if (metric_df["group"] == g).any()
            else 0
            for g in groups
        ]
        ax1.bar(x + i * width, vals, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1.0)

    ax1.set_ylabel("η² (Effect Size)", fontsize=14)
    ax1.set_xlabel("Group", fontsize=14)
    ax1.set_title("Effect Size", fontsize=16, fontweight="bold")
    ax1.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax1.set_xticklabels([split_name_mapping.get(g, g) for g in groups], fontsize=12)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for i, metric in enumerate(predictor_cols):
        metric_df = stability_df[stability_df["metric"] == metric]
        vals = [
            metric_df.loc[metric_df["group"] == g, "f_statistic"].iloc[0]
            if (metric_df["group"] == g).any()
            else 0
            for g in groups
        ]
        ax2.bar(x + i * width, vals, width, label=metric, alpha=0.8, edgecolor="black", linewidth=1.0)

    ax2.set_ylabel("F-statistic", fontsize=14)
    ax2.set_xlabel("Group", fontsize=14)
    ax2.set_title("ANOVA F-test", fontsize=16, fontweight="bold")
    ax2.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax2.set_xticklabels([split_name_mapping.get(g, g) for g in groups], fontsize=12)
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"04_eta_ftest_comparison{suffix}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_and_distributions(
    df: pd.DataFrame,
    metric: str,
    output_dir: str,
    ensemble_colors: dict[str, str],
    style: PlotStyle | None = None,
    marker_list: Iterable[str] | None = None,
    target_metric: str = "recovery_percent",
) -> None:
    """Save 05_scatter_and_distribution_<metric>.png."""
    if metric not in df.columns:
        return

    markers = list(marker_list) if marker_list is not None else list(_DEFAULT_MARKERS)
    split_colors = _get_split_colors(style, None)
    split_name_mapping = _get_split_name_mapping(style, None)

    ensembles = sorted(df["ensemble"].dropna().unique())
    has_bv_reg = "bv_reg_value" in df.columns
    n_cols = 6 if has_bv_reg else 5

    fig, axes = plt.subplots(len(ensembles), n_cols, figsize=(6 * n_cols, 5 * len(ensembles)))
    if len(ensembles) == 1:
        axes = axes.reshape(1, -1)

    for idx, ensemble in enumerate(ensembles):
        df_ens = df[df["ensemble"] == ensemble].copy()

        ax_dist = axes[idx, 0]
        if "split_type" in df_ens.columns:
            split_types = sorted(df_ens["split_type"].dropna().unique())
            for split_type in split_types:
                df_split = df_ens[df_ens["split_type"] == split_type]
                ax_dist.hist(
                    df_split[metric].dropna(),
                    bins=15,
                    alpha=0.6,
                    label=split_name_mapping.get(split_type, split_type),
                    color=split_colors.get(split_type, "gray"),
                )
        else:
            ax_dist.hist(df_ens[metric].dropna(), bins=20, alpha=0.7, color="steelblue")

        ax_dist.set_xlabel(metric, fontsize=12)
        ax_dist.set_ylabel("Frequency", fontsize=12)
        ax_dist.set_title(f"{ensemble} - Distribution", fontsize=13, fontweight="bold")
        ax_dist.legend(fontsize=10)
        ax_dist.grid(alpha=0.3)

        ax_scatter = axes[idx, 1]
        if "split_type" in df_ens.columns:
            split_types = sorted(df_ens["split_type"].dropna().unique())
            for split_type in split_types:
                df_split = df_ens[df_ens["split_type"] == split_type]
                if "split_idx" in df_split.columns:
                    unique_idxs = sorted(df_split["split_idx"].dropna().unique())
                    for i, s_idx in enumerate(unique_idxs):
                        df_sub = df_split[df_split["split_idx"] == s_idx]
                        valid = df_sub[[metric, target_metric]].dropna()
                        try:
                            m_idx = int(s_idx)
                        except (ValueError, TypeError):
                            m_idx = i
                        marker = markers[m_idx % len(markers)]
                        label = split_name_mapping.get(split_type, split_type) if i == 0 else None
                        ax_scatter.scatter(
                            valid[metric],
                            valid[target_metric],
                            alpha=0.6,
                            s=60,
                            color=split_colors.get(split_type, "gray"),
                            marker=marker,
                            label=label,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                else:
                    valid = df_split[[metric, target_metric]].dropna()
                    ax_scatter.scatter(
                        valid[metric],
                        valid[target_metric],
                        alpha=0.6,
                        s=60,
                        color=split_colors.get(split_type, "gray"),
                        label=split_name_mapping.get(split_type, split_type),
                        edgecolor="black",
                        linewidth=0.5,
                    )
        else:
            valid = df_ens[[metric, target_metric]].dropna()
            ax_scatter.scatter(valid[metric], valid[target_metric], alpha=0.6, s=60, color="steelblue", edgecolor="black", linewidth=0.5)

        valid = df_ens[[metric, target_metric]].dropna()
        if len(valid) > 2:
            X = valid[metric].values.reshape(-1, 1)
            y = valid[target_metric].values
            reg = LinearRegression().fit(X, y)
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = reg.predict(x_line.reshape(-1, 1))
            r2 = r2_score(y, reg.predict(X))
            ax_scatter.plot(x_line, y_line, "k--", linewidth=2.5, alpha=0.7, label=f"Linear fit (R²={r2:.3f})")

        ax_scatter.set_xlabel(metric, fontsize=12)
        ax_scatter.set_ylabel(target_metric, fontsize=12)
        ax_scatter.set_title(f"{ensemble} - Relationship", fontsize=13, fontweight="bold")
        ax_scatter.legend(fontsize=10)
        ax_scatter.grid(alpha=0.3)

        hue_specs = [
            ("maxent_value", axes[idx, 2], plt.cm.viridis, "MaxEnt Value", "Hue: MaxEnt"),
            ("convergence_value", axes[idx, 3], plt.cm.plasma, "Convergence Value", "Hue: Convergence"),
        ]

        ax_loss = axes[idx, 4]
        if "val_loss" in df_ens.columns:
            mask = (df_ens["val_loss"] > 0) & df_ens["val_loss"].notna()
            if mask.any():
                neg_log_loss_col = "_neg_log_val_loss"
                df_ens[neg_log_loss_col] = np.nan
                df_ens.loc[mask, neg_log_loss_col] = -np.log(df_ens.loc[mask, "val_loss"])
                hue_specs.append((neg_log_loss_col, ax_loss, plt.cm.ocean_r, "-log(Val Loss)", "Hue: -log(Val Loss)"))
        ax_loss.set_xlabel(metric, fontsize=12)
        ax_loss.set_ylabel(target_metric, fontsize=12)
        ax_loss.set_title(f"{ensemble} - Hue: -log(Val Loss)", fontsize=13, fontweight="bold")
        ax_loss.grid(alpha=0.3)

        if has_bv_reg:
            hue_specs.append(("bv_reg_value", axes[idx, 5], plt.cm.cividis, "BV Reg Value", "Hue: BV Reg"))

        for hue_col, axis, cmap, cbar_label, title in hue_specs:
            if hue_col not in df_ens.columns:
                continue
            vals = df_ens[hue_col].dropna()
            if vals.empty:
                continue
            if vals.max() > 0 and vals.min() > 0 and (vals.max() / vals.min() > 50):
                norm = matplotlib.colors.LogNorm(vmin=vals.min(), vmax=vals.max())
            else:
                norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())

            if "split_type" in df_ens.columns and "split_idx" in df_ens.columns:
                split_types = sorted(df_ens["split_type"].dropna().unique())
                for split_type in split_types:
                    df_split = df_ens[df_ens["split_type"] == split_type]
                    unique_idxs = sorted(df_split["split_idx"].dropna().unique())
                    for i, s_idx in enumerate(unique_idxs):
                        df_sub = df_split[df_split["split_idx"] == s_idx]
                        valid = df_sub[[metric, target_metric, hue_col]].dropna()
                        if valid.empty:
                            continue
                        try:
                            m_idx = int(s_idx)
                        except (ValueError, TypeError):
                            m_idx = i
                        marker = markers[m_idx % len(markers)]
                        axis.scatter(
                            valid[metric],
                            valid[target_metric],
                            c=valid[hue_col],
                            norm=norm,
                            cmap=cmap,
                            marker=marker,
                            s=60,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )
            else:
                valid = df_ens[[metric, target_metric, hue_col]].dropna()
                axis.scatter(
                    valid[metric],
                    valid[target_metric],
                    c=valid[hue_col],
                    norm=norm,
                    cmap=cmap,
                    s=60,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, label=cbar_label)
            axis.set_xlabel(metric, fontsize=12)
            axis.set_ylabel(target_metric, fontsize=12)
            axis.set_title(f"{ensemble} - {title}", fontsize=13, fontweight="bold")
            axis.grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"05_scatter_and_distribution_{metric}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_selection_performance(
    perf_df: pd.DataFrame,
    metric: str,
    output_dir: str,
    ensemble_colors: dict[str, str],
    split_colors: dict[str, str],
    split_name_mapping: dict[str, str],
    style: PlotStyle | None = None,
    target_metric: str = "recovery_percent",
) -> None:
    """Save 06_selection_performance_<metric>.png."""
    selected_models = perf_df[perf_df["score_metric"] == metric].copy()
    if selected_models.empty:
        return

    if "method_variant" in selected_models.columns:
        x_axis = "method_variant"
    elif "loss_function" in selected_models.columns:
        x_axis = "loss_function"
    elif "bv_reg_function" in selected_models.columns:
        x_axis = "bv_reg_function"
    else:
        selected_models["method_variant"] = "All"
        x_axis = "method_variant"

    palette = _get_split_colors(style, split_colors)
    g = sns.catplot(
        data=selected_models,
        x=x_axis,
        y=target_metric,
        hue="split_type",
        col="ensemble",
        kind="bar",
        height=5,
        aspect=1,
        sharey=True,
        palette=palette,
        errorbar="sd",
        capsize=0.1,
        edgecolor="black",
        linewidth=1.0,
    )

    direction = selected_models["direction"].iloc[0] if "direction" in selected_models.columns else "max"
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"Selection by {metric} ({direction}) -> {target_metric}", fontsize=16, fontweight="bold")
    g.set_axis_labels("Method Variant", target_metric)
    g.set_titles("{col_name}")

    safe_metric = "".join(c for c in metric if c.isalnum() or c in ("_", "-"))
    output_path = os.path.join(output_dir, f"06_selection_performance_{safe_metric}.png")
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def plot_correlations_bar_charts(
    corr_df: pd.DataFrame,
    split_type: str,
    output_dir: str,
    ensemble_colors: dict[str, str],
    style: PlotStyle | None = None,
) -> None:
    """Save 07_correlation_bars_<split_type>.png."""
    if corr_df.empty:
        return

    split_name_mapping = _get_split_name_mapping(style, None)
    df_subset = corr_df[corr_df["split_type"] == split_type]
    if df_subset.empty:
        return

    plt.figure(figsize=(max(10, len(df_subset["metric"].unique()) * 1.5), 6))
    hue_levels = list(df_subset["ensemble"].dropna().unique())
    palette = _safe_palette(ensemble_colors, hue_levels)
    sns.barplot(
        data=df_subset,
        x="metric",
        y="correlation",
        hue="ensemble",
        palette=palette,
        edgecolor="black",
    )
    plt.title(
        f"Correlation with Target (Split Type: {split_name_mapping.get(split_type, split_type)})",
        fontsize=16,
    )
    plt.xlabel("Score Metric", fontsize=14)
    plt.ylabel("Pearson Correlation", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Ensemble", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    safe_name = "".join(c for c in str(split_type) if c.isalnum() or c in ("_", "-"))
    output_path = os.path.join(output_dir, f"07_correlation_bars_{safe_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


__all__ = [
    "plot_coefficient_comparison",
    "plot_partial_r2_comparison",
    "plot_stability_comparison",
    "plot_eta_and_ftest",
    "plot_scatter_and_distributions",
    "plot_model_selection_performance",
    "plot_correlations_bar_charts",
]
