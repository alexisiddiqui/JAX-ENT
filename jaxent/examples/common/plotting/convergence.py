from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import PlotStyle
from .style import _get_style


def plot_loss_convergence(
    df: pd.DataFrame,
    convergence_rates: List[float],
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot error vs convergence with training and validation error separate.
    Ensembles are shown as different colors, loss functions as different markers.
    """
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize_wide)
    title_suffix = f" - {style.split_name_mapping.get(split_type, split_type)}" if split_type else ""
    fig.suptitle(f"Error vs Convergence Threshold{title_suffix}", fontsize=22, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    # Plot training errors
    ax = ax1
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]
                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"train_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["convergence_rate", "train_mean", "train_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    line = ax.errorbar(
                        stats["convergence_rate"],
                        stats["train_mean"],
                        yerr=stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df.get("convergence_threshold_step", df.get("convergence_step", 0)) > 0)
                ]
                if len(subset) > 0:
                    step_col = "convergence_threshold_step" if "convergence_threshold_step" in subset.columns else "convergence_step"
                    stats = subset.groupby(step_col).agg({"train_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["step", "train_mean", "train_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
                        color = style.ensemble_colors.get(ensemble, "grey")
                        marker = style.loss_markers.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"
                        line = ax.errorbar(
                            stats["convergence_rate"],
                            stats["train_mean"],
                            yerr=stats["train_std"],
                            label=label,
                            marker=marker,
                            color=color,
                            linewidth=2.5,
                            capsize=4,
                            markersize=8,
                            markeredgewidth=1.5,
                            markeredgecolor="black",
                        )
                        legend_handles.append(line)
                        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Training Error", fontweight="bold")
    ax.set_title("Training Error vs Convergence", fontweight="bold")
    
    if legend_handles:
        legend = ax.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 1),
            loc="upper left", frameon=True, fancybox=False, edgecolor="black"
        )
        for text, handle in zip(legend.get_texts(), legend_handles):
            label_text = text.get_text()
            ensemble_name = label_text.split(" - ")[0]
            text.set_color(style.ensemble_colors.get(ensemble_name, "black"))
    
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    # Plot validation errors
    ax = ax2
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]
                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"val_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["convergence_rate", "val_mean", "val_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    line = ax.errorbar(
                        stats["convergence_rate"],
                        stats["val_mean"],
                        yerr=stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df.get("convergence_threshold_step", df.get("convergence_step", 0)) > 0)
                ]
                if len(subset) > 0:
                    step_col = "convergence_threshold_step" if "convergence_threshold_step" in subset.columns else "convergence_step"
                    stats = subset.groupby(step_col).agg({"val_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["step", "val_mean", "val_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
                        color = style.ensemble_colors.get(ensemble, "grey")
                        marker = style.loss_markers.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"
                        line = ax.errorbar(
                            stats["convergence_rate"],
                            stats["val_mean"],
                            yerr=stats["val_std"],
                            label=label,
                            marker=marker,
                            color=color,
                            linewidth=2.5,
                            capsize=4,
                            markersize=8,
                            markeredgewidth=1.5,
                            markeredgecolor="black",
                        )
                        legend_handles.append(line)
                        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Validation Error", fontweight="bold")
    ax.set_title("Validation Error vs Convergence", fontweight="bold")
    
    if legend_handles:
        legend = ax.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 1),
            loc="upper left", frameon=True, fancybox=False, edgecolor="black"
        )
        for text, handle in zip(legend.get_texts(), legend_handles):
            label_text = text.get_text()
            ensemble_name = label_text.split(" - ")[0]
            text.set_color(style.ensemble_colors.get(ensemble_name, "black"))
    
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"error_vs_convergence_{split_type}.png" if split_type else "error_vs_convergence.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=style.dpi, bbox_inches="tight")
    plt.close()


def plot_split_variability(
    df: pd.DataFrame,
    convergence_rates: List[float],
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot standard deviation across splits for each convergence threshold."""
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize_wide)
    title_suffix = f" - {style.split_name_mapping.get(split_type, split_type)}" if split_type else ""
    fig.suptitle(f"Standard Deviation Across Splits{title_suffix}", fontsize=22, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())
    default_marker = "o"

    # Plot training loss std
    ax = ax1
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]
                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"train_loss": "std"}).reset_index()
                    std_stats.columns = ["convergence_rate", "train_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, default_marker)
                    label = f"{ensemble} - {loss_func}"

                    line, = ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df.get("convergence_threshold_step", df.get("convergence_step", 0)) > 0)
                ]
                if len(subset) > 0:
                    step_col = "convergence_threshold_step" if "convergence_threshold_step" in subset.columns else "convergence_step"
                    std_stats = subset.groupby(step_col).agg({"train_loss": "std"}).reset_index()
                    std_stats.columns = ["step", "train_std"]
                    std_stats["convergence_rate"] = std_stats["step"].apply(
                        lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                    )
                    std_stats = std_stats.dropna(subset=["convergence_rate"])

                    if len(std_stats) > 0:
                        color = style.ensemble_colors.get(ensemble, "grey")
                        marker = style.loss_markers.get(loss_func, default_marker)
                        label = f"{ensemble} - {loss_func}"

                        line, = ax.plot(
                            std_stats["convergence_rate"],
                            std_stats["train_std"],
                            label=label,
                            marker=marker,
                            color=color,
                            linewidth=2.5,
                            markersize=8,
                            markeredgewidth=1.5,
                            markeredgecolor="black",
                        )
                        legend_handles.append(line)
                        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Training Error Std Dev", fontweight="bold")
    ax.set_title("Training Error Standard Deviation", fontweight="bold")
    
    if legend_handles:
        legend = ax.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 1),
            loc="upper left", frameon=True, fancybox=False, edgecolor="black"
        )
        for text in legend.get_texts():
            label_text = text.get_text()
            ensemble_name = label_text.split(" - ")[0]
            text.set_color(style.ensemble_colors.get(ensemble_name, "black"))
    
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    # Plot validation loss std
    ax = ax2
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]
                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"val_loss": "std"}).reset_index()
                    std_stats.columns = ["convergence_rate", "val_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, default_marker)
                    label = f"{ensemble} - {loss_func}"

                    line, = ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df.get("convergence_threshold_step", df.get("convergence_step", 0)) > 0)
                ]
                if len(subset) > 0:
                    step_col = "convergence_threshold_step" if "convergence_threshold_step" in subset.columns else "convergence_step"
                    std_stats = subset.groupby(step_col).agg({"val_loss": "std"}).reset_index()
                    std_stats.columns = ["step", "val_std"]
                    std_stats["convergence_rate"] = std_stats["step"].apply(
                        lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                    )
                    std_stats = std_stats.dropna(subset=["convergence_rate"])

                    if len(std_stats) > 0:
                        color = style.ensemble_colors.get(ensemble, "grey")
                        marker = style.loss_markers.get(loss_func, default_marker)
                        label = f"{ensemble} - {loss_func}"

                        line, = ax.plot(
                            std_stats["convergence_rate"],
                            std_stats["val_std"],
                            label=label,
                            marker=marker,
                            color=color,
                            linewidth=2.5,
                            markersize=8,
                            markeredgewidth=1.5,
                            markeredgecolor="black",
                        )
                        legend_handles.append(line)
                        legend_labels.append(label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Validation Error Std Dev", fontweight="bold")
    ax.set_title("Validation Error Standard Deviation", fontweight="bold")
    
    if legend_handles:
        legend = ax.legend(
            legend_handles, legend_labels, bbox_to_anchor=(1.05, 1),
            loc="upper left", frameon=True, fancybox=False, edgecolor="black"
        )
        for text in legend.get_texts():
            label_text = text.get_text()
            ensemble_name = label_text.split(" - ")[0]
            text.set_color(style.ensemble_colors.get(ensemble_name, "black"))
    
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"split_variability_{split_type}.png" if split_type else "split_variability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=style.dpi, bbox_inches="tight")
    plt.close()


def plot_loss_convergence_2d(
    df: pd.DataFrame,
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot error vs hyperparameters with training and validation error separate for 2D sweeps."""
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize_wide)
    title_suffix = f" - {style.split_name_mapping.get(split_type, split_type)}" if split_type else ""
    fig.suptitle(f"Error vs Hyperparameters{title_suffix}", fontsize=22, fontweight="bold")

    split_types = sorted(df["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        split_df = df[df["split_type"] == stype] if stype else df

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        final_df = split_df.loc[
            split_df.groupby(["ensemble", "loss_function", "maxent_value", "bv_reg_value", "split"])[
                "convergence_step"
            ].idxmax()
        ]

        # Plot training errors (ax1)
        ax = ax1
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"train_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["maxent_value", "train_mean", "train_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["maxent_value"],
                        stats["train_mean"],
                        yerr=stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Training Error", fontweight="bold")
        ax.set_title("Training Error vs MaxEnt", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best", frameon=True, fancybox=False, edgecolor="black")
        sns.despine(ax=ax)

        # Plot validation errors (ax2)
        ax = ax2
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"val_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["maxent_value", "val_mean", "val_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["maxent_value"],
                        stats["val_mean"],
                        yerr=stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Validation Error", fontweight="bold")
        ax.set_title("Validation Error vs MaxEnt", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best", frameon=True, fancybox=False, edgecolor="black")
        sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"error_vs_hyperparameters_{split_type}.png" if split_type else "error_vs_hyperparameters.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=style.dpi, bbox_inches="tight")
    plt.close()


def plot_split_variability_2d(
    df: pd.DataFrame,
    output_dir: str,
    style: PlotStyle | None = None,
    split_type: str | None = None,
) -> None:
    """Plot standard deviation across splits for 2D sweeps."""
    style = _get_style(style)
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style.figsize_wide)
    title_suffix = f" - {style.split_name_mapping.get(split_type, split_type)}" if split_type else ""
    fig.suptitle(f"Standard Deviation Across Splits{title_suffix}", fontsize=22, fontweight="bold")

    split_types = sorted(df["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        split_df = df[df["split_type"] == stype] if stype else df

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        final_df = split_df.loc[
            split_df.groupby(["ensemble", "loss_function", "maxent_value", "bv_reg_value", "split"])[
                "convergence_step"
            ].idxmax()
        ]

        # Plot training loss std (ax1)
        ax = ax1
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"train_loss": "std"}).reset_index()
                    std_stats.columns = ["maxent_value", "train_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["maxent_value"],
                        std_stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Training Error Std Dev", fontweight="bold")
        ax.set_title("Training Error Standard Deviation", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        sns.despine(ax=ax)

        # Plot validation loss std (ax2)
        ax = ax2
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"val_loss": "std"}).reset_index()
                    std_stats.columns = ["maxent_value", "val_std"]

                    color = style.ensemble_colors.get(ensemble, "grey")
                    marker = style.loss_markers.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["maxent_value"],
                        std_stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Validation Error Std Dev", fontweight="bold")
        ax.set_title("Validation Error Standard Deviation", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"split_variability_{split_type}.png" if split_type else "split_variability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=style.dpi, bbox_inches="tight")
    plt.close()

