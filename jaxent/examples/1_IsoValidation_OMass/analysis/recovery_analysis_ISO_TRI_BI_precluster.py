"""
recovery_analysis_ISO_TRI_BI_precluster.py

Analyzes the recovery of open/closed states using pre-computed clusters.
Updated Analysis Script - Uses pre-computed clustering results from CSV files.

This script analyzes the %recovery of the ratio of the two conformations used in the IsoValidation process,
including analysis across different maxent regularization values.

Requirements:
    - Optimization results (results_EMA.hdf5)
    - Clustering results (_clustering_results/)
    - Data splits mapping (metadata)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/recovery_analysis_ISO_TRI_BI_precluster.py --results-dir ...
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

EMA = False


# ---------------------------------------------------------------------------
# Script-specific inline plotting functions (no common equivalent)
# ---------------------------------------------------------------------------


def plot_kl_vs_regularization_strength(kl_df, convergence_rates, output_dir):
    """
    Plot KL divergence vs combined regularization strength (maxent + convergence).

    Args:
        kl_df (pd.DataFrame): KL divergence analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    kl_df_copy = kl_df.copy()
    kl_df_copy["convergence_rate"] = kl_df_copy["convergence_threshold_step"].apply(
        lambda x: convergence_rates[int(x) - 1]
        if pd.notna(x) and x > 0 and int(x) - 1 < len(convergence_rates)
        else np.nan
    )

    kl_df_copy["regularization_strength"] = np.nan
    valid_mask = kl_df_copy["convergence_rate"].notna() & (kl_df_copy["maxent_value"] > 0)
    if valid_mask.any():
        conv_normalized = -np.log10(kl_df_copy.loc[valid_mask, "convergence_rate"])
        conv_normalized = (conv_normalized - conv_normalized.min()) / (
            conv_normalized.max() - conv_normalized.min()
        )
        maxent_normalized = kl_df_copy.loc[valid_mask, "maxent_value"]
        maxent_normalized = (maxent_normalized - maxent_normalized.min()) / (
            maxent_normalized.max() - maxent_normalized.min()
        )
        kl_df_copy.loc[valid_mask, "regularization_strength"] = (
            1 - conv_normalized
        ) + maxent_normalized

    split_types = kl_df_copy["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating KL regularization strength plot for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = kl_df_copy[
            (kl_df_copy["split_type"] == split_type)
            & (kl_df_copy["regularization_strength"].notna())
        ]

        if len(split_df) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())

            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s", "^", "D"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        grouped = (
                            subset.groupby("regularization_strength")
                            .agg({"kl_divergence": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = [
                            "regularization_strength",
                            "kl_mean",
                            "kl_std",
                        ]

                        ax.errorbar(
                            grouped["regularization_strength"],
                            grouped["kl_mean"],
                            yerr=grouped["kl_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                            alpha=0.7,
                        )

            ax.set_xlabel(
                "Combined Regularization Strength\n(Higher = More MaxEnt + Looser Convergence)"
            )
            ax.set_ylabel("KL Divergence")
            ax.set_title(f"KL Divergence vs Regularization Strength - {split_type} splits")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "kl_divergence_vs_regularization_strength.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_kl_maxent_comparison(kl_df, output_dir):
    """
    Plot comparison of KL divergence across different maxent values at final convergence.

    Args:
        kl_df (pd.DataFrame): KL divergence analysis results
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    final_data = (
        kl_df.groupby(["split_type", "ensemble", "loss_function", "maxent_value", "split"])
        .last()
        .reset_index()
    )

    split_types = final_data["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating KL maxent comparison for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = final_data[final_data["split_type"] == split_type]

        if len(split_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"KL Divergence MaxEnt Value Comparison - {split_type} splits",
                fontsize=16,
                fontweight="bold",
            )

            ax1 = axes[0]
            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())
            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        grouped = (
                            subset.groupby("maxent_value")
                            .agg({"kl_divergence": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = ["maxent_value", "kl_mean", "kl_std"]

                        ax1.errorbar(
                            grouped["maxent_value"],
                            grouped["kl_mean"],
                            yerr=grouped["kl_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                        )

            ax1.set_xlabel("MaxEnt Value")
            ax1.set_ylabel("KL Divergence")
            ax1.set_title("KL Divergence vs MaxEnt Value (Final Convergence)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            plot_data = []
            for _, row in split_df.iterrows():
                plot_data.append(
                    {
                        "MaxEnt": f"{row['maxent_value']:.0f}",
                        "KL": row["kl_divergence"],
                        "Combination": f"{row['ensemble']}-{row['loss_function']}",
                    }
                )
            plot_df = pd.DataFrame(plot_data)

            sns.boxplot(data=plot_df, x="MaxEnt", y="KL", hue="Combination", ax=ax2)

            ax2.set_xlabel("MaxEnt Value")
            ax2.set_ylabel("KL Divergence")
            ax2.set_title("KL Divergence Distribution by MaxEnt Value")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "kl_divergence_maxent_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_ess_vs_regularization_strength(ess_df, convergence_rates, output_dir):
    """
    Plot ESS vs combined regularization strength (maxent + convergence).

    Args:
        ess_df (pd.DataFrame): ESS analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    ess_df_copy = ess_df.copy()
    ess_df_copy["convergence_rate"] = ess_df_copy["convergence_threshold_step"].apply(
        lambda x: convergence_rates[int(x) - 1]
        if pd.notna(x) and x > 0 and int(x) - 1 < len(convergence_rates)
        else np.nan
    )

    ess_df_copy["regularization_strength"] = np.nan
    valid_mask = ess_df_copy["convergence_rate"].notna() & (ess_df_copy["maxent_value"] > 0)
    if valid_mask.any():
        conv_normalized = -np.log10(ess_df_copy.loc[valid_mask, "convergence_rate"])
        conv_normalized = (conv_normalized - conv_normalized.min()) / (
            conv_normalized.max() - conv_normalized.min()
        )
        maxent_normalized = ess_df_copy.loc[valid_mask, "maxent_value"]
        maxent_normalized = (maxent_normalized - maxent_normalized.min()) / (
            maxent_normalized.max() - maxent_normalized.min()
        )
        ess_df_copy.loc[valid_mask, "regularization_strength"] = (
            1 - conv_normalized
        ) + maxent_normalized

    split_types = ess_df_copy["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating ESS regularization strength plot for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = ess_df_copy[
            (ess_df_copy["split_type"] == split_type)
            & (ess_df_copy["regularization_strength"].notna())
        ]

        if len(split_df) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())

            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s", "^", "D"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        grouped = (
                            subset.groupby("regularization_strength")
                            .agg({"effective_sample_size": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = [
                            "regularization_strength",
                            "ess_mean",
                            "ess_std",
                        ]

                        ax.errorbar(
                            grouped["regularization_strength"],
                            grouped["ess_mean"],
                            yerr=grouped["ess_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                            alpha=0.7,
                        )

            ax.set_xlabel(
                "Combined Regularization Strength\n(Higher = More MaxEnt + Looser Convergence)"
            )
            ax.set_ylabel("Effective Sample Size")
            ax.set_title(f"Effective Sample Size vs Regularization Strength - {split_type} splits")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "ess_vs_regularization_strength.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_ess_maxent_comparison(ess_df, output_dir):
    """
    Plot comparison of ESS across different maxent values at final convergence.

    Args:
        ess_df (pd.DataFrame): ESS analysis results
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    final_data = (
        ess_df.groupby(["split_type", "ensemble", "loss_function", "maxent_value", "split"])
        .last()
        .reset_index()
    )

    split_types = final_data["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating ESS maxent comparison for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = final_data[final_data["split_type"] == split_type]

        if len(split_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Effective Sample Size MaxEnt Value Comparison - {split_type} splits",
                fontsize=16,
                fontweight="bold",
            )

            ax1 = axes[0]
            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())
            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        grouped = (
                            subset.groupby("maxent_value")
                            .agg({"effective_sample_size": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = ["maxent_value", "ess_mean", "ess_std"]

                        ax1.errorbar(
                            grouped["maxent_value"],
                            grouped["ess_mean"],
                            yerr=grouped["ess_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                        )

            ax1.set_xlabel("MaxEnt Value")
            ax1.set_ylabel("Effective Sample Size")
            ax1.set_title("ESS vs MaxEnt Value (Final Convergence)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            plot_data = []
            for _, row in split_df.iterrows():
                plot_data.append(
                    {
                        "MaxEnt": f"{row['maxent_value']:.0f}",
                        "ESS": row["effective_sample_size"],
                        "Combination": f"{row['ensemble']}-{row['loss_function']}",
                    }
                )
            plot_df = pd.DataFrame(plot_data)

            sns.boxplot(data=plot_df, x="MaxEnt", y="ESS", hue="Combination", ax=ax2)

            ax2.set_xlabel("MaxEnt Value")
            ax2.set_ylabel("Effective Sample Size")
            ax2.set_title("ESS Distribution by MaxEnt Value")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "ess_maxent_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_recovery_vs_regularization_strength(recovery_df, convergence_rates, output_dir):
    """
    Plot recovery vs combined regularization strength (maxent + convergence).

    Args:
        recovery_df (pd.DataFrame): Recovery analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Convert convergence_step to numeric
    recovery_df["convergence_step"] = pd.to_numeric(
        recovery_df["convergence_step"], errors="coerce"
    )

    # Calculate combined regularization strength
    # Higher maxent and looser convergence = higher regularization
    recovery_df_copy = recovery_df.copy()
    recovery_df_copy["convergence_rate"] = recovery_df_copy["convergence_step"].apply(
        lambda x: convergence_rates[int(x) - 1]
        if pd.notna(x) and x > 0 and int(x) - 1 < len(convergence_rates)
        else np.nan
    )

    # Create regularization strength metric
    # Normalize both to 0-1 scale and combine
    recovery_df_copy["regularization_strength"] = np.nan
    valid_mask = recovery_df_copy["convergence_rate"].notna() & (
        recovery_df_copy["maxent_value"] > 0
    )
    if valid_mask.any():
        # Log scale for convergence (inverse because smaller = tighter = less regularization)
        conv_normalized = -np.log10(recovery_df_copy.loc[valid_mask, "convergence_rate"])
        conv_normalized = (conv_normalized - conv_normalized.min()) / (
            conv_normalized.max() - conv_normalized.min()
        )

        # Linear scale for maxent (direct because higher = more regularization)
        maxent_normalized = recovery_df_copy.loc[valid_mask, "maxent_value"]
        maxent_normalized = (maxent_normalized - maxent_normalized.min()) / (
            maxent_normalized.max() - maxent_normalized.min()
        )

        # Combine (equal weighting)
        recovery_df_copy.loc[valid_mask, "regularization_strength"] = (
            1 - conv_normalized
        ) + maxent_normalized

    split_types = recovery_df_copy[recovery_df_copy["split_type"] != "N/A"]["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating regularization strength plot for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = recovery_df_copy[
            (recovery_df_copy["split_type"] == split_type)
            & (recovery_df_copy["regularization_strength"].notna())
        ]

        if len(split_df) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())

            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s", "^", "D"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        # Group by regularization strength and calculate mean/std
                        grouped = (
                            subset.groupby("regularization_strength")
                            .agg({"open_recovery": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = [
                            "regularization_strength",
                            "recovery_mean",
                            "recovery_std",
                        ]

                        ax.errorbar(
                            grouped["regularization_strength"],
                            grouped["recovery_mean"],
                            yerr=grouped["recovery_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                            alpha=0.7,
                        )

            ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
            ax.set_xlabel(
                "Combined Regularization Strength\n(Higher = More MaxEnt + Looser Convergence)"
            )
            ax.set_ylabel("Open State Recovery (%)")
            ax.set_title(f"Recovery vs Regularization Strength - {split_type} splits")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "recovery_vs_regularization_strength.png"),
                dpi=300,
                bbox_inches="tight",
            )


def plot_maxent_comparison(recovery_df, output_dir):
    """
    Plot comparison of recovery across different maxent values at final convergence.

    Args:
        recovery_df (pd.DataFrame): Recovery analysis results
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Get final convergence data for each maxent value
    final_data = recovery_df[recovery_df["loss_function"] != "Original"].copy()
    final_data = (
        final_data.groupby(["split_type", "ensemble", "loss_function", "maxent_value", "split"])
        .last()
        .reset_index()
    )

    split_types = final_data["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating maxent comparison for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = final_data[final_data["split_type"] == split_type]

        if len(split_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"MaxEnt Value Comparison - {split_type} splits", fontsize=16, fontweight="bold"
            )

            # Plot 1: Line plot of recovery vs maxent value
            ax1 = axes[0]

            ensembles = sorted(split_df["ensemble"].unique())
            loss_functions = sorted(split_df["loss_function"].unique())

            colors = sns.color_palette("husl", len(ensembles))
            markers = ["o", "s"]

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    subset = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(subset) > 0:
                        grouped = (
                            subset.groupby("maxent_value")
                            .agg({"open_recovery": ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = ["maxent_value", "recovery_mean", "recovery_std"]

                        ax1.errorbar(
                            grouped["maxent_value"],
                            grouped["recovery_mean"],
                            yerr=grouped["recovery_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                        )

            ax1.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
            ax1.set_xlabel("MaxEnt Value")
            ax1.set_ylabel("Open State Recovery (%)")
            ax1.set_title("Recovery vs MaxEnt Value (Final Convergence)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Box plot comparison
            ax2 = axes[1]

            # Prepare data for box plot
            plot_data = []
            for _, row in split_df.iterrows():
                plot_data.append(
                    {
                        "MaxEnt": f"{row['maxent_value']:.0f}",
                        "Recovery": row["open_recovery"],
                        "Combination": f"{row['ensemble']}-{row['loss_function']}",
                    }
                )
            plot_df = pd.DataFrame(plot_data)

            sns.boxplot(data=plot_df, x="MaxEnt", y="Recovery", hue="Combination", ax=ax2)

            ax2.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
            ax2.set_xlabel("MaxEnt Value")
            ax2.set_ylabel("Open State Recovery (%)")
            ax2.set_title("Recovery Distribution by MaxEnt Value")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "maxent_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir):
    """
    Plot volcano plot with KL divergence vs open recovery fold change.

    Args:
        kl_ess_df (pd.DataFrame): KL divergence and ESS analysis results
        recovery_df (pd.DataFrame): Recovery analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Creating volcano plot...")
    print(f"KL/ESS DataFrame shape: {kl_ess_df.shape}")
    print(f"Recovery DataFrame shape: {recovery_df.shape}")

    # Debug: Check what columns are available
    print(f"KL/ESS DataFrame columns: {kl_ess_df.columns.tolist()}")
    print(f"Recovery DataFrame columns: {recovery_df.columns.tolist()}")

    # Filter out Original data from recovery_df for this analysis
    recovery_filtered = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(recovery_filtered) == 0:
        print("No recovery data available for volcano plot")
        return

    if len(kl_ess_df) == 0:
        print("No KL divergence data available for volcano plot")
        return

    # Convert convergence steps to numeric - check which columns exist
    kl_ess_df = kl_ess_df.copy()
    recovery_filtered = recovery_filtered.copy()

    # Handle different possible column names for convergence steps
    kl_conv_col = None
    if "convergence_threshold_step" in kl_ess_df.columns:
        kl_conv_col = "convergence_threshold_step"
    elif "step" in kl_ess_df.columns:
        kl_conv_col = "step"
    elif "convergence_step" in kl_ess_df.columns:
        kl_conv_col = "convergence_step"
    else:
        print(
            f"Warning: No convergence step column found in KL data. Available columns: {kl_ess_df.columns.tolist()}"
        )
        return

    recovery_conv_col = None
    if "convergence_step" in recovery_filtered.columns:
        recovery_conv_col = "convergence_step"
    elif "step" in recovery_filtered.columns:
        recovery_conv_col = "step"
    else:
        print(
            f"Warning: No convergence step column found in recovery data. Available columns: {recovery_filtered.columns.tolist()}"
        )
        return

    kl_ess_df[kl_conv_col] = pd.to_numeric(kl_ess_df[kl_conv_col], errors="coerce")
    recovery_filtered[recovery_conv_col] = pd.to_numeric(
        recovery_filtered[recovery_conv_col], errors="coerce"
    )

    print(f"Using KL convergence column: {kl_conv_col}")
    print(f"Using recovery convergence column: {recovery_conv_col}")

    # Try exact merge first
    merge_columns = ["split_type", "ensemble", "loss_function", "maxent_value", "split"]

    # Create the left and right merge dictionaries
    left_cols = merge_columns + [kl_conv_col]
    right_cols = merge_columns + [recovery_conv_col]

    merged_df = pd.merge(
        kl_ess_df, recovery_filtered, left_on=left_cols, right_on=right_cols, how="inner"
    )

    if len(merged_df) == 0:
        print("No exact matches found, trying to merge by taking final convergence steps...")

        # Get final (last) convergence step for each combination in both dataframes
        kl_final = kl_ess_df.groupby(merge_columns).last().reset_index()
        recovery_final = recovery_filtered.groupby(merge_columns).last().reset_index()

        merged_df = pd.merge(
            kl_final, recovery_final, on=merge_columns, how="inner", suffixes=("_kl", "_recovery")
        )

        # Use recovery convergence step for plotting
        if recovery_conv_col in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]
        elif f"{recovery_conv_col}_recovery" in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[f"{recovery_conv_col}_recovery"]
        else:
            merged_df["plot_convergence_step"] = 1  # Default value

    else:
        merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]

    if len(merged_df) == 0:
        print("No matching data found for volcano plot after all merge attempts")
        print(f"KL data unique combinations: {len(kl_ess_df.groupby(merge_columns))}")
        print(f"Recovery data unique combinations: {len(recovery_filtered.groupby(merge_columns))}")
        return

    print(f"Merged {len(merged_df)} data points for volcano plot")

    # Get unweighted data for baseline calculations
    unweighted_recovery = recovery_df[recovery_df["loss_function"] == "Original"].copy()
    print(f"Found {len(unweighted_recovery)} unweighted baseline records")

    # Create a dictionary of unweighted open ratios per ensemble
    baseline_open_ratios = {}
    for ensemble in unweighted_recovery["ensemble"].unique():
        baseline_data = unweighted_recovery[unweighted_recovery["ensemble"] == ensemble]
        if not baseline_data.empty:
            col = "open_ratio" if "open_ratio" in baseline_data.columns else "open_recovery"
            baseline_open_ratios[ensemble] = baseline_data[col].iloc[0]
            print(f"Baseline for {ensemble}: {baseline_open_ratios[ensemble]:.3f}")
        else:
            baseline_open_ratios[ensemble] = 0.0
            print(f"Warning: No unweighted baseline found for {ensemble}")

    # Calculate fold change relative to unweighted open ratio for each combination
    fold_change_data = []

    for _, row in merged_df.iterrows():
        ensemble = row["ensemble"]
        baseline_open_ratio = baseline_open_ratios.get(ensemble, 0.0)

        if baseline_open_ratio > 0:
            # Use open_ratio if available, otherwise fall back to open_recovery
            if "open_ratio" in row and pd.notna(row["open_ratio"]):
                current_open_ratio = row["open_ratio"]
            else:
                current_open_ratio = row["open_recovery"]

            fold_change = current_open_ratio / baseline_open_ratio
            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
        else:
            fold_change = 1.0  # No change if baseline is 0
            log2_fold_change = 0.0

        fold_change_data.append(
            {
                **row.to_dict(),
                "open_recovery_fold_change": fold_change,
                "log2_fold_change": log2_fold_change,
                "baseline_open_ratio": baseline_open_ratio,
                "current_open_ratio": current_open_ratio
                if baseline_open_ratio > 0
                else row.get("open_recovery", 0),
            }
        )

    if not fold_change_data:
        print("No fold change data could be calculated")
        return

    volcano_df = pd.DataFrame(fold_change_data)
    print(f"Calculated fold changes for {len(volcano_df)} data points")

    # Calculate target fold changes for each ensemble based on target 40:60 ratio
    target_open_ratio = 0.4
    target_closed_ratio = 0.6
    target_fold_changes = {}

    print("Calculating target fold changes based on 40:60 Open:Closed ratio...")

    for ensemble in volcano_df["ensemble"].unique():
        unweighted_open_ratio = baseline_open_ratios.get(ensemble, 0.0)

        if unweighted_open_ratio > 0:
            target_fold_change = target_open_ratio / unweighted_open_ratio
            target_log2_fold_change = np.log2(target_fold_change)
            target_fold_changes[ensemble] = target_log2_fold_change
            print(
                f"  {ensemble}: Unweighted open ratio = {unweighted_open_ratio:.3f}, "
                + f"Target fold change = {target_fold_change:.3f} (log2 = {target_log2_fold_change:.3f})"
            )
        else:
            target_fold_changes[ensemble] = 0
            print(
                f"  {ensemble}: Warning - unweighted open ratio is 0, setting target fold change to 0"
            )

    # Create figure with subplots for each ensemble-loss combination
    split_types = volcano_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating volcano plot for split type: {split_type}")
        split_data = volcano_df[volcano_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Volcano Plot: KL Divergence vs Open Recovery Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        # Create colormap for MaxEnt values (log scale for better visualization)
        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])  # Avoid log(0)
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        # Size mapping for convergence rates (tighter = larger)
        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        print("  Size mapping (tighter convergence = larger points):")

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                # Tighter convergence (smaller rate, higher step number) = larger size
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
                if i < 3 or i == len(conv_steps) - 1:  # Only show first 3 and last
                    rate = convergence_rates[step_int - 1]
                    print(f"    Step {step_int} ({rate:.0e}): size {size:.0f}")
                elif i == 3:
                    print("    ...")
            else:
                size_map[step] = min_size

        # Marker styles for replicates (splits)
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "P", "X"]

        # Calculate global axis limits for consistency across panels
        all_x_data = split_data["log2_fold_change"]
        all_y_data = split_data["kl_divergence"]

        x_margin = (all_x_data.max() - all_x_data.min()) * 0.05  # 5% margin
        y_margin = (all_y_data.max() - all_y_data.min()) * 0.05  # 5% margin

        global_xlim = [all_x_data.min() - x_margin, all_x_data.max() + x_margin]
        global_ylim = [all_y_data.min() - y_margin, all_y_data.max() + y_margin]

        print(f"  Global X-axis range: {global_xlim[0]:.3f} to {global_xlim[1]:.3f}")
        print(f"  Global Y-axis range: {global_ylim[0]:.4f} to {global_ylim[1]:.4f}")

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    # Get unique splits for this combination
                    splits = sorted(combo_data["split"].unique())

                    # Plot each split (replicate) with different marker
                    for k, split_idx in enumerate(splits):
                        split_data_subset = combo_data[combo_data["split"] == split_idx]

                        if len(split_data_subset) > 0:
                            x_sub = split_data_subset["log2_fold_change"]
                            y_sub = split_data_subset["kl_divergence"]

                            # Colors based on log MaxEnt value for each point
                            colors_sub = []
                            for maxent_val in split_data_subset["maxent_value"]:
                                if len(maxent_values) > 1:
                                    colors_sub.append(cmap(norm(np.log10(max(1, maxent_val)))))
                                else:
                                    colors_sub.append("blue")

                            # Sizes based on convergence step
                            sizes_sub = [
                                size_map.get(step, min_size)
                                for step in split_data_subset["plot_convergence_step"]
                            ]

                            scatter = ax.scatter(
                                x_sub,
                                y_sub,
                                c=colors_sub,
                                s=sizes_sub,
                                marker=markers[k % len(markers)],
                                alpha=0.7,
                                edgecolors="black",
                                linewidth=0.5,
                                label=f"Replicate {split_idx}",
                            )

                    # Add reference lines
                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    # Add target fold change line for this ensemble
                    if ensemble in target_fold_changes:
                        target_x = target_fold_changes[ensemble]
                        ax.axvline(
                            x=target_x, color="orange", linestyle=":", alpha=0.8, linewidth=2
                        )

                        # Add text label for target line
                        ax.text(
                            target_x,
                            ax.get_ylim()[1] * 0.95,
                            "Target\n(40:60)",
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    # Set consistent axis limits
                    ax.set_xlim(-6, 6)
                    ax.set_ylim(global_ylim)

                    # Add quadrant labels
                    # Upper right: High KL, Increased Recovery
                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    # Upper left: High KL, Decreased Recovery
                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel("Log2 Fold Change (Open Recovery vs Unweighted Baseline)")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

                    # Add legend for first subplot only
                    if i == 0 and j == 0:
                        legend_elements = []

                        # Add replicate markers
                        for k, split_idx in enumerate(splits[:6]):  # Show max 6 replicates
                            legend_elements.append(
                                plt.Line2D(
                                    [0],
                                    [0],
                                    marker=markers[k % len(markers)],
                                    color="w",
                                    markerfacecolor="gray",
                                    markersize=8,
                                    label=f"Replicate {split_idx}",
                                    markeredgecolor="black",
                                    markeredgewidth=0.5,
                                )
                            )

                        # Add reference lines to legend
                        legend_elements.append(
                            plt.Line2D(
                                [0], [0], color="red", linestyle="--", label="No Fold Change"
                            )
                        )
                        legend_elements.append(
                            plt.Line2D(
                                [0], [0], color="orange", linestyle=":", label="Target (40:60)"
                            )
                        )

                        ax.legend(
                            handles=legend_elements,
                            bbox_to_anchor=(1.05, 1),
                            loc="upper left",
                            fontsize=8,
                        )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{ensemble} - {loss_func}")
                    # Set consistent axis limits even for empty plots
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    # Still add target line for empty plots if available
                    if ensemble in target_fold_changes:
                        target_x = target_fold_changes[ensemble]
                        ax.axvline(
                            x=target_x, color="orange", linestyle=":", alpha=0.8, linewidth=2
                        )
                        ax.text(
                            target_x,
                            ax.get_ylim()[1] * 0.95,
                            "Target\n(40:60)",
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    # Add baseline reference
                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

        plt.tight_layout()

        # Save the plot
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Add target fold change information to the volcano dataframe
    volcano_df["target_log2_fold_change"] = volcano_df["ensemble"].map(target_fold_changes)

    # Save the volcano plot dataset
    volcano_df_path = os.path.join(output_dir, "volcano_plot_data.csv")
    volcano_df.to_csv(volcano_df_path, index=False)
    print(f"Volcano plot dataset saved to: {volcano_df_path}")

    # Also save target fold change information separately
    target_df = pd.DataFrame(
        [
            {"ensemble": ensemble, "target_log2_fold_change": target_fc}
            for ensemble, target_fc in target_fold_changes.items()
        ]
    )
    target_path = os.path.join(output_dir, "target_fold_changes.csv")
    target_df.to_csv(target_path, index=False)
    print(f"Target fold changes saved to: {target_path}")

    # Print summary statistics
    print("\nVolcano Plot Summary:")
    print("-" * 40)
    print(f"Total data points: {len(volcano_df)}")
    print(f"MaxEnt values: {sorted(volcano_df['maxent_value'].unique())}")
    print(
        f"Fold change range: {volcano_df['log2_fold_change'].min():.2f} to {volcano_df['log2_fold_change'].max():.2f}"
    )
    print(
        f"KL divergence range: {volcano_df['kl_divergence'].min():.4f} to {volcano_df['kl_divergence'].max():.4f}"
    )

    print("\nTarget fold changes (40:60 Open:Closed ratio):")
    for ensemble, target_fc in target_fold_changes.items():
        print(f"  {ensemble}: Log2 fold change = {target_fc:.3f}")

    return volcano_df


def plot_volcano_kl_recovery_averaged(kl_ess_df, recovery_df, convergence_rates, output_dir):
    """
    Plot volcano plot with KL divergence vs open recovery fold change, showing averages across replicates with error bars.

    Args:
        kl_ess_df (pd.DataFrame): KL divergence and ESS analysis results
        recovery_df (pd.DataFrame): Recovery analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Creating averaged volcano plot with error bars...")

    # First run the regular volcano plot function to get the processed data
    volcano_df = plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir)

    if volcano_df is None or len(volcano_df) == 0:
        print("No volcano data available for averaged plot")
        return

    # Calculate averages and standard deviations across replicates
    grouping_cols = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
    ]

    averaged_df = (
        volcano_df.groupby(grouping_cols)
        .agg(
            {
                "log2_fold_change": ["mean", "std", "count"],
                "kl_divergence": ["mean", "std", "count"],
                "current_open_ratio": ["mean", "std"],
                "baseline_open_ratio": "first",
                "target_log2_fold_change": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    averaged_df.columns = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
        "log2_fold_change_mean",
        "log2_fold_change_std",
        "log2_fold_change_count",
        "kl_divergence_mean",
        "kl_divergence_std",
        "kl_divergence_count",
        "current_open_ratio_mean",
        "current_open_ratio_std",
        "baseline_open_ratio",
        "target_log2_fold_change",
    ]

    # Fill NaN standard deviations with 0 (for cases with only 1 replicate)
    averaged_df["log2_fold_change_std"] = averaged_df["log2_fold_change_std"].fillna(0)
    averaged_df["kl_divergence_std"] = averaged_df["kl_divergence_std"].fillna(0)
    averaged_df["current_open_ratio_std"] = averaged_df["current_open_ratio_std"].fillna(0)

    print(f"Averaged across replicates: {len(averaged_df)} unique parameter combinations")

    # Create figure with subplots for each ensemble-loss combination
    split_types = averaged_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating averaged volcano plot for split type: {split_type}")
        split_data = averaged_df[averaged_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Averaged Volcano Plot: KL Divergence vs Open Recovery Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        # Create colormap for MaxEnt values
        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        # Calculate global axis limits
        all_x_data = split_data["log2_fold_change_mean"]
        all_y_data = split_data["kl_divergence_mean"]

        x_with_error = np.concatenate(
            [
                all_x_data + split_data["log2_fold_change_std"],
                all_x_data - split_data["log2_fold_change_std"],
            ]
        )
        y_with_error = np.concatenate(
            [
                all_y_data + split_data["kl_divergence_std"],
                all_y_data - split_data["kl_divergence_std"],
            ]
        )

        x_margin = (x_with_error.max() - x_with_error.min()) * 0.05
        y_margin = (y_with_error.max() - y_with_error.min()) * 0.05

        global_xlim = [x_with_error.min() - x_margin, x_with_error.max() + x_margin]
        global_ylim = [y_with_error.min() - y_margin, y_with_error.max() + y_margin]

        # Size mapping for convergence rates
        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
            else:
                size_map[step] = min_size

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    # Plot averages with error bars
                    x_vals = combo_data["log2_fold_change_mean"]
                    y_vals = combo_data["kl_divergence_mean"]
                    x_errs = combo_data["log2_fold_change_std"]
                    y_errs = combo_data["kl_divergence_std"]

                    # Colors based on log MaxEnt value
                    colors_vals = []
                    for maxent_val in combo_data["maxent_value"]:
                        if len(maxent_values) > 1:
                            colors_vals.append(cmap(norm(np.log10(max(1, maxent_val)))))
                        else:
                            colors_vals.append("blue")

                    # Sizes based on convergence step
                    sizes_vals = [
                        size_map.get(step, min_size) for step in combo_data["plot_convergence_step"]
                    ]

                    # Plot points with error bars
                    scatter = ax.scatter(
                        x_vals,
                        y_vals,
                        c=colors_vals,
                        s=sizes_vals,
                        marker="o",
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=3,
                    )

                    # Add error bars
                    ax.errorbar(
                        x_vals,
                        y_vals,
                        xerr=x_errs,
                        yerr=y_errs,
                        fmt="none",
                        ecolor="black",
                        alpha=0.5,
                        capsize=2,
                        capthick=1,
                        zorder=2,
                    )

                    # Add reference lines
                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    # Add target fold change line
                    target_log2_fold_change = combo_data["target_log2_fold_change"].iloc[0]
                    if pd.notna(target_log2_fold_change):
                        ax.axvline(
                            x=target_log2_fold_change,
                            color="orange",
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                        )
                        ax.text(
                            target_log2_fold_change,
                            ax.get_ylim()[1] * 0.95,
                            "Target\n(40:60)",
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    # Set consistent axis limits
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    # Add quadrant labels
                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel("Log2 Fold Change (Open Recovery vs Unweighted Baseline)")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery_averaged.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save the averaged dataset
    averaged_df_path = os.path.join(output_dir, "volcano_plot_averaged_data.csv")
    averaged_df.to_csv(averaged_df_path, index=False)
    print(f"Averaged volcano plot dataset saved to: {averaged_df_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Main function to run the complete analysis using pre-computed clustering results.
    """
    parser = argparse.ArgumentParser(
        description="IsoValidation analysis: set results/output dirs and EMA flag. Paths are interpreted relative to the script unless --absolute-paths is given."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. If omitted, auto-discovered from config results_prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (relative to script dir by default). If omitted, derived from results-dir basename prefixed with '_analysis'.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results (results_EMA.hdf5). Default: False",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided results/output directories as absolute paths",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to ../config.yaml relative to this script.",
    )
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = script_dir.parent

    config_path = Path(args.config) if args.config else script_dir / ".." / "config.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)

    convergence_rates = cfg.convergence_rates

    # Resolve results directory
    if args.results_dir:
        results_dir = (
            Path(args.results_dir)
            if args.absolute_paths
            else (script_dir / args.results_dir).resolve()
        )
    elif cfg.results_prefix:
        prefix_path = Path(cfg.results_prefix)
        base_path = exp_dir / prefix_path.parent
        prefix = prefix_path.name
        found = find_most_recent_dir(base_path, prefix)
        if found is None:
            raise FileNotFoundError(
                f"No directory matching prefix '{prefix}' found in {base_path}"
            )
        results_dir = found
    elif cfg.results_dir:
        results_dir = exp_dir / cfg.results_dir
    else:
        raise ValueError("Provide --results-dir or set results_prefix/results_dir in config.")

    results_dir = str(results_dir)

    # clustering_dir stays as before (relative to script)
    clustering_dir = str(script_dir / ".." / "data" / "_clustering_results")
    ema_flag = args.ema

    # Determine output_dir:
    if args.output_dir:
        if args.absolute_paths:
            output_dir = args.output_dir
        else:
            output_dir = str((script_dir / args.output_dir).resolve())
    else:
        # Derive from results_dir basename, prefix with "_analysis"
        base_name = os.path.basename(os.path.normpath(results_dir))
        out_name = "_analysis" + base_name
        if args.absolute_paths:
            # Place output next to results_dir (same parent)
            parent = os.path.dirname(os.path.normpath(results_dir))
            output_dir = os.path.join(parent, out_name)
        else:
            # Place output inside script directory for relative mode
            output_dir = str(script_dir / out_name)

    # Show resolved paths and EMA flag
    print(f"Resolved results_dir: {results_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved output_dir: {output_dir}")
    print(f"EMA flag: {ema_flag}")

    # Check if required directories exist
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not os.path.exists(clustering_dir):
        raise FileNotFoundError(
            f"Clustering directory not found: {clustering_dir}. "
            "Please run clustering_analysis.py first."
        )

    print("Starting Complete IsoValidation Analysis with MaxEnt Values...")
    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {cfg.ensembles}")
    print(f"Loss functions: {cfg.loss_functions}")
    print(f"Number of splits: {cfg.num_splits}")
    print(f"Convergence rates: {convergence_rates}")
    print("-" * 60)

    # Load clustering results
    print("Loading clustering results...")
    clustering_results = loading.load_clustering_results(clustering_dir)

    # Load all optimization results with maxent
    print("Loading optimization results with maxent values...")
    results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        num_splits=cfg.num_splits,
        EMA=ema_flag,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: KL Divergence Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE AND ESS ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Extract KL divergences and ESS
    print("Extracting frame weights and calculating KL divergences and ESS...")
    kl_ess_df = analysis.extract_frame_weights_kl(results)

    if len(kl_ess_df) > 0:
        print(
            f"Extracted {len(kl_ess_df)} KL divergence and ESS data points from optimization histories"
        )

        # Save the KL divergence and ESS dataset
        kl_ess_df_path = os.path.join(output_dir, "kl_divergence_ess_analysis_maxent_data.csv")
        kl_ess_df.to_csv(kl_ess_df_path, index=False)
        print(f"KL divergence and ESS dataset saved to: {kl_ess_df_path}")

        # Rename step column to convergence_threshold_step for backward compat with inline plots
        if "step" in kl_ess_df.columns and "convergence_threshold_step" not in kl_ess_df.columns:
            kl_ess_df = kl_ess_df.rename(columns={"step": "convergence_threshold_step"})

        # Generate KL divergence plots
        print("Generating KL divergence heatmaps...")
        plotting.plot_metric_heatmap(
            kl_ess_df, "kl_divergence", convergence_rates, output_dir,
            cmap="Blues", columns_col="convergence_threshold_step"
        )

        print("Generating KL regularization strength plots...")
        plot_kl_vs_regularization_strength(kl_ess_df, convergence_rates, output_dir)

        print("Generating KL maxent comparison plots...")
        plot_kl_maxent_comparison(kl_ess_df, output_dir)

        # Generate ESS plots
        print("Generating ESS heatmaps...")
        plotting.plot_metric_heatmap(
            kl_ess_df, "effective_sample_size", convergence_rates, output_dir,
            cmap="viridis", columns_col="convergence_threshold_step"
        )

        print("Generating ESS regularization strength plots...")
        plot_ess_vs_regularization_strength(kl_ess_df, convergence_rates, output_dir)

        print("Generating ESS maxent comparison plots...")
        plot_ess_maxent_comparison(kl_ess_df, output_dir)

    else:
        print("No frame weights data found! Skipping KL divergence and ESS analysis.")
        kl_ess_df = pd.DataFrame()

    # Part 2: Conformational Recovery Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 2: CONFORMATIONAL RECOVERY ANALYSIS WITH MAXENT")
    print("=" * 60)

    if clustering_results:
        print("Analyzing conformational recovery with maxent values...")
        recovery_df = analysis.analyze_conformational_recovery(
            clustering_results,
            results,
            target_ratios=cfg.scoring.ground_truth_ratios,
            state_mapping=cfg.scoring.state_mapping,
            metric="ratio",
            best_step_only=False,
        )

        if len(recovery_df) > 0:
            print(f"Extracted {len(recovery_df)} conformational recovery data points")

            # Generate recovery plots
            print("Generating conformational recovery heatmaps...")
            plotting.plot_metric_heatmap(
                recovery_df, "open_recovery", convergence_rates, output_dir,
                cmap="PiYG", columns_col="convergence_step"
            )

            print("Generating regularization strength plots...")
            plot_recovery_vs_regularization_strength(recovery_df, convergence_rates, output_dir)

            print("Generating maxent comparison plots...")
            plot_maxent_comparison(recovery_df, output_dir)

            # Generate volcano plots if KL data is available
            if len(kl_ess_df) > 0:
                print("Generating volcano plot for KL divergence vs Open Recovery Fold Change...")
                plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir)

                print(
                    "Generating averaged volcano plot for KL divergence vs Open Recovery Fold Change..."
                )
                plot_volcano_kl_recovery_averaged(
                    kl_ess_df, recovery_df, convergence_rates, output_dir
                )
            else:
                print("No KL divergence data available for volcano plots")

            # Save the recovery dataset
            recovery_df_path = os.path.join(output_dir, "conformational_recovery_maxent_data.csv")
            recovery_df.to_csv(recovery_df_path, index=False)
            print(f"Conformational recovery dataset saved to: {recovery_df_path}")

            # Print summary statistics
            print("\nConformational Recovery Summary with MaxEnt:")
            print("-" * 40)

            # Summary by maxent value
            maxent_summary = (
                recovery_df[recovery_df["loss_function"] != "Original"]
                .groupby(["split_type", "ensemble", "loss_function", "maxent_value"])
                .last()
                .reset_index()
            )

            for split_type in maxent_summary["split_type"].unique():
                print(f"\nSplit Type: {split_type}")
                split_summary = maxent_summary[maxent_summary["split_type"] == split_type]

                for _, row in split_summary.iterrows():
                    open_ratio_str = (
                        f", Open Ratio = {row['open_ratio']:.3f}"
                        if "open_ratio" in row.index and pd.notna(row.get("open_ratio"))
                        else ""
                    )
                    print(
                        f"  {row['ensemble']} - {row['loss_function']} - MaxEnt {row['maxent_value']:.0f}: "
                        f"Open Recovery = {row['open_recovery']:.1f}%"
                        f"{open_ratio_str}"
                    )
        else:
            print("No conformational recovery data generated!")
    else:
        print("No clustering results available. Skipping conformational recovery analysis.")

    print("\n" + "=" * 60)
    print("ANALYSIS WITH MAXENT VALUES COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
