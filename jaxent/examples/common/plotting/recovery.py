from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metric_vs_regularization_strength(
    df,
    metric,
    metric_label,
    convergence_rates,
    output_dir,
    filename,
):
    """Plot a metric vs combined regularization strength (maxent + convergence)."""
    plt.style.use("seaborn-v0_8-whitegrid")

    df_copy = df.copy()
    df_copy["convergence_rate"] = df_copy["convergence_threshold_step"].apply(
        lambda x: convergence_rates[int(x) - 1]
        if pd.notna(x) and x > 0 and int(x) - 1 < len(convergence_rates)
        else np.nan
    )

    df_copy["regularization_strength"] = np.nan
    valid_mask = df_copy["convergence_rate"].notna() & (df_copy["maxent_value"] > 0)
    if valid_mask.any():
        conv_normalized = -np.log10(df_copy.loc[valid_mask, "convergence_rate"])
        conv_normalized = (conv_normalized - conv_normalized.min()) / (
            conv_normalized.max() - conv_normalized.min()
        )
        maxent_normalized = df_copy.loc[valid_mask, "maxent_value"]
        maxent_normalized = (maxent_normalized - maxent_normalized.min()) / (
            maxent_normalized.max() - maxent_normalized.min()
        )
        df_copy.loc[valid_mask, "regularization_strength"] = (1 - conv_normalized) + maxent_normalized

    split_types = df_copy["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating {metric_label} regularization strength plot for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df_copy[
            (df_copy["split_type"] == split_type) & (df_copy["regularization_strength"].notna())
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
                            .agg({metric: ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = ["regularization_strength", "metric_mean", "metric_std"]

                        ax.errorbar(
                            grouped["regularization_strength"],
                            grouped["metric_mean"],
                            yerr=grouped["metric_std"],
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
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} vs Regularization Strength - {split_type} splits")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, filename),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_metric_maxent_comparison(df, metric, metric_label, output_dir, filename):
    """Plot comparison of a metric across maxent values at final convergence."""
    plt.style.use("seaborn-v0_8-whitegrid")

    final_data = (
        df.groupby(["split_type", "ensemble", "loss_function", "maxent_value", "split"])
        .last()
        .reset_index()
    )

    split_types = final_data["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating {metric_label} maxent comparison for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = final_data[final_data["split_type"] == split_type]

        if len(split_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"{metric_label} MaxEnt Value Comparison - {split_type} splits",
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
                            .agg({metric: ["mean", "std"]})
                            .reset_index()
                        )
                        grouped.columns = ["maxent_value", "metric_mean", "metric_std"]

                        ax1.errorbar(
                            grouped["maxent_value"],
                            grouped["metric_mean"],
                            yerr=grouped["metric_std"],
                            label=f"{ensemble} - {loss_func}",
                            marker=markers[j % len(markers)],
                            color=colors[i],
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                        )

            ax1.set_xlabel("MaxEnt Value")
            ax1.set_ylabel(metric_label)
            if metric == "effective_sample_size":
                ax1.set_title("ESS vs MaxEnt Value (Final Convergence)")
            elif metric == "kl_divergence":
                ax1.set_title("KL Divergence vs MaxEnt Value (Final Convergence)")
            else:
                ax1.set_title(f"{metric_label} vs MaxEnt Value (Final Convergence)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            plot_data = []
            if metric == "effective_sample_size":
                box_col = "ESS"
                box_title = "ESS Distribution by MaxEnt Value"
            elif metric == "kl_divergence":
                box_col = "KL"
                box_title = "KL Divergence Distribution by MaxEnt Value"
            else:
                box_col = metric_label
                box_title = f"{metric_label} Distribution by MaxEnt Value"

            for _, row in split_df.iterrows():
                plot_data.append(
                    {
                        "MaxEnt": f"{row['maxent_value']:.0f}",
                        box_col: row[metric],
                        "Combination": f"{row['ensemble']}-{row['loss_function']}",
                    }
                )
            plot_df = pd.DataFrame(plot_data)

            sns.boxplot(data=plot_df, x="MaxEnt", y=box_col, hue="Combination", ax=ax2)

            ax2.set_xlabel("MaxEnt Value")
            ax2.set_ylabel(metric_label)
            ax2.set_title(box_title)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, filename),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_recovery_vs_regularization_strength(recovery_df, convergence_rates, output_dir):
    """Plot recovery vs combined regularization strength (maxent + convergence)."""
    plt.style.use("seaborn-v0_8-whitegrid")

    recovery_df["convergence_step"] = pd.to_numeric(
        recovery_df["convergence_step"], errors="coerce"
    )

    recovery_df_copy = recovery_df.copy()
    recovery_df_copy["convergence_rate"] = recovery_df_copy["convergence_step"].apply(
        lambda x: convergence_rates[int(x) - 1]
        if pd.notna(x) and x > 0 and int(x) - 1 < len(convergence_rates)
        else np.nan
    )

    recovery_df_copy["regularization_strength"] = np.nan
    valid_mask = recovery_df_copy["convergence_rate"].notna() & (
        recovery_df_copy["maxent_value"] > 0
    )
    if valid_mask.any():
        conv_normalized = -np.log10(recovery_df_copy.loc[valid_mask, "convergence_rate"])
        conv_normalized = (conv_normalized - conv_normalized.min()) / (
            conv_normalized.max() - conv_normalized.min()
        )

        maxent_normalized = recovery_df_copy.loc[valid_mask, "maxent_value"]
        maxent_normalized = (maxent_normalized - maxent_normalized.min()) / (
            maxent_normalized.max() - maxent_normalized.min()
        )

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
    """Plot comparison of recovery across different maxent values at final convergence."""
    plt.style.use("seaborn-v0_8-whitegrid")

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

            ax2 = axes[1]
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
