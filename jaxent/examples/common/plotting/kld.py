from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_kld_between_splits(
    kld_df, 
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Plot mean KLD between splits across maxent values as line plots."""
    print("Creating KLD between splits plot...")

    if kld_df is None or len(kld_df) == 0:
        print("  No KLD data available for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    kld_df = kld_df.copy()
    if "ensemble_loss" not in kld_df.columns:
        kld_df["ensemble_loss"] = kld_df["ensemble"] + "|" + kld_df["loss_function"]

    available_ensemble_loss = kld_df["ensemble_loss"].unique()
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break
        ax = axes[idx]
        el_data = kld_df[kld_df["ensemble_loss"] == ensemble_loss]

        if el_data.empty:
            ax.set_visible(False)
            continue

        available_split_types = el_data["split_type"].unique()
        n_st = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_st, 2))) # ensure at least 2 colors to avoid collapse
        color_map = dict(zip(available_split_types, colors))

        for split_type in available_split_types:
            st_data = el_data[el_data["split_type"] == split_type].sort_values("maxent_value")
            color = color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)
            ax.errorbar(
                st_data["maxent_value"],
                st_data["mean_kld_between_splits"],
                yerr=st_data["sem_kld_between_splits"],
                color=color,
                alpha=0.8,
                label=label,
                linewidth=2,
                marker="o",
                markersize=4,
                capsize=3,
            )

        ax.set_xscale("log")
        ax.set_xlabel("MaxEnt Value")
        ax.set_ylabel("Mean KLD Between Splits")
        ax.set_title(ensemble_loss.replace("|", "_"))
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Splits Across MaxEnt Values", fontsize=16)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "kld_between_splits_vs_maxent.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: kld_between_splits_vs_maxent.png")
    plt.close()


def plot_sequential_maxent_kld(
    sequential_kld_df, 
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Plot KLD between sequential maxent values."""
    print("Creating sequential maxent KLD plot...")

    if sequential_kld_df is None or len(sequential_kld_df) == 0:
        print("  No sequential KLD data available for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    seq_df = sequential_kld_df.copy()
    if "ensemble_loss" not in seq_df.columns:
        seq_df["ensemble_loss"] = seq_df["ensemble"] + "|" + seq_df["loss_function"]

    available_ensemble_loss = seq_df["ensemble_loss"].unique()
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break
        ax = axes[idx]
        el_data = seq_df[seq_df["ensemble_loss"] == ensemble_loss]

        if el_data.empty:
            ax.set_visible(False)
            continue

        available_split_types = el_data["split_type"].unique()
        n_st = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_st, 2)))
        color_map = dict(zip(available_split_types, colors))

        for split_type in available_split_types:
            st_data = el_data[el_data["split_type"] == split_type]
            color = color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            for split_idx in st_data["split_idx"].unique():
                si_data = st_data[st_data["split_idx"] == split_idx].sort_values("current_maxent")
                if len(si_data) > 0:
                    ax.plot(
                        si_data["current_maxent"],
                        si_data["kld_to_previous"],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                        marker=".",
                        markersize=2,
                    )

            maxent_stats = (
                st_data.groupby("current_maxent")["kld_to_previous"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            if len(maxent_stats) > 0:
                ax.errorbar(
                    maxent_stats["current_maxent"],
                    maxent_stats["mean"],
                    yerr=maxent_stats["std"] / np.sqrt(maxent_stats["count"]),
                    color=color,
                    alpha=0.8,
                    label=label,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    capsize=3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Current MaxEnt")
        ax.set_ylabel("KLD to Previous MaxEnt (or Uniform)")
        ax.set_title(ensemble_loss.replace("|", "_"))
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Sequential MaxEnt Values", fontsize=16)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "sequential_maxent_kld_vs_maxent.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: sequential_maxent_kld_vs_maxent.png")
    plt.close()
