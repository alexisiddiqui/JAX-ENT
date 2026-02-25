from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_split_distributions(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Plot peptide distribution across splits as a seaborn countplot."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.countplot(data=all_splits_df, x="peptide_index", hue="split_type")
    plt.title("Peptide Distribution Across All Splits")
    plt.xlabel("Peptide Index")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peptide_split_distribution.png"))
    plt.close()


def plot_enhanced_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Three-panel heatmap: training set, validation set, and combined train/val/gap."""
    os.makedirs(output_dir, exist_ok=True)

    all_peptides = sorted(all_splits_df["peptide_index"].unique())
    all_splits = sorted(all_splits_df["split_dir"].unique())

    train_matrix = np.zeros((len(all_peptides), len(all_splits)))
    val_matrix = np.zeros((len(all_peptides), len(all_splits)))

    for i, peptide in enumerate(all_peptides):
        for j, split_dir in enumerate(all_splits):
            train_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "train")
            )
            if train_mask.any():
                train_matrix[i, j] = 1

            val_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "validation")
            )
            if val_mask.any():
                val_matrix[i, j] = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    im1 = ax1.imshow(train_matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    ax1.set_title("Training Set Distribution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split Directory")
    ax1.set_ylabel("Peptide Index")
    ax1.set_xticks(range(len(all_splits)))
    ax1.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax1.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax1.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax1.grid(True, alpha=0.3)

    im2 = ax2.imshow(val_matrix, cmap="Reds", aspect="auto", interpolation="nearest")
    ax2.set_title("Validation Set Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split Directory")
    ax2.set_ylabel("Peptide Index")
    ax2.set_xticks(range(len(all_splits)))
    ax2.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax2.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax2.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax2.grid(True, alpha=0.3)

    combined_matrix = np.zeros((len(all_peptides), len(all_splits)))
    combined_matrix[train_matrix == 1] = 1
    combined_matrix[val_matrix == 1] = 2

    colors_list = ["white", "#2E86AB", "#A23B72"]
    cmap_combined = ListedColormap(colors_list)

    im3 = ax3.imshow(
        combined_matrix, cmap=cmap_combined, aspect="auto", interpolation="nearest", vmin=0, vmax=2
    )
    ax3.set_title(
        "Combined Train/Val Distribution\n(Blue=Train, Red=Val, White=Gap)",
        fontsize=14, fontweight="bold",
    )
    ax3.set_xlabel("Split Directory")
    ax3.set_ylabel("Peptide Index")
    ax3.set_xticks(range(len(all_splits)))
    ax3.set_xticklabels([s.replace("split_", "") for s in all_splits], rotation=45)
    ax3.set_yticks(range(0, len(all_peptides), max(1, len(all_peptides) // 10)))
    ax3.set_yticklabels(
        [str(all_peptides[i]) for i in range(0, len(all_peptides), max(1, len(all_peptides) // 10))]
    )
    ax3.grid(True, alpha=0.3)

    plt.colorbar(im1, ax=ax1, label="In Training")
    plt.colorbar(im2, ax=ax2, label="In Validation")
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2])
    cbar3.set_ticklabels(["Gap", "Train", "Val"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "enhanced_peptide_split_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Pivot-table heatmap showing which peptides appear in which split."""
    os.makedirs(output_dir, exist_ok=True)
    pivot_table = all_splits_df.pivot_table(
        index="peptide_index", columns="split_dir", values="present", aggfunc="first"
    ).fillna(0)
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, cmap="viridis", cbar=False)
    plt.title("Peptide Presence in Splits")
    plt.xlabel("Split Directory")
    plt.ylabel("Peptide Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peptide_split_heatmap.png"))
    plt.close()
