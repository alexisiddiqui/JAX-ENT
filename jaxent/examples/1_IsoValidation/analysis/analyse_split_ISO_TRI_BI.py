"""
This script analyses the splits created by the splitdata_ISO.py script for the ISO TRI BI dataset.
It loads the split data for each replicate and plots the distribution of the number of the splits across the replicates.
Plotting both the average distribution as well as each replicate individually.
This incudes heatmaps to show the splits themselves with clear gap visualization
As well as heatmaps that show the uptake curves for each split with separate colors for train/val.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

from jaxent.src.interfaces.topology import Partial_Topology


def load_split_data(
    split_dir: str,
) -> tuple[list[Partial_Topology], np.ndarray, list[Partial_Topology], np.ndarray]:
    """Load data for a single split."""
    train_top_file = os.path.join(split_dir, "train_topology.json")
    train_dfrac_file = os.path.join(split_dir, "train_dfrac.csv")
    val_top_file = os.path.join(split_dir, "val_topology.json")
    val_dfrac_file = os.path.join(split_dir, "val_dfrac.csv")

    train_top = Partial_Topology.load_list_from_json(train_top_file)
    train_dfrac = pd.read_csv(train_dfrac_file, header=None).to_numpy()
    val_top = Partial_Topology.load_list_from_json(val_top_file)
    val_dfrac = pd.read_csv(val_dfrac_file, header=None).to_numpy()

    return train_top, train_dfrac, val_top, val_dfrac


def plot_split_distributions(all_splits_df: pd.DataFrame, output_dir: str):
    """Plot the distribution of peptides in splits."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=all_splits_df, x="peptide_index", hue="split_type")
    plt.title("Peptide Distribution Across All Splits")
    plt.xlabel("Peptide Index")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "peptide_split_distribution.png"))
    plt.close()


def plot_enhanced_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str):
    """Plot enhanced heatmap showing train/val splits with clear gap visualization."""
    # Get all unique peptide indices and split directories
    all_peptides = sorted(all_splits_df["peptide_index"].unique())
    all_splits = sorted(all_splits_df["split_dir"].unique())

    # Create separate matrices for train and validation
    train_matrix = np.zeros((len(all_peptides), len(all_splits)))
    val_matrix = np.zeros((len(all_peptides), len(all_splits)))

    # Fill matrices
    for i, peptide in enumerate(all_peptides):
        for j, split_dir in enumerate(all_splits):
            # Check if peptide is in training set for this split
            train_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "train")
            )
            if train_mask.any():
                train_matrix[i, j] = 1

            # Check if peptide is in validation set for this split
            val_mask = (
                (all_splits_df["peptide_index"] == peptide)
                & (all_splits_df["split_dir"] == split_dir)
                & (all_splits_df["split_type"] == "validation")
            )
            if val_mask.any():
                val_matrix[i, j] = 1

    # Create combined visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    # Training set heatmap
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

    # Validation set heatmap
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

    # Combined heatmap with different colors for train/val/gaps
    combined_matrix = np.zeros((len(all_peptides), len(all_splits)))
    combined_matrix[train_matrix == 1] = 1  # Training
    combined_matrix[val_matrix == 1] = 2  # Validation
    # Gaps remain 0

    # Custom colormap: white (gaps), blue (train), red (val)
    colors = ["white", "#2E86AB", "#A23B72"]  # white, blue, red
    n_bins = 3
    cmap = ListedColormap(colors)

    im3 = ax3.imshow(
        combined_matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=2
    )
    ax3.set_title(
        "Combined Train/Val Distribution\n(Blue=Train, Red=Val, White=Gap)",
        fontsize=14,
        fontweight="bold",
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

    # Add colorbars
    plt.colorbar(im1, ax=ax1, label="In Training")
    plt.colorbar(im2, ax=ax2, label="In Validation")

    # Custom colorbar for combined plot
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2])
    cbar3.set_ticklabels(["Gap", "Train", "Val"])

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "enhanced_peptide_split_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_split_heatmap(all_splits_df: pd.DataFrame, output_dir: str):
    """Plot a heatmap showing which peptides are in which split (original version)."""
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


def plot_uptake_heatmap(
    peptides: list[Partial_Topology],
    dfracs: np.ndarray,
    title: str,
    filename: str,
    split_type: str = "train",
):
    """Plot a heatmap of uptake curves with color scheme based on split type."""
    peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in peptides]

    # Choose colormap based on split type
    if split_type == "train":
        colormap = "Blues"
    elif split_type == "validation":
        colormap = "Reds"
    else:
        colormap = "viridis"

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        dfracs,
        xticklabels=False,
        yticklabels=peptide_names,
        cmap=colormap,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Peptide")
    plt.xlabel("Timepoint")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_uptake_comparison(
    train_peptides: list[Partial_Topology],
    train_dfracs: np.ndarray,
    val_peptides: list[Partial_Topology],
    val_dfracs: np.ndarray,
    split_name: str,
    output_dir: str,
):
    """Plot side-by-side comparison of training and validation uptake curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Training heatmap
    train_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in train_peptides]
    sns.heatmap(
        train_dfracs,
        xticklabels=False,
        yticklabels=train_peptide_names,
        cmap="Blues",
        ax=ax1,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax1.set_title(f"Training Set Uptake - {split_name}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Peptide")
    ax1.set_xlabel("Timepoint")

    # Validation heatmap
    val_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in val_peptides]
    sns.heatmap(
        val_dfracs,
        xticklabels=False,
        yticklabels=val_peptide_names,
        cmap="Reds",
        ax=ax2,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax2.set_title(f"Validation Set Uptake - {split_name}", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Peptide")
    ax2.set_xlabel("Timepoint")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"combined_uptake_{split_name}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_gap_analysis(all_splits_df: pd.DataFrame, output_dir: str):
    """Plot detailed gap analysis showing missing peptides per split."""
    # Calculate gaps for each split
    all_peptides = set(all_splits_df["peptide_index"].unique())
    splits = all_splits_df["split_dir"].unique()

    gap_data = []
    for split_dir in splits:
        split_data = all_splits_df[all_splits_df["split_dir"] == split_dir]
        present_peptides = set(split_data["peptide_index"].unique())
        missing_peptides = all_peptides - present_peptides

        gap_data.append(
            {
                "split": split_dir.replace("split_", ""),
                "missing_count": len(missing_peptides),
                "present_count": len(present_peptides),
                "total_possible": len(all_peptides),
                "coverage_pct": (len(present_peptides) / len(all_peptides)) * 100,
            }
        )

    gap_df = pd.DataFrame(gap_data)

    # Plot gap analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Missing peptides count
    bars1 = ax1.bar(gap_df["split"], gap_df["missing_count"], color="lightcoral", alpha=0.7)
    ax1.set_title("Missing Peptides per Split", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("Number of Missing Peptides")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom"
        )

    # Coverage percentage
    bars2 = ax2.bar(gap_df["split"], gap_df["coverage_pct"], color="lightblue", alpha=0.7)
    ax2.set_title("Peptide Coverage per Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function to run the analysis."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "fitting", "jaxENT", "_datasplits")
    output_dir = os.path.join(os.path.dirname(__file__), "_analysis_split_iso_tri_bi")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(base_dir):
        print(f"Data directory not found: {base_dir}")
        print("Please run the `splitdata_ISO.py` script first.")
        return

    split_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("split_")
    ]

    all_peptides_data = []

    for split_dir in split_dirs:
        train_top, train_dfrac, val_top, val_dfrac = load_split_data(split_dir)

        split_name = os.path.basename(split_dir)

        # Individual uptake heatmaps with color coding
        plot_uptake_heatmap(
            train_top,
            train_dfrac,
            f"Training Set Uptake - {split_name}",
            os.path.join(output_dir, f"uptake_train_{split_name}.png"),
            split_type="train",
        )
        plot_uptake_heatmap(
            val_top,
            val_dfrac,
            f"Validation Set Uptake - {split_name}",
            os.path.join(output_dir, f"uptake_val_{split_name}.png"),
            split_type="validation",
        )

        # Combined uptake comparison
        plot_combined_uptake_comparison(
            train_top, train_dfrac, val_top, val_dfrac, split_name, output_dir
        )

        # Collect data for overall analysis
        for p in train_top:
            all_peptides_data.append(
                {
                    "peptide_index": p.fragment_index,
                    "split_dir": split_name,
                    "split_type": "train",
                    "present": 1,
                }
            )
        for p in val_top:
            all_peptides_data.append(
                {
                    "peptide_index": p.fragment_index,
                    "split_dir": split_name,
                    "split_type": "validation",
                    "present": 1,
                }
            )

    if not all_peptides_data:
        print("No data found in split directories.")
        return

    all_splits_df = pd.DataFrame(all_peptides_data)

    # Generate all plots
    plot_split_distributions(all_splits_df, output_dir)
    plot_split_heatmap(all_splits_df, output_dir)  # Original version
    plot_enhanced_split_heatmap(all_splits_df, output_dir)  # Enhanced version
    plot_gap_analysis(all_splits_df, output_dir)

    print(f"Analysis complete. Plots saved to {output_dir}")
    print("Enhanced visualizations include:")
    print("- Enhanced split heatmap with train/val color coding")
    print("- Combined uptake comparisons")
    print("- Gap analysis plots")


if __name__ == "__main__":
    main()
