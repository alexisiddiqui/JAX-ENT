from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_gap_analysis(all_splits_df: pd.DataFrame, output_dir: str) -> None:
    """Two-panel bar chart: missing peptide counts and coverage % per split."""
    os.makedirs(output_dir, exist_ok=True)

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    bars1 = ax1.bar(gap_df["split"], gap_df["missing_count"], color="lightcoral", alpha=0.7)
    ax1.set_title("Missing Peptides per Split", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Split")
    ax1.set_ylabel("Number of Missing Peptides")
    ax1.tick_params(axis="x", rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    bars2 = ax2.bar(gap_df["split"], gap_df["coverage_pct"], color="lightblue", alpha=0.7)
    ax2.set_title("Peptide Coverage per Split", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Split")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()
