from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_uptake_heatmap(
    peptides: list,
    dfracs: np.ndarray,
    title: str,
    filename: str,
    split_type: str = "train",
) -> None:
    """Heatmap of deuterium uptake curves with colormap based on split type.

    Parameters
    ----------
    peptides:
        List of topology objects with ``residue_start`` and ``residue_end``
        attributes (e.g. ``Partial_Topology``).
    dfracs:
        Uptake array of shape ``(n_peptides, n_timepoints)`` or
        ``(n_peptides, n_timepoints, 1)``; leading singleton axes are squeezed.
    title:
        Figure title.
    filename:
        Full output path (including directory and ``.png`` extension).
    split_type:
        ``"train"`` → Blues colormap, ``"validation"`` → Reds, else viridis.
    """
    peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in peptides]
    colormap = {"train": "Blues", "validation": "Reds"}.get(split_type, "viridis")

    # Squeeze trailing singleton dimensions so heatmap receives a 2-D array.
    plot_dfracs = np.squeeze(dfracs)
    if plot_dfracs.ndim == 1:
        plot_dfracs = plot_dfracs.reshape(-1, 1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        plot_dfracs,
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
    train_peptides: list,
    train_dfracs: np.ndarray,
    val_peptides: list,
    val_dfracs: np.ndarray,
    split_name: str,
    output_dir: str,
) -> None:
    """Side-by-side heatmaps comparing training and validation uptake curves."""
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    train_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in train_peptides]
    sns.heatmap(
        np.squeeze(train_dfracs),
        xticklabels=False,
        yticklabels=train_peptide_names,
        cmap="Blues",
        ax=ax1,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax1.set_title(f"Training Set Uptake — {split_name}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Peptide")
    ax1.set_xlabel("Timepoint")

    val_peptide_names = [f"{p.residue_start}-{p.residue_end}" for p in val_peptides]
    sns.heatmap(
        np.squeeze(val_dfracs),
        xticklabels=False,
        yticklabels=val_peptide_names,
        cmap="Reds",
        ax=ax2,
        cbar_kws={"label": "Deuterium Fraction"},
    )
    ax2.set_title(f"Validation Set Uptake — {split_name}", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Peptide")
    ax2.set_xlabel("Timepoint")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"combined_uptake_{split_name}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
