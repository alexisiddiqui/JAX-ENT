from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_weight_distribution_lines(
    weights_data,
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Plot frame-weight distributions as line plots with maxent as hue.

    Parameters
    ----------
    weights_data:
        DataFrame (or list of dicts) produced by
        ``analysis.extract_frame_weights_kl``.  Must contain columns
        ``ensemble``, ``loss_function``, ``split_type``, ``maxent_value``,
        ``weights``.
    output_dir:
        Directory to save the output PNG files.
    split_name_mapping:
        Optional ``{split_type_key: display_name}`` dict for axis titles.
        Defaults to identity (raw key used as title).
    """
    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    weights_df = pd.DataFrame(weights_data) if not isinstance(weights_data, pd.DataFrame) else weights_data.copy()

    if weights_df.empty:
        print("  No weights data available for plotting")
        return

    weights_df["ensemble_loss"] = weights_df["ensemble"] + "|" + weights_df["loss_function"]

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        split_types = sorted(ensemble_loss_data["split_type"].unique())
        if not split_types:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, split_type in enumerate(split_types[:4]):
            ax = axes[idx]
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            if split_data.empty:
                ax.set_visible(False)
                continue

            maxent_groups: Dict = {}
            for _, row in split_data.iterrows():
                maxent = row["maxent_value"]
                if maxent not in maxent_groups:
                    maxent_groups[maxent] = []
                maxent_groups[maxent].append(row["weights"])

            maxent_values = sorted(maxent_groups.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

            weight_bins = np.logspace(-50, 0, 50)
            bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

            for maxent_val, color in zip(maxent_values, colors):
                weights_list = maxent_groups[maxent_val]
                hist_counts = []
                for w in weights_list:
                    if len(w) > 0 and np.sum(w) > 0:
                        counts, _ = np.histogram(w, bins=weight_bins, density=True)
                        hist_counts.append(counts)

                if hist_counts:
                    mean_counts = np.mean(hist_counts, axis=0)
                    std_counts = (
                        np.std(hist_counts, axis=0) if len(hist_counts) > 1 else np.zeros_like(mean_counts)
                    )
                    ax.plot(bin_centers, mean_counts, color=color, alpha=0.8,
                            label=f"MaxEnt={maxent_val:.0e}", linewidth=2)
                    if len(hist_counts) > 1:
                        ax.fill_between(bin_centers, mean_counts - std_counts,
                                        mean_counts + std_counts, color=color, alpha=0.2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")
            ax.set_title(split_name_mapping.get(split_type, split_type))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(split_types), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Weight Distributions — {ensemble_loss}", fontsize=16, y=0.98)
        plt.tight_layout()
        filename = f"weight_distributions_lines_{ensemble_loss.replace('|', '_').replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
