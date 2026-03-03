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

def plot_weight_distribution_maxent_panels(
    conv_weights_df, 
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Weight distributions with maxent values as panels and convergence as lines."""
    print("Creating weight distribution plots with maxent panels...")

    if conv_weights_df is None or len(conv_weights_df) == 0:
        print("  No convergence weights data available for plotting")
        return

    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    weights_df = (
        pd.DataFrame(conv_weights_df)
        if not isinstance(conv_weights_df, pd.DataFrame)
        else conv_weights_df.copy()
    )
    weights_df["ensemble_loss"] = weights_df["ensemble"] + "|" + weights_df["loss_function"]

    weight_bins = np.logspace(-50, 0, 50)
    bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        el_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if el_data.empty:
            continue

        for split_type in sorted(el_data["split_type"].unique()):
            split_data = el_data[el_data["split_type"] == split_type]
            if split_data.empty:
                continue

            maxent_values = sorted(split_data["maxent_value"].unique())
            if len(maxent_values) < 2:
                continue

            n_maxent = len(maxent_values)
            n_cols = min(4, n_maxent)
            n_rows = (n_maxent + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_maxent > 1 else [axes]

            for idx, maxent_val in enumerate(maxent_values):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                maxent_data = split_data[split_data["maxent_value"] == maxent_val]
                if maxent_data.empty:
                    ax.set_visible(False)
                    continue

                conv_groups = {}
                for _, row in maxent_data.iterrows():
                    cf = row["convergence_fraction"]
                    if cf not in conv_groups:
                        conv_groups[cf] = []
                    conv_groups[cf].append(row["weights"])

                conv_fractions = sorted(conv_groups.keys())
                colors = plt.cm.plasma(np.linspace(0, 1, len(conv_fractions)))

                for conv_frac, color in zip(conv_fractions, colors):
                    hist_counts = []
                    for w in conv_groups[conv_frac]:
                        if len(w) > 0 and np.sum(w) > 0:
                            counts, _ = np.histogram(w, bins=weight_bins, density=True)
                            hist_counts.append(counts)
                    if hist_counts:
                        mean_counts = np.mean(hist_counts, axis=0)
                        std_counts = (
                            np.std(hist_counts, axis=0)
                            if len(hist_counts) > 1
                            else np.zeros_like(mean_counts)
                        )
                        ax.plot(
                            bin_centers, mean_counts,
                            color=color, alpha=0.8,
                            label=f"{conv_frac:.1%}", linewidth=2,
                        )
                        if len(hist_counts) > 1:
                            ax.fill_between(
                                bin_centers,
                                mean_counts - std_counts,
                                mean_counts + std_counts,
                                color=color, alpha=0.2,
                            )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"MaxEnt = {maxent_val:.0e}")
                ax.legend(title="Convergence", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(maxent_values), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            split_name = split_name_mapping.get(split_type, split_type)
            plt.suptitle(
                f"Weight Evolution Over Convergence — {ensemble_loss.replace('|', '_')} — {split_name}",
                fontsize=16, y=0.98,
            )
            plt.tight_layout()
            fname = (
                f"weight_distributions_maxent_panels_{ensemble_loss.replace('|', '_')}_{split_type}"
                .replace("/", "_").replace(" ", "_") + ".png"
            )
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
            print(f"  Saved: {fname}")
            plt.close()


def plot_weight_distribution_convergence_panels(
    conv_weights_df, 
    output_dir: str,
    split_name_mapping: Dict[str, str] | None = None,
) -> None:
    """Weight distributions with convergence fractions as panels and maxent as lines."""
    print("Creating weight distribution plots with convergence panels...")

    if conv_weights_df is None or len(conv_weights_df) == 0:
        print("  No convergence weights data available for plotting")
        return

    os.makedirs(output_dir, exist_ok=True)
    if split_name_mapping is None:
        split_name_mapping = {}

    weights_df = (
        pd.DataFrame(conv_weights_df)
        if not isinstance(conv_weights_df, pd.DataFrame)
        else conv_weights_df.copy()
    )
    weights_df["ensemble_loss"] = weights_df["ensemble"] + "|" + weights_df["loss_function"]

    weight_bins = np.logspace(-50, 0, 50)
    bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        el_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if el_data.empty:
            continue

        for split_type in sorted(el_data["split_type"].unique()):
            split_data = el_data[el_data["split_type"] == split_type]
            if split_data.empty:
                continue

            conv_fractions = sorted(split_data["convergence_fraction"].unique())
            if len(conv_fractions) < 2:
                continue

            n_conv = len(conv_fractions)
            n_cols = min(3, n_conv)
            n_rows = (n_conv + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_conv > 1 else [axes]

            for idx, conv_frac in enumerate(conv_fractions):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                conv_data = split_data[split_data["convergence_fraction"] == conv_frac]
                if conv_data.empty:
                    ax.set_visible(False)
                    continue

                maxent_groups = {}
                for _, row in conv_data.iterrows():
                    maxent = row["maxent_value"]
                    if maxent not in maxent_groups:
                        maxent_groups[maxent] = []
                    maxent_groups[maxent].append(row["weights"])

                maxent_values = sorted(maxent_groups.keys())
                colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

                for maxent_val, color in zip(maxent_values, colors):
                    hist_counts = []
                    for w in maxent_groups[maxent_val]:
                        if len(w) > 0 and np.sum(w) > 0:
                            counts, _ = np.histogram(w, bins=weight_bins, density=True)
                            hist_counts.append(counts)
                    if hist_counts:
                        mean_counts = np.mean(hist_counts, axis=0)
                        ax.plot(
                            bin_centers, mean_counts,
                            color=color, alpha=0.8,
                            label=f"{maxent_val:.0e}", linewidth=2,
                        )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Convergence: {conv_frac:.1%}")
                ax.legend(title="MaxEnt", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(conv_fractions), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            split_name = split_name_mapping.get(split_type, split_type)
            plt.suptitle(
                f"MaxEnt Comparison Across Convergence — {ensemble_loss.replace('|', '_')} — {split_name}",
                fontsize=16, y=0.98,
            )
            plt.tight_layout()
            fname = (
                f"weight_distributions_convergence_panels_{ensemble_loss.replace('|', '_')}_{split_type}"
                .replace("/", "_").replace(" ", "_") + ".png"
            )
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
            print(f"  Saved: {fname}")
            plt.close()
