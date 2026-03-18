"""
Analyse the results of the simple synthetic SAXS fitting.

Plots the KLD (weights to uniform) across the splits and replicates.
Selects best model based on validation error from the loss history.

For the best models (across split types, show variation across replicates):
- Plots Effective sample size (ESS) across the splits and replicates.
- Plots the recovery against the RMSD cluster assignments (APO: cluster 0: 100%, cluster 1: 0%, cluster -1: 0%)
- Plots the SAXS curve of the ensemble (weighted average) against the experimental curve.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.models.SAXS.parameters import SAXS_direct_Model_Parameters


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_run_name(filename: str) -> dict:
    """Parse run name to extract parameters."""
    # Format: {target}_{loss}_{split_type}_split{idx}_maxent{strength}
    pattern = r"([A-Za-z]+)_([A-Za-z0-9]+)_([\w-]+)_split(\d+)_maxent([\d.eE+-]+)"
    match = re.search(pattern, filename)
    if match:
        return {
            "target_curve": match.group(1),
            "loss_type": match.group(2),
            "split_type": match.group(3),
            "split_index": int(match.group(4)),
            "maxent_strength": float(match.group(5)),
        }
    return None


def compute_kld(weights):
    """Compute KL divergence from uniform distribution."""
    n = len(weights)
    uniform = 1.0 / n
    weights_safe = np.clip(weights, 1e-10, None)
    return np.sum(weights_safe * np.log((weights_safe + 1e-10) / (uniform + 1e-10)))


def compute_ess(weights):
    """Compute normalized effective sample size."""
    n = len(weights)
    return (1.0 / np.sum(weights**2)) / n


def compute_jsd(p, q):
    """Jensen-Shannon divergence (returns JSD, not sqrt)."""
    p = np.clip(p, 1e-15, None)
    p /= p.sum()
    q = np.clip(q, 1e-15, None)
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def load_results(results_dir: str) -> pd.DataFrame:
    """Walk results directory and build dataframe from HDF5 files."""
    results_dir = Path(results_dir)
    records = []

    for hdf_path in sorted(results_dir.rglob("*_results.hdf5")):
        # Skip EMA files
        if "_EMA" in hdf_path.name:
            continue

        # Parse run name
        params = parse_run_name(hdf_path.name)
        if params is None:
            print(f"Warning: Could not parse {hdf_path.name}")
            continue

        # Load optimization history
        try:
            history = load_optimization_history_from_file(
                str(hdf_path),
                default_model_params_cls=SAXS_direct_Model_Parameters
            )
        except Exception as e:
            print(f"Warning: Could not load {hdf_path}: {e}")
            continue

        # Get best state
        best = history.get_best_state()
        weights = np.abs(np.array(best.params.frame_weights))
        weights /= weights.sum()

        # Compute metrics
        kld = compute_kld(weights)
        ess = compute_ess(weights)
        best_val_loss = float(best.losses.val_losses[0])

        # Extract error loss history (first component)
        train_error_history = [float(s.losses.train_losses[0]) for s in history.states]
        val_error_history = [float(s.losses.val_losses[0]) for s in history.states]

        record = {
            **params,
            "kld": kld,
            "ess": ess,
            "best_val_loss": best_val_loss,
            "train_error_history": train_error_history,
            "val_error_history": val_error_history,
            "final_frame_weights": weights,
            "hdf_path": str(hdf_path),
        }
        records.append(record)

    return pd.DataFrame(records)


def plot_kld_vs_maxent(df: pd.DataFrame, output_dir: Path):
    """Plot KLD vs maxent strength with mean ± SE bands across replicates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by (target_curve, loss_type, split_type)
    for (target, loss), group_tl in df.groupby(["target_curve", "loss_type"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(f"KLD vs MaxEnt Strength ({target}, {loss})")

        split_types = sorted(group_tl["split_type"].unique())
        for ax, split_type in zip(axes, split_types):
            group_st = group_tl[group_tl["split_type"] == split_type]

            # Group by maxent_strength and compute mean ± SE across replicates
            maxent_values = sorted(group_st["maxent_strength"].unique())
            means = []
            ses = []

            for maxent in maxent_values:
                klds = group_st[group_st["maxent_strength"] == maxent]["kld"].values
                means.append(np.mean(klds))
                if len(klds) > 1:
                    se = np.std(klds, ddof=1) / np.sqrt(len(klds))
                else:
                    se = 0
                ses.append(se)

            means = np.array(means)
            ses = np.array(ses)

            # Plot mean line
            ax.loglog(maxent_values, means, marker="o", color="blue", linewidth=2, label="Mean")

            # Fill between with SE bounds (clip lower bound to avoid log issues)
            lower_bounds = np.maximum(means - ses, 1e-15)
            upper_bounds = means + ses
            ax.fill_between(maxent_values, lower_bounds, upper_bounds, alpha=0.3, color="blue")

            ax.set_xlabel("MaxEnt Strength (log scale)")
            ax.set_ylabel("KLD" if ax == axes[0] else "")
            ax.set_title(split_type)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        fname = f"KLD_vs_maxent_{target}_{loss}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def select_best_per_replicate(df: pd.DataFrame) -> pd.DataFrame:
    """Select best model per (split_type, loss_type, target_curve, split_index) based on min val_loss."""
    best_models = []
    for (split_type, loss_type, target_curve, split_index), group in df.groupby(
        ["split_type", "loss_type", "target_curve", "split_index"]
    ):
        best_idx = group["best_val_loss"].idxmin()
        best_models.append(group.loc[best_idx])

    return pd.DataFrame(best_models)


def plot_ess(df: pd.DataFrame, output_dir: Path):
    """Plot ESS bar chart with mean ± SE across replicates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by loss_type and target_curve
    for (loss_type, target_curve), group in df.groupby(["loss_type", "target_curve"]):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Compute mean ± SE per split_type
        split_types = sorted(group["split_type"].unique())
        means = []
        ses = []

        for split_type in split_types:
            ess_values = group[group["split_type"] == split_type]["ess"].values
            means.append(np.mean(ess_values))
            if len(ess_values) > 1:
                se = np.std(ess_values, ddof=1) / np.sqrt(len(ess_values))
            else:
                se = 0
            ses.append(se)

        x = np.arange(len(split_types))
        colors = {"random": "blue", "stratified": "green", "random-stratified": "red"}
        bar_colors = [colors.get(st, "gray") for st in split_types]

        bars = ax.bar(x, means, yerr=ses, capsize=5, color=bar_colors, alpha=0.7)

        ax.set_xlabel("Split Type")
        ax.set_ylabel("Normalized ESS")
        ax.set_title(f"Effective Sample Size ({target_curve}, {loss_type})")
        ax.set_xticks(x)
        ax.set_xticklabels(split_types)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis="y")

        legend_elements = [
            Patch(facecolor=color, label=split_type)
            for split_type, color in colors.items()
            if split_type in split_types
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        fname = f"ESS_{target_curve}_{loss_type}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def plot_cluster_recovery(df: pd.DataFrame, output_dir: Path, clusters_df: pd.DataFrame):
    """Plot cluster weight recovery with reference panels and mean ± SE bars."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = clusters_df["cluster_id"].values
    unique_clusters = sorted(np.unique(cluster_ids))

    for (loss_type, target_curve), group in df.groupby(["loss_type", "target_curve"]):
        # Layout: 2 reference panels + 1 per split_type
        split_types = sorted(group["split_type"].unique())
        n_panels = 2 + len(split_types)

        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(f"Cluster Weight Recovery ({target_curve}, {loss_type})")

        # Panel 0: Target (ideal APO recovery)
        target_weights = np.zeros(len(unique_clusters))
        cluster_0_idx = list(unique_clusters).index(0)  # Find correct index for cluster 0
        target_weights[cluster_0_idx] = 1.0  # Cluster 0 = 100%

        bars = axes[0].bar(range(len(unique_clusters)), target_weights)
        for i, cid in enumerate(unique_clusters):
            bars[i].set_color("green" if cid == 0 else "gray")

        # Target always has 100% recovery (comparing against itself)
        axes[0].set_title("Target\n(Recovery: 100.0%)")
        axes[0].set_xticks(range(len(unique_clusters)))
        axes[0].set_xticklabels([str(c) for c in unique_clusters])
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].set_ylabel("Weight")

        # Panel 1: Uniform Prior (no-optimization baseline)
        frame_count = 12700
        uniform_weights = np.array([np.sum(cluster_ids == c) / frame_count for c in unique_clusters])

        bars = axes[1].bar(range(len(unique_clusters)), uniform_weights)
        for i, cid in enumerate(unique_clusters):
            bars[i].set_color("green" if cid == 0 else "gray")

        # Compute recovery for uniform prior
        uniform_recovery = (1 - np.sqrt(compute_jsd(uniform_weights, target_weights))) * 100
        axes[1].set_title(f"Uniform Prior\n(Recovery: {uniform_recovery:.1f}%)")
        axes[1].set_xticks(range(len(unique_clusters)))
        axes[1].set_xticklabels([str(c) for c in unique_clusters])
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis="y")

        # Panels 2+: One per split_type with mean ± SE
        for panel_idx, split_type in enumerate(split_types, start=2):
            ax = axes[panel_idx]
            group_st = group[group["split_type"] == split_type]

            # Build (n_replicates, n_clusters) weight array
            weight_array = []
            for _, row in group_st.iterrows():
                weights = row["final_frame_weights"]
                per_cluster = np.array([weights[cluster_ids == c].sum() for c in unique_clusters])
                weight_array.append(per_cluster)

            weight_array = np.array(weight_array)
            means = np.mean(weight_array, axis=0)
            if len(weight_array) > 1:
                ses = np.std(weight_array, axis=0, ddof=1) / np.sqrt(len(weight_array))
            else:
                ses = np.zeros(len(unique_clusters))

            bars = ax.bar(range(len(unique_clusters)), means, yerr=ses, capsize=5, alpha=0.7)
            for i, cid in enumerate(unique_clusters):
                bars[i].set_color("green" if cid == 0 else "gray")

            # Compute recovery for this split_type
            split_recovery = (1 - np.sqrt(compute_jsd(means, target_weights))) * 100
            ax.set_title(f"{split_type}\n(Recovery: {split_recovery:.1f}%)")
            ax.set_xticks(range(len(unique_clusters)))
            ax.set_xticklabels([str(c) for c in unique_clusters])
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = f"Cluster_recovery_{target_curve}_{loss_type}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def plot_loss_convergence(df: pd.DataFrame, output_dir: Path):
    """Plot training and validation error loss convergence over optimization steps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by (target_curve, loss_type, split_type)
    for (target, loss, split), group in df.groupby(["target_curve", "loss_type", "split_type"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique maxent strengths for coloring
        maxent_values = sorted(group["maxent_strength"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))
        color_map = dict(zip(maxent_values, colors))

        for _, row in group.iterrows():
            color = color_map[row["maxent_strength"]]
            
            # Plot training loss (solid)
            ax.plot(row["train_error_history"], 
                    color=color, 
                    linestyle="-", 
                    alpha=0.6,
                    label=f"MaxEnt {row['maxent_strength']} (Train)" if row["split_index"] == 0 else None)
            
            # Plot validation loss (dashed)
            ax.plot(row["val_error_history"], 
                    color=color, 
                    linestyle="--", 
                    alpha=0.8,
                    label=f"MaxEnt {row['maxent_strength']} (Val)" if row["split_index"] == 0 else None)

        ax.set_yscale("log")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Error Loss (Component 0)")
        ax.set_title(f"Loss Convergence ({target}, {loss}, {split})")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        fname = f"Loss_convergence_{target}_{loss}_{split}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def plot_recovery_vs_maxent(df: pd.DataFrame, output_dir: Path, clusters_df: pd.DataFrame):
    """Plot cluster recovery vs maxent strength with mean ± SE bands across replicates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = clusters_df["cluster_id"].values
    unique_clusters = sorted(np.unique(cluster_ids))

    # Target: cluster 0 = 100%
    target_weights = np.zeros(len(unique_clusters))
    cluster_0_idx = list(unique_clusters).index(0)
    target_weights[cluster_0_idx] = 1.0

    for (target, loss), group_tl in df.groupby(["target_curve", "loss_type"]):
        split_types = sorted(group_tl["split_type"].unique())
        fig, axes = plt.subplots(1, len(split_types), figsize=(5 * len(split_types), 4), sharey=True)
        if len(split_types) == 1:
            axes = [axes]

        fig.suptitle(f"Recovery vs MaxEnt Strength ({target}, {loss})")

        for ax, split_type in zip(axes, split_types):
            group_st = group_tl[group_tl["split_type"] == split_type]
            maxent_values = sorted(group_st["maxent_strength"].unique())
            means = []
            ses = []

            for maxent in maxent_values:
                group_m = group_st[group_st["maxent_strength"] == maxent]
                recoveries = []
                for _, row in group_m.iterrows():
                    weights = row["final_frame_weights"]
                    per_cluster = np.array([weights[cluster_ids == c].sum() for c in unique_clusters])
                    recovery = (1 - np.sqrt(compute_jsd(per_cluster, target_weights))) * 100
                    recoveries.append(recovery)
                means.append(np.mean(recoveries))
                ses.append(np.std(recoveries, ddof=1) / np.sqrt(len(recoveries)) if len(recoveries) > 1 else 0)

            means = np.array(means)
            ses = np.array(ses)

            ax.semilogx(maxent_values, means, marker="o", color="green", linewidth=2, label="Mean")
            ax.fill_between(maxent_values, means - ses, means + ses, alpha=0.3, color="green")

            ax.set_xlabel("MaxEnt Strength (log scale)")
            ax.set_ylabel("Recovery (%)" if ax == axes[0] else "")
            ax.set_title(split_type)
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        fname = f"Recovery_vs_maxent_{target}_{loss}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def plot_ensemble_saxs_curves(df: pd.DataFrame, output_dir: Path):
    """Plot ensemble-weighted SAXS curves with mean ± SE bands."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FOXS curves
    FOXS_DIR = SCRIPT_DIR.parent.parent / "4_SAXS" /"FOXS"
    foxs_data = np.load(FOXS_DIR / "CaM_SAXS_ordered.npz")
    curve_matrix = foxs_data["saxs"].T  # (501, 12700)

    for (loss_type, target_curve), group in df.groupby(["loss_type", "target_curve"]):
        # Load experimental curve
        fname = "1CLL_apo.pdb.dat" if target_curve == "APO" else "1CLL_nosol.pdb.dat"
        exp_data = np.loadtxt(FOXS_DIR / fname, comments="#")
        exp_q, exp_I, exp_err = exp_data[:, 0], exp_data[:, 1], exp_data[:, 2]

        split_types = sorted(group["split_type"].unique())
        fig, axes = plt.subplots(1, len(split_types), figsize=(5 * len(split_types), 4))
        if len(split_types) == 1:
            axes = [axes]

        fig.suptitle(f"Ensemble SAXS Curves ({target_curve}, {loss_type})")

        for ax, split_type in zip(axes, split_types):
            group_st = group[group["split_type"] == split_type]

            # Build (n_replicates, 501) curve array
            curve_array = []
            for _, row in group_st.iterrows():
                weights = row["final_frame_weights"]
                ens_curve = curve_matrix @ weights
                curve_array.append(ens_curve)

            curve_array = np.array(curve_array)
            mean_curve = np.mean(curve_array, axis=0)
            if len(curve_array) > 1:
                sd_curve = np.std(curve_array, axis=0, ddof=1)
            else:
                sd_curve = np.zeros_like(mean_curve)

            # Compute correlations
            pearson_r, _ = stats.pearsonr(mean_curve, exp_I)
            spearman_r, _ = stats.spearmanr(mean_curve, exp_I)

            ax.semilogy(exp_q, exp_I, "k-", linewidth=2, label="Experimental")
            ax.fill_between(exp_q, exp_I - exp_err, exp_I + exp_err, alpha=0.2, color="gray")
            ax.semilogy(exp_q, mean_curve, "r--", linewidth=2, label="Ensemble Mean")

            # Fill between with SD bounds (clip lower bound)
            lower_bounds = np.maximum(mean_curve - sd_curve, 1e-15)
            upper_bounds = mean_curve + sd_curve
            ax.fill_between(exp_q, lower_bounds, upper_bounds, alpha=0.3, color="red")

            ax.set_xlabel("q (Å⁻¹)")
            ax.set_ylabel("I(q)")
            ax.set_title(f"{split_type}\nPearson r={pearson_r:.3f}  Spearman ρ={spearman_r:.3f}")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend()

        plt.tight_layout()
        fname = f"SAXS_curves_{target_curve}_{loss_type}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAXS fitting results")
    parser.add_argument("--results-dir", required=True, help="Path to results directory")

    args = parser.parse_args()

    print("Loading results...")
    df = load_results(args.results_dir)

    if len(df) == 0:
        print("No results found!")
        return

    print(f"Loaded {len(df)} results")
    print(df.to_string())

    # Create analysis output directory
    output_dir = Path(args.results_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clusters_df once
    clusters_csv = SCRIPT_DIR.parent / "data" / "_RMSD_cluster_output" / "cluster_assignments.csv"
    clusters_df = pd.read_csv(clusters_csv) if clusters_csv.exists() else None

    # Step 1: Plot KLD vs maxent
    print("\nPlotting KLD vs maxent...")
    plot_kld_vs_maxent(df, output_dir)

    # Step 1b: Plot Recovery vs maxent
    print("Plotting Recovery vs maxent...")
    if clusters_df is not None:
        plot_recovery_vs_maxent(df, output_dir, clusters_df)
    else:
        print("Warning: Cluster assignments not found, skipping recovery vs maxent plot")

    # Step 2: Select best models per replicate
    print("Selecting best models per replicate...")
    best_per_replicate = select_best_per_replicate(df)
    print(f"Selected {len(best_per_replicate)} best models (per replicate)")
    print(best_per_replicate[["target_curve", "loss_type", "split_type", "split_index", "maxent_strength", "best_val_loss"]])

    # Step 3: Plot ESS
    print("\nPlotting ESS...")
    plot_ess(best_per_replicate, output_dir)

    # Step 4: Plot cluster recovery
    print("Plotting cluster recovery...")
    if clusters_df is not None:
        plot_cluster_recovery(best_per_replicate, output_dir, clusters_df)
    else:
        print("Warning: Cluster assignments not found, skipping cluster recovery plots")

    # Step 5: Plot ensemble SAXS curves
    print("Plotting ensemble SAXS curves...")
    plot_ensemble_saxs_curves(best_per_replicate, output_dir)

    # Step 6: Plot loss convergence (for all results, not just best)
    print("\nPlotting loss convergence curves...")
    plot_loss_convergence(df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
