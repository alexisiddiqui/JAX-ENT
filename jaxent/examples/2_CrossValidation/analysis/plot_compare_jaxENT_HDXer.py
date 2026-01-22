"""
Compare HDXer and jaxENT results with publication-ready plots.
Creates bar plots of recovery % and KL divergence, panelled by experiment and by ensemble.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Publication-ready style configuration
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    },
)

# Color schemes (unified from both scripts)
ENSEMBLE_COLOURS = {
    "ISO_BI": "indigo",
    "ISO_TRI": "saddlebrown",
    "ISO-BiModal": "indigo",
    "ISO-TriModal": "saddlebrown",
    "AF2_MSAss": "RoyalBlue",  # Blue
    "AF2_filtered": "Cyan",   # Cyan
    "MoPrP_af_dirty": "RoyalBlue", # Blue
    "MoPrP_af_clean": "Cyan",       # Cyan
}

SPLIT_TYPE_COLOURS = {
    "r": "fuchsia",
    "s": "black",
    "R3": "green",
    "sequence_cluster": "green",
    "Sp": "grey",
    "spatial": "grey",
}

SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",
}

# Reverse mapping for harmonization
SPLIT_TYPE_NORMALIZATION = {
    "sequence_cluster": "R3",
    "spatial": "Sp",
    "R3": "R3",
    "Sp": "Sp", 
    "r": "r",
    "s": "s",
    "R3_k_sequence": "R3",
    "Sp_res_neighbours": "Sp",
    "r_naive_random": "r",
}

# Create palette that maps split names to colors
SPLIT_NAME_COLOURS = {
    "Random": "fuchsia",
    "Sequence": "black",
    "Non-Redundant": "green",
    "Spatial": "grey",
}

# Experiment colors
EXPERIMENT_COLOURS = {
    "HDXer": "crimson",
    "mcMSE": "navy",
    "MSE": "blue",
    "Sigma_MSE": "indigo",
}
EXPERIMENT_HATCHING = {
    "HDXer": "",
    "mcMSE": "",
    "MSE": "",
    "Sigma_MSE": "",
}

def load_and_harmonize_data(hdxer_path, jaxent_path):
    """
    Load and harmonize HDXer and jaxENT data for comparison.
    
    Args:
        hdxer_path: Path to HDXer selected_metrics_all.csv
        jaxent_path: Path to jaxENT selected_metrics_all.csv
        
    Returns:
        Combined DataFrame with harmonized columns
    """
    # Load HDXer data
    hdxer_df = pd.read_csv(hdxer_path)
    
    # Check which columns exist and harmonize
    if "kl_div_uniform" in hdxer_df.columns:
        hdxer_df["kl_divergence"] = hdxer_df["kl_div_uniform"]
    if "open_state_recovery" in hdxer_df.columns:
        hdxer_df["recovery_percent"] = hdxer_df["open_state_recovery"]
    
    # Harmonize ensemble names
    hdxer_df["ensemble"] = hdxer_df["ensemble"].replace({
        "ISO-BiModal": "ISO_BI",
        "ISO-TriModal": "ISO_TRI"
    })
    
    # Add experiment identifier
    hdxer_df["experiment"] = "HDXer"
    hdxer_df["loss_function"] = "HDXer"
    
    # Load jaxENT data
    jaxent_df = pd.read_csv(jaxent_path)
    
    # Normalize split_type codes for both datasets
    for df in [hdxer_df, jaxent_df]:
        if "split_type" in df.columns:
            df["split_type"] = df["split_type"].map(SPLIT_TYPE_NORMALIZATION).fillna(df["split_type"])
    
    # Ensure split_name exists for both
    for df in [hdxer_df, jaxent_df]:
        if "split_name" not in df.columns and "split_type" in df.columns:
            df["split_name"] = df["split_type"].map(SPLIT_NAME_MAPPING)
        elif "split_name" in df.columns:
            # Ensure split_name is consistent with mapping
            df["split_name"] = df["split_type"].map(SPLIT_NAME_MAPPING)
    
    # Add experiment identifier based on loss function for jaxENT
    jaxent_df["experiment"] = jaxent_df["loss_function"]
    
    # Select common columns
    common_cols = [
        "ensemble", "split_type", "split_name", "experiment", 
        "loss_function", "kl_divergence", "recovery_percent"
    ]
    
    # Add replicate/split column if available
    if "replicate" in hdxer_df.columns:
        hdxer_df["replicate_id"] = hdxer_df["replicate"].astype(str)
        common_cols.append("replicate_id")
    if "split" in jaxent_df.columns:
        jaxent_df["replicate_id"] = jaxent_df["split"].astype(str)
        if "replicate_id" not in common_cols:
            common_cols.append("replicate_id")
    
    # Filter to only include columns that exist in both
    hdxer_cols = [col for col in common_cols if col in hdxer_df.columns]
    jaxent_cols = [col for col in common_cols if col in jaxent_df.columns]
    
    # Combine datasets
    combined_df = pd.concat([
        hdxer_df[hdxer_cols],
        jaxent_df[jaxent_cols]
    ], ignore_index=True)
    
    # Final check: ensure split_name is properly set
    combined_df["split_name"] = combined_df["split_type"].map(SPLIT_NAME_MAPPING)

    # Harmonize AF2 ensemble names
    combined_df["ensemble"] = combined_df["ensemble"].replace({
        "MoPrP_af_dirty": "AF2_MSAss",
        "MoPrP_af_clean": "AF2_filtered"
    })
    
    # Print diagnostics
    print("\nData Harmonization Summary:")
    print(f"  HDXer records: {len(hdxer_df)}")
    print(f"  jaxENT records: {len(jaxent_df)}")
    print(f"  Combined records: {len(combined_df)}")
    print(f"  Split types in combined data: {sorted(combined_df['split_type'].astype(str).unique())}")
    print(f"  Split names in combined data: {sorted(combined_df['split_name'].astype(str).unique())}")
    print(f"  Ensembles: {sorted(combined_df['ensemble'].unique())}")
    print(f"  Experiments: {sorted(combined_df['experiment'].unique())}")
    
    return combined_df


def _apply_hatches_by_experiment_and_split(ax, experiment_order, split_name_order):
    """
    Apply hatch patterns so that all bars for a given experiment share the same hatch.

    Important: do NOT rely on tick labels (sharex=True often means top-row tick labels are blank).
    Instead, infer the x-category index from patch x-center and map it to experiment_order.
    """
    # Group patches by their x "bin" (experiment position)
    grouped = {}
    for patch in ax.patches:
        x_center = patch.get_x() + patch.get_width() / 2.0
        x_group = int(round(x_center))
        grouped.setdefault(x_group, []).append(patch)

    # seaborn uses categorical positions 0..N-1 for order=experiment_order
    for x_group, patches in grouped.items():
        if x_group < 0 or x_group >= len(experiment_order):
            continue
        exp_name = experiment_order[x_group]
        hatch = EXPERIMENT_HATCHING.get(exp_name, "")

        for p in patches:
            p.set_hatch(hatch)
            p.set_edgecolor("black")
            p.set_linewidth(1.2)


def _annotate_bar_heights(ax):
    """
    Annotate bar heights on the recovery % plots for clarity.
    
    Args:
        ax: The axis object to annotate
    """
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2., height),
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=12,
            color="black",
            weight="bold",
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
        )


def plot_comparison_by_experiment(data, output_dir, analysis_split_types=None):
    """
    Create bar plots panelled by experiment (HDXer vs loss functions).
    Separate plots for recovery % and KL divergence.
    
    Args:
        data: Combined DataFrame
        output_dir: Output directory path
        analysis_split_types: List of split types to include (default: ["R3", "Sp"])
    """
    # Default to only Non-Redundant and Spatial
    if analysis_split_types is None:
        analysis_split_types = ["R3", "Sp"]
    
    # Filter for desired split types
    plot_df = data[data["split_type"].isin(analysis_split_types)].copy()
    
    if plot_df.empty:
        print("No data available for experiment comparison plot")
        return
    
    # Get unique experiments in desired order
    experiment_order = ["HDXer", "mcMSE", "MSE", "Sigma_MSE"]
    available_experiments = [exp for exp in experiment_order if exp in plot_df["experiment"].unique()]
    
    if not available_experiments:
        print("No matching experiments found for experiment comparison plot.")
        return

    # Determine split name order based on available data
    all_split_names = sorted(plot_df["split_name"].dropna().unique())
    preferred_order = ["Non-Redundant", "Spatial"]  # Only these two
    split_name_order = [name for name in preferred_order if name in all_split_names]
    # Add any remaining split names not in preferred order (shouldn't happen with filtered data)
    split_name_order.extend([name for name in all_split_names if name not in split_name_order])
    
    # Create figure with 2 rows (recovery and KL), columns for each experiment
    fig, axes = plt.subplots(
        2, len(available_experiments), 
        figsize=(6 * len(available_experiments), 10),
        sharex=True
    )
    
    if len(available_experiments) == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot recovery percentages (top row)
    for col_idx, experiment in enumerate(available_experiments):
        ax = axes[0, col_idx]
        exp_data = plot_df[plot_df["experiment"] == experiment]
        
        if not exp_data.empty:
            sns.barplot(
                data=exp_data,
                x="ensemble",
                y="recovery_percent",
                hue="split_name",
                hue_order=split_name_order,
                ax=ax,
                palette=SPLIT_NAME_COLOURS,
                estimator=np.mean,
                errorbar="se",
                capsize=0.06,
                errwidth=1.5,
                edgecolor="black",
                linewidth=1.2,
            )
            
            # Apply hatch pattern to bars
            hatch = EXPERIMENT_HATCHING.get(experiment, "")
            for patch in ax.patches:
                patch.set_hatch(hatch)
            
            # Styling
            ax.set_title(f"{experiment}", fontsize=22, fontweight="bold", 
                        color=EXPERIMENT_COLOURS.get(experiment, "black"))
            ax.set_ylabel("Ground Truth Recovery (%)" if col_idx == 0 else "", fontsize=18)
            ax.set_xlabel("")
            ax.set_ylim(0, 100)
            ax.grid(False)
            
            # Color x-tick labels by ensemble
            for tick in ax.get_xticklabels():
                ensemble_name = tick.get_text()
                tick.set_color(ENSEMBLE_COLOURS.get(ensemble_name, "black"))
                tick.set_fontweight("bold")
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            _annotate_bar_heights(ax)
        else:
            ax.set_visible(False)
    
    # Plot KL divergence (bottom row)
    for col_idx, experiment in enumerate(available_experiments):
        ax = axes[1, col_idx]
        exp_data = plot_df[plot_df["experiment"] == experiment]
        
        if not exp_data.empty:
            sns.barplot(
                data=exp_data,
                x="ensemble",
                y="kl_divergence",
                hue="split_name",
                hue_order=split_name_order,
                ax=ax,
                palette=SPLIT_NAME_COLOURS,
                estimator=np.mean,
                errorbar="se",
                capsize=0.06,
                errwidth=1.5,
                edgecolor="black",
                linewidth=1.2,
            )
            
            # Apply hatch pattern to bars
            hatch = EXPERIMENT_HATCHING.get(experiment, "")
            for patch in ax.patches:
                patch.set_hatch(hatch)
            
            # Styling
            ax.set_ylabel(r"KL(P||U$_{\mathrm{uniform}}$)" if col_idx == 0 else "", fontsize=18)
            ax.set_xlabel("Ensemble", fontsize=18)
            ax.set_ylim(0, None)
            ax.grid(False)
            
            # Color x-tick labels by ensemble
            for tick in ax.get_xticklabels():
                ensemble_name = tick.get_text()
                tick.set_color(ENSEMBLE_COLOURS.get(ensemble_name, "black"))
                tick.set_fontweight("bold")
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            ax.set_visible(False)
    
    # Create unified legend
    handles = [
        Patch(facecolor=SPLIT_NAME_COLOURS.get(name, "grey"),
              edgecolor="black", label=name)
        for name in split_name_order
    ]
    
    fig.legend(
        handles,
        split_name_order,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        title="Split Type",
        title_fontsize=16,
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    output_path = os.path.join(output_dir, "comparison_by_experiment.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved experiment comparison plot to {output_path}")
    plt.close(fig)
    
    return fig


def plot_comparison_by_ensemble(data, output_dir, analysis_split_types=None):
    """
    Create bar plots panelled by ensemble.
    Separate plots for recovery % and KL divergence.
    
    Args:
        data: Combined DataFrame
        output_dir: Output directory path
        analysis_split_types: List of split types to include (default: ["R3", "Sp"])
    """
    # Default to only Non-Redundant and Spatial
    if analysis_split_types is None:
        analysis_split_types = ["R3", "Sp"]
    
    # Filter for desired split types
    plot_df = data[data["split_type"].isin(analysis_split_types)].copy()
    
    if plot_df.empty:
        print("No data available for ensemble comparison plot")
        return
    
    # Get unique ensembles
    ensemble_order = ["ISO_BI", "ISO_TRI", "AF2_MSAss", "AF2_filtered"]
    available_ensembles = [ens for ens in ensemble_order if ens in plot_df["ensemble"].unique()]
    
    if not available_ensembles:
        print("No matching ensembles found for ensemble comparison plot.")
        return

    # Determine split name order based on available data
    all_split_names = sorted(plot_df["split_name"].dropna().unique())
    preferred_order = ["Non-Redundant", "Spatial"]  # Only these two
    split_name_order = [name for name in preferred_order if name in all_split_names]
    # Add any remaining split names not in preferred order (shouldn't happen with filtered data)
    split_name_order.extend([name for name in all_split_names if name not in split_name_order])
    
    # Create figure with 2 rows (recovery and KL), columns for each ensemble
    fig, axes = plt.subplots(
        2, len(available_ensembles),
        figsize=(8 * len(available_ensembles), 10),
        sharex=True
    )
    
    if len(available_ensembles) == 1:
        axes = axes.reshape(-1, 1)
    
    experiment_order = ["HDXer", "mcMSE", "MSE", "Sigma_MSE"]
    
    # Plot recovery percentages (top row)
    for col_idx, ensemble in enumerate(available_ensembles):
        ax = axes[0, col_idx]
        ens_data = plot_df[plot_df["ensemble"] == ensemble]
        
        if not ens_data.empty:
            sns.barplot(
                data=ens_data,
                x="experiment",
                y="recovery_percent",
                hue="split_name",
                order=experiment_order,
                hue_order=split_name_order,
                ax=ax,
                palette=SPLIT_NAME_COLOURS,
                estimator=np.mean,
                errorbar="se",
                capsize=0.06,
                errwidth=1.5,
                edgecolor="black",
                linewidth=1.2,
            )
            
            # Apply hatch patterns robustly (consistent across split types)
            _apply_hatches_by_experiment_and_split(ax, experiment_order, split_name_order)

            # Styling
            display_name = {
                "ISO_BI": "ISO-BiModal",
                "ISO_TRI": "ISO-TriModal",
                "AF2_MSAss": "AF2 MSAss",
                "AF2_filtered": "AF2 Filtered"
            }.get(ensemble, ensemble)

            ax.set_title(f"MoPrP | {display_name}", fontsize=22, fontweight="bold",
                        color=ENSEMBLE_COLOURS.get(ensemble, "black"))
            ax.set_ylabel("Ground Truth Recovery (%)" if col_idx == 0 else "", fontsize=18)
            ax.set_xlabel("")
            ax.set_ylim(0, 100)
            ax.grid(False)
            
            # Color x-tick labels by experiment
            for tick in ax.get_xticklabels():
                exp_name = tick.get_text()
                tick.set_color(EXPERIMENT_COLOURS.get(exp_name, "black"))
                tick.set_fontweight("bold")
                tick.set_rotation(45)
                tick.set_ha("right")
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            _annotate_bar_heights(ax)
        else:
            ax.set_visible(False)
    
    # Plot KL divergence (bottom row)
    for col_idx, ensemble in enumerate(available_ensembles):
        ax = axes[1, col_idx]
        ens_data = plot_df[plot_df["ensemble"] == ensemble]
        
        if not ens_data.empty:
            sns.barplot(
                data=ens_data,
                x="experiment",
                y="kl_divergence",
                hue="split_name",
                order=experiment_order,
                hue_order=split_name_order,
                ax=ax,
                palette=SPLIT_NAME_COLOURS,
                estimator=np.mean,
                errorbar="se",
                capsize=0.06,
                errwidth=1.5,
                edgecolor="black",
                linewidth=1.2,
            )
            
            # Apply hatch patterns robustly (consistent across split types)
            _apply_hatches_by_experiment_and_split(ax, experiment_order, split_name_order)

            # Styling
            ax.set_ylabel(r"KL(P||U$_{\mathrm{uniform}}$)" if col_idx == 0 else "", fontsize=18)
            ax.set_xlabel("Method", fontsize=18)
            ax.set_ylim(0, None)
            ax.grid(False)
            
            # Color x-tick labels by experiment
            for tick in ax.get_xticklabels():
                exp_name = tick.get_text()
                tick.set_color(EXPERIMENT_COLOURS.get(exp_name, "black"))
                tick.set_fontweight("bold")
                tick.set_rotation(45)
                tick.set_ha("right")
            
            # Remove individual legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            # NOTE: Removed _annotate_bar_heights here so KL plots do NOT get text annotations.
        else:
            ax.set_visible(False)
    
    # Create unified legend
    handles = [
        Patch(facecolor=SPLIT_NAME_COLOURS.get(name, "grey"),
              edgecolor="black", label=name)
        for name in split_name_order
    ]
    
    fig.legend(
        handles,
        split_name_order,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        title="Split Type",
        title_fontsize=16,
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    output_path = os.path.join(output_dir, "comparison_by_ensemble.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ensemble comparison plot to {output_path}")
    plt.close(fig)
    
    return fig


def plot_summary_statistics(data, output_dir, analysis_split_types=None):
    """
    Generate summary statistics table comparing HDXer and jaxENT.
    
    Args:
        data: Combined DataFrame
        output_dir: Output directory path
        analysis_split_types: List of split types to include (default: ["R3", "Sp"])
    """
    # Default to only Non-Redundant and Spatial
    if analysis_split_types is None:
        analysis_split_types = ["R3", "Sp"]
    
    # Filter data
    filtered_data = data[data["split_type"].isin(analysis_split_types)]
    
    summary_stats = filtered_data.groupby(["experiment", "ensemble", "split_type"]).agg({
        "recovery_percent": ["mean", "std", "count"],
        "kl_divergence": ["mean", "std", "count"]
    }).round(2)
    
    # Flatten column names
    summary_stats.columns = ["_".join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()
    
    # Save to CSV
    output_path = os.path.join(output_dir, "comparison_summary_statistics.csv")
    summary_stats.to_csv(output_path, index=False)
    print(f"Saved summary statistics to {output_path}")
    
    # Print to console
    print("\nSummary Statistics:")
    print("=" * 100)
    print(summary_stats.to_string())
    
    return summary_stats


def main():
    """
    Main function to run the comparison analysis.
    """
    # Define paths
    base_dir = "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation"
    hdxer_path = os.path.join(base_dir, "data/_HDXer/selected_metrics_all.csv")
    jaxent_path = os.path.join(base_dir, "analysis/_analysis_optimise_quick_test_FIGURE_SIGMA_5000__20251120_213245/selected_metrics/selected_metrics_all.csv")
    output_dir = os.path.join(base_dir, "analysis/_comparison_plots")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("HDXer vs jaxENT Comparison Analysis")
    print("=" * 80)
    print(f"HDXer data: {hdxer_path}")
    print(f"jaxENT data: {jaxent_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if files exist
    if not os.path.exists(hdxer_path):
        print(f"ERROR: HDXer data file not found: {hdxer_path}")
        return
    
    if not os.path.exists(jaxent_path):
        print(f"ERROR: jaxENT data file not found: {jaxent_path}")
        return
    
    # Load and harmonize data
    print("Loading and harmonizing data...")
    combined_data = load_and_harmonize_data(hdxer_path, jaxent_path)
    print(f"Loaded {len(combined_data)} total data points")
    print(f"Experiments: {combined_data['experiment'].unique()}")
    print(f"Ensembles: {combined_data['ensemble'].unique()}")
    print(f"Split types: {combined_data['split_type'].unique()}")
    print()
    
    # Generate plots
    print("Generating comparison plots...")
    print("-" * 80)
    print("NOTE: Only plotting Non-Redundant (R3) and Spatial (Sp) split types")
    print("-" * 80)
    
    print("\n1. Plotting comparison by experiment...")
    plot_comparison_by_experiment(combined_data, output_dir)
    
    print("\n2. Plotting comparison by ensemble...")
    plot_comparison_by_ensemble(combined_data, output_dir)
    
    print("\n3. Generating summary statistics...")
    plot_summary_statistics(combined_data, output_dir)
    
    # Print split type coverage by experiment (filtered to R3 and Sp only)
    print("\n" + "=" * 80)
    print("Split Type Coverage by Experiment (Non-Redundant & Spatial only):")
    print("=" * 80)
    analysis_split_types = ["R3", "Sp"]
    for experiment in sorted(combined_data['experiment'].unique()):
        exp_data = combined_data[
            (combined_data['experiment'] == experiment) & 
            (combined_data['split_type'].isin(analysis_split_types))
        ]
        if len(exp_data) > 0:
            split_types = sorted(exp_data['split_type'].unique())
            split_names = [SPLIT_NAME_MAPPING.get(st, st) for st in split_types]
            print(f"\n{experiment}:")
            print(f"  Split types: {', '.join(split_types)}")
            print(f"  Split names: {', '.join(split_names)}")
            print(f"  Total records: {len(exp_data)}")
        else:
            print(f"\n{experiment}: No data for R3/Sp splits")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()