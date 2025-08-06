"""
This script loads the optimization histories of the ISO TRI and BI models over both loss functions
and performs some standard analyses.
It generates plots to visualize the training and validation losses over the variance convergence thresholds:
 convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
It also plots the standard deviation across splits

Updated to handle multiple split types like the ratio recovery script.
"""

import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

# Import the HDF5 loading functions from the provided script
from jaxent.src.utils.hdf import load_optimization_history_from_file


def load_all_optimization_results(
    results_dir: str,
    split_type: str = None,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    num_splits: int = 3,
) -> Dict:
    """
    Load all optimization results from HDF5 files for a specific split type.

    Args:
        results_dir: Directory containing the HDF5 result files
        split_type: Name of the split type subdirectory (if None, loads from results_dir directly)
        ensembles: List of ensemble names
        loss_functions: List of loss function names
        num_splits: Number of data splits

    Returns:
        Dictionary containing loaded optimization histories organized by ensemble, loss, and split
    """
    results = {}

    # Determine the actual directory to load from
    if split_type:
        load_dir = os.path.join(results_dir, split_type)
    else:
        load_dir = results_dir

    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return results

    for ensemble in ensembles:
        results[ensemble] = {}

        for loss_name in loss_functions:
            results[ensemble][loss_name] = {}

            for split_idx in range(num_splits):
                if split_type:
                    filename = (
                        f"{ensemble}_{loss_name}_{split_type}_split{split_idx:03d}_results.hdf5"
                    )
                else:
                    filename = f"{ensemble}_{loss_name}_split{split_idx:03d}_results.hdf5"
                filepath = os.path.join(load_dir, filename)

                if os.path.exists(filepath):
                    try:
                        history = load_optimization_history_from_file(filepath)
                        results[ensemble][loss_name][split_idx] = history
                        print(f"Loaded: {filename}")
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
                        results[ensemble][loss_name][split_idx] = None
                else:
                    print(f"File not found: {filename}")
                    results[ensemble][loss_name][split_idx] = None

    return results


def extract_loss_trajectories(results: Dict, split_type: str = None) -> pd.DataFrame:
    """
    Extract loss trajectories from optimization results.

    Args:
        results: Dictionary containing optimization histories
        split_type: Name of the split type (for labeling)

    Returns:
        DataFrame containing loss trajectories for analysis
    """
    data_rows = []

    for ensemble in results:
        for loss_name in results[ensemble]:
            for split_idx in results[ensemble][loss_name]:
                history = results[ensemble][loss_name][split_idx]

                if history is not None and history.states:
                    for step_idx, state in enumerate(history.states):
                        if state.losses is not None:
                            data_rows.append(
                                {
                                    "ensemble": ensemble,
                                    "loss_function": loss_name,
                                    "split": split_idx,
                                    "split_type": split_type,
                                    "step": step_idx,
                                    "convergence_threshold_step": step_idx,  # Each saved state represents a convergence threshold
                                    "train_loss": float(state.losses.total_train_loss),
                                    "val_loss": float(state.losses.total_val_loss),
                                    "step_number": state.step,
                                }
                            )

    return pd.DataFrame(data_rows)


def plot_loss_convergence(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot error vs convergence with training and validation error separate.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}  # Blue and Orange
    loss_markers = {"mcMSE": "o", "MSE": "s"}  # Circle and Square

    # Create separate plots for training and validation errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Error vs Convergence Threshold{title_suffix}", fontsize=16, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    # Plot training errors
    ax = ax1
    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination and exclude step 0 (pre-optimization)
            subset = df[
                (df["ensemble"] == ensemble)
                & (df["loss_function"] == loss_func)
                & (df["convergence_threshold_step"] > 0)  # Skip pre-optimization step
            ]

            if len(subset) > 0:
                # Calculate mean and std across splits for each convergence step
                stats = (
                    subset.groupby("convergence_threshold_step")
                    .agg({"train_loss": ["mean", "std"]})
                    .reset_index()
                )

                # Flatten column names
                stats.columns = ["step", "train_mean", "train_std"]

                # Map steps to convergence rates (step 1 -> convergence_rates[0], step 2 -> convergence_rates[1], etc.)
                stats["convergence_rate"] = stats["step"].apply(
                    lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                )

                # Remove rows where convergence rate mapping failed
                stats = stats.dropna(subset=["convergence_rate"])

                if len(stats) > 0:
                    # Plot training loss
                    color = ensemble_colors[ensemble]
                    marker = loss_markers[loss_func]
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["convergence_rate"],
                        stats["train_mean"],
                        yerr=stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        capsize=3,
                        markersize=6,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold")
    ax.set_ylabel("Training Error")
    ax.set_title("Training Error vs Convergence")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot validation errors
    ax = ax2
    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination and exclude step 0 (pre-optimization)
            subset = df[
                (df["ensemble"] == ensemble)
                & (df["loss_function"] == loss_func)
                & (df["convergence_threshold_step"] > 0)  # Skip pre-optimization step
            ]

            if len(subset) > 0:
                # Calculate mean and std across splits for each convergence step
                stats = (
                    subset.groupby("convergence_threshold_step")
                    .agg({"val_loss": ["mean", "std"]})
                    .reset_index()
                )

                # Flatten column names
                stats.columns = ["step", "val_mean", "val_std"]

                # Map steps to convergence rates (step 1 -> convergence_rates[0], step 2 -> convergence_rates[1], etc.)
                stats["convergence_rate"] = stats["step"].apply(
                    lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                )

                # Remove rows where convergence rate mapping failed
                stats = stats.dropna(subset=["convergence_rate"])

                if len(stats) > 0:
                    # Plot validation loss
                    color = ensemble_colors[ensemble]
                    marker = loss_markers[loss_func]
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["convergence_rate"],
                        stats["val_mean"],
                        yerr=stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        capsize=3,
                        markersize=6,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold")
    ax.set_ylabel("Validation Error")
    ax.set_title("Validation Error vs Convergence")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = (
        f"error_vs_convergence_{split_type}.png" if split_type else "error_vs_convergence.png"
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_split_variability(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot standard deviation across splits for each convergence threshold.
    Organized the same way as error convergence plots with training and validation separate.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers (same as main plots)
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}  # Blue and Orange
    loss_markers = {"mcMSE": "o", "MSE": "s"}  # Circle and Square

    # Create separate plots for training and validation standard deviations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Standard Deviation Across Splits{title_suffix}", fontsize=16, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    # Plot training loss standard deviations
    ax = ax1
    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination and exclude step 0 (pre-optimization)
            subset = df[
                (df["ensemble"] == ensemble)
                & (df["loss_function"] == loss_func)
                & (df["convergence_threshold_step"] > 0)  # Skip pre-optimization step
            ]

            if len(subset) > 0:
                # Calculate std across splits for each convergence step
                std_stats = (
                    subset.groupby("convergence_threshold_step")
                    .agg({"train_loss": "std"})
                    .reset_index()
                )

                # Flatten column names
                std_stats.columns = ["step", "train_std"]

                # Map steps to convergence rates (step 1 -> convergence_rates[0], step 2 -> convergence_rates[1], etc.)
                std_stats["convergence_rate"] = std_stats["step"].apply(
                    lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                )

                # Remove rows where convergence rate mapping failed
                std_stats = std_stats.dropna(subset=["convergence_rate"])

                if len(std_stats) > 0:
                    # Plot training loss standard deviation
                    color = ensemble_colors[ensemble]
                    marker = loss_markers[loss_func]
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        markersize=6,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold")
    ax.set_ylabel("Training Error Std Dev")
    ax.set_title("Training Error Standard Deviation")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Plot validation loss standard deviations
    ax = ax2
    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination and exclude step 0 (pre-optimization)
            subset = df[
                (df["ensemble"] == ensemble)
                & (df["loss_function"] == loss_func)
                & (df["convergence_threshold_step"] > 0)  # Skip pre-optimization step
            ]

            if len(subset) > 0:
                # Calculate std across splits for each convergence step
                std_stats = (
                    subset.groupby("convergence_threshold_step")
                    .agg({"val_loss": "std"})
                    .reset_index()
                )

                # Flatten column names
                std_stats.columns = ["step", "val_std"]

                # Map steps to convergence rates (step 1 -> convergence_rates[0], step 2 -> convergence_rates[1], etc.)
                std_stats["convergence_rate"] = std_stats["step"].apply(
                    lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                )

                # Remove rows where convergence rate mapping failed
                std_stats = std_stats.dropna(subset=["convergence_rate"])

                if len(std_stats) > 0:
                    # Plot validation loss standard deviation
                    color = ensemble_colors[ensemble]
                    marker = loss_markers[loss_func]
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        markersize=6,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold")
    ax.set_ylabel("Validation Error Std Dev")
    ax.set_title("Validation Error Standard Deviation")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"split_variability_{split_type}.png" if split_type else "split_variability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_performance_comparison(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Plot comparison of final performance across ensembles and loss functions.

    Args:
        df: DataFrame containing loss data
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Get final convergence step data (last step for each combination)
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Final Performance Comparison{title_suffix}", fontsize=16, fontweight="bold")

    # Training loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="train_loss", hue="loss_function", ax=ax1)
    ax1.set_yscale("log")
    ax1.set_title("Final Training Loss")
    ax1.set_ylabel("Training Loss (log scale)")

    # Validation loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="val_loss", hue="loss_function", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("Final Validation Loss")
    ax2.set_ylabel("Validation Loss (log scale)")

    plt.tight_layout()
    filename = (
        f"final_performance_comparison_{split_type}.png"
        if split_type
        else "final_performance_comparison.png"
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Generate summary statistics and save to CSV.

    Args:
        df: DataFrame containing loss data
        output_dir: Directory to save summary
        split_type: Name of the split type (for filename)
    """
    # Summary statistics for final performance
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    summary = (
        final_data.groupby(["ensemble", "loss_function"])
        .agg(
            {"train_loss": ["mean", "std", "min", "max"], "val_loss": ["mean", "std", "min", "max"]}
        )
        .round(6)
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save to CSV
    filename = f"summary_statistics_{split_type}.csv" if split_type else "summary_statistics.csv"
    summary_path = os.path.join(output_dir, filename)
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary to console
    print(f"\nSummary Statistics for {split_type if split_type else 'All Data'}:")
    print("=" * 80)
    print(summary)

    return summary


def plot_cross_split_type_comparison(
    all_data: pd.DataFrame, convergence_rates: List[float], output_dir: str
):
    """
    Plot comparison across different split types.

    Args:
        all_data: DataFrame containing data from all split types
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors for split types
    split_type_colors = plt.cm.Set1.colors  # Use qualitative colormap
    split_types = sorted(all_data["split_type"].unique())
    color_map = {
        split_type: split_type_colors[i % len(split_type_colors)]
        for i, split_type in enumerate(split_types)
    }

    # Create plots for final performance comparison across split types
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Performance Comparison Across Split Types", fontsize=16, fontweight="bold")

    # Get final convergence step data
    final_data = (
        all_data.groupby(["split_type", "ensemble", "loss_function", "split"]).last().reset_index()
    )

    # Training loss by split type and ensemble
    sns.boxplot(data=final_data, x="split_type", y="train_loss", hue="ensemble", ax=ax1)
    ax1.set_yscale("log")
    ax1.set_title("Final Training Loss by Split Type and Ensemble")
    ax1.set_ylabel("Training Loss (log scale)")
    ax1.tick_params(axis="x", rotation=45)

    # Validation loss by split type and ensemble
    sns.boxplot(data=final_data, x="split_type", y="val_loss", hue="ensemble", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("Final Validation Loss by Split Type and Ensemble")
    ax2.set_ylabel("Validation Loss (log scale)")
    ax2.tick_params(axis="x", rotation=45)

    # Training loss by split type and loss function
    sns.boxplot(data=final_data, x="split_type", y="train_loss", hue="loss_function", ax=ax3)
    ax3.set_yscale("log")
    ax3.set_title("Final Training Loss by Split Type and Loss Function")
    ax3.set_ylabel("Training Loss (log scale)")
    ax3.tick_params(axis="x", rotation=45)

    # Validation loss by split type and loss function
    sns.boxplot(data=final_data, x="split_type", y="val_loss", hue="loss_function", ax=ax4)
    ax4.set_yscale("log")
    ax4.set_title("Final Validation Loss by Split Type and Loss Function")
    ax4.set_ylabel("Validation Loss (log scale)")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "cross_split_type_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Also create convergence comparison across split types
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Convergence Comparison Across Split Types", fontsize=16, fontweight="bold")

    # For simplicity, show ISO_TRI with mcMSE across split types
    ensemble_filter = "ISO_TRI"
    loss_filter = "mcMSE"

    for split_type in split_types:
        subset = all_data[
            (all_data["split_type"] == split_type)
            & (all_data["ensemble"] == ensemble_filter)
            & (all_data["loss_function"] == loss_filter)
            & (all_data["convergence_threshold_step"] > 0)
        ]

        if len(subset) > 0:
            # Calculate mean across splits for each convergence step
            stats = (
                subset.groupby("convergence_threshold_step")
                .agg({"train_loss": "mean", "val_loss": "mean"})
                .reset_index()
            )

            # Map steps to convergence rates
            stats["convergence_rate"] = stats["convergence_threshold_step"].apply(
                lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
            )

            # Remove rows where convergence rate mapping failed
            stats = stats.dropna(subset=["convergence_rate"])

            if len(stats) > 0:
                color = color_map[split_type]

                # Training loss
                ax1.plot(
                    stats["convergence_rate"],
                    stats["train_loss"],
                    label=f"{split_type}",
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                # Validation loss
                ax2.plot(
                    stats["convergence_rate"],
                    stats["val_loss"],
                    label=f"{split_type}",
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Convergence Threshold")
    ax1.set_ylabel("Training Error")
    ax1.set_title(f"Training Error Convergence - {ensemble_filter} {loss_filter}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Convergence Threshold")
    ax2.set_ylabel("Validation Error")
    ax2.set_title(f"Validation Error Convergence - {ensemble_filter} {loss_filter}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "convergence_across_split_types.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """
    Main function to run the complete analysis with multiple split types.
    """
    # Define parameters (should match those used in the optimization script)
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    num_splits = 3
    # Remove the '0' convergence rate as it represents pre-optimization values
    convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # Define directories
    base_results_dir = "../fitting/jaxENT/_optimise"
    base_results_dir = os.path.join(os.path.dirname(__file__), base_results_dir)

    output_base_dir = "_analysis"
    output_base_dir = os.path.join(os.path.dirname(__file__), output_base_dir)

    # Check if results directory exists
    if not os.path.exists(base_results_dir):
        raise FileNotFoundError(f"Results directory not found: {base_results_dir}")

    print("Starting ISO Model Analysis with Multiple Split Types...")
    print(f"Base results directory: {base_results_dir}")
    print(f"Base output directory: {output_base_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print("-" * 60)

    # Detect split types
    split_types = [
        d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d))
    ]

    # If no subdirectories found, assume flat structure (backward compatibility)
    if not split_types:
        print("No split type subdirectories found. Using flat structure.")
        split_types = [None]

    print(f"Found split types: {split_types}")

    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Store all data for cross-split-type analysis
    all_split_data = []

    for split_type in split_types:
        if split_type:
            print(f"\n--- Analysing split type: {split_type} ---")
            output_dir = os.path.join(output_base_dir, split_type)
        else:
            print("\n--- Analysing results (flat structure) ---")
            output_dir = output_base_dir

        os.makedirs(output_dir, exist_ok=True)

        # Load optimization results for this split type
        print("Loading optimization results...")
        results = load_all_optimization_results(
            results_dir=base_results_dir,
            split_type=split_type,
            ensembles=ensembles,
            loss_functions=loss_functions,
            num_splits=num_splits,
        )

        # Extract loss trajectories
        print("Extracting loss trajectories...")
        df = extract_loss_trajectories(results, split_type)

        if len(df) == 0:
            print(f"No data found for split type {split_type}! Skipping.")
            continue

        print(f"Extracted {len(df)} data points from optimization histories")

        # Store data for cross-split-type analysis
        all_split_data.append(df)

        # Generate plots for this split type
        print("Generating error vs convergence plots...")
        plot_loss_convergence(df, convergence_rates, output_dir, split_type)

        print("Generating split variability plots...")
        plot_split_variability(df, convergence_rates, output_dir, split_type)

        print("Generating final performance comparison...")
        plot_final_performance_comparison(df, output_dir, split_type)

        # Generate summary statistics
        print("Generating summary statistics...")
        summary = generate_summary_statistics(df, output_dir, split_type)

        # Save the full dataset for this split type
        filename = (
            f"full_analysis_data_{split_type}.csv" if split_type else "full_analysis_data.csv"
        )
        df_path = os.path.join(output_dir, filename)
        df.to_csv(df_path, index=False)
        print(f"Dataset saved to: {df_path}")

        print(
            f"Analysis for {split_type if split_type else 'flat structure'} complete. Outputs saved to {output_dir}"
        )

    # Generate cross-split-type analysis if we have multiple split types
    if len(all_split_data) > 1 and split_types != [None]:
        print("\n--- Generating cross-split-type comparisons ---")

        # Combine all data
        combined_data = pd.concat(all_split_data, ignore_index=True)

        # Generate cross-split-type plots
        plot_cross_split_type_comparison(combined_data, convergence_rates, output_base_dir)

        # Save combined dataset
        combined_path = os.path.join(output_base_dir, "combined_analysis_data.csv")
        combined_data.to_csv(combined_path, index=False)
        print(f"Combined dataset saved to: {combined_path}")

    print("\nAnalysis completed successfully!")
    print(f"All outputs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()
