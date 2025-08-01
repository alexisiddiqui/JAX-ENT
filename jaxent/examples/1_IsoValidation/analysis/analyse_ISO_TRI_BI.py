"""
This script loads the optimization histories of the ISO TRI and BI models over both loss functions
and performs some standard analyses.
It generates plots to visualize the training and validation losses over the variance convergence thresholds:
 convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
It also plots the standard deviation across splits
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
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    num_splits: int = 3,
) -> Dict:
    """
    Load all optimization results from HDF5 files.

    Args:
        results_dir: Directory containing the HDF5 result files
        ensembles: List of ensemble names
        loss_functions: List of loss function names
        num_splits: Number of data splits

    Returns:
        Dictionary containing loaded optimization histories organized by ensemble, loss, and split
    """
    results = {}

    for ensemble in ensembles:
        results[ensemble] = {}

        for loss_name in loss_functions:
            results[ensemble][loss_name] = {}

            for split_idx in range(num_splits):
                filename = f"{ensemble}_{loss_name}_split{split_idx:03d}_results.hdf5"
                filepath = os.path.join(results_dir, filename)

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


def extract_loss_trajectories(results: Dict) -> pd.DataFrame:
    """
    Extract loss trajectories from optimization results.

    Args:
        results: Dictionary containing optimization histories

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
                                    "step": step_idx,
                                    "convergence_threshold_step": step_idx,  # Each saved state represents a convergence threshold
                                    "train_loss": float(state.losses.total_train_loss),
                                    "val_loss": float(state.losses.total_val_loss),
                                    "step_number": state.step,
                                }
                            )

    return pd.DataFrame(data_rows)


def plot_loss_convergence(df: pd.DataFrame, convergence_rates: List[float], output_dir: str):
    """
    Plot error vs convergence with training and validation error separate.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}  # Blue and Orange
    loss_markers = {"mcMSE": "o", "MSE": "s"}  # Circle and Square

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create separate plots for training and validation errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Error vs Convergence Threshold", fontsize=16, fontweight="bold")

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
    plt.savefig(os.path.join(output_dir, "error_vs_convergence.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_split_variability(df: pd.DataFrame, convergence_rates: List[float], output_dir: str):
    """
    Plot standard deviation across splits for each convergence threshold.
    Organized the same way as error convergence plots with training and validation separate.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers (same as main plots)
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}  # Blue and Orange
    loss_markers = {"mcMSE": "o", "MSE": "s"}  # Circle and Square

    # Create separate plots for training and validation standard deviations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Standard Deviation Across Splits", fontsize=16, fontweight="bold")

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
    plt.savefig(os.path.join(output_dir, "split_variability.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_final_performance_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot comparison of final performance across ensembles and loss functions.

    Args:
        df: DataFrame containing loss data
        output_dir: Directory to save plots
    """
    # Get final convergence step data (last step for each combination)
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Final Performance Comparison", fontsize=16, fontweight="bold")

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
    plt.savefig(
        os.path.join(output_dir, "final_performance_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def generate_summary_statistics(df: pd.DataFrame, output_dir: str):
    """
    Generate summary statistics and save to CSV.

    Args:
        df: DataFrame containing loss data
        output_dir: Directory to save summary
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
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary to console
    print("\nSummary Statistics:")
    print("=" * 80)
    print(summary)

    return summary


def main():
    """
    Main function to run the complete analysis.
    """
    # Define parameters (should match those used in the optimization script)
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    num_splits = 3
    # Remove the '0' convergence rate as it represents pre-optimization values
    convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # Define directories
    results_dir = "../fitting/jaxENT/_optimise"
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    output_dir = "_analysis"
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Check if results directory exists
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print("Starting ISO Model Analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print("-" * 60)

    # Load all optimization results
    print("Loading optimization results...")
    results = load_all_optimization_results(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        num_splits=num_splits,
    )

    # Extract loss trajectories
    print("Extracting loss trajectories...")
    df = extract_loss_trajectories(results)

    if len(df) == 0:
        print("No data found! Check that the result files exist and are properly formatted.")
        return

    print(f"Extracted {len(df)} data points from optimization histories")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("Generating error vs convergence plots...")
    plot_loss_convergence(df, convergence_rates, output_dir)

    print("Generating split variability plots...")
    plot_split_variability(df, convergence_rates, output_dir)

    print("Generating final performance comparison...")
    plot_final_performance_comparison(df, output_dir)

    # Generate summary statistics
    print("Generating summary statistics...")
    summary = generate_summary_statistics(df, output_dir)

    # Save the full dataset for further analysis
    df_path = os.path.join(output_dir, "full_analysis_data.csv")
    df.to_csv(df_path, index=False)
    print(f"Full dataset saved to: {df_path}")

    print("\nAnalysis completed successfully!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
