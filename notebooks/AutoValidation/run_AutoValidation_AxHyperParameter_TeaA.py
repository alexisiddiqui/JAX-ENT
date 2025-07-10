#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter tuning script for TeaA autovalidation model using Ax.
This script tunes l2_weight, x_regularisation, and maxent_regularisation
parameters to find the optimal configuration.
"""

import datetime
import json
import os
from typing import List

import jax
import matplotlib.pyplot as plt
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties

# Import from the main script
from jaxENT_AX_QuickTrial_run_TeaA import (
    closed_path,
    open_path,
    output_dir,
    setup_experiment,
    setup_experiment_complete,
    topology_path,
    train_evaluate,
    trajectory_path,
)

output_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/TeaA_complete/quick_AV_HyperParameterTuning"


def create_timestamped_directory():
    """Create a directory with timestamp for storing results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"hyperparameter_tuning_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def run_hyperparameter_tuning(
    num_trials: int = 20,
    num_seeds: int = 3,
    reference_paths: List[str] = None,
    complete: bool = False,
):
    """
    Run hyperparameter tuning using Ax.

    Args:
        num_trials: Number of hyperparameter configurations to try
        num_seeds: Number of seeds to use for each configuration
        reference_paths: Paths to reference structures

    Returns:
        Best parameters and experiment results
    """
    if reference_paths is None:
        reference_paths = [open_path, closed_path]

    # Create directory for results
    result_dir = create_timestamped_directory()
    print(f"Results will be saved to: {result_dir}")

    # Setup the experiment
    print("Setting up experiment...")
    if not complete:
        (
            datasets,
            models,
            opt_settings,
            features,
            bv_config,
            synthetic_data,
            rmsd_values,
            pairwise_similarity,
            cluster_assignments,
        ) = setup_experiment(
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            open_path=open_path,
            closed_path=closed_path,
            seeds=num_seeds,
        )
    elif complete:
        (
            datasets,
            models,
            opt_settings,
            features,
            bv_config,
            synthetic_data,
            rmsd_values,
            pairwise_similarity,
            cluster_assignments,
        ) = setup_experiment_complete(
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            open_path=open_path,
            closed_path=closed_path,
            seeds=num_seeds,
        )

    # Define Ax client for hyperparameter optimization
    ax_client = AxClient(random_seed=42)
    ax_client.create_experiment(
        name="TeaA_hyperparameter_tuning",
        parameters=[
            {
                "name": "l2_weight",
                "type": "range",
                "bounds": [1e-2, 1e2],
                "log_scale": True,
            },
            {
                "name": "x_regularisation",
                "type": "range",
                "bounds": [1e0, 1e3],
                "log_scale": True,
            },
            {
                "name": "maxent_regularisation",
                "type": "range",
                "bounds": [1e-1, 1e2],
                "log_scale": True,
            },
        ],
        objectives={"mse": ObjectiveProperties(minimize=True)},  # arameter
    )

    results = []

    # Evaluation function for Ax
    def evaluate_parameters(parameters):
        print(f"\nTrial parameters: {parameters}")
        jax.clear_caches()

        # Create a trial directory
        trial_idx = len(results)
        trial_dir = os.path.join(result_dir, f"trial_{trial_idx}")
        os.makedirs(trial_dir, exist_ok=True)

        # Save parameter configuration
        with open(os.path.join(trial_dir, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=2)

        try:
            # Run evaluation with these parameters
            mse, sem = train_evaluate(
                output_dir=trial_dir,
                rmsd_values=rmsd_values,
                cluster_assignments=cluster_assignments,
                datasets=datasets,
                bv_config=bv_config,
                pairwise_similarity=pairwise_similarity,
                models=models,
                features=features,
                optimizer="adamw",
                l2_weight=parameters["l2_weight"],
                x_regularisation=parameters["x_regularisation"],
                maxent_regularisation=parameters["maxent_regularisation"],
            )

            # Save results
            result = {
                "trial": trial_idx,
                "parameters": parameters,
                "mse": float(mse),
                "sem": float(sem),
            }
            results.append(result)

            with open(os.path.join(trial_dir, "results.json"), "w") as f:
                json.dump(result, f, indent=2)

            # Clear JAX caches to prevent memory buildup
            jax.clear_caches()

            print(f"Trial {trial_idx} - MSE: {mse:.6f}, SEM: {sem:.6f}")
            return {"mse": (mse, sem)}

        except Exception as e:
            print(f"Error in trial {trial_idx}: {e}")
            # Return a high MSE value on error
            return {"mse": (1e10, 0.0)}

    # Run the trials
    for i in range(num_trials):
        print(f"\n=== Running Trial {i + 1}/{num_trials} ===")
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate_parameters(parameters))

    # Get best parameters
    best_parameters, values = ax_client.get_best_parameters()
    best_mse = values[0]["mse"]
    print(f"\nBest parameters: {best_parameters}")
    print(f"Best MSE: {best_mse:.6f}")

    # Save best parameters
    with open(os.path.join(result_dir, "best_parameters.json"), "w") as f:
        json.dump(
            {
                "parameters": best_parameters,
                "mse": best_mse,
            },
            f,
            indent=2,
        )

    # Save all results
    with open(os.path.join(result_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate and save plots
    generate_plots(ax_client, result_dir)

    return best_parameters, ax_client


def generate_plots(ax_client, result_dir):
    """Generate and save visualization plots using direct matplotlib plotting."""

    # Create plots directory
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Get trials data from experiment
        trials_df = pd.DataFrame(
            [
                {**t.arm.parameters, "mse": t.objective_mean, "trial_index": t.index}
                for t in ax_client.experiment.trials.values()
                if t.status.is_completed
            ]
        )

        if len(trials_df) < 3:
            print("Not enough completed trials for meaningful plots")
            return

        # Create parameter pair plots
        param_pairs = [
            ("l2_weight", "x_regularisation"),
            ("l2_weight", "maxent_regularisation"),
            ("x_regularisation", "maxent_regularisation"),
        ]

        for param_x, param_y in param_pairs:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot with color representing MSE
            scatter = ax.scatter(
                trials_df[param_x],
                trials_df[param_y],
                c=trials_df["mse"],
                cmap="cool",
                s=100,
                alpha=0.7,
            )

            # Add colorbar
            cbar = fig.colorbar(scatter)
            cbar.set_label("MSE (lower is better)")

            # Mark the best point
            best_idx = trials_df["mse"].idxmin()
            best_x = trials_df.loc[best_idx, param_x]
            best_y = trials_df.loc[best_idx, param_y]
            ax.scatter(best_x, best_y, s=200, facecolors="none", edgecolors="r", linewidth=2)

            # Set logarithmic scales
            ax.set_xscale("log")
            ax.set_yscale("log")

            # Add labels and title
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_title(f"Hyperparameter Optimization: {param_x} vs {param_y}")

            # Add trial indices as labels
            for i, row in trials_df.iterrows():
                ax.annotate(
                    f"{int(row['trial_index'])}",
                    (row[param_x], row[param_y]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            # Add grid
            ax.grid(True, which="both", ls="--", alpha=0.3)

            # Save figure
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, f"scatter_{param_x}_vs_{param_y}.png"), dpi=300)
            plt.close(fig)

        # Create individual parameter plots
        params = ["l2_weight", "x_regularisation", "maxent_regularisation"]
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        for i, param in enumerate(params):
            # Sort by parameter value
            sorted_df = trials_df.sort_values(by=param)

            # Plot parameter vs MSE
            axs[i].plot(sorted_df[param], sorted_df["mse"], "o-", markersize=8)
            axs[i].set_xscale("log")
            axs[i].set_title(f"MSE vs {param}")
            axs[i].set_xlabel(param)
            axs[i].set_ylabel("MSE")
            axs[i].grid(True, which="both", ls="--", alpha=0.3)

            # Mark best point
            best_idx = sorted_df["mse"].idxmin()
            best_x = sorted_df.loc[best_idx, param]
            best_y = sorted_df.loc[best_idx, "mse"]
            axs[i].scatter(best_x, best_y, s=150, facecolors="none", edgecolors="r", linewidth=2)

            # Add annotation for best value
            axs[i].annotate(
                f"Best: {best_x:.4g}",
                (best_x, best_y),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "parameter_trends.png"), dpi=300)
        plt.close(fig)

        # Create a summary results table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")

        # Sort by MSE (best to worst)
        sorted_df = trials_df.sort_values(by="mse").head(10)
        table_data = sorted_df[
            ["trial_index", "l2_weight", "x_regularisation", "maxent_regularisation", "mse"]
        ]

        # Create table
        table = ax.table(
            cellText=table_data.values.round(6),
            colLabels=table_data.columns,
            loc="center",
            cellLoc="center",
        )

        # Set table properties
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Highlight best row
        for key, cell in table.get_celld().items():
            if key[0] == 1:  # First row after header
                cell.set_facecolor("#d4f7d4")

        ax.set_title("Top 10 Hyperparameter Configurations (Sorted by MSE)", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "best_parameters_table.png"), dpi=300)
        plt.close(fig)

        # Save the full results table as CSV
        trials_df.to_csv(os.path.join(plots_dir, "all_trials_results.csv"), index=False)

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()

    plt.close("all")


if __name__ == "__main__":
    print("Starting TeaA model hyperparameter tuning...")
    best_parameters, ax_client = run_hyperparameter_tuning(
        complete=True,
        num_trials=20,
        num_seeds=5,
    )

    print("\nHyperparameter tuning completed!")
    print(f"Best parameters: {best_parameters}")
