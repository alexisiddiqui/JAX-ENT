"""
compute_sigma_synthetic.py

Computes covariance matrices (sigma) for the synthetic data using clustering results.

Requirements:
    - Clustering results (_clustering_results/)
    - Featurized data (_featurise/)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \\
        --clustering_dir ... \\
        --features_dir ... \\
        --ensemble_name ISO_BI \\
        --output_dir ...

Output:
    - Sigma matrices in _covariance_matrices_sigma/
"""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import PTSerialiser
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.predict import run_predict


# --- Helper Functions ---
def plot_heatmap(
    matrix,
    title,
    filename,
    output_dir,
    cmap="viridis",
    annot=False,
    fmt=".2f",
    log_scale=False,
    eps=1e-12,
):
    plt.figure(figsize=(10, 8))
    if log_scale:
        matrix_to_plot = np.abs(np.array(matrix, dtype=float))
        matrix_to_plot[matrix_to_plot <= eps] = eps
        norm = LogNorm(vmin=matrix_to_plot.min(), vmax=matrix_to_plot.max())
        sns.heatmap(
            matrix_to_plot,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            norm=norm,
            cbar_kws={"label": "Value (log scale)"},
        )
    else:
        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, cbar_kws={"label": "Value"})
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_diagonal_bar(matrix, title, filename, output_dir, log_scale=False, eps=1e-12):
    diag = np.diag(matrix).astype(float)
    indices = np.arange(len(diag))
    plt.figure(figsize=(10, 4.5))
    if log_scale:
        diag_plot = np.abs(diag)
        diag_plot[diag_plot <= eps] = eps
        plt.bar(indices, diag_plot)
        plt.yscale("log")
        plt.ylabel("Absolute diagonal value (log scale)")
    else:
        plt.bar(indices, diag)
        plt.ylabel("Diagonal value")
    plt.xlabel("Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def compute_cluster_weights(cluster_assignments, target_ratios):
    """
    Compute frame weights to achieve target cluster ratios.

    Args:
        cluster_assignments (np.ndarray): Cluster assignments (0=open, 1=closed, -1=unclustered)
        target_ratios (dict): Target ratios {'open': 0.4, 'closed': 0.6}

    Returns:
        np.ndarray: Normalized frame weights
    """
    n_frames = len(cluster_assignments)
    frame_weights = np.zeros(n_frames)

    # Count frames in each cluster
    open_mask = cluster_assignments == 0
    closed_mask = cluster_assignments == 1
    unclustered_mask = cluster_assignments == -1

    n_open = np.sum(open_mask)
    n_closed = np.sum(closed_mask)
    n_unclustered = np.sum(unclustered_mask)

    print(f"  Cluster counts - Open: {n_open}, Closed: {n_closed}, Unclustered: {n_unclustered}")

    # Compute weights for each cluster
    target_open = target_ratios["open"]
    target_closed = target_ratios["closed"]

    if n_open > 0:
        frame_weights[open_mask] = target_open / n_open
    if n_closed > 0:
        frame_weights[closed_mask] = target_closed / n_closed

    # Distribute remaining weight to unclustered frames
    if n_unclustered > 0:
        remaining_weight = 1.0 - (target_open + target_closed)
        if remaining_weight > 0:
            frame_weights[unclustered_mask] = remaining_weight / n_unclustered
        else:
            # If target ratios sum to 1, don't weight unclustered frames
            frame_weights[unclustered_mask] = 0.0

    # Normalize weights to sum to 1
    total_weight = np.sum(frame_weights)
    if total_weight > 0:
        frame_weights = frame_weights / total_weight

    return frame_weights


def compute_weighted_covariance(data, weights):
    """
    Compute weighted covariance matrix.

    Args:
        data (np.ndarray): Data matrix (n_variables, n_observations)
        weights (np.ndarray): Weights for each observation (n_observations,)

    Returns:
        np.ndarray: Weighted covariance matrix (n_variables, n_variables)
    """
    # Ensure weights are normalized
    weights = weights / np.sum(weights)

    # Compute weighted mean
    weighted_mean = np.sum(data * weights[np.newaxis, :], axis=1, keepdims=True)

    # Center the data
    centered_data = data - weighted_mean

    # Compute weighted covariance
    weighted_cov = (centered_data * weights[np.newaxis, :]) @ centered_data.T

    return weighted_cov


def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted Sigma covariance matrices from clustering and ensemble predictions."
    )

    script_dir = os.path.dirname(__file__)

    # Default paths - matching existing scripts
    default_clustering_dir = os.path.join(script_dir, "../../data/_clustering_results")
    default_features_dir = os.path.join(script_dir, "_featurise")
    default_output_dir = os.path.join(script_dir, "_covariance_matrices_sigma")

    parser.add_argument(
        "--clustering_dir",
        type=str,
        default=default_clustering_dir,
        help="Directory containing clustering results.",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=default_features_dir,
        help="Directory containing featurized data.",
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        default="ISO_BI",
        help="Name of the ensemble to use (default: ISO_BI).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save the output weighted Sigma matrices and plots. Defaults to '_covariance_matrices'.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("WEIGHTED SIGMA COMPUTATION")
    print("=" * 80)

    # --- Load Clustering Results ---
    print(f"\n--- Loading Clustering Results for {args.ensemble_name} ---")
    cluster_file = os.path.join(
        args.clustering_dir, f"cluster_assignments_{args.ensemble_name}.csv"
    )

    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Clustering file not found: {cluster_file}")

    cluster_df = pd.read_csv(cluster_file)
    cluster_assignments = cluster_df["cluster_assignment"].values
    n_frames = len(cluster_assignments)

    print(f"  Loaded {n_frames} cluster assignments from {cluster_file}")

    # --- Compute Cluster Weights ---
    print("\n--- Computing Cluster-Based Weights ---")
    target_ratios = {"open": 0.4, "closed": 0.6}
    print(
        f"  Target ratios - Open: {target_ratios['open']:.1%}, Closed: {target_ratios['closed']:.1%}"
    )

    frame_weights = compute_cluster_weights(cluster_assignments, target_ratios)

    # Verify achieved ratios
    open_mask = cluster_assignments == 0
    closed_mask = cluster_assignments == 1
    achieved_open = np.sum(frame_weights[open_mask])
    achieved_closed = np.sum(frame_weights[closed_mask])

    print(f"  Achieved ratios - Open: {achieved_open:.1%}, Closed: {achieved_closed:.1%}")

    # Save weights
    weights_df = pd.DataFrame(
        {
            "frame": np.arange(n_frames),
            "cluster_assignment": cluster_assignments,
            "frame_weight": frame_weights,
        }
    )
    weights_path = os.path.join(args.output_dir, f"{args.ensemble_name}_frame_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    print(f"  Saved frame weights to: {weights_path}")

    # Plot weight distribution
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(n_frames), frame_weights, width=1.0, edgecolor="none")
    plt.xlabel("Frame")
    plt.ylabel("Weight")
    plt.title(
        f"{args.ensemble_name} Frame Weights (Open: {achieved_open:.1%}, Closed: {achieved_closed:.1%})"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.ensemble_name}_frame_weights.png"))
    plt.close()

    # --- Load Features and Topology ---
    print(f"\n--- Loading Features and Topology for {args.ensemble_name} ---")
    topology_file = os.path.join(args.features_dir, f"topology_{args.ensemble_name.lower()}.json")
    features_file = os.path.join(args.features_dir, f"features_{args.ensemble_name.lower()}.npz")

    if not os.path.exists(topology_file):
        raise FileNotFoundError(f"Topology file not found: {topology_file}")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")

    topology = PTSerialiser.load_list_from_json(topology_file)
    features = BV_input_features.load(features_file)

    print(f"  Loaded topology with {len(topology)} peptides")
    print(f"  Loaded features with shape: {features.features_shape}")

    # --- Initialize BV_model ---
    print("\n--- Initializing BV_model ---")
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params

    print(f"  BV_model initialized with {len(bv_config.timepoints)} timepoints")

    # --- Predict Uptake for All Frames ---
    print("\n--- Predicting HDX Uptake for All Frames ---")

    # Create dummy Simulation_Parameters
    dummy_sim_params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    predictions_output_features = run_predict(
        input_features=[features],
        forward_models=[bv_model],
        model_parameters=dummy_sim_params,
        validate=False,
    )

    # Extract predictions: (num_timepoints, num_peptides, num_frames)
    y_pred_all_frames = predictions_output_features[0].y_pred()
    print(f"  Prediction shape: {y_pred_all_frames.shape}")

    # Average across timepoints to get (num_peptides, num_frames)
    y_pred_avg_timepoints = np.array(np.mean(y_pred_all_frames, axis=0))
    print(f"  Time-averaged prediction shape: {y_pred_avg_timepoints.shape}")

    # --- Compute Unweighted Covariance ---
    print("\n--- Computing Unweighted Sigma ---")
    Sigma_unweighted = np.cov(y_pred_avg_timepoints) + np.diag(
        np.full(y_pred_avg_timepoints.shape[0], 1e-6)
    )

    print(f"  Unweighted Sigma shape: {Sigma_unweighted.shape}")

    plot_heatmap(
        Sigma_unweighted,
        f"{args.ensemble_name} Unweighted Sigma",
        f"{args.ensemble_name}_Sigma_unweighted_heatmap.png",
        args.output_dir,
    )
    plot_heatmap(
        np.linalg.inv(Sigma_unweighted),
        f"{args.ensemble_name} Inverse Unweighted Sigma",
        f"{args.ensemble_name}_Sigma_unweighted_inv_heatmap.png",
        args.output_dir,
        cmap="magma",
        log_scale=True,
    )
    plot_diagonal_bar(
        Sigma_unweighted,
        f"{args.ensemble_name} Unweighted Sigma Diagonal",
        f"{args.ensemble_name}_Sigma_unweighted_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_unweighted.npz"),
        Sigma=Sigma_unweighted,
        Sigma_inv=np.linalg.inv(Sigma_unweighted),
    )

    # --- Compute Weighted Covariance ---
    print("\n--- Computing Weighted Sigma (Cluster-Based) ---")
    Sigma_weighted = compute_weighted_covariance(y_pred_avg_timepoints, frame_weights)

    # Add small diagonal for numerical stability
    Sigma_weighted += np.diag(np.full(Sigma_weighted.shape[0], 1e-6))

    print(f"  Weighted Sigma shape: {Sigma_weighted.shape}")

    plot_heatmap(
        Sigma_weighted,
        f"{args.ensemble_name} Weighted Sigma (40:60 Open:Closed)",
        f"{args.ensemble_name}_Sigma_weighted_heatmap.png",
        args.output_dir,
    )
    plot_heatmap(
        np.linalg.inv(Sigma_weighted),
        f"{args.ensemble_name} Inverse Weighted Sigma",
        f"{args.ensemble_name}_Sigma_weighted_inv_heatmap.png",
        args.output_dir,
        cmap="magma",
        log_scale=True,
    )
    plot_diagonal_bar(
        Sigma_weighted,
        f"{args.ensemble_name} Weighted Sigma Diagonal",
        f"{args.ensemble_name}_Sigma_weighted_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_weighted.npz"),
        Sigma=Sigma_weighted,
        Sigma_inv=np.linalg.inv(Sigma_weighted),
        frame_weights=frame_weights,
        cluster_assignments=cluster_assignments,
        target_ratios=target_ratios,
        achieved_ratios={"open": achieved_open, "closed": achieved_closed},
    )

    print(f"  Weighted Sigma computed and saved to: {args.output_dir}")

    # --- Compute Difference Between Weighted and Unweighted ---
    print("\n--- Computing Difference Matrix ---")
    Sigma_diff = Sigma_weighted - Sigma_unweighted

    plot_heatmap(
        Sigma_diff,
        f"{args.ensemble_name} Sigma Difference (Weighted - Unweighted)",
        f"{args.ensemble_name}_Sigma_diff_heatmap.png",
        args.output_dir,
        cmap="RdBu_r",
    )
    plot_diagonal_bar(
        Sigma_diff,
        f"{args.ensemble_name} Sigma Difference Diagonal",
        f"{args.ensemble_name}_Sigma_diff_diagonal_bar.png",
        args.output_dir,
    )

    np.savez(
        os.path.join(args.output_dir, f"{args.ensemble_name}_Sigma_diff.npz"),
        Sigma_diff=Sigma_diff,
    )

    # --- Summary Statistics ---
    print("\n--- Summary Statistics ---")
    print("  Unweighted Sigma:")
    print(f"    Trace: {np.trace(Sigma_unweighted):.6f}")
    print(f"    Frobenius norm: {np.linalg.norm(Sigma_unweighted, 'fro'):.6f}")
    print("  Weighted Sigma:")
    print(f"    Trace: {np.trace(Sigma_weighted):.6f}")
    print(f"    Frobenius norm: {np.linalg.norm(Sigma_weighted, 'fro'):.6f}")
    print("  Difference:")
    print(f"    Max absolute difference: {np.max(np.abs(Sigma_diff)):.6f}")
    print(f"    Mean absolute difference: {np.mean(np.abs(Sigma_diff)):.6f}")

    # Save summary
    summary_data = {
        "ensemble": args.ensemble_name,
        "n_frames": n_frames,
        "n_peptides": y_pred_avg_timepoints.shape[0],
        "target_open_ratio": target_ratios["open"],
        "target_closed_ratio": target_ratios["closed"],
        "achieved_open_ratio": achieved_open,
        "achieved_closed_ratio": achieved_closed,
        "unweighted_trace": np.trace(Sigma_unweighted),
        "weighted_trace": np.trace(Sigma_weighted),
        "unweighted_frobenius": np.linalg.norm(Sigma_unweighted, "fro"),
        "weighted_frobenius": np.linalg.norm(Sigma_weighted, "fro"),
        "max_abs_diff": np.max(np.abs(Sigma_diff)),
        "mean_abs_diff": np.mean(np.abs(Sigma_diff)),
    }

    summary_df = pd.DataFrame([summary_data])
    summary_path = os.path.join(args.output_dir, f"{args.ensemble_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("WEIGHTED SIGMA COMPUTATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

