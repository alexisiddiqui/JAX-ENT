"""
MoPrP Score Models Script

This script loads the outputs of MoPrP_Process_Optimisation_results.py and computes various scores and metrics.
This includes error metrics: MSE(train/val/test), dMSE(train/val/test),
and Work Metrics: Shape, Density, Fitting, Scale, and KL Divergence(weights|uniform)

As the target data is already split into train/val (test is the entire dataset), the error metrics are computed 
by applying the train/val sparse mapping to the predicted and prior uptake data.

Work metrics are computed based on the entire set of predicted and prior lnPF values.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add the base directory to the path to import JAX-ENT modules
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Datapoint, ExpD_Dataloader
import jaxent.src.interfaces.topology as pt
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.models.HDX.BV.features import BV_input_features

# Mapping of ensemble names to feature file names (case-sensitive for MoPrP)
ENSEMBLE_FEATURE_MAP = {
    "AF2_MSAss": "AF2_MSAss",
    "AF2_filtered": "AF2_filtered",
}

# Ensemble to clustering subdirectory mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": ("AF2_MSAss", "AF2_MSAss_frame_to_cluster.csv"),
    "AF2_filtered": ("AF2_Filtered", "AF2_Filtered_frame_to_cluster.csv"),
}

# --- Data Loading Helpers ---
def load_experimental_data(
    datasplit_dir: str, split_type: str, split_idx: str
) -> Tuple[List[HDX_peptide], List[HDX_peptide], List[HDX_peptide]]:
    """
    Load train, validation, and full (test) experimental data for a given split.
    """
    split_path = os.path.join(datasplit_dir, split_type, f"split_{split_idx}")
    
    # If exact match not found, try integer match (e.g. split_002 vs split_2)
    if not os.path.exists(split_path):
        try:
            split_path_int = os.path.join(datasplit_dir, split_type, f"split_{int(split_idx)}")
            if os.path.exists(split_path_int):
                split_path = split_path_int
        except ValueError:
            pass

    if not os.path.exists(split_path):
        # Try checking if the split_type directory even exists
        type_dir = os.path.join(datasplit_dir, split_type)
        if not os.path.exists(type_dir):
             raise FileNotFoundError(f"Split type directory not found: {type_dir}")
        raise FileNotFoundError(f"Data split path not found: {split_path} (checked exact and int variants)")
    if not os.path.exists(os.path.join(datasplit_dir, "full_dataset_topology.json")):
        raise FileNotFoundError(f"Full dataset topology not found: {os.path.join(datasplit_dir, 'full_dataset_topology.json')}")
    if not os.path.exists(os.path.join(datasplit_dir, "full_dataset_dfrac.csv")):
        raise FileNotFoundError(f"Full dataset dfrac not found: {os.path.join(datasplit_dir, 'full_dataset_dfrac')}")

    train_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "train_topology.json"),
        csv_path=os.path.join(split_path, "train_dfrac.csv"),
    )
    val_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "val_topology.json"),
        csv_path=os.path.join(split_path, "val_dfrac.csv"),
    )
    test_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(datasplit_dir, "full_dataset_topology.json"),
        csv_path=os.path.join(datasplit_dir, "full_dataset_dfrac.csv"),
    )
    return train_data, val_data, test_data


def load_feature_topology(features_dir: str, ensemble: str) -> List[pt.Partial_Topology]:
    """Load the feature topology for a given ensemble."""
    # Use the ensemble feature map to get the correct case-sensitive name
    feature_name = ENSEMBLE_FEATURE_MAP.get(ensemble, ensemble.lower())
    topology_path = os.path.join(features_dir, f"topology_{feature_name}.json")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Feature topology not found: {topology_path}")
    return pt.PTSerialiser.load_list_from_json(topology_path)


def get_experimental_uptake(data: List[ExpD_Datapoint]) -> np.ndarray:
    """Extracts experimental uptake values into a numpy array."""
    if not data:
        return np.array([])
    return np.array([d.dfrac for d in data]).squeeze()


def _align_uptake_shape(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    """
    Remove leading singleton axes until the array matches the target dimensionality.
    Used to squeeze arrays shaped like (1, N, T) down to (N, T).
    """
    aligned = arr
    while aligned.ndim > target_ndim and aligned.shape[0] == 1:
        aligned = np.squeeze(aligned, axis=0)
    return aligned


def _is_valid_array(arr: np.ndarray) -> bool:
    """Check if array is valid for computation (non-empty and has non-NaN values)."""
    if arr is None:
        return False
    if arr.size == 0:
        return False
    if np.all(np.isnan(arr)):
        return False
    return True


# --- Provided functions for dMSE and Work Metrics ---
def calculate_dMSE(pred_uptake: np.ndarray, 
                   prior_uptake: np.ndarray, 
                   experimental_uptake: np.ndarray) -> float:
    """
    Computes the change in Validation Error (dMSE) between a predicted model 
    and a prior model, relative to experimental data.
    
    Metric: dMSE_val (Delta MSE_fitting)
    
    Args:
        pred_uptake (np.array): Shape (n_segments, n_timepoints). Fitted/Predicted uptake.
        prior_uptake (np.array): Shape (n_segments, n_timepoints). Uptake from the Prior model.
        experimental_uptake (np.array): Shape (n_segments, n_timepoints). Target/Experimental uptake.
        
    Returns:
        float: The dMSE value. Negative values indicate the fit improved over the prior.
    """
    # Validate inputs
    if not _is_valid_array(pred_uptake) or not _is_valid_array(prior_uptake) or not _is_valid_array(experimental_uptake):
        return np.nan

    pred_uptake = _align_uptake_shape(pred_uptake, experimental_uptake.ndim)
    prior_uptake = _align_uptake_shape(prior_uptake, experimental_uptake.ndim)

    # Check shapes after alignment
    if pred_uptake.size == 0 or prior_uptake.size == 0:
        return np.nan

    # Calculate squared residuals
    # Note: nan_mean is used to handle potential missing experimental data points
    mse_pred = np.nanmean(np.abs(pred_uptake - experimental_uptake)**2)
    mse_prior = np.nanmean(np.abs(prior_uptake - experimental_uptake)**2)
    
    # dMSE = Error_Posterior - Error_Prior
    d_mse = mse_pred - mse_prior
    
    return d_mse

def calculate_work_metrics(pred_lnpf: np.ndarray, 
                           prior_lnpf: np.ndarray, 
                           T: float = 300.0) -> dict:
    """
    Computes thermodynamic work metrics (Shape, Density, Fitting, Scale) based on 
    Log Protection Factors (lnPF) derived from the 'calc_pmf' logic.
    
    Args:
        pred_lnpf (np.array): Shape (n_residues,). Natural log of protection factors from the fitted model.
        prior_lnpf (np.array): Shape (n_residues,). Natural log of protection factors from the prior.
        T (float): Temperature in Kelvin. Default is 300.
        
    Returns:
        dict: Dictionary containing the computed metrics in kJ/mol.
    """
    R = 8.314  # J/(mol K)
    
    # --- Helper function to compute H, S, G vectors for a given lnPF distribution ---
    def get_thermo_props(lnpf_array):
        # 1. Central Tendency (Average lnPF of the ensemble)
        avg_logPF = np.mean(lnpf_array)
        
        # 2. Deviation from the mean (log_delta_PFi)
        delta_phi = np.abs(lnpf_array - avg_logPF)
        d_phi = lnpf_array - avg_logPF

        # 3. Enthalpy component (H_opt)
        # Based on script: H_opt = R * T * |lnPF - mean|
        H = R * T * delta_phi
        
        # 4. Partition Function components
        # q = exp(-|lnPF - mean|)
        q = np.abs(np.exp(-delta_phi))
        Z = np.sum(q) # Scalar
        
        # 5. Probability (Pi_opt)
        Pi = q * Z  # Shape (n_residues,)
        
        # 6. Entropy component (S_opt)
        # Based on script: S_opt = -R * Pi * log(Pi)
        # Note: We add a small epsilon to log to prevent log(0) if Pi is extremely small
        S = -R * Pi * np.log(Pi + 1e-16)
        
        # 7. Gibbs Free Energy component (G_opt)
        # G = H - TS
        G = H - (T * S)
        
        return avg_logPF, H, S, G

    # --- Calculate properties for Predicted and Prior ---
    mu_pred, H_pred, S_pred, G_pred = get_thermo_props(pred_lnpf)
    mu_prior, H_prior, S_prior, G_prior = get_thermo_props(prior_lnpf)
    
    # --- Compute Work Metrics (in J/mol, then convert to kJ/mol) ---
    
    # 1. Work Scale (delta_H_abs_kj)
    # Represents the shift in the global magnitude/scale of protection factors
    # Formula: RT * |mean(pred) - mean(prior)|
    work_scale_j = R * T * np.abs(mu_pred - mu_prior)
    
    # 2. Work Shape (delta_H_opt_kj)
    # Represents the energetic cost of changing the relative profile "shape"
    # Formula: mean( |H_pred - H_prior| )
    work_shape_j = np.mean(np.abs(H_pred - H_prior))
    
    # 3. Work Density (-Tdelta_S_opt_kj)
    # Represents the entropic cost (redistribution of PFs)
    # Formula: T * mean( |S_pred - S_prior| )
    # Note: The script calculates delta_S as abs diff, then multiplies by T.
    work_density_j = T * np.mean(np.abs(S_pred - S_prior))
    
    # 4. Work Fitting (delta_G_opt_kj)
    # Total optimization work done
    # Formula: mean( |G_pred - G_prior| )
    work_fitting_j = np.mean(np.abs(G_pred - G_prior))
    
    work_fitting_f =  work_scale_j + work_density_j 

    work_magnitude_j =  work_shape_j - work_scale_j
    # --- Convert to kJ/mol and Pack ---
    metrics = {
        "delta_H_abs_kj": work_scale_j / 1000.0,     # Work_scale
        "delta_H_opt_kj": work_shape_j / 1000.0,     # Work_shape
        "-Tdelta_S_opt_kj": work_density_j / 1000.0, # Work_density
        "delta_G_opt_kj": work_fitting_j / 1000.0,    # Work_opt
        "delta_G_fit_kj": work_fitting_f / 1000.0,    # Work_fitting
        "delta_G_mag_kj": work_magnitude_j / 1000.0  # Work_magnitude
    }
    
    return metrics

def calculate_mse(predicted_mapped_uptake: np.ndarray, experimental_uptake: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between predicted mapped uptake and experimental uptake.
    
    Args:
        predicted_mapped_uptake (np.ndarray): Predicted uptake values mapped to peptides (n_peptides, n_timepoints).
        experimental_uptake (np.ndarray): Experimental uptake values (n_peptides, n_timepoints).
        
    Returns:
        float: The Mean Squared Error.
    """
    # Validate inputs
    if not _is_valid_array(predicted_mapped_uptake) or not _is_valid_array(experimental_uptake):
        return np.nan

    # Align shapes by removing leading singleton axes in predictions
    predicted_mapped_uptake = _align_uptake_shape(predicted_mapped_uptake, experimental_uptake.ndim)

    # Check if array is empty after alignment
    if predicted_mapped_uptake.size == 0:
        return np.nan

    # Ensure shapes are compatible
    if predicted_mapped_uptake.shape != experimental_uptake.shape:
        # Attempt to proceed if only timepoints differ, otherwise return NaN
        if len(predicted_mapped_uptake.shape) >= 2 and len(experimental_uptake.shape) >= 2:
            if predicted_mapped_uptake.shape[0] == experimental_uptake.shape[0]:
                min_timepoints = min(predicted_mapped_uptake.shape[1], experimental_uptake.shape[1])
                predicted_mapped_uptake = predicted_mapped_uptake[:, :min_timepoints]
                experimental_uptake = experimental_uptake[:, :min_timepoints]
            else:
                return np.nan
        else:
            return np.nan
    
    # Use nanmean to handle potential missing data (NaNs)
    return np.nanmean(np.abs(predicted_mapped_uptake - experimental_uptake)**2)

def calculate_recovery_percentage(cluster_assignments: np.ndarray, frame_weights: np.ndarray, target_ratios: Dict[str, float], state_mapping: Dict[int, str]) -> float:
    """
    Compute recovery percentage using 1 - sqrt(JSD) based on state clustering.
    For MoPrP: States are Folded (0), PUF1 (1), PUF2 (2), unfolded (4)
    Target distribution based on experimental data.
    """
    if cluster_assignments is None or len(cluster_assignments) == 0:
        return np.nan

    if target_ratios is None:
        return np.nan

    if frame_weights is None:
        weights = np.ones(len(cluster_assignments), dtype=float)
    else:
        weights = np.asarray(frame_weights, dtype=float)

    if len(weights) != len(cluster_assignments) or np.sum(weights) <= 0:
        return np.nan

    # Normalize weights
    weights = weights / np.sum(weights)

    # Invert mapping: state -> cluster ids
    state_to_clusters = {}
    for cluster_id, state_name in state_mapping.items():
        state_to_clusters.setdefault(state_name, []).append(cluster_id)

    # Compute current proportions (weighted)
    current_proportions = {state: 0.0 for state in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        # Use np.isin for numpy array
        state_mask = np.isin(cluster_assignments, cluster_ids)
        current_proportions[state_name] = float(np.sum(weights[state_mask]))

    # Order states consistently with target_ratios
    states = list(target_ratios.keys())
    P = np.array([current_proportions.get(s, 0.0) for s in states], dtype=float)
    Q = np.array([target_ratios.get(s, 0.0) for s in states], dtype=float)

    # Normalize distributions
    if P.sum() > 0:
        P = P / P.sum()
    else:
        return np.nan
        
    if Q.sum() > 0:
        Q = Q / Q.sum()
    else:
        return np.nan

    # Jensen-Shannon divergence (base 2)
    M = 0.5 * (P + Q)

    def kld(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    jsd = 0.5 * (kld(P, M) + kld(Q, M))
    jsd = max(0.0, jsd)

    return 100.0 * (1.0 - np.sqrt(jsd))

def load_cluster_assignments(clustering_dir: str, ensemble: str) -> Optional[np.ndarray]:
    """
    Load cluster assignments for a specific ensemble from nested subdirectories (MoPrP structure).
    
    Expected structure:
    - AF2_MSAss/AF2_MSAss_frame_to_cluster.csv
    - AF2_Filtered/AF2_Filtered_frame_to_cluster.csv
    """
    if ensemble not in ENSEMBLE_CLUSTERING_MAP:
        print(f"Warning: No clustering mapping for ensemble {ensemble}")
        return None
    
    subdir, filename = ENSEMBLE_CLUSTERING_MAP[ensemble]
    cluster_path = os.path.join(clustering_dir, subdir, filename)
    
    if os.path.exists(cluster_path):
        df = pd.read_csv(cluster_path)
        if "cluster_label" in df.columns:
            return df["cluster_label"].values
        else:
            print(f"Warning: 'cluster_label' column not found in {cluster_path}")
            return None
    else:
        print(f"Warning: Clustering file not found: {cluster_path}")
        return None

def create_violin_plots(scores_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create violin plots for each score metric, faceted by ensemble and loss type,
    with hue by split type and convergence values on x-axis.
    
    Args:
        scores_df (pd.DataFrame): DataFrame containing all scores.
        output_dir (str): Directory to save the plots.
    """
    # Define score columns to plot
    score_columns = [
        "train_mse", "val_mse", "test_mse",
        "d_mse_train", "d_mse_val", "d_mse_test",
        "work_scale", "work_shape", "work_density", "work_fitting",
        "kl_divergence", "recovery_percent", "val_loss"
    ]
    
    # Filter to columns that exist in the dataframe
    available_scores = [col for col in score_columns if col in scores_df.columns]
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    for score_col in available_scores:
        # Skip if all values are NaN
        if scores_df[score_col].isna().all():
            print(f"  Skipping {score_col} (all NaN)")
            continue
        
        # Get unique ensembles and loss functions
        ensembles = sorted(scores_df['ensemble'].unique())
        loss_functions = sorted(scores_df['loss_function'].unique())
        
        # Create facet grid
        n_ensembles = len(ensembles)
        n_losses = len(loss_functions)
        
        fig, axes = plt.subplots(n_ensembles, n_losses, figsize=(5*n_losses, 5*n_ensembles))
        
        # Handle case where there's only one subplot
        if n_ensembles == 1 and n_losses == 1:
            axes = np.array([[axes]])
        elif n_ensembles == 1:
            axes = axes.reshape(1, -1)
        elif n_losses == 1:
            axes = axes.reshape(-1, 1)
        
        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]
                
                # Filter data for this ensemble and loss function
                subset = scores_df[
                    (scores_df['ensemble'] == ensemble) & 
                    (scores_df['loss_function'] == loss_func)
                ].copy()
                
                if subset.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f'{ensemble} - {loss_func}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Round convergence values for better x-axis labels
                subset['convergence_rounded'] = subset['convergence_value'].round(4)
                
                # Create violin plot
                sns.violinplot(
                    data=subset,
                    x='convergence_rounded',
                    y=score_col,
                    hue='split_type',
                    ax=ax,
                    palette='Set2'
                )
                
                ax.set_xlabel('Convergence Value', fontsize=10)
                ax.set_ylabel(score_col, fontsize=10)
                ax.set_title(f'{ensemble} - {loss_func}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                
                # Move legend outside if there are multiple hues
                if len(subset['split_type'].unique()) > 1:
                    ax.legend(title='Split Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    ax.get_legend().remove()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"violin_{score_col}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Calculate various scores and metrics from processed MoPrP optimization results."
    )
    parser.add_argument(
        "--processed-data-dir",
        default="../fitting/jaxENT/_processed_optimise_quick_test_splits__20250918_171508",
        help="Directory containing processed .npy files from MoPrP_Process_Optimisation_results.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the final scores CSV. If omitted, creates '_scores' subdirectory inside processed-data-dir.",
    )
    parser.add_argument(
        "--datasplit-dir",
        default="../fitting/jaxENT/_datasplits",
        help="Directory containing data splits (train/val/full datasets).",
    )
    parser.add_argument(
        "--features-dir",
        default="../fitting/jaxENT/_featurise",
        help="Directory containing featurized data (features_*.npz and topology_*.json)",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Directory containing cluster assignment subdirectories (AF2_MSAss, AF2_Filtered)",
    )
    parser.add_argument(
        "--state-ratios-json",
        default="../analysis/state_ratios.json",
        help="Path to JSON with target state ratios",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided directories as absolute paths.",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(__file__)
    if args.absolute_paths:
        processed_data_dir = args.processed_data_dir
        datasplit_dir = args.datasplit_dir
        features_dir = args.features_dir
        clustering_dir = args.clustering_dir
    else:
        processed_data_dir = os.path.abspath(os.path.join(script_dir, args.processed_data_dir))
        datasplit_dir = os.path.abspath(os.path.join(script_dir, args.datasplit_dir))
        features_dir = os.path.abspath(os.path.join(script_dir, args.features_dir))
        clustering_dir = os.path.abspath(os.path.join(script_dir, args.clustering_dir))

    if args.output_dir:
        if args.absolute_paths:
            output_scores_dir = args.output_dir
        else:
            output_scores_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    else:
        # Create output directory as a subdirectory of processed_data_dir
        basename = os.path.basename(processed_data_dir.rstrip('/'))
        output_scores_dir = os.path.join(processed_data_dir, f"_scores_{basename}")
    
    os.makedirs(output_scores_dir, exist_ok=True)

    print(f"Resolved processed_data_dir: {processed_data_dir}")
    print(f"Resolved datasplit_dir: {datasplit_dir}")
    print(f"Resolved features_dir: {features_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved output_scores_dir: {output_scores_dir}")

    # State mapping for MoPrP
    state_mapping = {
        0: "Folded",
        1: "PUF1",
        2: "PUF2",
        4: "unfolded",
    }

    # Load target ratios
    state_ratios_path = os.path.join(script_dir, args.state_ratios_json)
    target_ratios = None
    if os.path.exists(state_ratios_path):
        try:
            with open(state_ratios_path, "r") as f:
                ratios_data = json.load(f)
            
            target_ratios = {
                "Folded": ratios_data["fractional_populations"]["folded"]["fraction"],
                "PUF1": ratios_data["fractional_populations"]["PUF1"]["fraction"],
                "PUF2": ratios_data["fractional_populations"]["PUF2"]["fraction"],
                "unfolded": 0.0,
            }
            print(f"Loaded target ratios from {state_ratios_path}: {target_ratios}")
        except Exception as e:
            print(f"Error loading state ratios: {e}")
    else:
        print(f"State ratios file not found: {state_ratios_path}")

    print("-" * 60)

    # --- 1. First pass: Collect all runs and unique data configurations ---
    all_run_info = []
    if not os.path.exists(processed_data_dir):
        print(f"Processed data directory not found: {processed_data_dir}")
        return

    for split_type_dir in os.listdir(processed_data_dir):
        full_split_type_path = os.path.join(processed_data_dir, split_type_dir)
        if not os.path.isdir(full_split_type_path):
            continue
        
        actual_split_type = split_type_dir if split_type_dir != "_flat" else "flat"

        for run_id in os.listdir(full_split_type_path):
            full_run_path = os.path.join(full_split_type_path, run_id)
            if not os.path.isdir(full_run_path):
                continue
            
            # Updated regex for MoPrP ensembles
            match = re.match(
                r"(AF2_MSAss|AF2_filtered)_(mcMSE|MSE|Sigma_MSE)_(.+?)_split(\d+)_maxent([\d\.]+)_bvreg([\d\.]+)_bvregfn([A-Za-z0-9]+)",
                run_id,
            )
            if not match:
                continue

            (
                ensemble,
                loss_name,
                run_split_type,
                split_idx_str,
                maxent_str,
                bv_reg_str,
                bv_reg_fn,
            ) = match.groups()
            
            if "_cluster" in run_split_type:
                # Check if the full version exists first
                full_path = os.path.join(datasplit_dir, run_split_type)
                stripped_type = run_split_type.replace("_cluster", "")
                stripped_path = os.path.join(datasplit_dir, stripped_type)
                
                if os.path.exists(full_path):
                    effective_split_type = run_split_type
                elif os.path.exists(stripped_path):
                    effective_split_type = stripped_type
                else:
                    # Fallback to stripped as per original logic
                    effective_split_type = stripped_type
            else:
                effective_split_type = run_split_type

            run_info = {
                "run_id": run_id,
                "full_run_path": full_run_path,
                "ensemble": ensemble,
                "loss_name": loss_name,
                "run_split_type": run_split_type,
                "effective_split_type": effective_split_type,
                "split_idx_str": split_idx_str,
                "split_idx": int(split_idx_str),
                "maxent_value": float(maxent_str),
                "bv_reg_value": float(bv_reg_str),
                "bv_reg_function": bv_reg_fn,
                "actual_split_type": actual_split_type,
            }
            all_run_info.append(run_info)

    unique_configs = sorted(list(set((r['ensemble'], r['effective_split_type'], r['split_idx_str']) for r in all_run_info)))

    # --- 2. Pre-load and cache all data ---
    data_cache = {}
    print("--- Pre-caching data ---")
    for ensemble, effective_split_type, split_idx_str in unique_configs:
        cache_key = (ensemble, effective_split_type, split_idx_str)
        print(f"Caching data for {cache_key}...")
        
        # Find a valid run to load priors from (some runs might be incomplete)
        runs_in_group = [r for r in all_run_info if (r['ensemble'], r['effective_split_type'], r['split_idx_str']) == cache_key]
        valid_run = None
        for r in runs_in_group:
            p1 = os.path.join(r['full_run_path'], "prior_ln_pf.npy")
            p2 = os.path.join(r['full_run_path'], "prior_uptake.npy")
            if os.path.exists(p1) and os.path.exists(p2):
                valid_run = r
                break
        
        if valid_run is None:
            print(f"  ERROR: No valid run found with prior files for config {cache_key}. Skipping group.")
            continue

        full_run_path = valid_run['full_run_path']

        try:
            train_data, val_data, test_data = load_experimental_data(datasplit_dir, effective_split_type, split_idx_str)
            feature_top = load_feature_topology(features_dir, ensemble)
            
            # Use the ensemble feature map to get the correct case-sensitive name
            feature_name = ENSEMBLE_FEATURE_MAP.get(ensemble, ensemble.lower())
            feature_path = os.path.join(features_dir, f"features_{feature_name}.npz")
            features = BV_input_features.load(feature_path)

            full_dataset_loader = ExpD_Dataloader(data=test_data)
            full_dataset_loader.create_datasets(
                features=features,
                feature_topology=feature_top,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )

            prior_ln_pf = np.load(os.path.join(full_run_path, "prior_ln_pf.npy"))
            prior_uptake = np.load(os.path.join(full_run_path, "prior_uptake.npy"))

            train_map = full_dataset_loader.train.residue_feature_ouput_mapping
            val_map = full_dataset_loader.val.residue_feature_ouput_mapping
            test_map = full_dataset_loader.test.residue_feature_ouput_mapping
            
            mapped_prior_uptake_train = np.array([apply_sparse_mapping(train_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T
            mapped_prior_uptake_val = np.array([apply_sparse_mapping(val_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T
            mapped_prior_uptake_test = np.array([apply_sparse_mapping(test_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T

            data_cache[cache_key] = {
                "loader": full_dataset_loader,
                "y_true_train": get_experimental_uptake(train_data),
                "y_true_val": get_experimental_uptake(val_data),
                "y_true_test": get_experimental_uptake(test_data),
                "prior_ln_pf": prior_ln_pf,
                "mapped_prior_uptake_train": mapped_prior_uptake_train,
                "mapped_prior_uptake_val": mapped_prior_uptake_val,
                "mapped_prior_uptake_test": mapped_prior_uptake_test,
            }
        except Exception as e:
            print(f"  ERROR caching data for {cache_key}: {e}")
            import traceback
            traceback.print_exc()
    print("--- Caching complete ---")

    all_scores = []
    cluster_assignments_cache = {}

    # --- 3. Second pass: Process runs using cached data ---
    print("\n--- Processing runs ---")
    for run_info in tqdm(all_run_info, desc="Processing runs"):
        run_id = run_info['run_id']
        full_run_path = run_info['full_run_path']

        try:
            ensemble = run_info['ensemble']
            effective_split_type = run_info['effective_split_type']
            split_idx_str = run_info['split_idx_str']
            cache_key = (ensemble, effective_split_type, split_idx_str)

            if cache_key not in data_cache:
                print(f"  Warning: Data not found in cache for key {cache_key}. Skipping run {run_id}.")
                continue

            cached_data = data_cache[cache_key]
            loader = cached_data["loader"]
            y_true_train = cached_data["y_true_train"]
            y_true_val = cached_data["y_true_val"]
            y_true_test = cached_data["y_true_test"]
            prior_ln_pf = cached_data["prior_ln_pf"]
            mapped_prior_uptake_train = cached_data["mapped_prior_uptake_train"]
            mapped_prior_uptake_val = cached_data["mapped_prior_uptake_val"]
            mapped_prior_uptake_test = cached_data["mapped_prior_uptake_test"]

            # Load only the per-run data
            convergence_thresholds = []
            conv_path = os.path.join(full_run_path, "convergence_thresholds.txt")
            if os.path.exists(conv_path):
                with open(conv_path, "r") as f:
                    convergence_thresholds = [float(line.strip()) for line in f]

            pred_ln_pf_stack = np.load(os.path.join(full_run_path, "pred_ln_pf.npy"))
            pred_uptake_stack = np.load(os.path.join(full_run_path, "pred_uptake.npy"))
            kl_divergence_stack = np.load(os.path.join(full_run_path, "kl_divergence.npy"))
            frame_weights_stack = np.load(os.path.join(full_run_path, "frame_weights.npy"))

            val_loss_stack = None
            val_loss_path = os.path.join(full_run_path, "val_loss.npy")
            if os.path.exists(val_loss_path):
                val_loss_stack = np.load(val_loss_path)

            cluster_ratios_df = None
            cluster_ratios_path = os.path.join(full_run_path, "cluster_ratios.csv")
            if os.path.exists(cluster_ratios_path):
                cluster_ratios_df = pd.read_csv(cluster_ratios_path)

            if not (len(pred_ln_pf_stack) == len(pred_uptake_stack) == len(kl_divergence_stack) == len(frame_weights_stack) == len(convergence_thresholds)):
                print(f"  Warning: Data stacks have inconsistent lengths for run {run_id}. Skipping.")
                continue

            train_map = loader.train.residue_feature_ouput_mapping
            val_map = loader.val.residue_feature_ouput_mapping
            test_map = loader.test.residue_feature_ouput_mapping
            
            if ensemble not in cluster_assignments_cache:
                cluster_assignments_cache[ensemble] = load_cluster_assignments(clustering_dir, ensemble)
            cluster_assignments = cluster_assignments_cache[ensemble]

            # Loop over each convergence point
            for i, convergence_val in enumerate(convergence_thresholds):
                pred_ln_pf = pred_ln_pf_stack[i]
                pred_uptake = pred_uptake_stack[i]
                kl_divergence = kl_divergence_stack[i]
                frame_weights = frame_weights_stack[i]

                val_loss = np.nan
                if val_loss_stack is not None and i < len(val_loss_stack):
                    val_loss = val_loss_stack[i]

                # Apply mapping
                mapped_pred_uptake_train = np.array([apply_sparse_mapping(train_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]).T
                mapped_pred_uptake_val = np.array([apply_sparse_mapping(val_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]).T
                mapped_pred_uptake_test = np.array([apply_sparse_mapping(test_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]).T
                
                # Calculate scores
                train_mse = calculate_mse(mapped_pred_uptake_train, y_true_train)
                val_mse = calculate_mse(mapped_pred_uptake_val, y_true_val)
                test_mse = calculate_mse(mapped_pred_uptake_test, y_true_test)
                d_mse_train = calculate_dMSE(mapped_pred_uptake_train, mapped_prior_uptake_train, y_true_train)
                d_mse_val = calculate_dMSE(mapped_pred_uptake_val, mapped_prior_uptake_val, y_true_val)
                d_mse_test = calculate_dMSE(mapped_pred_uptake_test, mapped_prior_uptake_test, y_true_test)
                
                work_metrics = calculate_work_metrics(pred_ln_pf, prior_ln_pf)
                
                recovery_percent = np.nan
                if cluster_assignments is not None:
                    recovery_percent = calculate_recovery_percentage(cluster_assignments, frame_weights, target_ratios, state_mapping)

                cluster_ratios = {}
                if cluster_ratios_df is not None:
                    row = cluster_ratios_df[cluster_ratios_df['convergence'] == convergence_val]
                    if not row.empty:
                        cluster_ratios = row.iloc[0].to_dict()
                        cluster_ratios.pop('convergence', None)

                scores_entry = {
                    "ensemble": ensemble,
                    "loss_function": run_info['loss_name'],
                    "bv_reg_function": run_info['bv_reg_function'],
                    "split_type": run_info['actual_split_type'],
                    "split_idx": run_info['split_idx'],
                    "maxent_value": run_info['maxent_value'],
                    "bv_reg_value": run_info['bv_reg_value'],
                    "convergence_value": convergence_val,
                    "kl_divergence": kl_divergence,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "test_mse": test_mse,
                    "d_mse_train": d_mse_train,
                    "d_mse_val": d_mse_val,
                    "d_mse_test": d_mse_test,
                    "work_scale": work_metrics.get("delta_H_abs_kj", np.nan),
                    "work_shape": work_metrics.get("delta_H_opt_kj", np.nan),
                    "work_density": work_metrics.get("-Tdelta_S_opt_kj", np.nan),
                    "work_opt": work_metrics.get("delta_G_opt_kj", np.nan),
                    "work_fitting": work_metrics.get("delta_G_fit_kj", np.nan),
                    "work_magnitude": work_metrics.get("delta_G_mag_kj", np.nan),
                    "recovery_percent": recovery_percent,
                    "val_loss": val_loss,
                }
                scores_entry.update(cluster_ratios)
                all_scores.append(scores_entry)

        except Exception as e:
            print(f"  ERROR processing run {run_id}: {e}")
            import traceback
            traceback.print_exc()

    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        output_csv_path = os.path.join(output_scores_dir, "model_scores.csv")
        scores_df.to_csv(output_csv_path, index=False)
        print(f"\nAll scores saved to: {output_csv_path}")
        
        # Generate violin plots
        print("\n--- Generating violin plots ---")
        create_violin_plots(scores_df, output_scores_dir)
    else:
        print("\nNo scores were generated.")


if __name__ == "__main__":
    main()