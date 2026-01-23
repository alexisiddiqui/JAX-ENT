"""
This script loads the outputs of process_optimisation_results.py and computes various scores and metrics.
This includes error metrics: MSE(train/val/test), dMSE(train/val/test),
and Work Metrics: Shape, Density, Fitting, Scale, and KL Divergence(weights|uniform)

As the target data is already split into train/val (test is the entire dataset), the error metrics are computed by applying the train/val sparse mapping to the predicted and prior uptake data.

Work metrics are computed based on the entire set of predicted and prior lnPF values.

/home/alexi/Documents/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/optimise_ISO_TRI_BI_splits_Sigma.py
^ contains some code to apply the sparse mapping to get the mapped uptake values as well as get the appropriate file paths.

"""

import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Add the base directory to the path to import JAX-ENT modules
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Datapoint, ExpD_Dataloader
import jaxent.src.interfaces.topology as pt
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping


# --- Data Loading Helpers ---
def load_experimental_data(
    datasplit_dir: str, split_type: str, split_idx: int
) -> Tuple[List[HDX_peptide], List[HDX_peptide], List[HDX_peptide]]:
    """
    Load train, validation, and full (test) experimental data for a given split.
    """
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data split path not found: {split_path}")
    if not os.path.exists(os.path.join(datasplit_dir, "full_dataset_topology.json")):
        raise FileNotFoundError(f"Full dataset topology not found: {os.path.join(datasplit_dir, 'full_dataset_topology.json')}")
    if not os.path.exists(os.path.join(datasplit_dir, "full_dataset_dfrac.csv")):
        raise FileNotFoundError(f"Full dataset dfrac not found: {os.path.join(datasplit_dir, 'full_dataset_dfrac.csv')}")

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
    topology_path = os.path.join(features_dir, f"topology_{ensemble.lower()}.json")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Feature topology not found: {topology_path}")
    return pt.PTSerialiser.load_list_from_json(topology_path)


def get_experimental_uptake(data: List[ExpD_Datapoint]) -> np.ndarray:
    """Extracts experimental uptake values into a numpy array."""
    if not data:
        return np.array([])
    return np.array([d.y_true for d in data]).squeeze()



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
        
        # 3. Enthalpy component (H_opt)
        # Based on script: H_opt = R * T * |lnPF - mean|
        H = R * T * delta_phi
        
        # 4. Partition Function components
        # q = exp(-|lnPF - mean|)
        q = np.exp(-delta_phi)
        Z = np.sum(q) # Scalar
        
        # 5. Probability (Pi_opt)
        Pi = q / Z
        
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
    
    # --- Convert to kJ/mol and Pack ---
    metrics = {
        "delta_H_abs_kj": work_scale_j / 1000.0,     # Work_scale
        "delta_H_opt_kj": work_shape_j / 1000.0,     # Work_shape
        "-Tdelta_S_opt_kj": work_density_j / 1000.0, # Work_density
        "delta_G_opt_kj": work_fitting_j / 1000.0    # Work_fitting
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
    # Ensure shapes are compatible
    if predicted_mapped_uptake.shape != experimental_uptake.shape:
        print(f"Warning: Shape mismatch in calculate_mse. Predicted: {predicted_mapped_uptake.shape}, Experimental: {experimental_uptake.shape}")
        # Attempt to proceed if only timepoints differ, otherwise return NaN
        if predicted_mapped_uptake.shape[0] == experimental_uptake.shape[0]:
            min_timepoints = min(predicted_mapped_uptake.shape[1], experimental_uptake.shape[1])
            predicted_mapped_uptake = predicted_mapped_uptake[:, :min_timepoints]
            experimental_uptake = experimental_uptake[:, :min_timepoints]
        else:
            return np.nan
    
    # Use nanmean to handle potential missing data (NaNs)
    return np.nanmean(np.abs(predicted_mapped_uptake - experimental_uptake)**2)

def calculate_recovery_percentage(cluster_assignments: np.ndarray, frame_weights: np.ndarray = None) -> float:
    """
    Compute open state recovery percentage using 1 - sqrt(JSD) against the target distribution.
    This function is copied from analyse_loss_ISO_TRI_BI.py.
    """
    if cluster_assignments is None or len(cluster_assignments) == 0:
        return np.nan

    if frame_weights is None:
        weights = np.ones(len(cluster_assignments), dtype=float)
    else:
        weights = np.asarray(frame_weights, dtype=float)

    if len(weights) != len(cluster_assignments) or np.sum(weights) <= 0:
        return np.nan

    weights = weights / np.sum(weights)

    # Predicted distribution (open=0, closed=1, remaining treated as intermediate)
    open_ratio = float(np.sum(weights[cluster_assignments == 0]))
    closed_ratio = float(np.sum(weights[cluster_assignments == 1]))
    intermediate_ratio = max(0.0, 1.0 - open_ratio - closed_ratio)

    pred_dist = np.array([open_ratio, closed_ratio, intermediate_ratio], dtype=float)
    pred_dist = np.clip(pred_dist, 0.0, 1.0)
    if pred_dist.sum() == 0:
        return np.nan
    pred_dist = pred_dist / pred_dist.sum()

    # Ground-truth distribution (assuming 40:60 open:closed for IsoValidation)
    gt_dist = np.array([0.4, 0.6, 0.0], dtype=float)
    gt_dist = gt_dist / gt_dist.sum()

    eps = 1e-12
    pred_dist = np.clip(pred_dist, eps, 1.0)
    gt_dist = np.clip(gt_dist, eps, 1.0)
    midpoint = np.clip(0.5 * (pred_dist + gt_dist), eps, 1.0)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log2(p / q)))

    jsd = 0.5 * _kl(pred_dist, midpoint) + 0.5 * _kl(gt_dist, midpoint)
    jsd = max(0.0, jsd)

    return 100.0 * max(0.0, 1.0 - np.sqrt(jsd))

def load_cluster_assignments(clustering_dir: str, ensemble: str) -> Optional[np.ndarray]:
    """
    Load cluster assignments for a specific ensemble.
    """
    cluster_path = os.path.join(clustering_dir, f"cluster_assignments_{ensemble}.csv")
    if os.path.exists(cluster_path):
        df = pd.read_csv(cluster_path)
        return df["cluster_assignment"].values
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Calculate various scores and metrics from processed optimization results."
    )
    parser.add_argument(
        "--processed-data-dir",
        default="../fitting/jaxENT/_processed_optimise_quick_test_splits__20250915_125135",
        help="Directory containing processed .npy files from process_optimisation_results.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the final scores CSV. If omitted, derived from processed-data-dir basename prefixed with '_scores'.",
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
        default="../data/_clustering_results",
        help="Directory containing cluster assignment CSV files.",
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
        base_name = os.path.basename(os.path.normpath(processed_data_dir))
        output_scores_dir = os.path.join(os.path.dirname(processed_data_dir), f"_scores_{base_name}")
    
    os.makedirs(output_scores_dir, exist_ok=True)

    print(f"Resolved processed_data_dir: {processed_data_dir}")
    print(f"Resolved datasplit_dir: {datasplit_dir}")
    print(f"Resolved features_dir: {features_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved output_scores_dir: {output_scores_dir}")
    print("-" * 60)

    all_scores = []

    # Traverse processed data directory
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

            print(f"Processing run: {run_id}")

            try:
                # Extract metadata from run_id
                match = re.match(r"(.+?)_(.+?)_(.+?)_split(\d+)_maxent([\d\.]+)", run_id)
                if not match:
                    print(f"  Warning: Could not parse run_id: {run_id}. Skipping.")
                    continue
                
                ensemble, loss_name, run_split_type, split_idx_str, maxent_str = match.groups()
                split_idx = int(split_idx_str)
                maxent_value = float(maxent_str)

                # Load convergence values
                convergence_thresholds = []
                conv_path = os.path.join(full_run_path, "convergence_thresholds.txt")
                if os.path.exists(conv_path):
                    with open(conv_path, "r") as f:
                        convergence_thresholds = [float(line.strip()) for line in f]

                # Load processed data stacks
                pred_ln_pf_stack = np.load(os.path.join(full_run_path, "pred_ln_pf.npy"))
                prior_ln_pf = np.load(os.path.join(full_run_path, "prior_ln_pf.npy"))
                pred_uptake_stack = np.load(os.path.join(full_run_path, "pred_uptake.npy"))
                prior_uptake = np.load(os.path.join(full_run_path, "prior_uptake.npy"))
                kl_divergence_stack = np.load(os.path.join(full_run_path, "kl_divergence.npy"))
                frame_weights_stack = np.load(os.path.join(full_run_path, "frame_weights.npy"))

                cluster_ratios_df = None
                cluster_ratios_path = os.path.join(full_run_path, "cluster_ratios.csv")
                if os.path.exists(cluster_ratios_path):
                    cluster_ratios_df = pd.read_csv(cluster_ratios_path)

                if not (len(pred_ln_pf_stack) == len(pred_uptake_stack) == len(kl_divergence_stack) == len(frame_weights_stack) == len(convergence_thresholds)):
                    print(f"  Warning: Data stacks have inconsistent lengths for run {run_id}. Skipping.")
                    continue

                # Load experimental data and create sparse maps
                exp_data_split_type = actual_split_type if actual_split_type != 'flat' else 'random' # Assume random if flat
                train_data, val_data, test_data = load_experimental_data(datasplit_dir, exp_data_split_type, split_idx)
                feature_top = load_feature_topology(features_dir, ensemble)

                train_loader = ExpD_Dataloader(data=train_data)
                train_loader.create_datasets_from_topology(feature_topology=feature_top)
                train_map = train_loader.train.residue_feature_ouput_mapping
                
                val_loader = ExpD_Dataloader(data=val_data)
                val_loader.create_datasets_from_topology(feature_topology=feature_top)
                val_map = val_loader.train.residue_feature_ouput_mapping
                
                test_loader = ExpD_Dataloader(data=test_data)
                test_loader.create_datasets_from_topology(feature_topology=feature_top)
                test_map = test_loader.train.residue_feature_ouput_mapping

                y_true_train = get_experimental_uptake(train_data)
                y_true_val = get_experimental_uptake(val_data)
                y_true_test = get_experimental_uptake(test_data)
                
                # Map prior once
                mapped_prior_uptake_train = np.array([apply_sparse_mapping(train_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T
                mapped_prior_uptake_val = np.array([apply_sparse_mapping(val_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T
                mapped_prior_uptake_test = np.array([apply_sparse_mapping(test_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]).T

                cluster_assignments = load_cluster_assignments(clustering_dir, ensemble)

                # Loop over each convergence point
                for i, convergence_val in enumerate(convergence_thresholds):
                    pred_ln_pf = pred_ln_pf_stack[i]
                    pred_uptake = pred_uptake_stack[i]
                    kl_divergence = kl_divergence_stack[i]
                    frame_weights = frame_weights_stack[i]

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
                        recovery_percent = calculate_recovery_percentage(cluster_assignments, frame_weights)

                    cluster_ratios = {}
                    if cluster_ratios_df is not None:
                        row = cluster_ratios_df[cluster_ratios_df['convergence'] == convergence_val]
                        if not row.empty:
                            cluster_ratios = row.iloc[0].to_dict()
                            cluster_ratios.pop('convergence', None)

                    scores_entry = {
                        "ensemble": ensemble,
                        "loss_function": loss_name,
                        "split_type": actual_split_type,
                        "split_idx": split_idx,
                        "maxent_value": maxent_value,
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
                        "work_fitting": work_metrics.get("delta_G_opt_kj", np.nan),
                        "recovery_percent": recovery_percent,
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
    else:
        print("\nNo scores were generated.")


if __name__ == "__main__":
    main()
