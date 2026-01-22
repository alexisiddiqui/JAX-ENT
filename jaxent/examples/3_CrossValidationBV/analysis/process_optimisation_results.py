import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# Add the base directory to the path to import JAX-ENT modules
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.utils.hdf import load_optimization_history_from_file
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.interfaces.simulation import Simulation_Parameters
import jaxent.src.interfaces.topology as pt
from jaxent.src.predict import run_predict
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.models.config import BV_model_Config
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.custom_types.key import m_key
from jaxent.src.utils.jax_fn import frame_average_features


from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters



class BV_uptake_ForwardPass_frames(
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        # print("heavy_contacts", heavy_contacts.shape)
        # print("acceptor_contacts", acceptor_contacts.shape)
        kints = jnp.asarray(input_features.k_ints)
        # print("kints", kints.shape)
        time_points = parameters.timepoints.reshape(-1)
        # print("timepoint shape", time_points.shape)
        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        # print("logpf", log_pf)

        pf = jnp.exp(log_pf)
        
        # Ensure pf has shape for broadcasting if it is (N,)
        # If features were averaged, pf is (N,). We need (N, 1) to broadcast with kints (N, 1)
        if pf.ndim == 1:
            pf = pf.reshape(-1, 1)

        # Vectorized computation of uptake for each timepoint
        def compute_uptake_for_timepoint(timepoint):
            # Reshape kints to allow broadcasting over frames
            kints_reshaped = kints.reshape(-1, 1)
            # Calculate uptake for each residue: Df_i = 1 - exp(-kint_i * timepoint/ Pf_i)
            # Shapes: (res, 1) * scalar / (res, frames) -> (res, frames)
            uptake = 1 - jnp.exp(-kints_reshaped * timepoint / pf)
            return uptake

        # Compute uptake for each timepoint
        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)
        # print("uptake_per_timepoint", uptake_per_timepoint.shape)
        # raise NotImplementedError("stop here")
        # Return the list of timepoint-wise residue-wise uptake arrays
        return uptake_BV_output_features(uptake_per_timepoint)


def load_experimental_data(results_dir: str, datasplit_dir: str, split_type: str, split_idx: int) -> Tuple[List[HDX_peptide], List[HDX_peptide], List[HDX_peptide], np.ndarray]:
    """
    Load train, validation, and full (test) experimental data for a given split, and extract timepoints.

    Args:
        datasplit_dir: Base directory containing data splits.
        split_type: Type of split (e.g., 'random', 'sequence').
        split_idx: Index of the split.

    Returns:
        Tuple of (train_data, val_data, test_data, timepoints)
    """
    split_path = os.path.join(datasplit_dir, split_type, f"split_{split_idx:03d}")
    parent_dir = os.path.dirname(results_dir)
    full_dataset_path = os.path.join(parent_dir, "_datasplits")

    train_csv_path = os.path.join(split_path, "train_dfrac.csv")
    train_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "train_topology.json"),
        csv_path=train_csv_path,
    )

    # Extract timepoints from the header of the training CSV file
    df_train = pd.read_csv(train_csv_path)
    required_cols = ["datapoint_type", "feature_length"]
    timepoints = [float(col) for col in df_train.columns if col not in required_cols]

    val_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "val_topology.json"),
        csv_path=os.path.join(split_path, "val_dfrac.csv"),
    )
    test_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(full_dataset_path, "full_dataset_topology.json"),
        csv_path=os.path.join(full_dataset_path, "full_dataset_dfrac.csv"),
    )
    return train_data, val_data, test_data, np.array(timepoints)




def load_all_optimization_results_2d(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    bv_reg_functions: List[str],
    num_splits: int,
    EMA: bool = False,
) -> Dict:
    """
    Load all optimization results from HDF5 files for 2D hyperparameter sweep.
    
    Returns nested dict: results[split_type][ensemble][loss_fn][bv_reg_fn][maxent][bv_reg][split_idx] = history
    """
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]
    if not split_types:
        split_types = ["_flat"]

    hdf_pattern = "results_EMA.hdf5" if EMA else "results.hdf5"

    for split_type in split_types:
        results[split_type] = {}
        
        if split_type == "_flat":
            split_type_dir = results_dir
        else:
            split_type_dir = os.path.join(results_dir, split_type)

        if not os.path.exists(split_type_dir):
            continue

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                for bv_reg_fn in bv_reg_functions:
                    results[split_type][ensemble][loss_name][bv_reg_fn] = {}

                    # Filter files matching this combination
                    if split_type == "_flat":
                        pattern = f"{ensemble}_{loss_name}_split"
                    else:
                        pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                    
                    all_files = os.listdir(split_type_dir)
                    files = [
                        f
                        for f in all_files
                        if f.startswith(pattern)
                        and f"bvregfn{bv_reg_fn}" in f
                        and f.endswith(hdf_pattern)
                    ]

                    for filename in files:
                        # Extract maxent, bvreg, split_idx from filename
                        match = re.search(
                            r"split(\d{3})_maxent([\d.]+)_bvreg([\d.]+)_bvregfn([A-Za-z0-9]+)",
                            filename,
                        )
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = float(match.group(2))
                            bvreg_val = float(match.group(3))
                            bvreg_fn_found = match.group(4)

                            if bvreg_fn_found != bv_reg_fn:
                                continue

                            if maxent_val not in results[split_type][ensemble][loss_name][bv_reg_fn]:
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val] = {}

                            if (
                                bvreg_val
                                not in results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val]
                            ):
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ] = {}

                            filepath = os.path.join(split_type_dir, filename)

                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ][split_idx] = history
                            except Exception as e:
                                print(f"    ✗ Failed to load {filename}: {str(e)[:100]}")
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ][split_idx] = None
    
    if "_flat" in results and not results["_flat"]:
        del results["_flat"]
        
    return results

def load_clustering_results(clustering_dir: str) -> Dict:
    """
    Load clustering results from nested subdirectories (MoPrP structure).
    
    Expected structure:
    - AF2_MSAss/AF2_MSAss_frame_to_cluster.csv
    - AF2_Filtered/AF2_Filtered_frame_to_cluster.csv

    Args:
        clustering_dir (str): Directory containing clustering subdirectories

    Returns:
        dict: Dictionary containing clustering results by ensemble
    """
    if not os.path.exists(clustering_dir):
        print(f"Clustering directory not found: {clustering_dir}")
        return {}

    clustering_results = {}
    
    # Mapping of ensemble names to their subdirectory and filename
    ensemble_map = {
        "AF2_MSAss": ("AF2_MSAss", "AF2_MSAss_frame_to_cluster.csv"),
        "AF2_filtered": ("AF2_Filtered", "AF2_Filtered_frame_to_cluster.csv"),
    }

    for ensemble_name, (subdir, filename) in ensemble_map.items():
        cluster_path = os.path.join(clustering_dir, subdir, filename)
        
        if os.path.exists(cluster_path):
            cluster_df = pd.read_csv(cluster_path)
            clustering_results[ensemble_name] = {
                "cluster_assignments": cluster_df["cluster_label"].values,
                "frame_data": cluster_df,
            }
            print(f"Loaded cluster assignments for {ensemble_name}: {len(cluster_df)} frames")
        else:
            print(f"Warning: Clustering file not found for {ensemble_name} at {cluster_path}")

    return clustering_results

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calculate KL divergence between two probability distributions.

    Args:
        p: First probability distribution (frame_weights)
        q: Second probability distribution (uniform prior)
        eps: Small value to avoid log(0)

    Returns:
        KL divergence KL(p||q)
    """
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Add small epsilon to avoid log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Calculate KL divergence: KL(p||q) = Σ p(i) * log(p(i)/q(i))
    return np.sum(p * np.log(p / q))

def calculate_cluster_ratios(cluster_assignments: np.ndarray, frame_weights: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate ratios of clusters based on assignments and optional frame weights.

    Args:
        cluster_assignments (np.ndarray): Cluster assignments
        frame_weights (np.ndarray, optional): Frame weights from optimization

    Returns:
        dict: Cluster ratios
    """
    if frame_weights is None:
        frame_weights = np.ones(len(cluster_assignments))
    
    # Normalize frame weights
    if np.sum(frame_weights) == 0:
        return {} # Return empty if no weights

    # Calculate weighted ratios
    ratios = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster in unique_clusters:
        if cluster >= 0:  # Skip invalid clusters (-1)
            mask = cluster_assignments == cluster
            ratios[f"cluster_{cluster}"] = np.sum(frame_weights[mask])

    return ratios

def main():
    parser = argparse.ArgumentParser(
        description="Process optimization results to extract predictions, KL divergence, and cluster ratios."
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test_SIGMA_50_lr0.1_BV_objectve_20250918_171508",
        help="Directory containing optimization HDF5 result files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for processed .npy files. If omitted, derived from results-dir basename prefixed with '_processed'.",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Directory containing cluster assignment subdirectories (AF2_MSAss, AF2_Filtered)",
    )
    parser.add_argument(
        "--features-dir",
        default="../fitting/jaxENT/_featurise",
        help="Directory containing featurized data (features_*.npz and topology_*.json)",
    )
    parser.add_argument(
        "--datasplit-dir",
        default="../fitting/jaxENT/_datasplits",
        help="Directory containing data splits (train/val/full datasets).",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results (results_EMA.hdf5). Default: False",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided results/output/clustering/features directories as absolute paths",
    )
    args = parser.parse_args()

    # Define parameters (should match those used in optimization)
    ensembles = ["AF2_MSAss", "AF2_filtered"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    bv_reg_functions = ["L1", "L2"]
    num_splits = 3
    convergence_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # Resolve paths
    script_dir = os.path.dirname(__file__)
    if args.absolute_paths:
        results_dir = args.results_dir
        clustering_dir = args.clustering_dir
        features_dir = args.features_dir
        datasplit_dir = args.datasplit_dir
    else:
        results_dir = os.path.abspath(os.path.join(script_dir, args.results_dir))
        clustering_dir = os.path.abspath(os.path.join(script_dir, args.clustering_dir))
        features_dir = os.path.abspath(os.path.join(script_dir, args.features_dir))
        datasplit_dir = os.path.abspath(os.path.join(script_dir, args.datasplit_dir))

    if args.output_dir:
        if args.absolute_paths:
            output_base_dir = args.output_dir
        else:
            output_base_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_base_dir = os.path.join(os.path.dirname(results_dir), f"_processed_{base_name}")
    
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Resolved results_dir: {results_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved features_dir: {features_dir}")
    print(f"Resolved datasplit_dir: {datasplit_dir}")
    print(f"Resolved output_base_dir: {output_base_dir}")
    print(f"EMA flag: {args.ema}")
    print("-" * 60)

    # Load cluster assignments
    print("Loading cluster assignments...")
    clustering_results = load_clustering_results(clustering_dir)

    # Load all optimization results
    print("Loading optimization results...")
    all_optim_results = load_all_optimization_results_2d(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        bv_reg_functions=bv_reg_functions,
        num_splits=num_splits,
        EMA=args.ema,
    )

    if not all_optim_results:
        print("No optimization results found. Exiting.")
        return

    # Reorganize results by ensemble first for efficiency
    results_by_ensemble = {}
    for split_type, ensembles_data in all_optim_results.items():
        for ensemble, loss_data in ensembles_data.items():
            if ensemble not in results_by_ensemble:
                results_by_ensemble[ensemble] = {}
            results_by_ensemble[ensemble][split_type] = loss_data

    # Mapping of ensemble names to feature file names (case-sensitive for MoPrP)
    ensemble_feature_map = {
        "AF2_MSAss": "AF2_MSAss",
        "AF2_filtered": "AF2_filtered",
    }

    # Process each optimization run, iterating by ensemble first
    print("\nProcessing each optimization run...")
    for ensemble, split_type_data in results_by_ensemble.items():
        # Load features and topology for the current ensemble (once per ensemble)
        # Use the ensemble feature map to get the correct case-sensitive name
        feature_name = ensemble_feature_map.get(ensemble, ensemble.lower())
        feature_path = os.path.join(features_dir, f"features_{feature_name}.npz")
        topology_path = feature_path.replace("features_", "topology_").replace(".npz", ".json")

        if not os.path.exists(feature_path) or not os.path.exists(topology_path):
            print(f"  Skipping {ensemble}: Features or topology not found at {feature_path} / {topology_path}")
            continue
        
        print(f"\nLoading features for ensemble: {ensemble}")
        features = BV_input_features.load(feature_path)
        feature_top = pt.PTSerialiser.load_list_from_json(topology_path)

        # --- Get timepoints from data ---
        # Find a representative split_idx to load one data file
        first_split_idx = None
        first_split_type = None
        for split_type_from_loop, loss_data in split_type_data.items():
            for loss_name, bv_reg_data in loss_data.items():
                for bv_reg_fn, maxent_data in bv_reg_data.items():
                    for maxent_val, bvreg_val_data in maxent_data.items():
                        for bvreg_val, splits_data in bvreg_val_data.items():
                            if splits_data:
                                first_split_idx = next(iter(splits_data))
                                first_split_type = split_type_from_loop
                                break
                        if first_split_idx is not None: break
                    if first_split_idx is not None: break
                if first_split_idx is not None: break
            if first_split_idx is not None: break
        
        if first_split_idx is None:
            print(f"  No run data found for ensemble {ensemble}. Skipping.")
            continue

        exp_split_type = first_split_type if first_split_type != '_flat' else "random"
        _, _, _, timepoints_from_data = load_experimental_data(results_dir, datasplit_dir, exp_split_type, first_split_idx)
        num_timepoints = len(timepoints_from_data)
        print(f"  Inferred {num_timepoints} timepoints from data file: {timepoints_from_data}")

        # Setup BV model for ln_pf prediction (HDX_resPF)
        bv_config_lnpf = BV_model_Config(num_timepoints=0)
        bv_model_lnpf = BV_model(config=bv_config_lnpf)

        # Setup BV model for uptake prediction (HDX_peptide)
        bv_config_uptake = BV_model_Config(num_timepoints=num_timepoints, timepoints=jnp.array(timepoints_from_data))
        bv_model_uptake = BV_model(config=bv_config_uptake)


        # --- Compute Prior Predictions (from uniform weights and initial params) ---
        print(f"  Computing prior predictions for {ensemble}...")
        n_frames = features.features_shape[1]
        uniform_frame_weights = jnp.ones(n_frames) / n_frames
        
        # Average features using uniform weights
        prior_averaged_features = frame_average_features(features, uniform_frame_weights)

        # Get the forward pass functions from the models
        forward_pass_lnpf = bv_model_lnpf.forward[m_key("HDX_resPF")]
        forward_pass_uptake = BV_uptake_ForwardPass_frames()

        # Run forward pass with averaged features and initial parameters
        prior_lnpf_output = forward_pass_lnpf(prior_averaged_features, bv_model_lnpf.params)
        prior_uptake_output = forward_pass_uptake(prior_averaged_features, bv_model_uptake.params)
        
        prior_ln_pf = prior_lnpf_output.log_Pf
        prior_uptake = prior_uptake_output.uptake

        print(f"  Shapes: prior_ln_pf={prior_ln_pf.shape}, prior_uptake={prior_uptake.shape}")

        for split_type, loss_data in split_type_data.items():
            print(f"  Processing split type: {split_type} for ensemble: {ensemble}")
            if split_type == "_flat":
                current_output_dir = output_base_dir
            else:
                current_output_dir = os.path.join(output_base_dir, split_type)
            os.makedirs(current_output_dir, exist_ok=True)

            # --- Collect all parameters for batching ---
            run_infos = []
            for loss_name, bv_reg_data in loss_data.items():
                for bv_reg_fn, maxent_data in bv_reg_data.items():
                    for maxent_val, bvreg_val_data in maxent_data.items():
                        for bvreg_val, splits_data in bvreg_val_data.items():
                            for split_idx, history in splits_data.items():
                                run_id = f"{ensemble}_{loss_name}_{split_type if split_type != '_flat' else 'flat'}_split{split_idx:03d}_maxent{maxent_val:.1f}_bvreg{bvreg_val:.2f}_bvregfn{bv_reg_fn}"
                                
                                if history is None or not history.states:
                                    # print(f"    Skipping {run_id}: No history found.")
                                    continue

                                # Save the met convergence rates to a file for this run
                                met_convergence_rates = []
                                for i in range(len(history.states)):
                                    if i < len(convergence_rates):
                                        met_convergence_rates.append(convergence_rates[i])

                                # Create a directory for the run and save convergence thresholds
                                run_output_dir_for_run = os.path.join(current_output_dir, run_id)
                                os.makedirs(run_output_dir_for_run, exist_ok=True)
                                with open(os.path.join(run_output_dir_for_run, "convergence_thresholds.txt"), "w") as f:
                                    for rate in met_convergence_rates:
                                        f.write(f"{rate}\n")

                                for i, state in enumerate(history.states):
                                    if i < len(convergence_rates):
                                        convergence_val = convergence_rates[i]
                                    else:
                                        # print(f"    Warning: More states in history than convergence rates defined for {run_id}. Using state index.")
                                        convergence_val = f"state_{i}"

                                    if not hasattr(state, "params") or state.params is None:
                                        continue

                                    run_infos.append({
                                        "run_id": run_id,
                                        "split_idx": split_idx,
                                        "params": state.params,
                                        "losses": state.losses,
                                        "convergence": convergence_val,
                                    })

            if not run_infos:
                print(f"  No valid runs found for ensemble {ensemble} and split_type {split_type}. Skipping.")
                continue

            # Filter out runs with missing parameters before batching
            valid_run_infos = [
                info for info in run_infos 
                if (info["params"] is not None and 
                    info["params"].frame_weights is not None and 
                    info["params"].model_parameters is not None)
            ]

            if not valid_run_infos:
                print(f"  No runs with complete parameters for ensemble {ensemble} and split_type {split_type}. Skipping.")
                continue

            # --- Process and Save Individual Results ---
            # Group runs by run_id
            runs_to_process = {}
            for info in valid_run_infos:
                run_id = info['run_id']
                if run_id not in runs_to_process:
                    runs_to_process[run_id] = []
                runs_to_process[run_id].append(info)
            
            print(f"  Processing {len(runs_to_process)} runs for {ensemble}/{split_type}...")

            for run_id, infos in runs_to_process.items():
                
                all_pred_ln_pf = []
                all_pred_uptake = []
                all_kl_div = []
                all_frame_weights = []
                all_cluster_ratios = []
                all_val_losses = []
                all_bv_bc = []
                all_bv_bh = []

                # The infos are already sorted by convergence because of how they were added.
                for info in infos:
                    params = info["params"]
                    losses = info["losses"]
                    convergence = info["convergence"]

                    # --- Efficiently calculate reweighted predictions ---
                    frame_weights = jnp.array(params.frame_weights)
                    
                    # Average features using frame_weights
                    averaged_features = frame_average_features(features, frame_weights)
                    
                    # Extract optimized BV parameters
                    # Assuming params.model_parameters has bv_bc and bv_bh
                    # We need to construct a BV_Model_Parameters object
                    # Note: We use the timepoints from data, assuming they are constant
                    
                    # params.model_parameters is a list, so we take the first element
                    model_params = params.model_parameters[0]

                    optimized_bv_params = BV_Model_Parameters(
                        bv_bc=model_params.bv_bc,
                        bv_bh=model_params.bv_bh,
                        timepoints=jnp.array(timepoints_from_data)
                    )
                    
                    # Extract scalar values safely (handling 0-d or 1-d arrays)
                    bc_scalar = np.array(model_params.bv_bc).ravel()[0]
                    bh_scalar = np.array(model_params.bv_bh).ravel()[0]
                    
                    all_bv_bc.append(float(bc_scalar))
                    all_bv_bh.append(float(bh_scalar))

                    # Run forward pass with averaged features and optimized parameters
                    pred_lnpf_output = forward_pass_lnpf(averaged_features, optimized_bv_params)
                    pred_uptake_output = forward_pass_uptake(averaged_features, optimized_bv_params)

                    pred_ln_pf = pred_lnpf_output.log_Pf
                    pred_uptake = pred_uptake_output.uptake
                    
                    all_pred_ln_pf.append(pred_ln_pf)
                    all_pred_uptake.append(pred_uptake)
                    all_frame_weights.append(frame_weights)

                    # Extract frame weights and KL-divergence
                    kl_div = 0.0
                    if len(frame_weights) > 0 and np.sum(frame_weights) > 0:
                        uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)
                        kl_div = kl_divergence(np.array(frame_weights), uniform_prior)
                    all_kl_div.append(kl_div)
                    
                    # Extract validation loss
                    val_loss = np.nan
                    if losses is not None and hasattr(losses, 'val_losses') and losses.val_losses is not None:
                        try:
                            # Take the first element of val_losses
                            val_loss = float(losses.val_losses[0])
                        except (IndexError, TypeError, ValueError):
                            pass
                    all_val_losses.append(val_loss)

                    # Compute cluster population ratios
                    cluster_ratios = {}
                    if ensemble in clustering_results:
                        cluster_assignments = clustering_results[ensemble]["cluster_assignments"]
                        if len(frame_weights) == len(cluster_assignments):
                            cluster_ratios = calculate_cluster_ratios(cluster_assignments, np.array(frame_weights))
                    
                    cr_with_conv = {"convergence": convergence}
                    cr_with_conv.update(cluster_ratios)
                    all_cluster_ratios.append(cr_with_conv)

                # Save combined results
                run_output_dir = os.path.join(current_output_dir, run_id)
                
                if all_pred_ln_pf:
                    np.save(os.path.join(run_output_dir, "pred_ln_pf.npy"), np.stack(all_pred_ln_pf))
                if all_pred_uptake:
                    np.save(os.path.join(run_output_dir, "pred_uptake.npy"), np.stack(all_pred_uptake))
                if all_kl_div:
                    np.save(os.path.join(run_output_dir, "kl_divergence.npy"), np.array(all_kl_div))
                if all_frame_weights:
                    np.save(os.path.join(run_output_dir, "frame_weights.npy"), np.stack(all_frame_weights))
                
                if all_val_losses:
                    np.save(os.path.join(run_output_dir, "val_loss.npy"), np.array(all_val_losses))
                
                if all_bv_bc:
                    np.save(os.path.join(run_output_dir, "bv_bc.npy"), np.array(all_bv_bc))
                if all_bv_bh:
                    np.save(os.path.join(run_output_dir, "bv_bh.npy"), np.array(all_bv_bh))
                
                # Priors are constant for the run, save them once
                np.save(os.path.join(run_output_dir, "prior_ln_pf.npy"), np.array(prior_ln_pf))
                np.save(os.path.join(run_output_dir, "prior_uptake.npy"), np.array(prior_uptake))

                # Save cluster ratios to CSV
                if all_cluster_ratios:
                    cluster_df = pd.DataFrame(all_cluster_ratios)
                    # Reorder columns to have 'convergence' first
                    cols = ['convergence'] + [col for col in cluster_df.columns if col != 'convergence']
                    cluster_df = cluster_df[cols]
                    cluster_df.to_csv(os.path.join(run_output_dir, "cluster_ratios.csv"), index=False)


    print("\nAll optimization results processed successfully!")
    print(f"Outputs saved to: {output_base_dir}")

if __name__ == "__main__":
    main()