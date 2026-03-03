"""
Process MoPrP Optimization Results

Function:
Processes HDF5 optimization history files to extract:
- Residue-wise log protection factors (lnPF) and peptide uptake predictions.
- KL divergence of frame weights relative to a uniform prior.
- Cluster population ratios based on external clustering assignments.
Outputs processed data as .npy and .csv files for downstream scoring.

Requirements:
- `--results-dir`: Directory containing `results.hdf5` files.
- `--clustering-dir`: Directory with `frame_to_cluster.csv` files (e.g., from example 2).
- `--features-dir`: Directory with `features_*.npz` and `topology_*.json` (from example 2).
- `--datasplit-dir`: Directory with data splits (from example 2).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

# Add the base directory to the path to import JAX-ENT modules
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.models.HDX.BV.features import BV_input_features, uptake_BV_output_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.custom_types.key import m_key
from jaxent.src.utils.jax_fn import frame_average_features

# common modules
from jaxent.examples.common import analysis, loading, paths
from jaxent.examples.common.paths import derive_processed_output_dir, resolve_script_paths
from jaxent.examples.common.optimization import BV_uptake_ForwardPass_frames


class BV_uptake_ForwardPass_averaged(BV_uptake_ForwardPass_frames):
    """Variant for averaged (non-per-frame) input features — adds pf reshape."""
    def __call__(self, input_features, parameters):
        bc, bh = parameters.bv_bc, parameters.bv_bh
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)
        time_points = parameters.timepoints.reshape(-1)
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)
        pf = jnp.exp(log_pf)
        if pf.ndim == 1:
            pf = pf.reshape(-1, 1)

        def compute_uptake_for_timepoint(timepoint):
            kints_reshaped = kints.reshape(-1, 1)
            uptake = 1 - jnp.exp(-kints_reshaped * timepoint / pf)
            return uptake

        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)
        return uptake_BV_output_features(uptake_per_timepoint)


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
    ensemble_feature_map = {"AF2_MSAss": "AF2_MSAss", "AF2_filtered": "AF2_filtered"}
    ensemble_clustering_map = {"AF2_MSAss": "AF2_MSAss", "AF2_filtered": "AF2_Filtered"}
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    bv_reg_functions = ["L1", "L2"]
    num_splits = 3
    convergence_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # Resolve paths
    resolved = resolve_script_paths(args, Path(__file__).parent)
    results_dir    = resolved["results_dir"]
    clustering_dir = resolved["clustering_dir"]
    features_dir   = resolved["features_dir"]
    datasplit_dir  = resolved["datasplit_dir"]
    output_base_dir = resolved.get("output_dir") or str(derive_processed_output_dir(results_dir))
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
    clustering_results = loading.load_clustering_results(clustering_dir, ensemble_clustering_map)

    # Load all optimization results (2D sweep)
    print("Loading optimization results...")
    all_optim_results = loading.load_all_optimization_results_2d(
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

    # Process each optimization run, iterating by ensemble first
    print("\nProcessing each optimization run...")
    for ensemble, split_type_data in results_by_ensemble.items():
        # Load features and topology for the current ensemble (once per ensemble)
        try:
            features, feature_top = loading.load_features_and_topology(features_dir, ensemble, ensemble_feature_map)
        except FileNotFoundError as e:
            print(f"  Skipping {ensemble}: {e}")
            continue

        print(f"\nLoading features for ensemble: {ensemble}")

        # --- Get timepoints from data ---
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
        _, _, _, timepoints_from_data = loading.load_experimental_data(results_dir, datasplit_dir, exp_split_type, first_split_idx)
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

        # Forward pass functions
        forward_pass_lnpf = bv_model_lnpf.forward[m_key("HDX_resPF")]
        forward_pass_uptake = BV_uptake_ForwardPass_averaged()

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
                                    continue

                                met_convergence_rates = []
                                for i in range(len(history.states)):
                                    if i < len(convergence_rates):
                                        met_convergence_rates.append(convergence_rates[i])

                                run_output_dir_for_run = os.path.join(current_output_dir, run_id)
                                os.makedirs(run_output_dir_for_run, exist_ok=True)
                                with open(os.path.join(run_output_dir_for_run, "convergence_thresholds.txt"), "w") as f:
                                    for rate in met_convergence_rates:
                                        f.write(f"{rate}\n")

                                for i, state in enumerate(history.states):
                                    if i < len(convergence_rates):
                                        convergence_val = convergence_rates[i]
                                    else:
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

            valid_run_infos = [
                info for info in run_infos
                if (info["params"] is not None and
                    info["params"].frame_weights is not None and
                    info["params"].model_parameters is not None)
            ]

            if not valid_run_infos:
                print(f"  No runs with complete parameters for ensemble {ensemble} and split_type {split_type}. Skipping.")
                continue

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

                for info in infos:
                    params = info["params"]
                    losses = info["losses"]
                    convergence = info["convergence"]

                    frame_weights = jnp.array(params.frame_weights)

                    # Average features using frame_weights
                    averaged_features = frame_average_features(features, frame_weights)

                    # Extract optimized BV parameters from first model
                    model_params = params.model_parameters[0]
                    optimized_bv_params = BV_Model_Parameters(
                        bv_bc=model_params.bv_bc,
                        bv_bh=model_params.bv_bh,
                        timepoints=jnp.array(timepoints_from_data)
                    )

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

                    kl_div = 0.0
                    if len(frame_weights) > 0 and np.sum(frame_weights) > 0:
                        uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)
                        kl_div = analysis.kl_divergence(np.array(frame_weights), uniform_prior)
                    all_kl_div.append(kl_div)

                    val_loss = np.nan
                    if losses is not None and hasattr(losses, 'val_losses') and losses.val_losses is not None:
                        try:
                            val_loss = float(losses.val_losses[0])
                        except (IndexError, TypeError, ValueError):
                            pass
                    all_val_losses.append(val_loss)

                    cluster_ratios = {}
                    if ensemble in clustering_results:
                        cluster_assignments = clustering_results[ensemble]["cluster_assignments"]
                        if len(frame_weights) == len(cluster_assignments):
                            cluster_ratios = analysis.calculate_cluster_ratios(cluster_assignments, np.array(frame_weights))

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

                np.save(os.path.join(run_output_dir, "prior_ln_pf.npy"), np.array(prior_ln_pf))
                np.save(os.path.join(run_output_dir, "prior_uptake.npy"), np.array(prior_uptake))

                if all_cluster_ratios:
                    cluster_df = pd.DataFrame(all_cluster_ratios)
                    cols = ['convergence'] + [col for col in cluster_df.columns if col != 'convergence']
                    cluster_df = cluster_df[cols]
                    cluster_df.to_csv(os.path.join(run_output_dir, "cluster_ratios.csv"), index=False)


    print("\nAll optimization results processed successfully!")
    print(f"Outputs saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
