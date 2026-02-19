"""
process_optimisation_results.py

Processes raw optimization outputs into a structured format for scoring/plotting.

Requirements:
    - Optimization results (results_EMA.hdf5)
    - Data splits (_datasplits/)
    - Features (_featurise/)
    - Clustering results (_clustering_results/)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/process_optimisation_results.py --results-dir ...

Output:
    - Processed data directory (e.g., _processed_...)
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


def main():
    parser = argparse.ArgumentParser(
        description="Process optimization results to extract predictions, KL divergence, and cluster ratios."
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_test_SIGMA_500__20260216_224925",
        help="Directory containing optimization HDF5 result files (e.g., from optimise_ISO_TRI_BI_splits_Sigma.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for processed .npy files. If omitted, derived from results-dir basename prefixed with '_processed'.",
    )
    parser.add_argument(
        "--clustering-dir",
        default="../data/_clustering_results",
        help="Directory containing cluster assignment CSV files (e.g., from clustering_analysis.py)",
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
    ensembles = ["ISO_TRI", "ISO_BI"]
    ensemble_feature_map = {ens: ens.lower() for ens in ensembles}
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
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
    clustering_results = loading.load_clustering_results(clustering_dir)

    # Load all optimization results
    print("Loading optimization results...")
    all_optim_results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
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
        # Find a representative split_idx to load one data file
        first_split_idx = None
        first_split_type = None
        for split_type_from_loop, loss_data in split_type_data.items():
            for maxent_data in loss_data.values():
                for splits_data in maxent_data.values():
                    if splits_data:
                        first_split_idx = next(iter(splits_data))
                        first_split_type = split_type_from_loop
                        break
                if first_split_idx is not None:
                    break
            if first_split_idx is not None:
                break

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

        # --- Compute frame-wise predictions (once per ensemble) ---
        print(f"  Computing frame-wise predictions for {ensemble}...")

        forward_pass_lnpf = bv_model_lnpf.forward[m_key("HDX_resPF")]
        forward_pass_uptake = BV_uptake_ForwardPass_frames()

        framewise_output_lnpf = forward_pass_lnpf(features, bv_model_lnpf.params)
        framewise_output_uptake = forward_pass_uptake(features, bv_model_uptake.params)

        # --- Compute Prior Predictions (from uniform weights) ---
        print(f"  Computing prior predictions for {ensemble}...")
        n_frames = features.features_shape[1]
        uniform_frame_weights = jnp.ones(n_frames) / n_frames

        prior_lnpf_output = frame_average_features(framewise_output_lnpf, uniform_frame_weights)
        prior_uptake_output = frame_average_features(framewise_output_uptake, uniform_frame_weights)

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
            for loss_name, maxent_data in loss_data.items():
                for maxent_val, splits_data in maxent_data.items():
                    for split_idx, history in splits_data.items():
                        run_id = f"{ensemble}_{loss_name}_{split_type if split_type != '_flat' else 'flat'}_split{split_idx:03d}_maxent{maxent_val:.1f}"

                        if history is None or not history.states:
                            print(f"    Skipping {run_id}: No history found.")
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
                                print(f"    Warning: More states in history than convergence rates defined for {run_id}. Using state index.")
                                convergence_val = f"state_{i}"

                            if not hasattr(state, "params") or state.params is None:
                                print(f"    Skipping {run_id} for convergence {convergence_val}: No parameters in state.")
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

                for info in infos:
                    params = info["params"]
                    losses = info["losses"]
                    convergence = info["convergence"]

                    frame_weights = jnp.array(params.frame_weights)

                    pred_lnpf_output = frame_average_features(framewise_output_lnpf, frame_weights)
                    pred_uptake_output = frame_average_features(framewise_output_uptake, frame_weights)

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
                        else:
                            print(f"    Warning: Frame weights length mismatch for {run_id} at convergence {convergence}. Skipping cluster ratios.")
                    else:
                        print(f"    Warning: No clustering results for {ensemble}. Skipping cluster ratios.")

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
