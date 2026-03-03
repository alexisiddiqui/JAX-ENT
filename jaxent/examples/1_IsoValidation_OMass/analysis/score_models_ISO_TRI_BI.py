"""
score_models_ISO_TRI_BI.py  (Exp1 — IsoValidation_OMass)

Computes scores (MSE, dMSE, work metrics, recovery %) for optimized models.
Loads outputs of process_optimisation_results.py and writes model_scores.csv.

Requirements:
    - Processed data directory (_processed_...)
    - Data splits (_datasplits/)
    - Features (_featurise/)
    - Clustering results (_clustering_results/)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/score_models_ISO_TRI_BI.py \
        --processed-data-dir ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig

ENSEMBLE_PATTERN = r"(ISO_BI|ISO_TRI)_(mcMSE|MSE|Sigma_MSE)_(.+?)_split(\d+)_maxent([\d.]+)"


def _resolve_effective_split_type(run_split_type: str, datasplit_dir: str) -> str:
    """Return the datasplit directory name that actually exists for this run_split_type."""
    if "_cluster" in run_split_type:
        full_path = os.path.join(datasplit_dir, run_split_type)
        stripped = run_split_type.replace("_cluster", "")
        stripped_path = os.path.join(datasplit_dir, stripped)
        if os.path.exists(full_path):
            return run_split_type
        elif os.path.exists(stripped_path):
            return stripped
        return stripped
    return run_split_type


def main():
    parser = argparse.ArgumentParser(
        description="Calculate scores and metrics from processed Exp1 optimization results."
    )
    parser.add_argument(
        "--processed-data-dir",
        default="../fitting/jaxENT/_optimise_test_SIGMA_500__20260216_224925",
        help="Directory containing processed .npy files from process_optimisation_results.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, creates '_scores_<basename>' inside processed-data-dir.",
    )
    parser.add_argument(
        "--datasplit-dir",
        default="../fitting/jaxENT/_datasplits",
        help="Directory containing data splits (train/val/full datasets).",
    )
    parser.add_argument(
        "--features-dir",
        default="../fitting/jaxENT/_featurise",
        help="Directory containing featurized data (features_*.npz and topology_*.json).",
    )
    parser.add_argument(
        "--clustering-dir",
        default="../data/_clustering_results",
        help="Directory containing cluster assignment CSV files.",
    )
    parser.add_argument(
        "--config",
        default="../config.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided directories as absolute paths.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    if args.absolute_paths:
        processed_data_dir = args.processed_data_dir
        datasplit_dir = args.datasplit_dir
        features_dir = args.features_dir
        clustering_dir = args.clustering_dir
        config_path = args.config
    else:
        processed_data_dir = os.path.abspath(os.path.join(script_dir, args.processed_data_dir))
        datasplit_dir = os.path.abspath(os.path.join(script_dir, args.datasplit_dir))
        features_dir = os.path.abspath(os.path.join(script_dir, args.features_dir))
        clustering_dir = os.path.abspath(os.path.join(script_dir, args.clustering_dir))
        config_path = os.path.abspath(os.path.join(script_dir, args.config))

    if args.output_dir:
        output_scores_dir = (
            args.output_dir
            if args.absolute_paths
            else os.path.abspath(os.path.join(script_dir, args.output_dir))
        )
    else:
        basename = os.path.basename(processed_data_dir.rstrip("/"))
        output_scores_dir = os.path.join(processed_data_dir, f"_scores_{basename}")
    os.makedirs(output_scores_dir, exist_ok=True)

    print(f"processed_data_dir: {processed_data_dir}")
    print(f"datasplit_dir:      {datasplit_dir}")
    print(f"features_dir:       {features_dir}")
    print(f"clustering_dir:     {clustering_dir}")
    print(f"output_scores_dir:  {output_scores_dir}")
    print("-" * 60)

    # Load config
    config = ExperimentConfig.from_yaml(config_path)
    target_ratios = config.scoring.ground_truth_ratios
    state_mapping = config.scoring.state_mapping
    ensemble_feature_map = config.scoring.ensemble_feature_map

    # Load clustering
    clustering_data = loading.load_clustering_results(clustering_dir)

    # First pass: collect run metadata
    all_run_info, unique_configs = loading.load_processed_run_info(
        processed_data_dir, ENSEMBLE_PATTERN
    )

    # Augment each run_info with effective_split_type
    for r in all_run_info:
        r["effective_split_type"] = _resolve_effective_split_type(
            r["run_split_type"], datasplit_dir
        )

    unique_configs = sorted(
        set((r["ensemble"], r["effective_split_type"], r["split_idx_str"]) for r in all_run_info)
    )

    # Pre-cache loop
    data_cache = {}
    print("--- Pre-caching data ---")
    for ensemble, effective_split_type, split_idx_str in unique_configs:
        cache_key = (ensemble, effective_split_type, split_idx_str)
        print(f"Caching {cache_key}...")

        runs_in_group = [
            r for r in all_run_info
            if (r["ensemble"], r["effective_split_type"], r["split_idx_str"]) == cache_key
        ]
        valid_run = next(
            (
                r for r in runs_in_group
                if os.path.exists(os.path.join(r["full_run_path"], "prior_ln_pf.npy"))
                and os.path.exists(os.path.join(r["full_run_path"], "prior_uptake.npy"))
            ),
            None,
        )
        if valid_run is None:
            print(f"  ERROR: No valid run with prior files for {cache_key}. Skipping.")
            continue

        try:
            train_data, val_data, test_data, _ = loading.load_experimental_data(
                processed_data_dir, datasplit_dir, effective_split_type, int(split_idx_str)
            )
            features, feature_top = loading.load_features_and_topology(
                features_dir, ensemble, ensemble_feature_map
            )

            full_loader = ExpD_Dataloader(data=test_data)
            full_loader.create_datasets(
                features=features,
                feature_topology=feature_top,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
            )

            prior_ln_pf = np.load(os.path.join(valid_run["full_run_path"], "prior_ln_pf.npy"))
            prior_uptake = np.load(os.path.join(valid_run["full_run_path"], "prior_uptake.npy"))

            train_map = full_loader.train.residue_feature_ouput_mapping
            val_map = full_loader.val.residue_feature_ouput_mapping
            test_map = full_loader.test.residue_feature_ouput_mapping

            mapped_prior_train = np.array(
                [apply_sparse_mapping(train_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]
            ).T
            mapped_prior_val = np.array(
                [apply_sparse_mapping(val_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]
            ).T
            mapped_prior_test = np.array(
                [apply_sparse_mapping(test_map, prior_uptake[t]) for t in range(prior_uptake.shape[0])]
            ).T

            data_cache[cache_key] = {
                "loader": full_loader,
                "y_true_train": analysis.get_experimental_uptake(train_data),
                "y_true_val": analysis.get_experimental_uptake(val_data),
                "y_true_test": analysis.get_experimental_uptake(test_data),
                "prior_ln_pf": prior_ln_pf,
                "mapped_prior_train": mapped_prior_train,
                "mapped_prior_val": mapped_prior_val,
                "mapped_prior_test": mapped_prior_test,
            }
        except Exception as e:
            import traceback
            print(f"  ERROR caching {cache_key}: {e}")
            traceback.print_exc()
    print("--- Caching complete ---")

    all_scores = []

    print("\n--- Processing runs ---")
    for run_info in tqdm(all_run_info, desc="Processing runs"):
        run_id = run_info["run_id"]
        full_run_path = run_info["full_run_path"]
        try:
            ensemble = run_info["ensemble"]
            cache_key = (ensemble, run_info["effective_split_type"], run_info["split_idx_str"])

            if cache_key not in data_cache:
                print(f"  Warning: cache miss for {cache_key}. Skipping {run_id}.")
                continue

            cached = data_cache[cache_key]
            loader = cached["loader"]
            y_true_train = cached["y_true_train"]
            y_true_val = cached["y_true_val"]
            y_true_test = cached["y_true_test"]
            prior_ln_pf = cached["prior_ln_pf"]
            mapped_prior_train = cached["mapped_prior_train"]
            mapped_prior_val = cached["mapped_prior_val"]
            mapped_prior_test = cached["mapped_prior_test"]

            conv_path = os.path.join(full_run_path, "convergence_thresholds.txt")
            if not os.path.exists(conv_path):
                continue
            with open(conv_path) as f:
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

            if len(pred_ln_pf_stack) != len(convergence_thresholds):
                print(f"  Warning: inconsistent stack lengths for {run_id}. Skipping.")
                continue

            train_map = loader.train.residue_feature_ouput_mapping
            val_map = loader.val.residue_feature_ouput_mapping
            test_map = loader.test.residue_feature_ouput_mapping

            cluster_assignments = (clustering_data.get(ensemble, {}) or {}).get("cluster_assignments")

            for i, convergence_val in enumerate(convergence_thresholds):
                pred_ln_pf = pred_ln_pf_stack[i]
                pred_uptake = pred_uptake_stack[i]
                kl_div = kl_divergence_stack[i]
                frame_weights = frame_weights_stack[i]

                val_loss = np.nan
                if val_loss_stack is not None and i < len(val_loss_stack):
                    val_loss = val_loss_stack[i]

                mapped_pred_train = np.array(
                    [apply_sparse_mapping(train_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]
                ).T
                mapped_pred_val = np.array(
                    [apply_sparse_mapping(val_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]
                ).T
                mapped_pred_test = np.array(
                    [apply_sparse_mapping(test_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]
                ).T

                work_metrics = analysis.calculate_work_metrics(pred_ln_pf, prior_ln_pf)

                recovery_percent = np.nan
                if cluster_assignments is not None:
                    recovery_percent = analysis.calculate_recovery_percentage(
                        cluster_assignments, frame_weights, target_ratios, state_mapping
                    )

                cluster_ratios = {}
                if cluster_ratios_df is not None:
                    row = cluster_ratios_df[cluster_ratios_df["convergence"] == convergence_val]
                    if not row.empty:
                        cluster_ratios = row.iloc[0].to_dict()
                        cluster_ratios.pop("convergence", None)

                scores_entry = {
                    "ensemble": ensemble,
                    "loss_function": run_info["loss_name"],
                    "split_type": run_info["actual_split_type"],
                    "split_idx": run_info["split_idx"],
                    "maxent_value": run_info["maxent_value"],
                    "convergence_value": convergence_val,
                    "kl_divergence": kl_div,
                    "train_mse": analysis.calculate_mse(mapped_pred_train, y_true_train),
                    "val_mse": analysis.calculate_mse(mapped_pred_val, y_true_val),
                    "test_mse": analysis.calculate_mse(mapped_pred_test, y_true_test),
                    "d_mse_train": analysis.calculate_dMSE(mapped_pred_train, mapped_prior_train, y_true_train),
                    "d_mse_val": analysis.calculate_dMSE(mapped_pred_val, mapped_prior_val, y_true_val),
                    "d_mse_test": analysis.calculate_dMSE(mapped_pred_test, mapped_prior_test, y_true_test),
                    "work_scale_kj": work_metrics.get("work_scale_kj", np.nan),
                    "work_shape_kj": work_metrics.get("work_shape_kj", np.nan),
                    "work_density_kj": work_metrics.get("work_density_kj", np.nan),
                    "work_fitting_kj": work_metrics.get("work_fitting_kj", np.nan),
                    "work_magnitude_kj": work_metrics.get("work_magnitude_kj", np.nan),
                    "recovery_percent": recovery_percent,
                    "val_loss": val_loss,
                }
                scores_entry.update(cluster_ratios)
                all_scores.append(scores_entry)

        except Exception as e:
            import traceback
            print(f"  ERROR processing {run_id}: {e}")
            traceback.print_exc()

    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        output_csv = os.path.join(output_scores_dir, "model_scores.csv")
        scores_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(scores_df)} rows to: {output_csv}")
        print("\n--- Generating violin plots ---")
        plotting.create_violin_plots(scores_df, output_scores_dir)
    else:
        print("\nNo scores were generated.")


if __name__ == "__main__":
    main()
