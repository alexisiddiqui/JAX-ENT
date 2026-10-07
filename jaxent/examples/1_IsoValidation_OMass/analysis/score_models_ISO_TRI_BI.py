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
from jaxent.examples.common.manifest import (
    load_processing_manifest, atomic_to_csv, ConvergenceLabelMismatchError,
)

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
    parser.add_argument(
        "--allow-legacy-without-manifest",
        action="store_true",
        help="Allow known legacy processed artifacts that predate manifest files.",
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
    load_processing_manifest(
        processed_data_dir,
        allow_legacy_missing=args.allow_legacy_without_manifest,
    )

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

            data_cache[cache_key] = {
                "loader": full_loader,
                "y_true_val": analysis.get_experimental_uptake(val_data),
                "y_true_test": analysis.get_experimental_uptake(test_data),
            }
        except ConvergenceLabelMismatchError:
            raise
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
            y_true_val = cached["y_true_val"]
            y_true_test = cached["y_true_test"]

            conv_path = os.path.join(full_run_path, "convergence_thresholds.txt")
            if not os.path.exists(conv_path):
                continue
            with open(conv_path) as f:
                convergence_thresholds = [float(line.strip()) for line in f]

            pred_uptake_stack = np.load(os.path.join(full_run_path, "pred_uptake.npy"))
            if len(pred_uptake_stack) != len(convergence_thresholds):
                raise ConvergenceLabelMismatchError(
                    f"inconsistent stack lengths for {run_id}"
                )

            val_map = loader.val.residue_feature_ouput_mapping
            test_map = loader.test.residue_feature_ouput_mapping

            for i, convergence_val in enumerate(convergence_thresholds):
                pred_uptake = pred_uptake_stack[i]
                mapped_pred_val = np.array(
                    [apply_sparse_mapping(val_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]
                ).T
                mapped_pred_test = np.array(
                    [apply_sparse_mapping(test_map, pred_uptake[t]) for t in range(pred_uptake.shape[0])]
                ).T

                scores_entry = {
                    "ensemble": ensemble,
                    "loss_function": run_info["loss_name"],
                    "split_type": run_info["actual_split_type"],
                    "split_idx": run_info["split_idx"],
                    "maxent_value": run_info["maxent_value"],
                    "convergence_value": convergence_val,
                    "val_mse": analysis.calculate_mse(mapped_pred_val, y_true_val),
                    "test_mse": analysis.calculate_mse(mapped_pred_test, y_true_test),
                }
                all_scores.append(scores_entry)

        except ConvergenceLabelMismatchError:
            raise
        except Exception as e:
            import traceback
            print(f"  ERROR processing {run_id}: {e}")
            traceback.print_exc()

    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        output_csv = os.path.join(output_scores_dir, "model_scores.csv")
        atomic_to_csv(scores_df, output_csv)
        print(f"\nSaved {len(scores_df)} rows to: {output_csv}")
    else:
        print("\nNo scores were generated.")


if __name__ == "__main__":
    main()
