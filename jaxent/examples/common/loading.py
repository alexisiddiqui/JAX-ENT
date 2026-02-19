"""
Shared data loading functions for JAX-ENT example scripts.

Consolidates the 5+ near-identical ``load_all_optimization_results`` variants,
``load_clustering_results``, ``load_features_and_topology``, and other loading
helpers scattered across example scripts.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from jaxent.src.utils.hdf import load_optimization_history_from_file


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------


def extract_maxent_value_from_filename(filename: str) -> float | None:
    """Extract the maxent value from a result filename.

    The convention is ``..._maxent<value>_...`` where *value* is a float.
    Returns ``None`` if no match is found.
    """
    match = re.search(r"_maxent(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Optimisation result loading
# ---------------------------------------------------------------------------


def load_all_optimization_results(
    results_dir: str,
    split_type: str | None = None,
    ensembles: List[str] | None = None,
    loss_functions: List[str] | None = None,
    num_splits: int = 3,
    EMA: bool = False,
    maxent_values: List[float] | None = None,
) -> Dict:
    """Load optimisation results from HDF5 files for a single split type.

    Returns ``{ensemble: {loss_name: {split_idx: {maxent_val: history}}}}``.
    """
    if ensembles is None:
        ensembles = []
    if loss_functions is None:
        loss_functions = []

    results: Dict = {}
    load_dir = os.path.join(results_dir, split_type) if split_type else results_dir
    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return results

    hdf_suffix = "results_EMA.hdf5" if EMA else "results.hdf5"

    for ensemble in ensembles:
        results[ensemble] = {}
        for loss_name in loss_functions:
            results[ensemble][loss_name] = {}
            for split_idx in range(num_splits):
                if split_type:
                    pattern = f"{ensemble}_{loss_name}_{split_type}_split{split_idx:03d}_maxent*_{hdf_suffix}"
                else:
                    pattern = f"{ensemble}_{loss_name}_split{split_idx:03d}_maxent*_{hdf_suffix}"

                matching_files = glob.glob(os.path.join(load_dir, pattern))
                if matching_files:
                    results[ensemble][loss_name][split_idx] = {}
                    for filepath in matching_files:
                        filename = os.path.basename(filepath)
                        maxent_val = extract_maxent_value_from_filename(filename)
                        if maxent_val is not None:
                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[ensemble][loss_name][split_idx][maxent_val] = history
                                print(f"Loaded: {filename}")
                            except Exception as e:
                                print(f"Failed to load {filename}: {e}")
                else:
                    results[ensemble][loss_name][split_idx] = None
    return results


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    num_splits: int,
    EMA: bool = False,
) -> Dict:
    """Load results including maxent values, auto-discovering split types.

    Returns ``{split_type: {ensemble: {loss_name: {maxent_val: {split_idx: history}}}}}``
    """
    results: Dict = {}
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
        split_type_dir = results_dir if split_type == "_flat" else os.path.join(results_dir, split_type)

        if not os.path.exists(split_type_dir):
            continue

        for ensemble in ensembles:
            results[split_type][ensemble] = {}
            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                prefix = f"{ensemble}_{loss_name}_split" if split_type == "_flat" else f"{ensemble}_{loss_name}_{split_type}_split"
                files = [
                    f for f in os.listdir(split_type_dir)
                    if f.startswith(prefix) and f.endswith(hdf_pattern)
                ]

                for filename in files:
                    match = re.search(r"split(\d{3})_maxent(\d+(?:\.\d+)?)", filename)
                    if match:
                        split_idx = int(match.group(1))
                        maxent_val = float(match.group(2))
                    else:
                        match = re.search(r"split(\d{3})", filename)
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = 0.0
                        else:
                            continue

                    if maxent_val not in results[split_type][ensemble][loss_name]:
                        results[split_type][ensemble][loss_name][maxent_val] = {}

                    filepath = os.path.join(split_type_dir, filename)
                    try:
                        history = load_optimization_history_from_file(filepath)
                        results[split_type][ensemble][loss_name][maxent_val][split_idx] = history
                        print(f"Loaded: {filepath}")
                    except Exception as e:
                        print(f"Failed to load {filepath}: {e}")
                        results[split_type][ensemble][loss_name][maxent_val][split_idx] = None

    # Clean up empty _flat placeholder
    if "_flat" in results and not any(results["_flat"].values()):
        del results["_flat"]
    return results


def load_all_optimization_results_2d(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    bv_reg_functions: List[str],
    num_splits: int = 3,
    EMA: bool = False,
    verbose: bool = True,
) -> Dict:
    """Load results for 2D hyperparameter sweeps (maxent + bv_reg).

    Returns ``{split_type: {ensemble: {loss_fn: {bv_reg_fn: {maxent: {bv_reg: {split_idx: history}}}}}}}``.
    """
    results: Dict = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]

    hdf_pattern = "results_EMA.hdf5" if EMA else "results.hdf5"

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        for ensemble in ensembles:
            results[split_type][ensemble] = {}
            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}
                for bv_reg_fn in bv_reg_functions:
                    results[split_type][ensemble][loss_name][bv_reg_fn] = {}

                    files = [
                        f
                        for f in os.listdir(split_type_dir)
                        if f.startswith(f"{ensemble}_{loss_name}_{split_type}_split")
                        and f"bvregfn{bv_reg_fn}" in f
                        and f.endswith(hdf_pattern)
                    ]

                    for filename in files:
                        match = re.search(
                            r"split(\d{3})_maxent([\d.]+)_bvreg([\d.]+)_bvregfn([A-Za-z0-9]+)",
                            filename,
                        )
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = float(match.group(2))
                            bvreg_val = float(match.group(3))
                            
                            if maxent_val not in results[split_type][ensemble][loss_name][bv_reg_fn]:
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val] = {}

                            if bvreg_val not in results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val]:
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][bvreg_val] = {}

                            filepath = os.path.join(split_type_dir, filename)
                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][bvreg_val][split_idx] = history
                            except Exception as e:
                                print(f"Failed to load {filename}: {e}")
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][bvreg_val][split_idx] = None
    return results


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def load_clustering_results(
    clustering_dir: str,
    ensemble_clustering_map: Dict[str, str] | None = None,
) -> Dict:
    """Load clustering results from nested subdirectories.

    Parameters
    ----------
    clustering_dir:
        Base directory containing per-ensemble clustering subdirectories.
    ensemble_clustering_map:
        ``{ensemble_name: subdirectory_name}`` mapping.  When ``None``, the
        subdirectory is discovered automatically.

    Returns
    -------
    ``{ensemble_name: {"cluster_assignments": ndarray, "frame_data": DataFrame}}``
    """
    if not os.path.exists(clustering_dir):
        print(f"Clustering directory not found: {clustering_dir}")
        return {}

    clustering_results: Dict = {}

    # Detect Exp1 flat-CSV format: cluster_assignments_<ensemble>.csv files directly
    # in clustering_dir (as opposed to the nested subdirectory format used by Exp2/3).
    flat_csv_files = glob.glob(os.path.join(clustering_dir, "cluster_assignments_*.csv"))
    if flat_csv_files:
        for csv_path in sorted(flat_csv_files):
            filename = os.path.basename(csv_path)
            ensemble_name = filename.replace("cluster_assignments_", "").replace(".csv", "")
            cluster_df = pd.read_csv(csv_path)
            # Normalise column name: flat format uses "cluster_assignment" (singular),
            # whereas the nested format uses "cluster_label".
            if "cluster_assignment" in cluster_df.columns and "cluster_label" not in cluster_df.columns:
                cluster_df = cluster_df.rename(columns={"cluster_assignment": "cluster_label"})
            clustering_results[ensemble_name] = {
                "cluster_assignments": cluster_df["cluster_label"].values,
                "frame_data": cluster_df,
            }
            print(
                f"Loaded cluster assignments (flat format) for {ensemble_name}: "
                f"{len(cluster_df)} frames"
            )
        return clustering_results

    if ensemble_clustering_map is None:
        # Auto-discover: each subdirectory that contains a *_frame_to_cluster.csv
        for subdir in os.listdir(clustering_dir):
            subdir_path = os.path.join(clustering_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            csv_files = glob.glob(os.path.join(subdir_path, "*_frame_to_cluster.csv"))
            for csv_path in csv_files:
                cluster_df = pd.read_csv(csv_path)
                clustering_results[subdir] = {
                    "cluster_assignments": cluster_df["cluster_label"].values,
                    "frame_data": cluster_df,
                }
                print(f"Loaded cluster assignments for {subdir}: {len(cluster_df)} frames")
    else:
        for ensemble_name, subdir in ensemble_clustering_map.items():
            csv_path = os.path.join(clustering_dir, subdir, f"{subdir}_frame_to_cluster.csv")
            if os.path.exists(csv_path):
                cluster_df = pd.read_csv(csv_path)
                clustering_results[ensemble_name] = {
                    "cluster_assignments": cluster_df["cluster_label"].values,
                    "frame_data": cluster_df,
                }
                print(f"Loaded cluster assignments for {ensemble_name}: {len(cluster_df)} frames")
            else:
                print(f"Warning: Clustering file not found for {ensemble_name} at {csv_path}")

    return clustering_results


def load_clustering_for_ensemble(
    ensemble_name: str,
    clustering_base_dir: str,
    ensemble_clustering_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load clustering results for a single ensemble.

    Returns a DataFrame with a ``cluster_label`` column.
    """
    if ensemble_clustering_map is None:
        subdir = ensemble_name
    else:
        if ensemble_name not in ensemble_clustering_map:
            raise ValueError(
                f"Unknown ensemble: {ensemble_name}. Expected one of {list(ensemble_clustering_map.keys())}"
            )
        subdir = ensemble_clustering_map[ensemble_name]

    csv_path = os.path.join(clustering_base_dir, subdir, f"{subdir}_frame_to_cluster.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clustering file not found: {csv_path}")

    cluster_df = pd.read_csv(csv_path)
    if "cluster_label" not in cluster_df.columns:
        raise ValueError(f"Expected 'cluster_label' column in {csv_path}")
    print(f"Loaded {len(cluster_df)} frames with {cluster_df['cluster_label'].nunique()} clusters for {ensemble_name}")
    return cluster_df


# ---------------------------------------------------------------------------
# Experimental data loading
# ---------------------------------------------------------------------------


def load_experimental_data(
    results_dir: str,
    datasplit_dir: str,
    split_type: str,
    split_idx: int,
) -> Tuple:
    """Load train/val/test experimental data for one split.

    Returns ``(train_data, val_data, test_data, timepoints_array)``.
    """
    from jaxent.src.custom_types.HDX import HDX_peptide

    split_path = os.path.join(datasplit_dir, split_type, f"split_{split_idx:03d}")

    train_csv_path = os.path.join(split_path, "train_dfrac.csv")
    train_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "train_topology.json"),
        csv_path=train_csv_path,
    )

    df_train = pd.read_csv(train_csv_path)
    required_cols = {"datapoint_type", "feature_length"}
    timepoints = [float(col) for col in df_train.columns if col not in required_cols]

    val_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_path, "val_topology.json"),
        csv_path=os.path.join(split_path, "val_dfrac.csv"),
    )
    test_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(datasplit_dir, "full_dataset_topology.json"),
        csv_path=os.path.join(datasplit_dir, "full_dataset_dfrac.csv"),
    )

    return train_data, val_data, test_data, np.array(timepoints)


# ---------------------------------------------------------------------------
# Features & topology
# ---------------------------------------------------------------------------


def load_features_and_topology(
    features_dir: str,
    ensemble_name: str,
    ensemble_feature_map: Dict[str, str] | None = None,
):
    """Load BV features and topology for an ensemble.

    Returns ``(features: BV_input_features, topology: list[Partial_Topology])``.
    """
    from jaxent.src.models.HDX.BV.features import BV_input_features
    import jaxent.src.interfaces.topology as pt

    feature_name = (ensemble_feature_map or {}).get(ensemble_name, ensemble_name)
    feature_path = os.path.join(features_dir, f"features_{feature_name}.npz")
    topology_path = feature_path.replace("features_", "topology_").replace(".npz", ".json")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    features = BV_input_features.load(feature_path)
    feature_top = pt.PTSerialiser.load_list_from_json(topology_path)

    return features, feature_top


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def augment_best_models_with_metrics(
    best_models_df: pd.DataFrame,
    results: Dict,
    clustering_data: Dict[str, pd.DataFrame],
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
) -> pd.DataFrame:
    """Augment best-models DataFrame with KL divergence and JSD recovery metrics."""
    from .analysis import kl_divergence, calculate_recovery_JSD

    augmented_rows: list[dict] = []

    for _, row in best_models_df.iterrows():
        ensemble = row["ensemble"]
        loss_func = row["loss_function"]
        split_idx = int(row["split"])
        maxent_val = row["maxent_value"]
        conv_step = int(row["convergence_step"])
        split_type = row.get("split_type", None)

        # Navigate nested dict
        history = None
        entry = results.get(ensemble, {}).get(loss_func, {}).get(split_idx, None)
        if entry is None:
            continue
        if isinstance(entry, dict):
            history = entry.get(maxent_val) or entry.get(str(maxent_val))
        else:
            history = entry

        if history is None or not hasattr(history, "states") or not history.states:
            continue
        if conv_step <= 0 or conv_step > len(history.states):
            continue

        state = history.states[conv_step - 1]
        if not (hasattr(state, "params") and hasattr(state.params, "frame_weights") and state.params.frame_weights is not None):
            continue

        weights = np.array(state.params.frame_weights)
        uniform_prior = np.ones(len(weights)) / len(weights)
        kl_div = kl_divergence(weights, uniform_prior)

        js_div = np.nan
        recovery_percent = np.nan

        if ensemble in clustering_data:
            cluster_df = clustering_data[ensemble]
            cluster_assignments = cluster_df["cluster_label"] if isinstance(cluster_df, pd.DataFrame) else cluster_df
            if len(weights) == len(cluster_assignments):
                normalized_weights = weights / np.sum(weights)
                js_div_val, _ = calculate_recovery_JSD(
                    cluster_assignments, normalized_weights, target_ratios, state_mapping
                )
                if not np.isnan(js_div_val):
                    js_div = js_div_val
                    recovery_percent = (1.0 - np.sqrt(js_div_val)) * 100.0

        row_dict = row.to_dict()
        row_dict.update(
            {
                "kl_divergence": kl_div,
                "js_divergence": float(js_div) if not np.isnan(js_div) else 0.0,
                "js_distance": float(np.sqrt(js_div)) if not np.isnan(js_div) else 0.0,
                "recovery_percent": float(recovery_percent) if not np.isnan(recovery_percent) else 0.0,
            }
        )
        augmented_rows.append(row_dict)

    return pd.DataFrame(augmented_rows)


# ---------------------------------------------------------------------------
# Data splitting (Saving)
# ---------------------------------------------------------------------------


def save_split_data(
    output_dir: str,
    split_type: str,
    split_idx: int,
    train_data: List[Any],
    val_data: List[Any],
    test_data: List[Any],
):
    """Save train/val/test data splits to CSV and JSON (topology)."""
    split_path = os.path.join(output_dir, split_type, f"split_{split_idx:03d}")
    os.makedirs(split_path, exist_ok=True)

    from jaxent.src.custom_types.HDX import HDX_peptide

    # Save training data
    HDX_peptide.save_list_to_files(
        data_list=train_data,
        json_path=os.path.join(split_path, "train_topology.json"),
        csv_path=os.path.join(split_path, "train_dfrac.csv"),
    )

    # Save validation data
    HDX_peptide.save_list_to_files(
        data_list=val_data,
        json_path=os.path.join(split_path, "val_topology.json"),
        csv_path=os.path.join(split_path, "val_dfrac.csv"),
    )

    # Note: Test data is usually saved once as full_dataset at the root
    # but we can save it per split if requested by experiment structure.
    HDX_peptide.save_list_to_files(
        data_list=test_data,
        json_path=os.path.join(output_dir, "full_dataset_topology.json"),
        csv_path=os.path.join(output_dir, "full_dataset_dfrac.csv"),
    )
    print(f"Saved split {split_idx} to {split_path}")


# ---------------------------------------------------------------------------
# Processed run scanning
# ---------------------------------------------------------------------------


def load_processed_run_info(
    processed_data_dir: str,
    ensemble_pattern: str,
    extra_group_names: List[str] | None = None,
) -> Tuple[List[Dict], List[Tuple]]:
    """Scan processed output directories and extract run metadata.

    Parameters
    ----------
    processed_data_dir:
        Directory containing per-split-type subdirectories, each with
        processed run sub-directories named by run ID.
    ensemble_pattern:
        Compiled-ready regex string with **at least 5 capture groups** in order:
        ``(ensemble, loss_name, split_type, split_idx, maxent_value, ...)``.

        Example (Exp1/Exp2)::

            r"(ISO_BI|ISO_TRI)_(mcMSE|MSE|Sigma_MSE)_(.+?)_split(\\d+)_maxent([\\d.]+)"

        Example (Exp3 with extra groups)::

            r"(AF2_MSAss|AF2_filtered)_..._maxent([\\d.]+)_bvreg([\\d.]+)_bvregfn([A-Za-z0-9]+)"

    extra_group_names:
        Optional list of names for capture groups beyond the fixed 5.  When
        provided, the extra captured values are added to each ``run_info`` dict
        under the given names.  ``None`` (default) means no extra groups.

    Returns
    -------
    all_run_info:
        List of dicts, one per matched run directory.  Keys: ``run_id``,
        ``full_run_path``, ``ensemble``, ``loss_name``, ``run_split_type``,
        ``split_idx_str``, ``split_idx``, ``maxent_value``,
        ``actual_split_type``, plus any keys from *extra_group_names*.
    unique_configs:
        Sorted list of ``(ensemble, run_split_type, split_idx_str)`` tuples
        representing distinct data-loading configurations.
    """
    all_run_info: List[Dict] = []
    if not os.path.exists(processed_data_dir):
        print(f"Processed data directory not found: {processed_data_dir}")
        return all_run_info, []

    compiled = re.compile(ensemble_pattern)

    for split_type_dir in os.listdir(processed_data_dir):
        full_split_type_path = os.path.join(processed_data_dir, split_type_dir)
        if not os.path.isdir(full_split_type_path):
            continue

        actual_split_type = split_type_dir if split_type_dir != "_flat" else "flat"

        for run_id in os.listdir(full_split_type_path):
            full_run_path = os.path.join(full_split_type_path, run_id)
            if not os.path.isdir(full_run_path):
                continue

            match = compiled.match(run_id)
            if not match:
                continue

            groups = match.groups()
            ensemble, loss_name, run_split_type, split_idx_str, maxent_str = groups[:5]

            run_info: Dict = {
                "run_id": run_id,
                "full_run_path": full_run_path,
                "ensemble": ensemble,
                "loss_name": loss_name,
                "run_split_type": run_split_type,
                "split_idx_str": split_idx_str,
                "split_idx": int(split_idx_str),
                "maxent_value": float(maxent_str),
                "actual_split_type": actual_split_type,
            }
            if extra_group_names:
                for name, value in zip(extra_group_names, groups[5:]):
                    run_info[name] = value
            all_run_info.append(run_info)

    unique_configs = sorted(
        set((r["ensemble"], r["run_split_type"], r["split_idx_str"]) for r in all_run_info)
    )
    return all_run_info, unique_configs


# ---------------------------------------------------------------------------
# Split data loading
# ---------------------------------------------------------------------------


def load_split_data(
    split_dir: str,
) -> Tuple[List, np.ndarray, List, np.ndarray]:
    """Load train and validation data from a split directory.

    Parameters
    ----------
    split_dir:
        Path to a split directory containing ``train_topology.json``,
        ``train_dfrac.csv``, ``val_topology.json``, and ``val_dfrac.csv``.

    Returns
    -------
    ``(train_top, train_dfrac, val_top, val_dfrac)`` where ``*_top`` are lists
    of ``Partial_Topology`` objects and ``*_dfrac`` are ``np.ndarray`` of
    shape ``(n_peptides, n_timepoints)``.
    """
    from jaxent.src.custom_types.datapoint import ExpD_Datapoint
    from jaxent.src.custom_types.HDX import HDX_peptide

    train_datapoints = ExpD_Datapoint.load_list_from_files(
        json_path=os.path.join(split_dir, "train_topology.json"),
        csv_path=os.path.join(split_dir, "train_dfrac.csv"),
        datapoint_class=HDX_peptide,
    )
    val_datapoints = ExpD_Datapoint.load_list_from_files(
        json_path=os.path.join(split_dir, "val_topology.json"),
        csv_path=os.path.join(split_dir, "val_dfrac.csv"),
        datapoint_class=HDX_peptide,
    )

    train_top = [dp.top for dp in train_datapoints]
    train_dfrac = np.array([dp.dfrac for dp in train_datapoints])
    val_top = [dp.top for dp in val_datapoints]
    val_dfrac = np.array([dp.dfrac for dp in val_datapoints])

    return train_top, train_dfrac, val_top, val_dfrac
