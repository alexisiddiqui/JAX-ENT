"""Checkpoint 7: exact nearest-training PF distance for conditional scale."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE, atomic_yaml, integrated_autocorrelation_frames, load_config,
    load_contact_coordinates, load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import align_to_structure, make_fold_pairs
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions, w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    absolute_change_vectors, deterministic_cap, make_inner_directions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import exact_neighbors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet, checkpoint_paths, likelihood_rows, valid_checkpoint,
)
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import (
    PREPROCESSING, _inner_arrays, fit_selected, preprocess_features, select_ridge,
)
from jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_checkpoint5 import (
    calibrated_q90, cross_replica_mean_residuals, fit_scale_calibrator, select_bins,
)


def inner_nearest_residual_sets(inner, mean_residual_sets, sigma_floor: float):
    """Nearest PF distance into the opposite training replica, paired with mean residual."""
    sets = []
    for arrays, (_, residual) in zip(inner, mean_residual_sets):
        fit_x, validation_x, *_ = arrays
        fit, validation, *_ = preprocess_features(fit_x, validation_x, "zscore", sigma_floor)
        distance, _ = exact_neighbors(fit, validation, 1)
        coordinate = distance[:, 0] / np.sqrt(fit.shape[1])
        sets.append((coordinate, residual))
    return sets


def calibrated_variance(q90: np.ndarray) -> np.ndarray:
    return np.maximum(np.finfo(float).eps, (q90 / 1.6448536269514722) ** 2)


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict]]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    vector = settings["vector_audit"]
    candidates = vector["likelihood"]["mean_scale_bins"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = align_to_structure(coordinates, reference)
    distributions = intraframe_distance_distributions(coordinates)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    centered_reference = reference - reference.mean(axis=0)
    rmsd_to_start = np.sqrt(np.mean(np.sum((aligned - centered_reference) ** 2, axis=2), axis=1))
    global_pf = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_to_start[mask]),
            integrated_autocorrelation_frames(global_pf[mask]),
        )

    results, hyperparameters = [], []
    seed = config["analysis"]["seed"]
    for heldout in (1, 2, 3):
        train_pairs, test_pairs = make_fold_pairs(
            aligned, replicas, frames, heldout, theiler,
            settings["train_pairs"], settings["test_pairs"], seed,
        )
        train_take = deterministic_cap(
            len(train_pairs.left), vector["knn_train_pairs"], seed + 1009 * heldout
        )
        test_take = deterministic_cap(
            len(test_pairs.left), vector["knn_test_pairs"], seed + 2003 * heldout
        )
        train_x = absolute_change_vectors(
            z, train_pairs.left[train_take], train_pairs.right[train_take]
        )
        test_x = absolute_change_vectors(z, test_pairs.left[test_take], test_pairs.right[test_take])
        train_w1_full = w1_pair_distance(
            distributions, train_pairs.left, train_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        test_w1_full = w1_pair_distance(
            distributions, test_pairs.left, test_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        targets = {
            "rmsd": (train_pairs.rmsd[train_take], test_pairs.rmsd[test_take]),
            "w1": (train_w1_full[train_take], test_w1_full[test_take]),
        }
        directions = make_inner_directions(
            aligned, replicas, frames, heldout, theiler,
            vector["inner_pairs_per_replica"], seed + 3001 * heldout,
        )
        replica_targets = {}
        for direction in directions:
            for replica, pairs in ((direction.fit_replica, direction.fit_pairs),
                                   (direction.validation_replica, direction.validation_pairs)):
                if replica not in replica_targets:
                    replica_targets[replica] = {
                        "rmsd": pairs.rmsd,
                        "w1": w1_pair_distance(
                            distributions, pairs.left, pairs.right,
                            settings["support_audit"]["w1_max_chunk_values"],
                        ),
                    }
        inner = _inner_arrays(z, directions, replica_targets)
        nearest_sets_cache = {}
        outer_fit, outer_validation, *_ = preprocess_features(
            train_x, test_x, "zscore", settings["sigma_floor"]
        )
        outer_distance, _ = exact_neighbors(outer_fit, outer_validation, 1)
        outer_nearest = outer_distance[:, 0] / np.sqrt(outer_fit.shape[1])
        for target, (train_target, test_target) in targets.items():
            for preprocessing in PREPROCESSING:
                selected_mean = select_ridge(
                    inner, target, preprocessing, vector["ridge_alphas"], settings["sigma_floor"]
                )
                mean_sets = cross_replica_mean_residuals(
                    inner, target, preprocessing, selected_mean.alpha, settings["sigma_floor"]
                )
                nearest_sets = inner_nearest_residual_sets(
                    inner, mean_sets, settings["sigma_floor"]
                )
                selected = select_bins(nearest_sets, candidates, monotone=True)
                mean, _, metadata = fit_selected(
                    selected_mean, train_x, train_target, test_x, settings["sigma_floor"]
                )
                pooled_nearest = np.concatenate([item[0] for item in nearest_sets])
                pooled_residual = np.concatenate([item[1] for item in nearest_sets])
                arms = [
                    ("constant_scale", fit_scale_calibrator(
                        pooled_nearest, pooled_residual, 1, monotone=True
                    )),
                    ("nearest_distance_scale", fit_scale_calibrator(
                        pooled_nearest, pooled_residual, selected.bins, monotone=True
                    )),
                ]
                for model, calibrator in arms:
                    q90 = calibrated_q90(calibrator, outer_nearest)
                    rows = likelihood_rows(
                        system, heldout, target, preprocessing, train_target, test_target,
                        mean, calibrated_variance(q90), settings, model=model,
                    )
                    for result in rows:
                        result["mean_q90_width"] = float(2 * np.mean(q90))
                        result["median_nearest_distance"] = float(np.median(outer_nearest))
                    results.extend(rows)
                hyperparameters.append({
                    "system_id": system, "heldout_replica": heldout, "target": target,
                    "preprocessing": preprocessing, "mean_alpha": selected_mean.alpha,
                    "nearest_bins": selected.bins,
                    "inner_interval_score": selected.inner_interval_score,
                    "train_pairs": len(train_take), "test_pairs": len(test_take), **metadata,
                })
    return results, hyperparameters


def aggregate(results: pd.DataFrame, hyperparameters: pd.DataFrame) -> dict:
    primary = results[
        ((results.target == "rmsd") & (results.band == "global"))
        | ((results.target == "w1") & (results.band == "q5"))
    ]
    summary = []
    for keys, group in primary.groupby(["target", "model", "preprocessing", "band"]):
        target, model, preprocessing, band = keys
        summary.append({
            "target": target, "model": model, "preprocessing": preprocessing, "band": band,
            "system_folds": len(group),
            "median_recovery_percent": float(100 * group.distribution_recovery.median()),
            "median_coverage_90_percent": float(100 * group.coverage_90.median()),
            "median_nll": float(group.mean_nll.median()),
        })
    return {
        "checkpoint": "7A", "status": "measurement_complete",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "selected_bins_percent": {
            str(int(k)): float(v) for k, v in
            hyperparameters.nearest_bins.value_counts(normalize=True).mul(100).sort_index().items()
        },
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true"); args = parser.parse_args()
    config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint7_nearest"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]; results, hyperparameters = future.result()
            result_path, hyperparameter_path = checkpoint_paths(parts, row["system_id"])
            atomic_parquet(pd.DataFrame(results), result_path)
            atomic_parquet(pd.DataFrame(hyperparameters), hyperparameter_path)
            print(f"[{resumed + index}/{len(systems)}] {row['system_id']} nearest checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoints: {missing}")
    result_frame = pd.concat([pd.read_parquet(checkpoint_paths(parts, r["system_id"])[0]) for r in systems], ignore_index=True)
    hyperparameter_frame = pd.concat([pd.read_parquet(checkpoint_paths(parts, r["system_id"])[1]) for r in systems], ignore_index=True)
    result_frame.to_parquet(output / "nearest_results.parquet", index=False)
    hyperparameter_frame.to_parquet(output / "nearest_hyperparameters.parquet", index=False)
    report = aggregate(result_frame, hyperparameter_frame)
    atomic_yaml(output / "checkpoint7a_report.yaml", report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
