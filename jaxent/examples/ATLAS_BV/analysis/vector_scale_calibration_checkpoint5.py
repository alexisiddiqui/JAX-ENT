"""Checkpoint 5: nonlinear predicted-mean scale calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

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
    absolute_change_vectors, make_inner_directions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet, checkpoint_paths, likelihood_rows, valid_checkpoint,
)
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import (
    PREPROCESSING, _inner_arrays, fit_selected, preprocess_features, select_ridge,
)


@dataclass(frozen=True)
class ScaleCalibrator:
    centers: np.ndarray
    quantiles: np.ndarray


@dataclass(frozen=True)
class SelectedBins:
    bins: int
    inner_interval_score: float


def fit_scale_calibrator(
    mean: np.ndarray, residual: np.ndarray, bins: int, monotone: bool = False
) -> ScaleCalibrator:
    """Fit piecewise-linear conditional q90 absolute residual against predicted mean."""
    edges = np.unique(np.quantile(mean, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return ScaleCalibrator(np.array([float(np.median(mean))]),
                               np.array([float(np.quantile(np.abs(residual), 0.9, method="higher"))]))
    labels = np.clip(np.digitize(mean, edges[1:-1]), 0, len(edges) - 2)
    centers, quantiles = [], []
    for label in range(len(edges) - 1):
        mask = labels == label
        if mask.any():
            centers.append(float(np.median(mean[mask])))
            quantiles.append(float(np.quantile(np.abs(residual[mask]), 0.9, method="higher")))
    quantiles = np.maximum(np.asarray(quantiles), np.finfo(float).eps)
    if monotone:
        quantiles = np.maximum.accumulate(quantiles)
    return ScaleCalibrator(np.asarray(centers), quantiles)


def calibrated_q90(calibrator: ScaleCalibrator, mean: np.ndarray) -> np.ndarray:
    if len(calibrator.centers) == 1:
        return np.full(len(mean), calibrator.quantiles[0])
    return np.interp(
        mean, calibrator.centers, calibrator.quantiles,
        left=calibrator.quantiles[0], right=calibrator.quantiles[-1],
    )


def interval_score(residual: np.ndarray, q90: np.ndarray, alpha: float = 0.1) -> float:
    """Central interval score; lower is better and the target quantile is proper."""
    excess = np.maximum(np.abs(residual) - q90, 0.0)
    return float(np.mean(2 * q90 + (2 / alpha) * excess))


def select_bins(
    mean_residual_sets: list[tuple[np.ndarray, np.ndarray]], candidates: list[int],
    monotone: bool = False,
) -> SelectedBins:
    scored = []
    for bins in candidates:
        losses = []
        for fit_index, validation_index in ((0, 1), (1, 0)):
            fit_mean, fit_residual = mean_residual_sets[fit_index]
            validation_mean, validation_residual = mean_residual_sets[validation_index]
            calibrator = fit_scale_calibrator(fit_mean, fit_residual, bins, monotone=monotone)
            losses.append(interval_score(
                validation_residual, calibrated_q90(calibrator, validation_mean)
            ))
        scored.append((float(np.mean(losses)), int(bins)))
    score, bins = min(scored, key=lambda item: (item[0], item[1]))
    return SelectedBins(bins, score)


def cross_replica_mean_residuals(inner, target: str, preprocessing: str, alpha: float, sigma_floor: float):
    sets = []
    for fit_x, validation_x, fit_targets, validation_targets in inner:
        fit, validation, *_ = preprocess_features(fit_x, validation_x, preprocessing, sigma_floor)
        model = Ridge(alpha=alpha, fit_intercept=True).fit(fit, fit_targets[target])
        mean = np.maximum(0.0, model.predict(validation))
        sets.append((mean, validation_targets[target] - mean))
    return sets


def cross_replica_novelty_residuals(
    inner, mean_residual_sets: list[tuple[np.ndarray, np.ndarray]], sigma_floor: float
):
    """Pair residuals with radial PF novelty fitted on the opposite replica."""
    sets = []
    for arrays, (_, residual) in zip(inner, mean_residual_sets):
        fit_x, validation_x, *_ = arrays
        _, standardized, *_ = preprocess_features(fit_x, validation_x, "zscore", sigma_floor)
        novelty = np.linalg.norm(standardized, axis=1) / np.sqrt(standardized.shape[1])
        sets.append((novelty, residual))
    return sets


def analyse_system(
    row: dict[str, str], config: dict, scale_coordinate: str = "predicted_mean"
) -> tuple[list[dict], list[dict]]:
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
        directions = make_inner_directions(
            aligned, replicas, frames, heldout, theiler,
            vector["inner_pairs_per_replica"], seed + 3001 * heldout,
        )
        train_x = absolute_change_vectors(z, train_pairs.left, train_pairs.right)
        test_x = absolute_change_vectors(z, test_pairs.left, test_pairs.right)
        train_w1 = w1_pair_distance(distributions, train_pairs.left, train_pairs.right,
                                    settings["support_audit"]["w1_max_chunk_values"])
        test_w1 = w1_pair_distance(distributions, test_pairs.left, test_pairs.right,
                                   settings["support_audit"]["w1_max_chunk_values"])
        replica_targets = {}
        for direction in directions:
            for replica, pairs in ((direction.fit_replica, direction.fit_pairs),
                                   (direction.validation_replica, direction.validation_pairs)):
                if replica not in replica_targets:
                    replica_targets[replica] = {
                        "rmsd": pairs.rmsd,
                        "w1": w1_pair_distance(distributions, pairs.left, pairs.right,
                            settings["support_audit"]["w1_max_chunk_values"]),
                    }
        inner = _inner_arrays(z, directions, replica_targets)
        for target, (train_target, test_target) in {
            "rmsd": (train_pairs.rmsd, test_pairs.rmsd), "w1": (train_w1, test_w1)
        }.items():
            for preprocessing in PREPROCESSING:
                selected_mean = select_ridge(
                    inner, target, preprocessing, vector["ridge_alphas"], settings["sigma_floor"]
                )
                mean_residual_sets = cross_replica_mean_residuals(
                    inner, target, preprocessing, selected_mean.alpha, settings["sigma_floor"]
                )
                if scale_coordinate == "predicted_mean":
                    calibration_sets = mean_residual_sets
                elif scale_coordinate == "pf_novelty":
                    calibration_sets = cross_replica_novelty_residuals(
                        inner, mean_residual_sets, settings["sigma_floor"]
                    )
                else:
                    raise ValueError(f"unknown scale coordinate: {scale_coordinate}")
                monotone_scale = scale_coordinate == "pf_novelty"
                selected_bins = select_bins(
                    calibration_sets, candidates, monotone=monotone_scale
                )
                mean, _, metadata = fit_selected(
                    selected_mean, train_x, train_target, test_x, settings["sigma_floor"]
                )
                pooled_coordinate = np.concatenate([item[0] for item in calibration_sets])
                pooled_residual = np.concatenate([item[1] for item in calibration_sets])
                calibrator = fit_scale_calibrator(
                    pooled_coordinate, pooled_residual, selected_bins.bins,
                    monotone=monotone_scale,
                )
                if scale_coordinate == "predicted_mean":
                    test_coordinate = mean
                    model_name = "predicted_mean_binned_scale"
                else:
                    _, standardized_test, *_ = preprocess_features(
                        train_x, test_x, "zscore", settings["sigma_floor"]
                    )
                    test_coordinate = (
                        np.linalg.norm(standardized_test, axis=1)
                        / np.sqrt(standardized_test.shape[1])
                    )
                    model_name = "pf_novelty_binned_scale"
                q90 = calibrated_q90(calibrator, test_coordinate)
                variance = np.maximum(np.finfo(float).eps, (q90 / 1.6448536269514722) ** 2)
                rows = likelihood_rows(
                    system, heldout, target, preprocessing, train_target, test_target,
                    mean, variance, settings, model=model_name,
                )
                for result in rows:
                    result["mean_q90_width"] = float(2 * np.mean(q90))
                    result["scale_coordinate"] = scale_coordinate
                results.extend(rows)
                hyperparameters.append({
                    "system_id": system, "heldout_replica": heldout, "target": target,
                    "preprocessing": preprocessing, "mean_alpha": selected_mean.alpha,
                    "scale_bins": selected_bins.bins,
                    "scale_coordinate": scale_coordinate,
                    "inner_interval_score": selected_bins.inner_interval_score, **metadata,
                })
    return results, hyperparameters


def aggregate(
    results: pd.DataFrame, hyperparameters: pd.DataFrame, checkpoint: str = "5A"
) -> dict:
    primary = results[
        ((results.target == "rmsd") & (results.band == "global"))
        | ((results.target == "w1") & (results.band == "q5"))
    ]
    summary = []
    for keys, group in primary.groupby(["target", "preprocessing", "band"]):
        target, preprocessing, band = keys
        summary.append({
            "target": target, "preprocessing": preprocessing, "band": band,
            "system_folds": len(group),
            "median_recovery_percent": float(100 * group.distribution_recovery.median()),
            "median_coverage_90_percent": float(100 * group.coverage_90.median()),
            "median_nll": float(group.mean_nll.median()),
        })
    return {
        "checkpoint": checkpoint, "status": "measurement_complete",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "selected_bins_percent": {
            str(int(key)): float(value) for key, value in
            hyperparameters.scale_bins.value_counts(normalize=True).mul(100).sort_index().items()
        },
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--scale-coordinate", choices=("predicted_mean", "pf_novelty"),
        default="predicted_mean",
    )
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    checkpoint = "checkpoint5_scale" if args.scale_coordinate == "predicted_mean" else "checkpoint6_novelty"
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / checkpoint
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(analyse_system, row, config, args.scale_coordinate): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]; results, hyperparameters = future.result()
            result_path, hyperparameter_path = checkpoint_paths(parts, row["system_id"])
            atomic_parquet(pd.DataFrame(results), result_path)
            atomic_parquet(pd.DataFrame(hyperparameters), hyperparameter_path)
            print(f"[{resumed + index}/{len(systems)}] {row['system_id']} scale checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoints: {missing}")
    result_frame = pd.concat([pd.read_parquet(checkpoint_paths(parts, r["system_id"])[0]) for r in systems], ignore_index=True)
    hyperparameter_frame = pd.concat([pd.read_parquet(checkpoint_paths(parts, r["system_id"])[1]) for r in systems], ignore_index=True)
    result_frame.to_parquet(output / "scale_results.parquet", index=False)
    hyperparameter_frame.to_parquet(output / "scale_hyperparameters.parquet", index=False)
    report = aggregate(
        result_frame, hyperparameter_frame,
        "5A" if args.scale_coordinate == "predicted_mean" else "6A",
    )
    report_name = "checkpoint5a_report.yaml" if args.scale_coordinate == "predicted_mean" else "checkpoint6a_report.yaml"
    atomic_yaml(output / report_name, report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
