"""Checkpoint 4: replica-isolated heteroscedastic per-residue ridge likelihood."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from scipy.special import digamma, ndtr
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE, atomic_yaml, integrated_autocorrelation_frames, load_config,
    load_contact_coordinates, load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import align_to_structure, make_fold_pairs
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions, target_bands, w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    absolute_change_vectors, make_inner_directions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import probability_mass_errors
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import (
    PREPROCESSING, _inner_arrays, fit_selected, preprocess_features, select_ridge,
)


@dataclass(frozen=True)
class VarianceSelection:
    alpha: float
    inner_nll: float


LOG_CHI_SQUARE_1_MEAN = float(digamma(0.5) + np.log(2.0))


def log_variance_target(residual: np.ndarray, variance_floor: float) -> np.ndarray:
    """Bias-correct log squared Gaussian residuals to target log variance."""
    return np.log(residual**2 + variance_floor) - LOG_CHI_SQUARE_1_MEAN


def gaussian_nll(residual: np.ndarray, variance: np.ndarray) -> float:
    variance = np.maximum(np.asarray(variance, float), np.finfo(float).tiny)
    return float(np.mean(0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)))


def select_variance_ridge(
    residual_sets: list[tuple[np.ndarray, np.ndarray]],
    alphas: list[float],
    variance_floor: float,
) -> VarianceSelection:
    """Swap-validate log-variance between two cross-replica residual sets."""
    scored = []
    for alpha in alphas:
        losses = []
        for fit_index, validation_index in ((0, 1), (1, 0)):
            fit_x, fit_residual = residual_sets[fit_index]
            validation_x, validation_residual = residual_sets[validation_index]
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(fit_x, log_variance_target(fit_residual, variance_floor))
            variance = np.maximum(variance_floor, np.exp(model.predict(validation_x)))
            losses.append(gaussian_nll(validation_residual, variance))
        scored.append((float(np.mean(losses)), float(alpha)))
    loss, alpha = min(scored, key=lambda item: (item[0], -item[1]))
    return VarianceSelection(alpha, loss)


def cross_fitted_scale(
    residual_sets: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    variance_floor: float,
    coverage: float = 0.90,
) -> float:
    """Calibrate scale from variance predictions made across replicas."""
    standardized = []
    for fit_index, validation_index in ((0, 1), (1, 0)):
        fit_x, fit_residual = residual_sets[fit_index]
        validation_x, validation_residual = residual_sets[validation_index]
        model = Ridge(alpha=alpha, fit_intercept=True).fit(
            fit_x, log_variance_target(fit_residual, variance_floor)
        )
        sigma = np.sqrt(np.maximum(variance_floor, np.exp(model.predict(validation_x))))
        standardized.append(np.abs(validation_residual) / sigma)
    quantile = float(np.quantile(np.concatenate(standardized), coverage, method="higher"))
    return max(quantile / 1.6448536269514722, np.finfo(float).eps)


def cross_replica_residual_sets(
    inner: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    target: str,
    preprocessing: str,
    mean_alpha: float,
    sigma_floor: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Residuals are always predictions into the other training replica."""
    residual_sets = []
    for fit_x, validation_x, fit_targets, validation_targets in inner:
        fit, validation, *_ = preprocess_features(fit_x, validation_x, preprocessing, sigma_floor)
        mean_model = Ridge(alpha=mean_alpha, fit_intercept=True).fit(fit, fit_targets[target])
        residual = validation_targets[target] - np.maximum(0, mean_model.predict(validation))
        residual_sets.append((validation, residual))
    return residual_sets


def gaussian_conditional_mass(
    mean: np.ndarray,
    sigma: np.ndarray,
    low: float,
    high: float,
    bins: int,
) -> np.ndarray:
    """Pooled Gaussian mass conditional on the requested target interval."""
    edges = np.linspace(low, high, bins + 1)
    z = (edges[None, :] - mean[:, None]) / sigma[:, None]
    mass = np.maximum(0.0, np.diff(ndtr(z), axis=1))
    row_mass = mass.sum(axis=1, keepdims=True)
    bad = row_mass[:, 0] <= np.finfo(float).tiny
    mass[~bad] /= row_mass[~bad]
    if bad.any():
        nearest = np.clip(np.digitize(mean[bad], edges[1:-1]), 0, bins - 1)
        mass[bad] = 0.0
        mass[np.flatnonzero(bad), nearest] = 1.0
    return mass.mean(axis=0)


def likelihood_rows(
    system: str,
    heldout: int,
    target_name: str,
    preprocessing: str,
    train_target: np.ndarray,
    test_target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    settings: dict,
    model: str = "heteroscedastic_ridge_gaussian",
) -> list[dict]:
    boundary = settings["boundary_audit"]
    sigma = np.sqrt(variance)
    rows = []
    for band, mask, low, high in target_bands(target_name, train_target, test_target):
        if not mask.any():
            continue
        fit_low = low if np.isfinite(low) else float(train_target.min())
        fit_high = high if np.isfinite(high) else float(train_target.max())
        if fit_high <= fit_low:
            continue
        predicted_mass = gaussian_conditional_mass(
            mean[mask], sigma[mask], fit_low, fit_high, boundary["distribution_bins"]
        )
        edges = np.linspace(fit_low, fit_high, boundary["distribution_bins"] + 1)
        target_mass = np.histogram(np.clip(test_target[mask], fit_low, fit_high), bins=edges)[0]
        errors = probability_mass_errors(
            predicted_mass, target_mass, boundary["distribution_smoothing"]
        )
        residual = test_target[mask] - mean[mask]
        rows.append({
            "system_id": system,
            "heldout_replica": heldout,
            "target": target_name,
            "model": model,
            "preprocessing": preprocessing,
            "band": band,
            "pairs": int(mask.sum()),
            "mean_nll": gaussian_nll(residual, variance[mask]),
            "coverage_90": float(np.mean(np.abs(residual) <= 1.6448536269514722 * sigma[mask])),
            "mean_sigma": float(np.mean(sigma[mask])),
            **errors,
        })
    return rows


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict]]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    vector = settings["vector_audit"]
    likelihood = vector["likelihood"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = align_to_structure(coordinates, reference)
    distributions = intraframe_distance_distributions(coordinates)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    reference_centered = reference - reference.mean(axis=0)
    rmsd_to_start = np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=2), axis=1))
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
        train_w1 = w1_pair_distance(
            distributions, train_pairs.left, train_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        test_w1 = w1_pair_distance(
            distributions, test_pairs.left, test_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
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
        for target_name, (train_target, test_target) in {
            "rmsd": (train_pairs.rmsd, test_pairs.rmsd), "w1": (train_w1, test_w1)
        }.items():
            target_iqr = float(np.subtract(*np.quantile(train_target, [0.75, 0.25])))
            variance_floor = max(np.finfo(float).eps, (likelihood["sigma_floor_iqr"] * target_iqr) ** 2)
            for preprocessing in PREPROCESSING:
                selected_mean = select_ridge(
                    inner, target_name, preprocessing, vector["ridge_alphas"], settings["sigma_floor"]
                )
                residual_sets = cross_replica_residual_sets(
                    inner, target_name, preprocessing, selected_mean.alpha, settings["sigma_floor"]
                )
                selected_variance = select_variance_ridge(
                    residual_sets, likelihood["variance_alphas"], variance_floor
                )
                calibration_scale = cross_fitted_scale(
                    residual_sets, selected_variance.alpha, variance_floor
                )
                mean, _, metadata = fit_selected(
                    selected_mean, train_x, train_target, test_x, settings["sigma_floor"]
                )
                outer_train, outer_test, *_ = preprocess_features(
                    train_x, test_x, preprocessing, settings["sigma_floor"]
                )
                # Re-express cross-replica validation features in the outer-training transform.
                residual_x, residual_y = [], []
                for direction in directions:
                    validation_raw = absolute_change_vectors(
                        z, direction.validation_pairs.left, direction.validation_pairs.right
                    )
                    _, transformed, *_ = preprocess_features(
                        train_x, validation_raw, preprocessing, settings["sigma_floor"]
                    )
                    # residual_sets follow the same direction ordering.
                    residual_x.append(transformed)
                residual_y = [item[1] for item in residual_sets]
                variance_model = Ridge(alpha=selected_variance.alpha, fit_intercept=True).fit(
                    np.concatenate(residual_x),
                    log_variance_target(np.concatenate(residual_y), variance_floor),
                )
                variance = np.maximum(
                    variance_floor,
                    np.exp(variance_model.predict(outer_test)) * calibration_scale**2,
                )
                results.extend(likelihood_rows(
                    system, heldout, target_name, preprocessing, train_target, test_target,
                    mean, variance, settings,
                ))
                hyperparameters.append({
                    "system_id": system, "heldout_replica": heldout, "target": target_name,
                    "preprocessing": preprocessing, "mean_alpha": selected_mean.alpha,
                    "variance_alpha": selected_variance.alpha,
                    "inner_variance_nll": selected_variance.inner_nll,
                    "variance_floor": variance_floor,
                    "calibration_scale": calibration_scale, **metadata,
                })
    return results, hyperparameters


def aggregate(results: pd.DataFrame) -> dict:
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
            "median_nll": float(group.mean_nll.median()),
            "median_coverage_90_percent": float(100 * group.coverage_90.median()),
        })
    return {
        "checkpoint": "4A", "status": "measurement_complete",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "summary": summary,
    }


def checkpoint_paths(parts: "Path", system: str) -> tuple["Path", "Path"]:
    return parts / f"{system}.results.parquet", parts / f"{system}.hyperparameters.parquet"


def atomic_parquet(frame: pd.DataFrame, path: "Path") -> None:
    """Write a parquet checkpoint without exposing a partial destination."""
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def valid_checkpoint(parts: "Path", system: str) -> bool:
    result_path, hyperparameter_path = checkpoint_paths(parts, system)
    if not result_path.exists() or not hyperparameter_path.exists():
        return False
    try:
        results = pd.read_parquet(result_path, columns=["system_id", "heldout_replica"])
        hyperparameters = pd.read_parquet(
            hyperparameter_path, columns=["system_id", "heldout_replica"]
        )
    except Exception:
        return False
    return (
        set(results.system_id) == {system}
        and set(hyperparameters.system_id) == {system}
        and set(results.heldout_replica) == {1, 2, 3}
        and set(hyperparameters.heldout_replica) == {1, 2, 3}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true", help="ignore valid per-system checkpoints")
    args = parser.parse_args()
    config = load_config(); systems = load_systems()[:args.limit]
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint4_likelihood"
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed:
        print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            system_results, system_hyperparameters = future.result()
            result_path, hyperparameter_path = checkpoint_paths(parts, row["system_id"])
            atomic_parquet(pd.DataFrame(system_results), result_path)
            atomic_parquet(pd.DataFrame(system_hyperparameters), hyperparameter_path)
            print(
                f"[{resumed + index}/{len(systems)}] {row['system_id']} likelihood checkpointed",
                flush=True,
            )
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing:
        raise RuntimeError(f"cannot assemble population; missing checkpoints: {missing}")
    result_frame = pd.concat(
        [pd.read_parquet(checkpoint_paths(parts, row["system_id"])[0]) for row in systems],
        ignore_index=True,
    )
    hyperparameter_frame = pd.concat(
        [pd.read_parquet(checkpoint_paths(parts, row["system_id"])[1]) for row in systems],
        ignore_index=True,
    )
    result_frame.to_parquet(output / "likelihood_results.parquet", index=False)
    hyperparameter_frame.to_parquet(output / "likelihood_hyperparameters.parquet", index=False)
    report = aggregate(result_frame)
    atomic_yaml(output / "checkpoint4a_report.yaml", report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
