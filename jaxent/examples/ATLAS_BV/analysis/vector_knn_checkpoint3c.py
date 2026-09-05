"""Checkpoint 3C: capped exact kNN on complete per-residue PF-change vectors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 import probability_distribution_errors
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import align_to_structure, make_fold_pairs
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions,
    target_bands,
    w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    InnerDirection,
    absolute_change_vectors,
    deterministic_cap,
    make_inner_directions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import (
    normalized_mae,
    preprocess_features,
)


PREPROCESSING = ("raw", "zscore")


@dataclass(frozen=True)
class SelectedKNN:
    preprocessing: str
    neighbors: int
    inner_normalized_mae: float


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    """Row-normalized weights with exact matches receiving all mass."""
    weights = np.empty_like(distances, dtype=np.float64)
    zero = distances <= np.finfo(np.float64).eps
    rows_with_zero = zero.any(axis=1)
    weights[rows_with_zero] = zero[rows_with_zero]
    weights[~rows_with_zero] = 1.0 / np.maximum(
        distances[~rows_with_zero], np.finfo(np.float64).eps
    )
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def exact_neighbors(
    train: np.ndarray,
    query: np.ndarray,
    maximum_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = min(maximum_neighbors, len(train))
    model = NearestNeighbors(n_neighbors=neighbors, algorithm="brute", metric="euclidean", n_jobs=1)
    model.fit(train)
    return model.kneighbors(query, return_distance=True)


def point_prediction(
    distances: np.ndarray,
    indices: np.ndarray,
    train_target: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    neighbors = min(neighbors, indices.shape[1])
    weights = inverse_distance_weights(distances[:, :neighbors])
    return np.maximum(0.0, np.sum(weights * train_target[indices[:, :neighbors]], axis=1))


def probability_mass_errors(
    predicted_mass: np.ndarray,
    target_mass: np.ndarray,
    smoothing: float,
) -> dict[str, float]:
    predicted = predicted_mass.astype(float) + smoothing
    target = target_mass.astype(float) + smoothing
    predicted /= predicted.sum()
    target /= target.sum()
    midpoint = 0.5 * (predicted + target)
    delta = predicted - target
    jsd = 0.5 * (
        np.sum(predicted * np.log(predicted / midpoint))
        + np.sum(target * np.log(target / midpoint))
    )
    # JSD is non-negative analytically; guard sqrt against sub-ulp cancellation.
    jsd = max(0.0, float(jsd))
    return {
        "distribution_l1": float(np.sum(np.abs(delta))),
        "distribution_l2": float(np.sqrt(np.sum(delta**2))),
        "distribution_jsd": jsd,
        "distribution_sqrt_jsd": float(np.sqrt(jsd)),
        "distribution_kld_target_to_prediction": float(np.sum(target * np.log(target / predicted))),
        "distribution_recovery": float(1.0 - np.sqrt(jsd)),
    }


def conditional_distribution_errors(
    distances: np.ndarray,
    indices: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    query_mask: np.ndarray,
    neighbors: int,
    low: float,
    high: float,
    bins: int,
    smoothing: float,
) -> dict[str, float]:
    """Compare the pooled per-query neighbor mixture with observed target mass."""
    neighbors = min(neighbors, indices.shape[1])
    local_distances = distances[query_mask, :neighbors]
    local_indices = indices[query_mask, :neighbors]
    weights = inverse_distance_weights(local_distances)
    edges = np.linspace(low, high, bins + 1)
    values = np.clip(train_target[local_indices], low, high)
    labels = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    predicted_mass = np.zeros(bins, dtype=float)
    for label in range(bins):
        predicted_mass[label] = np.sum(weights[labels == label])
    predicted_mass /= np.count_nonzero(query_mask)
    target_mass = np.histogram(np.clip(test_target[query_mask], low, high), bins=edges)[0]
    return probability_mass_errors(predicted_mass, target_mass, smoothing)


def select_k(
    inner_neighbors: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    target: str,
    candidates: list[int],
) -> SelectedKNN:
    scored = []
    for neighbors in candidates:
        scores = []
        for distances, indices, fit_targets, validation_targets in inner_neighbors:
            prediction = point_prediction(distances, indices, fit_targets[target], neighbors)
            scores.append(
                normalized_mae(validation_targets[target], prediction, fit_targets[target])
            )
        scored.append((float(np.mean(scores)), int(neighbors)))
    score, neighbors = min(scored, key=lambda item: (item[0], -item[1]))
    return SelectedKNN("", neighbors, score)


def _inner_neighbor_arrays(
    z: np.ndarray,
    directions: tuple[InnerDirection, InnerDirection],
    targets: dict[int, dict[str, np.ndarray]],
    preprocessing: str,
    sigma_floor: float,
    maximum_neighbors: int,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    arrays = []
    for direction in directions:
        fit_x = absolute_change_vectors(z, direction.fit_pairs.left, direction.fit_pairs.right)
        validation_x = absolute_change_vectors(
            z, direction.validation_pairs.left, direction.validation_pairs.right
        )
        fit, validation, *_ = preprocess_features(
            fit_x, validation_x, preprocessing, sigma_floor
        )
        distances, indices = exact_neighbors(fit, validation, maximum_neighbors)
        arrays.append(
            (
                distances,
                indices,
                targets[direction.fit_replica],
                targets[direction.validation_replica],
            )
        )
    return arrays


def metric_rows(
    system: str,
    heldout: int,
    selected: SelectedKNN,
    target_name: str,
    train_target: np.ndarray,
    test_target: np.ndarray,
    prediction: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    settings: dict,
) -> list[dict]:
    rows = []
    boundary = settings["boundary_audit"]
    target_iqr = float(np.subtract(*np.quantile(train_target, [0.75, 0.25])))
    rho = (
        0.0
        if np.ptp(test_target) <= 1e-15 or np.ptp(prediction) <= 1e-15
        else float(spearmanr(test_target, prediction).statistic)
    )
    for band, mask, low, high in target_bands(target_name, train_target, test_target):
        if not mask.any():
            continue
        fit_low = low if np.isfinite(low) else float(train_target.min())
        fit_high = high if np.isfinite(high) else float(train_target.max())
        if fit_high <= fit_low:
            continue
        point_errors = probability_distribution_errors(
            prediction[mask], test_target[mask], fit_low, fit_high,
            boundary["distribution_bins"], boundary["distribution_smoothing"],
        )
        conditional_errors = conditional_distribution_errors(
            distances, indices, train_target, test_target, mask, selected.neighbors,
            fit_low, fit_high, boundary["distribution_bins"],
            boundary["distribution_smoothing"],
        )
        mae = float(np.mean(np.abs(prediction[mask] - test_target[mask])))
        rows.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "target": target_name,
                "model": "knn",
                "preprocessing": selected.preprocessing,
                "band": band,
                "pairs": int(mask.sum()),
                "neighbors": selected.neighbors,
                "mae": mae,
                "normalized_mae": mae / target_iqr if target_iqr > 0 else np.nan,
                "spearman_rho": rho,
                **point_errors,
                **{f"conditional_{key}": value for key, value in conditional_errors.items()},
            }
        )
    return rows


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict]]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    vector = settings["vector_audit"]
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
    max_neighbors = max(vector["knn_neighbors"])
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
        train_left, train_right = train_pairs.left[train_take], train_pairs.right[train_take]
        test_left, test_right = test_pairs.left[test_take], test_pairs.right[test_take]
        train_x = absolute_change_vectors(z, train_left, train_right)
        test_x = absolute_change_vectors(z, test_left, test_right)

        train_w1_full = w1_pair_distance(
            distributions, train_pairs.left, train_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        test_w1_full = w1_pair_distance(
            distributions, test_pairs.left, test_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        outer_targets = {
            "rmsd": (train_pairs.rmsd[train_take], test_pairs.rmsd[test_take]),
            "w1": (train_w1_full[train_take], test_w1_full[test_take]),
        }
        directions = make_inner_directions(
            aligned, replicas, frames, heldout, theiler,
            vector["inner_pairs_per_replica"], seed + 3001 * heldout,
        )
        replica_targets = {}
        for direction in directions:
            for replica, pairs in (
                (direction.fit_replica, direction.fit_pairs),
                (direction.validation_replica, direction.validation_pairs),
            ):
                if replica not in replica_targets:
                    replica_targets[replica] = {
                        "rmsd": pairs.rmsd,
                        "w1": w1_pair_distance(
                            distributions, pairs.left, pairs.right,
                            settings["support_audit"]["w1_max_chunk_values"],
                        ),
                    }

        for preprocessing in PREPROCESSING:
            inner = _inner_neighbor_arrays(
                z, directions, replica_targets, preprocessing,
                settings["sigma_floor"], max_neighbors,
            )
            outer_train, outer_test, _, _, floored = preprocess_features(
                train_x, test_x, preprocessing, settings["sigma_floor"]
            )
            distances, indices = exact_neighbors(outer_train, outer_test, max_neighbors)
            for target_name, (train_target, test_target) in outer_targets.items():
                selected_base = select_k(inner, target_name, vector["knn_neighbors"])
                selected = SelectedKNN(
                    preprocessing, selected_base.neighbors, selected_base.inner_normalized_mae
                )
                prediction = point_prediction(
                    distances, indices, train_target, selected.neighbors
                )
                results.extend(
                    metric_rows(
                        system, heldout, selected, target_name,
                        train_target, test_target, prediction, distances, indices, settings,
                    )
                )
                hyperparameters.append(
                    {
                        "system_id": system,
                        "heldout_replica": heldout,
                        "target": target_name,
                        "model": "knn",
                        "preprocessing": preprocessing,
                        "neighbors": selected.neighbors,
                        "inner_normalized_mae": selected.inner_normalized_mae,
                        "floored_residues": floored,
                        "train_pairs": len(train_take),
                        "test_pairs": len(test_take),
                    }
                )
    return results, hyperparameters


def aggregate(results: pd.DataFrame, hyperparameters: pd.DataFrame) -> dict:
    summary = []
    for keys, group in results.groupby(["target", "preprocessing", "band"]):
        target, preprocessing, band = keys
        summary.append(
            {
                "target": target,
                "preprocessing": preprocessing,
                "band": band,
                "system_folds": len(group),
                "median_point_recovery_percent": float(100 * group.distribution_recovery.median()),
                "median_conditional_recovery_percent": float(
                    100 * group.conditional_distribution_recovery.median()
                ),
                "median_normalized_mae": float(group.normalized_mae.median()),
            }
        )
    return {
        "checkpoint": "3C",
        "status": "measurement_complete",
        "decision": "pause_for_review",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "train_pair_cap": int(hyperparameters.train_pairs.median()),
        "test_pair_cap": int(hyperparameters.test_pairs.median()),
        "summary": summary,
        "next_step": "final paired familywise comparison; requires review",
    }


def write_plots(results: pd.DataFrame, output) -> None:
    import matplotlib.pyplot as plt

    tail = results[
        ((results.target == "rmsd") & (results.band == "global"))
        | ((results.target == "w1") & (results.band == "q5"))
    ]
    labels, point, conditional = [], [], []
    for target in ("rmsd", "w1"):
        for preprocessing in PREPROCESSING:
            group = tail[(tail.target == target) & (tail.preprocessing == preprocessing)]
            labels.append(f"{target.upper()}\n{preprocessing}")
            point.append(100 * group.distribution_recovery.median())
            conditional.append(100 * group.conditional_distribution_recovery.median())
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, point, width, label="Point-prediction distribution")
    ax.bar(x + width / 2, conditional, width, label="Neighbour conditional mixture")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median distribution recovery (%)")
    ax.set_title("Exact complete-vector kNN tail recovery")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "knn_tail_recovery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    systems = load_systems()[: args.limit]
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    results, hyperparameters = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in systems}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            system_results, system_hyperparameters = future.result()
            results.extend(system_results)
            hyperparameters.extend(system_hyperparameters)
            print(f"[{index}/{len(systems)}] {row['system_id']} exact kNN complete", flush=True)
    results_frame = pd.DataFrame(results)
    hyperparameter_frame = pd.DataFrame(hyperparameters)
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint3_vector"
    output.mkdir(parents=True, exist_ok=True)
    results_frame.to_parquet(output / "knn_results.parquet", index=False)
    hyperparameter_frame.to_parquet(output / "knn_hyperparameters.parquet", index=False)
    report = aggregate(results_frame, hyperparameter_frame)
    atomic_yaml(output / "checkpoint3c_report.yaml", report)
    write_plots(results_frame, output)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
