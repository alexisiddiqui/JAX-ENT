"""Checkpoint 3B: replica-tuned per-residue ridge and PCA-ridge models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    align_to_structure,
    make_fold_pairs,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions,
    target_bands,
    w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    InnerDirection,
    absolute_change_vectors,
    feature_standardizer,
    make_inner_directions,
)


PREPROCESSING = ("raw", "zscore")


@dataclass(frozen=True)
class SelectedModel:
    model: str
    preprocessing: str
    alpha: float
    pca_variance: float | None
    inner_normalized_mae: float


def preprocess_features(
    train: np.ndarray,
    other: np.ndarray,
    preprocessing: str,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit pair-feature preprocessing on train only and apply it to another set."""
    if preprocessing == "raw":
        mean = np.zeros(train.shape[1])
        sigma = np.ones(train.shape[1])
        return train, other, mean, sigma, 0
    if preprocessing != "zscore":
        raise ValueError(f"unknown preprocessing: {preprocessing}")
    mean, sigma, floored = feature_standardizer(train, sigma_floor)
    return (train - mean) / sigma, (other - mean) / sigma, mean, sigma, floored


def normalized_mae(target: np.ndarray, prediction: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.subtract(*np.quantile(reference, [0.75, 0.25])))
    if scale <= 0:
        return float("inf")
    return float(np.mean(np.abs(target - np.maximum(0.0, prediction))) / scale)


def _inner_arrays(
    z: np.ndarray,
    directions: tuple[InnerDirection, InnerDirection],
    targets: dict[int, dict[str, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]]:
    arrays = []
    for direction in directions:
        fit_x = absolute_change_vectors(z, direction.fit_pairs.left, direction.fit_pairs.right)
        validation_x = absolute_change_vectors(
            z, direction.validation_pairs.left, direction.validation_pairs.right
        )
        arrays.append(
            (
                fit_x,
                validation_x,
                targets[direction.fit_replica],
                targets[direction.validation_replica],
            )
        )
    return arrays


def select_ridge(
    inner: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    target: str,
    preprocessing: str,
    alphas: list[float],
    sigma_floor: float,
) -> SelectedModel:
    scored = []
    for alpha in alphas:
        direction_scores = []
        for fit_x, validation_x, fit_targets, validation_targets in inner:
            fit, validation, *_ = preprocess_features(
                fit_x, validation_x, preprocessing, sigma_floor
            )
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(fit, fit_targets[target])
            direction_scores.append(
                normalized_mae(
                    validation_targets[target], model.predict(validation), fit_targets[target]
                )
            )
        scored.append((float(np.mean(direction_scores)), float(alpha)))
    score, alpha = min(scored, key=lambda item: (item[0], -item[1]))
    return SelectedModel("ridge", preprocessing, alpha, None, score)


def select_pca_ridge(
    inner: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]],
    target: str,
    preprocessing: str,
    alphas: list[float],
    variance_grid: list[float],
    sigma_floor: float,
) -> SelectedModel:
    transformed = []
    for fit_x, validation_x, fit_targets, validation_targets in inner:
        fit, validation, *_ = preprocess_features(
            fit_x, validation_x, preprocessing, sigma_floor
        )
        pca = PCA(svd_solver="full")
        fit_pc = pca.fit_transform(fit)
        validation_pc = pca.transform(validation)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        transformed.append((fit_pc, validation_pc, cumulative, fit_targets, validation_targets))

    scored = []
    for retained_variance in variance_grid:
        for alpha in alphas:
            direction_scores = []
            for fit_pc, validation_pc, cumulative, fit_targets, validation_targets in transformed:
                components = min(len(cumulative), int(np.searchsorted(cumulative, retained_variance) + 1))
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(fit_pc[:, :components], fit_targets[target])
                direction_scores.append(
                    normalized_mae(
                        validation_targets[target],
                        model.predict(validation_pc[:, :components]),
                        fit_targets[target],
                    )
                )
            scored.append(
                (float(np.mean(direction_scores)), float(retained_variance), float(alpha))
            )
    score, retained_variance, alpha = min(
        scored, key=lambda item: (item[0], item[1], -item[2])
    )
    return SelectedModel("pca_ridge", preprocessing, alpha, retained_variance, score)


def fit_selected(
    selected: SelectedModel,
    train_x: np.ndarray,
    train_target: np.ndarray,
    test_x: np.ndarray,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Outer refit and coefficients back-projected to original per-residue features."""
    train, test, mean, sigma, floored = preprocess_features(
        train_x, test_x, selected.preprocessing, sigma_floor
    )
    pca = None
    if selected.model == "pca_ridge":
        pca = PCA(svd_solver="full")
        train_pc = pca.fit_transform(train)
        test_pc = pca.transform(test)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        components = min(
            len(cumulative), int(np.searchsorted(cumulative, selected.pca_variance) + 1)
        )
        design_train, design_test = train_pc[:, :components], test_pc[:, :components]
    else:
        components = train.shape[1]
        design_train, design_test = train, test
    model = Ridge(alpha=selected.alpha, fit_intercept=True)
    model.fit(design_train, train_target)
    prediction = np.maximum(0.0, model.predict(design_test))
    if pca is None:
        processed_coefficient = model.coef_
    else:
        processed_coefficient = pca.components_[:components].T @ model.coef_
    coefficient = processed_coefficient / sigma
    metadata = {
        "components": int(components),
        "floored_residues": int(floored),
        "intercept": float(model.intercept_),
        "feature_mean_norm": float(np.linalg.norm(mean)),
    }
    return prediction, coefficient, metadata


def metric_rows(
    system: str,
    heldout: int,
    selected: SelectedModel,
    target_name: str,
    train_target: np.ndarray,
    test_target: np.ndarray,
    prediction: np.ndarray,
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
        distribution = probability_distribution_errors(
            prediction[mask],
            test_target[mask],
            fit_low,
            fit_high,
            boundary["distribution_bins"],
            boundary["distribution_smoothing"],
        )
        mae = float(np.mean(np.abs(prediction[mask] - test_target[mask])))
        rows.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "target": target_name,
                "model": selected.model,
                "preprocessing": selected.preprocessing,
                "band": band,
                "pairs": int(mask.sum()),
                "mae": mae,
                "normalized_mae": mae / target_iqr if target_iqr > 0 else np.nan,
                "spearman_rho": rho,
                **distribution,
            }
        )
    return rows


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict], list[dict]]:
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

    result_rows, hyperparameter_rows, coefficient_rows = [], [], []
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
        inner = _inner_arrays(z, directions, replica_targets)
        train_x = absolute_change_vectors(z, train_pairs.left, train_pairs.right)
        test_x = absolute_change_vectors(z, test_pairs.left, test_pairs.right)
        targets = {
            "rmsd": (train_pairs.rmsd, test_pairs.rmsd),
            "w1": (train_w1, test_w1),
        }
        for target_name, (train_target, test_target) in targets.items():
            selected_models = []
            for preprocessing in PREPROCESSING:
                selected_models.append(
                    select_ridge(
                        inner, target_name, preprocessing, vector["ridge_alphas"],
                        settings["sigma_floor"],
                    )
                )
                selected_models.append(
                    select_pca_ridge(
                        inner, target_name, preprocessing, vector["ridge_alphas"],
                        vector["pca_variance"], settings["sigma_floor"],
                    )
                )
            for selected in selected_models:
                prediction, coefficient, metadata = fit_selected(
                    selected, train_x, train_target, test_x, settings["sigma_floor"]
                )
                result_rows.extend(
                    metric_rows(
                        system, heldout, selected, target_name,
                        train_target, test_target, prediction, settings,
                    )
                )
                hyperparameter_rows.append(
                    {
                        "system_id": system,
                        "heldout_replica": heldout,
                        "target": target_name,
                        "model": selected.model,
                        "preprocessing": selected.preprocessing,
                        "alpha": selected.alpha,
                        "pca_variance": selected.pca_variance,
                        "inner_normalized_mae": selected.inner_normalized_mae,
                        **metadata,
                    }
                )
                coefficient_rows.extend(
                    {
                        "system_id": system,
                        "heldout_replica": heldout,
                        "target": target_name,
                        "model": selected.model,
                        "preprocessing": selected.preprocessing,
                        "residue_index": residue,
                        "coefficient": float(value),
                    }
                    for residue, value in enumerate(coefficient)
                )
    return result_rows, hyperparameter_rows, coefficient_rows


def aggregate(results: pd.DataFrame, hyperparameters: pd.DataFrame) -> dict:
    summary = []
    for keys, group in results.groupby(["target", "model", "preprocessing", "band"]):
        target, model, preprocessing, band = keys
        summary.append(
            {
                "target": target,
                "model": model,
                "preprocessing": preprocessing,
                "band": band,
                "system_folds": len(group),
                "median_recovery_percent": float(100 * group.distribution_recovery.median()),
                "median_normalized_mae": float(group.normalized_mae.median()),
                "median_spearman_rho": float(group.spearman_rho.median()),
            }
        )
    return {
        "checkpoint": "3B",
        "status": "measurement_complete",
        "decision": "pause_for_review",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "models": ["ridge", "pca_ridge"],
        "preprocessing": list(PREPROCESSING),
        "hyperparameter_rows": len(hyperparameters),
        "summary": summary,
        "next_step": "capped exact kNN; requires review",
    }


def write_plots(results: pd.DataFrame, output) -> None:
    import matplotlib.pyplot as plt

    selected = results[
        ((results.target == "rmsd") & (results.band == "global"))
        | ((results.target == "w1") & (results.band == "q5"))
    ].copy()
    selected["arm"] = selected.model + " / " + selected.preprocessing
    summary = 100 * selected.groupby(["arm", "target"]).distribution_recovery.median()
    summary = summary.unstack("target").reindex(columns=["rmsd", "w1"])
    baseline_path = output.parent / "checkpoint2_boundary" / "boundary_results.parquet"
    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
        baseline = baseline[
            (baseline.method == "clipped_all")
            & (baseline.representation == "raw")
            & (baseline.metric == "l1")
            & (
                ((baseline.target == "rmsd") & (baseline.band == "global"))
                | ((baseline.target == "w1") & (baseline.band == "q5"))
            )
        ]
        summary.loc["Absolute-L1 scalar"] = (
            100 * baseline.groupby("target").distribution_recovery.median()
        ).reindex(["rmsd", "w1"])
    order = [
        "Absolute-L1 scalar", "ridge / raw", "ridge / zscore",
        "pca_ridge / raw", "pca_ridge / zscore",
    ]
    summary = summary.reindex(order)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, summary["rmsd"], width, label="Global RMSD")
    ax.bar(x + width / 2, summary["w1"], width, label="W1 q5")
    ax.set_xticks(x, order, rotation=18, ha="right")
    ax.set_ylabel("Median distribution recovery (%)")
    ax.set_title("Full-pair per-residue vector models")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "ridge_pca_tail_recovery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    systems = load_systems()[: args.limit]
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    results, hyperparameters, coefficients = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in systems}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            system_results, system_hyperparameters, system_coefficients = future.result()
            results.extend(system_results)
            hyperparameters.extend(system_hyperparameters)
            coefficients.extend(system_coefficients)
            print(f"[{index}/{len(systems)}] {row['system_id']} ridge/PCA complete", flush=True)
    results_frame = pd.DataFrame(results)
    hyperparameter_frame = pd.DataFrame(hyperparameters)
    coefficient_frame = pd.DataFrame(coefficients)
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint3_vector"
    output.mkdir(parents=True, exist_ok=True)
    results_frame.to_parquet(output / "ridge_pca_results.parquet", index=False)
    hyperparameter_frame.to_parquet(output / "ridge_pca_hyperparameters.parquet", index=False)
    coefficient_frame.to_parquet(output / "ridge_pca_coefficients.parquet", index=False)
    report = aggregate(results_frame, hyperparameter_frame)
    atomic_yaml(output / "checkpoint3b_report.yaml", report)
    write_plots(results_frame, output)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
