"""Measure replica-held-out PF geometry against pairwise C-alpha RMSD.

This is a measurement-only successor to the historical occupancy-based Stage 1.
It deliberately does not fit BV coefficients or authorize a later stage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.within_basin_stage1 import native_contacts


REPRESENTATIONS = ("raw", "frame_centered", "residue_standardized", "centered_standardized")
PF_METRICS = ("l1", "l2")
RMSD_QUANTILES = np.asarray([0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])


@dataclass(frozen=True)
class PairSet:
    left: np.ndarray
    right: np.ndarray
    rmsd: np.ndarray


def align_to_structure(coordinates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Kabsch-align every frame to a fixed structure without using held-out frames."""
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)
    covariance = np.einsum("fai,aj->fij", centered, reference)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    reflected = np.linalg.det(rotation) < 0
    left[reflected, :, -1] *= -1
    rotation = left @ right_t
    return centered @ rotation


def pair_rmsd(aligned: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = aligned[left] - aligned[right]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))


def transform_logpf(
    z: np.ndarray,
    train_frames: np.ndarray,
    representation: str,
    sigma_floor: float,
) -> tuple[np.ndarray, int]:
    """Apply a fold-local representation transform to residues x frames log-PF."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    transformed = np.asarray(z, dtype=np.float64).copy()
    if representation in {"frame_centered", "centered_standardized"}:
        transformed -= transformed.mean(axis=0, keepdims=True)
    floored = 0
    if representation in {"residue_standardized", "centered_standardized"}:
        mean = transformed[:, train_frames].mean(axis=1, keepdims=True)
        sigma = transformed[:, train_frames].std(axis=1, keepdims=True)
        floored = int(np.count_nonzero(sigma < sigma_floor))
        sigma = np.maximum(sigma, sigma_floor)
        transformed = (transformed - mean) / sigma
    return transformed, floored


def pf_pair_distance(
    z: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Normalized residue-wise PF distance for a sampled set of frame pairs."""
    result = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        left_values = z[:, left[start:stop]]
        right_values = z[:, right[start:stop]]
        delta = left_values - right_values
        if metric == "l1":
            result[start:stop] = np.mean(np.abs(delta), axis=0)
        elif metric == "l2":
            result[start:stop] = np.sqrt(np.mean(delta * delta, axis=0))
        elif metric in {"cosine", "correlation"}:
            if metric == "correlation":
                left_values = left_values - left_values.mean(axis=0, keepdims=True)
                right_values = right_values - right_values.mean(axis=0, keepdims=True)
            numerator = np.sum(left_values * right_values, axis=0)
            left_norm = np.sqrt(np.sum(left_values * left_values, axis=0))
            right_norm = np.sqrt(np.sum(right_values * right_values, axis=0))
            denominator = left_norm * right_norm
            distance = np.ones(stop - start, dtype=np.float64)
            valid = denominator > np.finfo(np.float64).eps
            distance[valid] = 1.0 - numerator[valid] / denominator[valid]
            both_zero = (left_norm <= np.finfo(np.float64).eps) & (
                right_norm <= np.finfo(np.float64).eps
            )
            distance[both_zero] = 0.0
            result[start:stop] = np.clip(distance, 0.0, 2.0)
        else:
            raise ValueError(f"unknown PF metric: {metric}")
    return result


def _within_candidates(indices: np.ndarray, frames: np.ndarray, theiler: int) -> tuple[np.ndarray, np.ndarray]:
    local_left, local_right = np.triu_indices(len(indices), k=1)
    keep = np.abs(frames[indices[local_left]] - frames[indices[local_right]]) > theiler
    return indices[local_left[keep]], indices[local_right[keep]]


def _cross_candidates(left_indices: np.ndarray, right_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(left_indices, len(right_indices)), np.tile(right_indices, len(left_indices))


def _sample_pairs(
    left: np.ndarray,
    right: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(left):
        return left, right
    chosen = rng.choice(len(left), size=min(count, len(left)), replace=False)
    return left[chosen], right[chosen]


def make_fold_pairs(
    aligned: np.ndarray,
    replicas: np.ndarray,
    frames: np.ndarray,
    heldout: int,
    theiler_by_replica: dict[int, int],
    train_pair_count: int,
    test_pair_count: int,
    seed: int,
) -> tuple[PairSet, PairSet]:
    """Create common deterministic train/test support for every metric arm."""
    rng = np.random.default_rng(seed + heldout)
    training_replicas = [replica for replica in (1, 2, 3) if replica != heldout]
    per_source = max(1, train_pair_count // 3)
    train_left, train_right = [], []
    train_indices = []
    for replica in training_replicas:
        indices = np.flatnonzero(replicas == replica)
        train_indices.append(indices)
        left, right = _within_candidates(indices, frames, theiler_by_replica[replica])
        left, right = _sample_pairs(left, right, per_source, rng)
        train_left.append(left)
        train_right.append(right)
    left, right = _cross_candidates(train_indices[0], train_indices[1])
    left, right = _sample_pairs(left, right, train_pair_count - sum(map(len, train_left)), rng)
    train_left.append(left)
    train_right.append(right)
    train_left_array = np.concatenate(train_left)
    train_right_array = np.concatenate(train_right)

    test_indices = np.flatnonzero(replicas == heldout)
    test_left, test_right = _within_candidates(
        test_indices, frames, theiler_by_replica[heldout]
    )
    test_left, test_right = _sample_pairs(test_left, test_right, test_pair_count, rng)
    return (
        PairSet(train_left_array, train_right_array, pair_rmsd(aligned, train_left_array, train_right_array)),
        PairSet(test_left, test_right, pair_rmsd(aligned, test_left, test_right)),
    )


def calibration_metrics(
    train_distance: np.ndarray,
    train_rmsd: np.ndarray,
    test_distance: np.ndarray,
    test_rmsd: np.ndarray,
    interval_bins: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Fit continuous monotone calibration and score its held-out distribution error."""
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(train_distance, train_rmsd)
    train_prediction = model.predict(train_distance)
    prediction = model.predict(test_distance)
    error = np.abs(prediction - test_rmsd)
    mae = float(error.mean())
    scale = float(np.subtract(*np.quantile(train_rmsd, [0.75, 0.25])))
    normalized_mae = mae / scale if scale > 0 else float("nan")
    null_mae = float(np.mean(np.abs(test_rmsd - np.median(train_rmsd))))
    skill = 1.0 - mae / null_mae if null_mae > 0 else float("nan")
    rho = (
        0.0
        if np.ptp(test_distance) <= 1e-15 or np.ptp(test_rmsd) <= 1e-15
        else spearmanr(test_distance, test_rmsd).statistic
    )

    edges = np.unique(np.quantile(train_distance, np.linspace(0, 1, interval_bins + 1)))
    coverage = np.full(len(test_distance), False)
    if len(edges) >= 2:
        train_labels = np.clip(np.digitize(train_distance, edges[1:-1]), 0, len(edges) - 2)
        test_labels = np.clip(np.digitize(test_distance, edges[1:-1]), 0, len(edges) - 2)
        residual = train_rmsd - train_prediction
        for label in range(len(edges) - 1):
            train_mask = train_labels == label
            if np.count_nonzero(train_mask) < 2:
                continue
            low, high = np.quantile(residual[train_mask], [0.05, 0.95])
            test_mask = test_labels == label
            coverage[test_mask] = (test_rmsd[test_mask] >= prediction[test_mask] + low) & (
                test_rmsd[test_mask] <= prediction[test_mask] + high
            )
    metrics = {
        "mae_angstrom": mae,
        "normalized_mae": float(normalized_mae),
        "null_mae_angstrom": null_mae,
        "skill_vs_train_median": float(skill),
        "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "interval_90_coverage": float(coverage.mean()),
    }
    return metrics, prediction, coverage


def conditional_curve_rows(
    system: str,
    heldout: int,
    scope: str,
    representation: str,
    metric: str,
    train_rmsd: np.ndarray,
    test_rmsd: np.ndarray,
    test_distance: np.ndarray,
    prediction: np.ndarray,
    coverage: np.ndarray,
) -> list[dict]:
    edges = np.unique(np.quantile(train_rmsd, RMSD_QUANTILES))
    rows = []
    if len(edges) < 2:
        return rows
    labels = np.clip(np.digitize(test_rmsd, edges[1:-1]), 0, len(edges) - 2)
    for label in range(len(edges) - 1):
        mask = labels == label
        if not mask.any():
            continue
        rows.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "scope": scope,
                "representation": representation,
                "metric": metric,
                "rmsd_band": label,
                "rmsd_low": float(edges[label]),
                "rmsd_high": float(edges[label + 1]),
                "pairs": int(mask.sum()),
                "median_rmsd": float(np.median(test_rmsd[mask])),
                "median_pf_distance": float(np.median(test_distance[mask])),
                "pf_distance_q25": float(np.quantile(test_distance[mask], 0.25)),
                "pf_distance_q75": float(np.quantile(test_distance[mask], 0.75)),
                "mae_angstrom": float(np.mean(np.abs(prediction[mask] - test_rmsd[mask]))),
                "interval_90_coverage": float(np.mean(coverage[mask])),
                "coverage_error_vs_nominal": float(np.mean(coverage[mask]) - 0.90),
            }
        )
    return rows


def _score_arm(
    system: str,
    heldout: int,
    scope: str,
    representation: str,
    metric: str,
    train_distance: np.ndarray,
    train_rmsd: np.ndarray,
    test_distance: np.ndarray,
    test_rmsd: np.ndarray,
    floored_residues: int,
    interval_bins: int,
) -> tuple[dict, list[dict]]:
    metrics, prediction, coverage = calibration_metrics(
        train_distance, train_rmsd, test_distance, test_rmsd, interval_bins
    )
    row = {
        "system_id": system,
        "heldout_replica": heldout,
        "scope": scope,
        "representation": representation,
        "metric": metric,
        "train_pairs": len(train_distance),
        "test_pairs": len(test_distance),
        "floored_residues": floored_residues,
        **metrics,
    }
    curves = conditional_curve_rows(
        system,
        heldout,
        scope,
        representation,
        metric,
        train_rmsd,
        test_rmsd,
        test_distance,
        prediction,
        coverage,
    )
    return row, curves


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict], dict]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = align_to_structure(coordinates, reference)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    if z.shape[1] != len(coordinates):
        raise ValueError(f"{system}: PF/coordinate frame mismatch {z.shape[1]} != {len(coordinates)}")

    rmsd_to_start = np.sqrt(np.mean(np.sum((aligned - (reference - reference.mean(axis=0))) ** 2, axis=2), axis=1))
    g = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_to_start[mask]),
            integrated_autocorrelation_frames(g[mask]),
        )

    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))
    result_rows: list[dict] = []
    curve_rows: list[dict] = []
    fold_metadata = []
    for heldout in (1, 2, 3):
        train_mask = replicas != heldout
        train_frames = np.flatnonzero(train_mask)
        reference_index = int(train_frames[np.argmin(rmsd_to_start[train_mask])])
        native = native_contacts(coordinates, reference_index, config)
        train_pairs, test_pairs = make_fold_pairs(
            aligned,
            replicas,
            frames,
            heldout,
            theiler,
            settings["train_pairs"],
            settings["test_pairs"],
            config["analysis"]["seed"],
        )
        fold_metadata.append(
            {
                "heldout_replica": heldout,
                "theiler_frames": theiler[heldout],
                "train_pairs": len(train_pairs.left),
                "test_pairs": len(test_pairs.left),
            }
        )
        train_flat = aligned[train_mask].reshape(np.count_nonzero(train_mask), -1)
        medoid_local = int(
            np.argmin(np.linalg.norm(train_flat - train_flat.mean(axis=0), axis=1))
        )
        medoid = train_frames[medoid_local]
        heldout_frames = np.flatnonzero(~train_mask)
        folded_train = train_frames[train_frames != medoid]
        folded_train_rmsd = pair_rmsd(
            aligned, folded_train, np.full(len(folded_train), medoid)
        )
        folded_test_rmsd = pair_rmsd(
            aligned, heldout_frames, np.full(len(heldout_frames), medoid)
        )
        for representation in REPRESENTATIONS:
            transformed, floored = transform_logpf(
                z, train_frames, representation, settings["sigma_floor"]
            )
            for metric in PF_METRICS:
                train_distance = pf_pair_distance(
                    transformed, train_pairs.left, train_pairs.right, metric, settings["distance_chunk_size"]
                )
                test_distance = pf_pair_distance(
                    transformed, test_pairs.left, test_pairs.right, metric, settings["distance_chunk_size"]
                )
                score, curves = _score_arm(
                    system,
                    heldout,
                    "pairwise",
                    representation,
                    metric,
                    train_distance,
                    train_pairs.rmsd,
                    test_distance,
                    test_pairs.rmsd,
                    floored,
                    settings["interval_bins"],
                )
                result_rows.append(score)
                curve_rows.extend(curves)

                # Secondary folded-proxy view: closest-to-centroid training frame.
                folded_train_distance = pf_pair_distance(
                    transformed,
                    folded_train,
                    np.full(len(folded_train), medoid),
                    metric,
                    settings["distance_chunk_size"],
                )
                folded_test_distance = pf_pair_distance(
                    transformed,
                    heldout_frames,
                    np.full(len(heldout_frames), medoid),
                    metric,
                    settings["distance_chunk_size"],
                )
                score, curves = _score_arm(
                    system,
                    heldout,
                    "folded_proxy",
                    representation,
                    metric,
                    folded_train_distance,
                    folded_train_rmsd,
                    folded_test_distance,
                    folded_test_rmsd,
                    floored,
                    settings["interval_bins"],
                )
                result_rows.append(score)
                curve_rows.extend(curves)

        for metric, scalar in (("rg", rg), ("native_contacts", native)):
            train_distance = np.abs(scalar[train_pairs.left] - scalar[train_pairs.right])
            test_distance = np.abs(scalar[test_pairs.left] - scalar[test_pairs.right])
            score, curves = _score_arm(
                system,
                heldout,
                "pairwise",
                "control",
                metric,
                train_distance,
                train_pairs.rmsd,
                test_distance,
                test_pairs.rmsd,
                0,
                settings["interval_bins"],
            )
            result_rows.append(score)
            curve_rows.extend(curves)
    metadata = {
        "system_id": system,
        "frames": len(coordinates),
        "residues": z.shape[0],
        "ca_atoms": aligned.shape[1],
        "folds": fold_metadata,
    }
    return result_rows, curve_rows, metadata


def aggregate(results: pd.DataFrame, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pairwise = results[results.scope == "pairwise"]
    per_system = (
        pairwise.groupby(["system_id", "representation", "metric"], as_index=False)
        .agg(
            normalized_mae=("normalized_mae", "mean"),
            skill_vs_train_median=("skill_vs_train_median", "mean"),
            spearman_rho=("spearman_rho", "mean"),
            interval_90_coverage=("interval_90_coverage", "mean"),
        )
    )
    arms = []
    for (representation, metric), group in per_system.groupby(["representation", "metric"]):
        values = group["skill_vs_train_median"].to_numpy()
        draws = np.median(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
        arms.append(
            {
                "representation": representation,
                "metric": metric,
                "systems": len(group),
                "median_normalized_mae": float(group.normalized_mae.median()),
                "median_skill_vs_train_median": float(np.median(values)),
                "skill_system_bootstrap_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
                "median_spearman_rho": float(group.spearman_rho.median()),
                "median_interval_90_coverage": float(group.interval_90_coverage.median()),
            }
        )
    return {
        "redesign_version": "pairwise_rmsd_geometry_v1",
        "decision": "measurement_only",
        "stage2_authorized": False,
        "systems": int(per_system.system_id.nunique()),
        "replica_validation": "leave_one_replica_out",
        "bv_coefficients": "fixed_default",
        "arms": arms,
    }


def write_plots(results: pd.DataFrame, curves: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pairwise = results[(results.scope == "pairwise") & (results.representation != "control")]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, values = [], []
    for (representation, metric), group in pairwise.groupby(["representation", "metric"]):
        per_system = group.groupby("system_id").skill_vs_train_median.mean()
        labels.append(f"{representation}\n{metric.upper()}")
        values.append(per_system.to_numpy())
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Held-out skill vs training-median RMSD")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_skill.png", dpi=180)
    plt.close(fig)

    selected = curves[(curves.scope == "pairwise") & (curves.representation == "raw")]
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, group in selected.groupby("metric"):
        summary = group.groupby("rmsd_band").agg(rmsd=("median_rmsd", "median"), pf=("median_pf_distance", "median"))
        ax.plot(summary.rmsd, summary.pf, marker="o", label=metric.upper())
    ax.set_xlabel("Pairwise C-alpha RMSD (angstrom)")
    ax.set_ylabel("Median fixed-BV PF distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "raw_distance_vs_rmsd.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    settings = config["analysis"]["pairwise_geometry"]
    workers = args.workers or settings["workers"]
    rows = load_systems()[: args.limit]
    all_results, all_curves, metadata = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result, curves, system_metadata = future.result()
            all_results.extend(result)
            all_curves.extend(curves)
            metadata.append(system_metadata)
            print(f"[{index}/{len(rows)}] {row['system_id']} geometry complete", flush=True)
    results = pd.DataFrame(all_results).sort_values(
        ["system_id", "scope", "heldout_replica", "representation", "metric"]
    )
    curves = pd.DataFrame(all_curves).sort_values(
        ["system_id", "scope", "heldout_replica", "representation", "metric", "rmsd_band"]
    )
    output_dir = HERE / "outputs" / "analysis" / "pairwise_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output_dir / "fold_results.parquet", index=False)
    curves.to_parquet(output_dir / "conditional_curves.parquet", index=False)
    pd.DataFrame(metadata).to_parquet(output_dir / "systems.parquet", index=False)
    report = aggregate(
        results, settings["system_bootstrap_samples"], config["analysis"]["seed"]
    )
    atomic_yaml(output_dir / "aggregate.yaml", report)
    atomic_yaml(HERE / "outputs" / "analysis" / "stage1_geometry_measurement.yaml", report)
    write_plots(results, curves, output_dir)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
