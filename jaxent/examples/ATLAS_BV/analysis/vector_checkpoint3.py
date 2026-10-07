"""Checkpoint 3A: fold-safe per-residue vector support and scalar baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 import (
    ARMS,
    boundary_predictions,
    probability_distribution_errors,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    PairSet,
    align_to_structure,
    make_fold_pairs,
    pair_rmsd,
    pf_pair_distance,
    transform_logpf,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    intraframe_distance_distributions,
    target_bands,
    w1_pair_distance,
)


@dataclass(frozen=True)
class InnerDirection:
    fit_replica: int
    validation_replica: int
    fit_pairs: PairSet
    validation_pairs: PairSet


def deterministic_cap(length: int, cap: int, seed: int) -> np.ndarray:
    """Return a stable, sorted no-replacement subset."""
    if length <= cap:
        return np.arange(length, dtype=np.int64)
    chosen = np.random.default_rng(seed).choice(length, size=cap, replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def absolute_change_vectors(
    z: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Pairs x residues absolute fixed-BV changes."""
    return np.abs(z[:, left] - z[:, right]).T


def feature_standardizer(
    train: np.ndarray,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Training-pair-only feature mean and scale."""
    mean = train.mean(axis=0)
    sigma = train.std(axis=0)
    floored = sigma < sigma_floor
    return mean, np.maximum(sigma, sigma_floor), int(floored.sum())


def _within_pair_set(
    aligned: np.ndarray,
    replicas: np.ndarray,
    frames: np.ndarray,
    replica: int,
    theiler: int,
    cap: int,
    seed: int,
) -> PairSet:
    indices = np.flatnonzero(replicas == replica)
    local_left, local_right = np.triu_indices(len(indices), k=1)
    keep = np.abs(frames[indices[local_left]] - frames[indices[local_right]]) > theiler
    left = indices[local_left[keep]]
    right = indices[local_right[keep]]
    selected = deterministic_cap(len(left), cap, seed)
    left, right = left[selected], right[selected]
    return PairSet(left, right, pair_rmsd(aligned, left, right))


def make_inner_directions(
    aligned: np.ndarray,
    replicas: np.ndarray,
    frames: np.ndarray,
    heldout: int,
    theiler: dict[int, int],
    pair_cap: int,
    seed: int,
) -> tuple[InnerDirection, InnerDirection]:
    """A→B and B→A tuning splits with no cross-replica pairs."""
    first, second = (replica for replica in (1, 2, 3) if replica != heldout)
    first_pairs = _within_pair_set(
        aligned, replicas, frames, first, theiler[first], pair_cap, seed + 101 * first
    )
    second_pairs = _within_pair_set(
        aligned, replicas, frames, second, theiler[second], pair_cap, seed + 101 * second
    )
    return (
        InnerDirection(first, second, first_pairs, second_pairs),
        InnerDirection(second, first, second_pairs, first_pairs),
    )


def _scalar_rows(
    system: str,
    heldout: int,
    representation: str,
    metric: str,
    train_distance: np.ndarray,
    test_distance: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    settings: dict,
) -> list[dict]:
    rows = []
    boundary = settings["boundary_audit"]
    for target_name, (train_target, test_target) in targets.items():
        fitted = boundary_predictions(
            train_distance,
            train_target,
            test_distance,
            boundary["tail_fraction"],
            boundary["tail_minimum_pairs"],
        )
        prediction = fitted["clipped_test"]
        target_iqr = float(np.subtract(*np.quantile(train_target, [0.75, 0.25])))
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
                    "representation": representation,
                    "metric": metric,
                    "model": "scalar_isotonic",
                    "preprocessing": representation,
                    "band": band,
                    "pairs": int(mask.sum()),
                    "mae": mae,
                    "normalized_mae": mae / target_iqr if target_iqr > 0 else np.nan,
                    **distribution,
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
    if z.shape[1] != len(coordinates):
        raise ValueError(f"{system}: PF/coordinate frame mismatch")

    reference_centered = reference - reference.mean(axis=0)
    rmsd_to_start = np.sqrt(
        np.mean(np.sum((aligned - reference_centered) ** 2, axis=2), axis=1)
    )
    global_pf = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_to_start[mask]),
            integrated_autocorrelation_frames(global_pf[mask]),
        )

    audit_rows, scalar_rows = [], []
    base_seed = config["analysis"]["seed"]
    for heldout in (1, 2, 3):
        train_frames = np.flatnonzero(replicas != heldout)
        train_pairs, test_pairs = make_fold_pairs(
            aligned,
            replicas,
            frames,
            heldout,
            theiler,
            settings["train_pairs"],
            settings["test_pairs"],
            base_seed,
        )
        train_take = deterministic_cap(
            len(train_pairs.left), vector["knn_train_pairs"], base_seed + 1009 * heldout
        )
        test_take = deterministic_cap(
            len(test_pairs.left), vector["knn_test_pairs"], base_seed + 2003 * heldout
        )
        train_left, train_right = train_pairs.left[train_take], train_pairs.right[train_take]
        test_left, test_right = test_pairs.left[test_take], test_pairs.right[test_take]
        inner = make_inner_directions(
            aligned,
            replicas,
            frames,
            heldout,
            theiler,
            vector["inner_pairs_per_replica"],
            base_seed + 3001 * heldout,
        )
        train_x = absolute_change_vectors(z, train_left, train_right)
        _, sigma, floored = feature_standardizer(train_x, settings["sigma_floor"])
        outer_clean = bool(
            np.all(replicas[train_left] != heldout)
            and np.all(replicas[train_right] != heldout)
            and np.all(replicas[test_left] == heldout)
            and np.all(replicas[test_right] == heldout)
        )
        inner_clean = all(
            np.all(replicas[direction.fit_pairs.left] == direction.fit_replica)
            and np.all(replicas[direction.fit_pairs.right] == direction.fit_replica)
            and np.all(replicas[direction.validation_pairs.left] == direction.validation_replica)
            and np.all(replicas[direction.validation_pairs.right] == direction.validation_replica)
            and heldout not in {direction.fit_replica, direction.validation_replica}
            for direction in inner
        )
        if not outer_clean or not inner_clean:
            raise AssertionError(f"{system} fold {heldout}: replica leakage")

        audit_rows.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "residues": z.shape[0],
                "outer_train_pairs_full": len(train_pairs.left),
                "outer_test_pairs_full": len(test_pairs.left),
                "capped_train_pairs": len(train_take),
                "capped_test_pairs": len(test_take),
                "inner_pairs_first": len(inner[0].fit_pairs.left),
                "inner_pairs_second": len(inner[0].validation_pairs.left),
                "zscore_floored_residues": floored,
                "minimum_feature_sigma": float(sigma.min()),
                "outer_replica_isolation": outer_clean,
                "inner_replica_isolation": inner_clean,
                "train_cap_checksum": int(np.dot(train_take + 1, np.arange(1, len(train_take) + 1))),
                "test_cap_checksum": int(np.dot(test_take + 1, np.arange(1, len(test_take) + 1))),
            }
        )

        train_w1_full = w1_pair_distance(
            distributions,
            train_pairs.left,
            train_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        test_w1_full = w1_pair_distance(
            distributions,
            test_pairs.left,
            test_pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        )
        targets = {
            "rmsd": (train_pairs.rmsd[train_take], test_pairs.rmsd[test_take]),
            "w1": (train_w1_full[train_take], test_w1_full[test_take]),
        }
        for representation, metric in ARMS:
            transformed, _ = transform_logpf(
                z, train_frames, representation, settings["sigma_floor"]
            )
            train_distance = pf_pair_distance(
                transformed, train_left, train_right, metric, settings["distance_chunk_size"]
            )
            test_distance = pf_pair_distance(
                transformed, test_left, test_right, metric, settings["distance_chunk_size"]
            )
            scalar_rows.extend(
                _scalar_rows(
                    system,
                    heldout,
                    representation,
                    metric,
                    train_distance,
                    test_distance,
                    targets,
                    settings,
                )
            )
    return audit_rows, scalar_rows


def aggregate(audit: pd.DataFrame, scalar: pd.DataFrame) -> dict:
    recovery = []
    for keys, group in scalar.groupby(["target", "representation", "metric", "band"]):
        target, representation, metric, band = keys
        recovery.append(
            {
                "target": target,
                "representation": representation,
                "metric": metric,
                "band": band,
                "system_folds": len(group),
                "median_pairs": float(group.pairs.median()),
                "median_recovery_percent": float(100 * group.distribution_recovery.median()),
                "median_sqrt_jsd": float(group.distribution_sqrt_jsd.median()),
            }
        )
    return {
        "checkpoint": "3A",
        "status": "measurement_complete",
        "decision": "pause_for_review",
        "systems": int(audit.system_id.nunique()),
        "folds": len(audit),
        "outer_replica_isolation_all": bool(audit.outer_replica_isolation.all()),
        "inner_replica_isolation_all": bool(audit.inner_replica_isolation.all()),
        "median_residues": float(audit.residues.median()),
        "median_zscore_floored_residues": float(audit.zscore_floored_residues.median()),
        "scalar_baseline_summary": recovery,
        "next_step": "full-pair ridge and PCA-ridge; requires review",
    }


def write_plots(audit: pd.DataFrame, scalar: pd.DataFrame, output) -> None:
    import matplotlib.pyplot as plt

    labels = {
        ("raw", "l1"): "Absolute-L1",
        ("raw", "l2"): "Raw L2",
        ("frame_centered", "l2"): "Frame-centred L2",
        ("raw", "cosine"): "Cosine",
        ("raw", "correlation"): "Correlation",
    }
    tail = scalar[
        ((scalar.target == "rmsd") & (scalar.band == "global"))
        | ((scalar.target == "w1") & (scalar.band == "q5"))
    ].copy()
    tail["arm"] = [labels[(row.representation, row.metric)] for row in tail.itertuples()]
    summary = (
        100
        * tail.groupby(["target", "arm"]).distribution_recovery.median()
    ).unstack("target")
    order = ["Absolute-L1", "Raw L2", "Frame-centred L2", "Cosine", "Correlation"]
    summary = summary.reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(order))
    width = 0.36
    ax.bar(x - width / 2, summary["rmsd"], width, label="Global RMSD")
    ax.bar(x + width / 2, summary["w1"], width, label="W1 q5")
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("Median distribution recovery (%)")
    ax.set_title("Capped scalar baselines for the vector-model comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "capped_scalar_tail_recovery.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(audit.residues, bins=15, color="#4477AA", alpha=0.85)
    axes[0].set_xlabel("Residues / vector dimensions")
    axes[0].set_ylabel("System × replica folds")
    axes[1].hist(audit.minimum_feature_sigma, bins=15, color="#228833", alpha=0.85)
    axes[1].set_xlabel("Minimum training-pair feature standard deviation")
    axes[1].set_ylabel("System × replica folds")
    fig.tight_layout()
    fig.savefig(output / "vector_feature_support.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config()
    systems = load_systems()[: args.limit]
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    audit_rows, scalar_rows = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in systems}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            audit, scalar = future.result()
            audit_rows.extend(audit)
            scalar_rows.extend(scalar)
            print(f"[{index}/{len(systems)}] {row['system_id']} vector support complete", flush=True)
    audit = pd.DataFrame(audit_rows)
    scalar = pd.DataFrame(scalar_rows)
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint3_vector"
    output.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output / "feature_support_audit.parquet", index=False)
    scalar.to_parquet(output / "capped_scalar_baselines.parquet", index=False)
    report = aggregate(audit, scalar)
    atomic_yaml(output / "checkpoint3a_report.yaml", report)
    write_plots(audit, scalar, output)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
