"""Checkpoint 1: structural support, effective-frame, and C-alpha W1 audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    integrated_autocorrelation_frames,
    load_config,
    load_contact_coordinates,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    PF_METRICS,
    REPRESENTATIONS,
    RMSD_QUANTILES,
    align_to_structure,
    calibration_metrics,
    make_fold_pairs,
    pf_pair_distance,
    transform_logpf,
)


RMSD_REGIMES = (
    ("hyperlocal", -np.inf, 1.25),
    ("local", 1.25, 2.5),
    ("global", 2.5, np.inf),
)


def intraframe_distance_distributions(coordinates: np.ndarray) -> np.ndarray:
    """Sorted unique C-alpha pair distances for every frame."""
    return np.asarray([np.sort(pdist(frame)) for frame in coordinates], dtype=np.float32)


def w1_pair_distance(
    distributions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    max_chunk_values: int = 2_000_000,
) -> np.ndarray:
    """Exact empirical W1 for equal-sized sorted one-dimensional samples."""
    features = distributions.shape[1]
    chunk = max(1, max_chunk_values // features)
    result = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        result[start:stop] = np.mean(
            np.abs(distributions[left[start:stop]] - distributions[right[start:stop]]),
            axis=1,
        )
    return result


def quantile_signatures(distributions: np.ndarray, count: int) -> np.ndarray:
    """Compressed inverse-CDF signatures used only for nearest-frame support."""
    positions = np.linspace(0, distributions.shape[1] - 1, count)
    low = np.floor(positions).astype(int)
    high = np.ceil(positions).astype(int)
    fraction = (positions - low)[None, :]
    return distributions[:, low] * (1.0 - fraction) + distributions[:, high] * fraction


def effective_endpoint_frames(left: np.ndarray, right: np.ndarray) -> tuple[int, float]:
    if not len(left):
        return 0, 0.0
    indices, counts = np.unique(np.concatenate([left, right]), return_counts=True)
    effective = float(counts.sum() ** 2 / np.sum(counts.astype(float) ** 2))
    return int(len(indices)), effective


def _nearest_cross_support(
    representation: np.ndarray,
    replicas: np.ndarray,
    training_replicas: tuple[int, int],
    heldout: int,
    metric: str,
) -> tuple[np.ndarray, float]:
    first = np.flatnonzero(replicas == training_replicas[0])
    second = np.flatnonzero(replicas == training_replicas[1])
    test = np.flatnonzero(replicas == heldout)
    if metric == "euclidean":
        cross = pairwise_distances(representation[first], representation[second])
    elif metric == "manhattan_mean":
        cross = pairwise_distances(representation[first], representation[second], metric="manhattan")
        cross /= representation.shape[1]
    else:
        raise ValueError(metric)
    reference_nearest = np.concatenate([cross.min(axis=1), cross.min(axis=0)])
    threshold = float(np.quantile(reference_nearest, 0.95))
    training = np.concatenate([first, second])
    if metric == "euclidean":
        nearest = pairwise_distances(representation[test], representation[training]).min(axis=1)
    else:
        nearest = pairwise_distances(
            representation[test], representation[training], metric="manhattan"
        ).min(axis=1) / representation.shape[1]
    novelty = np.zeros(len(replicas), dtype=bool)
    novelty[test] = nearest > threshold
    return novelty, threshold


def support_novelty(
    aligned: np.ndarray,
    w1_signatures: np.ndarray,
    replicas: np.ndarray,
    heldout: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    training = tuple(replica for replica in (1, 2, 3) if replica != heldout)
    rmsd_vectors = aligned.reshape(len(aligned), -1) / np.sqrt(aligned.shape[1])
    rmsd_novel, rmsd_threshold = _nearest_cross_support(
        rmsd_vectors, replicas, training, heldout, "euclidean"
    )
    w1_novel, w1_threshold = _nearest_cross_support(
        w1_signatures, replicas, training, heldout, "manhattan_mean"
    )
    return {"rmsd": rmsd_novel, "w1": w1_novel}, {
        "rmsd": rmsd_threshold,
        "w1": w1_threshold,
    }


def target_bands(target: str, train: np.ndarray, test: np.ndarray) -> list[tuple[str, np.ndarray, float, float]]:
    if target == "rmsd":
        return [
            (name, (test >= low) & (test < high), low, high)
            for name, low, high in RMSD_REGIMES
        ]
    edges = np.unique(np.quantile(train, RMSD_QUANTILES))
    if len(edges) < 2:
        return []
    labels = np.clip(np.digitize(test, edges[1:-1]), 0, len(edges) - 2)
    return [
        (f"q{index}", labels == index, float(edges[index]), float(edges[index + 1]))
        for index in range(len(edges) - 1)
    ]


def audit_rows(
    system: str,
    heldout: int,
    target_name: str,
    representation: str,
    metric: str,
    train_pf: np.ndarray,
    test_pf: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    test_left: np.ndarray,
    test_right: np.ndarray,
    frame_novelty: np.ndarray,
    coverage: np.ndarray,
    prediction: np.ndarray,
) -> list[dict]:
    pf_in_support = (test_pf >= train_pf.min()) & (test_pf <= train_pf.max())
    target_in_support = (test_target >= train_target.min()) & (test_target <= train_target.max())
    pair_novel = frame_novelty[test_left] | frame_novelty[test_right]
    rows = []
    for band, mask, low, high in target_bands(target_name, train_target, test_target):
        if not mask.any():
            continue
        unique_frames, effective_frames = effective_endpoint_frames(test_left[mask], test_right[mask])
        rows.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "target": target_name,
                "representation": representation,
                "metric": metric,
                "band": band,
                "band_low": low,
                "band_high": high,
                "pairs": int(mask.sum()),
                "unique_frames": unique_frames,
                "effective_frames": effective_frames,
                "median_target": float(np.median(test_target[mask])),
                "median_pf_distance": float(np.median(test_pf[mask])),
                "pf_in_support_fraction": float(np.mean(pf_in_support[mask])),
                "target_in_support_fraction": float(np.mean(target_in_support[mask])),
                "novel_endpoint_pair_fraction": float(np.mean(pair_novel[mask])),
                "interval_90_coverage": float(np.mean(coverage[mask])),
                "mae": float(np.mean(np.abs(prediction[mask] - test_target[mask]))),
            }
        )
    return rows


def analyse_system(row: dict[str, str], config: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    audit_settings = settings["support_audit"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = align_to_structure(coordinates, reference)
    distributions = intraframe_distance_distributions(coordinates)
    signatures = quantile_signatures(distributions, audit_settings["w1_support_quantiles"])
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    rmsd_to_start = np.sqrt(
        np.mean(
            np.sum((aligned - (reference - reference.mean(axis=0))) ** 2, axis=2),
            axis=1,
        )
    )
    g = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_to_start[mask]),
            integrated_autocorrelation_frames(g[mask]),
        )

    rows, folds, pair_samples = [], [], []
    approximation_errors = []
    for heldout in (1, 2, 3):
        train_mask = replicas != heldout
        train_frames = np.flatnonzero(train_mask)
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
        train_w1 = w1_pair_distance(
            distributions,
            train_pairs.left,
            train_pairs.right,
            audit_settings["w1_max_chunk_values"],
        )
        test_w1 = w1_pair_distance(
            distributions,
            test_pairs.left,
            test_pairs.right,
            audit_settings["w1_max_chunk_values"],
        )
        approximate = np.mean(
            np.abs(signatures[test_pairs.left] - signatures[test_pairs.right]), axis=1
        )
        approximation_errors.append(np.abs(approximate - test_w1))
        novelty, thresholds = support_novelty(aligned, signatures, replicas, heldout)
        fold_rho = float(spearmanr(test_pairs.rmsd, test_w1).statistic)
        folds.append(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "rmsd_novelty_threshold": thresholds["rmsd"],
                "w1_novelty_threshold": thresholds["w1"],
                "rmsd_w1_spearman": fold_rho,
                "test_pairs": len(test_w1),
            }
        )
        sample_count = min(audit_settings["target_pair_sample"], len(test_w1))
        sample_index = np.linspace(0, len(test_w1) - 1, sample_count, dtype=int)
        pair_samples.extend(
            {
                "system_id": system,
                "heldout_replica": heldout,
                "rmsd": float(test_pairs.rmsd[index]),
                "w1": float(test_w1[index]),
            }
            for index in sample_index
        )
        targets = {
            "rmsd": (train_pairs.rmsd, test_pairs.rmsd),
            "w1": (train_w1, test_w1),
        }
        for representation in REPRESENTATIONS:
            transformed, _ = transform_logpf(
                z, train_frames, representation, settings["sigma_floor"]
            )
            for metric in PF_METRICS:
                train_pf = pf_pair_distance(
                    transformed,
                    train_pairs.left,
                    train_pairs.right,
                    metric,
                    settings["distance_chunk_size"],
                )
                test_pf = pf_pair_distance(
                    transformed,
                    test_pairs.left,
                    test_pairs.right,
                    metric,
                    settings["distance_chunk_size"],
                )
                for target_name, (train_target, test_target) in targets.items():
                    _, prediction, coverage = calibration_metrics(
                        train_pf,
                        train_target,
                        test_pf,
                        test_target,
                        settings["interval_bins"],
                    )
                    rows.extend(
                        audit_rows(
                            system,
                            heldout,
                            target_name,
                            representation,
                            metric,
                            train_pf,
                            test_pf,
                            train_target,
                            test_target,
                            test_pairs.left,
                            test_pairs.right,
                            novelty[target_name],
                            coverage,
                            prediction,
                        )
                    )
    approximation = np.concatenate(approximation_errors)
    metadata = {
        "system_id": system,
        "frames": len(coordinates),
        "ca_atoms": aligned.shape[1],
        "unique_ca_pairs": distributions.shape[1],
        "w1_support_quantiles": signatures.shape[1],
        "w1_support_approx_mae": float(approximation.mean()),
        "w1_support_approx_max_error": float(approximation.max()),
    }
    return rows, folds, pair_samples, metadata


def aggregate(audit: pd.DataFrame, folds: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    selected = audit[
        ((audit.representation == "raw") & (audit.metric == "l2"))
        | ((audit.representation == "frame_centered") & (audit.metric == "l2"))
    ]
    bands = []
    for keys, group in selected.groupby(["target", "representation", "metric", "band"]):
        target, representation, metric, band = keys
        bands.append(
            {
                "target": target,
                "representation": representation,
                "metric": metric,
                "band": band,
                "system_folds": len(group),
                "median_pairs": float(group.pairs.median()),
                "median_unique_frames": float(group.unique_frames.median()),
                "median_effective_frames": float(group.effective_frames.median()),
                "median_pf_in_support_fraction": float(group.pf_in_support_fraction.median()),
                "median_target_in_support_fraction": float(group.target_in_support_fraction.median()),
                "median_novel_endpoint_pair_fraction": float(group.novel_endpoint_pair_fraction.median()),
                "median_interval_90_coverage": float(group.interval_90_coverage.median()),
            }
        )
    return {
        "checkpoint": 1,
        "status": "measurement_complete",
        "decision": "pause_for_review",
        "systems": int(audit.system_id.nunique()),
        "w1_definition": "mean absolute difference between sorted unique intraframe C-alpha pair distances",
        "w1_nearest_support": "256-quantile inverse-CDF approximation",
        "median_w1_support_approx_mae": float(metadata.w1_support_approx_mae.median()),
        "max_w1_support_approx_error": float(metadata.w1_support_approx_max_error.max()),
        "median_rmsd_w1_spearman": float(folds.rmsd_w1_spearman.median()),
        "bands": bands,
    }


def write_plots(audit: pd.DataFrame, pairs: pd.DataFrame, output_dir) -> None:
    import matplotlib.pyplot as plt

    selected = audit[(audit.representation == "raw") & (audit.metric == "l2")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, target in zip(axes, ("rmsd", "w1")):
        sub = selected[selected.target == target]
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else sorted(sub.band.unique())
        grouped = sub.groupby("band")
        coverage = [
            100 * grouped.get_group(band).interval_90_coverage.median()
            if band in grouped.groups
            else np.nan
            for band in order
        ]
        effective = [
            grouped.get_group(band).effective_frames.median()
            if band in grouped.groups
            else np.nan
            for band in order
        ]
        axis.plot(order, coverage, marker="o", label="Coverage (%)")
        axis.axhline(90, color="black", linestyle="--", linewidth=1)
        twin = axis.twinx()
        twin.plot(order, effective, marker="s", color="tab:orange", label="Effective frames")
        axis.set_title(target.upper())
        axis.set_ylabel("Nominal 90% interval coverage (%)")
        twin.set_ylabel("Effective endpoint frames")
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_and_effective_frames.png", dpi=180)
    plt.close(fig)

    sample = pairs.sample(min(100_000, len(pairs)), random_state=20260826)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    hexbin = ax.hexbin(sample.rmsd, sample.w1, gridsize=70, bins="log", mincnt=1)
    fig.colorbar(hexbin, ax=ax, label="Pair count (log colour scale)")
    ax.axvline(1.25, color="black", linestyle=":", linewidth=1)
    ax.axvline(2.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Pairwise C-alpha RMSD (angstrom)")
    ax.set_ylabel("C-alpha internal-distance W1 (angstrom)")
    fig.tight_layout()
    fig.savefig(output_dir / "rmsd_vs_w1.png", dpi=180)
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
    audits, folds, samples, metadata = [], [], [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config): row for row in rows}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            system_audit, system_folds, system_samples, system_metadata = future.result()
            audits.extend(system_audit)
            folds.extend(system_folds)
            samples.extend(system_samples)
            metadata.append(system_metadata)
            print(f"[{index}/{len(rows)}] {row['system_id']} support/W1 complete", flush=True)
    audit = pd.DataFrame(audits)
    fold_table = pd.DataFrame(folds)
    pair_table = pd.DataFrame(samples)
    metadata_table = pd.DataFrame(metadata)
    output_dir = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint1_support_w1"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(output_dir / "support_audit.parquet", index=False)
    fold_table.to_parquet(output_dir / "fold_summary.parquet", index=False)
    pair_table.to_parquet(output_dir / "target_pair_sample.parquet", index=False)
    metadata_table.to_parquet(output_dir / "systems.parquet", index=False)
    report = aggregate(audit, fold_table, metadata_table)
    atomic_yaml(output_dir / "report.yaml", report)
    write_plots(audit, pair_table, output_dir)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
