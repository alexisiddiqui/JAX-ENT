from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import MDAnalysis as mda
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
    replica_paths,
)


def load_ca_coordinates(row: dict[str, str], config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection = config["analysis"]["basins"]["atom_selection"]
    universe = mda.Universe(HERE / row["pdb_path"], *replica_paths(row))
    ca = universe.select_atoms(selection)
    if len(ca) != int(row["length"]):
        raise ValueError(f"{row['system_id']}: expected {row['length']} CAs, found {len(ca)}")
    frames, replicas, coordinates = [], [], []
    per_replica = universe.trajectory.n_frames // 3
    keep = np.flatnonzero(
        np.arange(per_replica) * config["analysis"]["frame_interval_ns"]
        > config["analysis"]["equilibration_ns"]
    )
    for replica in range(3):
        for frame in keep:
            universe.trajectory[replica * per_replica + frame]
            coordinates.append(ca.positions.copy())
            frames.append(int(frame))
            replicas.append(replica + 1)
    return np.asarray(coordinates, dtype=np.float64), np.asarray(replicas), np.asarray(frames)


def align_to_reference(coordinates: np.ndarray) -> np.ndarray:
    """Kabsch-align every frame to frame zero in one batched SVD."""
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    reference = centered[0]
    covariance = np.einsum("fai,aj->fij", centered, reference)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    reflected = np.linalg.det(rotation) < 0
    left[reflected, :, -1] *= -1
    rotation = left @ right_t
    return centered @ rotation


def analyse_system(row: dict[str, str], config: dict, force: bool = False) -> dict:
    system = row["system_id"]
    output_dir = HERE / "outputs" / "analysis" / "basins" / system
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "assignments.parquet"
    summary_path = output_dir / "summary.yaml"
    if assignments_path.exists() and summary_path.exists() and not force:
        import yaml

        return yaml.safe_load(summary_path.read_text())

    coordinates, replicas, frames = load_ca_coordinates(row, config)
    aligned = align_to_reference(coordinates)
    distances = pairwise_distances(aligned.reshape(len(aligned), -1), metric="euclidean")
    distances /= np.sqrt(aligned.shape[1])
    basin_config = config["analysis"]["basins"]
    labels = DBSCAN(
        eps=basin_config["dbscan_eps_angstrom"],
        min_samples=basin_config["dbscan_min_samples"],
        metric="precomputed",
        n_jobs=1,
    ).fit_predict(distances)
    basin_labels, basin_counts = np.unique(labels[labels >= 0], return_counts=True)
    reference_label = int(basin_labels[np.argmax(basin_counts)]) if len(basin_labels) else None
    medoid_index = None
    if reference_label is not None:
        members = np.flatnonzero(labels == reference_label)
        medoid_index = int(members[np.argmin(distances[np.ix_(members, members)].sum(axis=1))])

    table = pd.DataFrame(
        {
            "pooled_index": np.arange(len(labels)),
            "replica": replicas,
            "frame": frames,
            "basin": labels,
            "rmsd_to_medoid": np.nan if medoid_index is None else distances[:, medoid_index],
        }
    )
    table.to_parquet(assignments_path, index=False)
    counts = {
        int(label): {replica: int(np.sum((labels == label) & (replicas == replica))) for replica in (1, 2, 3)}
        for label in basin_labels
    }
    usable = [
        label
        for label, replica_counts in counts.items()
        if all(value >= basin_config["min_frames_per_replica"] for value in replica_counts.values())
    ]
    summary = {
        "system_id": system,
        "frames": len(labels),
        "ca_atoms": aligned.shape[1],
        "basins": len(basin_labels),
        "usable_basins": len(usable),
        "usable_labels": usable,
        "noise_frames": int(np.sum(labels == -1)),
        "reference_label": reference_label,
        "medoid_pooled_index": medoid_index,
        "medoid_replica": None if medoid_index is None else int(replicas[medoid_index]),
        "medoid_frame": None if medoid_index is None else int(frames[medoid_index]),
        "counts_by_replica": counts,
        "assignments": str(assignments_path.relative_to(HERE)),
        "alignment": "all frames Kabsch-aligned to pooled frame zero before Euclidean C-alpha RMSD",
    }
    atomic_yaml(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    workers = args.workers or config["analysis"]["basins"]["workers"]
    rows = load_systems()[: args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, args.force): row for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{count}/{len(rows)}] {result['system_id']}: {result['usable_basins']} usable basins")

    output = HERE / "outputs" / "analysis" / "basin_census.csv"
    fields = ["system_id", "frames", "ca_atoms", "basins", "usable_basins", "noise_frames", "reference_label"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["system_id"]):
            writer.writerow({key: result[key] for key in fields})
    print(f"Basin census complete: report={output}")


if __name__ == "__main__":
    main()
