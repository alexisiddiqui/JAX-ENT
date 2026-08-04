#!/usr/bin/env python3
"""Generate ISO synthetic HDX targets with explicit frame-averaging semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_fast,
    residue_uptake_legacy,
    residue_uptake_slow2,
    weights_from_cluster_populations,
)

TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0])
HEADER = "#\t0.167\t1.0\t10.0\t60.0\t120.0\t times/min\n"
LABELS = {"open": 0, "closed": 1, "intermediate": -1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_population(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in text.split(","):
        try:
            name, value = item.split("=", 1)
            values[name.strip().lower()] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("population must use name=value pairs") from exc
    unknown = set(values) - set(LABELS)
    if unknown or any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
        raise argparse.ArgumentTypeError(f"invalid populations (unknown={sorted(unknown)})")
    return {name: value / sum(values.values()) for name, value in values.items()}


def peptide_aggregate(
    residue_uptake: np.ndarray, residue_segments: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Bradshaw's non-overlapping contributing-residue segment rule."""
    residue_uptake = np.asarray(residue_uptake)
    residue_segments = np.asarray(residue_segments, dtype=int)
    if residue_uptake.shape[1] != len(residue_segments):
        raise ValueError("one segment row is required per residue-uptake column")
    if width < 2:
        raise ValueError("peptide width must be at least 2")
    starts = list(range(1, 311, width - 1))
    ends = list(range(width, 311, width - 1))
    starts.pop(-1)
    ends.pop(-1)
    ends.append(310)
    values = []
    segments = []
    residue_ids = residue_segments[:, 1]
    for start, end in zip(starts, ends, strict=True):
        mask = (residue_ids >= start + 1) & (residue_ids <= end)
        if not np.any(mask):
            raise ValueError(f"segment {start}-{end} contains no represented residues")
        values.append(residue_uptake[:, mask].mean(axis=1))
        segments.append((start, end))
    return np.stack(values, axis=1), np.asarray(segments, dtype=int)


def align_residue_layout(
    uptake: np.ndarray, topology_path: Path, shipped_segments_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the 293 feature rows against the 294-row experimental layout."""
    topology = json.loads(topology_path.read_text())
    topologies = topology["topologies"]
    topology_residues = np.asarray([item["residues"][0] for item in topologies], dtype=int)
    segments = np.loadtxt(shipped_segments_path, dtype=int)
    if uptake.shape[1] != len(topologies):
        raise ValueError(
            f"feature uptake has {uptake.shape[1]} residues but topology has {len(topologies)}"
        )
    # Feature residues match the first 293 segment endpoints exactly.  The
    # sole unmatched segment is therefore terminal at the end of the layout.
    if len(segments) != len(topologies) + 1 or not np.array_equal(
        topology_residues, segments[:-1, 1]
    ):
        raise ValueError("feature topology cannot be aligned exactly to the shipped residue layout")
    terminal = np.ones((uptake.shape[0], 1), dtype=uptake.dtype)
    return np.concatenate((uptake, terminal), axis=1), segments


def write_target(output_dir: Path, uptake: np.ndarray, segments: np.ndarray) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dfrac_path = output_dir / "target_dfrac.dat"
    segs_path = output_dir / "target_segs.txt"
    with dfrac_path.open("w") as handle:
        handle.write(HEADER)
        np.savetxt(handle, uptake.T, delimiter="\t", fmt="%.5f")
    np.savetxt(segs_path, segments, fmt="%d", delimiter=" ")
    return dfrac_path, segs_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", choices=["iso_tri", "iso_bi"], required=True)
    parser.add_argument("--semantics", choices=["legacy", "fast", "slow2"], required=True)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--population", type=parse_population, required=True)
    parser.add_argument("--peptide-width", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.tau < 0 or args.peptide_width == 1 or args.peptide_width < 0:
        parser.error("tau and peptide width must be non-negative; width 1 is invalid")

    example_root = Path(__file__).resolve().parents[1]
    fitting_root = example_root / "fitting/jaxENT"
    suffix = args.ensemble.removeprefix("iso_")
    feature_path = fitting_root / f"_featurise/features_{args.ensemble}.npz"
    topology_path = fitting_root / f"_featurise/topology_{args.ensemble}.json"
    assignment_path = example_root / f"data/_clustering_results/cluster_assignments_ISO_{suffix.upper()}.csv"
    shipped_segments_path = example_root / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"

    with np.load(feature_path) as features:
        log_pf = 0.35 * features["heavy_contacts"] + 2.0 * features["acceptor_contacts"]
        k_ints = np.asarray(features["k_ints"])
    assignments = pd.read_csv(assignment_path)["cluster_assignment"].to_numpy(dtype=int)
    populations = {LABELS[name]: value for name, value in args.population.items()}
    for label in np.unique(assignments):
        populations.setdefault(int(label), 0.0)
    weights = weights_from_cluster_populations(assignments, populations)
    function = {
        "legacy": residue_uptake_legacy,
        "fast": residue_uptake_fast,
        "slow2": residue_uptake_slow2,
    }[args.semantics]
    positional = (log_pf, k_ints, TIMEPOINTS, weights)
    uptake = function(*positional, assignments, args.tau) if args.semantics == "slow2" else function(*positional, args.tau)
    uptake, segments = align_residue_layout(uptake, topology_path, shipped_segments_path)
    layout = "residue"
    if args.peptide_width:
        uptake, segments = peptide_aggregate(uptake, segments, args.peptide_width)
        layout = f"peptide_width_{args.peptide_width}"
    dfrac_path, segs_path = write_target(args.output_dir, uptake, segments)
    manifest = {
        "ensemble": args.ensemble,
        "semantics": args.semantics,
        "tau": args.tau,
        "populations": args.population,
        "layout": layout,
        "bc": 0.35,
        "bh": 2.0,
        "timepoints_min": TIMEPOINTS.tolist(),
        "outputs": {"dfrac": dfrac_path.name, "segments": segs_path.name},
        "input_hashes": {
            str(path.relative_to(example_root)): sha256(path)
            for path in (feature_path, topology_path, assignment_path, shipped_segments_path)
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
