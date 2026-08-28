#!/usr/bin/env python
"""Resumable parallel Stage-1 featurisation for all selected ATLAS replicas."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from benchmark_featurisation import parse_timing


HERE = Path(__file__).resolve().parent


def validate_features(path: Path, expected_frames: int = 1001) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as features:
        heavy = features["heavy_contacts"]
        acceptor = features["acceptor_contacts"]
        valid = (
            heavy.shape == acceptor.shape
            and heavy.ndim == 2
            and heavy.shape[1] == expected_frames
            and np.isfinite(heavy).all()
            and np.isfinite(acceptor).all()
            and (heavy >= 0).all()
            and (acceptor >= 0).all()
        )
        return {
            "valid": bool(valid),
            "eligible_residues": int(heavy.shape[0]),
            "frames": int(heavy.shape[1]),
        }


def load_jobs() -> list[dict[str, object]]:
    jobs = []
    with (HERE / "data" / "systems.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            for replica, relative_trajectory in enumerate(row["replica_paths"].split(";"), 1):
                jobs.append(
                    {
                        "system_id": row["system_id"],
                        "length": int(row["length"]),
                        "replica": replica,
                        "topology": HERE / row["pdb_path"],
                        "trajectory": HERE / relative_trajectory,
                    }
                )
    return jobs


def run_job(job: dict[str, object], protocol: dict[str, object], force: bool) -> dict[str, object]:
    system = str(job["system_id"])
    replica = int(job["replica"])
    output_dir = HERE / "outputs" / "stage1" / system / f"R{replica}"
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.npz"
    if features_path.exists() and not force:
        validation = validate_features(features_path)
        if validation["valid"]:
            return {**job, "status": "skipped_valid", **validation}

    timing_path = output_dir / "timing.txt"
    stdout_path = output_dir / "featurise.log"
    executable = Path(sys.executable).parent / "jaxent-featurise"
    command = [
        "/usr/bin/time", "-v", "-o", str(timing_path), str(executable),
        "--top_path", str(job["topology"]),
        "--trajectory_path", str(job["trajectory"]),
        "--output_dir", str(output_dir),
        "--name", f"atlas_bv_{system}_R{replica}",
        "bv",
        "--temperature", str(protocol["temperature_k"]),
        "--bv_bc", str(protocol["bv_bc"]),
        "--bv_bh", str(protocol["bv_bh"]),
        "--heavy_radius", str(protocol["heavy_midpoint_angstrom"]),
        "--o_radius", str(protocol["acceptor_midpoint_angstrom"]),
        "--num_timepoints", "0",
        "--residue_ignore", *(str(value) for value in protocol["residue_ignore"]),
        "--contact_mode", str(protocol["contact_mode"]),
        "--switch_scale_nc", str(protocol["switch_scale_nc_angstrom"]),
        "--switch_scale_nh", str(protocol["switch_scale_nh_angstrom"]),
        "--mda_contact_environment", str(protocol["contact_environment"]),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(protocol["contact_threads"])
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    started = datetime.now(timezone.utc).isoformat()
    with stdout_path.open("w") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        return {
            **job,
            "status": "failed",
            "returncode": completed.returncode,
            "started_utc": started,
            "log": str(stdout_path.relative_to(HERE)),
        }
    validation = validate_features(features_path)
    return {
        **job,
        "status": "complete" if validation["valid"] else "invalid",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        **parse_timing(timing_path),
        **validation,
        "output_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
    }


def serialise_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: str(value.relative_to(HERE)) if isinstance(value, Path) else value
        for key, value in result.items()
    }


def write_report(path: Path, started: str, workers: int, results: list[dict[str, object]]) -> None:
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "started_utc": started,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "status_counts": counts,
        "results": [serialise_result(result) for result in results],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config = yaml.safe_load((HERE / "config.yaml").read_text())
    protocol = config["protocol"]
    jobs = load_jobs()
    for job in jobs:
        for key in ("topology", "trajectory"):
            if not Path(job[key]).is_file():
                raise FileNotFoundError(job[key])

    report_path = HERE / "outputs" / "stage1" / "batch_report.yaml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(
        f"Starting {len(jobs)} replicas with {args.workers} workers × "
        f"{protocol['contact_threads']} contact threads; existing valid outputs are resumed.",
        flush=True,
    )
    wall_start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job, protocol, args.force): job for job in jobs}
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            write_report(report_path, started, args.workers, results)
            if completed_count % 10 == 0 or result["status"] not in {"complete", "skipped_valid"}:
                elapsed = (time.monotonic() - wall_start) / 60
                print(
                    f"[{completed_count}/{len(jobs)}] {result['system_id']}_R{result['replica']} "
                    f"{result['status']} ({elapsed:.1f} min elapsed)",
                    flush=True,
                )

    failures = [result for result in results if result["status"] not in {"complete", "skipped_valid"}]
    print(f"Batch finished in {(time.monotonic() - wall_start) / 60:.1f} min; failures={len(failures)}")
    print(f"Report: {report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
