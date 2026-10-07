#!/usr/bin/env python
"""Run the ATLAS Stage-1 median-system featurisation benchmark only."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent


def parse_elapsed(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"unrecognised elapsed time: {value!r}")


def parse_timing(path: Path) -> dict[str, float]:
    text = path.read_text()

    def field(pattern: str) -> str:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"timing field not found: {pattern}")
        return match.group(1).strip()

    return {
        "elapsed_seconds": parse_elapsed(
            field(r"Elapsed \(wall clock\) time .*?:\s*(\S+)$")
        ),
        "user_seconds": float(field(r"User time \(seconds\):\s*(\S+)$")),
        "system_seconds": float(field(r"System time \(seconds\):\s*(\S+)$")),
        "maximum_rss_kib": float(
            field(r"Maximum resident set size \(kbytes\):\s*(\S+)$")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing benchmark")
    args = parser.parse_args()

    config_path = HERE / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    protocol = config["protocol"]
    benchmark = config["benchmark"]
    full = config["full_dataset"]

    system = benchmark["system_id"]
    replica = benchmark["replica"]
    raw_dir = HERE / "data" / "raw" / system
    top_path = raw_dir / f"{system}.pdb"
    trajectory_path = raw_dir / f"{system}_R{replica}.xtc"
    for required in (top_path, trajectory_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_dir = HERE / "outputs" / "benchmark" / system / f"R{replica}"
    features_path = output_dir / "features.npz"
    if features_path.exists() and not args.force:
        raise FileExistsError(f"benchmark exists at {output_dir}; use --force to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / "timing.txt"

    print(
        "PRE-RUN ESTIMATE: "
        f"{system}_R{replica} ~{benchmark['estimated_seconds']:.1f} s; "
        f"full 111-system serial projection ~{full['estimated_serial_seconds'] / 3600:.2f} h; "
        "this command runs one replica only.",
        flush=True,
    )

    executable = Path(sys.executable).parent / "jaxent-featurise"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(timing_path),
        str(executable),
        "--top_path",
        str(top_path),
        "--trajectory_path",
        str(trajectory_path),
        "--output_dir",
        str(output_dir),
        "--name",
        f"atlas_bv_{system}_R{replica}",
        "bv",
        "--temperature",
        str(protocol["temperature_k"]),
        "--bv_bc",
        str(protocol["bv_bc"]),
        "--bv_bh",
        str(protocol["bv_bh"]),
        "--heavy_radius",
        str(protocol["heavy_midpoint_angstrom"]),
        "--o_radius",
        str(protocol["acceptor_midpoint_angstrom"]),
        "--num_timepoints",
        "0",
        "--residue_ignore",
        *(str(value) for value in protocol["residue_ignore"]),
        "--contact_mode",
        protocol["contact_mode"],
        "--switch_scale_nc",
        str(protocol["switch_scale_nc_angstrom"]),
        "--switch_scale_nh",
        str(protocol["switch_scale_nh_angstrom"]),
        "--mda_contact_environment",
        protocol["contact_environment"],
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(protocol["contact_threads"])
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    timing = parse_timing(timing_path)
    with np.load(features_path, allow_pickle=False) as features:
        heavy = features["heavy_contacts"]
        acceptor = features["acceptor_contacts"]
        validation = {
            "heavy_shape": list(heavy.shape),
            "acceptor_shape": list(acceptor.shape),
            "all_finite": bool(np.isfinite(heavy).all() and np.isfinite(acceptor).all()),
            "all_nonnegative": bool((heavy >= 0).all() and (acceptor >= 0).all()),
        }
    if not validation["all_finite"] or not validation["all_nonnegative"]:
        raise ValueError(f"invalid benchmark features: {validation}")

    workload_ratio = (
        full["summed_residues"]
        * full["replicas_per_system"]
        * benchmark["frames"]
        / (benchmark["residues"] * benchmark["frames"])
    )
    projected_seconds = timing["elapsed_seconds"] * workload_ratio
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "median system, replica 1 only; full batch not run",
        "config": str(config_path.relative_to(HERE)),
        "inputs": {
            "system_id": system,
            "replica": replica,
            "topology": str(top_path.relative_to(HERE)),
            "trajectory": str(trajectory_path.relative_to(HERE)),
        },
        "protocol": protocol,
        "command": command,
        "timing": timing,
        "validation": validation,
        "output_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "projection": {
            "workload_ratio": workload_ratio,
            "full_serial_seconds": projected_seconds,
            "full_serial_hours": projected_seconds / 3600,
            "two_concurrent_jobs_ideal_hours": projected_seconds / 7200,
            "full_serial_hours_with_20_percent_margin": projected_seconds * 1.2 / 3600,
        },
    }
    report_path = output_dir / "benchmark_report.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    print(yaml.safe_dump({"timing": timing, "validation": validation, "projection": report["projection"]}, sort_keys=False))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

