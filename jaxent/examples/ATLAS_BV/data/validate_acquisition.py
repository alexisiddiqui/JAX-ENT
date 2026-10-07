#!/usr/bin/env python3
"""Validate ATLAS analysis-tier archives and extracted trajectory systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import MDAnalysis as mda
import numpy as np

EXPECTED_FRAMES = 1001
EXPECTED_START_PS = 0.0
EXPECTED_END_PS = 100_000.0
EXPECTED_DT_PS = 100.0


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in archive member {bad}")
        for member in archive.infolist():
            candidate = PurePosixPath(member.filename)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")


def _required_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty required file: {path}")
    return path


def _donor_counts(universe: mda.Universe) -> tuple[int, int, int]:
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        raise ValueError("PDB contains no atoms selected by MDAnalysis 'protein'")
    residues_by_chain: dict[str, list] = {}
    for residue in protein.residues:
        chain = str(
            getattr(residue, "chainID", "")
            or getattr(residue, "segid", "")
            or "_"
        )
        residues_by_chain.setdefault(chain, []).append(residue)
    eligible = []
    for residues in residues_by_chain.values():
        eligible.extend(residue for residue in residues[1:] if residue.resname != "PRO")
    missing_n = 0
    missing_h = 0
    for residue in eligible:
        names = list(residue.atoms.names)
        missing_n += names.count("N") != 1
        missing_h += sum(name in {"H", "HN"} for name in names) != 1
    return len(eligible), missing_n, missing_h


def validate_system(root: Path, system_id: str, expected_length: int | None = None) -> dict[str, object]:
    pdb = _required_file(root, f"{system_id}.pdb")
    _required_file(root, f"{system_id}_corresp.tsv")
    for suffix in ("RMSD.tsv", "gyrate.tsv", "RMSF.tsv", "Neq.tsv", "Bfactor.tsv", "pLDDT.tsv"):
        _required_file(root, f"{system_id}_{suffix}")
    universe = mda.Universe(str(pdb))
    protein = universe.select_atoms("protein")
    n_residues = len(protein.residues)
    if expected_length is not None and n_residues != expected_length:
        raise ValueError(
            f"Protein length mismatch for {system_id}: catalogue={expected_length}, PDB={n_residues}"
        )
    n_hydrogens = len(universe.select_atoms("type H"))
    if n_hydrogens == 0:
        raise ValueError(f"No hydrogens found in {pdb}")
    eligible, missing_n, missing_h = _donor_counts(universe)
    if eligible == 0 or missing_n or missing_h:
        raise ValueError(
            f"BV donor validation failed: eligible={eligible}, missing_N={missing_n}, missing_H/HN={missing_h}"
        )

    result: dict[str, object] = {
        "system_id": system_id,
        "status": "valid",
        "pdb_path": str(pdb),
        "n_atoms": len(universe.atoms),
        "n_residues": n_residues,
        "n_hydrogens": n_hydrogens,
        "n_bv_donors": eligible,
    }
    for replica in (1, 2, 3):
        xtc = _required_file(root, f"{system_id}_R{replica}.xtc")
        _required_file(root, f"{system_id}_R{replica}.tpr")
        trajectory = mda.Universe(str(pdb), str(xtc))
        if len(trajectory.atoms) != len(universe.atoms):
            raise ValueError(f"Atom-count mismatch in {xtc}")
        n_frames = trajectory.trajectory.n_frames
        first = trajectory.trajectory[0]
        start = float(first.time)
        dt = float(trajectory.trajectory.dt)
        end = float(trajectory.trajectory[-1].time)
        if n_frames != EXPECTED_FRAMES:
            raise ValueError(f"{xtc} has {n_frames} frames; expected {EXPECTED_FRAMES}")
        if not np.isclose(start, EXPECTED_START_PS, atol=1e-3):
            raise ValueError(f"{xtc} starts at {start} ps")
        if not np.isclose(end, EXPECTED_END_PS, atol=1e-2):
            raise ValueError(f"{xtc} ends at {end} ps")
        if not np.isclose(dt, EXPECTED_DT_PS, atol=1e-4):
            raise ValueError(f"{xtc} has dt={dt} ps")
        result[f"r{replica}_frames"] = n_frames
        result[f"r{replica}_start_ps"] = start
        result[f"r{replica}_end_ps"] = end
        result[f"r{replica}_dt_ps"] = dt
    result["n_frames"] = sum(int(result[f"r{replica}_frames"]) for replica in (1, 2, 3))
    return result


def atomic_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.remove(temporary_name)
        except FileNotFoundError:
            pass
        raise


def record_download(manifest: Path, row: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    if manifest.exists():
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows = [existing for existing in rows if existing.get("system_id") != row["system_id"]]
    rows.append(row)
    rows.sort(key=lambda item: str(item["system_id"]))
    atomic_csv(rows, manifest)


def audit(systems_csv: Path, raw_root: Path, downloads_csv: Path, report: Path) -> int:
    with systems_csv.open(newline="") as handle:
        systems = list(csv.DictReader(handle))
    downloads: dict[str, dict[str, str]] = {}
    if downloads_csv.exists():
        with downloads_csv.open(newline="") as handle:
            downloads = {row["system_id"]: row for row in csv.DictReader(handle)}
    report_rows: list[dict[str, object]] = []
    failures = 0
    total_frames = 0
    for system in systems:
        system_id = system["system_id"]
        base: dict[str, object] = dict(downloads.get(system_id, {}))
        try:
            validated = validate_system(raw_root / system_id, system_id, int(system["length"]))
            validated["pdb_path"] = system["pdb_path"]
            base.update(validated)
            total_frames += int(validated["n_frames"])
        except Exception as exc:  # keep a complete batch report
            failures += 1
            base.update({"system_id": system_id, "status": "invalid", "error": str(exc)})
        report_rows.append(base)
    atomic_csv(report_rows, report)
    expected_frames = len(systems) * 3 * EXPECTED_FRAMES
    if failures or total_frames != expected_frames:
        print(
            f"Audit failed: valid={len(systems) - failures}/{len(systems)}, "
            f"frames={total_frames}/{expected_frames}",
            file=sys.stderr,
        )
        return 1
    print(f"Audit passed: {len(systems)} systems, {total_frames} frames -> {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("path", type=Path)
    system_parser = subparsers.add_parser("system")
    system_parser.add_argument("--root", type=Path, required=True)
    system_parser.add_argument("--system-id", required=True)
    system_parser.add_argument("--expected-length", type=int)
    record_parser = subparsers.add_parser("record-download")
    record_parser.add_argument("--manifest", type=Path, required=True)
    record_parser.add_argument("--system-id", required=True)
    record_parser.add_argument("--url", required=True)
    record_parser.add_argument("--content-length", type=int, required=True)
    record_parser.add_argument("--archive", type=Path, required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--systems-csv", type=Path, required=True)
    audit_parser.add_argument("--raw-root", type=Path, required=True)
    audit_parser.add_argument("--downloads-csv", type=Path, required=True)
    audit_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "archive":
        check_archive(args.path)
        print(f"Archive valid: {args.path}")
        return 0
    if args.command == "system":
        print(json.dumps(validate_system(args.root, args.system_id, args.expected_length), sort_keys=True))
        return 0
    if args.command == "record-download":
        record_download(
            args.manifest,
            {
                "system_id": args.system_id,
                "url": args.url,
                "remote_content_length": args.content_length,
                "archive_sha256": hash_file(args.archive),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0
    return audit(args.systems_csv, args.raw_root, args.downloads_csv, args.report)


if __name__ == "__main__":
    sys.exit(main())
