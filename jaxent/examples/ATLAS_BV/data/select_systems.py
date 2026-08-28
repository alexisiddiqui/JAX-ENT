#!/usr/bin/env python3
"""Fetch the ATLAS catalogue and create the pinned ATLAS_BV system manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

CATALOG_URL = "https://www.dsimb.inserm.fr/ATLAS/api/parsable?dataset=ATLAS"
PARAMETERS_URL = "https://www.dsimb.inserm.fr/ATLAS/api/MD_parameters"
EXPECTED_SOURCE_COUNT = 1938
EXPECTED_SELECTED_COUNT = 111
REQUIRED_COLUMNS = {
    "PDB",
    "length",
    "PDB_resolution",
    "contact_ligand",
    "contact_nucleotide",
    "avg_RMSF",
    "avg_gyration",
    "CATH_class",
    "non_redundant_protein",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "JAX-ENT-ATLAS-BV/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
    temporary.replace(destination)


def safe_catalog_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    candidates: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe catalogue archive member: {member.filename}")
        if member.filename.endswith("_ATLAS_info.tsv"):
            candidates.append(member)
    if len(candidates) != 1:
        raise ValueError(f"Expected one *_ATLAS_info.tsv member, found {len(candidates)}")
    return candidates[0]


def parse_bool(value: str, *, field: str, system_id: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"true", "t", "yes", "y", "1"}:
        return True
    if normalised in {"false", "f", "no", "n", "0", "", "none", "nan", "na"}:
        return False
    raise ValueError(f"Unknown boolean {value!r} for {field} in {system_id}")


def parse_float(value: str, *, field: str, system_id: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}={value!r} in {system_id}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {field}={value!r} in {system_id}")
    return result


def normalise_cath_class(value: str) -> str:
    classes: list[str] = []
    for item in value.split(","):
        item = item.strip().strip('"').strip()
        if item in {"", "-"}:
            continue
        if item not in classes:
            classes.append(item)
    return "|".join(classes) if classes else "unclassified"


def select_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], tuple[float, float]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        system_id = row["PDB"].strip()
        if not re.fullmatch(r"[A-Za-z0-9]{4}_.+", system_id):
            raise ValueError(f"Invalid ATLAS system identifier: {system_id!r}")
        if system_id in seen:
            raise ValueError(f"Duplicate ATLAS system identifier: {system_id}")
        seen.add(system_id)
        length = int(row["length"])
        non_redundant = parse_bool(
            row["non_redundant_protein"], field="non_redundant_protein", system_id=system_id
        )
        ligand = parse_bool(row["contact_ligand"], field="contact_ligand", system_id=system_id)
        nucleotide = parse_bool(
            row["contact_nucleotide"], field="contact_nucleotide", system_id=system_id
        )
        if not (60 <= length <= 250 and non_redundant and not ligand and not nucleotide):
            continue
        rmsf = parse_float(row["avg_RMSF"], field="avg_RMSF", system_id=system_id)
        gyration = parse_float(row["avg_gyration"], field="avg_gyration", system_id=system_id)
        resolution_raw = row["PDB_resolution"].strip()
        resolution = ""
        if resolution_raw.lower() not in {"", "none", "nan", "na"}:
            resolution = parse_float(
                resolution_raw, field="PDB_resolution", system_id=system_id
            )
        replicas = ";".join(
            f"data/raw/{system_id}/{system_id}_R{replica}.xtc" for replica in (1, 2, 3)
        )
        selected.append(
            {
                "system_id": system_id,
                "length": length,
                "cath_class": normalise_cath_class(row["CATH_class"]),
                "avg_RMSF": rmsf,
                "avg_gyration": gyration,
                "resolution": resolution,
                "rmsf_tercile": "",
                "pdb_path": f"data/raw/{system_id}/{system_id}.pdb",
                "replica_paths": replicas,
                "n_frames": 3003,
            }
        )
    values = np.asarray([float(row["avg_RMSF"]) for row in selected], dtype=float)
    if values.size == 0:
        raise ValueError("Selection produced no systems")
    lower, upper = (float(value) for value in np.quantile(values, [1 / 3, 2 / 3]))
    for row in selected:
        value = float(row["avg_RMSF"])
        row["rmsf_tercile"] = "low" if value <= lower else "middle" if value <= upper else "high"
    selected.sort(key=lambda row: str(row["system_id"]))
    return selected, (lower, upper)


def atomic_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def acquire_parameters(url: str, archive_path: Path, provenance_path: Path, refresh: bool) -> None:
    if refresh or not archive_path.exists():
        download(url, archive_path)
    extraction_root = archive_path.parent / "extracted"
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe simulation-parameter member: {member.filename}")
        candidates = [member for member in archive.infolist() if member.filename.endswith("3_production.mdp")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one 3_production.mdp, found {len(candidates)}")
        member = candidates[0]
        mdp_path = extraction_root / "3_production.mdp"
        with archive.open(member) as source:
            mdp_path.write_bytes(source.read())
    settings: dict[str, str] = {}
    for raw_line in mdp_path.read_text().splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        settings[key] = value
    expected = {
        "integrator": "md",
        "dt": "0.002",
        "nsteps": "50000000",
        "nstxout_compressed": "5000",
        "tcoupl": "Nose-Hoover",
        "pcoupl": "Parrinello-Rahman",
    }
    for key, value in expected.items():
        if settings.get(key, "").lower() != value.lower():
            raise ValueError(f"Unexpected {key}={settings.get(key)!r}; expected {value!r}")
    temperatures = [float(item) for item in settings.get("ref_t", "").split()]
    if not temperatures or any(not np.isclose(item, 300.0) for item in temperatures):
        raise ValueError(f"Unexpected ref_t={settings.get('ref_t')!r}")
    pressures = [float(item) for item in settings.get("ref_p", "").split()]
    if not pressures or any(not np.isclose(item, 1.0) for item in pressures):
        raise ValueError(f"Unexpected ref_p={settings.get('ref_p')!r}")
    provenance = (
        "manifest_version: 1\n"
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"source_url: {url}\n"
        f"archive_sha256: {sha256(archive_path)}\n"
        f"production_mdp_member: {member.filename}\n"
        f"production_mdp_sha256: {sha256(mdp_path)}\n"
        f"integrator: {settings['integrator']}\n"
        f"dt_ps: {settings['dt']}\n"
        f"nsteps: {settings['nsteps']}\n"
        f"nstxout_compressed: {settings['nstxout_compressed']}\n"
        f"temperature_k: {temperatures[0]:g}\n"
        f"temperature_coupling: {settings['tcoupl']}\n"
        f"pressure_bar: {pressures[0]:g}\n"
        f"pressure_coupling: {settings['pcoupl']}\n"
        "ensemble: NPT\n"
    )
    atomic_text(provenance, provenance_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-url", default=CATALOG_URL)
    parser.add_argument("--catalog-archive", type=Path, default=Path(__file__).parent / "catalog/parsable.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "systems.csv")
    parser.add_argument(
        "--provenance", type=Path, default=Path(__file__).parent / "selection_provenance.yaml"
    )
    parser.add_argument("--expected-source-count", type=int, default=EXPECTED_SOURCE_COUNT)
    parser.add_argument("--expected-selected-count", type=int, default=EXPECTED_SELECTED_COUNT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--parameters-url", default=PARAMETERS_URL)
    parser.add_argument(
        "--parameters-archive",
        type=Path,
        default=Path(__file__).parent / "metadata/MD_parameters.zip",
    )
    parser.add_argument(
        "--parameters-provenance",
        type=Path,
        default=Path(__file__).parent / "md_parameters_provenance.yaml",
    )
    parser.add_argument("--skip-parameters", action="store_true")
    args = parser.parse_args(argv)

    if args.refresh or not args.catalog_archive.exists():
        download(args.catalog_url, args.catalog_archive)
    archive_hash = sha256(args.catalog_archive)
    with zipfile.ZipFile(args.catalog_archive) as archive:
        member = safe_catalog_member(archive)
        with archive.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig")
            reader = csv.DictReader(text, delimiter="\t")
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"ATLAS catalogue is missing columns: {sorted(missing)}")
            rows = list(reader)

    if len(rows) != args.expected_source_count:
        raise ValueError(
            f"Catalogue drift: expected {args.expected_source_count} rows, found {len(rows)}"
        )
    selected, terciles = select_rows(rows)
    if len(selected) != args.expected_selected_count:
        raise ValueError(
            f"Selection drift: expected {args.expected_selected_count} systems, found {len(selected)}"
        )
    atomic_csv(selected, args.output)
    provenance = (
        "manifest_version: 1\n"
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"catalog_url: {args.catalog_url}\n"
        f"catalog_member: {member.filename}\n"
        f"catalog_sha256: {archive_hash}\n"
        f"source_rows: {len(rows)}\n"
        f"selected_rows: {len(selected)}\n"
        "filters:\n"
        "  minimum_length: 60\n"
        "  maximum_length: 250\n"
        "  non_redundant_protein: true\n"
        "  contact_ligand: false\n"
        "  contact_nucleotide: false\n"
        "  contact_ion: allowed\n"
        f"rmsf_tercile_lower: {terciles[0]:.12g}\n"
        f"rmsf_tercile_upper: {terciles[1]:.12g}\n"
    )
    atomic_text(provenance, args.provenance)
    if not args.skip_parameters:
        acquire_parameters(
            args.parameters_url,
            args.parameters_archive,
            args.parameters_provenance,
            args.refresh,
        )
    print(f"Selected {len(selected)} of {len(rows)} ATLAS systems -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
