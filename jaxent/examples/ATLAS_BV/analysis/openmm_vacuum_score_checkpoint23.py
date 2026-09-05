"""Audit and score protein-only ATLAS trajectories with OpenMM in vacuum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import yaml

from openmm_vacuum_common import canonical_atom_name, unwrap_positions


HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint23_openmm_vacuum"
ENERGY_DIR = OUTPUT / "energies"
TOTAL_ONLY_DIR = OUTPUT / "energies_total_only"
FORCE_GROUPS = {"bond": 0, "angle": 1, "torsion": 2, "nonbonded": 3, "other": 4}
FORCE_CLASSES = {
    "HarmonicBondForce": "bond",
    "HarmonicAngleForce": "angle",
    "PeriodicTorsionForce": "torsion",
    "CMAPTorsionForce": "torsion",
    "CustomTorsionForce": "torsion",
    "NonbondedForce": "nonbonded",
    "CustomNonbondedForce": "nonbonded",
    "CustomBondForce": "nonbonded",
}


def systems() -> list[dict[str, str]]:
    with (HERE / "data/systems.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_identity(atoms) -> list[tuple[int, str, str]]:
    return [
        (int(resid), str(resname), canonical_atom_name(name))
        for resid, resname, name in zip(atoms.resids, atoms.resnames, atoms.names)
    ]


def build_system(
    pdb_path: Path,
) -> tuple[app.PDBFile, openmm.System, dict[str, list[str]]]:
    pdb = app.PDBFile(str(pdb_path))
    forcefield = app.ForceField("charmm36.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    classes: dict[str, list[str]] = defaultdict(list)
    for force in system.getForces():
        class_name = force.__class__.__name__
        category = FORCE_CLASSES.get(class_name, "other")
        force.setForceGroup(FORCE_GROUPS[category])
        classes[category].append(class_name)
    return pdb, system, dict(classes)


def bond_edges(topology: app.Topology) -> tuple[list[list[int]], int]:
    atoms = list(topology.atoms())
    index = {atom: atom.index for atom in atoms}
    adjacency = [[] for _ in atoms]
    for left, right in topology.bonds():
        i, j = index[left], index[right]
        adjacency[i].append(j)
        adjacency[j].append(i)
    components = 0
    seen: set[int] = set()
    for root in range(len(atoms)):
        if root in seen:
            continue
        components += 1
        queue = [root]
        seen.add(root)
        while queue:
            for neighbour in adjacency[queue.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return adjacency, components


def max_bond_length_angstrom(
    positions: np.ndarray, adjacency: list[list[int]]
) -> float:
    maximum = 0.0
    for left, neighbours in enumerate(adjacency):
        for right in neighbours:
            if right > left:
                maximum = max(
                    maximum, float(np.linalg.norm(positions[left] - positions[right]))
                )
    return maximum


def audit_row(row: dict[str, str]) -> dict:
    system_id = row["system_id"]
    root = HERE / "data/raw" / system_id
    pdb_path = root / f"{system_id}.pdb"
    result = {"system_id": system_id, "status": "failed", "error": ""}
    try:
        pdb_universe = mda.Universe(str(pdb_path))
        pdb_identity = topology_identity(pdb_universe.atoms)
        for replica in (1, 2, 3):
            tpr = mda.Universe(str(root / f"{system_id}_R{replica}.tpr"))
            protein = tpr.select_atoms("protein")
            if topology_identity(protein) != pdb_identity:
                raise ValueError(
                    f"R{replica} TPR protein atom identity/order differs from PDB"
                )
            trajectory = mda.Universe(
                str(pdb_path), str(root / f"{system_id}_R{replica}.xtc")
            )
            if (
                len(trajectory.atoms) != len(pdb_identity)
                or len(trajectory.trajectory) != 1001
            ):
                raise ValueError(
                    f"R{replica} XTC shape differs from PDB/expected frames"
                )
        pdb, openmm_system, classes = build_system(pdb_path)
        adjacency, components = bond_edges(pdb.topology)
        if openmm_system.getNumParticles() != len(pdb_identity):
            raise ValueError("OpenMM changed the particle count")
        result.update(
            status="ok",
            atoms=len(pdb_identity),
            residues=pdb.topology.getNumResidues(),
            bonds=sum(map(len, adjacency)) // 2,
            connected_components=components,
            force_classes=json.dumps(classes, sort_keys=True),
            pdb_sha256=sha256(pdb_path),
        )
    except Exception as error:  # persisted audit must retain all failures
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def score_replica(
    system_id: str,
    replica: int,
    pdb_path: Path,
    openmm_system: openmm.System,
    topology: app.Topology,
    platform_name: str,
    device: str | None,
    threads: int | None,
    frame_limit: int | None,
    force: bool,
    total_only: bool,
    energy_dir: Path,
) -> dict:
    xtc_path = pdb_path.parent / f"{system_id}_R{replica}.xtc"
    universe = mda.Universe(str(pdb_path), str(xtc_path))
    adjacency, components = bond_edges(topology)
    output = energy_dir / system_id / f"{system_id}_R{replica}.energies.npz"
    count = min(len(universe.trajectory), frame_limit or len(universe.trajectory))
    if not force and valid_cached(output, count):
        return {
            "system_id": system_id,
            "replica": replica,
            "frames": count,
            "status": "cached",
            "platform": platform_name,
            "output": str(output.relative_to(HERE)),
            "sha256": sha256(output),
        }
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform_object = openmm.Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name in {"CUDA", "OpenCL"}:
        properties = {"Precision": "mixed"}
        if device is not None:
            properties["DeviceIndex"] = device
    elif platform_name == "CPU" and threads is not None:
        properties["Threads"] = str(threads)
    context = openmm.Context(openmm_system, integrator, platform_object, properties)
    energy_names = ("total",) if total_only else ("total", *FORCE_GROUPS)
    energies = {name: np.empty(count) for name in energy_names}
    maximum_bond = 0.0
    start = time.perf_counter()
    for frame, timestep in enumerate(universe.trajectory[:count]):
        box = np.asarray(timestep.triclinic_dimensions, dtype=float)
        positions = unwrap_positions(timestep.positions, box, adjacency)
        maximum_bond = max(maximum_bond, max_bond_length_angstrom(positions, adjacency))
        context.setPositions(positions * unit.angstrom)
        energies["total"][frame] = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if not total_only:
            for name, group in FORCE_GROUPS.items():
                energies[name][frame] = (
                    context.getState(getEnergy=True, groups=1 << group)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilojoule_per_mole)
                )
    elapsed = time.perf_counter() - start
    del context, integrator
    atomic_npz(
        output,
        frame=np.arange(count, dtype=np.int32),
        **{f"energy_{name}_kj_mol": value for name, value in energies.items()},
    )
    component_sum_error = (
        np.nan
        if total_only
        else np.max(
            np.abs(energies["total"] - sum(energies[name] for name in FORCE_GROUPS))
        )
    )
    return {
        "system_id": system_id,
        "replica": replica,
        "frames": count,
        "status": "scored",
        "seconds": elapsed,
        "frames_per_second": count / elapsed,
        "platform": platform_name,
        "connected_components": components,
        "max_bond_length_angstrom": maximum_bond,
        "max_component_sum_error_kj_mol": float(component_sum_error),
        "output": str(output.relative_to(HERE)),
        "sha256": sha256(output),
    }


def valid_cached(path: Path, expected_frames: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                len(archive["frame"]) == expected_frames
                and all(np.all(np.isfinite(archive[key])) for key in archive.files)
            )
    except Exception:
        return False


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "score"))
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--platform", choices=("CUDA", "OpenCL", "CPU", "Reference"), default="CUDA"
    )
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--total-only", action="store_true")
    args = parser.parse_args()
    selected = systems()
    if args.pilot:
        import pandas as pd  # only needed when choosing the persisted pilot

        pilot = set(
            pd.read_parquet(
                HERE
                / "outputs/analysis/pairwise_geometry/checkpoint19_thermodynamic_combination_pilot/pilot_systems.parquet"
            ).system_id
        )
        selected = [row for row in selected if row["system_id"] in pilot]
    if args.systems:
        requested = set(args.systems)
        selected = [row for row in selected if row["system_id"] in requested]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        rows = []
        for index, row in enumerate(selected, 1):
            result = audit_row(row)
            rows.append(result)
            print(
                f"[{index}/{len(selected)}] {row['system_id']}: {result['status']}",
                flush=True,
            )
        import pandas as pd

        pd.DataFrame(rows).to_parquet(OUTPUT / "topology_audit.parquet", index=False)
        write_yaml(
            OUTPUT / "topology_audit.yaml",
            {
                "systems": len(rows),
                "passed": sum(row["status"] == "ok" for row in rows),
                "failed": sum(row["status"] != "ok" for row in rows),
                "openmm_version": openmm.__version__,
            },
        )
        return
    score_rows = []
    energy_dir = TOTAL_ONLY_DIR if args.total_only else ENERGY_DIR
    for index, row in enumerate(selected, 1):
        system_id = row["system_id"]
        pdb_path = HERE / "data/raw" / system_id / f"{system_id}.pdb"
        pdb, openmm_system, classes = build_system(pdb_path)
        for replica in (1, 2, 3):
            score_rows.append(
                score_replica(
                    system_id,
                    replica,
                    pdb_path,
                    openmm_system,
                    pdb.topology,
                    args.platform,
                    args.device,
                    args.threads,
                    args.frame_limit,
                    args.force,
                    args.total_only,
                    energy_dir,
                )
            )
        write_yaml(
            energy_dir / system_id / "manifest.yaml",
            {
                "system_id": system_id,
                "force_field": "charmm36.xml",
                "nonbonded_method": "NoCutoff",
                "constraints": None,
                "temperature_k_for_analysis": 300.0,
                "openmm_version": openmm.__version__,
                "python": sys.version,
                "host_platform": platform.platform(),
                "force_classes": classes,
                "total_only": args.total_only,
                "replicas": [
                    item for item in score_rows if item["system_id"] == system_id
                ],
            },
        )
        print(f"[{index}/{len(selected)}] {system_id}: scored", flush=True)
    import pandas as pd

    manifest_name = (
        f"scoring_manifest.{selected[0]['system_id']}.parquet"
        if len(selected) == 1
        else "scoring_manifest.parquet"
    )
    pd.DataFrame(score_rows).to_parquet(OUTPUT / manifest_name, index=False)


if __name__ == "__main__":
    main()
