"""Audit and rescore ATLAS trajectories with PyRosetta ref2015 score functions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from openmm_vacuum_common import unwrap_positions
from pyrosetta_energy_common import build_atom_mapping


HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint24_pyrosetta_energy"
ENERGY_DIR = OUTPUT / "energies"
DEFAULT_PYROSETTA_SITE = Path("/home/alexi/anaconda3/lib/python3.11/site-packages")
PILOT_SYSTEMS = {
    "2ad6_D", "1pch_A", "6fub_B", "1u6t_A", "7bwf_B", "6yhu_B",
    "1dvo_A", "2in8_A", "5bnh_A", "4qmd_A", "1k7j_A", "1ah7_A",
}
SCOREFUNCTIONS = ("ref2015", "ref2015_cart")


def load_pyrosetta(site: Path):
    if str(site) not in sys.path:
        sys.path.append(str(site))
    import pyrosetta

    pyrosetta.init("-mute all -ignore_unrecognized_res true")
    return pyrosetta


def systems() -> list[dict[str, str]]:
    with (HERE / "data/systems.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def pose_atom_records(pose) -> list[dict]:
    records = []
    for resindex in range(1, pose.total_residue() + 1):
        residue = pose.residue(resindex)
        for atomno in range(1, residue.natoms() + 1):
            atom_type = residue.atom_type(atomno)
            xyz = residue.xyz(atomno)
            records.append(
                {
                    "resindex": resindex,
                    "atomno": atomno,
                    "name": residue.atom_name(atomno).strip(),
                    "element": str(atom_type.element()).strip(),
                    "virtual": bool(atom_type.is_virtual()),
                    "position": [xyz.x, xyz.y, xyz.z],
                }
            )
    return records


def build_mapping(pdb_path: Path, pose) -> tuple[mda.Universe, np.ndarray, dict]:
    # Explicit formats avoid MDAnalysis' ParmEd format probe.  ParmEd imports
    # OpenMM plugins lazily, which is unsafe after Rosetta's shared libraries
    # have been loaded in this mixed environment.
    universe = mda.Universe(str(pdb_path), topology_format="PDB")
    atoms = universe.atoms
    mapping, metadata = build_atom_mapping(
        list(atoms.names),
        list(atoms.resnames),
        np.asarray(atoms.resindices, dtype=int) + 1,
        list(atoms.elements),
        np.asarray(atoms.positions, dtype=float),
        pose_atom_records(pose),
    )
    virtual_atoms = sum(atom["virtual"] for atom in pose_atom_records(pose))
    metadata.update(
        residues=len(universe.residues),
        pose_residues=pose.total_residue(),
        pose_atoms=pose.total_atoms(),
        pose_virtual_atoms=virtual_atoms,
    )
    if len(universe.residues) != pose.total_residue():
        raise ValueError("PDB and Pose residue counts differ")
    return universe, mapping, metadata


def guessed_adjacency(universe: mda.Universe) -> list[list[int]]:
    universe.atoms.guess_bonds()
    adjacency = [[] for _ in universe.atoms]
    for left, right in universe.bonds.indices:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return adjacency


def score_terms(pyrosetta, pose, scorefxn) -> dict[str, float]:
    total = float(scorefxn(pose))
    weights = scorefxn.weights()
    energies = pose.energies().total_energies()
    values = {"total": total}
    for score_type in scorefxn.get_nonzero_weighted_scoretypes():
        name = pyrosetta.rosetta.core.scoring.name_from_score_type(score_type)
        values[name] = float(weights[score_type] * energies[score_type])
    return values


def set_pose_coordinates(pyrosetta, pose, mapping: np.ndarray, xyz: np.ndarray) -> None:
    atom_id = pyrosetta.rosetta.core.id.AtomID
    vector = pyrosetta.rosetta.numeric.xyzVector_double_t
    for pdb_index, (resindex, atomno) in enumerate(mapping):
        if resindex == 0:
            continue
        position = xyz[pdb_index]
        pose.set_xyz(
            atom_id(int(atomno), int(resindex)),
            vector(float(position[0]), float(position[1]), float(position[2])),
        )
    pose.update_residue_neighbors()


def audit_system(pyrosetta, row: dict[str, str]) -> dict:
    system = row["system_id"]
    root = HERE / "data/raw" / system
    pdb_path = root / f"{system}.pdb"
    result = {"system_id": system, "status": "failed", "error": ""}
    try:
        pose = pyrosetta.pose_from_pdb(str(pdb_path))
        universe, mapping, metadata = build_mapping(pdb_path, pose)
        score_values = {}
        for name in SCOREFUNCTIONS:
            score_values[name] = score_terms(
                pyrosetta, pose, pyrosetta.create_score_function(name)
            )["total"]
        for replica in (1, 2, 3):
            trajectory = mda.Universe(
                str(pdb_path),
                str(root / f"{system}_R{replica}.xtc"),
                topology_format="PDB",
                format="XTC",
            )
            if len(trajectory.atoms) != len(mapping):
                raise ValueError(f"R{replica} trajectory atom count differs from PDB")
            if len(trajectory.trajectory) != 1001:
                raise ValueError(f"R{replica} has {len(trajectory.trajectory)} frames")
        result.update(
            status="ok",
            pdb_sha256=sha256(pdb_path),
            initial_scores=score_values,
            **metadata,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


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


def score_replica(
    pyrosetta,
    system: str,
    replica: int,
    pdb_path: Path,
    frame_limit: int | None,
    force: bool,
) -> dict:
    xtc_path = pdb_path.parent / f"{system}_R{replica}.xtc"
    trajectory = mda.Universe(
        str(pdb_path), str(xtc_path), topology_format="PDB", format="XTC"
    )
    count = min(len(trajectory.trajectory), frame_limit or len(trajectory.trajectory))
    output = ENERGY_DIR / system / f"{system}_R{replica}.energies.npz"
    if not force and valid_cached(output, count):
        return {
            "system_id": system, "replica": replica, "frames": count,
            "status": "cached", "output": str(output.relative_to(HERE)),
            "sha256": sha256(output),
        }
    pose = pyrosetta.pose_from_pdb(str(pdb_path))
    pdb_universe, mapping, mapping_metadata = build_mapping(pdb_path, pose)
    adjacency = guessed_adjacency(pdb_universe)
    scorefxns = {
        name: pyrosetta.create_score_function(name) for name in SCOREFUNCTIONS
    }
    term_names = {
        name: [
            pyrosetta.rosetta.core.scoring.name_from_score_type(score_type)
            for score_type in scorefxn.get_nonzero_weighted_scoretypes()
        ]
        for name, scorefxn in scorefxns.items()
    }
    arrays = {
        f"{name}__{term}": np.empty(count, dtype=np.float64)
        for name, terms in term_names.items()
        for term in ("total", *terms)
    }
    maximum_bond = 0.0
    start = time.perf_counter()
    for frame, timestep in enumerate(trajectory.trajectory[:count]):
        box = np.asarray(timestep.triclinic_dimensions, dtype=float)
        xyz = unwrap_positions(timestep.positions, box, adjacency)
        for left, neighbours in enumerate(adjacency):
            for right in neighbours:
                if right > left:
                    maximum_bond = max(
                        maximum_bond, float(np.linalg.norm(xyz[left] - xyz[right]))
                    )
        set_pose_coordinates(pyrosetta, pose, mapping, xyz)
        for name, scorefxn in scorefxns.items():
            values = score_terms(pyrosetta, pose, scorefxn)
            for term, value in values.items():
                arrays[f"{name}__{term}"][frame] = value
    elapsed = time.perf_counter() - start
    atomic_npz(output, frame=np.arange(count, dtype=np.int32), **arrays)
    return {
        "system_id": system,
        "replica": replica,
        "frames": count,
        "status": "scored",
        "seconds": elapsed,
        "frames_per_second": count / elapsed,
        "max_bond_length_angstrom": maximum_bond,
        **mapping_metadata,
        "output": str(output.relative_to(HERE)),
        "sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "score"))
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pyrosetta-site", type=Path, default=DEFAULT_PYROSETTA_SITE)
    args = parser.parse_args()
    pyrosetta = load_pyrosetta(args.pyrosetta_site)
    selected = systems()
    if args.pilot:
        selected = [row for row in selected if row["system_id"] in PILOT_SYSTEMS]
    if args.systems:
        requested = set(args.systems)
        selected = [row for row in selected if row["system_id"] in requested]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        results = []
        for index, row in enumerate(selected, 1):
            result = audit_system(pyrosetta, row)
            results.append(result)
            print(f"[{index}/{len(selected)}] {row['system_id']}: {result['status']}", flush=True)
        atomic_json(OUTPUT / "topology_audit.json", results)
        atomic_json(
            OUTPUT / "topology_audit_summary.json",
            {
                "systems": len(results),
                "passed": sum(row["status"] == "ok" for row in results),
                "failed": sum(row["status"] != "ok" for row in results),
                "pyrosetta_version": pyrosetta._version_string(),
            },
        )
        return
    manifests = []
    for index, row in enumerate(selected, 1):
        system = row["system_id"]
        pdb_path = HERE / "data/raw" / system / f"{system}.pdb"
        replicas = [
            score_replica(
                pyrosetta, system, replica, pdb_path, args.frame_limit, args.force
            )
            for replica in (1, 2, 3)
        ]
        manifest = {
            "system_id": system,
            "scorefunctions": list(SCOREFUNCTIONS),
            "relaxation": "none",
            "pyrosetta_version": pyrosetta._version_string(),
            "python": sys.version,
            "host_platform": platform.platform(),
            "replicas": replicas,
        }
        atomic_json(ENERGY_DIR / system / "manifest.json", manifest)
        manifests.extend(replicas)
        print(f"[{index}/{len(selected)}] {system}: scored", flush=True)
    manifest_name = (
        f"scoring_manifest.{selected[0]['system_id']}.json"
        if len(selected) == 1
        else "scoring_manifest.json"
    )
    atomic_json(OUTPUT / manifest_name, manifests)


if __name__ == "__main__":
    main()
