"""Checkpoint 36: corrected-contact direct and local-variance recovery controls.

This is deliberately independent of the OMC/Laplacian analyses.  It rebuilds BV
features into an isolated cache, fits on replica A, tunes local statistics on B,
and evaluates once on C.  The default single-system run is a mandatory review
checkpoint; the 24-system cohort requires a separate ``--scope pilot`` command.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-cp36-matplotlib")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.alpha_variance_checkpoint25 import (
    OPENMM_TOTAL_ONLY_DIR,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 import (
    ENERGY_DIR as PYROSETTA_ENERGY_DIR,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    load_config,
    load_systems,
    post_equilibration_indices,
    replica_paths,
)
from jaxent.examples.ATLAS_BV.analysis.graph_representation_audit import rmsd_matrix
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    density_targets,
    frame_w1_signatures,
    mass_metrics,
    w1_matrices,
)
from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import (
    direct_distance,
    local_statistics,
    nearest,
    pair_local_features,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 import (
    finite_spearman,
    fitted_scale,
    global_to_local,
)
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import (
    _pair_sets_from_audit,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    entropy_contributions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)
from jaxent.src.interfaces.topology.mda_adapter import mda_TopologyAdapter
from jaxent.src.models.func.uptake import calculate_HDXrate


OUTPUT = (
    HERE / "outputs/analysis/pairwise_geometry/checkpoint36_corrected_variance_recovery"
)
GRAPH_REFERENCE = (
    HERE / "outputs/analysis/pairwise_geometry/graph_representation_audit_bradshaw_0p1"
)
PILOT_TABLE = (
    HERE
    / "outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph/pilot_systems.parquet"
)
HISTORICAL_FITS = (
    HERE
    / "outputs/analysis/pairwise_geometry/checkpoint28_local_variance/pilot/local_variance_fits.parquet"
)
FIT, TUNE, TEST = 1, 2, 3
K_VALUES = (5, 10, 20, 50)
SHRINKAGES = (0.001, 0.01, 0.1)
SEED = 20260914
SINGLE_SYSTEM = "2w86_A"

PROTOCOL = {
    "temperature_k": 300.0,
    "pd": 7.0,
    "bv_bc": 0.35,
    "bv_bh": 2.0,
    "heavy_midpoint_angstrom": 6.5,
    "acceptor_midpoint_angstrom": 2.4,
    "contact_mode": "bradshaw_switch",
    "switch_function": "rational_6_12",
    "switch_scale_nc_angstrom": 0.1,
    "switch_scale_nh_angstrom": 0.1,
    "residue_ignore": [-2, 2],
    "contact_environment": "all",
    "kint_unit": "min^-1",
}

CLEAN_WORK = ("work_scale", "work_shape", "work_density", "work_opt")
DERIVED_WORK = ("work_fitting", "work_magnitude")
PF_METRICS = ("pf_l1", "pf_l2")
COORDINATES = ("rmsd", "w1", "rg")
ENERGY_METRICS = ("pyro_ref2015", "openmm_total")
ENERGY_SOURCES = {
    "pyro_ref2015": (PYROSETTA_ENERGY_DIR, "ref2015__total", "REU"),
    "openmm_total": (OPENMM_TOTAL_ONLY_DIR, "energy_total_kj_mol", "kJ/mol"),
}
DIRECT_METRICS = (
    *CLEAN_WORK,
    *DERIVED_WORK,
    *PF_METRICS,
    *COORDINATES,
    *ENERGY_METRICS,
)
VARIANCE_METRICS = (*CLEAN_WORK, *PF_METRICS, *ENERGY_METRICS)


def energy_representations(data: dict, config: dict) -> tuple[dict, dict]:
    """Load cached total energies, requiring exact replica/frame alignment."""
    representations, provenance = {}, {}
    system = data["system"]
    for metric, (root, key, unit) in ENERGY_SOURCES.items():
        values = np.empty(len(data["frames"]), dtype=float)
        sources = []
        for replica in (1, 2, 3):
            path = root / system / f"{system}_R{replica}.energies.npz"
            mask = data["replicas"] == replica
            with np.load(path, allow_pickle=False) as archive:
                frames = archive["frame"]
                keep = (
                    frames * config["analysis"]["frame_interval_ns"]
                    > config["analysis"]["equilibration_ns"]
                )
                if not np.array_equal(frames[keep], data["frames"][mask]):
                    raise ValueError(f"Energy frame alignment mismatch: {path}")
                energy = np.asarray(archive[key], dtype=float)[keep]
                if energy.shape != values[mask].shape or not np.isfinite(energy).all():
                    raise ValueError(f"Invalid energy values: {path}")
                values[mask] = energy
            sources.append({"path": str(path), "sha256": digest(path)})
        representations[metric] = (values, "l1")
        provenance[metric] = {"unit": unit, "key": key, "sources": sources}
    return representations, provenance


def digest(path: Path | str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scientific_code_identity() -> dict[str, str]:
    repo = HERE.parents[2]
    paths = [
        repo / "jaxent/cli/featurise.py",
        repo / "jaxent/src/models/config.py",
        repo / "jaxent/src/models/func/contacts.py",
        repo / "jaxent/src/models/func/uptake.py",
        repo / "jaxent/src/models/HDX/BV/forwardmodel.py",
    ]
    return {str(path): digest(path) for path in paths}


def feature_identity(row: dict[str, str], replica: int) -> dict:
    trajectory = replica_paths(row)[replica - 1]
    pdb = HERE / row["pdb_path"]
    return {
        "system_id": row["system_id"],
        "replica": replica,
        "protocol": PROTOCOL,
        "inputs": {str(pdb): digest(pdb), str(trajectory): digest(trajectory)},
        "code": scientific_code_identity(),
    }


def feature_folder(output: Path, system: str, replica: int) -> Path:
    return output / "features" / system / f"R{replica}"


def validate_feature_marker(folder: Path, identity: dict) -> dict:
    marker = json.loads((folder / "complete.json").read_text())
    if marker["identity"] != identity:
        raise ValueError(f"Feature provenance changed; use a fresh output: {folder}")
    for name, expected in marker["artifacts"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Corrupt feature artifact: {folder / name}")
    if not marker["audit"]["passed"]:
        raise ValueError(f"Saved feature audit did not pass: {folder}")
    return marker


def _residue_mapping(row: dict[str, str], replica: int, folder: Path):
    trajectory = replica_paths(row)[replica - 1]
    universe = mda.Universe(str(HERE / row["pdb_path"]), str(trajectory))
    topologies = json.loads((folder / "topology.json").read_text())["topologies"]
    protein = universe.select_atoms("protein").residues
    chains = {r.resindex: str(mda_TopologyAdapter._get_chain_id(r)) for r in protein}
    if len(set(chains.values())) != 1 or len(set(protein.resids)) != len(protein):
        raise ValueError("Checkpoint 36 requires one chain with unique residue numbers")
    # Partial_Topology residue numbers are renumbered after exclusions. Recover
    # physical residues from the explicit amide-H criterion in original chain order.
    donor_hydrogens = universe.select_atoms(
        "protein and not resname PRO and (name H or name HN)"
    )
    residues = donor_hydrogens.residues
    if len(donor_hydrogens) != len(residues):
        raise ValueError("Each represented BV residue must have exactly one amide H/HN")
    if len(residues) != len(topologies):
        raise ValueError(
            "Physical amide donors do not align one-to-one with the feature topology"
        )
    ordinal = {residue.resindex: index for index, residue in enumerate(protein)}
    return universe, protein, residues, donor_hydrogens, ordinal, chains


def audit_features(row: dict[str, str], replica: int, folder: Path) -> dict:
    with np.load(folder / "features.npz", allow_pickle=False) as archive:
        cached = {
            key: np.asarray(archive[key])
            for key in ("heavy_contacts", "acceptor_contacts", "k_ints")
        }
    universe, protein, residues, donor_hydrogens, ordinal, chains = _residue_mapping(
        row, replica, folder
    )
    rate_map = calculate_HDXrate(
        residues,
        PROTOCOL["temperature_k"],
        PROTOCOL["pd"],
        unit=PROTOCOL["kint_unit"],
    )
    expected_rates = np.asarray([rate_map[residue] for residue in residues])
    rate_error = float(np.max(np.abs(cached["k_ints"] - expected_rates)))
    rate_passed = bool(
        np.allclose(cached["k_ints"], expected_rates, rtol=2e-5, atol=1e-8)
    )
    targets = {
        "heavy_contacts": universe.atoms[
            [residue.atoms.select_atoms("name N")[0].index for residue in residues]
        ],
        "acceptor_contacts": donor_hydrogens,
    }
    records = []
    for name, atom_selection, radius, scale in (
        ("heavy_contacts", "not type H", 6.5, 0.1),
        ("acceptor_contacts", "type O", 2.4, 0.1),
    ):
        environment = universe.select_atoms(
            PROTOCOL["contact_environment"]
        ).select_atoms(atom_selection)
        ignored = np.asarray(
            [
                [
                    atom.resindex in ordinal
                    and chains[atom.resindex] == chains[residue.resindex]
                    and PROTOCOL["residue_ignore"][0]
                    <= ordinal[atom.resindex] - ordinal[residue.resindex]
                    <= PROTOCOL["residue_ignore"][1]
                    for atom in environment
                ]
                for residue in residues
            ],
            dtype=bool,
        )
        for frame in (101, 550, 1000):
            universe.trajectory[frame]
            distances = distance_array(
                targets[name].positions,
                environment.positions,
                box=universe.dimensions,
                backend="serial",
            )
            expected = 1.0 / (1.0 + ((distances - radius) / scale) ** 6)
            expected[ignored] = 0.0
            expected = expected.sum(axis=1)
            error = float(np.max(np.abs(cached[name][:, frame] - expected)))
            records.append(
                {
                    "feature": name,
                    "frame": frame,
                    "max_abs_error": error,
                    "passed": bool(
                        np.allclose(
                            cached[name][:, frame], expected, rtol=2e-5, atol=2e-4
                        )
                    ),
                }
            )
    reference = GRAPH_REFERENCE / "systems" / row["system_id"]
    reference_records = []
    if (reference / "features.npz").exists() and (reference / "frames.npz").exists():
        with (
            np.load(reference / "features.npz", allow_pickle=False) as old,
            np.load(reference / "frames.npz", allow_pickle=False) as frame_data,
        ):
            mask = frame_data["replicas"] == replica
            frames = frame_data["frames"][mask].astype(int)
            offset = np.flatnonzero(mask)
            for name in ("heavy_contacts", "acceptor_contacts"):
                error = float(
                    np.max(np.abs(cached[name][:, frames] - old[name][:, offset]))
                )
                reference_records.append(
                    {
                        "feature": name,
                        "max_abs_error": error,
                        "passed": bool(
                            np.allclose(
                                cached[name][:, frames],
                                old[name][:, offset],
                                rtol=2e-5,
                                atol=2e-4,
                            )
                        ),
                    }
                )
            error = float(np.max(np.abs(cached["k_ints"] - old["k_ints"])))
            reference_records.append(
                {
                    "feature": "k_ints",
                    "max_abs_error": error,
                    "passed": bool(
                        np.allclose(
                            cached["k_ints"], old["k_ints"], rtol=2e-5, atol=1e-8
                        )
                    ),
                }
            )
    shape_passed = (
        cached["heavy_contacts"].shape == cached["acceptor_contacts"].shape
        and cached["heavy_contacts"].shape[1] == 1001
        and cached["heavy_contacts"].shape[0] == len(cached["k_ints"])
        and all(np.isfinite(value).all() for value in cached.values())
    )
    passed = bool(
        shape_passed
        and rate_passed
        and all(item["passed"] for item in records)
        and (not reference_records or all(item["passed"] for item in reference_records))
    )
    return {
        "passed": passed,
        "shape_passed": bool(shape_passed),
        "contact_passed": all(item["passed"] for item in records),
        "rate_passed": rate_passed,
        "rate_unit": PROTOCOL["kint_unit"],
        "rates_max_abs_error": rate_error,
        "contact_checks": records,
        "reference_checks": reference_records,
    }


def generate_features(row: dict[str, str], replica: int, output: Path) -> dict:
    folder = feature_folder(output, row["system_id"], replica)
    identity = feature_identity(row, replica)
    if (folder / "complete.json").exists():
        marker = validate_feature_marker(folder, identity)
        audit_code = digest(Path(__file__))
        if marker.get("audit_code_sha256") != audit_code:
            marker["audit"] = audit_features(row, replica, folder)
            if not marker["audit"]["passed"]:
                raise ValueError(f"Re-audit failed after audit code changed: {folder}")
            marker["audit_code_sha256"] = audit_code
            write_json(folder / "complete.json", marker)
        return marker
    executable = Path(sys.executable).parent / "jaxent-featurise"
    command = [
        str(executable),
        "--top_path",
        str(HERE / row["pdb_path"]),
        "--trajectory_path",
        str(replica_paths(row)[replica - 1]),
        "--output_dir",
        str(folder),
        "--name",
        f"atlas_cp36_{row['system_id']}_R{replica}",
        "bv",
        "--temperature",
        "300",
        "--ph",
        "7",
        "--kint_unit",
        "min^-1",
        "--bv_bc",
        "0.35",
        "--bv_bh",
        "2.0",
        "--heavy_radius",
        "6.5",
        "--o_radius",
        "2.4",
        "--num_timepoints",
        "0",
        "--residue_ignore",
        "-2",
        "2",
        "--contact_mode",
        "bradshaw_switch",
        "--switch_scale_nc",
        "0.1",
        "--switch_scale_nh",
        "0.1",
        "--mda_contact_environment",
        "all",
    ]
    artifacts = ("features.npz", "topology.json")
    if folder.exists() and all((folder / name).exists() for name in artifacts):
        audit = audit_features(row, replica, folder)
        if not audit["passed"]:
            write_json(folder / "failure.json", {"identity": identity, "audit": audit})
            raise ValueError(f"Recovered feature audit failed: {folder}")
        marker = {
            "identity": identity,
            "command": command,
            "runtime_seconds": None,
            "recovered_after_interruption": True,
            "audit": audit,
            "audit_code_sha256": digest(Path(__file__)),
            "artifacts": {name: digest(folder / name) for name in artifacts},
        }
        write_json(folder / "complete.json", marker)
        return marker
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(
            f"Incomplete feature folder exists; inspect or choose a fresh output: {folder}"
        )
    folder.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "10", "OPENBLAS_NUM_THREADS": "1"}
    started = time.perf_counter()
    with (folder / "featurise.log").open("w") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
        )
    if completed.returncode:
        raise RuntimeError(f"Featurisation failed; see {folder / 'featurise.log'}")
    audit = audit_features(row, replica, folder)
    if not audit["passed"]:
        write_json(folder / "failure.json", {"identity": identity, "audit": audit})
        raise ValueError(f"Feature audit failed: {folder}")
    marker = {
        "identity": identity,
        "command": command,
        "runtime_seconds": time.perf_counter() - started,
        "audit": audit,
        "audit_code_sha256": digest(Path(__file__)),
        "artifacts": {name: digest(folder / name) for name in artifacts},
    }
    write_json(folder / "complete.json", marker)
    return marker


def corrected_system_data(row: dict[str, str], config: dict, output: Path) -> dict:
    system = row["system_id"]
    audit_path = (
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint8_strict_conformal/parts"
        / f"{system}.pairs.parquet"
    )
    pairs = _pair_sets_from_audit(pd.read_parquet(audit_path))
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    signatures = frame_w1_signatures(
        coordinates,
        config["analysis"]["pairwise_geometry"]["support_audit"][
            "w1_support_quantiles"
        ],
    )
    matrices = w1_matrices(signatures, replicas)
    keep = post_equilibration_indices(
        1001,
        config["analysis"]["equilibration_ns"],
        config["analysis"]["frame_interval_ns"],
    )
    z_parts = []
    quantiles = []
    for replica in (1, 2, 3):
        folder = feature_folder(output, system, replica)
        with np.load(folder / "features.npz", allow_pickle=False) as archive:
            heavy = np.asarray(archive["heavy_contacts"][:, keep], dtype=float)
            acceptor = np.asarray(archive["acceptor_contacts"][:, keep], dtype=float)
        z = PROTOCOL["bv_bc"] * heavy + PROTOCOL["bv_bh"] * acceptor
        z_parts.append(z)
        quantiles.append(
            {
                "system_id": system,
                "replica": replica,
                "contact": "corrected_bradshaw_0p1",
                "logpf_min": float(z.min()),
                "logpf_median": float(np.median(z)),
                "logpf_max": float(z.max()),
                "heavy_median": float(np.median(heavy)),
                "acceptor_median": float(np.median(acceptor)),
            }
        )
    return {
        "system": system,
        "pairs": pairs,
        "coordinates": coordinates,
        "replicas": replicas,
        "frames": frames,
        "matrices": matrices,
        "z": np.concatenate(z_parts, axis=1),
        "feature_quantiles": quantiles,
    }


def work_representations(z: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    mean = np.mean(z, axis=0)
    shape = np.abs(z - mean[None, :])
    density = entropy_contributions(shape, "legacy_zq")
    # G/(RT) = H/(RT) - S/R.  ``density`` is S/R = -Pi log(Pi).
    work_opt = shape - density
    return {
        "work_scale": (mean, "l1"),
        "work_shape": (shape, "l1"),
        "work_density": (density, "l1"),
        "work_opt": (work_opt, "l1"),
        "pf_l1": (z, "l1"),
        "pf_l2": (z, "l2"),
    }


def ca_radius_of_gyration(coordinates: np.ndarray) -> np.ndarray:
    """Equal-weight C-alpha radius about the centroid, in Angstroms per frame."""
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))


def coordinate_dispersion(
    metric: np.ndarray,
    neighbours: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    shrinkage: float,
    reference: float | None,
) -> tuple[np.ndarray, float]:
    local = metric[np.arange(len(metric))[:, None], neighbours]
    radius_squared = np.mean(np.square(local), axis=1)
    if reference is None:
        positive = radius_squared[radius_squared > 0]
        reference = float(np.median(positive)) if len(positive) else 1.0
    variance = radius_squared + shrinkage * reference
    feature = np.abs(0.5 * np.log(variance[left]) - 0.5 * np.log(variance[right]))
    return feature, reference


def normalized_scale(alpha: float, feature: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target))
    return (
        float(alpha * np.linalg.norm(feature) / denominator) if denominator else np.nan
    )


def evaluate_system(row: dict[str, str], config: dict, edges: np.ndarray, output: Path):
    started = time.perf_counter()
    data = corrected_system_data(row, config, output)
    system = data["system"]
    signed, bandwidth = density_targets(data, FIT, 10)
    take = {
        replica: sampled_indices(system, replica, len(signed[replica]), PAIR_CAP)
        for replica in (1, 2, 3)
    }
    target = {replica: np.abs(signed[replica][take[replica]]) for replica in (1, 2, 3)}
    endpoints = {
        replica: (
            data["pairs"][replica].left_frame.to_numpy()[take[replica]],
            data["pairs"][replica].right_frame.to_numpy()[take[replica]],
        )
        for replica in (1, 2, 3)
    }
    local_endpoints = {
        replica: tuple(
            global_to_local(data["matrices"][replica][0], endpoint)
            for endpoint in endpoints[replica]
        )
        for replica in (1, 2, 3)
    }
    representations = work_representations(data["z"])
    energies, energy_provenance = energy_representations(data, config)
    representations.update(energies)
    direct: dict[str, dict[int, np.ndarray]] = {
        name: {
            replica: direct_distance(values, *endpoints[replica], kind)
            for replica in (1, 2, 3)
        }
        for name, (values, kind) in representations.items()
    }
    direct["work_fitting"] = {
        replica: direct["work_scale"][replica] + direct["work_density"][replica]
        for replica in (1, 2, 3)
    }
    direct["work_magnitude"] = {
        replica: direct["work_shape"][replica] - direct["work_scale"][replica]
        for replica in (1, 2, 3)
    }
    rg = ca_radius_of_gyration(data["coordinates"])
    direct["rg"] = {
        replica: np.abs(rg[endpoints[replica][0]] - rg[endpoints[replica][1]])
        for replica in (1, 2, 3)
    }
    for coordinate in ("rmsd", "w1"):
        direct[coordinate] = {
            replica: data["pairs"][replica][coordinate].to_numpy()[take[replica]]
            for replica in (1, 2, 3)
        }
    predictions = []
    fits = []
    for metric in DIRECT_METRICS:
        alpha = fitted_scale(direct[metric][FIT], target[FIT])
        prediction = alpha * direct[metric][TEST]
        predictions.append((metric, "direct", prediction))
        fits.append(
            {
                "system_id": system,
                "metric": metric,
                "model": "direct",
                "alpha": alpha,
                "normalized_alpha": normalized_scale(
                    alpha, direct[metric][FIT], target[FIT]
                ),
                "feature_rms": float(np.sqrt(np.mean(np.square(direct[metric][FIT])))),
                "target_rms": float(np.sqrt(np.mean(np.square(target[FIT])))),
                "k": np.nan,
                "shrinkage": np.nan,
                "tune_mae": float(
                    np.mean(np.abs(target[TUNE] - alpha * direct[metric][TUNE]))
                ),
                "negative_prediction_fraction": float(np.mean(prediction < 0)),
                "pathology_alpha_ge_1000": bool(alpha >= 1000),
            }
        )
    for metric in VARIANCE_METRICS:
        values, kind = representations[metric]
        candidates = []
        for k in K_VALUES:
            for shrinkage in SHRINKAGES:
                features = {}
                reference = None
                for replica, (global_indices, matrix) in data["matrices"].items():
                    local_values = (
                        values[global_indices]
                        if np.asarray(values).ndim == 1
                        else values[:, global_indices]
                    )
                    local_mean, local_variance, reference = local_statistics(
                        local_values,
                        nearest(matrix, k),
                        reference if replica != FIT else None,
                        shrinkage,
                    )
                    features[replica] = pair_local_features(
                        local_values,
                        local_mean,
                        local_variance,
                        *local_endpoints[replica],
                        kind,
                    )["variance_magnitude"]
                alpha = fitted_scale(features[FIT], target[FIT])
                tune_mae = float(np.mean(np.abs(target[TUNE] - alpha * features[TUNE])))
                candidates.append((tune_mae, k, shrinkage, alpha, features))
        tune_mae, k, shrinkage, alpha, features = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        prediction = alpha * features[TEST]
        predictions.append((metric, "variance_magnitude", prediction))
        fits.append(
            {
                "system_id": system,
                "metric": metric,
                "model": "variance_magnitude",
                "alpha": alpha,
                "normalized_alpha": normalized_scale(alpha, features[FIT], target[FIT]),
                "feature_rms": float(np.sqrt(np.mean(np.square(features[FIT])))),
                "target_rms": float(np.sqrt(np.mean(np.square(target[FIT])))),
                "k": k,
                "shrinkage": shrinkage,
                "tune_mae": tune_mae,
                "negative_prediction_fraction": 0.0,
                "pathology_alpha_ge_1000": bool(alpha >= 1000),
            }
        )
    coordinate_matrices: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in COORDINATES
    }
    for replica, (global_indices, w1) in data["matrices"].items():
        coordinate_matrices["w1"][replica] = w1
        coordinate_matrices["rmsd"][replica] = rmsd_matrix(
            data["coordinates"][global_indices]
        )
        local_rg = rg[global_indices]
        coordinate_matrices["rg"][replica] = np.abs(
            local_rg[:, None] - local_rg[None, :]
        )
    for metric in COORDINATES:
        candidates = []
        for k in K_VALUES:
            neighbours = {
                replica: nearest(data["matrices"][replica][1], k)
                for replica in (1, 2, 3)
            }
            for shrinkage in SHRINKAGES:
                features = {}
                reference = None
                for replica in (FIT, TUNE, TEST):
                    features[replica], reference = coordinate_dispersion(
                        coordinate_matrices[metric][replica],
                        neighbours[replica],
                        *local_endpoints[replica],
                        shrinkage,
                        reference if replica != FIT else None,
                    )
                alpha = fitted_scale(features[FIT], target[FIT])
                tune_mae = float(np.mean(np.abs(target[TUNE] - alpha * features[TUNE])))
                candidates.append((tune_mae, k, shrinkage, alpha, features))
        tune_mae, k, shrinkage, alpha, features = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        prediction = alpha * features[TEST]
        predictions.append((metric, "local_dispersion", prediction))
        fits.append(
            {
                "system_id": system,
                "metric": metric,
                "model": "local_dispersion",
                "alpha": alpha,
                "normalized_alpha": normalized_scale(alpha, features[FIT], target[FIT]),
                "feature_rms": float(np.sqrt(np.mean(np.square(features[FIT])))),
                "target_rms": float(np.sqrt(np.mean(np.square(target[FIT])))),
                "k": k,
                "shrinkage": shrinkage,
                "tune_mae": tune_mae,
                "negative_prediction_fraction": 0.0,
                "pathology_alpha_ge_1000": bool(alpha >= 1000),
            }
        )
    test_w1 = data["pairs"][TEST].w1.to_numpy()[take[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    results = []
    for metric, model, prediction in predictions:
        for band in range(6):
            mask = (test_w1 >= edges[band]) & (
                test_w1 < edges[band + 1] if band < 5 else True
            )
            if mask.sum() < 30:
                continue
            results.append(
                {
                    "system_id": system,
                    "metric": metric,
                    "model": model,
                    "band": f"q{band}",
                    "pairs": int(mask.sum()),
                    "mae": float(
                        np.mean(np.abs(target[TEST][mask] - prediction[mask]))
                    ),
                    "spearman": finite_spearman(target[TEST][mask], prediction[mask]),
                    **mass_metrics(
                        target[TEST][mask],
                        prediction[mask],
                        target[FIT],
                        settings["distribution_bins"],
                        settings["distribution_smoothing"],
                    ),
                }
            )
    runtime = {
        "system_id": system,
        "runtime_seconds": time.perf_counter() - started,
        "bandwidth_angstrom": bandwidth,
        "energy_provenance_json": json.dumps(energy_provenance, sort_keys=True),
    }
    return results, fits, data["feature_quantiles"], runtime


def selected_rows(scope: str, system: str) -> list[dict[str, str]]:
    rows = load_systems()
    if scope == "single":
        selected = [row for row in rows if row["system_id"] == system]
        if len(selected) != 1:
            raise ValueError(f"Unknown ATLAS system: {system}")
        return selected
    cohort = pd.read_parquet(PILOT_TABLE).query("pilot")
    if cohort.system_id.nunique() != 24:
        raise ValueError("Frozen pilot table must contain exactly 24 systems")
    ids = set(cohort.system_id)
    return [row for row in rows if row["system_id"] in ids]


def save_evaluation(
    row: dict[str, str], config: dict, edges: np.ndarray, output: Path, parts: Path
) -> str:
    """Evaluate one system and atomically materialise its resumable tables."""
    result = evaluate_system(row, config, edges, output)
    for name, records in zip(("results", "fits", "quantiles", "runtime"), result):
        records = records if isinstance(records, list) else [records]
        atomic_parquet(
            pd.DataFrame(records), parts / f"{row['system_id']}.{name}.parquet"
        )
    return row["system_id"]


def bootstrap_interval(
    values: np.ndarray, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def build_report(destination: Path, rows: list[dict[str, str]], scope: str) -> None:
    tables = {}
    for name in ("results", "fits", "quantiles", "runtime"):
        parts = [
            destination / "parts" / f"{row['system_id']}.{name}.parquet" for row in rows
        ]
        missing = [path for path in parts if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing analysis parts: {missing[:3]}")
        tables[name] = pd.concat(
            [pd.read_parquet(path) for path in parts], ignore_index=True
        )
        atomic_parquet(tables[name], destination / f"corrected_variance_{name}.parquet")
    summary_rows = []
    for key, block in tables["results"].groupby(["metric", "model", "band"]):
        values = block.distribution_recovery.to_numpy()
        low, high = bootstrap_interval(values, SEED + sum(map(ord, "".join(key))))
        summary_rows.append(
            {
                "metric": key[0],
                "model": key[1],
                "band": key[2],
                "recovery_mean": float(values.mean()),
                "recovery_median": float(np.median(values)),
                "recovery_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "recovery_ci_low": low,
                "recovery_ci_high": high,
                "systems": int(block.system_id.nunique()),
                "pairs": int(block.pairs.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_parquet(summary, destination / "corrected_variance_summary.parquet")
    historical = pd.DataFrame()
    if HISTORICAL_FITS.exists():
        old = pd.read_parquet(HISTORICAL_FITS)
        mapping = {"work_density_legacy_zq": "work_density"}
        old["metric"] = old.metric.replace(mapping)
        historical = old[
            old.system_id.isin([row["system_id"] for row in rows])
            & old.metric.isin((*CLEAN_WORK[:3], *PF_METRICS))
            & old.model.isin(("direct", "variance_magnitude"))
        ][["system_id", "metric", "model", "alpha"]].rename(
            columns={"alpha": "historical_alpha"}
        )
        comparison = tables["fits"].merge(
            historical, on=["system_id", "metric", "model"], how="left"
        )
        comparison["alpha_ratio_to_historical"] = (
            comparison.alpha / comparison.historical_alpha
        )
        comparison.to_csv(destination / "alpha_historical_comparison.csv", index=False)
    families = {
        "Work": (*CLEAN_WORK, *DERIVED_WORK),
        "Protection factor": PF_METRICS,
        "Coordinates": COORDINATES,
        "Total energy": ENERGY_METRICS,
    }
    fig, axes = plt.subplots(2, 4, figsize=(25, 10), sharey=True)
    for column, (family, metrics) in enumerate(families.items()):
        for row_index, models in enumerate(
            (("direct",), ("variance_magnitude", "local_dispersion"))
        ):
            axis = axes[row_index, column]
            block = summary[summary.metric.isin(metrics) & summary.model.isin(models)]
            for metric, points in block.groupby("metric"):
                points = points.assign(
                    order=points.band.str[1:].astype(int)
                ).sort_values("order")
                x = points.order.to_numpy()
                y = 100 * points.recovery_mean.to_numpy()
                label = {
                    "rg": "Cα radius of gyration",
                    "pyro_ref2015": "PyRosetta ref2015 total",
                    "openmm_total": "OpenMM total (vacuum)",
                }.get(metric, metric)
                axis.plot(x, y, marker="o", label=label)
                axis.fill_between(
                    x,
                    100 * points.recovery_ci_low.to_numpy(),
                    100 * points.recovery_ci_high.to_numpy(),
                    alpha=0.10,
                )
            axis.set_title(
                f"{family}: {'direct' if row_index == 0 else 'local variance/dispersion'}"
            )
            axis.set_xticks(range(6), [f"q{i}" for i in range(6)])
            axis.set_xlabel("Global structural-W1 band")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Mean recovery (%)")
    axes[1, 0].set_ylabel("Mean recovery (%)")
    fig.tight_layout()
    fig.savefig(destination / "recovery_vs_w1.png", dpi=180)
    fig.savefig(destination / "recovery_vs_w1.pdf")
    plt.close(fig)
    fit_table = tables["fits"].copy()
    fit_table["display_alpha"] = fit_table.alpha.where(fit_table.alpha > 0, np.nan)
    labels = (
        fit_table.metric + ":" + fit_table.model.str.replace("variance_magnitude", "VM")
    )
    order = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for index, label in enumerate(order):
        block = fit_table[labels == label]
        axes[0].scatter(
            np.full(len(block), index), block.display_alpha, alpha=0.7, s=22
        )
        axes[1].scatter(
            np.full(len(block), index), block.normalized_alpha, alpha=0.7, s=22
        )
    axes[0].axhline(
        1000, color="red", linestyle="--", linewidth=1, label="pathology warning"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Raw fitted alpha")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel(r"alpha ||x|| / ||y||")
    local = fit_table[fit_table.k.notna()]
    if len(local):
        choice = local.groupby(["k", "shrinkage"]).size().unstack(fill_value=0)
        choice.plot.bar(ax=axes[2])
    axes[2].set_title("Selected local parameters")
    axes[2].set_ylabel("Fits")
    for axis in axes[:2]:
        axis.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination / "parameter_distributions.png", dpi=180)
    plt.close(fig)
    pathology = bool(
        tables["fits"].pathology_alpha_ge_1000.any()
        or not np.isfinite(
            tables["fits"][
                ["alpha", "normalized_alpha", "feature_rms", "target_rms"]
            ].to_numpy()
        ).all()
        or (tables["fits"].feature_rms <= np.finfo(float).eps).any()
    )
    feature_audits = []
    output = destination.parent
    for row in rows:
        for replica in (1, 2, 3):
            marker_path = (
                feature_folder(output, row["system_id"], replica) / "complete.json"
            )
            feature_audits.append(json.loads(marker_path.read_text())["audit"])
    finite_normalized = tables["fits"].normalized_alpha[
        np.isfinite(tables["fits"].normalized_alpha)
    ]
    report = {
        "checkpoint": 36,
        "scope": scope,
        "systems": len(rows),
        "system_ids": [row["system_id"] for row in rows],
        "protocol": PROTOCOL,
        "assignment": "A-fit/B-tune/C-test",
        "target": "magnitude of rank-10 structural-W1 KDE log-density change",
        "direct_metrics": list(DIRECT_METRICS),
        "variance_metrics": list(VARIANCE_METRICS),
        "coordinate_dispersion": list(COORDINATES),
        "radius_of_gyration": "Equal-weight C-alpha Rg (Angstrom); direct predictor |Rg_i-Rg_j|; local dispersion uses W1 neighbours and squared Rg differences",
        "energy_metrics": {
            name: {"unit": unit, "key": key, "source_directory": str(root)}
            for name, (root, key, unit) in ENERGY_SOURCES.items()
        },
        "kint_used_in_recovery": False,
        "feature_audit": {
            "all_passed": all(audit["passed"] for audit in feature_audits),
            "max_rate_abs_error": max(
                audit["rates_max_abs_error"] for audit in feature_audits
            ),
            "max_contact_abs_error": max(
                check["max_abs_error"]
                for audit in feature_audits
                for check in audit["contact_checks"]
            ),
        },
        "alpha_diagnostics": {
            "raw_min": float(tables["fits"].alpha.min()),
            "raw_max": float(tables["fits"].alpha.max()),
            "normalized_min": float(finite_normalized.min()),
            "normalized_max": float(finite_normalized.max()),
            "warning_threshold": 1000.0,
        },
        "parameter_pathology_detected": pathology,
        "pilot_status": "pending_manual_review" if scope == "single" else "completed",
        "analysis_code_sha256": digest(Path(__file__)),
    }
    report_path = destination / "checkpoint36_report.yaml"
    temporary = report_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(report, sort_keys=False))
    temporary.replace(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("single", "pilot"), default="single")
    parser.add_argument("--system", default=SINGLE_SYSTEM)
    parser.add_argument(
        "--phase", choices=("features", "analyse", "report", "all"), default="all"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--approve-pilot",
        action="store_true",
        help="Acknowledge review of the clean single-system alpha report.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    if args.scope == "pilot":
        review = output / "single/checkpoint36_report.yaml"
        if not args.approve_pilot:
            parser.error("--scope pilot requires --approve-pilot after manual review")
        if not review.exists():
            parser.error("run and review the single-system checkpoint before the pilot")
        single_report = yaml.safe_load(review.read_text())
        if single_report.get("parameter_pathology_detected", True):
            parser.error("the single-system report contains a parameter pathology")
    rows = selected_rows(args.scope, args.system)
    destination = output / args.scope
    destination.mkdir(parents=True, exist_ok=True)
    if args.phase in ("features", "all"):
        for row in rows:
            for replica in (1, 2, 3):
                marker = generate_features(row, replica, output)
                print(
                    f"features {row['system_id']} R{replica}: audit={marker['audit']['passed']}",
                    flush=True,
                )
    if args.phase in ("analyse", "all"):
        config = load_config()
        with (
            HERE
            / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
        ).open() as handle:
            edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
        parts = destination / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        pending = []
        for row in rows:
            required = [
                parts / f"{row['system_id']}.{name}.parquet"
                for name in ("results", "fits", "quantiles", "runtime")
            ]
            if all(path.exists() for path in required):
                fits = pd.read_parquet(parts / f"{row['system_id']}.fits.parquet")
                expected = (
                    {(metric, "direct") for metric in DIRECT_METRICS}
                    | {(metric, "variance_magnitude") for metric in VARIANCE_METRICS}
                    | {(metric, "local_dispersion") for metric in COORDINATES}
                )
                if expected.issubset(set(zip(fits.metric, fits.model))):
                    print(f"{row['system_id']} resumed", flush=True)
                else:
                    pending.append(row)
            else:
                pending.append(row)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(save_evaluation, row, config, edges, output, parts): row
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                print(
                    f"[{index}/{len(pending)}] {future.result()} complete", flush=True
                )
    if args.phase in ("report", "all"):
        build_report(destination, rows, args.scope)
        print(f"report: {destination / 'checkpoint36_report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
