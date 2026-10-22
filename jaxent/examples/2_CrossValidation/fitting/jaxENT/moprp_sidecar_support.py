"""MoPrP adapters around the existing native fitting and sidecar conventions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.lib.util import convert_aa_code
import numpy as np
from scipy.spatial.distance import pdist, squareform

from jaxent.examples.common import loading
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.moprp_weighted_loss import peptide_time_weights
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.custom_types.key import m_key
from jaxent.src.opt.loss.graph_laplacian import FrameGraph, build_frame_graph_from_distances
from jaxent.src.opt.loss.original_omc_laplacian import build_omc_kernel, create_original_omc_loss

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
ISO = EXPERIMENT.parent/"1_IsoValidation_OMass/fitting/jaxENT"
sys.path.insert(0, str(ISO))
from analyze_within_trajectory_selection import candidates
from iso_sidecar_geometry import median_scale
from run_maxent_omc_sidecar import sparse_omc_loss

saved_candidates = candidates


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def model():
    times = loading.load_hdx_timepoints_minutes(EXPERIMENT/"data/_MoPrP/moprp.times")
    result = build_uptake_model("standard", times, kint_unit="min^-1")
    result.forward[m_key("HDX_peptide")].frame_averaging_mode = "frame_uptake"
    return result


def raw_weighted_mse(predicted, observed, weights):
    predicted, observed, weights = [np.asarray(x, dtype=float) for x in (predicted, observed, weights)]
    if predicted.shape != observed.shape or weights.shape != observed.shape:
        raise ValueError("Peptide/timepoint predictions, targets and file weights must align")
    if not all(np.isfinite(x).all() for x in (predicted, observed, weights)) or np.any(weights <= 0):
        raise ValueError("Require finite predictions and positive file weights")
    return float(np.mean(weights*(predicted-observed)**2))


def config_and_targets():
    config = ExperimentConfig.from_yaml(str(EXPERIMENT/"config.yaml"))
    populations = json.loads((EXPERIMENT/"analysis/state_ratios.json").read_text())["fractional_populations"]
    target = {name: populations.get(key, {}).get("fraction", 0.) for name, key in
        [("Folded", "folded"), ("PUF1", "PUF1"), ("PUF2", "PUF2"), ("PUF3", "PUF3"), ("unfolded", "unfolded")]}
    for name in config.scoring.state_mapping.values():
        target.setdefault(name, 0.)
    return config, target


def aligned_file_weights(data):
    return np.asarray(peptide_time_weights(data, 1), dtype=float)


def prepare_geometry(output, features_dir, topology_path, trajectory_path):
    from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import periodic_dihedral_pair_distance
    config, _ = config_and_targets()
    features, feature_top = loading.load_features_and_topology(str(features_dir), "AF2_MSAss", config.scoring.ensemble_feature_map)
    physical = model().params
    z = np.asarray(physical.bv_bc).item()*np.asarray(features.heavy_contacts, dtype=float) + np.asarray(physical.bv_bh).item()*np.asarray(features.acceptor_contacts, dtype=float)
    inputs = {str(p): digest(p) for p in [features_dir/"features_AF2_MSAss.npz", features_dir/"topology_AF2_MSAss.json",
        topology_path, trajectory_path, Path(__file__)]}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["input_sha256"] != inputs:
            raise ValueError("Geometry inputs changed; use a new campaign output directory")
        for name, fingerprint in metadata["file_sha256"].items():
            if digest(output/name) != fingerprint:
                raise ValueError(f"Geometry file changed: {name}")
        return metadata
    universe = mda.Universe(str(topology_path), str(trajectory_path))
    if len(universe.trajectory) != features.features_shape[1]:
        raise ValueError("Geometry trajectory and native features must have identical frame counts/order")
    feature_residues = {int(r) for top in feature_top for r in top.residues}
    residues = [res for res in universe.select_atoms("protein").residues if int(res.resid) in feature_residues]
    if {int(res.resid) for res in residues} != feature_residues:
        raise ValueError("Trajectory does not cover the feature residue IDs")
    sequence = "".join(convert_aa_code(res.resname) for res in residues)
    if sequence != feature_top[0].fragment_sequence:
        raise ValueError("Trajectory sequence does not match the native feature topology")
    atomgroups, angle_ids = [], []
    for residue in residues:
        phi, psi = residue.phi_selection(), residue.psi_selection()
        if phi is not None and psi is not None:
            atomgroups.extend((phi, psi))
            angle_ids.extend((f"{residue.resid}:phi", f"{residue.resid}:psi"))
    if not atomgroups:
        raise ValueError("No matching backbone phi/psi angles")
    angles = np.deg2rad(np.asarray(Dihedral(atomgroups).run().results.angles, dtype=float)).T
    if not np.isfinite(angles).all():
        raise ValueError("Nonfinite dihedral angles")
    n = z.shape[1]
    left, right = np.triu_indices(n, 1)
    torsion = np.zeros((n, n))
    torsion[left, right] = periodic_dihedral_pair_distance(angles, left, right)
    torsion += torsion.T
    distances = {"work_scale": np.abs(z.mean(axis=0)[:, None]-z.mean(axis=0)[None, :]), "dihedral_rmsd": torsion}
    metrics = {}
    for name, raw in distances.items():
        scaled, median = median_scale(raw)
        np.savez_compressed(output/f"{name}.npz", raw_distances=raw, normalized_distances=scaled,
            median_positive_distance=median, frame=np.arange(n), metric=name)
        metrics[name] = dict(median_positive_distance=median, native_units="work/RT" if name == "work_scale" else "degrees")
    graph = build_frame_graph_from_distances(squareform(pdist(z.T)), k=20, metric="logpf_l2",
        weighted=False, symmetrization="union")
    np.savez_compressed(output/"logpf_knn20.npz", edge_sources=np.asarray(graph.edge_sources),
        edge_targets=np.asarray(graph.edge_targets), n_nodes=n, k=20)
    metadata = dict(input_sha256=inputs, frames=n, feature_residues=len(residues), dihedral_angles=angle_ids,
        angle_difference="wrapped phi/psi RMS in degrees; existing periodic_dihedral_pair_distance",
        neighbour_metric="Euclidean L2 over frozen residue log-PF profiles", neighbour_k=20,
        symmetrization="union", edges=len(graph.edge_sources), metrics=metrics,
        file_sha256={name: digest(output/name) for name in ["work_scale.npz", "dihedral_rmsd.npz", "logpf_knn20.npz"]})
    write_json(manifest_path, metadata)
    return metadata


def regularizer(spec):
    if spec.method == "maxent":
        return None
    with np.load(Path(spec.geometry_dir)/f"{spec.edge_type}.npz") as archive:
        distances = archive["normalized_distances"]
    if not spec.graph_k:
        kernel = build_omc_kernel(distances, bandwidth=spec.bandwidth, metric=f"{spec.edge_type}_median_scaled")
        return "original_omc_laplacian", create_original_omc_loss(normalise=False), kernel, 1/spec.strength
    with np.load(Path(spec.geometry_dir)/"logpf_knn20.npz") as archive:
        if int(archive["k"]) != spec.graph_k or int(archive["n_nodes"]) != len(distances):
            raise ValueError("Shared L2-logPF neighbour graph mismatch")
        left, right = archive["edge_sources"], archive["edge_targets"]
        weights = np.exp(-np.minimum((distances[left, right]/spec.bandwidth)**2/2, 80))
        graph = FrameGraph(jnp.asarray(left), jnp.asarray(right), jnp.asarray(weights), len(distances),
            f"logpf_l2_neighbours_{spec.edge_type}_median_scaled", spec.graph_k)
    return "original_omc_laplacian_knn", sparse_omc_loss, graph, 1/spec.strength
