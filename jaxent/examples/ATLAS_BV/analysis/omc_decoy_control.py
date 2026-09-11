"""Frozen k=100 candidates, empirical targets and physical-BV decoy challenges.

The physical preflight is mandatory. A completed preparation/report with blocked
fits is a valid diagnostic outcome, never evidence for or against OMC recovery.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import fcntl
from importlib.metadata import version
import json
import multiprocessing
import os
import subprocess
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .common import HERE, load_config, load_systems, post_equilibration_indices

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/omc_decoy_control"
TIMES = np.array([10.0, 60.0, 600.0, 3600.0, 14400.0])
QUANTILES = np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64])
SEEDS = tuple(range(20260909, 20260914))
CHECKPOINTS = (1000, 3000, 10000)
# Absolute fractional uptake, not a relative threshold on arbitrarily tiny data.
SIGNAL_FLOOR = 1e-8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def uptake(log_pf, rates, weights, times=TIMES):
    """Default BV mean-log-PF model, with stable small-uptake evaluation."""
    mean = np.asarray(log_pf) @ np.asarray(weights)
    exposure = np.asarray(times)[:, None] * np.asarray(rates)[None, :] * np.exp(-mean)
    return -np.expm1(-exposure)


def partition_candidates(vectors, count=100, seed=SEEDS[0]):
    fit = KMeans(n_clusters=count, n_init=20, random_state=seed).fit(vectors)
    reps = np.array(
        [
            members[
                np.argmin(
                    np.sum((vectors[members] - fit.cluster_centers_[i]) ** 2, axis=1)
                )
            ]
            for i in range(count)
            for members in [np.flatnonzero(fit.labels_ == i)]
        ]
    )
    if len(np.unique(reps)) != count:
        raise ValueError("Candidate representatives must be unique")
    return reps, fit.labels_


def partition_targets(vectors, seed=SEEDS[0], minimum=20):
    distance = squareform(pdist(vectors))
    rows, candidates = [], {}
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(vectors)
        sizes = np.bincount(labels, minlength=k)
        valid = bool(sizes.min() >= minimum)
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        rows.append(
            dict(k=k, silhouette=score, min_size=int(sizes.min()), eligible=valid)
        )
        candidates[k] = labels
    eligible = sorted(
        (r for r in rows if r["eligible"]), key=lambda r: (-r["silhouette"], r["k"])
    )
    selected = eligible[0]["k"] if eligible else None
    for row in rows:
        row["selected"] = row["k"] == selected
    return pd.DataFrame(rows), candidates.get(selected)


def empirical_target(labels, removed=-1):
    keep = np.asarray(labels) != removed
    if not keep.any():
        raise ValueError("Target cannot be empty")
    weights = keep.astype(float) / keep.sum()
    masses = np.bincount(labels, weights=weights, minlength=int(max(labels)) + 1)
    return weights, masses


def random_decoys(log_pf, count, seed):
    rng = np.random.default_rng(seed)
    frame = rng.integers(log_pf.shape[1], size=(log_pf.shape[0], count))
    return log_pf[np.arange(log_pf.shape[0])[:, None], frame], frame


def donor_decoys(log_pf, recipient_residues, count, seed):
    """Random contiguous crops of ordered donor profiles; tile short donors."""
    if min(log_pf.shape) < 1 or recipient_residues < 1:
        raise ValueError("Nonempty donor and recipient required")
    rng = np.random.default_rng(seed)
    frame = rng.integers(log_pf.shape[1], size=count)
    length = log_pf.shape[0]
    # Allow a random phase for tiled profiles, and all valid crops for long ones.
    offset = rng.integers(
        length if length < recipient_residues else length - recipient_residues + 1,
        size=count,
    )
    mapping = (np.arange(recipient_residues)[:, None] + offset) % length
    repeats = (offset + recipient_residues + length - 1) // length
    return log_pf[mapping, frame[None, :]], dict(
        frame=frame, offset=offset, repeats=repeats, residue_mapping=mapping
    )


def graph(log_pf, sigmas=None):
    values = np.asarray(log_pf).mean(axis=0)
    distance = abs(values[:, None] - values[None, :])
    if sigmas is None:
        # Matches the preceding experiment's positive off-diagonal quantiles.
        pairs = distance[np.triu_indices(len(values), 1)]
        pairs = pairs[pairs > 0]
        if not len(pairs):
            raise ValueError("No positive graph distances")
        sigmas = np.maximum(np.quantile(pairs, QUANTILES), np.finfo(float).eps)
    sigmas = np.asarray(sigmas)
    if len(sigmas) != 6 or not np.isfinite(sigmas).all() or (sigmas <= 0).any():
        raise ValueError("Six positive, finite frozen bandwidths required")
    kernels = np.exp(
        -np.minimum(0.5 * (distance[None] / sigmas[:, None, None]) ** 2, 80)
    )
    return kernels, sigmas


def regularisation(weights, kernel):
    w = np.asarray(weights)
    return (
        0.5
        * len(w) ** 2
        * np.sum(w[:, None] * w[None, :] * kernel * (w[:, None] - w[None, :]) ** 2)
    )


def ess_stats(weights, native_count):
    w = np.asarray(weights)
    mass = w[:native_count].sum()
    native = w[:native_count] / mass if mass > 0 else np.zeros(native_count)
    ess = 1 / np.dot(w, w)
    conditional = 1 / np.dot(native, native) if mass > 0 else 0.0
    return dict(
        ess=float(ess),
        ess_fraction=float(ess / len(w)),
        native_mass=float(mass),
        native_ess=float(conditional),
        native_ess_fraction=float(conditional / native_count),
    )


def physical_check(log_pf, rates, targets, contact_ok=True, rates_ok=True):
    predictions = np.stack([uptake(log_pf, rates, w) for w in targets])
    differences = np.max(abs(predictions - predictions[0]), axis=(1, 2))
    finite = bool(
        np.isfinite(predictions).all()
        and np.isfinite(log_pf).all()
        and np.isfinite(rates).all()
        and (rates > 0).all()
    )
    informative = bool(
        ((predictions[0] > SIGNAL_FLOOR) & (predictions[0] < 1 - SIGNAL_FLOOR)).any()
    )
    contrast = bool((differences[1:] > SIGNAL_FLOOR).any())
    reasons = []
    if not contact_ok:
        reasons.append(
            "Cached contacts do not match the current configured contact calculation"
        )
    if not rates_ok:
        reasons.append(
            "Cached intrinsic rates do not match recipient sequence rates at 300 K, pD 7, seconds^-1"
        )
    if not finite:
        reasons.append("Nonfinite features/predictions or invalid intrinsic rates")
    if not informative:
        reasons.append(
            "Full-reference uptake has no observable between 1e-8 and 1-1e-8"
        )
    if not contrast:
        reasons.append("Every target-state removal changes uptake by at most 1e-8")
    return dict(
        passed=not reasons,
        reasons=reasons,
        signal_floor=SIGNAL_FLOOR,
        uptake_min=float(predictions.min()),
        uptake_max=float(predictions.max()),
        informative_reference_observations=int(
            (
                (predictions[0] > SIGNAL_FLOOR) & (predictions[0] < 1 - SIGNAL_FLOOR)
            ).sum()
        ),
        target_max_abs_change=differences.tolist(),
        log_pf_quantiles=np.quantile(log_pf, [0, 0.1, 0.5, 0.9, 1]).tolist(),
    ), predictions


def experiment_config():
    """Override contact construction for this experiment only."""
    config = load_config()
    override = yaml.safe_load((HERE / "omc_decoy_config.yaml").read_text())
    config["protocol"].update(override["protocol"])
    return config


def prepare_features(sid, config, output):
    """Use the existing BV featuriser with the experiment's explicit contact mode."""
    row = next(r for r in load_systems() if r["system_id"] == sid)
    protocol = config["protocol"]
    folder = output / "features" / sid / "R1"
    folder.mkdir(parents=True, exist_ok=True)
    pdb = HERE / row["pdb_path"]
    trajectory = HERE / row["replica_paths"].split(";")[0]
    identity = dict(
        protocol=protocol,
        pdb=digest(pdb),
        trajectory=digest(trajectory),
        code=code_identity(),
    )
    marker = folder / "provenance.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Contact-feature provenance changed; use a fresh output directory"
            )
        if all(
            digest(folder / name) == value for name, value in saved["artifacts"].items()
        ):
            return folder
    executable = Path(sys.executable).parent / "jaxent-featurise"
    command = [
        str(executable),
        "--top_path",
        str(pdb),
        "--trajectory_path",
        str(trajectory),
        "--output_dir",
        str(folder),
        "--name",
        f"omc_decoy_{sid}",
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
        *map(str, protocol["residue_ignore"]),
        "--contact_mode",
        protocol["contact_mode"],
        "--switch_scale_nc",
        str(protocol["switch_scale_nc_angstrom"]),
        "--switch_scale_nh",
        str(protocol["switch_scale_nh_angstrom"]),
        "--mda_contact_environment",
        protocol["contact_environment"],
    ]
    print(
        f"Featurising {sid}: {protocol['contact_mode']}, scale 0.5 Angstrom", flush=True
    )
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
    }
    with (folder / "featurise.log").open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    if result.returncode:
        raise RuntimeError(f"Featurisation failed; see {folder / 'featurise.log'}")
    write_json(
        marker,
        dict(
            identity=identity,
            command=command,
            artifacts={
                name: digest(folder / name)
                for name in ("features.npz", "topology.json")
            },
        ),
    )
    return folder


def feature_audit(sid, config, frames, feature_folder=None):
    """Check stage1 cache against independent pair sums and full-sequence kints.

    This preserves the existing configured contact definition. A mismatch blocks
    fitting and is reported; it never triggers a silent contact-model replacement.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
    from MDAnalysis.lib.util import convert_aa_code
    from jaxent.src.models.func.uptake import calculate_HDXrate_from_sequence
    from jaxent.src.interfaces.topology.mda_adapter import mda_TopologyAdapter

    row = next(r for r in load_systems() if r["system_id"] == sid)
    folder = feature_folder or HERE / f"outputs/stage1/{sid}/R1"
    paths = [
        folder / "features.npz",
        folder / "topology.json",
        HERE / row["pdb_path"],
        HERE / row["replica_paths"].split(";")[0],
    ]
    data = load_npz(paths[0])
    topologies = json.loads(paths[1].read_text())["topologies"]
    u = mda.Universe(str(paths[2]), str(paths[3]))
    protein = u.select_atoms("protein").residues
    # These single-chain ATLAS systems have unique simulated residue numbers.
    if len(set(protein.resids)) != len(protein):
        raise ValueError("Ambiguous simulated residue numbering")
    lookup = {int(r.resid): r for r in protein}
    residues = [lookup[int(t["residues"][0])] for t in topologies]
    n_atoms = u.atoms[[r.atoms.select_atoms("name N")[0].index for r in residues]]
    h_atoms = u.atoms[
        [r.atoms.select_atoms("name H or name HN")[0].index for r in residues]
    ]
    protocol = config["protocol"]
    chain = {r.resindex: str(mda_TopologyAdapter._get_chain_id(r)) for r in protein}
    ordinal = {}
    for key in sorted(set(chain.values())):
        ordinal.update(
            {
                r.resindex: i
                for i, r in enumerate([r for r in protein if chain[r.resindex] == key])
            }
        )
    sequence = "".join(convert_aa_code(r.resname) for r in protein)
    if len(set(chain.values())) != 1:
        raise ValueError("This initial pilot requires one protein chain")
    raw_rates = calculate_HDXrate_from_sequence(
        sequence, protocol["temperature_k"], 7.0, unit="s^-1"
    )
    expected_rates = raw_rates[[ordinal[r.resindex] for r in residues]]
    audit_frames = np.unique(np.asarray(frames)[[0, len(frames) // 2, -1]])
    records = []
    for kind, targets, atom_selection, radius, scale in [
        (
            "heavy_contacts",
            n_atoms,
            "not type H",
            protocol["heavy_midpoint_angstrom"],
            protocol["switch_scale_nc_angstrom"],
        ),
        (
            "acceptor_contacts",
            h_atoms,
            "type O",
            protocol["acceptor_midpoint_angstrom"],
            protocol["switch_scale_nh_angstrom"],
        ),
    ]:
        environment = u.select_atoms(protocol["contact_environment"]).select_atoms(
            atom_selection
        )
        ignored = np.array(
            [
                [
                    a.resindex in ordinal
                    and chain[a.resindex] == chain[r.resindex]
                    and protocol["residue_ignore"][0]
                    <= ordinal[a.resindex] - ordinal[r.resindex]
                    <= protocol["residue_ignore"][1]
                    for a in environment
                ]
                for r in residues
            ]
        )
        for frame in audit_frames:
            u.trajectory[int(frame)]
            distance = distance_array(
                targets.positions,
                environment.positions,
                box=u.dimensions,
                backend="serial",
            )
            mode = protocol["contact_mode"]
            if mode == "bradshaw_switch":
                contribution = 1 / (1 + ((distance - radius) / scale) ** 6)
            elif mode == "smooth_cutoff":
                contribution = np.ones_like(distance)
                beyond = distance > radius
                contribution[beyond] = 1 / (
                    1 + ((distance[beyond] - radius) / scale) ** 6
                )
            elif mode == "hard":
                contribution = (distance <= radius).astype(float)
            elif mode == "legacy_switch":
                contribution = (distance <= radius) / (1 + (distance / radius) ** 6)
            else:
                raise ValueError(f"Unsupported configured contact mode: {mode}")
            contribution[ignored] = 0
            expected = contribution.sum(axis=1)
            cached = data[kind][:, frame]
            records.append(
                dict(
                    system=sid,
                    feature=kind,
                    frame=int(frame),
                    max_abs_error=float(np.max(abs(cached - expected))),
                    passed=bool(np.allclose(cached, expected, rtol=2e-5, atol=2e-4)),
                )
            )
    ca = u.select_atoms("protein and name CA")
    xyz = np.stack([ca.positions.copy() for _ in u.trajectory[frames]])
    rates = np.asarray(data["k_ints"], float)
    audit = dict(
        system=sid,
        contact_passed=all(r["passed"] for r in records),
        rates_passed=bool(np.allclose(rates, expected_rates, rtol=2e-5, atol=1e-8)),
        rates_max_abs_error=float(np.max(abs(rates - expected_rates))),
        contacts=records,
        input_hashes={str(p): digest(p) for p in paths},
    )
    return dict(
        log_pf=protocol["bv_bc"] * data["heavy_contacts"][:, frames].astype(float)
        + protocol["bv_bh"] * data["acceptor_contacts"][:, frames].astype(float),
        rates=rates,
        residues=np.array([r.resid for r in residues]),
        xyz=xyz,
        heavy=data["heavy_contacts"][:, frames],
        acceptor=data["acceptor_contacts"][:, frames],
    ), audit


def code_identity():
    paths = [
        Path(__file__),
        HERE / "config.yaml",
        HERE / "omc_decoy_config.yaml",
        HERE.parents[1] / "src/models/config.py",
        HERE.parents[1] / "src/models/HDX/BV/forwardmodel.py",
        HERE.parents[1] / "cli/featurise.py",
        HERE / "analysis/common.py",
        HERE.parents[1] / "src/models/func/contacts.py",
        HERE.parents[1] / "src/models/func/uptake.py",
        HERE.parents[1] / "src/models/HDX/forward.py",
        HERE / "featurise_batch.py",
    ]
    return {str(p): digest(p) for p in paths}


def load_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["code"] != code_identity():
        raise ValueError("Experiment code/config changed; use a fresh output directory")
    for p, expected in manifest["input_hashes"].items():
        if digest(p) != expected:
            raise ValueError(f"Frozen input changed: {p}")
    for p, expected in manifest["artifacts"].items():
        if digest(output / p) != expected:
            raise ValueError(f"Prepared artifact changed: {p}")
    return manifest


def prepare(output):
    output = Path(output)
    if (output / "manifest.json").exists():
        return load_manifest(output)
    config = experiment_config()
    previous = (
        HERE
        / "outputs/analysis/pairwise_geometry/omc_bandwidth_control/systems/1tzw_A/source.npz"
    )
    indices = load_npz(previous)["indices"]
    with np.load(HERE / "outputs/stage1/1tzw_A/R1/features.npz") as data:
        n_frames = data["heavy_contacts"].shape[1]
    post = post_equilibration_indices(
        n_frames,
        config["analysis"]["equilibration_ns"],
        config["analysis"]["frame_interval_ns"],
    )
    frames = post[indices]
    recipient_folder = prepare_features("1tzw_A", config, output)
    donor_folder = prepare_features("2w86_A", config, output)
    print("Auditing fresh recipient and donor features", flush=True)
    source, audit = feature_audit("1tzw_A", config, frames, recipient_folder)
    with np.load(HERE / "outputs/stage1/2w86_A/R1/features.npz") as data:
        donor_post = post_equilibration_indices(
            data["heavy_contacts"].shape[1],
            config["analysis"]["equilibration_ns"],
            config["analysis"]["frame_interval_ns"],
        )
    donor_frames = donor_post[
        np.linspace(0, len(donor_post) - 1, min(512, len(donor_post)), dtype=int)
    ]
    donor, donor_audit = feature_audit("2w86_A", config, donor_frames, donor_folder)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "feature_audit.json", dict(recipient=audit, donor=donor_audit))
    vectors = np.stack([pdist(frame) for frame in source["xyz"]])
    print("Selecting 100 representatives and independent target states", flush=True)
    reps, candidate_labels = partition_candidates(vectors)
    table, labels = partition_targets(vectors)
    table.to_csv(output / "target_clustering.csv", index=False)
    if labels is None or set(labels[reps]) != set(labels):
        raise ValueError(
            "No eligible, fully represented target partition; see target_clustering.csv"
        )
    native_labels = labels[reps]
    distance = cdist(vectors, vectors[reps]) / np.sqrt(vectors.shape[1])
    mixing = pd.crosstab(
        pd.Series(candidate_labels, name="candidate_cluster"),
        pd.Series(labels, name="target_state"),
    )
    mixing.to_csv(output / "candidate_target_mixing.csv")
    core = source["log_pf"][:, reps]
    _, sigmas = graph(core)
    target_specs = [
        empirical_target(labels, removed)
        for removed in [-1, *range(int(labels.max()) + 1)]
    ]
    preflight, predictions = physical_check(
        source["log_pf"],
        source["rates"],
        [w for w, _ in target_specs],
        audit["contact_passed"] and donor_audit["contact_passed"],
        audit["rates_passed"] and donor_audit["rates_passed"],
    )
    preflight["target_names"] = [
        "full_reference",
        *[f"remove_state_{i}" for i in range(len(target_specs) - 1)],
    ]
    preflight["contact_method"] = config["protocol"]
    write_json(output / "preflight.json", preflight)
    save_npz(
        output / "source.npz",
        **source,
        source_frames=frames,
        candidate_indices=reps,
        candidate_frames=frames[reps],
        candidate_labels=candidate_labels,
        target_labels=labels,
        native_labels=native_labels,
        structural_distance=distance,
        sigmas=sigmas,
        target_predictions=predictions,
        times=TIMES,
    )
    save_npz(
        output / "donor.npz",
        log_pf=donor["log_pf"],
        source_frames=donor_frames,
        residues=donor["residues"],
    )
    scale = float(np.var(predictions[0]) + 1e-8)
    cases = []
    for index, (weights, masses) in enumerate(target_specs):
        cases.append(
            dict(
                name="baseline" if index == 0 else f"internal_{index - 1}",
                kind="baseline" if index == 0 else "internal",
                removed=index - 1,
                seed=SEEDS[0],
                values=core,
                target_weights=weights,
                target_masses=masses,
            )
        )
    for seed in SEEDS:
        random, sampled = random_decoys(source["log_pf"], 25, seed)
        external, provenance = donor_decoys(donor["log_pf"], core.shape[0], 25, seed)
        for kind, decoys, origin in [
            ("random", random, dict(sampled_source_frames=frames[sampled])),
            (
                "donor",
                external,
                {**provenance, "source_frames": donor_frames[provenance["frame"]]},
            ),
        ]:
            name = f"{kind}_{seed}"
            save_npz(output / "cases" / name / "decoy_provenance.npz", **origin)
            cases.append(
                dict(
                    name=name,
                    kind=kind,
                    removed=-1,
                    seed=seed,
                    values=np.column_stack([core, decoys]),
                    target_weights=target_specs[0][0],
                    target_masses=target_specs[0][1],
                )
            )
    summaries = []
    for case in cases:
        values = case.pop("values")
        w = case.pop("target_weights")
        masses = case.pop("target_masses")
        target = uptake(source["log_pf"], source["rates"], w)
        kernels, _ = graph(values, sigmas)
        supported = np.flatnonzero(native_labels != case["removed"])
        # Diagnostic quadrature: project retained reference mass to supported representatives.
        nearest = supported[np.argmin(distance[:, supported], axis=1)]
        projection = np.bincount(nearest, weights=w, minlength=values.shape[1])
        decoy = np.r_[
            native_labels == case["removed"], np.ones(values.shape[1] - 100, bool)
        ]
        save_npz(
            output / "cases" / case["name"] / "input.npz",
            log_pf=values,
            target=target,
            rates=source["rates"],
            times=TIMES,
            kernels=kernels,
            sigmas=sigmas,
            target_weights=w,
            target_masses=masses,
            native_labels=native_labels,
            decoy=decoy,
            projection=projection,
            scale=np.array(scale),
        )
        case["informative"] = bool(
            preflight["passed"]
            and (
                case["removed"] < 0
                or preflight["target_max_abs_change"][case["removed"] + 1]
                > SIGNAL_FLOOR
            )
        )
        summaries.append(case)
    write_json(output / "cases.json", summaries)
    prepared_paths = [
        output / name
        for name in (
            "source.npz",
            "donor.npz",
            "feature_audit.json",
            "preflight.json",
            "target_clustering.csv",
            "candidate_target_mixing.csv",
            "cases.json",
        )
    ]
    prepared_paths += list((output / "features").glob("*/R1/provenance.json"))
    prepared_paths += list((output / "cases").glob("*/input.npz"))
    prepared_paths += list((output / "cases").glob("*/decoy_provenance.npz"))
    artifacts = {str(p.relative_to(output)): digest(p) for p in prepared_paths}
    manifest = dict(
        version=2,
        system="1tzw_A",
        donor="2w86_A",
        candidate_count=100,
        external_count=25,
        seed=SEEDS[0],
        external_seeds=list(SEEDS),
        times_seconds=TIMES.tolist(),
        frame_averaging="log_pf",
        intrinsic_rates_unit="s^-1",
        pD=7.0,
        noise=0.0,
        checkpoints=list(CHECKPOINTS),
        data_scale=scale,
        cases=summaries,
        fit_gate_passed=preflight["passed"],
        code=code_identity(),
        artifacts=artifacts,
        software={
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "jax",
                "optax",
                "MDAnalysis",
                "hdxrate",
            )
        },
        input_hashes={
            **audit["input_hashes"],
            **donor_audit["input_hashes"],
            str(previous): digest(previous),
            str(HERE / "data/systems.csv"): digest(HERE / "data/systems.csv"),
        },
    )
    write_json(output / "manifest.json", manifest)
    print(
        f"Prepared {len(cases)} cases; physical preflight passed={preflight['passed']}",
        flush=True,
    )
    return manifest


def arm_specs():
    return (
        [dict(family="omc", strength=0.1, quantile=float(q)) for q in QUANTILES]
        + [
            dict(family="maxent", strength=s, quantile=None)
            for s in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
        ]
        + [dict(family="unregularised", strength=0.0, quantile=None)]
    )


def jax_objective(logits, log_pf, rates, target, times, scale, kernel, strength, kind):
    import jax
    import jax.numpy as jnp

    w = jax.nn.softmax(logits)
    prediction = -jnp.expm1(-times[:, None] * rates[None, :] * jnp.exp(-(log_pf @ w)))
    mse = jnp.mean((prediction - target) ** 2) / scale
    c = w - 1 / len(w)
    raw = len(w) ** 2 * (
        jnp.dot(w * c**2, kernel @ w) - jnp.dot(w * c, kernel @ (w * c))
    )
    entropy = jnp.mean(-jnp.log(len(w)) - jax.nn.log_softmax(logits))
    return mse + strength * jnp.where(kind == 0, jnp.maximum(raw, 0.0), entropy)


def fit_case(data, checkpoints=CHECKPOINTS, window=250):
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)
    specs = arm_specs()
    count = data["log_pf"].shape[1]
    kernels = np.concatenate([data["kernels"], np.zeros((7, count, count))])
    kinds = np.array([0] * 6 + [1] * 7)
    strengths = np.array([s["strength"] for s in specs])
    if not checkpoints or any(
        b - a < window for a, b in zip((0, *checkpoints), checkpoints)
    ):
        raise ValueError("Checkpoint intervals must contain the convergence window")

    def losses(logits):
        return jax.vmap(
            lambda x, k, s, kind: jax_objective(
                x,
                jnp.asarray(data["log_pf"]),
                jnp.asarray(data["rates"]),
                jnp.asarray(data["target"]),
                jnp.asarray(data["times"]),
                data["scale"],
                k,
                s,
                kind,
            )
        )(logits, jnp.asarray(kernels), jnp.asarray(strengths), jnp.asarray(kinds))

    optimiser = optax.adam(0.05)
    results = []
    for initial in [
        np.zeros((13, count)),
        np.random.default_rng(SEEDS[0]).normal(0, 0.01, (13, count)),
    ]:
        x = jnp.asarray(initial)
        state = optimiser.init(x)
        active = np.ones(13, bool)
        steps = np.zeros(13, int)

        @jax.jit
        def advance(x, state, enabled, count_steps):
            def step(_, carry):
                x, state = carry
                updates, state = optimiser.update(
                    jax.grad(lambda z: losses(z).sum())(x), state, x
                )
                return jnp.where(
                    enabled[:, None], optax.apply_updates(x, updates), x
                ), state

            return jax.lax.fori_loop(0, count_steps, step, (x, state))

        previous = 0
        relative = np.full(13, np.inf)
        for checkpoint in checkpoints:
            x, state = advance(
                x, state, jnp.asarray(active), checkpoint - previous - window
            )
            before = np.asarray(losses(x))
            x, state = advance(x, state, jnp.asarray(active), window)
            after = np.asarray(losses(x))
            relative = np.where(
                active, abs(after - before) / np.maximum(abs(after), 1e-12), relative
            )
            steps[active] = checkpoint
            active &= (relative > 0.01) | ~np.isfinite(relative)
            previous = checkpoint
        results.append(
            dict(
                weights=np.asarray(jax.nn.softmax(x)),
                objective=np.asarray(losses(x)),
                relative_change=relative,
                steps=steps,
                grad_norm=np.asarray(
                    jnp.linalg.norm(jax.grad(lambda z: losses(z).sum())(x), axis=1)
                ),
            )
        )
    objectives = np.stack([r["objective"] for r in results])
    best = np.argmin(objectives, axis=0)
    selected = {
        key: np.stack([r[key] for r in results])[best, np.arange(13)]
        for key in results[0]
    }
    selected["initialisation_objectives"] = objectives
    selected["initialisation_weights"] = np.stack([r["weights"] for r in results])
    selected["initialisation_relative_change"] = np.stack(
        [r["relative_change"] for r in results]
    )
    selected["initialisation_objective_gap"] = abs(
        objectives[0] - objectives[1]
    ) / np.maximum(abs(selected["objective"]), 1e-12)
    selected["initialisation_weight_tv"] = 0.5 * abs(
        results[0]["weights"] - results[1]["weights"]
    ).sum(axis=1)
    selected["converged"] = (
        (selected["relative_change"] <= 0.01)
        & (selected["initialisation_objective_gap"] <= 0.01)
        & (selected["initialisation_relative_change"] <= 0.01).all(axis=0)
        & np.isfinite(selected["weights"]).all(axis=1)
        & np.isfinite(selected["objective"])
        & np.isfinite(selected["grad_norm"])
    )
    return selected


def worker_init():
    # Limit JAX's CPU thread pools as well as BLAS/OpenMP pools in each process.
    if hasattr(os, "sched_getaffinity"):
        slots = sorted(os.sched_getaffinity(0))
        identity = multiprocessing.current_process()._identity
        start = 2 * ((identity[-1] - 1) if identity else 0) % len(slots)
        os.sched_setaffinity(0, {slots[start], slots[(start + 1) % len(slots)]})


def run_worker(args):
    folder = Path(args[0]) / "cases" / args[1]
    with (folder / ".fit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_worker_locked(args)


def _run_worker_locked(args):
    output, name, identity = args
    output = Path(output)
    folder = output / "cases" / name
    marker = folder / "complete.json"
    if marker.exists():
        complete = json.loads(marker.read_text())
        if (
            complete["identity"] == identity
            and digest(folder / "fit.npz") == complete["sha256"]
        ):
            return name, "resumed"
    data = load_npz(folder / "input.npz")
    result = fit_case(data)
    if not np.isfinite(result["weights"]).all() or not np.allclose(
        result["weights"].sum(axis=1), 1, atol=1e-10
    ):
        raise ValueError(f"Invalid weights for {name}")
    save_npz(folder / "fit.npz", **result)
    write_json(marker, dict(identity=identity, sha256=digest(folder / "fit.npz")))
    return name, "complete"


def run(output, workers=10, limit=None):
    manifest = load_manifest(output)
    if not manifest["fit_gate_passed"]:
        write_json(
            output / "run_status.json",
            dict(
                status="blocked_physical_preflight",
                fitted_cases=0,
                reasons=json.loads((output / "preflight.json").read_text())["reasons"],
            ),
        )
        print(
            "No fits started: physical preflight failed. Generating diagnostic report.",
            flush=True,
        )
        return
    cases = [c for c in manifest["cases"] if c["informative"]]
    cases = cases[:limit] if limit is not None else cases
    jobs = [(str(output), c["name"], digest(output / "manifest.json")) for c in cases]
    os.environ.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_PLATFORMS="cpu",
    )
    if workers == 1:
        completed = list(map(run_worker, jobs))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
        ) as pool:
            completed = list(pool.map(run_worker, jobs))
    for row in completed:
        print(*row, flush=True)
    write_json(
        output / "run_status.json",
        dict(
            status="complete" if limit is None else "partial",
            fitted_cases=len(completed),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["prepare", "run", "report", "all"], default="all"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit fitting cases for a smoke run; preparation remains complete",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.phase in ("prepare", "all"):
        prepare(args.output)
    if args.phase in ("run", "all"):
        run(args.output, args.workers, args.limit)
    # Preparation always provides a visible report, including failed preflight.
    from .omc_decoy_report import report

    report(args.output)


if __name__ == "__main__":
    main()
