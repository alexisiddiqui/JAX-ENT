#!/usr/bin/env python3
"""Generate reproducible, first-party JAX-ENT IsoValidation HDX targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np
import pandas as pd

import jaxent.src.interfaces.topology as pt
from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_fast,
    residue_uptake_frame,
    residue_uptake_legacy,
    residue_uptake_slow2,
    weights_from_cluster_populations,
)
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.HDX.forward import BV_uptake_ForwardPass
from jaxent.src.models.config import BV_model_Config
from jaxent.src.utils.jax_fn import frame_average_features

TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0])
HEADER = "#\t0.167\t1.0\t10.0\t60.0\t120.0\t times/min\n"
LABELS = {"open": 0, "closed": 1, "intermediate": -1}
LEGACY_MODES = {"legacy": "log_pf", "fast": "rate", "slow2": "uptake"}
MODES = ("log_pf", "rate", "uptake", "frame_uptake")


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
    """Aggregate represented amides into non-overlapping fixed-width peptides."""
    residue_uptake = np.asarray(residue_uptake)
    residue_segments = np.asarray(residue_segments, dtype=int)
    if residue_uptake.shape[1] != len(residue_segments):
        raise ValueError("one segment row is required per residue-uptake column")
    if width < 2:
        raise ValueError("peptide width must be at least 2")
    terminal = 310
    starts = list(range(1, terminal + 1, width - 1))
    ends = list(range(width, terminal + 1, width - 1))
    starts.pop(-1)
    ends.pop(-1)
    ends.append(terminal)
    values, segments = [], []
    residue_ids = residue_segments[:, 1]
    for start, end in zip(starts, ends, strict=True):
        mask = (residue_ids >= start + 1) & (residue_ids <= end)
        if np.any(mask):
            values.append(residue_uptake[:, mask].mean(axis=1))
            segments.append((start, end))
    return np.stack(values, axis=1), np.asarray(segments, dtype=int)


def aggregate_segment_map(
    residue_uptake: np.ndarray, residue_segments: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Average represented exchangeable residues for an arbitrary peptide map."""
    residue_ids = np.asarray(residue_segments, dtype=int)[:, 1]
    values, kept = [], []
    for start, stop in np.asarray(segments, dtype=int):
        mask = (residue_ids >= start + 1) & (residue_ids <= stop)
        if np.any(mask):
            values.append(np.asarray(residue_uptake)[:, mask].mean(axis=1))
            kept.append((start, stop))
    if not values:
        raise ValueError("peptide map contains no represented exchangeable residues")
    return np.stack(values, axis=1), np.asarray(kept, dtype=int)


def align_residue_layout(
    uptake: np.ndarray, topology_path: Path, shipped_segments_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy alignment retained for old phase-matrix commands."""
    topology = json.loads(topology_path.read_text())
    topologies = topology["topologies"]
    topology_residues = np.asarray([item["residues"][0] for item in topologies], dtype=int)
    segments = np.loadtxt(shipped_segments_path, dtype=int)
    if uptake.shape[1] != len(topologies):
        raise ValueError("feature uptake and topology lengths differ")
    if len(segments) != len(topologies) + 1 or not np.array_equal(
        topology_residues, segments[:-1, 1]
    ):
        raise ValueError("feature topology cannot be aligned to the legacy layout")
    terminal = np.ones((uptake.shape[0], 1), dtype=uptake.dtype)
    return np.concatenate((uptake, terminal), axis=1), segments


def write_target(output_dir: Path, uptake: np.ndarray, segments: np.ndarray) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dfrac_path = output_dir / "target_dfrac.dat"
    segs_path = output_dir / "target_segs.txt"
    with dfrac_path.open("w") as handle:
        handle.write(HEADER)
        np.savetxt(handle, np.asarray(uptake).T, delimiter="\t", fmt="%.8f")
    np.savetxt(segs_path, segments, fmt="%d", delimiter=" ")
    return dfrac_path, segs_path


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_features(topology_path, open_path, closed_path, cache_dir, temperature, pd_value):
    topology_path, open_path, closed_path, cache_dir = (
        Path(topology_path).resolve(),
        Path(open_path).resolve(),
        Path(closed_path).resolve(),
        Path(cache_dir).resolve(),
    )
    inputs = {str(path): sha256(path) for path in (topology_path, open_path, closed_path)}
    expected = {
        "input_hashes": inputs,
        "temperature_k": temperature,
        "effective_pd": pd_value,
        "kint_unit": "min^-1",
        "contact_mode": "hard",
        "sequence_normalization": "standard_protonation_aliases_v1",
    }
    feature_path = cache_dir / "features_open_closed.npz"
    topology_json = cache_dir / "topology_open_closed.json"
    cache_manifest = cache_dir / "manifest.json"
    if feature_path.exists() and topology_json.exists() and cache_manifest.exists():
        metadata = json.loads(cache_manifest.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            features = BV_input_features.load(str(feature_path))
            topology = pt.PTSerialiser.load_list_from_json(str(topology_json))
            assignments = np.concatenate(
                (np.zeros(metadata["open_frames"], int), np.ones(metadata["closed_frames"], int))
            )
            return features, topology, assignments, metadata

    open_universe = mda.Universe(str(topology_path), str(open_path))
    closed_universe = mda.Universe(str(topology_path), str(closed_path))
    config = BV_model_Config(
        timepoints=jnp.asarray(TIMEPOINTS), contact_mode="hard", kint_unit="min^-1"
    )
    config.temperature, config.ph = temperature, pd_value
    builder = Experiment_Builder(
        universes=[open_universe, closed_universe], forward_models=[BV_model(config)]
    )
    feature_sets, topology_sets = run_featurise(
        builder, FeaturiserSettings(name="iso_target_open_closed", batch_size=None)
    )
    features, topology = feature_sets[0], topology_sets[0]
    n_open = open_universe.trajectory.n_frames
    n_closed = closed_universe.trajectory.n_frames
    assignments = np.concatenate((np.zeros(n_open, int), np.ones(n_closed, int)))
    if features.features_shape[1] != len(assignments):
        raise ValueError("featurised frame count does not match source trajectories")
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.save(str(feature_path))
    pt.PTSerialiser.save_list_to_json(topology, str(topology_json))
    metadata = {**expected, "open_frames": n_open, "closed_frames": n_closed}
    cache_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    return features, topology, assignments, metadata


def _predict(features, weights, assignments, mode, temperature, pd_value):
    config = BV_model_Config(
        timepoints=jnp.asarray(TIMEPOINTS), contact_mode="hard", kint_unit="min^-1"
    )
    config.temperature, config.ph = temperature, pd_value
    forward = BV_uptake_ForwardPass(mode, assignments if mode == "uptake" else None)
    if mode == "log_pf":
        output = forward(frame_average_features(features, jnp.asarray(weights)), config.forward_parameters)
    else:
        output = forward.average_frames(features, config.forward_parameters, jnp.asarray(weights))
    return np.asarray(output.uptake)


def _legacy_target(args, example_root: Path, mode: str) -> None:
    if not args.ensemble:
        raise ValueError("--ensemble is required with deprecated --semantics")
    fitting_root = example_root / "fitting/jaxENT"
    suffix = args.ensemble.removeprefix("iso_")
    feature_path = fitting_root / f"_featurise/features_{args.ensemble}.npz"
    topology_path = fitting_root / f"_featurise/topology_{args.ensemble}.json"
    assignment_path = example_root / f"data/_clustering_results/cluster_assignments_ISO_{suffix.upper()}.csv"
    shipped = example_root / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    features = BV_input_features.load(str(feature_path))
    log_pf = 0.35 * np.asarray(features.heavy_contacts) + 2.0 * np.asarray(features.acceptor_contacts)
    assignments = pd.read_csv(assignment_path)["cluster_assignment"].to_numpy(dtype=int)
    populations = {LABELS[name]: value for name, value in args.population.items()}
    for label in np.unique(assignments):
        populations.setdefault(int(label), 0.0)
    weights = weights_from_cluster_populations(assignments, populations)
    functions = {"log_pf": residue_uptake_legacy, "rate": residue_uptake_fast, "uptake": residue_uptake_slow2, "frame_uptake": residue_uptake_frame}
    values = (log_pf, np.asarray(features.k_ints), TIMEPOINTS, weights)
    uptake = functions[mode](*values, assignments, args.tau) if mode == "uptake" else functions[mode](*values, args.tau)
    uptake, segments = align_residue_layout(uptake, topology_path, shipped)
    if args.peptide_width:
        uptake, segments = peptide_aggregate(uptake, segments, args.peptide_width)
    dfrac, segs = write_target(args.output_dir, uptake, segments)
    (args.output_dir / "manifest.json").write_text(json.dumps({"legacy": True, "averaging_mode": mode, "populations": args.population, "outputs": {"dfrac": dfrac.name, "segments": segs.name}}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--averaging-mode", choices=MODES, default="frame_uptake")
    parser.add_argument("--semantics", choices=tuple(LEGACY_MODES), help="Deprecated legacy alias")
    parser.add_argument("--ensemble", choices=["iso_tri", "iso_bi"], help="Required only by legacy mode")
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--population", type=parse_population, default=parse_population("open=0.4,closed=0.6"))
    parser.add_argument("--layouts", default="residue,width10")
    parser.add_argument("--peptide-width", type=int, default=None, help="Deprecated single-layout output")
    parser.add_argument(
        "--peptide-segments",
        type=Path,
        help="Optional arbitrary peptide segment map; writes an additional pepsin layout.",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pd", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    args = parser.parse_args()
    if args.tau != 0:
        parser.error("the production JAX-ENT target currently supports EX2 (tau=0) only")
    if args.temperature <= 0 or not np.isfinite(args.pd):
        parser.error("temperature must be positive and pD finite")

    example_root = Path(__file__).resolve().parents[1]
    repo_root = example_root.parents[2]
    mode = LEGACY_MODES[args.semantics] if args.semantics else args.averaging_mode
    if args.semantics:
        _legacy_target(args, example_root, mode)
        return

    trajectory_root = example_root / "data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    topology_path = trajectory_root / "TeaA_ref_closed_state.pdb"
    open_path = trajectory_root / "TeaA_open_reimaged.xtc"
    closed_path = trajectory_root / "TeaA_closed_reimaged.xtc"
    cache_dir = args.feature_cache_dir or example_root / "data/_self_consistent_target_features"
    features, topology, assignments, source_metadata = _source_features(
        topology_path, open_path, closed_path, cache_dir, args.temperature, args.pd
    )
    populations = {LABELS[name]: value for name, value in args.population.items()}
    for label in np.unique(assignments):
        populations.setdefault(int(label), 0.0)
    weights = weights_from_cluster_populations(assignments, populations)
    uptake = _predict(features, weights, assignments, mode, args.temperature, args.pd)
    residue_ids = np.asarray(
        [top._get_active_residues(check_trim=False)[0] for top in topology], dtype=int
    )
    residue_segments = np.column_stack((residue_ids - 1, residue_ids))

    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    if set(layouts) - {"residue", "width10"} or not layouts:
        parser.error("--layouts must contain residue and/or width10")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for layout in layouts:
        layout_uptake, segments = uptake, residue_segments
        if layout == "width10":
            layout_uptake, segments = peptide_aggregate(uptake, residue_segments, 10)
        dfrac, segs = write_target(args.output_dir / layout, layout_uptake, segments)
        outputs[layout] = {
            "dfrac": str(dfrac.relative_to(args.output_dir)),
            "segments": str(segs.relative_to(args.output_dir)),
            "n_observations": len(segments),
        }
    if args.peptide_segments is not None:
        peptide_segments = np.loadtxt(args.peptide_segments, dtype=int, ndmin=2)
        peptide_uptake, peptide_segments = aggregate_segment_map(
            uptake, residue_segments, peptide_segments
        )
        dfrac, segs = write_target(
            args.output_dir / "pepsin", peptide_uptake, peptide_segments
        )
        outputs["pepsin"] = {
            "dfrac": str(dfrac.relative_to(args.output_dir)),
            "segments": str(segs.relative_to(args.output_dir)),
            "source_segments": str(args.peptide_segments.resolve()),
            "source_segments_sha256": sha256(args.peptide_segments),
            "n_observations": len(peptide_segments),
        }
    np.save(args.output_dir / "target_frame_weights.npy", weights)
    np.save(args.output_dir / "target_state_labels.npy", assignments)
    manifest = {
        "schema_version": 1,
        "averaging_mode": mode,
        "default_averaging_mode": "frame_uptake",
        "populations": args.population,
        "timepoints_min": TIMEPOINTS.tolist(),
        "bv": {"bc": 0.35, "bh": 2.0, "contact_mode": "hard", "temperature_k": args.temperature, "effective_pd": args.pd, "kint_unit": "min^-1"},
        "source": source_metadata,
        "outputs": outputs,
        "git_commit": _git_commit(repo_root),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
