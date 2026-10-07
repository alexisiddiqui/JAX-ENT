#!/usr/bin/env python3
"""MSE-fitted full-uptake ISO SI: MaxEnt and median-scaled all-pairs OMC.

Run from the repository with .venv/bin/python, --phase all --jobs 8.
Historical campaigns remain untouched; both selectors use the existing saved-state candidate pool.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np
import pandas as pd

import analyze_within_trajectory_selection as original
import run_maxent_omc_sidecar as base
import run_maxent_omc_strength_sidecar as strength
from iso_sidecar_geometry import median_scale, pairwise_ca_rmsd
from sidecar_selection import closed_validation_mse, select_best_rows
from plot_iso_policy_sidecars import export_figures
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.examples.common.analysis.rescore_iso_coordinate_sigma import make_predict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
VERSION = "iso_policy_full_uptake_v1"
SPLITS = ("sequence_cluster", "spatial")
METRICS = ("work_scale", "rmsd", "pyrosetta")
SELECTORS = ("val_mse", "val_closed_sigma_mse")
MAXENT = {"ISO_BI": tuple(10.**i for i in range(-2, 7)),
          "ISO_TRI": tuple(10.**i for i in range(-2, 10))}
CAMPAIGN = HERE / "_maxent_weighted_mse_per_peptide_tri1e9_closed_selection_jobs8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


@dataclasses.dataclass(frozen=True)
class RunSpec(strength.RunSpec):
    graph_metric: str = "none"
    distance_path: str = ""
    reuse_history: str = ""
    reuse_config: str = ""

    @property
    def run_id(self):
        h = f"_h{base.shrinkage.alpha_token(self.kernel_bandwidth)}" if self.method == "omc" else ""
        return (f"{VERSION}_{self.ensemble}_{self.split_type}_{self.method}_{self.graph_metric}"
                f"{h}_S{base.shrinkage.alpha_token(self.sweep_maxent)}_split{self.split_idx:03d}")

    @property
    def curve(self):
        return "MaxEnt" if self.method == "maxent" else f"{self.graph_metric} h={self.kernel_bandwidth:g}"

    @property
    def saved_history(self):
        return Path(self.reuse_history) if self.reuse_history else self.history_path

    @property
    def saved_config(self):
        return Path(self.reuse_config) if self.reuse_config else self.config_path


def prepare_splits(args):
    target = args.output_dir / "datasplits"
    source = base.shrinkage.DEFAULT_DATASPLIT_DIR
    target.mkdir(exist_ok=True)
    for name in ("full_dataset_dfrac.csv", "full_dataset_topology.json"):
        dest = target / name
        if dest.exists() and digest(dest) != digest(source / name):
            raise ValueError("Self-consistent dataset changed")
        if not dest.exists():
            shutil.copyfile(source / name, dest)
    if not (target / "sequence_cluster").exists():
        shutil.copytree(source / "sequence_cluster", target / "sequence_cluster")
    spatial = target / "spatial"
    expected = [spatial / f"split_{i:03d}" / name for i in range(args.n_splits)
                for name in ("train_dfrac.csv", "train_topology.json", "val_dfrac.csv", "val_topology.json")]
    if not all(p.exists() for p in expected):
        _, topology = base.shrinkage.load_features(args.features_dir, "ISO_TRI")
        full = base.shrinkage.HDX_peptide.load_list_from_files(
            json_path=str(target/"full_dataset_topology.json"), csv_path=str(target/"full_dataset_dfrac.csv"))
        reference = mda.Universe(str(args.trajectory_dir/"TeaA_ref_closed_state.pdb"))
        for index in range(args.n_splits):
            splitter = DataSplitter(train_size=.5, dataset=ExpD_Dataloader(data=full),
                common_residues=set(topology), peptide_trim=1, centrality=False,
                check_trim=True, random_seed=42*index)
            # The target includes residue 310. The general spatial default
            # excludes termini and would leave that observation without atoms.
            train, val = splitter.spatial_split(universe=reference,
                remove_overlap=True, exclude_termini=False)
            part = spatial/f"split_{index:03d}"
            part.mkdir(parents=True, exist_ok=True)
            for name, data in (("train", train), ("val", val)):
                ExpD_Datapoint.save_list_to_files(data,
                    json_path=str(part/f"{name}_topology.json"), csv_path=str(part/f"{name}_dfrac.csv"))
    split_audit = []
    for split in SPLITS:
        for index in range(args.n_splits):
            train, val = base.shrinkage.load_split(target, split, index)
            # Preserve repository topology/trim mapping; check effective residues.
            def residues(data):
                return {r for point in data for r in tuple(point.top.residues)[point.top.peptide_trim:]}
            if residues(train) & residues(val):
                raise ValueError(f"Overlapping effective train/validation residues: {split}/{index}")
            split_audit.append(dict(split_type=split, split_idx=index, train=len(train), val=len(val),
                                     seed=42*index, train_fraction=.5,
                                     spatial_exclude_termini=False if split=="spatial" else None))
    write_json(args.output_dir/"split_audit.json", split_audit)
    return target


def prepare_distances(args):
    directory = args.output_dir / "geometry"
    directory.mkdir(exist_ok=True)
    topology = args.trajectory_dir / "TeaA_ref_closed_state.pdb"
    trajectory = args.trajectory_dir / base.source.TRAJECTORIES["ISO_TRI"]
    fingerprints = {str(p): digest(p) for p in (topology, trajectory,
                     args.features_dir/"features_iso_tri.npz", HERE/"iso_sidecar_geometry.py")}
    metadata_path = directory / "manifest.json"
    expected_paths = {metric: directory/f"{metric}.npz" for metric in METRICS}
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["input_sha256"] != fingerprints:
            raise ValueError("Geometry inputs/code changed; use a new output directory")
        bonded_scores = directory/"pyrosetta_scores_final.npz"
        if (bonded_scores.exists() and previous.get("pyro_scores_sha256") == digest(bonded_scores)
                and previous.get("pyro_scoring_code_sha256") == digest(HERE/"score_iso_pyrosetta.py")
                and all(path.exists() and digest(path) == previous["distances"][m]["sha256"]
                        for m, path in expected_paths.items())):
            return expected_paths
    features, _ = base.shrinkage.load_features(args.features_dir, "ISO_TRI")
    model = base.shrinkage.configure_model("uptake", np.empty(0, dtype=int))
    n = features.features_shape[1]
    distances = {"work_scale": base.work_distances(features, model)}
    mobile = mda.Universe(str(topology), str(trajectory))
    coords = np.stack([mobile.select_atoms("protein and name CA").positions.copy()
                      for _ in mobile.trajectory]).astype(float)
    if len(coords) != n:
        raise ValueError("Trajectory/features frame mismatch")
    print(f"Prepare all-pairs RMSD for {n} frames", flush=True)
    distances["rmsd"] = pairwise_ca_rmsd(coords)
    pyro_path = directory/"pyrosetta_scores_final.npz"
    pyro_metadata = pyro_path.with_suffix(".json")
    inputs = {str(p): digest(p) for p in (topology, trajectory)}
    if not (pyro_path.exists() and pyro_metadata.exists()
            and json.loads(pyro_metadata.read_text())["input_sha256"] == inputs
            and json.loads(pyro_metadata.read_text()).get("virtual_atom_policy") == "rebuild internal coordinates; real atoms unchanged"
            and json.loads(pyro_metadata.read_text()).get("scoring_code_sha256") == digest(HERE/"score_iso_pyrosetta.py")):
        with (directory/"pyrosetta.log").open("w") as log:
            subprocess.run([str(args.pyrosetta_python), str(HERE/"score_iso_pyrosetta.py"),
                "--topology", str(topology), "--trajectory", str(trajectory), "--output", str(pyro_path)],
                stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(pyro_path) as archive:
        np.testing.assert_array_equal(archive["frame"], np.arange(n))
        scores = archive["ref2015_total"]
        distances["pyrosetta"] = np.abs(scores[:, None] - scores[None, :])
    descriptions = {}
    for metric, matrix in distances.items():
        normalized, scale = median_scale(matrix)
        path = expected_paths[metric]
        np.savez_compressed(path, raw_distances=matrix, normalized_distances=normalized,
                            median_positive_distance=scale, frame=np.arange(n), metric=metric)
        descriptions[metric] = dict(path=str(path), sha256=digest(path), frames=n,
            median_positive_distance=scale,
            native_units={"work_scale": "work/RT", "rmsd": "Angstrom", "pyrosetta": "REU"}[metric])
    write_json(metadata_path, dict(input_sha256=fingerprints, distances=descriptions,
                                  pyro_scores_sha256=digest(pyro_path),
                                  pyro_scoring_code_sha256=digest(HERE/"score_iso_pyrosetta.py"),
                                  normalization="median positive upper-triangle distance"))
    return expected_paths


def reuse_index(args):
    if args.smoke:
        return {}
    specs = original.collect_specs(args.reuse_campaign)
    result = {}
    for spec in specs:
        if spec.sigma_source != "mse" or spec.uptake_mode != "uptake":
            continue
        if not spec.history_path.exists() or not spec.config_path.exists():
            continue
        cfg = json.loads(spec.config_path.read_text())
        opt = cfg["opt_config"]
        matches = all(opt.get(k) == v for k, v in dict(n_steps=args.n_steps, learning_rate=1.,
            ema_alpha=.5, forward_model_scaling=1000., optimizer="adam", step_chunk_size=100,
            lr_adjustment=True, frame_average_impl="tensordot").items())
        matches &= cfg["effective_settings"].get("frame_averaging_mode") == "frame_uptake"
        matches &= not cfg["loss_config"]["optimize_bv_params"]
        matches &= Path(spec.features_dir).resolve() == args.features_dir
        matches &= all(digest(Path(spec.datasplit_dir)/spec.split_type/f"split_{spec.split_idx:03d}"/name)
            == digest(args.datasplit_dir/spec.split_type/f"split_{spec.split_idx:03d}"/name)
            for name in ("train_dfrac.csv", "val_dfrac.csv", "train_topology.json", "val_topology.json"))
        if matches:
            result[spec.ensemble, spec.split_type, spec.split_idx, spec.sweep_maxent] = spec
    return result


def build_specs(args, distance_paths, reused=None):
    specs = []
    reused = {} if reused is None else reused
    for ensemble in ("ISO_BI", "ISO_TRI"):
        for split in SPLITS:
            for index in range(args.n_splits):
                common = dict(ensemble=ensemble, sigma_source="mse", alpha=0., split_type=split,
                    split_idx=index, sigma_path="", output_dir=str(args.output_dir),
                    features_dir=str(args.features_dir), datasplit_dir=str(args.datasplit_dir),
                    clustering_dir=str(args.clustering_dir), n_steps=args.n_steps,
                    learning_rate=1., ema_alpha=.5, forward_model_scaling=1000., execution_mode="compiled",
                    uptake_mode="uptake", graph_k=0, graph_path="")
                for value in ([10.] if args.smoke else MAXENT[ensemble]):
                    old = reused.get((ensemble, split, index, value))
                    specs.append(RunSpec(**common, method="maxent", sweep_maxent=value, omc_strength=0.,
                        reuse_history=str(old.history_path) if old else "",
                        reuse_config=str(old.config_path) if old else ""))
                if ensemble == "ISO_TRI":
                    for metric in METRICS:
                        for h in args.bandwidths:
                            for value in args.strengths:
                                specs.append(RunSpec(**common, method="omc", sweep_maxent=value,
                                    omc_strength=1/value, graph_metric=metric,
                                    kernel_bandwidth=h, distance_path=str(distance_paths[metric])))
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("Duplicate run identities")
    return specs


def completed(spec):
    if not spec.saved_history.exists() or not spec.saved_config.exists():
        return False
    try:
        config = json.loads(spec.saved_config.read_text())
        effective = config["effective_settings"]
        # Full-length controls showed that the accelerated cached/batched path
        # changes Adam trajectories and native convergence candidates. Replaying
        # its saved MSE is insufficient to establish fitting equivalence.
        if effective.get("execution_mode") == "native_strength_batch" or effective.get("frozen_forward_cache", False):
            return False
        if spec.method == "omc":
            if effective.get("omc_diagonal_policy") != "pairwise_exact_zero":
                return False
        history = base.source.load_optimization_history_from_file(str(spec.saved_history))
        return bool(history.states) and bool(np.isfinite(float(history.states[-1].losses.total_train_loss)))
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def run_fit(spec):
    if spec.config_path.exists():
        old = json.loads(spec.config_path.read_text())
        if old.get("effective_settings", {}).get("execution_mode") == "native_strength_batch":
            archive = Path(spec.output_dir)/"invalid_acceleration_archive"/"fits"/spec.split_type/spec.run_id
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise ValueError(f"Refusing to overwrite archived fit: {archive}")
            shutil.move(str(spec.run_dir), str(archive))
    with contextlib.redirect_stdout(io.StringIO()):
        base.run_fit(spec)
    config = json.loads(spec.config_path.read_text())
    if spec.method == "omc":
        config["effective_settings"]["omc_diagonal_policy"] = "pairwise_exact_zero"
    config["sidecar_settings"].update(training_loss="MSE", selection_metrics=list(SELECTORS),
        data_strength=spec.sweep_maxent, regularizer_coefficient=1/spec.sweep_maxent,
        graph_metric=spec.graph_metric, distance_normalization="median_positive_off_diagonal",
        sigma_shrinkage=0., kernel_normalization=False)
    write_json(spec.config_path, config)
    print(f"Completed {spec.run_id}", flush=True)


def execute(specs, args):
    pending = [s for s in specs if not completed(s)]
    logs, specdir = args.output_dir/"logs", args.output_dir/"specs"
    logs.mkdir(exist_ok=True)
    specdir.mkdir(exist_ok=True)
    cores = sorted(os.sched_getaffinity(0))
    # Bound each subprocess to two CPUs; no JAX oversubscription across workers.
    # Fit each trajectory through the original scalar optimizer, including the
    # native forward pass and direct pairwise OMC objective. Parallelism is only
    # across independent subprocesses, never across optimizer lanes.
    groups = [[spec] for spec in pending]
    def worker(group, slot):
        path = specdir/f"{group[0].run_id}__batch.json"
        write_json(path, dataclasses.asdict(group[0]))
        affinity = cores[2*slot:2*slot+2] or cores[:1]
        command = ["taskset", "-c", ",".join(map(str, affinity)), sys.executable,
                   str(Path(__file__).resolve()), "--worker-spec", str(path)]
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with (logs/f"{group[0].run_id}__batch.log").open("w") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    failures = []
    # One queue per slot avoids assigning the same cores to simultaneous jobs.
    def queue(slot):
        for group in groups[slot::args.jobs]:
            code = worker(group, slot)
            if code:
                failures.extend(s.run_id for s in group)
            print(f"Native fit {group[0].run_id}: exit {code}", flush=True)
    print(f"Fit grid: {len(specs)}, reused {sum(bool(s.reuse_history) for s in specs)}, pending {len(pending)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(queue, range(args.jobs)))
    if failures:
        raise RuntimeError(f"Failed fits: {failures}")


def analyze(specs, args):
    unfinished = [spec.run_id for spec in specs if not completed(spec)]
    if unfinished:
        raise RuntimeError(f"Refusing figure export before native fits are complete: {len(unfinished)} pending")
    cache, rows, audits, missing = {}, [], [], []
    shards = args.output_dir/"score_shards"
    shards.mkdir(exist_ok=True)
    scoring_code = digest(Path(__file__)) + digest(HERE/"sidecar_selection.py")
    for number, spec in enumerate(specs, 1):
        if not completed(spec):
            missing.append(dict(run_id=spec.run_id, reason="missing_or_nonfinite_fit"))
            continue
        cfg = json.loads(spec.saved_config.read_text())
        if cfg["effective_settings"]["frame_averaging_mode"] != "frame_uptake" or cfg["loss_config"]["optimize_bv_params"]:
            raise ValueError(f"Unexpected forward model or trainable BV: {spec.run_id}")
        key = spec.ensemble, spec.split_type, spec.split_idx
        if key not in cache:
            with contextlib.redirect_stdout(io.StringIO()):
                features, top = base.shrinkage.load_features(spec.features_dir, spec.ensemble)
                labels = base.shrinkage.load_cluster_assignments(spec.clustering_dir, spec.ensemble)
                model = base.shrinkage.configure_model("uptake", labels)
                train, val = base.shrinkage.load_split(spec.datasplit_dir, spec.split_type, spec.split_idx)
                loader = base.source.create_data_loaders(train+val, train, val, features, top)
            mapping = np.asarray(loader.val.residue_feature_ouput_mapping.todense())
            np.testing.assert_array_equal(mapping, np.eye(features.features_shape[0])[
                [p.top.fragment_index for p in val]])
            target = np.asarray(loader.val.y_true)[..., 0]
            forward = model.forward[base.shrinkage.m_key("HDX_peptide")]
            cache[key] = (features, labels, model, mapping, target, make_predict(forward, features))
        features, labels, model, mapping, target, predict = cache[key]
        signature = hashlib.sha256((digest(spec.saved_history)+scoring_code).encode()).hexdigest()
        shard = shards/f"{spec.run_id}.csv"
        auditpath = shards/f"{spec.run_id}.json"
        if shard.exists() and auditpath.exists() and json.loads(auditpath.read_text())["signature"] == signature:
            rows.extend(pd.read_csv(shard).to_dict("records"))
            audit = json.loads(auditpath.read_text())
            audits.append(audit)
            if not audit["candidates"]:
                missing.append(dict(run_id=spec.run_id, reason="no_saved_candidates"))
            continue
        history = base.source.load_optimization_history_from_file(str(spec.saved_history))
        convergence = {id(item.state): item for item in base.source.iter_labeled_convergence_states(history)}
        errors = []
        local = []
        # Reuse the same candidate enumeration as the earlier ISO comparison.
        # Both validation selectors see all saved trajectory/convergence/best
        # states, deduplicated by step and weights; no threshold-only filter.
        for candidate_index, (kind, state, weights) in enumerate(original.candidates(history)):
            labeled = convergence.get(id(state))
            parameters = state.params.model_parameters[0]
            for actual, expected in zip(jax.tree_util.tree_leaves(parameters), jax.tree_util.tree_leaves(model.params), strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
            predicted = mapping @ np.asarray(predict(jnp.asarray(weights), parameters)).T
            mse = float(np.mean((predicted-target)**2))
            native = float(state.losses.val_losses[0])
            np.testing.assert_allclose(.5*mse, native, rtol=3e-3, atol=2e-7,
                err_msg=f"Native fit loss replay: {spec.run_id}")
            errors.append(abs(.5*mse-native))
            sigma = closed_validation_mse(spec, predicted, target)
            ess = base.source.effective_sample_size(weights)
            local.append(dict(run_id=spec.run_id, ensemble=spec.ensemble, method=spec.method,
                graph_metric=spec.graph_metric, split_type=spec.split_type, split_idx=spec.split_idx,
                data_strength=spec.sweep_maxent, bandwidth=spec.kernel_bandwidth if spec.method=="omc" else 0.,
                training_loss="MSE", forward_model="frame_uptake", step=int(state.step),
                candidate_index=candidate_index, candidate_kind=kind,
                convergence_rank=labeled.rank if labeled else np.nan,
                convergence_threshold=labeled.threshold if labeled else np.nan,
                val_mse=mse, val_closed_sigma_mse=sigma, native_validation_loss=native,
                recovery_percent=base.source.calculate_recovery_percentage(labels, weights,
                    base.shrinkage.GROUND_TRUTH, base.shrinkage.STATE_MAPPING),
                ess_percent=100*ess/len(weights), history_path=str(spec.saved_history),
                reused=bool(spec.reuse_history)))
        audit = dict(run_id=spec.run_id, signature=signature, candidates=len(local),
                     native_loss_max_absolute_delta=max(errors, default=0), history_sha256=digest(spec.saved_history),
                     reused=bool(spec.reuse_history))
        pd.DataFrame(local, columns=list(local[0]) if local else ["run_id"]).to_csv(shard, index=False)
        write_json(auditpath, audit)
        audits.append(audit)
        rows.extend(local)
        if not local:
            missing.append(dict(run_id=spec.run_id, reason="no_saved_candidates"))
        if number % 25 == 0:
            print(f"Scored {number}/{len(specs)}", flush=True)
    candidates = pd.DataFrame(rows)
    candidates.to_csv(args.output_dir/"candidates.csv", index=False)
    pd.DataFrame(missing, columns=["run_id", "reason"]).to_csv(args.output_dir/"missing_selection.csv", index=False)
    if candidates.empty:
        if not args.smoke:
            raise RuntimeError("No saved candidates")
        selected = pd.DataFrame()
    else:
        selected = pd.concat([select_best_rows(candidates, metric) for metric in SELECTORS], ignore_index=True)
        selected.to_csv(args.output_dir/"selected.csv", index=False)
    groups = ["ensemble", "method", "graph_metric", "split_type", "data_strength", "bandwidth", "selection_metric"]
    incomplete = []
    if not selected.empty:
        summary = selected.groupby(groups)[["recovery_percent", "ess_percent", "val_mse", "val_closed_sigma_mse"]].agg(["mean", "std", "count"])
        summary.to_csv(args.output_dir/"summary.csv")
        counts = selected.groupby(groups).size().reset_index(name="replicates")
        expected = pd.DataFrame([dict(ensemble=s.ensemble, method=s.method,
            graph_metric=s.graph_metric, split_type=s.split_type, data_strength=s.sweep_maxent,
            bandwidth=s.kernel_bandwidth if s.method=="omc" else 0., selection_metric=selector)
            for s in specs for selector in SELECTORS]).drop_duplicates()
        counts = expected.merge(counts, on=groups, how="left").fillna({"replicates": 0})
        incomplete = counts[counts.replicates != args.n_splits].to_dict("records")
        if not args.smoke:
            export_figures(selected, args.output_dir/"figures")
    audit = dict(expected_fits=len(specs), completed_fits=len(audits),
        reused_fits=sum(bool(s.reuse_history) for s in specs), native_candidates=len(candidates),
        selected_rows=len(selected), missing_selection=missing, incomplete_cells=incomplete,
        native_loss_max_absolute_delta=max((a["native_loss_max_absolute_delta"] for a in audits), default=0),
        fit_audits=audits, training_loss="MSE", selectors=list(SELECTORS),
        shrinkage_alpha=0., covariance_split_order="inverse_then_split",
        same_optimization_forward_model_verified=True, smoke=args.smoke,
        candidate_policy="all saved trajectory/convergence/running-best states; deduplicate step and simplex")
    audit["analysis_code_sha256"] = {p.name: digest(p) for p in (Path(__file__),
        HERE/"sidecar_selection.py", HERE/"plot_iso_policy_sidecars.py")}
    audit["jax_version"] = jax.__version__
    audit["numpy_version"] = np.__version__
    audit["geometry_manifest"] = json.loads((args.output_dir/"geometry/manifest.json").read_text())
    write_json(args.output_dir/"audit.json", audit)
    status_path = args.output_dir/"regression_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status.update(status="native_scalar_refit_audited", completed_fits=len(audits),
            native_candidates=len(candidates), missing_native_candidates=len(missing))
        write_json(status_path, status)
    if not args.smoke:
        figures = "\n".join(f'<h2>{name.replace("_", " ")}</h2><img src="figures/{name}.png">'
            for name in (f"{layout}_{metric}" for layout in ("maxent", "omc_curves", "omc_heatmaps")
                         for metric in ("recovery_percent", "ess_percent")))
        warning = f"Missing selection: {len(missing)} trajectories; incomplete observed cells: {len(incomplete)}. Replicate counts are recorded in the summary tables."
        (args.output_dir/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>ISO policy sidecars</title>'
            '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{max-width:100%}</style>'
            '<h1>ISO full uptake: MSE fitting, MSE versus closed-Sigma selection</h1>'
            f'<p>{len(audits)} fits; {len(candidates)} saved-state candidates. Mean ± sample SD across split replicates.</p>'
            '<p>Both selectors use the existing candidate pool: all saved trajectory, convergence, and running-best states, deduplicated by step and weights.</p>'
            '<p>All-pairs Gaussian OMC uses metric distances divided by their positive-distance median. Strength coefficient is 1/S. '
            'Closed-Sigma is alpha zero, numerical ridge, full inversion then validation subsetting, trace-normalized precision. '
            'Recovery and ESS are report-only. PyRosetta uses unrelaxed ref2015 coordinates.</p>'
            f'<p>{warning}</p><p><a href="selected.csv">Selected checkpoints</a> · <a href="summary.csv">Mean/SD tables</a> · '
            '<a href="audit.json">Replay audit</a> · <a href="manifest.json">Campaign manifest</a></p>'+figures)
    print(json.dumps({k: audit[k] for k in ("expected_fits", "completed_fits", "native_candidates", "selected_rows", "native_loss_max_absolute_delta")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO/"artifacts/iso_policy_sidecars")
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="all")
    parser.add_argument("--features-dir", type=Path, default=base.shrinkage.DEFAULT_FEATURES_DIR)
    parser.add_argument("--clustering-dir", type=Path, default=base.shrinkage.DEFAULT_CLUSTERING_DIR)
    parser.add_argument("--trajectory-dir", type=Path, default=base.source.DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--reuse-campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--pyrosetta-python", type=Path, default=Path("/home/alexi/anaconda3/envs/PLUMED_310/bin/python"))
    parser.add_argument("--strengths", default="10,100,1000,10000,100000,1000000")
    parser.add_argument("--bandwidths", default="0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        payload = json.loads(args.worker_spec.read_text())
        if isinstance(payload, list):
            pending = [s for s in (RunSpec(**s) for s in payload) if not completed(s)]
            for spec in pending:
                run_fit(spec)
        else:
            run_fit(RunSpec(**payload))
        return
    if args.smoke:
        if args.output_dir == REPO/"artifacts/iso_policy_sidecars":
            args.output_dir = REPO/"artifacts/iso_policy_sidecars_smoke"
        args.n_steps, args.n_splits, args.strengths, args.bandwidths = 20, 1, "10", "1"
    if args.n_splits != 3 and not args.smoke:
        parser.error("Publication figures require three split replicates")
    if args.n_steps < 1 or not 1 <= args.jobs <= max(1, len(os.sched_getaffinity(0))//2):
        parser.error("Steps must be positive and workers require two available CPUs each")
    for key in ("output_dir", "features_dir", "clustering_dir", "trajectory_dir", "reuse_campaign", "pyrosetta_python"):
        setattr(args, key, getattr(args, key).resolve())
    for key in ("strengths", "bandwidths"):
        values = list(base.shrinkage.parse_csv(getattr(args, key), float))
        if not values or len(set(values)) != len(values) or any(not np.isfinite(x) or x <= 0 for x in values):
            parser.error(f"Invalid {key}")
        setattr(args, key, sorted(values))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir/"manifest.json"
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"phase", "jobs", "worker_spec"}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            raise ValueError("Campaign settings changed; use a new output directory")
        specs = [RunSpec(**s) for s in previous["specs"]]
    else:
        args.datasplit_dir = prepare_splits(args)
        paths = prepare_distances(args)
        specs = build_specs(args, paths, reuse_index(args))
        write_json(manifest_path, dict(version=VERSION, settings=settings,
            expected_fits=len(specs), training_loss="MSE", selectors=list(SELECTORS),
            maxent_grid={k: list(v) for k, v in MAXENT.items()},
            code_sha256={p.name: digest(p) for p in (Path(__file__), HERE/"iso_sidecar_geometry.py",
                         HERE/"score_iso_pyrosetta.py", HERE/"plot_iso_policy_sidecars.py", HERE/"run_maxent_omc_sidecar.py")},
            specs=[dataclasses.asdict(s) for s in specs]))
    # Freeze data identities independently of file paths so resumed fits cannot
    # silently mix new inputs with completed trajectories.
    current = json.loads(manifest_path.read_text())
    input_paths = []
    for ensemble in ("ISO_BI", "ISO_TRI"):
        input_paths += [args.features_dir/f"features_{ensemble.lower()}.npz",
            args.features_dir/f"topology_{ensemble.lower()}.json",
            args.clustering_dir/f"cluster_assignments_{ensemble}.csv"]
    input_paths += sorted((args.output_dir/"datasplits").rglob("*.csv"))
    input_paths += sorted((args.output_dir/"datasplits").rglob("*.json"))
    fingerprints = {str(p): digest(p) for p in input_paths}
    if "input_sha256" in current and current["input_sha256"] != fingerprints:
        raise ValueError("Campaign data changed; use a new output directory")
    if "input_sha256" not in current:
        current["input_sha256"] = fingerprints
        write_json(manifest_path, current)
    print(f"{VERSION}: {len(specs)} trajectories ({sum(bool(s.reuse_history) for s in specs)} reused)", flush=True)
    if args.phase in {"fit", "all"}:
        execute(specs, args)
    if args.phase in {"analyze", "all"}:
        analyze(specs, args)


if __name__ == "__main__":
    main()
