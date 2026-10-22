#!/usr/bin/env python3
"""MoPrP sidecars: native MSE fits, raw file-weighted validation selection.

--phase prepare verifies reuse and freezes inputs. --smoke --phase all runs
36 short fits covering both splits, all four bandwidths and both graphs.
Production fitting is only launched with --phase fit or --phase all explicitly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.examples.common import loading
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.examples.common.analysis.rescore_iso_coordinate_sigma import make_predict
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import create_data_loaders, run_optimization
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.custom_types.key import m_key
from jaxent.src.utils.hdf import load_optimization_history_from_file
from moprp_sidecar_support import (EXPERIMENT, HERE, aligned_file_weights, config_and_targets,
    digest, model, prepare_geometry, raw_weighted_mse, regularizer, saved_candidates, write_json)

SOURCE = HERE/"_optimise_quick_test_SIGMA_5000_full_uptake__20260926_135652"
RATE_SOURCE = HERE/"_optimise_quick_test_SIGMA_5000__20260924_223118"
LINEAR_SOURCE = HERE/"_optimise_quick_test_SIGMA_5000_linear_bv_fixed__20260926_022111"
MODE_SUFFIXES = {"frame_uptake": "", "rate": "_rate", "linear_uptake": "_linear_bv"}
ENSEMBLES = ("AF2_MSAss", "AF2_filtered")
SPLITS = ("sequence_cluster", "spatial")
EDGE_TYPES = ("work_scale", "dihedral_rmsd")
STRENGTHS = (1., 5., 10., 50., 100., 500., 1000.)
BANDWIDTHS = (.001, .01, .1, 1.)


def experiment_specs(specs, experiment):
    if experiment == "all":
        return specs
    graph_k = {"all_pairs": 0, "knn20": 20}.get(experiment)
    return [s for s in specs if s.method == "maxent" or
        (experiment != "maxent" and s.graph_k == graph_k)]


@dataclass(frozen=True)
class RunSpec:
    ensemble: str
    method: str
    edge_type: str
    graph_k: int
    split_type: str
    split_idx: int
    strength: float
    bandwidth: float
    n_steps: int
    output_dir: str
    features_dir: str
    datasplit_dir: str
    clustering_dir: str
    geometry_dir: str
    template_config: str
    reuse_history: str = ""
    reuse_config: str = ""
    frame_averaging_mode: str = "frame_uptake"

    @property
    def run_id(self):
        tag = f"_k{self.graph_k}_{self.edge_type}_h{self.bandwidth:g}" if self.method == "omc" else ""
        mode = MODE_SUFFIXES[self.frame_averaging_mode]
        return f"MoPrP{mode}_{self.ensemble}_MSE_{self.method}{tag}_{self.split_type}_split{self.split_idx:03d}_maxent{self.strength:g}"

    @property
    def run_dir(self):
        return Path(self.output_dir)/self.split_type/self.run_id

    @property
    def history_path(self):
        return self.run_dir/f"{self.run_id}_results.hdf5"

    @property
    def config_path(self):
        return self.run_dir/f"{self.run_id}_config.json"

    @property
    def saved_history(self):
        return Path(self.reuse_history) if self.reuse_history else self.history_path


def build_specs(args):
    specs = []
    for ensemble in ENSEMBLES:
        templates = sorted(args.source_campaign.glob(f"*/{ensemble}_MSE_*_config.json"))
        if not templates:
            raise ValueError(f"No native MSE configuration for {ensemble}")
        for split in SPLITS:
            for index in range(args.n_splits):
                common = dict(ensemble=ensemble, split_type=split, split_idx=index, n_steps=args.n_steps,
                    output_dir=str(args.output_dir), features_dir=str(args.features_dir),
                    datasplit_dir=str(args.datasplit_dir), clustering_dir=str(args.clustering_dir),
                    geometry_dir=str(args.output_dir/"geometry"), template_config=str(templates[0]))
                if getattr(args, "frame_averaging_mode", "frame_uptake") != "frame_uptake":
                    common["frame_averaging_mode"] = args.frame_averaging_mode
                for strength in args.strengths:
                    specs.append(RunSpec(**common, method="maxent", edge_type="none", graph_k=0,
                        strength=strength, bandwidth=0.))
                if ensemble == "AF2_MSAss":
                    for graph_k in (0, 20):
                        for edge_type in EDGE_TYPES:
                            for bandwidth in args.bandwidths:
                                for strength in args.strengths:
                                    specs.append(RunSpec(**common, method="omc", edge_type=edge_type,
                                        graph_k=graph_k, strength=strength, bandwidth=bandwidth))
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("Duplicate fitting identities")
    return specs


def extend_bandwidth_manifest(previous, generated, settings):
    old_settings = previous["settings"]
    if {k: v for k, v in old_settings.items() if k != "bandwidths"} != {
            k: v for k, v in settings.items() if k != "bandwidths"}:
        raise ValueError("Bandwidth extension cannot change any other campaign setting")
    old_bandwidths, new_bandwidths = set(old_settings["bandwidths"]), set(settings["bandwidths"])
    if not old_bandwidths < new_bandwidths:
        raise ValueError("Bandwidth extension must strictly add to the existing grid")
    old_specs = {s.run_id: s for s in (RunSpec(**row) for row in previous["specs"])}
    generated_ids = {s.run_id for s in generated}
    if not set(old_specs) <= generated_ids:
        raise ValueError("Bandwidth extension would remove existing fits")
    specs = []
    for spec in generated:
        if spec.run_id in old_specs:
            old = old_specs[spec.run_id]
            if replace(old, reuse_history="", reuse_config="") != spec:
                raise ValueError(f"Existing fit configuration changed: {spec.run_id}")
            spec = old
        specs.append(spec)
    extension = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        previous_bandwidths=sorted(old_bandwidths), added_bandwidths=sorted(new_bandwidths-old_bandwidths),
        previous_fits=previous["expected_fits"], expected_fits=len(specs),
        new_fits=len(specs)-len(old_specs))
    return dict(previous, settings=settings, expected_fits=len(specs), specs=[asdict(s) for s in specs],
        extensions=previous.get("extensions", [])+[extension])


def revise_grid_manifest(previous, generated, settings):
    """Preserve retained native fits while changing the active S/h grid."""
    allowed = {"strengths", "bandwidths"}
    if {k: v for k, v in previous["settings"].items() if k not in allowed} != {
            k: v for k, v in settings.items() if k not in allowed}:
        raise ValueError("Grid revision cannot change other campaign settings")
    if not set(previous["settings"]["strengths"]) <= set(settings["strengths"]):
        raise ValueError("Strength extension must retain the existing strengths")
    if not generated:
        raise ValueError("The active grid cannot be empty")
    kind = type(generated[0])
    old = {s.run_id: s for s in (kind(**row) for row in previous["specs"])}
    specs = []
    for spec in generated:
        if spec.run_id in old:
            retained = old[spec.run_id]
            if replace(retained, reuse_history="", reuse_config="") != spec:
                raise ValueError(f"Retained fit configuration changed: {spec.run_id}")
            spec = retained
        specs.append(spec)
    ids = {s.run_id for s in specs}
    if len(ids) != len(specs):
        raise ValueError("Duplicate revised fitting identities")
    revision = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        previous_strengths=previous["settings"]["strengths"], strengths=settings["strengths"],
        previous_bandwidths=previous["settings"]["bandwidths"], bandwidths=settings["bandwidths"],
        previous_fits=len(old), expected_fits=len(specs), retained_fits=len(ids & old.keys()),
        inactive_fits=len(old.keys()-ids), new_fits=len(ids-old.keys()))
    return dict(previous, settings=settings, expected_fits=len(specs), specs=[asdict(s) for s in specs],
        verified_reuses=sum(bool(s.reuse_history) for s in specs),
        grid_revisions=previous.get("grid_revisions", [])+[revision])


def reset_grid_status(output, manifest):
    path = output/"background_status.json"
    if path.exists():
        status = json.loads(path.read_text())
        status.update(status="ready", current_stage="maxent", completed_stages=[],
            manifest_sha256=digest(output/"manifest.json"), grid_revision=manifest["grid_revisions"][-1])
        for key in ("finished_utc", "error", "failed_stage"):
            status.pop(key, None)
        write_json(path, status)


def configured_model(mode):
    native = model()
    if mode == "linear_uptake":
        return build_uptake_model("linear", native.params.timepoints, kint_unit="min^-1", time_unit="min")
    if mode not in MODE_SUFFIXES:
        raise ValueError(f"Unsupported MoPrP forward mode: {mode}")
    native.forward[m_key("HDX_peptide")].frame_averaging_mode = mode
    return native


def validate_protocol(parameters, native):
    if type(parameters) is not type(native) or parameters.temperature != native.temperature:
        raise ValueError("Frozen BV model/protocol changed")
    np.testing.assert_allclose(parameters.timepoints, native.timepoints, rtol=0, atol=1e-7,
        err_msg="Frozen uptake timepoints changed")
    if hasattr(native, "interval_offsets"):
        if parameters.kint_unit != native.kint_unit or parameters.time_unit != native.time_unit:
            raise ValueError("Frozen additive-linear-BV rate/time units changed")
        np.testing.assert_array_equal(parameters.interval_offsets, native.interval_offsets,
            err_msg="Additive-linear-BV interval offsets must remain fixed at zero")


def context(spec, cache):
    key = spec.ensemble, spec.split_type, spec.split_idx, spec.frame_averaging_mode
    if key not in cache:
        config, targets = config_and_targets()
        with contextlib.redirect_stdout(io.StringIO()):
            features, topology = loading.load_features_and_topology(spec.features_dir, spec.ensemble, config.scoring.ensemble_feature_map)
            train, val, full, _ = loading.load_experimental_data("", spec.datasplit_dir, spec.split_type, spec.split_idx)
            loader = create_data_loaders(train+val, train, val, features, topology)
            clusters = loading.load_clustering_results(spec.clustering_dir, config.scoring.ensemble_clustering_map)
        assignments = np.asarray(clusters[spec.ensemble]["cluster_assignments"])
        if len(assignments) != features.features_shape[1]:
            raise ValueError("Cluster assignments and feature frames do not match")
        forward_model = configured_model(spec.frame_averaging_mode)
        for data in (train, val, full):
            loading.validate_hdx_timepoint_count(data, np.asarray(forward_model.params.timepoints), label=spec.ensemble)
            aligned_file_weights(data)
        cache[key] = (features, topology, forward_model, train, val, loader, assignments, config, targets,
            make_predict(forward_model.forward[m_key("HDX_peptide")], features))
    return cache[key]


def score_history(spec, cache):
    features, _, native, _, val, loader, assignments, config, targets, predict = context(spec, cache)
    history = load_optimization_history_from_file(str(spec.saved_history))
    if not history.states or int(history.states[-1].step) != spec.n_steps:
        raise ValueError("History is incomplete or does not match the requested step count")
    weight_surface = aligned_file_weights(val)
    rows, errors = [], []
    for index, (kind, state, weights) in enumerate(saved_candidates(history)):
        parameters = state.params.model_parameters[0]
        if spec.frame_averaging_mode == "linear_uptake":
            validate_protocol(parameters, native.params)
        for actual, expected in zip(jax.tree_util.tree_leaves(parameters), jax.tree_util.tree_leaves(native.params), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8,
                err_msg="Frozen BV parameters/protocol changed")
        uptake = np.asarray(predict(jnp.asarray(weights), parameters)).T
        ordinary = {}
        predicted_val = None
        for name, dataset, saved in [("train", loader.train, state.losses.train_losses[0]),
                                    ("val", loader.val, state.losses.val_losses[0])]:
            observed = np.asarray(dataset.y_true)[..., 0]
            predicted = np.asarray(dataset.residue_feature_ouput_mapping.todense()) @ uptake
            value = float(np.mean((predicted-observed)**2))
            np.testing.assert_allclose(.5*value, float(saved), rtol=3e-3, atol=2e-7,
                err_msg=f"Native MSE replay mismatch: {spec.run_id}/{name}/step{state.step}")
            errors.append(abs(.5*value-float(saved)))
            ordinary[name] = value
            if name == "val":
                predicted_val = predicted
        observed_val = np.asarray(loader.val.y_true)[..., 0]
        weighted = raw_weighted_mse(predicted_val, observed_val, weight_surface)
        ess = 1/np.sum(weights**2)
        rows.append(dict(run_id=spec.run_id, ensemble=spec.ensemble, method=spec.method,
            edge_type=spec.edge_type, graph_k=spec.graph_k, split_type=spec.split_type,
            split_idx=spec.split_idx, strength=spec.strength, bandwidth=spec.bandwidth,
            candidate_index=index, candidate_kind=kind, step=int(state.step),
            val_weighted_mse=weighted, val_mse=ordinary["val"], train_mse=ordinary["train"],
            recovery_percent=calculate_recovery_percentage(assignments, weights, targets, config.scoring.state_mapping),
            ess_percent=100*ess/len(weights), history_path=str(spec.saved_history),
            forward_model=spec.frame_averaging_mode, training_loss="MSE", reused=bool(spec.reuse_history)))
    if not rows or not np.isfinite(pd.DataFrame(rows)[["val_weighted_mse", "recovery_percent", "ess_percent"]]).all().all():
        raise ValueError("Missing or nonfinite saved-state scores")
    return rows, dict(run_id=spec.run_id, candidates=len(rows), native_loss_max_absolute_delta=max(errors),
        history_sha256=digest(spec.saved_history), same_optimization_forward_model_verified=True)


def verified_reuse(specs, source):
    cache, result, audit = {}, [], []
    for spec in specs:
        if spec.method != "maxent":
            result.append(spec)
            continue
        pattern = f"{spec.ensemble}_MSE_{spec.split_type}_split{spec.split_idx:03d}_maxent*config.json"
        matched = [p for p in (source/spec.split_type).glob(pattern)
            if float(json.loads(p.read_text())["loss_config"]["maxent_scaling"]) == spec.strength]
        if len(matched) != 1:
            audit.append(dict(run_id=spec.run_id, reusable=False, reason="No unique source configuration"))
            result.append(spec)
            continue
        path = matched[0]
        old = replace(spec, reuse_config=str(path), reuse_history=str(path).replace("_config.json", "_results.hdf5"))
        try:
            _, verified = score_history(old, cache)
            result.append(old)
            audit.append(dict(run_id=spec.run_id, reusable=True, **{k: v for k, v in verified.items() if k != "run_id"}))
        except (OSError, ValueError, AssertionError, IndexError) as error:
            result.append(spec)
            audit.append(dict(run_id=spec.run_id, reusable=False, reason=str(error)[:1000]))
    return result, audit


def completed(spec):
    path = Path(spec.reuse_config) if spec.reuse_config else spec.config_path
    if not spec.saved_history.exists() or not path.exists():
        return False
    if spec.reuse_history:
        return True  # Accepted only after replay and recorded in the frozen manifest.
    try:
        config = json.loads(path.read_text())
        stored = config.get("moprp_sidecar", {}).get("spec")
        return stored is not None and type(spec)(**stored) == spec
    except (OSError, ValueError, TypeError):
        return False


def run_fit(spec):
    features, topology, native, train, val, loader, _, _, _, _ = context(spec, {})
    template = json.loads(Path(spec.template_config).read_text())
    if template["loss_config"]["optimize_bv_params"]:
        raise ValueError("Example 2 sidecars require fixed BV parameters")
    options = OptimizationConfig(**{**template["opt_config"], "n_steps": spec.n_steps})
    loss = LossConfig(primary_loss="hdx_uptake_MSE_loss", maxent_scaling=spec.strength if spec.method == "maxent" else 1.)
    with contextlib.redirect_stdout(io.StringIO()):
        run_optimization(train_data=train, val_data=val, prior_data=loader, features=features,
            feature_top=topology, forward_model=native, model_parameters=native.params,
            convergence=options.convergence_rates, loss_config=loss, opt_config=options,
            name=spec.run_id, output_dir=str(spec.run_dir), execution_mode="compiled", frame_regularizer=regularizer(spec))
    config = json.loads(spec.config_path.read_text())
    config["effective_settings"]["frame_averaging_mode"] = native.forward[m_key("HDX_peptide")].frame_averaging_mode
    config["moprp_sidecar"] = dict(spec=asdict(spec), training_loss="MSE", selection="mean(moprp.weights * residual²)",
        weight_power=1, weight_normalization=False, intrinsic_rate_unit="min^-1",
        neighbour_metric="logpf_l2" if spec.graph_k else None, coefficient=1/spec.strength,
        code_sha256=digest(Path(__file__)))
    write_json(spec.config_path, config)
    print(f"Completed {spec.run_id}", flush=True)


def execute(specs, args):
    pending = [s for s in specs if not completed(s)]
    cores = sorted(os.sched_getaffinity(0))
    for name in ("logs", "specs"):
        (args.output_dir/name).mkdir(exist_ok=True)
    def queue(slot):
        failures = []
        for spec in pending[slot::args.jobs]:
            path = args.output_dir/"specs"/f"{spec.run_id}.json"
            write_json(path, asdict(spec))
            with (args.output_dir/"logs"/f"{spec.run_id}.log").open("w") as log:
                command = ["taskset", "-c", ",".join(map(str, cores[2*slot:2*slot+2])), sys.executable,
                    str(Path(__file__).resolve()), "--worker-spec", str(path)]
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}).returncode
            print(f"Fit {spec.run_id}: exit {code}", flush=True)
            if code:
                failures.append(spec.run_id)
        return failures
    print(f"Fitting {len(pending)} trajectories; {sum(bool(s.reuse_history) for s in specs)} verified reuses", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = [name for group in pool.map(queue, range(args.jobs)) for name in group]
    if failures:
        raise RuntimeError(f"Failed native fits: {failures}")


def analyze(specs, args):
    cache, rows, audits = {}, [], []
    for index, spec in enumerate(specs, 1):
        if not completed(spec):
            raise ValueError(f"Incomplete native fit: {spec.run_id}")
        scores, audit = score_history(spec, cache)
        rows.extend(scores)
        audits.append(audit)
        if index % 10 == 0:
            print(f"Replayed {index}/{len(specs)} histories", flush=True)
    candidates = pd.DataFrame(rows)
    selected = candidates.sort_values(["val_weighted_mse", "step", "candidate_index"], kind="stable").groupby("run_id", sort=False).head(1)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.experiment == "all" else f"_{args.experiment}"
    candidates.to_csv(args.analysis_dir/f"candidates{suffix}.csv", index=False)
    selected.to_csv(args.analysis_dir/f"selected{suffix}.csv", index=False)
    keys = ["ensemble", "method", "edge_type", "graph_k", "split_type", "strength", "bandwidth"]
    counts = selected.groupby(keys).size()
    if not counts.eq(args.n_splits).all():
        raise ValueError("Selected replicate counts do not match the campaign")
    selected.groupby(keys)[["recovery_percent", "ess_percent", "val_weighted_mse"]].agg(["mean", "std", "count"]).to_csv(args.analysis_dir/f"summary{suffix}.csv")
    audit = dict(smoke=args.smoke, experiment=args.experiment, expected_fits=len(specs), completed_fits=len(audits),
        reused_fits=sum(bool(s.reuse_history) for s in specs), candidates=len(candidates), selected_rows=len(selected),
        selection="mean(moprp.weights * residual²)", weight_power=1, weight_normalization=False,
        candidate_policy="existing saved trajectory/convergence/running-best states, deduplicated by step and weights",
        training_loss="MSE", forward_model=args.frame_averaging_mode, same_optimization_forward_model_verified=True,
        native_loss_max_absolute_delta=max(a["native_loss_max_absolute_delta"] for a in audits), fit_audits=audits)
    write_json(args.analysis_dir/f"audit{suffix}.json", audit)
    sys.path.insert(0, str(EXPERIMENT/"analysis"))
    from plot_moprp_policy_sidecars import METRICS, VALIDATION_METRICS, export_figures
    export_figures(selected, args.analysis_dir/"figures", args.smoke,
        metrics=list(METRICS)+list(VALIDATION_METRICS))
    if args.experiment == "all" and (selected.method == "omc").any():
        from plot_moprp_policy_model_selection_bars import export_figures as export_bars
        export_bars(candidates, args.analysis_dir)
    layouts = ["maxent"]
    graphs = set(selected.loc[selected.method == "omc", "graph_k"])
    layouts += [name for k, name in [(0, "omc_all_pairs"), (20, "omc_knn20")] if k in graphs]
    figures = "".join(f'<h2>{layout}_{metric}</h2><img src="figures/{layout}_{metric}.png">'
        for metric in ("recovery_percent", "ess_percent") for layout in layouts)
    (args.analysis_dir/f"report{suffix}.html").write_text('<!doctype html><meta charset="utf-8"><title>MoPrP sidecars</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{max-width:100%}</style>'
        f'<h1>MoPrP: MSE fitting, raw peptide × timepoint weighted validation selection{" — SMOKE TEST" if args.smoke else ""}</h1>'
        f'<p>Fixed BV; native {args.frame_averaging_mode} uptake. File weights used directly, without squaring, inversion or weight-sum normalization. '
        'Recovery uses existing MoPrP target populations and zero-target decoy support. '
        'k=20 uses shared L2-logPF symmetric-union neighbours for both edge-weight types.</p>'
        f'<p>{len(audits)} histories replayed; {len(selected)} trajectory selections. '
        f'<a href="selected{suffix}.csv">Selected states</a> · <a href="summary{suffix}.csv">Mean/SD tables</a> · <a href="audit{suffix}.json">Audit</a></p>'+figures)
    print(json.dumps({key: audit[key] for key in ["completed_fits", "reused_fits", "candidates", "selected_rows", "native_loss_max_absolute_delta"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="prepare")
    parser.add_argument("--experiment", choices=("all", "maxent", "all_pairs", "knn20"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--features-dir", type=Path, default=HERE/"_featurise")
    parser.add_argument("--datasplit-dir", type=Path, default=HERE/"_datasplits")
    parser.add_argument("--clustering-dir", type=Path, default=EXPERIMENT/"analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters")
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--frame-averaging-mode", choices=tuple(MODE_SUFFIXES), default="frame_uptake")
    parser.add_argument("--topology", type=Path, default=EXPERIMENT/"data/MoPrP_max_plddt_4334.pdb")
    parser.add_argument("--trajectory", type=Path, default=EXPERIMENT/"data/_archived_MoPrP101/_cluster_MoPrP/clusters/all_clusters.xtc")
    parser.add_argument("--strengths", help="Defaults to the prepared campaign grid")
    parser.add_argument("--bandwidths", help="Defaults to the existing campaign grid, or 0.001,0.01,0.1,1 for a new campaign")
    parser.add_argument("--extend-bandwidths", action="store_true", help="Explicitly append bandwidths to a prepared campaign; --phase prepare only")
    parser.add_argument("--revise-grid", action="store_true", help="Explicitly revise active bandwidths and extend strengths; --phase prepare only")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        spec = RunSpec(**json.loads(args.worker_spec.read_text()))
        if not completed(spec):
            run_fit(spec)
        return
    suffix = MODE_SUFFIXES[args.frame_averaging_mode]+("_smoke" if args.smoke else "")
    args.output_dir = args.output_dir or HERE/f"_moprp_policy_sidecars{suffix}"
    args.analysis_dir = args.analysis_dir or EXPERIMENT/"analysis"/f"_moprp_policy_sidecars{suffix}"
    args.source_campaign = args.source_campaign or {
        "frame_uptake": SOURCE, "rate": RATE_SOURCE, "linear_uptake": LINEAR_SOURCE}[args.frame_averaging_mode]
    for name in ("output_dir", "analysis_dir", "features_dir", "datasplit_dir", "clustering_dir", "source_campaign", "topology", "trajectory"):
        setattr(args, name, getattr(args, name).resolve())
    manifest_path = args.output_dir/"manifest.json"
    if args.strengths is None:
        values = json.loads(manifest_path.read_text())["settings"]["strengths"] if manifest_path.exists() else STRENGTHS
        args.strengths = ",".join(map(str, values))
    args.strengths = sorted({float(x) for x in args.strengths.split(",")})
    if args.bandwidths is None:
        values = json.loads(manifest_path.read_text())["settings"]["bandwidths"] if manifest_path.exists() else BANDWIDTHS
        args.bandwidths = ",".join(map(str, values))
    args.bandwidths = sorted({float(x) for x in args.bandwidths.split(",")})
    args.n_splits = 1 if args.smoke else 3
    if args.smoke:
        args.n_steps, args.strengths = 20, [100.]
    if args.n_steps < 1 or not 1 <= args.jobs <= len(os.sched_getaffinity(0))//2:
        parser.error("Positive steps and two available CPUs per worker are required")
    if any(not np.isfinite(x) or x <= 0 for x in args.strengths+args.bandwidths):
        parser.error("Strengths and bandwidths must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension_lock = None
    if args.extend_bandwidths or args.revise_grid:
        if args.phase != "prepare":
            parser.error("Grid changes require --phase prepare")
        if args.extend_bandwidths and args.revise_grid:
            parser.error("Choose either --extend-bandwidths or --revise-grid")
        extension_lock = (args.output_dir/"background.lock").open("a")
        fcntl.flock(extension_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        if key not in {"jobs", "phase", "worker_spec", "experiment", "extend_bandwidths", "revise_grid"}}
    if args.frame_averaging_mode == "frame_uptake":
        settings.pop("frame_averaging_mode")  # Preserve existing full-uptake manifest identity.
    before_extension = None
    before_revision = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings:
            if args.revise_grid:
                before_revision = manifest
                manifest = revise_grid_manifest(manifest, build_specs(args), settings)
            elif args.extend_bandwidths:
                before_extension = manifest
                manifest = extend_bandwidth_manifest(manifest, build_specs(args), settings)
            else:
                raise ValueError("Campaign settings changed; append bandwidths with --extend-bandwidths --phase prepare")
        specs = [RunSpec(**row) for row in manifest["specs"]]
    else:
        if args.frame_averaging_mode != "frame_uptake" and not (args.output_dir/"geometry").exists():
            shutil.copytree(HERE/"_moprp_policy_sidecars/geometry", args.output_dir/"geometry")
        geometry = prepare_geometry(args.output_dir/"geometry", args.features_dir, args.topology, args.trajectory)
        specs = build_specs(args)
        if args.smoke:
            reuse = []
        else:
            specs, reuse = verified_reuse(specs, args.source_campaign)
        write_json(args.output_dir/"reuse_audit.json", reuse)
        manifest = dict(settings=settings, expected_fits=len(specs), geometry=geometry,
            specs=[asdict(s) for s in specs], verified_reuses=sum(bool(s.reuse_history) for s in specs))
        write_json(manifest_path, manifest)
    paths = [EXPERIMENT/"config.yaml", EXPERIMENT/"analysis/state_ratios.json",
        EXPERIMENT/"data/_MoPrP/moprp.weights", EXPERIMENT/"data/_MoPrP/moprp.dexp", EXPERIMENT/"data/_MoPrP/moprp.times",
        args.topology, args.trajectory]
    paths += sorted(args.features_dir.glob("*.npz")) + sorted(args.features_dir.glob("topology_*.json"))
    paths += sorted(args.datasplit_dir.rglob("*.csv")) + sorted(args.datasplit_dir.rglob("*.json"))
    paths += sorted(args.clustering_dir.rglob("*_frame_to_cluster.csv"))
    paths += sorted((args.output_dir/"geometry").glob("*.npz"))
    fingerprints = {str(path): digest(path) for path in paths}
    if "input_sha256" in manifest and manifest["input_sha256"] != fingerprints:
        raise ValueError("Frozen experimental/features/geometry inputs changed")
    manifest["input_sha256"] = fingerprints
    if before_extension is not None:
        backup = args.output_dir/f"manifest_before_extension_{len(manifest['extensions']):03d}.json"
        if backup.exists():
            raise ValueError(f"Refusing to overwrite prior manifest backup: {backup}")
        write_json(backup, before_extension)
    if before_revision is not None:
        backup = args.output_dir/f"manifest_before_grid_revision_{len(manifest['grid_revisions']):03d}.json"
        if backup.exists():
            raise ValueError(f"Refusing to overwrite manifest backup: {backup}")
        write_json(backup, before_revision)
    write_json(manifest_path, manifest)
    if before_revision is not None:
        reset_grid_status(args.output_dir, manifest)
    if before_extension is not None:
        status_path = args.output_dir/"background_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status.update(status="ready", current_stage="all_pairs", manifest_sha256=digest(manifest_path),
                completed_stages=[stage for stage in status.get("completed_stages", []) if stage == "maxent"],
                bandwidth_extension=manifest["extensions"][-1])
            for key in ("finished_utc", "error", "failed_stage"):
                status.pop(key, None)
            write_json(status_path, status)
    if extension_lock is not None:
        extension_lock.close()
    print(f"MoPrP {'SMOKE' if args.smoke else 'production'}: {len(specs)} histories, {manifest['verified_reuses']} verified reuses", flush=True)
    specs = experiment_specs(specs, args.experiment)
    print(f"Experiment {args.experiment}: {len(specs)} histories", flush=True)
    if args.phase in ("fit", "all"):
        execute(specs, args)
    if args.phase in ("analyze", "all"):
        analyze(specs, args)


if __name__ == "__main__":
    main()
