#!/usr/bin/env python3
"""Native MoPrP full-uptake sidecars jointly fitting frame weights and BV."""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]
EXAMPLE2 = EXPERIMENT.parent/"2_CrossValidation"
sys.path.insert(0, str(EXAMPLE2/"fitting/jaxENT"))
import run_moprp_policy_sidecars as fixed
from moprp_sidecar_support import aligned_file_weights, digest, raw_weighted_mse, regularizer, write_json
from jaxent.examples.common.config import ExperimentConfig, LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import run_optimization
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
from jaxent.src.utils.hdf import load_optimization_history_from_file

SOURCE = HERE/"_optimise_quick_test_SIGMA_5000_full_uptake_BV__20260926_140412"
RATE_SOURCE = HERE/"_optimise_quick_test_SIGMA_5000_lr1.0_BV_objectve_scale1.0__20260924_223526"
LINEAR_SOURCE = HERE/"_optimise_quick_test_SIGMA_5000_linear_bv_bc_bh__20260926_022424"
BANDWIDTHS = (.001, .01, .1, 1., 10., 100., 1000.)
BV_PENALTIES = (.5, 1.)


@dataclass(frozen=True)
class RunSpec(fixed.RunSpec):
    bv_reg_scaling: float = .5
    bv_reg_function: str = "L1"

    @property
    def run_id(self):
        return super().run_id+f"_bvreg{self.bv_reg_scaling:g}_bvregfn{self.bv_reg_function}"


def build_specs(args):
    specs = []
    templates = {}
    for ensemble in fixed.ENSEMBLES:
        for path in sorted(args.source_campaign.glob(f"*/{ensemble}_MSE_*_config.json")):
            loss = json.loads(path.read_text())["loss_config"]
            if loss.get("optimize_bv_params") and loss.get("regularization_losses") == [{"name": "model_params_L1_loss"}]:
                templates.setdefault((ensemble, float(loss["bv_reg_scaling"])), path)
    for spec in fixed.build_specs(args):
        for penalty in args.bv_reg_values:
            template = templates.get((spec.ensemble, penalty))
            if template is None:
                raise ValueError(f"No native BV L1 configuration for {spec.ensemble}/{penalty}")
            specs.append(RunSpec(**(asdict(spec) | dict(template_config=str(template),
                geometry_dir=str(args.geometry_dir))), bv_reg_scaling=penalty))
    if len(specs) != len({s.run_id for s in specs}):
        raise ValueError("Duplicate BV fitting identities")
    return specs


def saved_candidates(history):
    """Keep distinct fitted BV parameters even when step and weights coincide."""
    entries = [("trajectory", state) for state in history.states]
    entries += [("convergence", state) for state in history.convergence_states]
    if history.best_state is not None:
        entries.append(("running_best", history.best_state))
    seen = set()
    for kind, state in entries:
        weights = np.asarray(validated_frame_weight_simplex(state.params.frame_weight_simplex), dtype=float)
        weights /= weights.sum()
        parameters = state.params.model_parameters
        fingerprint = hashlib.sha256(weights.tobytes())
        fingerprint.update(str(jax.tree_util.tree_structure(parameters)).encode())
        for leaf in jax.tree_util.tree_leaves(parameters):
            values = np.asarray(leaf)
            fingerprint.update(str((values.dtype.str, values.shape)).encode())
            fingerprint.update(values.tobytes())
        key = int(state.step), fingerprint.hexdigest()
        if key not in seen:
            seen.add(key)
            yield kind, state, weights


def context(spec, cache):
    # Example 3 uses the identical MoPrP protocol, labels and observations, with
    # its own native feature/split paths carried explicitly by the specification.
    result = fixed.context(spec, cache)
    config = ExperimentConfig.from_yaml(str(EXPERIMENT/"config.yaml"))
    if config.scoring != result[7].scoring:
        raise ValueError("Experiment 3 MoPrP mappings differ from the shared protocol")
    return (*result[:7], config, *result[8:])


def validate_parameters(parameters, native):
    fixed.validate_protocol(parameters, native)
    values = np.asarray([np.asarray(parameters.bv_bc).item(), np.asarray(parameters.bv_bh).item()])
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Fitted BV coefficients must be finite and nonnegative")


def score_history(spec, cache):
    _, _, native, _, val, loader, assignments, config, targets, predict = context(spec, cache)
    history = load_optimization_history_from_file(str(spec.saved_history))
    if not history.states or int(history.states[-1].step) != spec.n_steps:
        raise ValueError("Incomplete BV history or incorrect step count")
    surface = aligned_file_weights(val)
    rows, errors = [], []
    for index, (kind, state, weights) in enumerate(saved_candidates(history)):
        parameters = state.params.model_parameters[0]
        validate_parameters(parameters, native.params)
        uptake = np.asarray(predict(jnp.asarray(weights), parameters)).T
        ordinary, predicted_val = {}, None
        for name, dataset, saved in [("train", loader.train, state.losses.train_losses[0]),
                                    ("val", loader.val, state.losses.val_losses[0])]:
            observed = np.asarray(dataset.y_true)[..., 0]
            predicted = np.asarray(dataset.residue_feature_ouput_mapping.todense()) @ uptake
            value = float(np.mean((predicted-observed)**2))
            np.testing.assert_allclose(.5*value, float(saved), rtol=3e-3, atol=2e-7,
                err_msg=f"Native BV-fit MSE replay mismatch: {spec.run_id}/{name}/step{state.step}")
            errors.append(abs(.5*value-float(saved)))
            ordinary[name] = value
            if name == "val":
                predicted_val = predicted
        weighted = raw_weighted_mse(predicted_val, np.asarray(loader.val.y_true)[..., 0], surface)
        rows.append(dict(run_id=spec.run_id, ensemble=spec.ensemble, method=spec.method,
            edge_type=spec.edge_type, graph_k=spec.graph_k, split_type=spec.split_type,
            split_idx=spec.split_idx, strength=spec.strength, bandwidth=spec.bandwidth,
            bv_reg_scaling=spec.bv_reg_scaling, bv_reg_function=spec.bv_reg_function,
            bv_bc=float(np.asarray(parameters.bv_bc).item()), bv_bh=float(np.asarray(parameters.bv_bh).item()),
            candidate_index=index, candidate_kind=kind, step=int(state.step),
            val_weighted_mse=weighted, val_mse=ordinary["val"], train_mse=ordinary["train"],
            recovery_percent=calculate_recovery_percentage(assignments, weights, targets, config.scoring.state_mapping),
            ess_percent=100/(len(weights)*np.sum(weights**2)), history_path=str(spec.saved_history),
            forward_model=spec.frame_averaging_mode, training_loss="MSE", bv_fitted=True, reused=bool(spec.reuse_history)))
    columns = ["val_weighted_mse", "recovery_percent", "ess_percent", "bv_bc", "bv_bh"]
    if not rows or not np.isfinite(pd.DataFrame(rows)[columns]).all().all():
        raise ValueError("Missing or nonfinite BV checkpoint scores")
    return rows, dict(run_id=spec.run_id, candidates=len(rows), native_loss_max_absolute_delta=max(errors),
        history_sha256=digest(spec.saved_history), same_optimization_forward_model_verified=True,
        checkpoint_specific_bv_parameters_verified=True)


def verified_reuse(specs, source):
    cache, result, audits = {}, [], []
    for spec in specs:
        if spec.method != "maxent":
            result.append(spec)
            continue
        pattern = f"{spec.ensemble}_MSE_{spec.split_type}_split{spec.split_idx:03d}_maxent*_config.json"
        paths = []
        for path in (source/spec.split_type).glob(pattern):
            loss = json.loads(path.read_text())["loss_config"]
            if (float(loss["maxent_scaling"]) == spec.strength and
                    float(loss["bv_reg_scaling"]) == spec.bv_reg_scaling and
                    loss.get("optimize_bv_params") and not loss.get("normalize_bv_reg") and
                    loss.get("regularization_losses") == [{"name": "model_params_L1_loss"}]):
                paths.append(path)
        if len(paths) != 1:
            result.append(spec)
            audits.append(dict(run_id=spec.run_id, reusable=False, reason="No unique compatible native BV configuration"))
            continue
        old = replace(spec, reuse_config=str(paths[0]),
            reuse_history=str(paths[0]).replace("_config.json", "_results.hdf5"))
        try:
            _, audit = score_history(old, cache)
            result.append(old)
            audits.append(dict(reusable=True, **audit))
        except (OSError, ValueError, AssertionError, IndexError) as error:
            result.append(spec)
            audits.append(dict(run_id=spec.run_id, reusable=False, reason=str(error)[:1000]))
    return result, audits


def loss_configuration(spec, template):
    source = template["loss_config"]
    if not source["optimize_bv_params"] or source["regularization_losses"] != [{"name": "model_params_L1_loss"}]:
        raise ValueError("Expected the original native BV L1 fitting configuration")
    overrides = dict(optimize_bv_params=True, normalize_bv_reg=False,
        bv_reg_scaling=spec.bv_reg_scaling,
        maxent_scaling=spec.strength if spec.method == "maxent" else 1.)
    if spec.frame_averaging_mode == "linear_uptake":
        overrides["trainable_model_parameters"] = frozenset({"raw_bv_bc", "raw_bv_bh"})
    return LossConfig(**(source | overrides))


def run_fit(spec):
    features, topology, native, train, val, loader, _, _, _, _ = context(spec, {})
    template = json.loads(Path(spec.template_config).read_text())
    options = OptimizationConfig(**(template["opt_config"] | dict(n_steps=spec.n_steps)))
    loss = loss_configuration(spec, template)
    with contextlib.redirect_stdout(io.StringIO()):
        run_optimization(train_data=train, val_data=val, prior_data=loader, features=features,
            feature_top=topology, forward_model=native, model_parameters=native.params,
            convergence=options.convergence_rates, loss_config=loss, opt_config=options,
            name=spec.run_id, output_dir=str(spec.run_dir), execution_mode="compiled", frame_regularizer=regularizer(spec))
    config = json.loads(spec.config_path.read_text())
    config["effective_settings"]["frame_averaging_mode"] = spec.frame_averaging_mode
    config["moprp_sidecar"] = dict(spec=asdict(spec), training_loss="MSE", bv_fitted=True,
        selection="mean(moprp.weights * residual²)", weight_power=1, weight_normalization=False,
        intrinsic_rate_unit="min^-1", neighbour_metric="logpf_l2" if spec.graph_k else None,
        geometry_bv_coefficients=[.35, 2.], geometry_frozen=True, coefficient=1/spec.strength,
        code_sha256=digest(Path(__file__)))
    write_json(spec.config_path, config)
    print(f"Completed {spec.run_id}", flush=True)


def execute(specs, args):
    pending = [s for s in specs if not fixed.completed(s)]
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
    print(f"Fitting {len(pending)} BV trajectories; {sum(bool(s.reuse_history) for s in specs)} verified reuses", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        failures = [name for group in pool.map(queue, range(args.jobs)) for name in group]
    if failures:
        raise RuntimeError(f"Failed native BV fits: {failures}")


def verify_geometry(args):
    metadata = json.loads((args.geometry_dir/"manifest.json").read_text())
    for filename in ("features_AF2_MSAss.npz", "topology_AF2_MSAss.json"):
        shared = EXAMPLE2/"fitting/jaxENT/_featurise"/filename
        if digest(args.features_dir/filename) != digest(shared) or metadata["input_sha256"].get(str(shared)) != digest(shared):
            raise ValueError(f"Frozen OMC geometry does not match Example 3 features: {filename}")
    for filename, fingerprint in metadata["file_sha256"].items():
        if digest(args.geometry_dir/filename) != fingerprint:
            raise ValueError(f"Frozen OMC geometry changed: {filename}")
    return metadata


def analyze(specs, args):
    cache, rows, audits = {}, [], []
    for index, spec in enumerate(specs, 1):
        if not fixed.completed(spec):
            raise ValueError(f"Incomplete native BV fit: {spec.run_id}")
        scored, audit = score_history(spec, cache)
        rows.extend(scored)
        audits.append(audit)
        if index % 10 == 0:
            print(f"Replayed {index}/{len(specs)} BV histories", flush=True)
    candidates = pd.DataFrame(rows)
    selected = candidates.sort_values(["val_weighted_mse", "step", "candidate_index"], kind="stable").groupby("run_id", sort=False).head(1)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.experiment == "all" else f"_{args.experiment}"
    candidates.to_csv(args.analysis_dir/f"candidates{suffix}.csv", index=False)
    selected.to_csv(args.analysis_dir/f"selected{suffix}.csv", index=False)
    keys = ["ensemble", "method", "edge_type", "graph_k", "split_type", "strength", "bandwidth", "bv_reg_scaling", "bv_reg_function"]
    if not selected.groupby(keys).size().eq(args.n_splits).all():
        raise ValueError("BV split-replicate counts do not match the campaign")
    selected.groupby(keys)[["recovery_percent", "ess_percent", "val_weighted_mse"]].agg(["mean", "std", "count"]).to_csv(args.analysis_dir/f"summary{suffix}.csv")
    audit = dict(smoke=args.smoke, experiment=args.experiment, expected_fits=len(specs), completed_fits=len(audits),
        reused_fits=sum(bool(s.reuse_history) for s in specs), candidates=len(candidates), selected_rows=len(selected),
        selection="mean(moprp.weights * residual²)", weight_power=1, weight_normalization=False,
        candidate_policy="all saved trajectory/convergence/running-best states; deduplicate step, weights and fitted model parameters",
        training_loss="MSE", forward_model=args.frame_averaging_mode, bv_fitted=True,
        bv_penalties=args.bv_reg_values, geometry_bv_coefficients=[.35, 2.], geometry_frozen=True,
        same_optimization_forward_model_verified=True, checkpoint_specific_bv_parameters_verified=True,
        native_loss_max_absolute_delta=max(a["native_loss_max_absolute_delta"] for a in audits), fit_audits=audits)
    write_json(args.analysis_dir/f"audit{suffix}.json", audit)
    sys.path.insert(0, str(EXPERIMENT/"analysis"))
    from plot_moprp_bv_policy_sidecars import export_figures
    layouts = export_figures(candidates, selected, args.analysis_dir, suffix, args.smoke)
    figures = "".join(f'<h2>{layout}_{metric}</h2><img src="figures/{layout}_{metric}.png">'
        for metric in ("recovery_percent", "ess_percent") for layout in layouts)
    (args.analysis_dir/f"report{suffix}.html").write_text('<!doctype html><meta charset="utf-8"><title>MoPrP BV sidecars</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}img{max-width:100%}</style>'
        f'<h1>MoPrP CrossValidationBV: {args.frame_averaging_mode} uptake, MSE fitting{" — SMOKE TEST" if args.smoke else ""}</h1>'
        '<p>Joint frame-weight and bc/bh fitting with original L1 penalties 0.5 and 1.0. '
        'Each saved checkpoint is evaluated with its own fitted BV coefficients. '
        'Selection uses direct peptide × timepoint MoPrP.weights, with no squaring or normalization. '
        'For sweep plots the BV penalty is selected independently within each split replicate and sweep point; '
        'final bars pool both split types after selecting all parameters within each split replicate. '
        'OMC geometry is frozen at published BV coefficients; k20 uses shared L2-logPF neighbours.</p>'
        f'<p>{len(audits)} audited histories; {len(selected)} trajectory selections. '
        f'<a href="selected{suffix}.csv">Selected states and BV coefficients</a> · '
        f'<a href="selected_profiled{suffix}.csv">BV-penalty-selected sweep points</a> · '
        f'<a href="audit{suffix}.json">Audit</a></p>'+figures)
    print(json.dumps({k: audit[k] for k in ("completed_fits", "reused_fits", "candidates", "selected_rows", "native_loss_max_absolute_delta")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="prepare")
    parser.add_argument("--experiment", choices=("all", "maxent", "all_pairs", "knn20"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--features-dir", type=Path, default=HERE/"_featurise")
    parser.add_argument("--datasplit-dir", type=Path, default=HERE/"_datasplits")
    parser.add_argument("--clustering-dir", type=Path, default=EXAMPLE2/"analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters")
    parser.add_argument("--geometry-dir", type=Path, default=EXAMPLE2/"fitting/jaxENT/_moprp_policy_sidecars/geometry")
    parser.add_argument("--source-campaign", type=Path)
    parser.add_argument("--frame-averaging-mode", choices=tuple(fixed.MODE_SUFFIXES), default="frame_uptake")
    parser.add_argument("--strengths")
    parser.add_argument("--bandwidths")
    parser.add_argument("--revise-grid", action="store_true")
    parser.add_argument("--bv-reg-values", default="0.5,1")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        spec = RunSpec(**json.loads(args.worker_spec.read_text()))
        if not fixed.completed(spec):
            run_fit(spec)
        return
    suffix = fixed.MODE_SUFFIXES[args.frame_averaging_mode]+("_smoke" if args.smoke else "")
    args.output_dir = args.output_dir or HERE/f"_moprp_policy_sidecars{suffix}"
    args.analysis_dir = args.analysis_dir or EXPERIMENT/"analysis"/f"_moprp_policy_sidecars{suffix}"
    args.source_campaign = args.source_campaign or {
        "frame_uptake": SOURCE, "rate": RATE_SOURCE, "linear_uptake": LINEAR_SOURCE}[args.frame_averaging_mode]
    for name in ("output_dir", "analysis_dir", "features_dir", "datasplit_dir", "clustering_dir", "geometry_dir", "source_campaign"):
        setattr(args, name, getattr(args, name).resolve())
    path = args.output_dir/"manifest.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    for name, default in (("strengths", fixed.STRENGTHS), ("bandwidths", BANDWIDTHS)):
        if getattr(args, name) is None:
            values = existing["settings"][name] if existing else default
            setattr(args, name, ",".join(map(str, values)))
    for name in ("strengths", "bandwidths", "bv_reg_values"):
        setattr(args, name, sorted({float(x) for x in getattr(args, name).split(",")}))
    args.n_splits = 1 if args.smoke else 3
    if args.smoke:
        args.n_steps, args.strengths = 20, [100.]
        args.bandwidths = [10., 100., 1000.] if args.frame_averaging_mode != "frame_uptake" else [.001, 1., 1000.]
    if args.n_steps < 1 or not 1 <= args.jobs <= len(os.sched_getaffinity(0))//2:
        parser.error("Positive steps and two CPUs per worker are required")
    if any(not np.isfinite(x) or x <= 0 for x in args.strengths+args.bandwidths+args.bv_reg_values):
        parser.error("Strengths, bandwidths and BV penalties must be positive and finite")
    geometry = verify_geometry(args)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        if k not in {"jobs", "phase", "worker_spec", "experiment", "revise_grid"}}
    if args.frame_averaging_mode == "frame_uptake":
        settings.pop("frame_averaging_mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision_lock = None
    before_revision = None
    if args.revise_grid:
        if args.phase != "prepare":
            parser.error("Grid changes require --phase prepare")
        revision_lock = (args.output_dir/"background.lock").open("a")
        fcntl.flock(revision_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["settings"] != settings:
            if not args.revise_grid:
                raise ValueError("Frozen BV campaign settings changed; use --revise-grid --phase prepare")
            before_revision = manifest
            manifest = fixed.revise_grid_manifest(manifest, build_specs(args), settings)
        specs = [RunSpec(**row) for row in manifest["specs"]]
    else:
        specs = build_specs(args)
        specs, reuse = (specs, []) if args.smoke else verified_reuse(specs, args.source_campaign)
        write_json(args.output_dir/"reuse_audit.json", reuse)
        manifest = dict(settings=settings, expected_fits=len(specs), geometry=geometry,
            specs=[asdict(s) for s in specs], verified_reuses=sum(bool(s.reuse_history) for s in specs))
    paths = [EXPERIMENT/"config.yaml", EXAMPLE2/"config.yaml", EXAMPLE2/"analysis/state_ratios.json"]
    paths += [EXAMPLE2/"data/_MoPrP"/name for name in ("moprp.weights", "moprp.times", "moprp.dexp")]
    paths += sorted(args.features_dir.glob("*.npz"))+sorted(args.features_dir.glob("topology_*.json"))
    paths += sorted(args.datasplit_dir.rglob("*.csv"))+sorted(args.datasplit_dir.rglob("*.json"))
    paths += sorted(args.clustering_dir.rglob("*_frame_to_cluster.csv"))
    paths += sorted(args.geometry_dir.glob("*.npz"))+[args.geometry_dir/"manifest.json"]
    paths += [Path(s.template_config) for s in specs]
    paths += [Path(p) for s in specs for p in (s.reuse_config, s.reuse_history) if p]
    fingerprints = {str(p): digest(p) for p in sorted(set(paths))}
    if "input_sha256" in manifest and manifest["input_sha256"] != fingerprints:
        raise ValueError("Frozen MoPrP BV campaign inputs changed")
    manifest["input_sha256"] = fingerprints
    if before_revision is not None:
        backup = args.output_dir/f"manifest_before_grid_revision_{len(manifest['grid_revisions']):03d}.json"
        if backup.exists():
            raise ValueError(f"Refusing to overwrite manifest backup: {backup}")
        write_json(backup, before_revision)
    write_json(path, manifest)
    if before_revision is not None:
        fixed.reset_grid_status(args.output_dir, manifest)
    if revision_lock is not None:
        revision_lock.close()
    print(f"MoPrP BV {'SMOKE' if args.smoke else 'production'}: {len(specs)} histories, {manifest['verified_reuses']} verified reuses", flush=True)
    specs = fixed.experiment_specs(specs, args.experiment)
    print(f"Experiment {args.experiment}: {len(specs)} histories", flush=True)
    if args.phase in ("fit", "all"):
        execute(specs, args)
    if args.phase in ("analyze", "all"):
        analyze(specs, args)


if __name__ == "__main__":
    main()
