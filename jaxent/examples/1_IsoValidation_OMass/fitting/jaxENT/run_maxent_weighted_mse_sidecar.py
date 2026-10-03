#!/usr/bin/env python3
"""BI/TRI MaxEnt sweep: Linear BV/frame uptake x four fixed data-fit metrics.

Coordinate Sigma uses alpha=0 (any stability ridge is reported). GT uptake
precision is diagonal over peptide x timepoint, not time-averaged covariance.
Default: 336 native fits; both final and closed-Sigma-MSE-selected plots.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path

import run_sigma_source_sidecar as source
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.examples.common.losses import LOSS_REGISTRY
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import run_optimization
from jaxent.src.custom_types.key import m_key

shrinkage = source.shrinkage
plt = source.plt
HERE = Path(__file__).resolve().parent
VERSION = "maxent_weighted_mse_stable_kl_v2"
ENSEMBLES = ("ISO_BI", "ISO_TRI")
MODES = ("linear", "uptake")
LOSSES = ("mse", "gt_coordinate", "closed_coordinate", "gt_uptake_weighted")
LABELS = {"mse": "MSE", "gt_coordinate": "GT coordinate Sigma-MSE",
          "closed_coordinate": "Closed coordinate Sigma-MSE",
          "gt_uptake_weighted": "GT uptake timepoint Weighted-MSE"}
DEFAULT_MAXENT = (.01, .1, 1., 10., 100., 1000., 10000.)
# Verified against the completed stable-KL campaign before adding normalization
# options. Its MSE/coordinate-Sigma fitting path is unchanged by this extension.
COMPATIBLE_BASELINE_HASH = "1653d6e9d8c1edf491cbc8f261d5077928177c57b9bc87142538443b2e8a1a83"


@dataclasses.dataclass(frozen=True)
class RunSpec(source.RunSpec):
    uptake_mode: str
    sweep_maxent: float
    weights_path: str
    precision_normalization: str = "global"

    @property
    def run_id(self):
        suffix = ("_per_peptide" if self.sigma_source == "gt_uptake_weighted"
                  and self.precision_normalization == "per_peptide" else "")
        return (f"{VERSION}_{self.ensemble}_{self.uptake_mode}_{self.sigma_source}"
                f"_M{shrinkage.alpha_token(self.sweep_maxent)}_split{self.split_idx:03d}{suffix}")


def marginal_precision(curves, oracle_weights, variance_floor=1e-4, normalization="global"):
    """Population variance over frames; curves are peptide x time x frame.

    Global: normalize over all peptide/time entries. Per-peptide: normalize
    over time separately for each peptide, giving every peptide equal weight.
    No effective-sample-size correction or variance of the estimated mean.
    """
    curves, weights = np.asarray(curves, dtype=float), np.asarray(oracle_weights, dtype=float)
    if (curves.ndim != 3 or weights.shape != (curves.shape[-1],)
            or not np.isfinite(curves).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0
            or not np.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("Invalid frame uptake, oracle weights or variance floor")
    weights = weights / weights.sum()
    mean = np.einsum("ptf,f->pt", curves, weights)
    variance = np.einsum("ptf,f->pt", (curves - mean[..., None])**2, weights)
    precision = 1. / np.maximum(variance, variance_floor)
    if normalization == "global":
        precision /= precision.mean()
    elif normalization == "per_peptide":
        precision /= precision.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown precision normalization: {normalization}")
    return mean, variance, precision


def make_weighted_loss(train_precision, val_precision):
    """Fixed observation-space precision; native model/loader/optimizer untouched."""
    precisions = [jnp.asarray(train_precision), jnp.asarray(val_precision)]

    def loss(model, dataset, prediction_index):
        uptake = model.outputs[prediction_index].uptake  # time x residue
        values = []
        for data, precision in zip((dataset.train, dataset.val), precisions):
            predicted = data.residue_feature_ouput_mapping.todense() @ uptake.T
            target = data.y_true[..., 0]
            if predicted.shape != precision.shape or target.shape != precision.shape:
                raise ValueError("Weighted-MSE observation/timepoint shape mismatch")
            values.append(.5 * jnp.mean(precision * (predicted - target)**2))
        return tuple(values)
    return loss


def prepare_weights(args):
    paths, diagnostics = {}, []
    directory = args.output_dir / "uptake_precision"
    directory.mkdir(exist_ok=True)
    for ensemble in getattr(args, "ensembles", ENSEMBLES):
        features, topology = shrinkage.load_features(args.features_dir, ensemble)
        assignments = shrinkage.load_cluster_assignments(args.clustering_dir, ensemble)
        oracle = source.population_weights(assignments, "gt")
        model = shrinkage.configure_model("uptake", assignments)
        # Repository HDXrate is min^-1; use the actual frame uptake, never grouped
        # uptake or the older time-averaged Sigma-coordinate helper.
        curves = np.asarray(model.forward[m_key("HDX_peptide")](features, model.params).uptake)
        for split in range(args.n_splits):
            train, val = shrinkage.load_split(args.datasplit_dir, "sequence_cluster", split)
            loader = source.create_data_loaders(train + val, train, val, features, topology)
            arrays = dict(oracle_weights=oracle, timepoints=np.asarray(shrinkage.TIMEPOINTS),
                          variance_floor=args.variance_floor,
                          precision_normalization=args.precision_normalization)
            for label, data in (("train", loader.train), ("val", loader.val)):
                mapping = np.asarray(data.residue_feature_ouput_mapping.todense())
                # Map EACH frame before calculating peptide variance, retaining
                # within-peptide residue covariance rather than averaging variances.
                mapped = np.einsum("pr,trf->ptf", mapping, curves)
                mean, variance, precision = marginal_precision(
                    mapped, oracle, args.variance_floor, args.precision_normalization)
                arrays.update({f"{label}_mean": mean, f"{label}_variance": variance,
                               f"{label}_precision": precision, f"{label}_mapping": mapping})
                diagnostics.append(dict(ensemble=ensemble, split_idx=split, partition=label,
                    floored_entries=int((variance < args.variance_floor).sum()),
                    entries=variance.size, min_variance=float(variance.min()),
                    max_precision=float(precision.max()), mean_precision=float(precision.mean()),
                    min_peptide_total_weight=float(precision.sum(axis=1).min()),
                    max_peptide_total_weight=float(precision.sum(axis=1).max()),
                    saturated_precision_fraction=float(
                        precision[(mean < .01) | (mean > .99)].sum() / precision.sum()),
                    precision_ess_percent=float(100 * precision.sum()**2 /
                                                (precision.size * np.sum(precision**2))),
                    floored_precision_fraction=float(
                        precision[variance < args.variance_floor].sum() / precision.sum())))
            path = directory / f"{ensemble}_split{split:03d}.npz"
            np.savez_compressed(path, **arrays)
            paths[ensemble, split] = path
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "uptake_precision_diagnostics.csv", index=False)
    maximum_mass = max(row["floored_precision_fraction"] for row in diagnostics)
    if maximum_mass > .5:
        print(f"WARNING: variance-floored entries receive up to {maximum_mass:.1%} "
              "of precision weight; inspect uptake_precision_diagnostics.csv", flush=True)
    return paths


def build_specs(args, sigma_paths, weight_paths):
    return [RunSpec(ensemble=ensemble, sigma_source=loss, alpha=0.,
        split_type="sequence_cluster", split_idx=split,
        sigma_path=str(sigma_paths.get((ensemble, loss, 0.), "")),
        output_dir=str(args.output_dir), features_dir=str(args.features_dir),
        datasplit_dir=str(args.datasplit_dir), clustering_dir=str(args.clustering_dir),
        n_steps=args.n_steps, learning_rate=1., ema_alpha=.5, forward_model_scaling=1000.,
        execution_mode="compiled", uptake_mode=mode, sweep_maxent=value,
        weights_path=str(weight_paths[ensemble, split]),
        precision_normalization=getattr(args, "precision_normalization", "global"))
        for ensemble in getattr(args, "ensembles", ENSEMBLES) for mode in MODES for loss in LOSSES
        for value in args.maxent_values for split in range(args.n_splits)]


def reuse_baselines(specs, baseline_dir):
    """Read-only reuse of verified stable-KL MSE/coordinate fits; never refit them."""
    manifest = json.loads((baseline_dir / "manifest.json").read_text())
    if manifest["settings"]["code_sha256"] != COMPATIBLE_BASELINE_HASH:
        raise ValueError("Baseline code is not the verified stable-KL campaign")
    def key(spec):
        return spec.ensemble, spec.uptake_mode, spec.sigma_source, spec.sweep_maxent, spec.split_idx
    baseline = {key(s): s for s in (RunSpec(**row) for row in manifest["specs"])}
    result = []
    for spec in specs:
        if spec.sigma_source == "gt_uptake_weighted":
            result.append(spec)
            continue
        old = baseline.get(key(spec))
        if old is None or not run_is_complete(old):
            raise ValueError(f"Missing complete baseline for {key(spec)}")
        excluded = {"output_dir", "sigma_path", "weights_path", "precision_normalization"}
        for field in dataclasses.fields(spec):
            if field.name not in excluded and getattr(old, field.name) != getattr(spec, field.name):
                raise ValueError(f"Baseline setting mismatch: {field.name}")
        result.append(old)
    return result


def run_fit(spec):
    features, topology = shrinkage.load_features(spec.features_dir, spec.ensemble)
    assignments = shrinkage.load_cluster_assignments(spec.clustering_dir, spec.ensemble)
    model = shrinkage.configure_model(spec.uptake_mode, assignments)
    train, val = shrinkage.load_split(spec.datasplit_dir, spec.split_type, spec.split_idx)
    precision = None
    loss_name = "hdx_uptake_MSE_loss"
    if spec.sigma_source.endswith("coordinate"):
        with np.load(spec.sigma_path) as archive:
            precision = jnp.asarray(archive["Sigma_inv_normalized"])
        loss_name = "hdx_uptake_sigma_MSE_loss"
    elif spec.sigma_source == "gt_uptake_weighted":
        loss_name = "sidecar_gt_uptake_timepoint_weighted_mse"
        with np.load(spec.weights_path) as archive:
            LOSS_REGISTRY[loss_name] = make_weighted_loss(archive["train_precision"], archive["val_precision"])
    run_optimization(train_data=train, val_data=val,
        prior_data=shrinkage.build_prior_dataset(model, features, topology),
        features=features, feature_top=topology, forward_model=model, model_parameters=model.params,
        convergence=list(shrinkage.CONVERGENCE_RATES),
        loss_config=LossConfig(primary_loss=loss_name, maxent_scaling=spec.sweep_maxent),
        opt_config=OptimizationConfig(n_steps=spec.n_steps, learning_rate=spec.learning_rate,
            ema_alpha=spec.ema_alpha, convergence_rates=list(shrinkage.CONVERGENCE_RATES),
            optimizer="adam", step_chunk_size=100, lr_adjustment=True, frame_average_impl="tensordot",
            reset_threshold_cooldown_on_oscillation=True, forward_model_scaling=spec.forward_model_scaling),
        name=spec.run_id, output_dir=str(spec.run_dir), cov_matrix=precision,
        execution_mode=spec.execution_mode)
    config = json.loads(spec.config_path.read_text())
    mode = "linear_uptake" if spec.uptake_mode == "linear" else "frame_uptake"
    config["effective_settings"]["frame_averaging_mode"] = mode
    config["sidecar_settings"] = dataclasses.asdict(spec) | dict(frame_averaging_mode=mode,
        sigma_shrinkage=0., kl_coefficient=1/spec.sweep_maxent,
        intrinsic_rate_provider="jaxent_calculate_HDXrate", intrinsic_rate_unit="min^-1")
    history = source.load_optimization_history_from_file(str(spec.history_path))
    if not history.states:
        raise RuntimeError("Optimization returned an empty history")
    terminal = history.states[-1]
    terminal_loss = float(terminal.losses.total_train_loss)
    config["sidecar_settings"].update(
        terminal_step=int(terminal.step), terminal_train_loss=terminal_loss,
        zero_step_reason=("nonfinite_initial_loss" if not np.isfinite(terminal_loss)
                          else "initial_loss_below_tolerance") if int(terminal.step) == 0 else None)
    spec.config_path.write_text(json.dumps(config, indent=2))
    if not np.isfinite(terminal_loss):
        raise RuntimeError("Nonfinite terminal loss; inspect saved history")


def run_is_complete(spec):
    if not source.run_is_complete(spec):
        return False
    history = source.load_optimization_history_from_file(str(spec.history_path))
    return bool(history.states)


def execute(specs, output_dir, jobs):
    for name in ("specs", "logs"):
        (output_dir / name).mkdir(exist_ok=True)
    pending = []
    for spec in specs:
        path = output_dir / "specs" / f"{spec.run_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(spec), indent=2))
        if not run_is_complete(spec):
            if Path(spec.output_dir).resolve() != output_dir.resolve():
                raise RuntimeError(f"Reused baseline is incomplete; will not overwrite {spec.run_id}")
            pending.append(path)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(source._worker_command, Path(__file__), path,
                               output_dir / "logs" / f"{path.stem}.log") for path in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, code = future.result()
            print(f"[{index}/{len(pending)}] {name}: return code {code}", flush=True)
            if code:
                failures.append(name)
    if failures:
        raise RuntimeError(f"Failed fits; inspect logs: {failures}")


def plot_metric(frame, metric, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10").colors
    for axis, ensemble in zip(axes, ENSEMBLES):
        for index, (mode, loss) in enumerate((m, l) for m in MODES for l in LOSSES):
            rows = frame[(frame.ensemble == ensemble) & (frame.uptake_mode == mode) & (frame.sigma_source == loss)]
            for _, trace in rows.groupby("split_idx"):
                trace = trace.sort_values("maxent")
                axis.plot(trace.maxent, trace[metric], color=colors[index], alpha=.2, lw=.7)
            summary = rows.groupby("maxent")[metric].agg(["mean", "std"]).sort_index()
            label = f"{'Linear BV' if mode == 'linear' else 'Uptake'} · {LABELS[loss]}"
            axis.plot(summary.index, summary["mean"], "o-", color=colors[index], label=label)
        axis.set(title=ensemble.replace("_", " "), xscale="log", xlabel="MaxEnt parameter (KL coefficient = 1 / MaxEnt)", ylim=(0, 102))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Recovery (%)" if metric == "recovery_percent" else "ESS (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    policy = "Final optimization step" if "final_step" in output.name else "Selected by closed-coordinate Sigma-MSE"
    weighted = frame[frame.sigma_source == "gt_uptake_weighted"]
    normalization = (weighted.precision_normalization.iloc[0]
                     if "precision_normalization" in weighted and not weighted.empty else "global")
    fig.suptitle(f"{policy} · Sigma shrinkage = 0 · Weighted-MSE: {normalization.replace('_', ' ')} normalization")
    fig.subplots_adjust(bottom=.28, top=.88, wspace=.08)
    output.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def combine_results(current, previous):
    """Append disjoint sweep points; never silently overwrite old observations."""
    if not current.empty and not previous.empty:
        keys = ["ensemble", "uptake_mode", "sigma_source", "maxent", "split_idx"]
        if not current[keys].merge(previous[keys], on=keys).empty:
            raise ValueError("Extended results overlap existing sweep points")
    return pd.concat([previous, current], ignore_index=True)


def validate_extension(args):
    old = json.loads((args.extend_results / "manifest.json").read_text())
    settings = old["settings"]
    for key in ("features_dir", "datasplit_dir", "clustering_dir", "trajectory_dir",
                "n_steps", "n_splits", "variance_floor", "precision_normalization"):
        if str(settings.get(key)) != str(getattr(args, key)):
            raise ValueError(f"Extended campaign setting mismatch: {key}")
    if old["version"] != VERSION or set(settings["maxent_values"]) & set(args.maxent_values):
        raise ValueError("Extension requires the same fitting version and disjoint MaxEnt values")
    for name in ("selected_results.csv", "final_step_results.csv", "convergence_scores.csv",
                 "incomplete_runs.csv", "missing_selection.csv", "frame_weights.npz"):
        if not (args.extend_results / name).is_file():
            raise ValueError(f"Previous campaign has no {name}")
    if len(pd.read_csv(args.extend_results / "incomplete_runs.csv")):
        raise ValueError("Previous campaign is incomplete")
    previous_scores = pd.read_csv(args.extend_results / "convergence_scores.csv", nrows=0)
    if "val_closed_sigma_mse" not in previous_scores:
        raise ValueError("Previous campaign uses old selection; rescore closed-Sigma validation before extending")
    return old.get("combined_expected_fits", old["expected_fits"])


def analyze(specs, output_dir, extend_results=None):
    cache, conv, final, incomplete, missing, weights = {}, [], [], [], [], {}
    for spec in specs:
        if not source.run_is_complete(spec):
            incomplete.append(dict(run_id=spec.run_id, reason="missing_or_invalid_history"))
            continue
        key = spec.ensemble, spec.split_idx, spec.uptake_mode
        if key not in cache:
            context = list(source.score_context(spec, {}))
            context[2] = shrinkage.configure_model(spec.uptake_mode, context[1])
            cache[key] = tuple(context)
        history = source.load_optimization_history_from_file(str(spec.history_path))
        if not history.states:
            incomplete.append(dict(run_id=spec.run_id, reason="empty_optimization_history"))
            continue
        labeled = source.iter_labeled_convergence_states(history)
        if not labeled:
            missing.append(dict(run_id=spec.run_id, reason="no_convergence_states"))
        states = [(x.state, "convergence_state", x.rank, x.threshold) for x in labeled]
        states.append((history.states[-1], "final_optimization_state", -1, np.nan))
        for state, policy, rank, threshold in states:
            row, w = source.score_state(dataclasses.replace(spec, sigma_source="gt_coordinate"), state, policy, cache[key])
            row.update(run_id=spec.run_id, sigma_source=spec.sigma_source,
                sigma_source_label=LABELS[spec.sigma_source], uptake_mode=spec.uptake_mode,
                maxent=spec.sweep_maxent, kl_coefficient=1/spec.sweep_maxent,
                precision_normalization=spec.precision_normalization,
                convergence_rank=rank, convergence_threshold=threshold)
            (final if rank == -1 else conv).append(row)
            weights[f"{spec.run_id}__{rank}"] = w
    for name, rows in (("incomplete_runs", incomplete), ("missing_selection", missing)):
        report = pd.DataFrame(rows, columns=["run_id", "reason"])
        if extend_results:
            report = pd.concat([pd.read_csv(extend_results / f"{name}.csv"), report], ignore_index=True)
        report.to_csv(output_dir / f"{name}.csv", index=False)
    if not final:
        raise RuntimeError("No completed fits to analyze")
    convergence = pd.DataFrame(conv)
    selected = source.select_best_rows(convergence) if conv else pd.DataFrame(columns=final[0])
    exported_convergence = convergence
    if extend_results:
        exported_convergence = pd.concat([
            pd.read_csv(extend_results / "convergence_scores.csv"), convergence], ignore_index=True)
    exported_convergence.to_csv(output_dir / "convergence_scores.csv", index=False)
    for policy, frame in (("selected", selected), ("final_step", pd.DataFrame(final))):
        if extend_results:
            frame = combine_results(frame, pd.read_csv(extend_results / f"{policy}_results.csv"))
        frame.to_csv(output_dir / f"{policy}_results.csv", index=False)
        if frame.empty:
            continue
        frame.groupby(["ensemble", "uptake_mode", "sigma_source", "maxent"])[
            ["recovery_percent", "ess_percent", "val_mse"]].agg(["mean", "std", "count"]).to_csv(
                output_dir / f"{policy}_summary.csv")
        for metric in ("recovery_percent", "ess_percent"):
            plot_metric(frame, metric, output_dir / "plots" / f"{metric}_vs_maxent_{policy}")
    if extend_results:
        with np.load(extend_results / "frame_weights.npz") as archive:
            if set(archive.files) & set(weights):
                raise ValueError("Overlapping saved frame-weight keys")
            weights.update({key: archive[key] for key in archive.files})
    np.savez_compressed(output_dir / "frame_weights.npz", **weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "_maxent_weighted_mse_sidecar")
    for name, default in (("features", shrinkage.DEFAULT_FEATURES_DIR),
                          ("datasplit", shrinkage.DEFAULT_DATASPLIT_DIR),
                          ("clustering", shrinkage.DEFAULT_CLUSTERING_DIR),
                          ("trajectory", source.DEFAULT_TRAJECTORY_DIR)):
        parser.add_argument(f"--{name}-dir", type=Path, default=default)
    parser.add_argument("--maxent-values", default=",".join(map(str, DEFAULT_MAXENT)))
    parser.add_argument("--ensembles", nargs="+", choices=ENSEMBLES, default=list(ENSEMBLES))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--variance-floor", type=float, default=1e-4,
                        help="Population variance floor in uptake-fraction squared; 1e-4 = 1%% SD")
    parser.add_argument("--precision-normalization", choices=("global", "per_peptide"), default="global")
    parser.add_argument("--reuse-baselines", type=Path,
                        help="Reuse verified stable-KL MSE/coordinate histories read-only; fit only Weighted-MSE")
    parser.add_argument("--extend-results", type=Path,
                        help="Append disjoint MaxEnt values to a completed campaign's tables and plots")
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="all")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        run_fit(RunSpec(**json.loads(args.worker_spec.read_text())))
        return
    args.maxent_values = list(shrinkage.parse_csv(args.maxent_values, float))
    if (len(set(args.ensembles)) != len(args.ensembles)
            or not args.maxent_values or len(set(args.maxent_values)) != len(args.maxent_values)
            or any(not np.isfinite(x) or x <= 0 for x in args.maxent_values)
            or min(args.n_splits, args.n_steps, args.jobs) < 1
            or not np.isfinite(args.variance_floor) or args.variance_floor <= 0):
        parser.error("Sweep values must be unique positive finite values; counts/floor must be positive")
    for name in ("output_dir", "features_dir", "datasplit_dir", "clustering_dir", "trajectory_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.reuse_baselines:
        args.reuse_baselines = args.reuse_baselines.resolve()
    previous_count = 0
    if args.extend_results:
        args.extend_results = args.extend_results.resolve()
        previous_count = validate_extension(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"phase", "jobs", "worker_spec"}}
    settings["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in (
        Path(__file__), Path(source.__file__), Path(shrinkage.__file__), HERE / "sidecar_selection.py",
        HERE.parents[2] / "common" / "optimization.py",
        HERE.parents[3] / "src" / "opt" / "losses.py",
        HERE.parents[3] / "src" / "opt" / "run.py",
        HERE.parents[3] / "src" / "opt" / "chunk.py"))).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings:
            raise ValueError("Settings/code changed; use a fresh output directory")
        specs = [RunSpec(**s) for s in manifest["specs"]]
    else:
        paths, _ = source.prepare_sigma_artifacts(args.output_dir, args.features_dir,
            args.clustering_dir, args.trajectory_dir, args.ensembles, LOSSES[1:3], (0.,), 1e8)
        specs = build_specs(args, paths, prepare_weights(args))
        if args.reuse_baselines:
            specs = reuse_baselines(specs, args.reuse_baselines)
        manifest = dict(version=VERSION, settings=settings, expected_fits=len(specs),
            selection_metric=source.SELECTION_POLICY,
            combined_expected_fits=len(specs) + previous_count,
            sigma_shrinkage=0., variance="oracle population frame-wise peptide/time uptake variance",
            variance_floor_interpretation="regularization floor, not measured experimental noise",
            precision_normalization=args.precision_normalization,
            new_fits=sum(Path(s.output_dir) == args.output_dir for s in specs),
            reused_fits=sum(Path(s.output_dir) != args.output_dir for s in specs),
            specs=[dataclasses.asdict(s) for s in specs])
        manifest_path.write_text(json.dumps(manifest, indent=2))
        pd.DataFrame([dataclasses.asdict(s) | dict(run_id=s.run_id) for s in specs]).to_csv(
            args.output_dir / "run_grid.csv", index=False)
    print(f"Grid: {len(specs)} fits, {args.jobs} workers; output: {args.output_dir}", flush=True)
    if args.phase in ("fit", "all"):
        execute(specs, args.output_dir, args.jobs)
    if args.phase in ("analyze", "all"):
        analyze(specs, args.output_dir, args.extend_results)
    manifest["completed_fits"] = sum(run_is_complete(s) for s in specs)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
