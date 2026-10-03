#!/usr/bin/env python3
"""ISO TRI calibrated Linear-BV: MaxEnt/OMC strength-bandwidth sweep.

Two panels use five frozen interval offsets fitted to the mean (across all
modelled residues) frame-wise uptake curve under either uniform or GT frame
weights. BV coefficients remain 0.35 and 2.0. Training is MSE; selection is
closed-coordinate Sigma-MSE alpha=0 over native convergence checkpoints.
OMC uses the all-pairs Work Scale graph. Default: 288 fits, ten workers.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import run_maxent_omc_strength_sidecar as strength
from sidecar_selection import SELECTION_POLICY, select_best_rows

base, source, shrinkage, plt = strength.base, strength.source, strength.shrinkage, strength.plt
HERE = Path(__file__).resolve().parent
VERSION = "tri_linear_mean_offset_strength_omc_allpairs_v1"
CALIBRATIONS = ("uniform_mean", "gt_mean")
CALIBRATION_LABELS = {"uniform_mean": "Offsets fitted to uniform mean",
                      "gt_mean": "Offsets fitted to GT mean"}


@dataclasses.dataclass(frozen=True)
class RunSpec(strength.RunSpec):
    calibration_source: str = "uniform_mean"
    interval_offsets: tuple[float, ...] = ()

    @property
    def panel(self):
        return CALIBRATION_LABELS[self.calibration_source]

    @property
    def run_id(self):
        token = shrinkage.alpha_token
        kernel = f"_h{token(self.kernel_bandwidth)}" if self.method == "omc" else ""
        return (f"{VERSION}_{self.calibration_source}_{self.method}{kernel}"
                f"_S{token(self.sweep_maxent)}_split{self.split_idx:03d}")


def fit_mean_offsets(features, assignments, frame_weights):
    """Fit five offsets to five residue-mean standard-uptake predictions."""
    standard = shrinkage.configure_model("uptake", assignments)
    linear = shrinkage.configure_model("linear", assignments)
    weights = np.asarray(frame_weights, dtype=float)
    weights /= weights.sum()
    target = np.asarray(shrinkage.predict_uptake(standard, features, weights), dtype=float).mean(axis=1)
    forward = linear.forward[shrinkage.m_key("HDX_peptide")]
    weights_jax = jnp.asarray(weights)

    def prediction(offsets):
        params = dataclasses.replace(linear.params, interval_offsets=offsets)
        return jnp.mean(forward.average_frames(features, params, weights_jax).uptake, axis=1)

    compiled = jax.jit(prediction)
    jacobian = jax.jit(jax.jacrev(prediction))
    predict_np = lambda x: np.asarray(compiled(jnp.asarray(x)), dtype=float)
    jac_np = lambda x: np.asarray(jacobian(jnp.asarray(x)), dtype=float)
    initial = np.zeros(len(shrinkage.TIMEPOINTS), dtype=float)
    fit = least_squares(lambda x: predict_np(x)-target, initial, jac=jac_np,
                        bounds=(-12., 12.), xtol=1e-13, ftol=1e-13, gtol=1e-13,
                        max_nfev=5000)
    fitted = predict_np(fit.x)
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Mean offset fit failed: {fit.message}")
    return np.asarray(fit.x), dict(target_mean=target, default_mean=predict_np(initial),
        fitted_mean=fitted, residual=fitted-target,
        rmse_default=float(np.sqrt(np.mean((predict_np(initial)-target)**2))),
        rmse_fitted=float(np.sqrt(np.mean((fitted-target)**2))),
        optimality=float(fit.optimality), nfev=int(fit.nfev))


def prepare_calibrations(args):
    features, _ = shrinkage.load_features(args.features_dir, "ISO_TRI")
    assignments = shrinkage.load_cluster_assignments(args.clustering_dir, "ISO_TRI")
    populations = {"uniform_mean": np.ones(len(assignments))/len(assignments),
                   "gt_mean": source.population_weights(assignments, "gt")}
    result, rows = {}, []
    payload = {"definition": "five global interval offsets fitted to residue-mean frame-wise uptake",
               "bv_bc": .35, "bv_bh": 2., "timepoints_min": list(map(float, shrinkage.TIMEPOINTS))}
    for name, weights in populations.items():
        offsets, diagnostics = fit_mean_offsets(features, assignments, weights)
        result[name] = tuple(map(float, offsets))
        payload[name] = dict(interval_offsets=list(map(float, offsets)),
                             interval_multipliers=list(map(float, np.exp(offsets))),
                             frame_weights="uniform" if name == "uniform_mean" else "GT 0.4 open / 0.6 closed",
                             **{key: value.tolist() if isinstance(value, np.ndarray) else value
                                for key, value in diagnostics.items()})
        for time, offset, multiplier, target, default, fitted in zip(
                shrinkage.TIMEPOINTS, offsets, np.exp(offsets), diagnostics["target_mean"],
                diagnostics["default_mean"], diagnostics["fitted_mean"]):
            rows.append(dict(calibration_source=name, timepoint_min=time, interval_offset=offset,
                interval_multiplier=multiplier, target_mean=target,
                default_linear_mean=default, fitted_linear_mean=fitted))
    payload["delta_gt_minus_uniform"] = list(map(float,
        np.asarray(result["gt_mean"])-np.asarray(result["uniform_mean"])))
    (args.output_dir/"offset_calibrations.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(rows).to_csv(args.output_dir/"offset_calibrations.csv", index=False)
    return result


def build_specs(args, calibrations):
    return [RunSpec(ensemble="ISO_TRI", sigma_source="mse", alpha=0.,
        split_type="sequence_cluster", split_idx=split, sigma_path="",
        output_dir=str(args.output_dir), features_dir=str(args.features_dir),
        datasplit_dir=str(args.datasplit_dir), clustering_dir=str(args.clustering_dir),
        n_steps=args.n_steps, learning_rate=1., ema_alpha=.5, forward_model_scaling=1000.,
        execution_mode="compiled", method=method, sweep_maxent=s,
        omc_strength=1/s if method == "omc" else 0., uptake_mode="linear",
        graph_k=0, graph_path="", kernel_bandwidth=h,
        calibration_source=calibration, interval_offsets=calibrations[calibration])
        for calibration in CALIBRATIONS
        for method, h in [("maxent", 1.)]+[("omc", h) for h in args.bandwidths]
        for s in args.strengths for split in range(args.n_splits)]


def run_fit(spec):
    strength.run_fit(spec)
    config = json.loads(spec.config_path.read_text())
    config["sidecar_settings"].update(calibration_source=spec.calibration_source,
        interval_offsets=list(spec.interval_offsets), interval_multipliers=list(np.exp(spec.interval_offsets)),
        offset_fit_reduction="mean_across_all_modelled_residues")
    spec.config_path.write_text(json.dumps(config, indent=2))


def execute(specs, directory, jobs):
    for name in ("specs", "logs"):
        (directory/name).mkdir(exist_ok=True)
    pending = []
    for spec in specs:
        path = directory/"specs"/f"{spec.run_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(spec), indent=2))
        if not strength.run_is_complete(spec):
            pending.append(path)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(source._worker_command, Path(__file__), path,
                               directory/"logs"/f"{path.stem}.log") for path in pending]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, code = future.result(); print(f"[{i}/{len(pending)}] {name}: {code}", flush=True)
            if code: failures.append(name)
    if failures: raise RuntimeError(f"Failed fits: {failures}")


def plot_metric(frame, metric, output):
    bandwidths = sorted(frame.loc[frame.method == "omc", "bandwidth"].unique())
    colors = dict(zip(bandwidths, plt.get_cmap("viridis")(np.linspace(.08, .9, len(bandwidths)))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    for axis, calibration in zip(axes, CALIBRATIONS):
        for method, bandwidth in [("maxent", None)]+[("omc", h) for h in bandwidths]:
            rows = frame[(frame.calibration_source == calibration) & (frame.method == method)]
            if bandwidth is not None: rows = rows[rows.bandwidth == bandwidth]
            color = "black" if method == "maxent" else colors[bandwidth]
            label = "MaxEnt" if method == "maxent" else f"OMC h={bandwidth:g}"
            for _, trace in rows.groupby("split_idx"):
                trace = trace.sort_values("data_strength")
                axis.plot(trace.data_strength, trace[metric], color=color, alpha=.18, lw=.7)
            mean = rows.groupby("data_strength")[metric].mean().sort_index()
            axis.plot(mean.index, mean, "o-", color=color, label=label,
                      lw=2.3 if method == "maxent" else 1.5)
        axis.set(title=CALIBRATION_LABELS[calibration], xscale="log",
                 xlabel="Data-fit strength S (regularizer coefficient = 1/S)")
        if metric != "val_closed_sigma_mse": axis.set_ylim(0, 102)
        elif bool((frame[metric] > 0).all()): axis.set_yscale("log")
        axis.grid(alpha=.2)
    labels = {"recovery_percent":"Recovery (%)", "ess_percent":"ESS (%)",
              "intermediate_percent":"Intermediate population weight (%)",
              "val_closed_sigma_mse":"Closed-coordinate Sigma validation error"}
    axes[0].set_ylabel(labels[metric])
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False)
    policy = "Final step" if "final_step" in output.name else "Selected by closed-coordinate Sigma-MSE (alpha=0)"
    fig.suptitle(f"ISO TRI · calibrated Linear-BV · MSE training · all-pairs Work Scale OMC\n{policy}")
    fig.subplots_adjust(bottom=.25, top=.82, wspace=.08)
    output.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"): fig.savefig(output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(specs, directory):
    cache, conv, final, incomplete, missing, weights = {}, [], [], [], [], {}
    for spec in specs:
        if not strength.run_is_complete(spec):
            incomplete.append(dict(run_id=spec.run_id, reason="missing_empty_or_nonfinite_history")); continue
        history = source.load_optimization_history_from_file(str(spec.history_path))
        context = base.score_context(spec, cache)
        labeled = source.iter_labeled_convergence_states(history)
        if not labeled: missing.append(dict(run_id=spec.run_id, reason="no_convergence_states"))
        states = [(x.state, "convergence_state", x.rank, x.threshold) for x in labeled]
        states.append((history.states[-1], "final_optimization_state", -1, np.nan))
        for state, policy, rank, threshold in states:
            row, w = base.score(spec, state, policy, context)
            assignments = np.asarray(context[1], dtype=int)
            row.update(calibration_source=spec.calibration_source,
                interval_offsets=json.dumps(list(spec.interval_offsets)),
                data_strength=spec.sweep_maxent, regularizer_coefficient=1/spec.sweep_maxent,
                curve=spec.curve, sweep_x=spec.sweep_maxent,
                intermediate_percent=100*float(np.asarray(w)[assignments == -1].sum()),
                convergence_rank=rank, convergence_threshold=threshold)
            (final if rank == -1 else conv).append(row); weights[f"{spec.run_id}__{rank}"] = w
    for name, rows in (("incomplete_runs", incomplete), ("missing_selection", missing)):
        pd.DataFrame(rows, columns=["run_id", "reason"]).to_csv(directory/f"{name}.csv", index=False)
    if not final: raise RuntimeError("No valid histories")
    convergence = pd.DataFrame(conv); convergence.to_csv(directory/"convergence_scores.csv", index=False)
    selected = select_best_rows(convergence) if not convergence.empty else pd.DataFrame(columns=final[0])
    for policy, frame in (("selected", selected), ("final_step", pd.DataFrame(final))):
        frame.to_csv(directory/f"{policy}_results.csv", index=False)
        if frame.empty: continue
        frame.groupby(["calibration_source", "curve", "data_strength"])[
            ["recovery_percent", "ess_percent", "intermediate_percent", "val_mse", "val_closed_sigma_mse"]
        ].agg(["mean", "std", "count"]).to_csv(directory/f"{policy}_summary.csv")
        for metric in ("recovery_percent", "ess_percent", "intermediate_percent", "val_closed_sigma_mse"):
            plot_metric(frame, metric, directory/"plots"/f"{metric}_vs_strength_{policy}")
    np.savez_compressed(directory/"frame_weights.npz", **weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"_maxent_omc_calibrated_linear_jobs10")
    for name, default in (("features", shrinkage.DEFAULT_FEATURES_DIR), ("datasplit", shrinkage.DEFAULT_DATASPLIT_DIR),
                          ("clustering", shrinkage.DEFAULT_CLUSTERING_DIR), ("trajectory", source.DEFAULT_TRAJECTORY_DIR)):
        parser.add_argument(f"--{name}-dir", type=Path, default=default)
    parser.add_argument("--strengths", default=",".join(map(str, strength.STRENGTHS)))
    parser.add_argument("--bandwidths", default=",".join(map(str, strength.BANDWIDTHS)))
    parser.add_argument("--n-splits", type=int, default=3); parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--phase", choices=("prepare","fit","analyze","all"), default="all")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec: run_fit(RunSpec(**json.loads(args.worker_spec.read_text()))); return
    for name in ("strengths", "bandwidths"):
        values=list(shrinkage.parse_csv(getattr(args,name),float))
        if not values or len(set(values)) != len(values) or any(not np.isfinite(x) or x<=0 for x in values):
            parser.error(f"Invalid {name}")
        setattr(args,name,values)
    if min(args.n_splits,args.n_steps,args.jobs)<1: parser.error("Counts must be positive")
    for name in ("output_dir","features_dir","datasplit_dir","clustering_dir","trajectory_dir"):
        setattr(args,name,getattr(args,name).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in {"phase","jobs","worker_spec"}}
    files=[Path(__file__),Path(strength.__file__),Path(base.__file__),Path(source.__file__),Path(shrinkage.__file__),
           HERE/"sidecar_selection.py",HERE.parents[2]/"common/optimization.py"]
    settings["code_sha256"]=hashlib.sha256(b"".join(x.read_bytes() for x in files)).hexdigest()
    manifest_path=args.output_dir/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["settings"] != settings:
        raise ValueError("Settings/code changed; use a new output directory")
    calibrations=prepare_calibrations(args); specs=build_specs(args,calibrations)
    manifest=dict(version=VERSION,settings=settings,expected_fits=len(specs),training_loss="MSE",
        selection_metric=SELECTION_POLICY,selection_sigma_shrinkage=0.,graph="all_pairs_work_scale",
        objective_coefficients="MSE + regularizer/S",offsets_frozen=True,offset_fit_reduction="mean_all_modelled_residues",
        specs=[dataclasses.asdict(s) for s in specs])
    manifest_path.write_text(json.dumps(manifest,indent=2))
    pd.DataFrame([dataclasses.asdict(s)|dict(run_id=s.run_id,data_strength=s.sweep_maxent,
        regularizer_coefficient=1/s.sweep_maxent,curve=s.curve) for s in specs]).to_csv(args.output_dir/"run_grid.csv",index=False)
    print(f"Grid: {len(specs)} fits, {args.jobs} workers; output: {args.output_dir}",flush=True)
    if args.phase in ("fit","all"): execute(specs,args.output_dir,args.jobs)
    if args.phase in ("analyze","all"): analyze(specs,args.output_dir)
    manifest["completed_fits"]=sum(strength.run_is_complete(s) for s in specs)
    manifest_path.write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
