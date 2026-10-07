#!/usr/bin/env python3
"""ISO TRI: MSE-only MaxEnt/OMC, shared data-strength axis, bandwidth hues.

Objective coefficients: MSE + (1/S)*regularizer. S=10^1..10^6.
OMC is original weight-weighted Laplacian, all-pairs Work Scale Gaussian;
bandwidth=10^-3..10^3 independently of S. No KL in OMC fits.
Linear-BV and frame-wise uptake panels; closed-coordinate alpha=0 selection
over native convergence checkpoints, plus separate final-step plots.
Default: 288 native fits, ten workers. kNN is deliberately out of scope.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_maxent_omc_sidecar as base
from sidecar_selection import SELECTION_POLICY, select_best_rows

source, shrinkage, plt = base.source, base.shrinkage, base.plt
HERE = Path(__file__).resolve().parent
VERSION = "tri_mse_strength_omc_allpairs_v1"
MODES = ("linear", "uptake")
STRENGTHS = tuple(10.**i for i in range(1, 7))
BANDWIDTHS = tuple(10.**i for i in range(-3, 4))


@dataclasses.dataclass(frozen=True)
class RunSpec(base.RunSpec):
    kernel_bandwidth: float = 1.

    @property
    def bandwidth(self):
        return self.kernel_bandwidth if self.method == "omc" else 1/self.sweep_maxent

    @property
    def panel(self):
        return "Linear BV" if self.uptake_mode == "linear" else "Frame-wise uptake"

    @property
    def curve(self):
        return "MaxEnt" if self.method == "maxent" else f"OMC h={self.kernel_bandwidth:g}"

    @property
    def run_id(self):
        token = shrinkage.alpha_token
        kernel = f"_h{token(self.kernel_bandwidth)}" if self.method == "omc" else ""
        return (f"{VERSION}_{self.uptake_mode}_{self.method}{kernel}"
                f"_S{token(self.sweep_maxent)}_split{self.split_idx:03d}")


def build_specs(args):
    return [RunSpec(ensemble="ISO_TRI", sigma_source="mse", alpha=0.,
        split_type="sequence_cluster", split_idx=split, sigma_path="",
        output_dir=str(args.output_dir), features_dir=str(args.features_dir),
        datasplit_dir=str(args.datasplit_dir), clustering_dir=str(args.clustering_dir),
        n_steps=args.n_steps, learning_rate=1., ema_alpha=.5, forward_model_scaling=1000.,
        execution_mode="compiled", method=method, sweep_maxent=strength,
        omc_strength=1/strength if method == "omc" else 0., uptake_mode=mode,
        graph_k=0, graph_path="", kernel_bandwidth=bandwidth)
        for mode in MODES for method, bandwidth in [("maxent", 1.)]+[("omc", h) for h in args.bandwidths]
        for strength in args.strengths for split in range(args.n_splits)]


def run_fit(spec):
    # Native forward models, optimizer and original OMC implementation are reused.
    # Only the previously coupled sweep parameters are now independent.
    base.run_fit(spec)
    config = json.loads(spec.config_path.read_text())
    config["sidecar_settings"].update(data_strength=spec.sweep_maxent,
        regularizer_coefficient=1/spec.sweep_maxent, sigma_shrinkage=None,
        selection_sigma_shrinkage=0., selection_metric=SELECTION_POLICY,
        training_loss="MSE", graph="all_pairs_work_scale", kernel_normalization=False)
    spec.config_path.write_text(json.dumps(config, indent=2))
    if not run_is_complete(spec):
        raise RuntimeError(f"Missing, empty or nonfinite terminal history: {spec.run_id}")


def run_is_complete(spec):
    if not spec.history_path.exists() or not spec.config_path.exists():
        return False
    try:
        history = source.load_optimization_history_from_file(str(spec.history_path))
        return bool(history.states) and bool(np.isfinite(float(history.states[-1].losses.total_train_loss)))
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def execute(specs, directory, jobs):
    for name in ("specs", "logs"):
        (directory/name).mkdir(exist_ok=True)
    pending = []
    for spec in specs:
        path = directory / "specs" / f"{spec.run_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(spec), indent=2))
        if not run_is_complete(spec):
            pending.append(path)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(source._worker_command, Path(__file__), path,
                               directory/"logs"/f"{path.stem}.log") for path in pending]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, code = future.result()
            print(f"[{i}/{len(pending)}] {name}: return code {code}", flush=True)
            if code:
                failures.append(name)
    if failures:
        raise RuntimeError(f"Failed fits: {failures}")


def plot_metric(frame, metric, output):
    bandwidths = sorted(frame.loc[frame.method == "omc", "bandwidth"].unique())
    colors = dict(zip(bandwidths, plt.get_cmap("viridis")(np.linspace(.08, .9, len(bandwidths)))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)
    for axis, mode in zip(axes, MODES):
        for method, bandwidth in [("maxent", None)]+[("omc", h) for h in bandwidths]:
            rows = frame[(frame.uptake_mode == mode) & (frame.method == method)]
            if bandwidth is not None:
                rows = rows[rows.bandwidth == bandwidth]
            color = "black" if method == "maxent" else colors[bandwidth]
            label = "MaxEnt" if method == "maxent" else f"OMC h={bandwidth:g}"
            for _, trace in rows.groupby("split_idx"):
                trace = trace.sort_values("data_strength")
                axis.plot(trace.data_strength, trace[metric], color=color, alpha=.18, lw=.7)
            mean = rows.groupby("data_strength")[metric].mean().sort_index()
            axis.plot(mean.index, mean, "o-", color=color, label=label,
                      lw=2.3 if method == "maxent" else 1.5, zorder=10 if method == "maxent" else 3)
        axis.set(title="Linear BV" if mode == "linear" else "Frame-wise uptake",
                 xscale="log", xlabel="Data-fit strength S (regularizer coefficient = 1/S)")
        if metric != "val_closed_sigma_mse":
            axis.set_ylim(0, 102)
        elif bool((frame[metric] > 0).all()):
            axis.set_yscale("log")
        axis.grid(alpha=.2)
    ylabel = {"recovery_percent": "Recovery (%)", "ess_percent": "ESS (%)",
              "intermediate_percent": "Intermediate population weight (%)",
              "val_closed_sigma_mse": "Closed-coordinate Sigma validation error"}[metric]
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    policy = "Final step" if "final_step" in output.name else "Selected by closed-coordinate Sigma-MSE (alpha=0)"
    fig.suptitle(f"ISO TRI · MSE training · all-pairs Work Scale OMC\n{policy}")
    fig.subplots_adjust(bottom=.25, top=.82, wspace=.08)
    output.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(specs, directory):
    cache, convergence, final, incomplete, missing, weights = {}, [], [], [], [], {}
    for spec in specs:
        if not run_is_complete(spec):
            incomplete.append(dict(run_id=spec.run_id, reason="missing_empty_or_nonfinite_history"))
            continue
        history = source.load_optimization_history_from_file(str(spec.history_path))
        context = base.score_context(spec, cache)
        labeled = source.iter_labeled_convergence_states(history)
        if not labeled:
            missing.append(dict(run_id=spec.run_id, reason="no_convergence_states"))
        states = [(s.state, "convergence_state", s.rank, s.threshold) for s in labeled]
        states.append((history.states[-1], "final_optimization_state", -1, np.nan))
        for state, policy, rank, threshold in states:
            row, w = base.score(spec, state, policy, context)
            assignments = np.asarray(context[1], dtype=int)
            row.update(data_strength=spec.sweep_maxent, regularizer_coefficient=1/spec.sweep_maxent,
                       curve=spec.curve, sweep_x=spec.sweep_maxent,
                       intermediate_percent=100*float(np.asarray(w)[assignments == -1].sum()),
                       convergence_rank=rank, convergence_threshold=threshold)
            (final if rank == -1 else convergence).append(row)
            weights[f"{spec.run_id}__{rank}"] = w
    for name, rows in (("incomplete_runs", incomplete), ("missing_selection", missing)):
        pd.DataFrame(rows, columns=["run_id", "reason"]).to_csv(directory/f"{name}.csv", index=False)
    if not final:
        raise RuntimeError("No valid histories to analyze")
    conv = pd.DataFrame(convergence)
    conv.to_csv(directory/"convergence_scores.csv", index=False)
    selected = select_best_rows(conv) if not conv.empty else pd.DataFrame(columns=final[0])
    for policy, frame in (("selected", selected), ("final_step", pd.DataFrame(final))):
        frame.to_csv(directory/f"{policy}_results.csv", index=False)
        if frame.empty:
            continue
        frame.groupby(["uptake_mode", "curve", "data_strength"])[
            ["recovery_percent", "ess_percent", "intermediate_percent", "val_mse", "val_closed_sigma_mse"]
        ].agg(["mean", "std", "count"]).to_csv(directory/f"{policy}_summary.csv")
        for metric in ("recovery_percent", "ess_percent", "intermediate_percent",
                       "val_closed_sigma_mse"):
            plot_metric(frame, metric, directory/"plots"/f"{metric}_vs_strength_{policy}")
    np.savez_compressed(directory/"frame_weights.npz", **weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"_maxent_omc_strength_allpairs_jobs10")
    for name, default in (("features", shrinkage.DEFAULT_FEATURES_DIR),
                          ("datasplit", shrinkage.DEFAULT_DATASPLIT_DIR),
                          ("clustering", shrinkage.DEFAULT_CLUSTERING_DIR),
                          ("trajectory", source.DEFAULT_TRAJECTORY_DIR)):
        parser.add_argument(f"--{name}-dir", type=Path, default=default)
    parser.add_argument("--strengths", default=",".join(map(str, STRENGTHS)))
    parser.add_argument("--bandwidths", default=",".join(map(str, BANDWIDTHS)))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="all")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        run_fit(RunSpec(**json.loads(args.worker_spec.read_text()))); return
    for name in ("strengths", "bandwidths"):
        values = list(shrinkage.parse_csv(getattr(args, name), float))
        if not values or len(set(values)) != len(values) or any(not np.isfinite(x) or x <= 0 for x in values):
            parser.error(f"{name} must contain unique positive finite values")
        setattr(args, name, values)
    if min(args.n_splits, args.n_steps, args.jobs) < 1:
        parser.error("Counts must be positive")
    for name in ("output_dir", "features_dir", "datasplit_dir", "clustering_dir", "trajectory_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                if k not in {"phase", "jobs", "worker_spec"}}
    files = [Path(__file__), Path(base.__file__), Path(source.__file__), Path(shrinkage.__file__),
             HERE/"sidecar_selection.py", HERE.parents[2]/"common/optimization.py"]
    files += [HERE.parents[3]/"src/opt"/name for name in
              ("losses.py", "run.py", "chunk.py", "loss/original_omc_laplacian.py")]
    settings["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    manifest_path = args.output_dir/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["settings"] != settings:
        raise ValueError("Settings/code changed; use a new output directory")
    specs = build_specs(args)
    manifest = dict(version=VERSION, settings=settings, expected_fits=len(specs),
        training_loss="MSE", selection_metric=SELECTION_POLICY, selection_sigma_shrinkage=0.,
        graph="all_pairs_work_scale", objective_coefficients="MSE + regularizer/S",
        bandwidth_independent_of_strength=True, omc_replaces_kl=True,
        specs=[dataclasses.asdict(s) for s in specs])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame([dataclasses.asdict(s)|dict(run_id=s.run_id, data_strength=s.sweep_maxent,
        regularizer_coefficient=1/s.sweep_maxent, curve=s.curve) for s in specs]).to_csv(
            args.output_dir/"run_grid.csv", index=False)
    print(f"Grid: {len(specs)} fits, {args.jobs} workers; output: {args.output_dir}", flush=True)
    if args.phase in ("fit", "all"):
        execute(specs, args.output_dir, args.jobs)
    if args.phase in ("analyze", "all"):
        analyze(specs, args.output_dir)
    manifest["completed_fits"] = sum(run_is_complete(s) for s in specs)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
