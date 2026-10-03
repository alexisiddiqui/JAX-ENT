#!/usr/bin/env python3
"""ISO TRI: MaxEnt versus original OMC, Work Scale/all pairs, Sigma alpha=0.1.

Run with --phase all --jobs 4; --phase prepare only writes the experiment grid.
Default: 252 fits (4 panels x 3 losses x 7 sweep values x 3 splits).
The shared x axis is 1/maxent = bandwidth; OMC strength is a separate coefficient.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import run_sigma_source_sidecar as source
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import run_optimization
from jaxent.src.opt.loss.original_omc_laplacian import (
    build_omc_kernel,
    create_original_omc_loss,
)
from jaxent.src.opt.loss.graph_laplacian import FrameGraph, build_frame_graph_from_distances

shrinkage = source.shrinkage
plt = source.plt
HERE = Path(__file__).resolve().parent
COMPARISONS = ("mse", "gt_coordinate", "closed_coordinate")
LABELS = dict(source.SOURCE_LABELS, mse="MSE")
COLORS = {"mse": "#009E73", "gt_coordinate": "#0072B2", "closed_coordinate": "#CC79A7"}
DEFAULT_MAXENT = (0.01, 0.1, 1., 10., 100., 1000., 10000.)
OMC_STRENGTHS = (1., 0.1, 0.01)
VERSION = "tri_maxent_omc_framewise_v1"


@dataclasses.dataclass(frozen=True)
class RunSpec(source.RunSpec):
    method: str
    sweep_maxent: float
    omc_strength: float
    uptake_mode: str = "uptake"
    graph_k: int = 0
    graph_path: str = ""

    @property
    def averaging_label(self):
        return "linear_uptake" if self.uptake_mode == "linear" else "frame_uptake"

    @property
    def bandwidth(self):
        return 1. / self.sweep_maxent

    @property
    def panel(self):
        return "MaxEnt" if self.method == "maxent" else f"OMC strength {self.omc_strength:g}"

    @property
    def run_id(self):
        token = shrinkage.alpha_token
        version = VERSION.replace("framewise", "linear_bv") if self.uptake_mode == "linear" else VERSION
        if self.graph_k:
            version += f"_logpf_knn{self.graph_k}"
        return (f"{version}_{self.method}_lambda{token(self.omc_strength)}_"
                f"{self.sigma_source}_M{token(self.sweep_maxent)}_split{self.split_idx:03d}")


def work_distances(features, model):
    """Repository Work Scale convention: absolute difference of mean log PF.

    Dimensionless (work/RT); no quantile scaling, pruning or coupling matching.
    """
    z = (np.asarray(model.params.bv_bc).item() * np.asarray(features.heavy_contacts)
         + np.asarray(model.params.bv_bh).item() * np.asarray(features.acceptor_contacts))
    values = z.mean(axis=0)
    return np.abs(values[:, None] - values[None, :])


def regularizer(spec, features, model):
    if spec.method == "maxent":
        return None
    if spec.graph_k:
        with np.load(spec.graph_path) as archive:
            if int(archive["n_nodes"]) != features.features_shape[1]:
                raise ValueError("Precomputed graph does not match feature frame count")
            if int(archive["k"]) != spec.graph_k:
                raise ValueError("Precomputed graph k mismatch")
            edge_weights = np.exp(-np.minimum(
                (archive["work_edge_distances"] / spec.bandwidth)**2 / 2., 80.))
            graph = FrameGraph(
                edge_sources=jnp.asarray(archive["edge_sources"]),
                edge_targets=jnp.asarray(archive["edge_targets"]),
                edge_weights=jnp.asarray(edge_weights), n_nodes=int(archive["n_nodes"]),
                metric="logpf_l2_neighbours_work_scale_weights", k=spec.graph_k,
            )
        return ("original_omc_laplacian_knn", sparse_omc_loss, graph, spec.omc_strength)
    # Freeze the same structural Work Scale metric across forward constructions.
    graph_model = (shrinkage.configure_model("uptake", np.empty(0, dtype=int))
                   if spec.uptake_mode == "linear" else model)
    kernel = build_omc_kernel(work_distances(features, graph_model),
                              bandwidth=spec.bandwidth, metric="work_scale_all_pairs")
    return ("original_omc_laplacian", create_original_omc_loss(normalise=False),
            kernel, spec.omc_strength)


def configure_spec_model(spec, assignments):
    """Configure the native forward model and apply optional frozen offsets."""
    model = shrinkage.configure_model(spec.uptake_mode, assignments)
    offsets = getattr(spec, "interval_offsets", ())
    if offsets:
        if spec.uptake_mode != "linear":
            raise ValueError("Interval offsets are only defined for Linear-BV")
        model.params = dataclasses.replace(
            model.params, interval_offsets=jnp.asarray(offsets))
    return model


def sparse_omc_loss(model, graph, prediction_index):
    """Original OMC energy over canonical undirected edges (no extra 1/2)."""
    del prediction_index
    w = model.params.frame_weight_simplex
    left, right = w[graph.edge_sources], w[graph.edge_targets]
    value = graph.n_nodes**2 * jnp.sum(graph.edge_weights * left * right * (left-right)**2)
    return value, value


def prepare_neighbours(args):
    """Precompute full-profile L2 distances and a fixed symmetric-union kNN graph."""
    if not args.graph_k:
        return ""
    features, _ = shrinkage.load_features(args.features_dir, "ISO_TRI")
    model = shrinkage.configure_model("uptake", np.empty(0, dtype=int))
    z = (np.asarray(model.params.bv_bc).item() * np.asarray(features.heavy_contacts, dtype=float)
         + np.asarray(model.params.bv_bh).item() * np.asarray(features.acceptor_contacts, dtype=float))
    fingerprint = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
    path = args.output_dir / f"ISO_TRI_logpf_l2_knn{args.graph_k}.npz"
    if path.exists():
        with np.load(path) as archive:
            if str(archive["logpf_sha256"]) != fingerprint or int(archive["k"]) != args.graph_k:
                raise ValueError("Precomputed graph input changed; use a new output directory")
        return str(path)
    distances = squareform(pdist(z.T, metric="euclidean"))
    graph = build_frame_graph_from_distances(distances, k=args.graph_k, metric="logpf_l2",
                                             weighted=False, symmetrization="union")
    left, right = np.asarray(graph.edge_sources), np.asarray(graph.edge_targets)
    scalar = z.mean(axis=0)
    np.savez_compressed(path, distances=distances, edge_sources=left, edge_targets=right,
                        work_edge_distances=np.abs(scalar[left]-scalar[right]),
                        n_nodes=z.shape[1], k=args.graph_k, logpf_sha256=fingerprint)
    degree = np.bincount(np.concatenate([left, right]), minlength=z.shape[1])
    (args.output_dir / "graph_metadata.json").write_text(json.dumps(dict(
        k=args.graph_k, n_nodes=z.shape[1], edges=len(left), symmetrization="union",
        distance="Euclidean L2 over residue log-PF profiles", edge_metric="Work Scale",
        degree_min=int(degree.min()), degree_max=int(degree.max()), degree_mean=float(degree.mean()),
        logpf_sha256=fingerprint,
    ), indent=2))
    return str(path)


def run_fit(spec):
    features, topology = shrinkage.load_features(spec.features_dir, spec.ensemble)
    assignments = shrinkage.load_cluster_assignments(spec.clustering_dir, spec.ensemble)
    model = configure_spec_model(spec, assignments)
    train, val = shrinkage.load_split(spec.datasplit_dir, spec.split_type, spec.split_idx)
    precision = None
    if spec.sigma_source != "mse":
        with np.load(spec.sigma_path) as archive:
            precision = jnp.asarray(archive["Sigma_inv_normalized"])
    run_optimization(
        train_data=train, val_data=val,
        prior_data=shrinkage.build_prior_dataset(model, features, topology),
        features=features, feature_top=topology, forward_model=model,
        model_parameters=model.params, convergence=list(shrinkage.CONVERGENCE_RATES),
        loss_config=LossConfig(
            primary_loss=("hdx_uptake_MSE_loss" if spec.sigma_source == "mse"
                          else "hdx_uptake_sigma_MSE_loss"),
            maxent_scaling=spec.sweep_maxent if spec.method == "maxent" else 1.,
        ),
        opt_config=OptimizationConfig(
            n_steps=spec.n_steps, learning_rate=spec.learning_rate, ema_alpha=spec.ema_alpha,
            convergence_rates=list(shrinkage.CONVERGENCE_RATES), optimizer="adam",
            step_chunk_size=100, lr_adjustment=True, frame_average_impl="tensordot",
            reset_threshold_cooldown_on_oscillation=True,
            forward_model_scaling=spec.forward_model_scaling,
        ),
        name=spec.run_id, output_dir=str(spec.run_dir), cov_matrix=precision,
        execution_mode=spec.execution_mode, frame_regularizer=regularizer(spec, features, model),
    )
    config = json.loads(spec.config_path.read_text())
    # Record the uptake pass rather than the generic recorder's first (PF) pass.
    config["effective_settings"]["frame_averaging_mode"] = spec.averaging_label
    config["sidecar_settings"] = dataclasses.asdict(spec) | {
        "bandwidth": spec.bandwidth if spec.method == "omc" else None,
        "kl_coefficient": spec.bandwidth if spec.method == "maxent" else 0.,
        "intrinsic_rate_provider": "jaxent_calculate_HDXrate", "intrinsic_rate_unit": "min^-1",
        "frame_averaging_mode": spec.averaging_label, "sigma_shrinkage": 0.1,
    }
    spec.config_path.write_text(json.dumps(config, indent=2))


def build_specs(args, sigma_paths):
    specs = []
    for method, strength in [("maxent", 0.)] + [("omc", s) for s in getattr(args, "omc_strengths", OMC_STRENGTHS)]:
        for comparison in COMPARISONS:
            for value in args.maxent_values:
                for split in range(args.n_splits):
                    specs.append(RunSpec(
                        ensemble="ISO_TRI", sigma_source=comparison, alpha=0.1,
                        split_type="sequence_cluster", split_idx=split,
                        sigma_path=str(sigma_paths.get(("ISO_TRI", comparison, 0.1), "")),
                        output_dir=str(args.output_dir), features_dir=str(args.features_dir),
                        datasplit_dir=str(args.datasplit_dir), clustering_dir=str(args.clustering_dir),
                        n_steps=args.n_steps, learning_rate=1., ema_alpha=0.5,
                        forward_model_scaling=1000., execution_mode="compiled",
                        method=method, sweep_maxent=value, omc_strength=strength,
                        uptake_mode=getattr(args, "uptake_mode", "uptake"),
                        graph_k=getattr(args, "graph_k", 0), graph_path=getattr(args, "graph_path", ""),
                    ))
    return specs


def execute(specs, output_dir, jobs):
    for directory in ("specs", "logs"):
        (output_dir / directory).mkdir(exist_ok=True)
    pending = []
    for spec in specs:
        path = output_dir / "specs" / f"{spec.run_id}.json"
        path.write_text(json.dumps(dataclasses.asdict(spec), indent=2))
        if not source.run_is_complete(spec):
            pending.append(path)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(source._worker_command, Path(__file__), path,
                                   output_dir / "logs" / f"{path.stem}.log") for path in pending]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, code = future.result()
            print(f"[{i}/{len(pending)}] {name}: return code {code}", flush=True)
            if code:
                failures.append(name)
    if failures:
        raise RuntimeError(f"Failed fits (see logs): {failures}")


def score_context(spec, cache):
    key = (spec.ensemble, spec.split_type, spec.split_idx, spec.uptake_mode)
    if key not in cache:
        context = list(source.score_context(spec, {}))
        context[2] = configure_spec_model(spec, context[1])
        cache[key] = tuple(context)
    return cache[key]


def score(spec, state, policy, context):
    # Reuse the source experiment's native predictions, mapping and recovery scoring.
    scoring_spec = dataclasses.replace(spec, sigma_source="gt_coordinate")
    row, weights = source.score_state(scoring_spec, state, policy, context)
    row.update(run_id=spec.run_id, sigma_source=spec.sigma_source,
               sigma_source_label=LABELS[spec.sigma_source], panel=spec.panel,
               method=spec.method, uptake_mode=spec.uptake_mode, graph_k=spec.graph_k,
               maxent=spec.sweep_maxent if spec.method == "maxent" else np.nan,
               bandwidth=spec.bandwidth if spec.method == "omc" else np.nan,
               sweep_x=spec.bandwidth, omc_strength=spec.omc_strength,
               kl_coefficient=spec.bandwidth if spec.method == "maxent" else 0.)
    return row, weights


def plot_metric(frame, metric, output):
    panels = list(dict.fromkeys(frame.panel))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.6),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    for axis, panel in zip(axes, panels):
        for comparison in COMPARISONS:
            rows = frame[(frame.panel == panel) & (frame.sigma_source == comparison)]
            for _, trace in rows.groupby("split_idx"):
                trace = trace.sort_values("sweep_x")
                axis.plot(trace.sweep_x, trace[metric], color=COLORS[comparison], alpha=.25, lw=.8)
            mean = rows.groupby("sweep_x")[metric].mean().sort_index()
            axis.plot(mean.index, mean, "o-", color=COLORS[comparison], label=LABELS[comparison])
        axis.set(title=panel, xscale="log", xlabel="1 / MaxEnt" if panel == "MaxEnt" else "OMC bandwidth (work / RT)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Recovery (%)" if metric == "recovery_percent" else "ESS (%)")
    axes[0].set_ylim(0, 102)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    mode_label = ("Linear BV" if "uptake_mode" in frame and frame.uptake_mode.iloc[0] == "linear"
                  else "frame-wise uptake")
    graph_label = (f" · L2 log-PF k={int(frame.graph_k.iloc[0])}"
                   if "graph_k" in frame and frame.graph_k.iloc[0] else " · all pairs")
    fig.suptitle(f"ISO TRI · {mode_label}{graph_label} · identity Sigma shrinkage 0.1", y=.995)
    fig.subplots_adjust(left=.05, right=.99, bottom=.18, top=.78, wspace=.08)
    output.parent.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze(specs, output_dir):
    cache, convergence, final, incomplete, weights = {}, [], [], [], {}
    missing_selection = []
    for spec in specs:
        if not source.run_is_complete(spec):
            incomplete.append({"run_id": spec.run_id, "reason": "missing_or_invalid_history"})
            continue
        history = source.load_optimization_history_from_file(str(spec.history_path))
        context = score_context(spec, cache)
        labeled_states = source.iter_labeled_convergence_states(history)
        if not labeled_states:
            missing_selection.append({"run_id": spec.run_id, "reason": "no_convergence_states"})
        for labeled in labeled_states:
            row, w = score(spec, labeled.state, "convergence_state", context)
            row.update(convergence_rank=labeled.rank, convergence_threshold=labeled.threshold)
            convergence.append(row)
            weights[f"{spec.run_id}__conv{labeled.rank}"] = w
        row, w = score(spec, history.states[-1], "final_optimization_state", context)
        final.append(row)
        weights[f"{spec.run_id}__final"] = w
    pd.DataFrame(incomplete, columns=["run_id", "reason"]).to_csv(output_dir / "incomplete_runs.csv", index=False)
    pd.DataFrame(missing_selection, columns=["run_id", "reason"]).to_csv(
        output_dir / "missing_selection.csv", index=False)
    if not final:
        raise RuntimeError("No complete optimization histories; inspect fit logs")
    conv = pd.DataFrame(convergence)
    conv.to_csv(output_dir / "convergence_scores.csv", index=False)
    selected = (source.select_best_rows(conv) if not conv.empty
                else pd.DataFrame(columns=list(final[0]) + ["convergence_rank", "convergence_threshold"]))
    for policy, frame in (("selected", selected), ("final_step", pd.DataFrame(final))):
        frame.to_csv(output_dir / f"{policy}_results.csv", index=False)
        if frame.empty:
            print(f"No {policy} states available; see missing_selection.csv", flush=True)
            continue
        summary = frame.groupby(["panel", "sigma_source", "sweep_x"])[
            ["recovery_percent", "ess_percent", "val_mse"]].agg(["mean", "std", "count"])
        summary.to_csv(output_dir / f"{policy}_summary.csv")
        for metric in ("recovery_percent", "ess_percent"):
            plot_metric(frame, metric, output_dir / "plots" / f"{metric}_vs_strength_{policy}")
    np.savez_compressed(output_dir / "frame_weights.npz", **weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--uptake-mode", choices=("uptake", "linear"), default="uptake")
    parser.add_argument("--graph-k", type=int, default=0,
                        help="0: all pairs; positive k: L2 log-PF neighbours, Work Scale edge weights")
    parser.add_argument("--features-dir", type=Path, default=shrinkage.DEFAULT_FEATURES_DIR)
    parser.add_argument("--datasplit-dir", type=Path, default=shrinkage.DEFAULT_DATASPLIT_DIR)
    parser.add_argument("--clustering-dir", type=Path, default=shrinkage.DEFAULT_CLUSTERING_DIR)
    parser.add_argument("--trajectory-dir", type=Path, default=source.DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--maxent-values", default=",".join(map(str, DEFAULT_MAXENT)))
    parser.add_argument("--omc-strengths", default=",".join(map(str, OMC_STRENGTHS)))
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--phase", choices=("prepare", "fit", "analyze", "all"), default="all")
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        run_fit(RunSpec(**json.loads(args.worker_spec.read_text())))
        return
    args.maxent_values = list(shrinkage.parse_csv(args.maxent_values, float))
    args.omc_strengths = list(shrinkage.parse_csv(args.omc_strengths, float))
    if args.output_dir is None:
        suffix = "linear_bv" if args.uptake_mode == "linear" else "framewise"
        if args.graph_k:
            suffix += f"_knn{args.graph_k}"
        args.output_dir = HERE / f"_maxent_omc_sidecar_{suffix}"
    if (not args.maxent_values or any(not np.isfinite(x) or x <= 0 for x in args.maxent_values)
            or len(set(args.maxent_values)) != len(args.maxent_values)
            or not args.omc_strengths
            or any(not np.isfinite(x) or x <= 0 for x in args.omc_strengths)
            or len(set(args.omc_strengths)) != len(args.omc_strengths)
            or min(args.n_splits, args.n_steps, args.jobs) < 1 or args.graph_k < 0):
        parser.error("Sweep values must be unique, positive and finite; counts must be positive")
    for name in ("output_dir", "features_dir", "datasplit_dir", "clustering_dir", "trajectory_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"phase", "jobs", "worker_spec"}}
    settings["version"] = VERSION
    # Reject changed code/settings before touching a resumable campaign.
    settings["code_sha256"] = hashlib.sha256(b"".join(path.read_bytes() for path in (
        Path(__file__), Path(source.__file__), Path(shrinkage.__file__), HERE / "sidecar_selection.py",
        HERE.parents[2] / "common" / "optimization.py",
    ))).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["settings"] != settings:
        raise ValueError("Campaign settings/code changed; use a new --output-dir")
    paths, _ = source.prepare_sigma_artifacts(
        args.output_dir, args.features_dir, args.clustering_dir, args.trajectory_dir,
        ("ISO_TRI",), COMPARISONS[1:], (0.1,), 1e8,
    )
    args.graph_path = prepare_neighbours(args)
    specs = build_specs(args, paths)
    manifest = dict(settings=settings, expected_fits=len(specs), sigma_shrinkage=.1,
                    selection_metric=source.SELECTION_POLICY,
                    frame_averaging_mode=("linear_uptake" if args.uptake_mode == "linear" else "frame_uptake"),
                    intrinsic_rate_provider="jaxent_calculate_HDXrate",
                    omc_metric="absolute_difference_mean_log_pf",
                    graph=(f"logpf_l2_knn{args.graph_k}_union" if args.graph_k else "all_pairs_gaussian"),
                    omc_strengths=args.omc_strengths, x_axis="1/maxent = OMC bandwidth")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    pd.DataFrame([dataclasses.asdict(s) | {"run_id": s.run_id, "sweep_x": s.bandwidth,
                                        "panel": s.panel} for s in specs]).to_csv(
        args.output_dir / "run_grid.csv", index=False)
    print(f"Grid: {len(specs)} fits, {args.jobs} workers; output: {args.output_dir}", flush=True)
    if args.phase in ("fit", "all"):
        execute(specs, args.output_dir, args.jobs)
    if args.phase in ("analyze", "all"):
        analyze(specs, args.output_dir)
    manifest["completed_fits"] = sum(source.run_is_complete(s) for s in specs)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
