#!/usr/bin/env python3
"""Compare four held-out selectors within each saved MaxEnt trajectory only.

No fitting or pooled-trajectory selection. Candidates are all saved trajectory,
convergence, and running-best states (deduplicated by step and frame weights).
GT recovery is evaluation-only, never a validation selector.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
from pathlib import Path

import run_maxent_weighted_mse_sidecar as sweep
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

source, shrinkage, plt = sweep.source, sweep.shrinkage, sweep.plt
SELECTORS = ("mse", "gt_coordinate", "closed_coordinate", "gt_uptake_weighted")
LABELS = ("MSE", "GT coordinate", "Closed coordinate", "Peptide × time")
HERE = Path(__file__).resolve().parent


def collect_specs(campaign, seen=None):
    seen = set() if seen is None else seen
    campaign = campaign.resolve()
    if campaign in seen:
        raise ValueError("Cyclic campaign extension")
    seen.add(campaign)
    manifest = json.loads((campaign / "manifest.json").read_text())
    specs = []
    previous = manifest["settings"].get("extend_results")
    if previous:
        specs.extend(collect_specs(Path(previous), seen))
    specs.extend(sweep.RunSpec(**row) for row in manifest["specs"])
    if len({s.run_id for s in specs}) != len(specs):
        raise ValueError("Repeated trajectory in campaign chain")
    return specs


def score_errors(residual, gt_precision, closed_precision, uptake_precision):
    """Residuals: peptide x time; fixed validation-only precisions."""
    if residual.shape != uptake_precision.shape:
        raise ValueError("Observation/timepoint precision shape mismatch")
    return dict(mse=float(np.mean(residual**2)),
        gt_coordinate=float(.5*np.einsum("pt,pq,qt->", residual, gt_precision, residual)/residual.size),
        closed_coordinate=float(.5*np.einsum("pt,pq,qt->", residual, closed_precision, residual)/residual.size),
        gt_uptake_weighted=float(.5*np.mean(uptake_precision*residual**2)))


def candidates(history):
    seen = set()
    entries = [("trajectory", s) for s in history.states]
    entries += [("convergence", s) for s in history.convergence_states]
    if history.best_state is not None:
        entries.append(("running_best", history.best_state))
    for kind, state in entries:
        weights = np.asarray(source.validated_frame_weight_simplex(state.params.frame_weight_simplex), dtype=float)
        weights /= weights.sum()
        key = (int(state.step), hashlib.sha256(weights.tobytes()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        yield kind, state, weights


def select_within(frame):
    selections = []
    for run_id, rows in frame.groupby("run_id", sort=False):
        if not np.isfinite(rows[list(SELECTORS)+["recovery_percent", "ess_percent"]]).all().all():
            raise ValueError(f"Nonfinite candidate scores in {run_id}")
        oracle = rows.sort_values(["recovery_percent", "step", "candidate_index"],
                                  ascending=[False, True, True], kind="stable").iloc[0]
        baseline = rows.sort_values(["mse", "step", "candidate_index"], kind="stable").iloc[0]
        for selector in SELECTORS:
            best = rows.sort_values([selector, "step", "candidate_index"], kind="stable").iloc[0]
            selections.append(best.to_dict() | dict(selector=selector, candidates=len(rows),
                oracle_recovery_percent=oracle.recovery_percent, oracle_step=int(oracle.step),
                recovery_regret_pp=float(oracle.recovery_percent-best.recovery_percent),
                recovery_gain_vs_mse_pp=float(best.recovery_percent-baseline.recovery_percent),
                same_candidate_as_mse=bool(best.candidate_index == baseline.candidate_index)))
    return pd.DataFrame(selections)


def worker(payload):
    specs = [sweep.RunSpec(**row) for row in payload["specs"]]
    first = specs[0]
    features, topology = shrinkage.load_features(first.features_dir, first.ensemble)
    assignments = shrinkage.load_cluster_assignments(first.clustering_dir, first.ensemble)
    model = shrinkage.configure_model(first.uptake_mode, assignments)
    train, val = shrinkage.load_split(first.datasplit_dir, first.split_type, first.split_idx)
    loader = source.create_data_loaders(train+val, train, val, features, topology)
    mapping = np.asarray(loader.val.residue_feature_ouput_mapping.todense())
    target = np.asarray(loader.val.y_true)[..., 0]
    indices = np.asarray([d.top.fragment_index for d in val], dtype=int)
    coordinate = []
    for name in SELECTORS[1:3]:
        with np.load(payload["sigma_paths"][name]) as archive:
            full = archive["Sigma_inv_normalized"]
        precision = full[np.ix_(indices, indices)]
        coordinate.append(precision * len(indices)/np.trace(precision))
    with np.load(payload["uptake_path"]) as archive:
        uptake_precision = archive["val_precision"]
        np.testing.assert_allclose(archive["val_mapping"], mapping)
        if str(archive["precision_normalization"]) != "per_peptide":
            raise ValueError("Selection comparison requires current per-peptide weights")
    forward = model.forward[shrinkage.m_key("HDX_peptide")]
    predict = jax.jit(lambda w: forward.average_frames(features, model.params, w).uptake)
    rows = []
    for spec in specs:
        config = json.loads(spec.config_path.read_text())
        if config["loss_config"]["optimize_bv_params"]:
            raise ValueError("This analysis expects the fixed-BV frame-weight-only campaigns")
        history = source.load_optimization_history_from_file(str(spec.history_path))
        if not history.states:
            raise ValueError(f"Empty history: {spec.run_id}")
        for index, (kind, state, weights) in enumerate(candidates(history)):
            predicted = np.asarray(predict(jnp.asarray(weights)), dtype=float)
            residual = mapping @ predicted.T - target
            scores = score_errors(residual, *coordinate, uptake_precision)
            # Cross-check the fit's own saved validation loss; no KL/scaling here.
            native = float(state.losses.val_losses[0]) if state.losses is not None else np.nan
            expected = scores[spec.sigma_source] * (.5 if spec.sigma_source == "mse" else 1.)
            if np.isfinite(native) and not np.isclose(native, expected, rtol=3e-3, atol=2e-7):
                raise ValueError(f"Native validation mismatch {spec.run_id} step {int(state.step)}: {native} vs {expected}")
            ess = source.effective_sample_size(weights)
            rows.append(dict(run_id=spec.run_id, ensemble=spec.ensemble, uptake_mode=spec.uptake_mode,
                training_loss=spec.sigma_source, maxent=spec.sweep_maxent, split_idx=spec.split_idx,
                candidate_index=index, candidate_kind=kind, step=int(state.step),
                is_final_step=int(state.step) == int(history.states[-1].step),
                native_validation_loss=native, **scores,
                recovery_percent=source.calculate_recovery_percentage(assignments, weights,
                    shrinkage.GROUND_TRUTH, shrinkage.STATE_MAPPING),
                ess_percent=100*ess/len(weights)))
        print(f"Scored {spec.run_id}", flush=True)
    pd.DataFrame(rows).to_csv(payload["result_path"], index=False)


def plot_heatmaps(selected, output_dir):
    for metric, title, cmap in (
        ("recovery_regret_pp", "Recovery gap to best saved state (pp; lower is better)", "viridis_r"),
        ("recovery_gain_vs_mse_pp", "Recovery change vs ordinary-MSE selection (pp)", "coolwarm"),
        ("recovery_percent", "Selected recovery (%)", "viridis"),
        ("ess_percent", "Selected ESS (%)", "viridis")):
        tables = []
        for ensemble in sweep.ENSEMBLES:
            for mode in sweep.MODES:
                rows = selected[(selected.ensemble == ensemble) & (selected.uptake_mode == mode)]
                tables.append(rows.pivot_table(index="training_loss", columns="selector", values=metric,
                                               aggfunc="mean").reindex(index=SELECTORS, columns=SELECTORS))
        values = np.concatenate([table.to_numpy().ravel() for table in tables])
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if metric == "recovery_gain_vs_mse_pp":
            high = max(abs(low), abs(high), 1e-6); low = -high
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for axis, table, (ensemble, mode) in zip(axes.flat, tables,
                [(e, m) for e in sweep.ENSEMBLES for m in sweep.MODES]):
            im = axis.imshow(table, cmap=cmap, vmin=low, vmax=high)
            axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
            axis.set_yticks(range(4), LABELS)
            axis.set(title=f"{ensemble} · {'Linear BV' if mode == 'linear' else 'Uptake'}",
                     xlabel="Validation selector", ylabel="Training loss")
            for i in range(4):
                for j in range(4):
                    axis.text(j, i, f"{table.iloc[i,j]:.2f}", ha="center", va="center",
                              bbox=dict(facecolor="white", alpha=.75, edgecolor="none", pad=1))
        fig.colorbar(im, ax=axes, shrink=.8)
        fig.suptitle(title + "\nWithin-trajectory selection; means across MaxEnt values and splits")
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / f"{metric}.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=HERE / "_maxent_weighted_mse_per_peptide_tri1e7_jobs8")
    parser.add_argument("--output-dir", type=Path, default=HERE / "_within_trajectory_selection")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--worker-spec", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        worker(json.loads(args.worker_spec.read_text())); return
    if args.jobs < 1:
        parser.error("jobs must be positive")
    args.output_dir = args.output_dir.resolve()
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = collect_specs(args.campaign)
    manifest = dict(campaign=str(args.campaign), trajectories=len(specs),
        candidate_policy="all saved trajectory/convergence/running-best states; deduplicate step and simplex",
        selector_policy="validation metric only; ties: earliest step then candidate index",
        pooling=False, precision_normalization="per_peptide", variance_floor=1e-4,
        code_sha256=hashlib.sha256(b"".join(Path(p).read_bytes() for p in (
            __file__, sweep.__file__, source.__file__, shrinkage.__file__))).hexdigest(),
        input_histories=[dict(run_id=s.run_id, path=str(s.history_path),
                             bytes=s.history_path.stat().st_size,
                             mtime_ns=s.history_path.stat().st_mtime_ns) for s in specs])
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Analysis settings changed; use a new directory")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    groups = {}
    for spec in specs:
        groups.setdefault((spec.ensemble, spec.uptake_mode, spec.split_idx), []).append(spec)
    for name in ("groups", "logs", "plots"):
        (args.output_dir / name).mkdir(exist_ok=True)
    jobs = []
    paths = []
    for (ensemble, mode, split), group in groups.items():
        token = f"{ensemble}_{mode}_split{split:03d}"
        result = args.output_dir / "groups" / f"{token}.csv"
        paths.append(result)
        weighted = next(s for s in group if s.sigma_source == "gt_uptake_weighted")
        sigma_paths = {name: next(s.sigma_path for s in group if s.sigma_source == name)
                       for name in SELECTORS[1:3]}
        payload = dict(specs=[dataclasses.asdict(s) for s in group], sigma_paths=sigma_paths,
                       uptake_path=weighted.weights_path, result_path=str(result))
        path = args.output_dir / "groups" / f"{token}.json"
        path.write_text(json.dumps(payload, indent=2))
        if not result.exists():
            jobs.append((path, args.output_dir / "logs" / f"{token}.log"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(source._worker_command, Path(__file__), path, log) for path, log in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name, code = future.result()
            print(f"[{i}/{len(jobs)}] {name}: {code}", flush=True)
            if code:
                raise RuntimeError(f"Analysis group failed: {name}; inspect logs")
    scored = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if scored.run_id.nunique() != len(specs):
        raise ValueError("Missing trajectories in score export")
    scored.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    selected = select_within(scored)
    selected.to_csv(args.output_dir / "selected_by_criterion.csv", index=False)
    selected.groupby(["ensemble", "uptake_mode", "training_loss", "selector"])[
        ["recovery_percent", "ess_percent", "recovery_regret_pp", "recovery_gain_vs_mse_pp"]
        ].agg(["mean", "std", "count"]).to_csv(args.output_dir / "summary.csv")
    selected.groupby(["ensemble", "uptake_mode", "training_loss", "maxent", "selector"])[
        ["recovery_percent", "ess_percent", "recovery_regret_pp", "recovery_gain_vs_mse_pp"]
        ].agg(["mean", "std", "count"]).to_csv(args.output_dir / "summary_by_maxent.csv")
    plot_heatmaps(selected, args.output_dir / "plots")
    print(f"Complete: {len(specs)} trajectories, {len(scored)} candidates, {len(selected)} selections", flush=True)


if __name__ == "__main__":
    main()
