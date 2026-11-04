"""Reselect saved MoPrP fits using raw weighted MSE on the full peptide dataset.

Writes sibling analysis folders; never runs optimization or changes source data.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

EXAMPLES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXAMPLES/"2_CrossValidation/fitting/jaxENT"))
sys.path.insert(0, str(EXAMPLES/"3_CrossValidationBV/fitting/jaxENT"))
sys.path.insert(0, str(EXAMPLES/"3_CrossValidationBV/analysis"))
import run_moprp_policy_sidecars as fixed
import run_moprp_bv_policy_sidecars as bv
from plot_moprp_policy_sidecars import ALL_PEPTIDE_METRICS, METRICS, VALIDATION_METRICS, export_figures
from plot_moprp_policy_model_selection_bars import plot_bars, select_across_parameters, summarize
from plot_moprp_bv_policy_sidecars import profile_bv_penalties

SCORE = "all_peptide_weighted_mse"
PREFIX = "all_pairs_knn20_parameter_selected"
IDENTITY = ["run_id", "candidate_index"]


def full_dataset(spec, runner, contexts, datasets):
    context = runner.context(spec, contexts)
    key = spec.ensemble, spec.split_type, spec.split_idx, spec.frame_averaging_mode
    if key not in datasets:
        features, topology, _, train, val, loader, *_ = context
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, full, _ = fixed.loading.load_experimental_data("", spec.datasplit_dir, spec.split_type, spec.split_idx)
            all_loader = fixed.create_data_loaders(full, train, val, features, topology)
        ids = [int(d.top.fragment_index) for d in full]
        train_ids = [int(d.top.fragment_index) for d in train]
        val_ids = [int(d.top.fragment_index) for d in val]
        if len(ids) != len(set(ids)) or not set(train_ids+val_ids).issubset(ids):
            raise ValueError("Require unique full-dataset peptide IDs and valid split references")
        mapping = np.asarray(all_loader.test.residue_feature_ouput_mapping.todense())
        observed = np.asarray(all_loader.test.y_true)[..., 0]
        weights = fixed.aligned_file_weights(full)
        val_indices = [ids.index(i) for i in val_ids]
        np.testing.assert_array_equal(mapping[val_indices], np.asarray(loader.val.residue_feature_ouput_mapping.todense()))
        np.testing.assert_array_equal(observed[val_indices], np.asarray(loader.val.y_true)[..., 0])
        np.testing.assert_array_equal(weights[val_indices], fixed.aligned_file_weights(val))
        datasets[key] = mapping, observed, weights, val_indices, ids
    return context, datasets[key]


def score_run(spec, expected, runner, contexts, datasets):
    context, (mapping, observed, surface, val_indices, _) = full_dataset(spec, runner, contexts, datasets)
    native, predict = context[2], context[-1]
    history = fixed.load_optimization_history_from_file(str(spec.saved_history))
    records = []
    expected = expected.sort_values("candidate_index")
    candidates = list(runner.saved_candidates(history))
    if len(candidates) != len(expected):
        raise ValueError(f"Saved candidate pool changed: {spec.run_id}")
    for index, ((kind, state, weights), row) in enumerate(zip(candidates, expected.itertuples(index=False), strict=True)):
        if row.candidate_index != index or kind != row.candidate_kind or int(state.step) != row.step:
            raise ValueError(f"Checkpoint identity changed: {spec.run_id}/{index}")
        parameters = state.params.model_parameters[0]
        if runner is bv:
            bv.validate_parameters(parameters, native.params)
            np.testing.assert_allclose([np.asarray(parameters.bv_bc).item(), np.asarray(parameters.bv_bh).item()],
                [row.bv_bc, row.bv_bh], rtol=0, atol=1e-12)
        else:
            fixed.validate_protocol(parameters, native.params)
            np.testing.assert_allclose([np.asarray(parameters.bv_bc).item(), np.asarray(parameters.bv_bh).item()],
                [np.asarray(native.params.bv_bc).item(), np.asarray(native.params.bv_bh).item()], rtol=1e-7)
        predicted = mapping @ np.asarray(predict(jnp.asarray(weights), parameters)).T
        residual = predicted-observed
        val_mse = float(np.mean(residual[val_indices]**2))
        val_weighted = fixed.raw_weighted_mse(predicted[val_indices], observed[val_indices], surface[val_indices])
        np.testing.assert_allclose([val_mse, val_weighted], [row.val_mse, row.val_weighted_mse], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(.5*val_mse, float(state.losses.val_losses[0]), rtol=3e-3, atol=2e-7)
        np.testing.assert_allclose(100/(len(weights)*np.sum(weights**2)), row.ess_percent, rtol=1e-6)
        records.append(dict(run_id=spec.run_id, candidate_index=index,
            all_peptide_mse=float(np.mean(residual**2)),
            all_peptide_weighted_mse=fixed.raw_weighted_mse(predicted, observed, surface),
            validation_replay_absolute_error=max(abs(val_mse-row.val_mse), abs(val_weighted-row.val_weighted_mse))))
    return records


def select_trajectories(candidates):
    return candidates.sort_values([SCORE, "step", "candidate_index"], kind="stable").groupby("run_id", sort=False).head(1).copy()


def comparison(original, selected, keys):
    fields = keys+["run_id", "candidate_index", "step", "strength", "bandwidth", "recovery_percent", "ess_percent"]
    if "bv_reg_scaling" in selected:
        fields += ["bv_reg_scaling", "bv_bc", "bv_bh"]
    merged = original[fields].merge(selected[fields], on=keys, suffixes=("_validation", "_all_peptide"), validate="one_to_one")
    for metric in ("recovery_percent", "ess_percent"):
        merged[f"delta_{metric}"] = merged[f"{metric}_all_peptide"]-merged[f"{metric}_validation"]
    return merged


def atomic_scores(path, records):
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def score_worker(path):
    task = json.loads(path.read_text())
    source = Path(task["source"])
    runner = bv if task["bv_fitted"] else fixed
    specs = [runner.RunSpec(**s) for s in task["specs"]]
    groups = dict(tuple(pd.read_csv(source/"candidates.csv").groupby("run_id", sort=False)))
    audits = {a["run_id"]: a for a in json.loads((source/"audit.json").read_text())["fit_audits"]}
    records, contexts, datasets = [], {}, {}
    for number, spec in enumerate(specs, 1):
        if fixed.digest(spec.saved_history) != audits[spec.run_id]["history_sha256"]:
            raise ValueError(f"Source optimization history changed: {spec.run_id}")
        records.extend(score_run(spec, groups[spec.run_id], runner, contexts, datasets))
        if number % 10 == 0 or number == len(specs):
            atomic_scores(Path(task["scores"]), records)
            print(f"Scored {number}/{len(specs)} histories", flush=True)


def worker_shards(specs, jobs):
    groups = {}
    for spec in specs:
        groups.setdefault((spec.ensemble, spec.split_type, spec.split_idx), []).append(spec)
    buckets = [[] for _ in range(min(jobs, len(groups)))]
    for group in sorted(groups.values(), key=len, reverse=True):
        min(buckets, key=len).extend(group)
    return buckets


def rescore(source, output=None, coordinate_sigma=True, jobs=10):
    source = source.resolve()
    output = (output or source.with_name(source.name+"_all_peptide_selection")).resolve()
    if output == source:
        raise ValueError("All-peptide selection must use a sibling output directory")
    experiment = source.parent.parent
    campaign = experiment/"fitting/jaxENT"/source.name
    manifest_path = campaign/"manifest.json"
    source_audit = json.loads((source/"audit.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    runner = bv if "bv_reg_values" in manifest["settings"] else fixed
    specs = [runner.RunSpec(**s) for s in manifest["specs"]]
    if source_audit["completed_fits"] != len(specs) or any(s.reuse_history or s.reuse_config for s in specs):
        raise ValueError("Require a complete, campaign-local source fit")
    fixed.validate_fitting_campaign(specs, manifest)
    for path, fingerprint in manifest["input_sha256"].items():
        if fixed.digest(path) != fingerprint:
            raise ValueError(f"Frozen fitting input changed: {path}")
    hashes = {str(p): fixed.digest(p) for p in (source/"candidates.csv", source/"audit.json", manifest_path)}
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output/"rescore_progress.json"
    if progress_path.exists() and json.loads(progress_path.read_text())["source_sha256"] != hashes:
        raise ValueError("Source results changed since rescoring began")
    candidates = pd.read_csv(source/"candidates.csv")
    score_path = output/"all_peptide_scores.csv"
    cached = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    worker_directory = output/"rescore_workers"
    for path in sorted(worker_directory.glob("scores_*.csv")):
        partial = pd.read_csv(path)
        if len(cached):
            partial = partial[~partial.run_id.isin(cached.run_id)]
        cached = pd.concat([cached, partial], ignore_index=True)
    records = cached.to_dict("records")
    done = set(cached.run_id) if len(cached) else set()
    contexts, datasets = {}, {}
    _, (_, full_observed, _, _, full_ids) = full_dataset(specs[0], runner, contexts, datasets)
    source_groups = dict(tuple(candidates.groupby("run_id", sort=False)))
    fit_audits = {a["run_id"]: a for a in source_audit["fit_audits"]}
    fixed.write_json(progress_path, dict(status="running", source_sha256=hashes, completed_histories=len(done),
        expected_histories=len(specs), no_refitting=True, selection_metric=SCORE))
    print(f"Rescore {experiment.name}/{source.name}: {len(specs)} histories, {len(candidates)} candidates; resume {len(done)} histories", flush=True)
    for run_id in done:
        if set(cached.loc[cached.run_id == run_id, "candidate_index"]) != set(source_groups[run_id].candidate_index):
            raise ValueError("Incomplete cached candidate pool")
    if records:
        atomic_scores(score_path, records)
    if jobs > 1 and len(done) < len(specs):
        from dataclasses import asdict
        worker_directory.mkdir(exist_ok=True)
        cores = sorted(os.sched_getaffinity(0))
        buckets = worker_shards([s for s in specs if s.run_id not in done], min(jobs, len(cores)//2))
        def launch(slot_and_specs):
            slot, subset = slot_and_specs
            task_path = worker_directory/f"task_{slot:02d}.json"
            scores_path = worker_directory/f"scores_{slot:02d}.csv"
            fixed.write_json(task_path, dict(source=str(source), bv_fitted=runner is bv,
                specs=[asdict(s) for s in subset], scores=str(scores_path)))
            with (worker_directory/f"worker_{slot:02d}.log").open("w") as log:
                subprocess.run(["taskset", "-c", ",".join(map(str, cores[2*slot:2*slot+2])),
                    sys.executable, str(Path(__file__).resolve()), "--score-worker", str(task_path)],
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            return pd.read_csv(scores_path)
        print(f"Read-only native scoring: {len(buckets)} workers, two CPUs per worker", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(buckets)) as pool:
            futures = [pool.submit(launch, item) for item in enumerate(buckets)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                records.extend(result.to_dict("records"))
                done.update(result.run_id)
                atomic_scores(score_path, records)
                fixed.write_json(progress_path, dict(status="rescoring", source_sha256=hashes, completed_histories=len(done),
                    expected_histories=len(specs), no_refitting=True, selection_metric=SCORE))
                print(f"Scored {len(done)}/{len(specs)} histories", flush=True)
        for path in worker_directory.glob("scores_*.csv"):
            path.unlink()
        cached = pd.DataFrame(records)
    for number, spec in enumerate(specs, 1):
        expected = source_groups[spec.run_id]
        if spec.run_id in done:
            if set(cached.loc[cached.run_id == spec.run_id, "candidate_index"]) != set(expected.candidate_index):
                raise ValueError("Incomplete cached candidate pool")
            continue
        if fixed.digest(spec.saved_history) != fit_audits[spec.run_id]["history_sha256"]:
            raise ValueError(f"Source optimization history changed: {spec.run_id}")
        records.extend(score_run(spec, expected, runner, contexts, datasets))
        done.add(spec.run_id)
        if number % 20 == 0 or number == len(specs):
            atomic_scores(score_path, records)
            fixed.write_json(progress_path, dict(status="rescoring", source_sha256=hashes, completed_histories=len(done),
                expected_histories=len(specs), no_refitting=True, selection_metric=SCORE))
            print(f"Scored {number}/{len(specs)} histories", flush=True)
    scores = pd.DataFrame(records)
    candidates = candidates.merge(scores, on=IDENTITY, validate="one_to_one")
    if len(candidates) != source_audit["candidates"] or not np.isfinite(candidates[SCORE]).all():
        raise ValueError("Missing or invalid all-peptide scores")
    candidates["selection_metric"] = SCORE
    candidates.to_csv(output/"candidates.csv", index=False)
    selected = select_trajectories(candidates)
    selected.to_csv(output/"selected.csv", index=False)
    fitted = runner is bv
    profiled = profile_bv_penalties(selected, SCORE) if fitted else selected
    if fitted:
        profiled.to_csv(output/"selected_profiled.csv", index=False)
    keys = ["ensemble", "method", "edge_type", "graph_k", "split_type", "strength", "bandwidth"]
    if not profiled.groupby(keys).size().eq(3).all():
        raise ValueError("Selected split-replicate counts changed")
    metrics = list(METRICS)+list(VALIDATION_METRICS)+list(ALL_PEPTIDE_METRICS)
    profiled.groupby(keys)[metrics].agg(["mean", "std", "count"]).to_csv(output/"summary.csv")
    export_figures(profiled, output/"figures", bv_fitted=fitted, metrics=metrics)
    final = select_across_parameters(candidates, SCORE)
    summary = summarize(final)
    final.to_csv(output/f"{PREFIX}.csv", index=False)
    summary.to_csv(output/f"{PREFIX}_summary.csv", index=False)
    if not summary.n.eq(6).all():
        raise ValueError("Final bars must contain all six split replicates")
    for metric in METRICS:
        plot_bars(final, summary, metric, output/"figures"/f"{PREFIX}_{metric}",
            manifest["settings"]["bandwidths"], bv_fitted=fitted)
    old = pd.read_csv(source/f"{PREFIX}.csv")
    delta = comparison(old, final, ["panel_graph_k", "model", "split_type", "split_idx"])
    delta.to_csv(output/"selection_comparison.csv", index=False)
    delta.groupby(["panel_graph_k", "model"])[["recovery_percent_validation", "recovery_percent_all_peptide",
        "delta_recovery_percent", "ess_percent_validation", "ess_percent_all_peptide", "delta_ess_percent"]].mean().to_csv(output/"selection_comparison_summary.csv")
    maxent = candidates[candidates.method == "maxent"].sort_values([SCORE, "strength"]+
        (["bv_reg_scaling"] if fitted else [])+["step", "candidate_index"], kind="stable").groupby(
            ["ensemble", "split_type", "split_idx"], sort=False).head(1)
    old_candidates = pd.read_csv(source/"candidates.csv")
    old_maxent = old_candidates[old_candidates.method == "maxent"].sort_values(["val_weighted_mse", "strength"]+
        (["bv_reg_scaling"] if fitted else [])+["step", "candidate_index"], kind="stable").groupby(
            ["ensemble", "split_type", "split_idx"], sort=False).head(1)
    comparison(old_maxent, maxent, ["ensemble", "split_type", "split_idx"]).to_csv(output/"maxent_selection_comparison.csv", index=False)
    audit = dict(source_audit, selection="mean(moprp.weights * residual²) over all peptides and timepoints",
        selection_metric=SCORE, selection_scope="full_dataset", no_refitting=True, source_analysis=str(source),
        source_campaign=str(campaign), source_sha256=hashes, unchanged_candidate_pool=True,
        checkpoint_specific_bv_parameters_verified=fitted,
        all_peptides=len(full_ids), all_timepoints=full_observed.shape[1],
        validation_replay_max_absolute_error=float(candidates.validation_replay_absolute_error.max()))
    fixed.write_json(output/"audit.json", audit)
    fixed.write_json(output/f"{PREFIX}_policy.json", dict(selection_metric=SCORE, selection_scope="full_dataset",
        weighting="mean(moprp.weights * residual squared); direct weights, no normalization", no_refitting=True,
        source_analysis=str(source), forward_model=source_audit["forward_model"], bv_fitted=fitted,
        bandwidths=manifest["settings"]["bandwidths"], aggregation="mean and sample SD over both split types"))
    if fitted and coordinate_sigma:
        subprocess.run([sys.executable, str(experiment/"analysis/plot_moprp_bv_coordinate_sigma.py"),
            "--analysis-dir", str(output), "--campaign-dir", str(campaign)], check=True)
    for path, fingerprint in hashes.items():
        if fixed.digest(path) != fingerprint:
            raise ValueError("Source results changed during analysis")
    layouts = ["maxent", "omc_all_pairs", "omc_knn20"]
    figures = "".join(f'<h2>{layout} · {metric}</h2><img src="figures/{layout}_{metric}.png">'
        for metric in METRICS for layout in layouts+[PREFIX])
    (output/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>MoPrP all-peptide selection</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}img{max-width:100%}</style>'
        f'<h1>MoPrP {source_audit["forward_model"]}: all-peptide weighted MSE selection</h1>'
        '<p>Same fitted histories and candidate pool; every saved state uses its own fitted BV parameters. '
        'Selection uses all 12 peptides × 15 timepoints and raw MoPrP.weights. '
        'Both split types pooled after selection; recovery and ESS use the existing definitions.</p>'
        f'<p><a href="../{source.name}/report.html">Validation-selected results</a> · '
        '<a href="selection_comparison_summary.csv">Recovery/ESS change</a></p>'+figures)
    fixed.write_json(progress_path, dict(status="complete", source_sha256=hashes, completed_histories=len(specs),
        expected_histories=len(specs), no_refitting=True, selection_metric=SCORE))
    print(summary.to_string(index=False), flush=True)
    print(f"All-peptide results: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-coordinate-sigma", action="store_true")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--score-worker", type=Path)
    args = parser.parse_args()
    if args.score_worker:
        score_worker(args.score_worker)
        return
    if not args.source_analysis or args.jobs < 1:
        parser.error("Specify --source-analysis and a positive --jobs count")
    if args.output_dir and len(args.source_analysis) != 1:
        parser.error("--output-dir requires one source campaign")
    for source in args.source_analysis:
        rescore(source, args.output_dir, not args.skip_coordinate_sigma, args.jobs)


if __name__ == "__main__":
    main()
