"""Plot ensemble-coordinate Sigma-MSE on existing weighted-selected BV models.

No optimization or reselection: evaluate exactly the sweep points already used
for recovery, ESS and validation-MSE figures.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT/"fitting/jaxENT"))
import run_moprp_bv_policy_sidecars as bv
sys.path.insert(0, str(bv.EXAMPLE2/"analysis"))
from plot_moprp_policy_sidecars import export_figures
from jaxent.examples.common.analysis.rescore_crossval_coordinate_sigma import (
    coordinate_covariance, make_loader, sigma_mse, split_precisions,
)

METRIC = "val_coordinate_sigma_mse"
TRAJECTORIES = {"AF2_MSAss": "_cluster_MoPrP", "AF2_filtered": "_cluster_MoPrP_filtered"}


def prepare_covariances(specs, directory):
    helper = importlib.import_module(
        "jaxent.examples.1_IsoValidation_OMass.fitting.jaxENT.compute_sigma_synthetic")
    config = bv.ExperimentConfig.from_yaml(str(EXPERIMENT/"config.yaml"))
    result, metadata = {}, {}
    for ensemble, trajectory_dir in TRAJECTORIES.items():
        spec = next(s for s in specs.values() if s.ensemble == ensemble)
        loader, features, topology = make_loader(Path(spec.features_dir).parent, config,
            ensemble, spec.split_type, spec.split_idx)
        trajectory = bv.EXAMPLE2/f"data/_archived_MoPrP101/{trajectory_dir}/clusters/all_clusters.xtc"
        residue, source = coordinate_covariance(topology, features.features_shape[1], trajectory=trajectory)
        ids = [int(d.top.fragment_index) for d in loader.test.data]
        if ids != list(range(len(ids))):
            raise ValueError("Full peptide mapping must follow the original fragment IDs")
        mapping = np.asarray(loader.test.residue_feature_ouput_mapping.todense(), dtype=float)
        raw = mapping @ residue @ mapping.T
        raw = (raw+raw.T)/2
        ridge = helper.ridge_thresholds(raw, condition_limit=1e8)["ridge_stable"]
        covariance = raw+ridge*np.eye(len(raw))
        source.update(ensemble=ensemble, peptide_ids=ids, shrinkage_alpha=0., ridge=ridge,
            condition_before=float(np.linalg.cond(raw)), condition_after=float(np.linalg.cond(covariance)),
            covariance_split_order="map to full peptide support, invert full covariance, subset validation precision",
            precision_normalization="validation precision trace equals validation peptide count",
            covariance_definition="M @ coordinate_residue_covariance @ M.T",
            features_sha256=bv.digest(Path(spec.features_dir)/f"features_{ensemble}.npz"))
        np.savez_compressed(directory/f"coordinate_sigma_{ensemble}.npz", residue_covariance=residue,
            mapping=mapping, peptide_covariance_raw=raw, peptide_covariance=covariance)
        result[ensemble] = covariance, mapping
        metadata[ensemble] = source
        print(f"{ensemble}: uniform {source['frames']}-frame covariance, {len(raw)} peptides, ridge={ridge:g}", flush=True)
    bv.write_json(directory/"coordinate_sigma_sources.json", metadata)
    return result, metadata


def evaluate_points(points, specs, covariances):
    cache, matrices, records = {}, {}, []
    for number, row in enumerate(points.itertuples(index=False), 1):
        spec = specs[row.run_id]
        _, _, native, _, val, loader, _, _, _, predict = bv.context(spec, cache)
        key = spec.ensemble, spec.split_type, spec.split_idx
        mapping = np.asarray(loader.val.residue_feature_ouput_mapping.todense())
        if key not in matrices:
            covariance, full_mapping = covariances[spec.ensemble]
            indices = [int(d.top.fragment_index) for d in val]
            np.testing.assert_array_equal(mapping, full_mapping[indices])
            matrices[key] = split_precisions(covariance, indices)["inverse_then_split"]
        history = bv.load_optimization_history_from_file(str(spec.saved_history))
        kind, state, weights = next(entry for index, entry in enumerate(bv.saved_candidates(history))
            if index == row.candidate_index)
        if kind != row.candidate_kind or int(state.step) != row.step:
            raise ValueError("Existing selected checkpoint identity changed")
        parameters = state.params.model_parameters[0]
        bv.validate_parameters(parameters, native.params)
        np.testing.assert_allclose([np.asarray(parameters.bv_bc).item(), np.asarray(parameters.bv_bh).item()],
            [row.bv_bc, row.bv_bh], rtol=0, atol=1e-12)
        predicted = mapping @ np.asarray(predict(jnp.asarray(weights), parameters)).T
        observed = np.asarray(loader.val.y_true)[..., 0]
        residual = predicted-observed
        ordinary = float(np.mean(residual**2))
        weighted = bv.raw_weighted_mse(predicted, observed, bv.aligned_file_weights(val))
        np.testing.assert_allclose([ordinary, weighted], [row.val_mse, row.val_weighted_mse], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(.5*ordinary, float(state.losses.val_losses[0]), rtol=3e-3, atol=2e-7)
        score = sigma_mse(residual, matrices[key])
        if not np.isfinite(score) or score < 0:
            raise ValueError("Invalid coordinate Sigma-MSE")
        records.append(row._asdict() | {METRIC: score, "coordinate_sigma_ensemble": spec.ensemble,
            "weighted_score_replay_absolute_error": abs(weighted-row.val_weighted_mse)})
        if number % 100 == 0:
            print(f"Evaluated {number}/{len(points)} existing sweep points", flush=True)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=EXPERIMENT/"analysis/_moprp_policy_sidecars")
    parser.add_argument("--campaign-dir", type=Path, default=EXPERIMENT/"fitting/jaxENT/_moprp_policy_sidecars")
    args = parser.parse_args()
    directory = args.analysis_dir.resolve()
    points_path = directory/"selected_profiled.csv"
    source_hash = bv.digest(points_path)
    manifest = json.loads((args.campaign_dir/"manifest.json").read_text())
    specs = {s.run_id: s for s in (bv.RunSpec(**row) for row in manifest["specs"])}
    points = pd.read_csv(points_path)
    output = directory/"selected_profiled_coordinate_sigma.csv"
    source_metadata = directory/"coordinate_sigma_sources.json"
    previous_sources = json.loads(source_metadata.read_text()) if source_metadata.exists() else None
    previous = pd.read_csv(output) if output.exists() else None
    with contextlib.redirect_stdout(io.StringIO()):
        covariances, sources = prepare_covariances(specs, directory)
    print("Coordinate sources: "+", ".join(f"{name}: ridge={source['ridge']:g}" for name, source in sources.items()), flush=True)
    reusable = []
    pending = []
    cached = {} if previous is None or previous_sources != sources else {
        (row.run_id, row.candidate_index): row for row in previous.itertuples(index=False)}
    for index, row in enumerate(points.itertuples(index=False)):
        old = cached.get((row.run_id, row.candidate_index))
        values = ("bv_bc", "bv_bh", "val_mse", "val_weighted_mse")
        if (old is not None and old.step == row.step and old.candidate_kind == row.candidate_kind
                and old.history_path == row.history_path and all(np.isclose(getattr(old, key),
                    getattr(row, key), rtol=1e-12, atol=1e-12) for key in values)):
            reusable.append(row._asdict() | {METRIC: getattr(old, METRIC),
                "coordinate_sigma_ensemble": row.ensemble,
                "weighted_score_replay_absolute_error": old.weighted_score_replay_absolute_error})
        else:
            pending.append(index)
    print(f"Reuse {len(reusable)} unchanged coordinate Sigma-MSE evaluations; evaluate {len(pending)} new points", flush=True)
    frames = [pd.DataFrame(reusable)] if reusable else []
    if pending:
        frames.append(evaluate_points(points.iloc[pending], specs, covariances))
    result = pd.concat(frames, ignore_index=True)
    order = {(row.run_id, row.candidate_index): index for index, row in enumerate(points.itertuples(index=False))}
    result = result.iloc[sorted(range(len(result)), key=lambda index: order[
        (result.iloc[index].run_id, result.iloc[index].candidate_index)])].reset_index(drop=True)
    if bv.digest(points_path) != source_hash:
        raise ValueError("Selected sweep points changed during evaluation")
    result.to_csv(output, index=False)
    keys = ["ensemble", "method", "edge_type", "graph_k", "split_type", "strength", "bandwidth"]
    result.groupby(keys)[[METRIC, "recovery_percent", "ess_percent"]].agg(["mean", "std", "count"]).to_csv(
        directory/"summary_profiled_coordinate_sigma.csv")
    smoke = json.loads((directory/"audit.json").read_text())["smoke"]
    export_figures(result, directory/"figures", smoke, bv_fitted=True, metrics=[METRIC])
    audit = dict(points=len(result), no_refitting=True, no_reselection=True,
        reused_coordinate_evaluations=len(reusable), new_coordinate_evaluations=len(pending),
        source_selected_points_sha256=source_hash, source_selected_points=str(points_path),
        checkpoint_specific_bv_parameters_verified=True,
        weighted_score_max_replay_absolute_error=float(result.weighted_score_replay_absolute_error.max()),
        metric="0.5 * sum_t(residual_t.T @ validation_precision @ residual_t) / (validation_peptides * timepoints)",
        source_selection="unchanged raw MoPrP.weights validation selection", coordinate_sources=sources)
    bv.write_json(directory/"audit_profiled_coordinate_sigma.json", audit)
    print(f"Saved three Sigma-MSE versus S plots for {len(result)} existing sweep points; no fitting or reselection.", flush=True)


if __name__ == "__main__":
    main()
