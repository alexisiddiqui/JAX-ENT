"""Reselect the documented CrossVal averaging benchmarks using MoPrP SD weights.

Run from the repository root with:
  .venv/bin/python -m jaxent.examples.common.analysis.rescore_crossval_weighted

This reads existing fits and writes a separate report; it never refits models.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import jax.numpy as jnp

from jaxent.examples.common import loading
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.examples.common.config import ExperimentConfig
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.custom_types.key import m_key
from jaxent.src.utils.hdf import load_model_parameters_from_hdf5


CAMPAIGNS = {
    "2_CrossValidation": {
        "rate": "_optimise_quick_test_SIGMA_5000__20260924_223118",
        "linear": "_optimise_quick_test_SIGMA_5000_linear_bv_fixed__20260926_022111",
        "full_uptake": "_optimise_quick_test_SIGMA_5000_full_uptake__20260926_135652",
    },
    "3_CrossValidationBV": {
        "rate": "_optimise_quick_test_SIGMA_5000_lr1.0_BV_objectve_scale1.0__20260924_223526",
        "linear": "_optimise_quick_test_SIGMA_5000_linear_bv_bc_bh__20260926_022424",
        "full_uptake": "_optimise_quick_test_SIGMA_5000_full_uptake_BV__20260926_140412",
    },
}
PATTERN = re.compile(
    r"(?P<ensemble>AF2_MSAss|AF2_filtered)_MSE_(?P<split_type>.+?)_split"
    r"(?P<split_idx>\d+)_maxent(?P<maxent_value>[\d.]+)"
    r"(?:_bvreg(?P<bv_reg_value>[\d.]+)_bvregfn(?P<bv_reg_function>[A-Za-z0-9]+))?"
)
METRICS = {"unweighted": 0, "1/SD": 1, "1/SD²": 2}
GROUP = ["experiment", "model", "ensemble", "split_type", "split_idx"]


def weighted_mse(pred, observed, weights, power):
    pred, observed, weights = (
        np.asarray(a, dtype=float) for a in (pred, observed, weights)
    )
    if pred.shape != observed.shape or weights.shape != observed.shape:
        raise ValueError("Prediction, uptake and weight shapes must match exactly")
    if not all(np.isfinite(a).all() for a in (pred, observed, weights)) or np.any(
        weights <= 0
    ):
        raise ValueError(
            "Require finite predictions/uptake and finite positive weights"
        )
    effective = weights**power
    return float(np.sum(effective * (pred - observed) ** 2) / np.sum(effective))


def verify_metric():
    pred = np.array([[1.0, 3.0], [2.0, 4.0]])
    observed = np.zeros_like(pred)
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    for power in (1, 2):
        np.testing.assert_allclose(
            weighted_mse(pred, observed, np.ones_like(pred), power), np.mean(pred**2)
        )
        np.testing.assert_allclose(
            weighted_mse(pred, observed, weights, power),
            weighted_mse(pred, observed, 10 * weights, power),
        )
    np.testing.assert_allclose(weighted_mse(pred, observed, weights, 1), 9.5)


def aligned_weights(data, inverse_sd, source_uptake):
    ids = [int(d.top.fragment_index) for d in data]
    observed = np.array([d.dfrac for d in data])
    if len(ids) != len(set(ids)) or any(i < 0 or i >= len(inverse_sd) for i in ids):
        raise ValueError("Invalid or duplicate peptide IDs")
    np.testing.assert_allclose(observed, source_uptake[ids], rtol=0, atol=1e-7)
    return observed, inverse_sd[ids]


def audit_provenance(raw, model):
    expected = "frame_uptake" if model == "full_uptake" else "rate"
    fit_logs = sorted(
        f for f in (raw / "logs").glob("*.log") if f.name.startswith("AF2_")
    )
    if not fit_logs:
        raise ValueError(f"Missing fit logs: {raw}")
    for log in fit_logs:
        if f"Frame averaging mode: {expected}" not in log.read_text():
            raise ValueError(f"Unconfirmed averaging mode: {log}")
    processing = (raw / "logs/process_optimisation_results.log").read_text()
    if f"Frame averaging mode: {expected}" not in processing:
        raise ValueError(f"Unconfirmed processing mode: {raw}")
    sidecar_modes = sorted(
        {
            json.loads(f.read_text())["effective_settings"]["frame_averaging_mode"]
            for f in raw.glob("*/*_config.json")
        }
    )
    raw_files = sorted(raw.glob("*/*_results.hdf5"))
    counts = []
    for file in raw_files:
        with h5py.File(file, "r") as h:
            counts.append(len(h["optimization_history/convergence_states"]))
    sidecar_expected = "linear_uptake" if model == "linear" else expected
    return {
        "source": str(raw.resolve()),
        "fit_log_count": len(fit_logs),
        "confirmed_mode": expected,
        "model_type": "linear" if model == "linear" else "standard",
        "sidecar_modes": sidecar_modes,
        "sidecar_mode_conflict": sidecar_modes != [sidecar_expected],
        "raw_result_count": len(raw_files),
        "raw_checkpoint_count": sum(counts),
        "runs_without_convergence_checkpoints": sum(count == 0 for count in counts),
    }


def campaign_rows(root, experiment, model, basename, inverse_sd, source_uptake):
    experiment_dir = root / "jaxent/examples" / experiment
    fitting = experiment_dir / "fitting/jaxENT"
    raw, processed = fitting / basename, fitting / ("_processed_" + basename)
    provenance = audit_provenance(raw, model)
    manifest = json.loads((processed / "manifest.json").read_text())
    if Path(manifest["source_results_dir"]).resolve() != raw.resolve():
        raise ValueError("Manifest source does not match campaign")
    saved = pd.read_csv(next(processed.glob("_scores*/model_scores.csv")))
    config = ExperimentConfig.from_yaml(str(experiment_dir / "config.yaml"))
    populations = json.loads(
        (experiment_dir / "analysis/state_ratios.json").read_text()
    )["fractional_populations"]
    targets = {
        s: populations.get(k, {}).get("fraction", 0.0)
        for s, k in [
            ("Folded", "folded"),
            ("PUF1", "PUF1"),
            ("PUF2", "PUF2"),
            ("PUF3", "PUF3"),
            ("unfolded", "unfolded"),
        ]
    }
    # The documented baseline keeps zero-target decoy mass in the JSD support.
    for state in config.scoring.state_mapping.values():
        targets.setdefault(state, 0.0)
    cache, rows, max_error = {}, [], 0.0
    runs = sorted(
        f for f in processed.glob("*/*") if f.is_dir() and PATTERN.fullmatch(f.name)
    )
    if len(runs) != manifest["n_runs_processed"]:
        raise ValueError("Run count does not match processing manifest")
    for run in runs:
        metadata = PATTERN.fullmatch(run.name).groupdict()
        metadata["split_idx"] = int(metadata["split_idx"])
        metadata["maxent_value"] = float(metadata["maxent_value"])
        metadata["bv_reg_value"] = (
            float(metadata["bv_reg_value"]) if metadata["bv_reg_value"] else np.nan
        )
        key = metadata["ensemble"], metadata["split_type"], metadata["split_idx"]
        if key not in cache:
            with contextlib.redirect_stdout(io.StringIO()):
                train, val, full, _ = loading.load_experimental_data(
                    str(processed), str(fitting / "_datasplits"), key[1], key[2]
                )
                features, topology = loading.load_features_and_topology(
                    str(fitting / "_featurise"),
                    key[0],
                    config.scoring.ensemble_feature_map,
                )
                loader = ExpD_Dataloader(data=full)
                loader.create_datasets(
                    features=features,
                    feature_topology=topology,
                    train_data=train,
                    val_data=val,
                    test_data=full,
                )
            yval, wval = aligned_weights(val, inverse_sd, source_uptake)
            yfull, wfull = aligned_weights(full, inverse_sd, source_uptake)
            cache[key] = (loader, yval, wval, yfull, wfull)
        loader, yval, wval, yfull, wfull = cache[key]
        predictions = np.load(run / "pred_uptake.npy", mmap_mode="r")
        thresholds = np.atleast_1d(np.loadtxt(run / "convergence_thresholds.txt"))
        clusters = pd.read_csv(run / "cluster_ratios.csv")
        frame_weights = np.load(run / "frame_weights.npy", mmap_mode="r")
        if (
            len(predictions) != len(thresholds)
            or len(clusters) != len(thresholds)
            or len(frame_weights) != len(thresholds)
        ):
            raise ValueError(f"Checkpoint stack mismatch: {run}")
        np.testing.assert_allclose(
            clusters["convergence"], thresholds, rtol=1e-12, atol=0
        )
        matched = saved[
            (saved.ensemble == key[0])
            & (saved.split_type == key[1])
            & (saved.split_idx == key[2])
            & (saved.maxent_value == metadata["maxent_value"])
        ]
        if "bv_reg_value" in saved:
            matched = matched[
                (matched.bv_reg_value == metadata["bv_reg_value"])
                & (matched.bv_reg_function == metadata["bv_reg_function"])
            ]
        if len(matched) != len(thresholds):
            raise ValueError(f"Saved scores missing candidates: {run}")
        for index, (pred, threshold) in enumerate(zip(predictions, thresholds)):
            mapped = []
            for dataset in (loader.val, loader.test):
                mapping = dataset.residue_feature_ouput_mapping
                mapped.append(
                    np.asarray(
                        [apply_sparse_mapping(mapping, jnp.asarray(p)) for p in pred]
                    ).T
                )
            val_scores = {
                name: weighted_mse(mapped[0], yval, wval, power)
                for name, power in METRICS.items()
            }
            full_scores = {
                name: weighted_mse(mapped[1], yfull, wfull, power)
                for name, power in METRICS.items()
            }
            original = matched.iloc[
                np.argmin(np.abs(matched.convergence_value.to_numpy() - threshold))
            ]
            np.testing.assert_allclose(
                original.convergence_value, threshold, rtol=1e-12, atol=0
            )
            error = max(
                abs(original.val_mse - val_scores["unweighted"]),
                abs(original.test_mse - full_scores["unweighted"]),
            )
            max_error = max(max_error, error)
            np.testing.assert_allclose(
                [val_scores["unweighted"], full_scores["unweighted"]],
                [original.val_mse, original.test_mse],
                rtol=1e-6,
                atol=1e-8,
            )
            cluster_ids = [
                int(c.removeprefix("cluster_"))
                for c in clusters
                if c.startswith("cluster_")
            ]
            mass = clusters.iloc[index][[f"cluster_{c}" for c in cluster_ids]].to_numpy(
                dtype=float
            )
            recovery = calculate_recovery_percentage(
                np.array(cluster_ids), mass, targets, config.scoring.state_mapping
            )
            weights = np.asarray(frame_weights[index], dtype=float)
            weights = weights / weights.sum()
            rows.append(
                {
                    **metadata,
                    "experiment": experiment,
                    "model": model,
                    "run_id": run.name,
                    "checkpoint_index": index,
                    "convergence_value": threshold,
                    "recovery_percent": recovery,
                    "ess": 1 / np.sum(weights**2),
                    **{f"val_{name}": score for name, score in val_scores.items()},
                    **{f"full_{name}": score for name, score in full_scores.items()},
                }
            )
    provenance.update(
        run_count=len(runs), candidate_count=len(rows), max_saved_mse_error=max_error
    )
    if len(rows) != provenance["raw_checkpoint_count"]:
        raise ValueError(
            "Processed candidates do not cover every saved raw convergence checkpoint"
        )
    return rows, provenance


def verify_baseline_and_predictions(root, selected, summary):
    """Check the published table, extracted selections and forward semantics."""
    baseline = summary[summary.weighting == "unweighted"].set_index(GROUP[:-1])
    section = (
        (root / "docs/averaging_mode_soft_regression.md")
        .read_text()
        .split("## Full-uptake reference refit", 1)[1]
        .split("For the same validation-selected", 1)[0]
    )
    document_checks = []
    for line in section.splitlines():
        if not line.startswith("| CrossValidation"):
            continue
        fields = [f.strip() for f in re.split(r"(?<!\\)\|", line)[1:-1]]
        experiment = (
            "2_CrossValidation"
            if fields[0] == "CrossValidation"
            else "3_CrossValidationBV"
        )
        ensemble, split = fields[1].rsplit(" ", 1)
        split = "sequence_cluster" if split == "sequence" else split
        for model, cell in zip(("rate", "linear", "full_uptake"), fields[2:]):
            recorded = np.array([float(n) for n in re.findall(r"\d*\.\d+", cell)])
            actual = baseline.loc[(experiment, model, ensemble, split)]
            difference = (
                np.array(
                    [
                        actual.recovery_mean,
                        actual.recovery_sd,
                        actual.val_mse,
                        actual.full_dataset_mse,
                    ]
                )
                - recorded
            )
            np.testing.assert_allclose(difference[:2], 0.0, rtol=0, atol=0.0051)
            np.testing.assert_allclose(difference[2:], 0.0, rtol=0, atol=0.000011)
            document_checks.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "ensemble": ensemble,
                    "split_type": split,
                    "recovery_mean_delta_pp": difference[0],
                    "recovery_sd_delta_pp": difference[1],
                    "val_mse_delta": difference[2],
                    "full_mse_delta": difference[3],
                }
            )
    if len(document_checks) != 24:
        raise ValueError("Could not check all 24 documented baseline cells")
    forward_checks, extracted_checks = [], []
    for experiment, models in CAMPAIGNS.items():
        expdir = root / "jaxent/examples" / experiment
        config = ExperimentConfig.from_yaml(str(expdir / "config.yaml"))
        fitting = expdir / "fitting/jaxENT"
        times = loading.load_hdx_timepoints_minutes(
            root / "jaxent/examples/2_CrossValidation/data/_MoPrP/moprp.times"
        )
        for model, basename in models.items():
            raw, processed = fitting / basename, fitting / ("_processed_" + basename)
            chosen = selected[
                (selected.experiment == experiment)
                & (selected.model == model)
                & (selected.weighting == "unweighted")
            ]
            for (ensemble, split), group in chosen.groupby(["ensemble", "split_type"]):
                suffix = "_L1" if experiment == "3_CrossValidationBV" else ""
                archive = next(
                    processed.glob(
                        f"_extracted*/val_mse_min/{ensemble}_MSE_{split}{suffix}_selected.npz"
                    )
                )
                extracted = np.load(archive)["frame_weights"]
                for row in group.itertuples():
                    actual = np.load(
                        processed / split / row.run_id / "frame_weights.npy"
                    )[row.checkpoint_index]
                    np.testing.assert_array_equal(actual, extracted[row.split_idx])
                    extracted_checks.append(
                        {
                            "experiment": experiment,
                            "model": model,
                            "run_id": row.run_id,
                            "split_idx": row.split_idx,
                            "max_frame_weight_delta": float(
                                np.max(np.abs(actual - extracted[row.split_idx]))
                            ),
                        }
                    )
            for ensemble, group in chosen.groupby("ensemble"):
                row = group.iloc[0]
                run = processed / row.split_type / row.run_id
                # Processed BV names format 1.0 as 1.00, unlike the raw files.
                raw_name = re.sub(
                    r"_bvreg([\d.]+)",
                    lambda m: f"_bvreg{float(m.group(1))}",
                    row.run_id,
                )
                with h5py.File(
                    raw / row.split_type / (raw_name + "_results.hdf5"), "r"
                ) as h:
                    params = load_model_parameters_from_hdf5(
                        h,
                        f"optimization_history/convergence_states/{row.checkpoint_index}/params/model_parameters/0",
                    )
                with contextlib.redirect_stdout(io.StringIO()):
                    features, _ = loading.load_features_and_topology(
                        str(fitting / "_featurise"),
                        ensemble,
                        config.scoring.ensemble_feature_map,
                    )
                uptake_model = build_uptake_model(
                    "linear" if model == "linear" else "standard", times
                )
                forward = uptake_model.forward[m_key("HDX_peptide")]
                if model != "linear":
                    forward.frame_averaging_mode = (
                        "frame_uptake" if model == "full_uptake" else "rate"
                    )
                weights = jnp.asarray(
                    np.load(run / "frame_weights.npy")[row.checkpoint_index]
                )
                predicted = np.asarray(
                    forward.average_frames(features, params, weights).uptake
                )
                stored = np.load(run / "pred_uptake.npy")[row.checkpoint_index]
                np.testing.assert_allclose(predicted, stored, rtol=1e-5, atol=1e-6)
                forward_checks.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "ensemble": ensemble,
                        "parameter_class": type(params).__name__,
                        "max_prediction_delta": float(
                            np.max(np.abs(predicted - stored))
                        ),
                    }
                )
    return {
        "document_baseline": document_checks,
        "extracted_selections": extracted_checks,
        "forward_predictions": forward_checks,
    }


def write_report(output, summary, verification):
    notes = (
        "Saved fits were reselected independently per replicate using normalized peptide × timepoint weighted validation MSE. "
        "1/SD uses moprp.weights directly; 1/SD² squares those values. No refitting was performed. "
        "Recovery is mean ± sample SD over three replicates, using the original benchmark's JSD support. "
        "Zero-target decoy mass, including PUF2-like, remains in that support. "
        "Full-dataset MSE includes training and validation peptides and is report-only. "
        "Fit and processing logs confirm averaging modes; sidecar mode conflicts are recorded in audit.json."
    )
    blocks = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>CrossVal weighted selection</title>",
        "<style>body{font:15px system-ui;margin:32px;line-height:1.5}table{border-collapse:collapse;margin:20px 0}td,th{padding:8px;border:1px solid #ddd;text-align:right}th{background:#f1f3f5}h2{margin-top:32px}p{max-width:1100px}</style>",
        "<h1>CrossVal weighted validation selection</h1><p>Source campaigns: September 24–26, 2026.</p><p>"
        + notes
        + "</p>",
        "<p>Weighted MSE = Σ(w<sup>q</sup> × residual²) / Σw<sup>q</sup>, where w = 1/SD and q = 1 or 2. The sum covers validation peptide × timepoint observations.</p>",
        "<p><a href='summary.csv'>Summary CSV</a> · <a href='selected.csv'>Selected checkpoints and hyperparameters</a> · <a href='candidates.csv'>All candidates</a> · <a href='audit.json'>Source audit</a> · <a href='verification.json'>Verification</a></p>",
    ]
    for metric in METRICS:
        for experiment in CAMPAIGNS:
            table = summary[
                (summary.weighting == metric) & (summary.experiment == experiment)
            ].copy()
            table["Recovery % ± SD"] = table.apply(
                lambda r: f"{r.recovery_mean:.2f} ± {r.recovery_sd:.2f}", axis=1
            )
            table = table.rename(
                columns={
                    "weighted_val_mse": "Selection val MSE",
                    "val_mse": "Ordinary val MSE",
                    "full_dataset_mse": "Full-dataset MSE",
                    "changed_selections": "Changed / 3",
                }
            )
            blocks += [
                f"<h2>{experiment.removeprefix('2_').removeprefix('3_')} — {metric}</h2>",
                table[
                    [
                        "ensemble",
                        "split_type",
                        "model",
                        "Recovery % ± SD",
                        "Selection val MSE",
                        "Ordinary val MSE",
                        "Full-dataset MSE",
                        "Changed / 3",
                    ]
                ].to_html(index=False, float_format=lambda v: f"{v:.5f}"),
            ]
    max_mse_delta = max(
        abs(r["val_mse_delta"]) for r in verification["document_baseline"]
    )
    blocks += [
        f"<p>Maximum ordinary validation MSE discrepancy versus the rounded documented table: {max_mse_delta:.3g}. Exact differences and archive checks are recorded in verification.json.</p>",
        "</html>",
    ]
    (output / "report.html").write_text("\n".join(blocks) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/crossval_weighted_selection")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    source = root / "jaxent/examples/2_CrossValidation/data/_MoPrP"
    raw_weights, uptake = (
        np.loadtxt(source / "moprp.weights"),
        np.loadtxt(source / "moprp.dexp"),
    )
    np.testing.assert_array_equal(raw_weights[:, 0], uptake[:, 0])
    np.testing.assert_allclose(
        raw_weights[:, 0] * 60,
        loading.load_hdx_timepoints_minutes(source / "moprp.times"),
        rtol=0,
        atol=1e-12,
    )
    verify_metric()
    rows, audit = [], []
    for experiment, models in CAMPAIGNS.items():
        for model, basename in models.items():
            print(f"Rescoring {experiment} {model}...", flush=True)
            candidates, provenance = campaign_rows(
                root, experiment, model, basename, raw_weights[:, 1:].T, uptake[:, 1:].T
            )
            rows.extend(candidates)
            audit.append(provenance)
    candidates = pd.DataFrame(rows)
    selections = []
    for metric in METRICS:
        chosen = (
            candidates.sort_values(f"val_{metric}", kind="stable")
            .drop_duplicates(GROUP)
            .copy()
        )
        chosen["weighting"] = metric
        chosen["weighted_val_mse"] = chosen[f"val_{metric}"]
        chosen["weighted_full_mse"] = chosen[f"full_{metric}"]
        selections.append(chosen)
    selected = pd.concat(selections, ignore_index=True)
    baseline = selections[0].set_index(GROUP)
    selected["changed_from_unweighted"] = [
        (row.run_id, row.checkpoint_index)
        != (
            baseline.loc[tuple(getattr(row, c) for c in GROUP)].run_id,
            baseline.loc[tuple(getattr(row, c) for c in GROUP)].checkpoint_index,
        )
        for row in selected.itertuples()
    ]
    summary = (
        selected.groupby(GROUP[:-1] + ["weighting"], sort=False)
        .agg(
            replicates=("split_idx", "size"),
            recovery_mean=("recovery_percent", "mean"),
            recovery_sd=("recovery_percent", "std"),
            weighted_val_mse=("weighted_val_mse", "mean"),
            val_mse=("val_unweighted", "mean"),
            full_dataset_mse=("full_unweighted", "mean"),
            weighted_full_mse=("weighted_full_mse", "mean"),
            ess_mean=("ess", "mean"),
            changed_selections=("changed_from_unweighted", "sum"),
        )
        .reset_index()
    )
    if not (summary.replicates == 3).all():
        raise ValueError("Require three replicates for every comparison cell")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / "candidates.csv", index=False)
    selected.to_csv(output / "selected.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    verification = verify_baseline_and_predictions(root, selected, summary)
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    write_report(output, summary, verification)
    print(summary.to_string(index=False))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
