"""Fit matched 1/SD and 1/SD² CrossVal grids and select by the fitted metric.

Run: .venv/bin/python -m jaxent.examples.common.analysis.fit_crossval_weighted
Use --smoke for 20-step checks, or rerun the same output path to resume.
Hyperparameters are batched within a split; independent worker processes use
two CPU cores each. Original fits and selection-only results are preserved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from dataclasses import replace
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.examples.common import loading
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.examples.common.analysis.rescore_crossval_weighted import (
    CAMPAIGNS,
    GROUP,
    METRICS,
    weighted_mse,
)
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.losses import (
    get_loss_function_by_name,
    maxent_convexKL_loss,
)
from jaxent.examples.common.moprp_weighted_loss import (
    peptide_time_weights,
    source_surface,
)
from jaxent.examples.common.optimization import create_data_loaders
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.analysis.frame_weights import validated_frame_weight_simplex
from jaxent.src.custom_types.config import Optimisable_Parameters, OptimiserSettings
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.opt.base import HParamBatch
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.utils.hdf import save_optimization_history_to_file
from jaxent.src.utils.jit_fn import jit_Guard


ROOT = Path(__file__).resolve().parents[4]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )
    tmp.replace(path)


def job_dir(output, experiment, model, power, ensemble, split):
    return output / f"1sd_power{power}" / experiment / model / f"{ensemble}_{split}"


def worker(
    output, experiment, model, power, ensemble, split, smoke, normalize_weights=True
):
    started = time.monotonic()
    out = job_dir(output, experiment, model, power, ensemble, split)
    out.mkdir(parents=True, exist_ok=True)
    expdir = ROOT / "jaxent/examples" / experiment
    fitting = expdir / "fitting/jaxENT"
    source = fitting / CAMPAIGNS[experiment][model]
    templates = [
        json.loads(p.read_text())
        for p in sorted(source.glob(f"{split}/{ensemble}_MSE_*config.json"))
    ]
    if not templates:
        raise ValueError(f"No source configurations: {source}")
    template = templates[0]
    source_opt, source_loss = template["opt_config"], template["loss_config"]
    combos = sorted(
        {
            (
                float(c["loss_config"]["maxent_scaling"]),
                float(c["loss_config"]["bv_reg_scaling"])
                if experiment == "3_CrossValidationBV"
                else None,
            )
            for c in templates
        }
    )
    if smoke:
        combos = combos[:1]
    steps = 20 if smoke else source_opt["n_steps"]
    times = loading.load_hdx_timepoints_minutes(
        ROOT / "jaxent/examples/2_CrossValidation/data/_MoPrP/moprp.times"
    )
    config = ExperimentConfig.from_yaml(str(expdir / "config.yaml"))
    with contextlib.redirect_stdout(io.StringIO()):
        features, topology = loading.load_features_and_topology(
            str(fitting / "_featurise"), ensemble, config.scoring.ensemble_feature_map
        )
        clustering = loading.load_clustering_results(
            str(
                expdir
                / "analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters"
            ),
            config.scoring.ensemble_clustering_map,
        )
    assignments = np.asarray(clustering[ensemble]["cluster_assignments"])
    populations = json.loads((expdir / "analysis/state_ratios.json").read_text())[
        "fractional_populations"
    ]
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
    for state in config.scoring.state_mapping.values():
        targets.setdefault(state, 0.0)
    model_obj = build_uptake_model("linear" if model == "linear" else "standard", times)
    mode = (
        "linear_uptake"
        if model == "linear"
        else "frame_uptake"
        if model == "full_uptake"
        else "rate"
    )
    forward = model_obj.forward[m_key("HDX_peptide")]
    if model != "linear":
        forward.frame_averaging_mode = mode
    parameters = model_obj.params
    optimize_bv = source_loss["optimize_bv_params"]
    primary_name = (
        "hdx_uptake_inv_sd_MSE_loss"
        if power == 1
        else "hdx_uptake_inv_variance_MSE_loss"
    )
    if not normalize_weights:
        if power != 1:
            raise ValueError("Raw MoPrP comparison uses the file weights directly")
        primary_name = "hdx_uptake_moprp_raw_weighted_MSE_loss"
    losses = [get_loss_function_by_name(primary_name), maxent_convexKL_loss]
    for reg in source_loss["regularization_losses"]:
        losses.append(get_loss_function_by_name(reg["name"]))
    nslots = len(losses)
    normalise = jnp.ones(nslots)
    if optimize_bv and not source_loss["normalize_bv_reg"]:
        normalise = normalise.at[-1].set(0.0)
    raw_weights = jnp.asarray(
        [[1.0, 1.0 / m] + ([bv] if bv is not None else []) for m, bv in combos]
    )
    masked = raw_weights * normalise
    lane_weights = raw_weights * (1.0 - normalise) + masked / masked.sum(
        axis=1, keepdims=True
    )
    hparams = HParamBatch(
        forward_model_weights=lane_weights,
        forward_model_scaling=jnp.full(
            (len(combos), nslots), source_opt["forward_model_scaling"]
        ),
        learning_rate=jnp.full((len(combos),), source_opt["learning_rate"]),
    )
    partitions = {Optimisable_Parameters.frame_weights}
    if optimize_bv:
        partitions.add(Optimisable_Parameters.model_parameters)
    trainable = (
        frozenset({"raw_bv_bc", "raw_bv_bh"})
        if model == "linear" and optimize_bv
        else None
    )
    settings = OptimiserSettings(
        name=f"weighted_{experiment}_{model}",
        n_steps=steps,
        tolerance=source_opt["tolerance"],
        convergence=source_opt["convergence_rates"],
        learning_rate=source_opt["learning_rate"],
        optimiser_type=source_opt["optimizer"],
        ema_alpha=source_opt["ema_alpha"],
        step_chunk_size=source_opt["step_chunk_size"],
        reset_threshold_cooldown_on_oscillation=source_opt[
            "reset_threshold_cooldown_on_oscillation"
        ],
        execution_mode="compiled",
        parameter_partitions=frozenset(partitions),
    )
    run_metadata = {
        "experiment": experiment,
        "model": model,
        "power": power,
        "normalize_weights": normalize_weights,
        "data_loss_definition": "0.5 * sum(w * residual²) / sum(w)"
        if normalize_weights
        else "0.5 * mean(moprp.weights * residual²)",
        "ensemble": ensemble,
        "split_type": split,
        "source_campaign": str(source),
        "optimizer": source_opt,
        "loss_config": {**source_loss, "primary_loss": primary_name},
        "frame_averaging_mode": mode,
        "trainable_model_parameters": sorted(trainable) if trainable else None,
        "actual_steps": steps,
        "execution": "native hyperparameter batch",
        "grid": combos,
        "weights_sha256": hashlib.sha256(
            (
                ROOT / "jaxent/examples/2_CrossValidation/data/_MoPrP/moprp.weights"
            ).read_bytes()
        ).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    atomic_json(out / "configuration.json", run_metadata)
    inv_sd, observed_source = source_surface()
    rows = []
    for split_idx in range(1 if smoke else 3):
        part = out / f"split_{split_idx:03d}"
        part.mkdir(exist_ok=True)
        if (part / "complete.json").exists():
            rows.extend(pd.read_csv(part / "candidates.csv").to_dict("records"))
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            train, val, full, _ = loading.load_experimental_data(
                "", str(fitting / "_datasplits"), split, split_idx
            )
            loader = create_data_loaders(train + val, train, val, features, topology)
            full_loader = create_data_loaders(full, full, full, features, topology)
        for data in (train, val, full):
            peptide_time_weights(data, power)
        initial = Simulation_Parameters.from_frame_weights(
            jnp.ones(features.features_shape[1]) / features.features_shape[1],
            model_parameters=(parameters,),
            forward_model_weights=lane_weights[0],
            normalise_loss_functions=normalise,
            forward_model_scaling=jnp.full(
                (nslots,), source_opt["forward_model_scaling"]
            ),
        )
        simulation = Simulation(
            input_features=(features,),
            forward_models=(model_obj,),
            params=initial,
            frame_average_impl=source_opt["frame_average_impl"],
        )
        optimizer = OptaxOptimizer(
            learning_rate=source_opt["learning_rate"],
            parameter_partition_masks=partitions,
            clip_value=source_opt["clip_value"],
            optimizer=source_opt["optimizer"],
            lr_adjustment=source_opt["lr_adjustment"],
            model_parameters_lr_scale=source_opt["model_parameters_lr_scale"],
            trainable_model_parameters=trainable,
        )
        data_targets = (loader, initial, initial) if optimize_bv else (loader, initial)
        print(
            f"Fit {experiment} {model} power={power} {ensemble}/{split}/split{split_idx:03d}: {len(combos)} lanes × {steps} steps",
            flush=True,
        )
        with jit_Guard(simulation, cleanup_on_exit=True) as simulation:
            simulation.initialise()
            result = batch_optimise(
                simulation=simulation,
                hparam_batch=hparams,
                batch_size=len(combos),
                data_to_fit=data_targets,
                config=settings,
                indexes=(0,) * nslots,
                loss_functions=tuple(losses),
                optimizer=optimizer,
            )
        split_rows = []
        for lane, (history, (maxent, bvreg)) in enumerate(
            zip(result.histories, combos, strict=True)
        ):
            # Native batching retains only optimized parameter partitions in
            # convergence snapshots. Restore known frozen values for replay.
            def restore(state):
                return state._replace(
                    params=replace(
                        state.params,
                        model_parameters=state.params.model_parameters
                        if optimize_bv
                        else (parameters,),
                        forward_model_weights=lane_weights[lane],
                        forward_model_scaling=initial.forward_model_scaling,
                        normalise_loss_functions=initial.normalise_loss_functions,
                    )
                )

            history.states = [restore(state) for state in history.states]
            history.convergence_states = [
                restore(state) for state in history.convergence_states
            ]
            history.best_state = restore(history.best_state)
            history.state_parameter_partitions = None
            name = f"{ensemble}_MSE_{split}_split{split_idx:03d}_maxent{maxent:.1f}"
            if bvreg is not None:
                name += f"_bvreg{bvreg:.1f}_bvregfnL1"
            save_optimization_history_to_file(
                str(part / f"{name}_results.hdf5"), history
            )
            candidates = list(history.iter_labeled_convergence_states())
            if smoke and not candidates:
                candidates = [(np.nan, history.best_state)]
            for index, (threshold, state) in enumerate(candidates):
                fitted = state.params.model_parameters[0]
                if model == "linear":
                    np.testing.assert_array_equal(
                        fitted.interval_offsets, parameters.interval_offsets
                    )
                if not optimize_bv:
                    for leaf, original in zip(
                        jax.tree_util.tree_leaves(fitted),
                        jax.tree_util.tree_leaves(parameters),
                        strict=True,
                    ):
                        np.testing.assert_array_equal(leaf, original)
                weights = validated_frame_weight_simplex(
                    state.params.frame_weight_simplex
                )
                uptake = forward.average_frames(features, fitted, weights).uptake
                scores = {}
                for label, ds in (
                    ("train", loader.train),
                    ("val", loader.val),
                    ("full", full_loader.train),
                ):
                    ids = [int(d.top.fragment_index) for d in ds.data]
                    mapped = np.asarray(
                        ds.residue_feature_ouput_mapping.todense() @ uptake.T
                    )
                    for metric, exponent in METRICS.items():
                        scores[f"{label}_{metric}"] = weighted_mse(
                            mapped, observed_source[ids], inv_sd[ids], exponent
                        )
                        if not normalize_weights:
                            scores[f"{label}_{metric}"] *= np.mean(
                                inv_sd[ids] ** exponent
                            )
                selected_metric = "1/SD" if power == 1 else "1/SD²"
                np.testing.assert_allclose(
                    2 * np.asarray(state.losses.val_losses[0]),
                    scores[f"val_{selected_metric}"],
                    rtol=1e-5,
                    atol=1e-7,
                )
                recovery = calculate_recovery_percentage(
                    assignments,
                    np.asarray(weights),
                    targets,
                    config.scoring.state_mapping,
                )
                split_rows.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "ensemble": ensemble,
                        "split_type": split,
                        "split_idx": split_idx,
                        "weighting": selected_metric,
                        "run_id": name,
                        "maxent_value": maxent,
                        "bv_reg_value": bvreg,
                        "checkpoint_index": index,
                        "convergence_value": threshold,
                        "recovery_percent": recovery,
                        "weighted_val_mse": scores[f"val_{selected_metric}"],
                        "weighted_train_mse": scores[f"train_{selected_metric}"],
                        "val_mse": scores["val_unweighted"],
                        "full_dataset_mse": scores["full_unweighted"],
                        "ess": 1 / float(jnp.sum(weights**2)),
                        "bc": float(np.asarray(fitted.bv_bc).ravel()[0]),
                        "bh": float(np.asarray(fitted.bv_bh).ravel()[0]),
                        **scores,
                    }
                )
        if not split_rows:
            raise ValueError(f"No selectable checkpoints for {part}")
        pd.DataFrame(split_rows).to_csv(part / "candidates.csv", index=False)
        atomic_json(
            part / "complete.json",
            {
                "fits": len(combos),
                "candidates": len(split_rows),
                "executed_steps": np.asarray(result.convergence_steps).tolist(),
            },
        )
        rows.extend(split_rows)
    pd.DataFrame(rows).to_csv(out / "candidates.csv", index=False)
    atomic_json(
        out / "complete.json",
        {
            "wall_seconds": time.monotonic() - started,
            "fits": len(combos) * (1 if smoke else 3),
            "candidates": len(rows),
        },
    )


def summarize(output, jobs, smoke):
    candidates = pd.concat(
        [pd.read_csv(job_dir(output, *job) / "candidates.csv") for job in jobs],
        ignore_index=True,
    )
    selected = candidates.sort_values(
        "weighted_val_mse", kind="stable"
    ).drop_duplicates(GROUP + ["weighting"])
    summary = (
        selected.groupby(GROUP[:-1] + ["weighting"])
        .agg(
            replicates=("split_idx", "size"),
            recovery_mean=("recovery_percent", "mean"),
            recovery_sd=("recovery_percent", "std"),
            weighted_val_mse=("weighted_val_mse", "mean"),
            weighted_train_mse=("weighted_train_mse", "mean"),
            val_mse=("val_mse", "mean"),
            full_dataset_mse=("full_dataset_mse", "mean"),
            ess_mean=("ess", "mean"),
        )
        .reset_index()
    )
    if not (summary.replicates == (1 if smoke else 3)).all():
        raise ValueError("Missing split replicates in weighted fit selection")
    baseline = pd.read_csv(ROOT / "artifacts/crossval_weighted_selection/summary.csv")
    for metric, column in (
        ("unweighted", "delta_vs_unweighted_pp"),
        (None, "delta_vs_selection_only_pp"),
    ):
        ref = (
            baseline[baseline.weighting == metric]
            if metric
            else baseline[baseline.weighting != "unweighted"]
        )
        keys = GROUP[:-1] + ([] if metric else ["weighting"])
        summary = summary.merge(
            ref[keys + ["recovery_mean"]].rename(
                columns={"recovery_mean": "baseline_recovery"}
            ),
            on=keys,
            how="left",
            validate="many_to_one",
        )
        summary[column] = summary.recovery_mean - summary.baseline_recovery
        summary = summary.drop(columns="baseline_recovery")
    candidates.to_csv(output / "candidates.csv", index=False)
    selected.to_csv(output / "selected.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparison_keys = GROUP[:-1]
    comparison_columns = comparison_keys + ["recovery_mean", "recovery_sd"]
    comparison = summary.loc[summary.weighting == "1/SD", comparison_columns].rename(
        columns={"recovery_mean": "inv_sd_recovery", "recovery_sd": "inv_sd_sd"}
    )
    comparison = comparison.merge(
        summary.loc[summary.weighting == "1/SD²", comparison_columns].rename(
            columns={
                "recovery_mean": "inv_variance_recovery",
                "recovery_sd": "inv_variance_sd",
            }
        ),
        on=comparison_keys,
        validate="one_to_one",
    )
    comparison["inv_variance_minus_inv_sd_pp"] = (
        comparison.inv_variance_recovery - comparison.inv_sd_recovery
    )
    comparison.to_csv(output / "comparison.csv", index=False)
    html = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Weighted CrossVal fits</title><style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}</style><h1>Weighted fitting and validation selection</h1>",
        "<p>Matched 1/SD and 1/SD² fits. Normalized peptide × timepoint weighted MSE is used for both optimization and selection. Native batching changes execution; original grids and optimizer settings are retained. Recovery is the mean and sample SD across three selected split replicates and retains zero-target decoy mass. Recovery gains are percentage points. Full-dataset MSE is report-only and includes training/validation peptides.</p>",
        "<p><a href='summary.csv'>Summary CSV</a> · <a href='comparison.csv'>Direct weighting comparison CSV</a> · <a href='selected.csv'>Selected hyperparameters and checkpoints</a> · <a href='candidates.csv'>All scored checkpoints</a> · <a href='completion.json'>Completion audit</a></p>",
    ]
    if smoke:
        html.append(
            "<p><strong>20-step smoke checks with one replicate; these are not full benchmark results.</strong></p>"
        )
    else:
        plot_recovery_gains(output, summary)
        html.append(
            "<p><a href='../crossval_weighted_selection/report.html'>Original fitting with weighted selection only</a> · <a href='recovery_gain.svg'>Export recovery-gain chart (SVG)</a></p><img src='recovery_gain.png' alt='Mean recovery gains from weighted fitting and selection versus ordinary fitting and selection' style='max-width:100%'>"
        )
    html.append("<h2>Direct comparison: weighted fitting and selection</h2>")
    html.append(
        "<p>Positive differences favor 1/SD²; negative differences favor 1/SD. Recovery and SD are percentages; differences are percentage points.</p>"
    )
    html.append(comparison.to_html(index=False, float_format=lambda x: f"{x:.2f}"))
    for (experiment, metric), group in summary.groupby(["experiment", "weighting"]):
        html.append(f"<h2>{experiment} — {metric}</h2>")
        html.append(
            group.drop(columns=["experiment", "weighting"]).to_html(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    (output / "report.html").write_text("\n".join(html) + "</html>\n")
    atomic_json(
        output / "completion.json",
        {
            "smoke": smoke,
            "job_count": len(jobs),
            "fit_count": sum(
                json.loads((job_dir(output, *job) / "complete.json").read_text())[
                    "fits"
                ]
                for job in jobs
            ),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "summary_cells": len(summary),
        },
    )
    print(summary.to_string(index=False), flush=True)


def plot_recovery_gains(output, summary):
    import matplotlib.pyplot as plt

    labels = [
        ("AF2_MSAss", "sequence_cluster"),
        ("AF2_MSAss", "spatial"),
        ("AF2_filtered", "sequence_cluster"),
        ("AF2_filtered", "spatial"),
    ]
    modes = ("rate", "linear", "full_uptake")
    limit = max(1.0, float(summary.delta_vs_unweighted_pp.abs().max()))
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for row, experiment in enumerate(CAMPAIGNS):
        for col, metric in enumerate(("1/SD", "1/SD²")):
            data = summary[
                (summary.experiment == experiment) & (summary.weighting == metric)
            ].set_index(["ensemble", "split_type", "model"])
            values = np.array(
                [
                    [
                        data.loc[(ensemble, split, mode)].delta_vs_unweighted_pp
                        for mode in modes
                    ]
                    for ensemble, split in labels
                ]
            )
            axis = axes[row, col]
            image = axis.imshow(
                values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
            )
            axis.set_xticks(range(3), ["Rate", "Linear BV", "Full uptake"])
            axis.set_yticks(
                range(4),
                [
                    "MSAss sequence",
                    "MSAss spatial",
                    "Filtered sequence",
                    "Filtered spatial",
                ],
            )
            name = "CrossVal (fixed BV)" if row == 0 else "CrossValBV (fitted BV)"
            axis.set_title(f"{name} — {metric}")
            for y, x in itertools.product(range(4), range(3)):
                axis.text(
                    x,
                    y,
                    f"{values[y, x]:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[y, x]) > limit * 0.55 else "black",
                )
    figure.colorbar(
        image, ax=axes, label="Mean recovery gain (percentage points)", shrink=0.8
    )
    figure.suptitle(
        "Weighted fitting + selection versus ordinary MSE fitting + selection"
    )
    figure.savefig(output / "recovery_gain.png", dpi=180)
    figure.savefig(output / "recovery_gain.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/crossval_weighted_fits")
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use moprp.weights directly without normalization; compare to existing MSE fits",
    )
    parser.add_argument(
        "--worker",
        nargs=5,
        metavar=("EXPERIMENT", "MODEL", "POWER", "ENSEMBLE", "SPLIT"),
    )
    args = parser.parse_args()
    if args.raw_weights and args.output_dir == Path("artifacts/crossval_weighted_fits"):
        args.output_dir = Path("artifacts/moprp_mse_vs_weighted_mse")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        exp, model, power, ensemble, split = args.worker
        worker(
            output,
            exp,
            model,
            int(power),
            ensemble,
            split,
            args.smoke,
            not args.raw_weights,
        )
        return
    ensembles = ("AF2_MSAss",) if args.smoke else ("AF2_MSAss", "AF2_filtered")
    splits = ("spatial",) if args.smoke else ("sequence_cluster", "spatial")
    jobs = list(
        itertools.product(
            CAMPAIGNS,
            ("rate", "linear", "full_uptake"),
            (1,) if args.raw_weights else (1, 2),
            ensembles,
            splits,
        )
    )
    cores = sorted(os.sched_getaffinity(0))
    pool = queue.Queue()
    count = min(args.jobs, max(1, len(cores) // 2))
    if count < 1:
        raise ValueError("--jobs must be positive")
    for index in range(count):
        pool.put(cores[2 * index : 2 * index + 2])
    env = {
        **os.environ,
        "JAX_PLATFORM_NAME": "cpu",
        "MPLCONFIGDIR": "/tmp/crossval-weighted-fit-mpl",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def launch(job):
        out = job_dir(output, *job)
        if (out / "complete.json").exists():
            metadata = json.loads((out / "configuration.json").read_text())
            if metadata.get("normalize_weights", True) == args.raw_weights:
                raise ValueError(
                    f"Weight normalization differs from completed group: {out}"
                )
            if metadata["actual_steps"] != (20 if args.smoke else 5000):
                raise ValueError(f"Step budget differs from completed group: {out}")
            return job, 0
        cpus = pool.get()
        try:
            out.mkdir(parents=True, exist_ok=True)
            command = [
                "taskset",
                "-c",
                ",".join(map(str, cpus)),
                sys.executable,
                "-m",
                "jaxent.examples.common.analysis.fit_crossval_weighted",
                "--output-dir",
                str(output),
                "--worker",
                *map(str, job),
            ]
            if args.smoke:
                command.append("--smoke")
            if args.raw_weights:
                command.append("--raw-weights")
            with (out / "worker.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            return job, result.returncode
        finally:
            pool.put(cpus)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        pending = [executor.submit(launch, job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(pending), 1):
            job, code = future.result()
            print(f"Completed {done}/{len(jobs)}: {job}, exit={code}", flush=True)
            if code:
                failed.append(job)
    if failed:
        atomic_json(output / "failures.json", failed)
        raise RuntimeError(f"{len(failed)} failed workers; see their worker.log files")
    if args.raw_weights:
        from jaxent.examples.common.analysis.compare_moprp_raw import summarize_raw

        summarize_raw(output, jobs, args.smoke)
    else:
        summarize(output, jobs, args.smoke)


if __name__ == "__main__":
    main()
