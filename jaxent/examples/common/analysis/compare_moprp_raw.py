"""Report raw MoPrP weighted fitting/selection versus ordinary MSE."""

from __future__ import annotations

import json
from pathlib import Path
import contextlib
import io

import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.rescore_crossval_weighted import GROUP

ROOT = Path(__file__).resolve().parents[4]


def summarize_raw(output, jobs, smoke=False):
    from jaxent.examples.common.analysis.fit_crossval_weighted import (
        atomic_json,
        job_dir,
    )

    candidates = pd.concat(
        [pd.read_csv(job_dir(output, *job) / "candidates.csv") for job in jobs],
        ignore_index=True,
    )
    selected = candidates.sort_values(
        "weighted_val_mse", kind="stable"
    ).drop_duplicates(GROUP)
    summary = (
        selected.groupby(GROUP[:-1])
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
    if not summary.replicates.eq(1 if smoke else 3).all():
        raise ValueError("Missing weighted-MSE split replicates")
    baseline = pd.read_csv(ROOT / "artifacts/crossval_weighted_selection/summary.csv")
    baseline = baseline[baseline.weighting.eq("unweighted")]
    baseline = baseline[
        GROUP[:-1] + ["recovery_mean", "recovery_sd", "val_mse", "full_dataset_mse"]
    ].rename(
        columns={
            "recovery_mean": "mse_recovery_mean",
            "recovery_sd": "mse_recovery_sd",
            "val_mse": "mse_val_mse",
            "full_dataset_mse": "mse_full_dataset_mse",
        }
    )
    comparison = summary.merge(baseline, on=GROUP[:-1], validate="one_to_one")
    # Evaluate the original MSE-selected models under the raw weighted metric
    # too, so both objectives can be compared on the same validation scores.
    from jaxent.examples.common import loading
    from jaxent.examples.common.moprp_weighted_loss import peptide_time_weights

    baseline_selected = pd.read_csv(
        ROOT / "artifacts/crossval_weighted_selection/selected.csv"
    )
    baseline_selected = baseline_selected[
        baseline_selected.weighting.eq("unweighted")
    ].copy()
    scales = {}
    for row in baseline_selected.itertuples():
        key = row.experiment, row.split_type, row.split_idx
        if key not in scales:
            with contextlib.redirect_stdout(io.StringIO()):
                _, val, full, _ = loading.load_experimental_data(
                    "",
                    str(
                        ROOT
                        / f"jaxent/examples/{row.experiment}/fitting/jaxENT/_datasplits"
                    ),
                    row.split_type,
                    row.split_idx,
                )
            scales[key] = tuple(
                float(np.asarray(peptide_time_weights(data, 1)).mean())
                for data in (val, full)
            )
    baseline_selected["mse_fit_weighted_val_mse"] = [
        row["val_1/SD"] * scales[(row.experiment, row.split_type, row.split_idx)][0]
        for _, row in baseline_selected.iterrows()
    ]
    baseline_selected["mse_fit_weighted_full_mse"] = [
        row["full_1/SD"] * scales[(row.experiment, row.split_type, row.split_idx)][1]
        for _, row in baseline_selected.iterrows()
    ]
    baseline_selected.to_csv(output / "mse_selected.csv", index=False)
    common_scores = (
        baseline_selected.groupby(GROUP[:-1])[
            ["mse_fit_weighted_val_mse", "mse_fit_weighted_full_mse"]
        ]
        .mean()
        .reset_index()
    )
    comparison = comparison.merge(common_scores, on=GROUP[:-1], validate="one_to_one")
    comparison["recovery_gain_pp"] = (
        comparison.recovery_mean - comparison.mse_recovery_mean
    )
    candidates.to_csv(output / "candidates.csv", index=False)
    selected.to_csv(output / "selected.csv", index=False)
    comparison.to_csv(output / "summary.csv", index=False)
    audit = dict(
        smoke=smoke,
        job_count=len(jobs),
        fit_count=sum(
            json.loads((job_dir(output, *job) / "complete.json").read_text())["fits"]
            for job in jobs
        ),
        candidate_count=len(candidates),
        selected_count=len(selected),
        summary_cells=len(comparison),
        file_weight_power=1,
        weight_normalization=False,
        fitting_loss="0.5 * mean(moprp.weights * residual²)",
        selection_metric="mean(moprp.weights * residual²)",
        baseline="Existing original MSE fits, selected by ordinary validation MSE",
        candidate_policy="native saved convergence checkpoints; hyperparameters selected separately for each split",
        recovery="mean and sample SD; all states including zero-target decoys retained",
    )
    atomic_json(output / "completion.json", audit)
    html = "<!doctype html><meta charset='utf-8'><title>MoPrP MSE vs raw weighted-MSE</title><style>body{font:15px system-ui;margin:30px}td,th{padding:6px}img{max-width:100%}</style><h1>MoPrP: MSE versus weighted-MSE</h1><p>Weighted-MSE uses moprp.weights directly: mean(w × residual²), with no squaring, inversion, or normalization by total weight. The optimization loss retains the standard ½ factor. Each objective is also used for its own validation selection. Original forward models, grids, regularization settings and optimizer settings are retained. Recovery is mean ± sample SD over three selected replicates; gains are percentage points. Full-dataset MSE includes training and validation and is report-only.</p><p><a href='summary.csv'>Comparison CSV</a> · <a href='selected.csv'>Weighted selected models</a> · <a href='completion.json'>Completion audit</a></p>"
    if smoke:
        html += "<p><strong>20-step smoke checks only; not full benchmark results.</strong></p>"
    else:
        plot_panel(output, comparison)
        html += "<p><a href='panel.svg'>Export panel SVG</a></p><img src='panel.png'>"
    html += comparison.to_html(index=False, float_format=lambda x: f"{x:.5f}")
    (output / "report.html").write_text(html)
    print(comparison.to_string(index=False), flush=True)


def plot_panel(output, summary):
    import matplotlib.pyplot as plt

    labels = [
        ("AF2_MSAss", "sequence_cluster"),
        ("AF2_MSAss", "spatial"),
        ("AF2_filtered", "sequence_cluster"),
        ("AF2_filtered", "spatial"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for i, experiment in enumerate(("2_CrossValidation", "3_CrossValidationBV")):
        for j, model in enumerate(("rate", "linear", "full_uptake")):
            g = (
                summary[(summary.experiment == experiment) & (summary.model == model)]
                .set_index(["ensemble", "split_type"])
                .loc[labels]
            )
            axis = axes[i, j]
            x = np.arange(len(g))
            axis.bar(
                x - 0.19,
                g.mse_recovery_mean,
                0.38,
                yerr=g.mse_recovery_sd,
                capsize=3,
                label="MSE",
                color="#4477AA",
            )
            axis.bar(
                x + 0.19,
                g.recovery_mean,
                0.38,
                yerr=g.recovery_sd,
                capsize=3,
                label="Weighted-MSE",
                color="#EE7733",
            )
            axis.set_xticks(
                x,
                [
                    "MSAss\nsequence",
                    "MSAss\nspatial",
                    "Filtered\nsequence",
                    "Filtered\nspatial",
                ],
                fontsize=8,
            )
            axis.set(
                ylim=(0, 110),
                ylabel="Recovery (%)",
                title=f"{'Fixed BV' if i == 0 else 'Fitted BV'} — {dict(rate='Rate', linear='Linear BV', full_uptake='Full uptake')[model]}",
            )
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            for k, gain in enumerate(g.recovery_gain_pp):
                axis.text(k, 105, f"{gain:+.2f} pp", ha="center", fontsize=8)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "MoPrP: MSE versus raw moprp.weights weighted-MSE\nMatched fitting and validation selection; mean ± sample SD, three replicates"
    )
    fig.savefig(output / "panel.png", dpi=180)
    fig.savefig(output / "panel.svg")
    plt.close(fig)
