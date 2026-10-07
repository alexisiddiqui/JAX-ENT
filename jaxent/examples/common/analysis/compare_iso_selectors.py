"""Compare MSE, closed-coordinate Sigma and GT-coordinate Sigma selectors."""

from pathlib import Path

import numpy as np
import pandas as pd


def export_comparison(output: Path, candidates: pd.DataFrame):
    index = [
        "training_loss",
        "ensemble",
        "model",
        "split_idx",
        "run_id",
        "convergence_rank",
    ]
    closed = candidates[candidates.covariance_source.eq("closed_coordinate")].copy()
    gt = candidates[candidates.covariance_source.eq("gt_coordinate")]
    if closed.duplicated(index).any() or gt.duplicated(index).any():
        raise ValueError("Ambiguous candidate identity")
    a = closed.set_index(index).sort_index()
    b = gt.set_index(index).sort_index()
    np.testing.assert_array_equal(a.index.to_numpy(), b.index.to_numpy())
    np.testing.assert_array_equal(a.ordinary_mse, b.ordinary_mse)
    np.testing.assert_array_equal(a.recovery_percent, b.recovery_percent)
    wide = closed.rename(columns={"inverse_then_split": "closed_sigma_mse"})
    wide = wide.merge(
        gt[index + ["inverse_then_split"]].rename(
            columns={"inverse_then_split": "gt_sigma_mse"}
        ),
        on=index,
        validate="one_to_one",
    )
    group = ["training_loss", "ensemble", "model", "split_idx"]
    selections = []
    selectors = {
        "MSE": "ordinary_mse",
        "Closed-Sigma": "closed_sigma_mse",
        "GT-Sigma": "gt_sigma_mse",
    }
    for label, metric in selectors.items():
        z = wide.sort_values(
            [metric, "maxent", "step", "convergence_rank"], kind="stable"
        ).drop_duplicates(group)
        selections.append(z.assign(selector=label, selection_score=z[metric]))
    selected = pd.concat(selections, ignore_index=True)
    summary = (
        selected.groupby(group[:-1] + ["selector"])
        .agg(
            replicates=("split_idx", "size"),
            recovery_mean=("recovery_percent", "mean"),
            recovery_sd=("recovery_percent", "std"),
            ordinary_val_mse=("ordinary_mse", "mean"),
            closed_sigma_val_mse=("closed_sigma_mse", "mean"),
            gt_sigma_val_mse=("gt_sigma_mse", "mean"),
        )
        .reset_index()
    )
    if not summary.replicates.eq(3).all():
        raise ValueError("Every selector must cover three replicates")
    comparison = summary.pivot(
        index=group[:-1], columns="selector", values="recovery_mean"
    ).reset_index()
    comparison["GT_minus_MSE_pp"] = comparison["GT-Sigma"] - comparison["MSE"]
    comparison["GT_minus_closed_pp"] = (
        comparison["GT-Sigma"] - comparison["Closed-Sigma"]
    )
    wide.to_csv(output / "selector_candidates.csv", index=False)
    selected.to_csv(output / "selector_selected.csv", index=False)
    summary.to_csv(output / "selector_summary.csv", index=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["MSE", "Closed-Sigma", "GT-Sigma"]
    colors = ["#4477AA", "#EE7733", "#228833"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (ensemble, model) in zip(
        axes.flat, [(e, m) for e in ["ISO_BI", "ISO_TRI"] for m in ["linear", "uptake"]]
    ):
        g = summary[
            (summary.ensemble == ensemble) & (summary.model == model)
        ].set_index(["training_loss", "selector"])
        for j, (label, color) in enumerate(zip(labels, colors)):
            z = g.loc[[(loss, label) for loss in ["mse", "closed_coordinate"]]]
            axis.bar(
                np.arange(2) + (j - 1) * 0.25,
                z.recovery_mean,
                0.25,
                yerr=z.recovery_sd,
                capsize=3,
                label=label,
                color=color,
            )
        axis.set_xticks([0, 1], ["MSE fitting", "Closed-Sigma fitting"])
        axis.set(
            ylim=(0, 105),
            ylabel="Recovery (%)",
            title=f"{ensemble} — {'Linear BV' if model == 'linear' else 'Full uptake'}",
        )
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="outside lower center", ncol=3, title="Validation selector"
    )
    fig.suptitle(
        "IsoValidation: MSE versus closed-Sigma fitting\nMSE / closed-Sigma / GT-Sigma validation selection; mean ± sample SD, three replicates"
    )
    fig.savefig(output / "panel.png", dpi=180)
    fig.savefig(output / "panel.svg")
    plt.close(fig)
    html = "<!doctype html><meta charset='utf-8'><title>IsoValidation selector comparison</title><style>body{font:15px system-ui;margin:30px}td,th{padding:6px}img{max-width:100%}</style><h1>IsoValidation fitting and selection comparison</h1><p>MSE and closed-coordinate Sigma-MSE fits, selected independently by MSE, closed-coordinate Sigma-MSE or GT-coordinate Sigma-MSE. Existing invert-then-split convention, alpha=0, original covariance and stability ridge. Native convergence checkpoints only; selection across checkpoints and MaxEnt separately for each replicate. Each candidate is replayed with the fitted forward model and saved parameters. Recovery is mean and sample SD across three replicates. Gains are percentage points. GT-Sigma uses the ground-truth population-weighted coordinate covariance; recovery is evaluation-only.</p><p><a href='comparison.csv'>Recovery and gains CSV</a> · <a href='selector_summary.csv'>Means and SDs</a> · <a href='selector_selected.csv'>Selected checkpoints</a> · <a href='audit.json'>Replay audit</a></p><img src='panel.png'><h2>Mean recovery and GT-selection gains</h2>"
    html += comparison.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    html += "<h2>Full results</h2>" + summary.to_html(
        index=False, float_format=lambda x: f"{x:.5f}"
    )
    (output / "report.html").write_text(html)
    print(comparison.to_string(index=False), flush=True)
