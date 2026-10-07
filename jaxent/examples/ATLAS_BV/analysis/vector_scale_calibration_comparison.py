"""Paired Checkpoint 5 comparison with the Checkpoint 4 likelihood."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


PRIMARY = (("rmsd", "global"), ("w1", "q5"))


def paired_differences(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["system_id", "heldout_replica", "target", "band", "preprocessing"]
    rows = []
    for target, band in PRIMARY:
        for preprocessing in ("raw", "zscore"):
            left = old[(old.target == target) & (old.band == band) &
                       (old.preprocessing == preprocessing)]
            right = new[(new.target == target) & (new.band == band) &
                         (new.preprocessing == preprocessing)]
            merged = right.merge(left, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
            for metric, multiplier in (("distribution_recovery", 100), ("coverage_90", 100),
                                       ("mean_nll", 1)):
                delta = multiplier * (merged[f"{metric}_new"] - merged[f"{metric}_old"])
                systems = delta.groupby(merged.system_id).mean().to_numpy()
                low, high = bootstrap_median_ci(systems, 10000, 47 + len(rows))
                rows.append({
                    "target": target, "band": band, "preprocessing": preprocessing,
                    "metric": metric, "systems": len(systems),
                    "median_binned_minus_linear": float(np.median(systems)),
                    "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                    "systems_positive_percent": float(100 * np.mean(systems > 0)),
                })
    return pd.DataFrame(rows)


def plot_comparison(old: pd.DataFrame, new: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, target in enumerate(("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for frame, model, linestyle in ((old, "linear variance", "--"), (new, "mean-binned scale", "-")):
            for preprocessing, marker in (("raw", "o"), ("zscore", "s")):
                block = frame[(frame.target == target) & (frame.preprocessing == preprocessing)]
                recovery = block.groupby("band").distribution_recovery.median()
                coverage = block.groupby("band").coverage_90.median()
                label = f"{model} / {preprocessing}"
                axes[0, column].plot(x, [100 * recovery.get(b, np.nan) for b in order],
                                     marker=marker, linestyle=linestyle, label=label)
                axes[1, column].plot(x, [100 * coverage.get(b, np.nan) for b in order],
                                     marker=marker, linestyle=linestyle, label=label)
        axes[0, column].set_title(target.upper()); axes[0, column].set_xticks(x, order)
        axes[1, column].set_xticks(x, order); axes[1, column].axhline(90, color="black", linestyle=":")
        axes[0, column].grid(alpha=.25); axes[1, column].grid(alpha=.25)
    axes[0, 0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1, 0].set_ylabel("Empirical 90% coverage (%)")
    axes[0, 1].legend(fontsize=8); fig.suptitle("Linear-variance versus predicted-mean scale calibration")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    root = HERE / "outputs" / "analysis" / "pairwise_geometry"
    output = root / "checkpoint5_scale"
    old = pd.read_parquet(root / "checkpoint4_likelihood" / "likelihood_results.parquet")
    new = pd.read_parquet(output / "scale_results.parquet")
    paired = paired_differences(old, new)
    paired.to_parquet(output / "scale_paired_vs_linear_variance.parquet", index=False)
    plot_comparison(old, new, output / "scale_recovery_coverage_comparison.png")
    hyperparameters = pd.read_parquet(output / "scale_hyperparameters.parquet")
    report = {
        "checkpoint": "5B", "status": "comparison_complete",
        "paired_differences": paired.to_dict(orient="records"),
        "constant_scale_selected_percent": float(100 * np.mean(hyperparameters.scale_bins == 1)),
        "conclusion": "predicted-mean scale bins improve pooled distribution fit but worsen W1-tail coverage; mean underprediction hides structural novelty from the scale calibrator",
    }
    atomic_yaml(output / "checkpoint5b_report.yaml", report)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
