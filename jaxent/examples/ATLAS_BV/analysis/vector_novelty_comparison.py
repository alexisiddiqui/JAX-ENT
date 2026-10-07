"""Compare PF-novelty calibration with both previous likelihood scale models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_comparison import paired_differences


def plot_models(models: list[tuple[pd.DataFrame, str, str]], path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, target in enumerate(("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for frame, model, linestyle in models:
            # Raw and z-scored curves are nearly coincident for novelty; retain both consistently.
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
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.suptitle("Conditional-scale models: mean, variance and PF novelty")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    root = HERE / "outputs" / "analysis" / "pairwise_geometry"
    linear = pd.read_parquet(root / "checkpoint4_likelihood" / "likelihood_results.parquet")
    mean = pd.read_parquet(root / "checkpoint5_scale" / "scale_results.parquet")
    novelty_output = root / "checkpoint6_novelty"
    novelty = pd.read_parquet(novelty_output / "scale_results.parquet")
    versus_linear = paired_differences(linear, novelty); versus_linear["baseline"] = "linear_variance"
    versus_mean = paired_differences(mean, novelty); versus_mean["baseline"] = "predicted_mean_scale"
    paired = pd.concat([versus_linear, versus_mean], ignore_index=True)
    paired.to_parquet(novelty_output / "novelty_paired_comparisons.parquet", index=False)
    plot_models(
        [(linear, "linear variance", "--"), (mean, "mean scale", "-."),
         (novelty, "PF novelty", "-")],
        novelty_output / "novelty_recovery_coverage_comparison.png",
    )
    report = {
        "checkpoint": "6B", "status": "comparison_complete",
        "paired_comparisons": paired.to_dict(orient="records"),
        "conclusion": "radial PF novelty does not identify structurally novel W1-tail pairs; calibration worsens sharply, supporting directional PF degeneracy",
    }
    atomic_yaml(novelty_output / "checkpoint6b_report.yaml", report)
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
