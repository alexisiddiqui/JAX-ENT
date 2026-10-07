"""Final paired comparison for exact nearest-PF support calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


PRIMARY = (("rmsd", "global"), ("w1", "q5"))


def paired_nearest_constant(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["system_id", "heldout_replica", "target", "band", "preprocessing"]
    rows = []
    for target, band in PRIMARY:
        for preprocessing in ("raw", "zscore"):
            block = results[(results.target == target) & (results.band == band) &
                            (results.preprocessing == preprocessing)]
            constant = block[block.model == "constant_scale"]
            nearest = block[block.model == "nearest_distance_scale"]
            merged = nearest.merge(constant, on=keys, suffixes=("_nearest", "_constant"),
                                   validate="one_to_one")
            for metric, multiplier in (("distribution_recovery", 100), ("coverage_90", 100),
                                       ("mean_nll", 1), ("mean_q90_width", 1)):
                delta = multiplier * (
                    merged[f"{metric}_nearest"] - merged[f"{metric}_constant"]
                )
                systems = delta.groupby(merged.system_id).mean().to_numpy()
                low, high = bootstrap_median_ci(systems, 10000, 71 + len(rows))
                rows.append({
                    "target": target, "band": band, "preprocessing": preprocessing,
                    "metric": metric, "systems": len(systems),
                    "median_nearest_minus_constant": float(np.median(systems)),
                    "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                    "systems_positive_percent": float(100 * np.mean(systems > 0)),
                })
    return pd.DataFrame(rows)


def knn_conditional_comparison(results: pd.DataFrame, knn: pd.DataFrame) -> pd.DataFrame:
    keys = ["system_id", "heldout_replica", "target", "band", "preprocessing"]
    rows = []
    nearest = results[results.model == "nearest_distance_scale"]
    for target, band in PRIMARY:
        for preprocessing in ("raw", "zscore"):
            left = nearest[(nearest.target == target) & (nearest.band == band) &
                           (nearest.preprocessing == preprocessing)]
            right = knn[(knn.target == target) & (knn.band == band) &
                        (knn.preprocessing == preprocessing)]
            merged = left.merge(right, on=keys, suffixes=("_nearest", "_knn"),
                                validate="one_to_one")
            delta = 100 * (
                merged.distribution_recovery_nearest
                - merged.conditional_distribution_recovery
            )
            systems = delta.groupby(merged.system_id).mean().to_numpy()
            low, high = bootstrap_median_ci(systems, 10000, 211 + len(rows))
            rows.append({
                "target": target, "band": band, "preprocessing": preprocessing,
                "systems": len(systems),
                "median_nearest_minus_knn_conditional_recovery_pp": float(np.median(systems)),
                "bootstrap_ci95_low_pp": low, "bootstrap_ci95_high_pp": high,
            })
    return pd.DataFrame(rows)


def plot_bands(results: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, target in enumerate(("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for model, linestyle in (("constant_scale", "--"), ("nearest_distance_scale", "-")):
            for preprocessing, marker in (("raw", "o"), ("zscore", "s")):
                block = results[(results.target == target) & (results.model == model) &
                                (results.preprocessing == preprocessing)]
                recovery = block.groupby("band").distribution_recovery.median()
                coverage = block.groupby("band").coverage_90.median()
                label = f"{model.replace('_', ' ')} / {preprocessing}"
                axes[0, column].plot(x, [100 * recovery.get(b, np.nan) for b in order],
                                     marker=marker, linestyle=linestyle, label=label)
                axes[1, column].plot(x, [100 * coverage.get(b, np.nan) for b in order],
                                     marker=marker, linestyle=linestyle, label=label)
        axes[0, column].set_title(target.upper()); axes[0, column].set_xticks(x, order)
        axes[1, column].set_xticks(x, order); axes[1, column].axhline(90, color="black", linestyle=":")
        axes[0, column].grid(alpha=.25); axes[1, column].grid(alpha=.25)
    axes[0, 0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1, 0].set_ylabel("Empirical 90% coverage (%)")
    axes[0, 1].legend(fontsize=8); fig.suptitle("Exact directional PF support calibration")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    root = HERE / "outputs" / "analysis" / "pairwise_geometry"
    output = root / "checkpoint7_nearest"
    results = pd.read_parquet(output / "nearest_results.parquet")
    knn = pd.read_parquet(root / "checkpoint3_vector" / "knn_results.parquet")
    paired = paired_nearest_constant(results)
    knn_comparison = knn_conditional_comparison(results, knn)
    paired.to_parquet(output / "nearest_paired_vs_constant.parquet", index=False)
    knn_comparison.to_parquet(output / "nearest_vs_knn_conditional.parquet", index=False)
    plot_bands(results, output / "nearest_recovery_coverage_bands.png")
    report = {
        "checkpoint": "7B", "status": "comparison_complete",
        "paired_nearest_minus_constant": paired.to_dict(orient="records"),
        "nearest_minus_knn_conditional": knn_comparison.to_dict(orient="records"),
        "conclusion": "directional nearest-PF distance transfers a modest support signal but leaves catastrophic W1-tail undercoverage; support calibration is insufficient",
    }
    atomic_yaml(output / "checkpoint7b_report.yaml", report)
    print(paired.to_string(index=False)); print("\n", knn_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
