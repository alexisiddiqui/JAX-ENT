"""Population comparison and calibration plots for Checkpoint 4 likelihoods."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import (
    bootstrap_median_ci, holm_adjust,
)


PRIMARY = (("rmsd", "global"), ("w1", "q5"))


def paired_preprocessing(results: pd.DataFrame, draws: int = 10000, seed: int = 29) -> pd.DataFrame:
    rows = []
    for index, (target, band) in enumerate(PRIMARY):
        block = results[(results.target == target) & (results.band == band)]
        wide = block.pivot(
            index=["system_id", "heldout_replica"], columns="preprocessing",
            values="distribution_recovery",
        ).dropna()
        fold_delta = 100 * (wide.zscore - wide.raw)
        system_delta = fold_delta.groupby("system_id").mean().to_numpy()
        low, high = bootstrap_median_ci(system_delta, draws, seed + index)
        nonzero = system_delta[system_delta != 0]
        p = float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        rows.append({
            "target": target, "band": band, "systems": len(system_delta),
            "median_zscore_minus_raw_recovery_pp": float(np.median(system_delta)),
            "bootstrap_ci95_low_pp": low, "bootstrap_ci95_high_pp": high,
            "systems_improved_percent": float(100 * np.mean(system_delta > 0)),
            "p_one_sided": p,
        })
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame.p_one_sided.to_numpy())
    return frame


def band_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "distribution_recovery", "distribution_l1", "distribution_l2",
        "distribution_sqrt_jsd", "distribution_kld_target_to_prediction",
        "mean_nll", "coverage_90", "mean_sigma",
    ]
    return results.groupby(["target", "band", "preprocessing"], as_index=False)[metrics].median()


def plot_bands(summary: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, target in enumerate(("rmsd", "w1")):
        block = summary[summary.target == target]
        order = [x for x in (["hyperlocal", "local", "global"] if target == "rmsd" else
                             ["q0", "q1", "q2", "q3", "q4", "q5"]) if x in set(block.band)]
        x = np.arange(len(order))
        for preprocessing, marker in (("raw", "o"), ("zscore", "s")):
            arm = block[block.preprocessing == preprocessing].set_index("band")
            axes[0, column].plot(
                x, [100 * arm.distribution_recovery.get(b, np.nan) for b in order],
                marker=marker, label=preprocessing,
            )
            axes[1, column].plot(
                x, [100 * arm.coverage_90.get(b, np.nan) for b in order],
                marker=marker, label=preprocessing,
            )
        axes[0, column].set_title(target.upper())
        axes[0, column].set_xticks(x, order)
        axes[1, column].set_xticks(x, order)
        axes[0, column].grid(alpha=0.25); axes[1, column].grid(alpha=0.25)
        axes[1, column].axhline(90, color="black", linestyle="--", linewidth=1, label="nominal 90%")
    axes[0, 0].set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1, 0].set_ylabel("Empirical 90% interval coverage (%)")
    axes[0, 1].legend(); axes[1, 1].legend()
    fig.suptitle("Per-residue heteroscedastic likelihood across structural scale")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint4_likelihood"
    results = pd.read_parquet(output / "likelihood_results.parquet")
    hyperparameters = pd.read_parquet(output / "likelihood_hyperparameters.parquet")
    paired = paired_preprocessing(results)
    summary = band_summary(results)
    paired.to_parquet(output / "likelihood_paired_preprocessing.parquet", index=False)
    summary.to_parquet(output / "likelihood_band_summary.parquet", index=False)
    plot_bands(summary, output / "likelihood_recovery_and_coverage_bands.png")
    report = {
        "checkpoint": "4B", "status": "comparison_complete",
        "systems": int(results.system_id.nunique()),
        "folds": int(results[["system_id", "heldout_replica"]].drop_duplicates().shape[0]),
        "paired_zscore_minus_raw": paired.to_dict(orient="records"),
        "variance_alpha_maximum_percent": float(
            100 * np.mean(hyperparameters.variance_alpha == hyperparameters.variance_alpha.max())
        ),
        "median_calibration_scale": float(hyperparameters.calibration_scale.median()),
        "conclusion": "distribution fit is strong, but nominal 90% intervals remain under-covered; z-scoring improves W1 q5 recovery while worsening its coverage",
    }
    atomic_yaml(output / "checkpoint4b_report.yaml", report)
    print(pd.concat([
        summary[(summary.target == target) & (summary.band == band)] for target, band in PRIMARY
    ])[['target','band','preprocessing','distribution_recovery','coverage_90','mean_nll']].to_string(index=False))
    print("\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
