"""Final paired comparison for scalar and per-residue vector model families."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml


METRICS = (
    "distribution_recovery",
    "distribution_l1",
    "distribution_l2",
    "distribution_sqrt_jsd",
    "distribution_kld_target_to_prediction",
)
PRIMARY_ENDPOINTS = (("rmsd", "global"), ("w1", "q5"))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm familywise adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_median_ci(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    """Percentile CI for a median over independent systems."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True)
    return tuple(float(x) for x in np.quantile(np.median(sampled, axis=1), [0.025, 0.975]))


def _arm_frames(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capped_scalar = pd.read_parquet(output / "capped_scalar_baselines.parquet").copy()
    capped_scalar["arm"] = capped_scalar["representation"] + "_" + capped_scalar["metric"]
    capped_scalar["family"] = "point"
    capped_scalar["evaluation_design"] = "capped_10000_test_pairs"

    full_scalar = pd.read_parquet(output.parent / "checkpoint2_boundary" / "boundary_results.parquet")
    full_scalar = full_scalar[full_scalar.method == "clipped_all"].copy()
    full_scalar["arm"] = full_scalar["representation"] + "_" + full_scalar["metric"]
    full_scalar["family"] = "point"
    full_scalar["evaluation_design"] = "all_test_pairs"

    ridge = pd.read_parquet(output / "ridge_pca_results.parquet").copy()
    ridge["arm"] = ridge["preprocessing"] + "_" + ridge["model"]
    ridge["family"] = "point"
    ridge["evaluation_design"] = "all_test_pairs"

    knn = pd.read_parquet(output / "knn_results.parquet").copy()
    point = knn.copy()
    point["arm"] = point["preprocessing"] + "_knn"
    point["family"] = "point"
    point["evaluation_design"] = "capped_10000_test_pairs"

    conditional = knn.copy()
    conditional["arm"] = conditional["preprocessing"] + "_knn_conditional"
    conditional["family"] = "conditional"
    conditional["evaluation_design"] = "capped_10000_test_pairs"
    for metric in METRICS:
        conditional[metric] = conditional["conditional_" + metric]

    columns = [
        "system_id", "heldout_replica", "target", "band", "arm", "family",
        "evaluation_design", *METRICS
    ]
    full = pd.concat([full_scalar[columns], ridge[columns]], ignore_index=True)
    capped = pd.concat([capped_scalar[columns], point[columns]], ignore_index=True)
    return full, capped, conditional[columns], capped_scalar[columns]


def paired_tests(
    models: pd.DataFrame,
    baseline: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """One-sided paired system tests, adjusted together within one model family."""
    rows = []
    keys = ["system_id", "heldout_replica"]
    for endpoint_index, (target, band) in enumerate(PRIMARY_ENDPOINTS):
        base = baseline[(baseline.target == target) & (baseline.band == band)][
            [*keys, "distribution_recovery"]
        ].rename(columns={"distribution_recovery": "baseline_recovery"})
        endpoint = models[(models.target == target) & (models.band == band)]
        for arm in sorted(endpoint.arm.unique()):
            if arm == "raw_l1":
                continue
            merged = endpoint[endpoint.arm == arm][[*keys, "distribution_recovery"]].merge(
                base, on=keys, validate="one_to_one"
            )
            merged["delta"] = 100 * (
                merged.distribution_recovery - merged.baseline_recovery
            )
            system_delta = merged.groupby("system_id", sort=True).delta.mean().to_numpy()
            system_delta = system_delta[np.isfinite(system_delta)]
            if not len(system_delta):
                continue
            nonzero = system_delta[system_delta != 0]
            p_value = (
                1.0
                if not len(nonzero)
                else float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
            )
            low, high = bootstrap_median_ci(
                system_delta, draws, seed + 1000 * endpoint_index + len(rows)
            )
            rows.append(
                {
                    "family": str(endpoint.family.iloc[0]),
                    "target": target,
                    "band": band,
                    "arm": arm,
                    "systems": len(system_delta),
                    "median_delta_recovery_pp": float(np.median(system_delta)),
                    "bootstrap_ci95_low_pp": low,
                    "bootstrap_ci95_high_pp": high,
                    "systems_improved_percent": float(100 * np.mean(system_delta > 0)),
                    "p_one_sided": p_value,
                }
            )
    result = pd.DataFrame(rows)
    return result


def familywise_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["p_holm_familywise"] = holm_adjust(frame.p_one_sided.to_numpy())
    frame["familywise_significant_0_05"] = frame.p_holm_familywise < 0.05
    return frame


def aggregate_metrics(models: pd.DataFrame) -> pd.DataFrame:
    selected = models[
        pd.MultiIndex.from_frame(models[["target", "band"]]).isin(PRIMARY_ENDPOINTS)
    ]
    return (
        selected.groupby(["family", "target", "band", "arm"], as_index=False)[list(METRICS)]
        .median()
        .sort_values(["family", "target", "band", "distribution_recovery"], ascending=[True, True, True, False])
    )


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    point = summary[summary.family == "point"]
    panels = [
        (["raw_l1", "raw_l2", "frame_centered_l2", "raw_ridge", "zscore_ridge",
          "raw_pca_ridge", "zscore_pca_ridge"],
         ["Abs-L1", "Raw L2", "Centred L2", "Raw ridge", "Z ridge", "Raw PCA", "Z PCA"],
         "All held-out pairs"),
        (["raw_l1_capped", "raw_knn", "zscore_knn"],
         ["Abs-L1", "Raw kNN", "Z kNN"], "Identical 10,000-pair cap"),
    ]
    width = 0.37
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, (arms, labels, title) in zip(axes, panels):
        x = np.arange(len(arms))
        for offset, (target, band), endpoint_label in zip(
            (-width / 2, width / 2), PRIMARY_ENDPOINTS, ("Global RMSD", "W1 q5")
        ):
            block = point[(point.target == target) & (point.band == band)].set_index("arm")
            values = [100 * block.distribution_recovery.get(arm, np.nan) for arm in arms]
            ax.bar(x + offset, values, width, label=endpoint_label)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 70)
    axes[0].set_ylabel(r"Median distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].legend()
    fig.suptitle("Final held-out point-model distribution fit")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_report(summary: pd.DataFrame, tests: pd.DataFrame, conditional_tests: pd.DataFrame) -> dict:
    def records(frame: pd.DataFrame) -> list[dict]:
        return frame.replace({np.nan: None}).to_dict(orient="records")
    return {
        "checkpoint": "3_final_familywise",
        "status": "complete",
        "primary_endpoints": ["global RMSD", "W1 q5"],
        "fit_metric": "100 * (1 - sqrt(JSD)); dimensionless probability-distribution recovery",
        "multiplicity": "one-sided paired Wilcoxon over system-mean fold deltas; Holm correction within each prediction family across both primary endpoints",
        "point_family_tests": records(tests),
        "conditional_family_tests": records(conditional_tests),
        "aggregate_metrics": records(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint3_vector"
    full, capped, conditional, capped_scalar = _arm_frames(output)
    full_tests = paired_tests(
        full, full[full.arm == "raw_l1"], args.bootstrap_draws, args.seed
    )
    full_tests["evaluation_design"] = "all_test_pairs"
    capped_tests = paired_tests(
        capped[capped.arm.str.endswith("knn")], capped_scalar[capped_scalar.arm == "raw_l1"],
        args.bootstrap_draws, args.seed + 25000,
    )
    capped_tests["evaluation_design"] = "capped_10000_test_pairs"
    point_tests = familywise_adjust(pd.concat([full_tests, capped_tests], ignore_index=True))
    conditional_tests = familywise_adjust(paired_tests(
        conditional, capped_scalar[capped_scalar.arm == "raw_l1"],
        args.bootstrap_draws, args.seed + 50000,
    ))
    conditional_tests["evaluation_design"] = "capped_10000_test_pairs"
    capped_baseline_summary = capped_scalar[capped_scalar.arm == "raw_l1"].copy()
    capped_baseline_summary["arm"] = "raw_l1_capped"
    summary = aggregate_metrics(pd.concat([
        full, capped_baseline_summary, capped[capped.arm.str.endswith("knn")], conditional
    ], ignore_index=True))
    point_tests.to_parquet(output / "final_point_familywise_tests.parquet", index=False)
    conditional_tests.to_parquet(output / "final_conditional_familywise_tests.parquet", index=False)
    summary.to_parquet(output / "final_distribution_metrics.parquet", index=False)
    plot_comparison(summary, output / "final_point_model_recovery.png")
    atomic_yaml(output / "final_familywise_report.yaml", make_report(summary, point_tests, conditional_tests))
    print(point_tests[["target", "arm", "median_delta_recovery_pp", "p_holm_familywise"]].to_string(index=False))
    print("\nConditional family")
    print(conditional_tests[["target", "arm", "median_delta_recovery_pp", "p_holm_familywise"]].to_string(index=False))


if __name__ == "__main__":
    main()
