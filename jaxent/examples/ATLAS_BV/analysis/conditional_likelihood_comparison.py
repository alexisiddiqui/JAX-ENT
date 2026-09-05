"""Checkpoint 11 population report for strict conditional likelihoods."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


METRICS = (
    "distribution_recovery", "distribution_l1", "distribution_l2", "distribution_sqrt_jsd",
    "distribution_kld_target_to_prediction", "distribution_cosine_distance",
    "distribution_correlation_distance", "coverage_90", "mean_interval_score",
    "median_interval_width", "effective_frames",
)


def aggregate(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["system_id", "target", "model", "calibration", "band", "stratum"]
    system = summary.groupby(keys, as_index=False).agg(**{metric: (metric, "mean") for metric in METRICS})
    population = system.groupby(keys[1:], as_index=False).median(numeric_only=True)
    return system, population


def paired_effects(system: pd.DataFrame) -> pd.DataFrame:
    primary = system[(system.stratum == "common_support")
                     & (((system.target == "w1") & (system.band == "q5"))
                        | ((system.target == "rmsd") & (system.band == "global")))]
    baseline_name = "logpf_ridge_gaussian"
    rows = []
    for (target, band, calibration), endpoint in primary.groupby(["target", "band", "calibration"]):
        baseline = endpoint[endpoint.model == baseline_name]
        for model in sorted(set(endpoint.model) - {baseline_name}):
            arm = endpoint[endpoint.model == model]
            merged = arm.merge(baseline, on="system_id", suffixes=("_arm", "_baseline"), validate="one_to_one")
            for metric, multiplier, sign in (
                ("distribution_recovery", 100.0, 1.0), ("coverage_90", 100.0, 1.0),
                ("mean_interval_score", 1.0, -1.0),
            ):
                values = multiplier * sign * (merged[f"{metric}_arm"] - merged[f"{metric}_baseline"])
                low, high = bootstrap_median_ci(values.to_numpy(), 10000, 1110 + len(rows))
                rows.append({
                    "target": target, "band": band, "calibration": calibration, "model": model,
                    "metric": metric, "positive_means_arm_better": True, "systems": len(values),
                    "median_effect": float(values.median()), "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high, "systems_improved_percent": float(100 * np.mean(values > 0)),
                })
    return pd.DataFrame(rows)


def gate(system: pd.DataFrame) -> dict:
    primary = system[(system.target == "w1") & (system.band == "q5")
                     & (system.stratum == "common_support")]
    candidates = []
    baseline_m = primary[(primary.model == "logpf_ridge_gaussian") & (primary.calibration == "marginal")]
    baseline_c = primary[(primary.model == "logpf_ridge_gaussian") & (primary.calibration == "mondrian")]
    for model in sorted(set(primary.model) - {"logpf_ridge_gaussian"}):
        marginal = primary[(primary.model == model) & (primary.calibration == "marginal")]
        mondrian = primary[(primary.model == model) & (primary.calibration == "mondrian")]
        paired = marginal[["system_id", "distribution_recovery"]].merge(
            baseline_m[["system_id", "distribution_recovery"]], on="system_id", suffixes=("_arm", "_baseline")
        )
        delta = 100 * (paired.distribution_recovery_arm - paired.distribution_recovery_baseline)
        ci = bootstrap_median_ci(delta.to_numpy(), 10000, 1190 + len(candidates))
        score_delta = float(
            mondrian.set_index("system_id").mean_interval_score.sub(
                baseline_c.set_index("system_id").mean_interval_score
            ).median()
        )
        recovery = float(100 * marginal.distribution_recovery.median())
        coverage = float(100 * mondrian.coverage_90.median())
        passed = recovery >= 85 and 85 <= coverage <= 95 and ci[0] > 0 and score_delta <= 0
        candidates.append({
            "model": model, "recovery_percent": recovery, "mondrian_coverage_percent": coverage,
            "recovery_improvement_pp": float(np.median(delta)),
            "recovery_improvement_ci95": [float(ci[0]), float(ci[1])],
            "mondrian_interval_score_delta": score_delta, "passed": bool(passed),
        })
    winner = max(candidates, key=lambda row: row["recovery_percent"])
    return {"target": "w1", "band": "q5", "stratum": "common_support",
            "baseline_stage9_recovery_percent": 83.14016971241543,
            "candidates": candidates, "winner": winner, "passed": any(row["passed"] for row in candidates)}


def plot_results(population: pd.DataFrame, path) -> None:
    models = ["logpf_ridge_gaussian", "opening_ridge_gaussian", "logpf_knn_mixture", "opening_knn_mixture"]
    labels = ["log-PF Gaussian", "opening Gaussian", "log-PF kNN mixture", "opening kNN mixture"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8)); order = [f"q{i}" for i in range(6)]; x = np.arange(6)
    for model, label in zip(models, labels):
        fit = population[(population.target == "w1") & (population.model == model)
                         & (population.calibration == "marginal") & (population.stratum == "common_support")].set_index("band")
        calibration = population[(population.target == "w1") & (population.model == model)
                                 & (population.calibration == "mondrian") & (population.stratum == "common_support")].set_index("band")
        axes[0].plot(x, [100 * fit.distribution_recovery.get(b, np.nan) for b in order], marker="o", label=label)
        axes[1].plot(x, [100 * calibration.coverage_90.get(b, np.nan) for b in order], marker="o", label=label)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].set_ylabel("Empirical 90% coverage (%)"); axes[1].axhline(90, color="black", linestyle=":")
    for axis in axes: axis.set_xticks(x, order); axis.grid(alpha=.25)
    axes[0].set_title("Conditional-distribution fit"); axes[1].set_title("Conformalized interval calibration")
    axes[1].legend(fontsize=7); fig.suptitle("Strict per-residue conditional likelihood")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint11_conditional_likelihood"
    summary = pd.read_parquet(output / "conditional_likelihood_assignment_summary.parquet")
    system, population = aggregate(summary); effects = paired_effects(system); decision = gate(system)
    system.to_parquet(output / "conditional_likelihood_system_summary.parquet", index=False)
    population.to_parquet(output / "conditional_likelihood_population.parquet", index=False)
    effects.to_parquet(output / "conditional_likelihood_paired_effects.parquet", index=False)
    plot_results(population, output / "conditional_likelihood_recovery_coverage.png")
    report = {"checkpoint": "11", "status": "complete", "systems": int(system.system_id.nunique()),
              "gate": decision, "decision": "stage_4_supported" if not decision["passed"] else "conditional_gate_passed"}
    atomic_yaml(output / "checkpoint11_report.yaml", report)
    print("Gate:", decision)


if __name__ == "__main__":
    main()
