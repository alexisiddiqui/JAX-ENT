"""Checkpoint 12A population report and contact-community trigger."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_comparison import METRICS, aggregate
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


BASELINE = "logpf_ridge_gaussian"
LOWRANK = ("logpf_lowrank_ridge_gaussian", "logpf_lowrank_knn_mixture")


def model_decisions(lowrank_system: pd.DataFrame, baseline_system: pd.DataFrame) -> list[dict]:
    lowrank = lowrank_system[(lowrank_system.target == "w1") & (lowrank_system.band == "q5")
                             & (lowrank_system.stratum == "common_support")]
    baseline = baseline_system[(baseline_system.target == "w1") & (baseline_system.band == "q5")
                               & (baseline_system.stratum == "common_support")]
    rows = []
    for model in LOWRANK:
        arm_m = lowrank[(lowrank.model == model) & (lowrank.calibration == "marginal")]
        arm_c = lowrank[(lowrank.model == model) & (lowrank.calibration == "mondrian")]
        base_m = baseline[(baseline.model == BASELINE) & (baseline.calibration == "marginal")]
        base_c = baseline[(baseline.model == BASELINE) & (baseline.calibration == "mondrian")]
        paired = arm_m[["system_id", "distribution_recovery"]].merge(
            base_m[["system_id", "distribution_recovery"]], on="system_id", suffixes=("_arm", "_baseline")
        )
        delta = 100 * (paired.distribution_recovery_arm - paired.distribution_recovery_baseline)
        ci = bootstrap_median_ci(delta.to_numpy(), 10000, 1210 + len(rows))
        score_delta = float(
            arm_c.set_index("system_id").mean_interval_score.sub(
                base_c.set_index("system_id").mean_interval_score
            ).median()
        )
        recovery = float(100 * arm_m.distribution_recovery.median())
        coverage = float(100 * arm_c.coverage_90.median())
        trigger = float(np.median(delta)) >= 2.0 and ci[0] > 0.0 and score_delta <= 0.0
        final_gate = trigger and recovery >= 85.0 and 85.0 <= coverage <= 95.0
        rows.append({
            "model": model, "recovery_percent": recovery,
            "mondrian_coverage_percent": coverage,
            "recovery_improvement_pp": float(np.median(delta)),
            "recovery_improvement_ci95": [float(ci[0]), float(ci[1])],
            "mondrian_interval_score_delta": score_delta,
            "contact_community_trigger": bool(trigger), "final_gate": bool(final_gate),
        })
    return rows


def plot_results(lowrank_population: pd.DataFrame, baseline_population: pd.DataFrame, path) -> None:
    combined = pd.concat([
        baseline_population[baseline_population.model == BASELINE], lowrank_population
    ], ignore_index=True)
    models = (BASELINE, *LOWRANK); labels = ("Raw log-PF Gaussian", "Low-rank Gaussian", "Low-rank kNN mixture")
    order = [f"q{i}" for i in range(6)]; x = np.arange(6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model, label in zip(models, labels):
        fit = combined[(combined.target == "w1") & (combined.model == model)
                       & (combined.calibration == "marginal") & (combined.stratum == "common_support")].set_index("band")
        calibration = combined[(combined.target == "w1") & (combined.model == model)
                               & (combined.calibration == "mondrian") & (combined.stratum == "common_support")].set_index("band")
        axes[0].plot(x, [100 * fit.distribution_recovery.get(b, np.nan) for b in order], marker="o", label=label)
        axes[1].plot(x, [100 * calibration.coverage_90.get(b, np.nan) for b in order], marker="o", label=label)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].set_ylabel("Empirical 90% coverage (%)"); axes[1].axhline(90, color="black", linestyle=":")
    for axis in axes: axis.set_xticks(x, order); axis.grid(alpha=.25)
    axes[0].set_title("Joint-mode distribution fit"); axes[1].set_title("Strict calibration")
    axes[1].legend(fontsize=8); fig.suptitle("A-only low-rank joint log-PF modes")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint12_joint_lowrank"
    summary = pd.read_parquet(output / "joint_lowrank_assignment_summary.parquet")
    lowrank_system, lowrank_population = aggregate(summary)
    previous = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint11_conditional_likelihood"
    baseline_system = pd.read_parquet(previous / "conditional_likelihood_system_summary.parquet")
    baseline_population = pd.read_parquet(previous / "conditional_likelihood_population.parquet")
    decisions = model_decisions(lowrank_system, baseline_system)
    trigger = any(row["contact_community_trigger"] for row in decisions)
    lowrank_system.to_parquet(output / "joint_lowrank_system_summary.parquet", index=False)
    lowrank_population.to_parquet(output / "joint_lowrank_population.parquet", index=False)
    plot_results(lowrank_population, baseline_population, output / "joint_lowrank_recovery_coverage.png")
    report = {
        "checkpoint": "12A", "status": "complete", "systems": int(lowrank_system.system_id.nunique()),
        "models": decisions, "contact_community_trigger": bool(trigger),
        "decision": "run_contact_communities" if trigger else "skip_contact_communities_proceed_to_bv_refit",
    }
    atomic_yaml(output / "checkpoint12a_report.yaml", report)
    print(report)


if __name__ == "__main__":
    main()
