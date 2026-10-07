"""Checkpoint 10 population inference and plots for opening-distance screening."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci, holm_adjust


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


def familywise_recovery(system: pd.DataFrame) -> pd.DataFrame:
    selected = system[
        (system.stratum == "common_support") & (system.calibration == "marginal")
        & (((system.target == "w1") & (system.band == "q5"))
           | ((system.target == "rmsd") & (system.band == "global")))
    ]
    rows = []
    for target, band in (("rmsd", "global"), ("w1", "q5")):
        endpoint = selected[(selected.target == target) & (selected.band == band)]
        for model in sorted(set(endpoint.model) - {"logpf_raw_l1", "logpf_raw_vector_ridge"}):
            baseline_name = "logpf_raw_vector_ridge" if ("vector" in model or model == "a_selected_opening") else "logpf_raw_l1"
            arm = endpoint[endpoint.model == model][["system_id", "distribution_recovery"]]
            baseline = endpoint[endpoint.model == baseline_name][["system_id", "distribution_recovery"]]
            merged = arm.merge(baseline, on="system_id", suffixes=("_arm", "_baseline"), validate="one_to_one")
            delta = 100 * (merged.distribution_recovery_arm - merged.distribution_recovery_baseline)
            low, high = bootstrap_median_ci(delta.to_numpy(), 10000, 1010 + len(rows))
            try:
                p = float(wilcoxon(delta, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            rows.append({
                "target": target, "band": band, "model": model, "baseline": baseline_name,
                "systems": len(delta), "median_recovery_improvement_pp": float(delta.median()),
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "systems_improved_percent": float(100 * np.mean(delta > 0)), "p_raw": p,
            })
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_raw.to_numpy())
    return result


def selected_gate(system: pd.DataFrame) -> dict:
    primary = system[(system.target == "w1") & (system.band == "q5")
                     & (system.stratum == "common_support")]
    adaptive_marginal = primary[(primary.model == "a_selected_opening") & (primary.calibration == "marginal")]
    adaptive_mondrian = primary[(primary.model == "a_selected_opening") & (primary.calibration == "mondrian")]
    baseline = primary[(primary.model == "logpf_raw_vector_ridge") & (primary.calibration == "marginal")]
    paired = adaptive_marginal[["system_id", "distribution_recovery"]].merge(
        baseline[["system_id", "distribution_recovery"]], on="system_id", suffixes=("_opening", "_baseline")
    )
    delta = 100 * (paired.distribution_recovery_opening - paired.distribution_recovery_baseline)
    improvement_ci = bootstrap_median_ci(delta.to_numpy(), 10000, 1099)
    recovery = float(100 * adaptive_marginal.distribution_recovery.median())
    coverage = float(100 * adaptive_mondrian.coverage_90.median())
    score_delta = float(
        adaptive_mondrian.set_index("system_id").mean_interval_score.sub(
            primary[(primary.model == "logpf_raw_vector_ridge") & (primary.calibration == "mondrian")]
            .set_index("system_id").mean_interval_score
        ).median()
    )
    passed = recovery >= 85 and 85 <= coverage <= 95 and improvement_ci[0] > 0 and score_delta <= 0
    return {
        "target": "w1", "band": "q5", "stratum": "common_support",
        "adaptive_recovery_percent": recovery, "adaptive_mondrian_coverage_percent": coverage,
        "paired_recovery_improvement_pp": float(np.median(delta)),
        "paired_recovery_improvement_ci95": [float(improvement_ci[0]), float(improvement_ci[1])],
        "mondrian_interval_score_delta": score_delta, "passed": bool(passed),
    }


def plot_top_models(population: pd.DataFrame, path) -> None:
    primary = population[(population.target == "w1") & (population.band == "q5")
                         & (population.stratum == "common_support") & (population.calibration == "marginal")]
    top = list(primary.nlargest(6, "distribution_recovery").model)
    for required in ("logpf_raw_l1", "logpf_raw_vector_ridge", "a_selected_opening"):
        if required not in top: top.append(required)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    order = [f"q{i}" for i in range(6)]; x = np.arange(6)
    for model in top:
        block = population[(population.target == "w1") & (population.model == model)
                           & (population.stratum == "common_support")
                           & (population.calibration == "marginal")].set_index("band")
        axes[0].plot(x, [100 * block.distribution_recovery.get(b, np.nan) for b in order], marker="o", label=model)
        block_c = population[(population.target == "w1") & (population.model == model)
                             & (population.stratum == "common_support")
                             & (population.calibration == "mondrian")].set_index("band")
        axes[1].plot(x, [100 * block_c.coverage_90.get(b, np.nan) for b in order], marker="o", label=model)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].set_ylabel("Empirical 90% coverage (%)"); axes[1].axhline(90, color="black", linestyle=":")
    for axis in axes:
        axis.set_xticks(x, order); axis.grid(alpha=.25)
    axes[0].set_title("Naive predicted-distribution fit"); axes[1].set_title("Mondrian calibration")
    axes[1].legend(fontsize=6, loc="lower left"); fig.suptitle("Opening-probability distance screen")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint10_opening_screen"
    summary = pd.read_parquet(output / "opening_screen_assignment_summary.parquet")
    hyper = pd.read_parquet(output / "opening_screen_hyperparameters.parquet")
    hyper["selection_mode"] = hyper.selection_mode.fillna("frame_disjoint")
    system, population = aggregate(summary); effects = familywise_recovery(system); gate = selected_gate(system)
    source_frequency = (
        hyper[hyper.model == "a_selected_opening"].groupby(["target", "selected_source"]).size()
        .rename("assignments").reset_index().sort_values(["target", "assignments"], ascending=[True, False])
    )
    selection_modes = (
        hyper[hyper.model == "a_selected_opening"].groupby(["target", "selection_mode"]).size()
        .rename("assignments").reset_index()
    )
    system.to_parquet(output / "opening_screen_system_summary.parquet", index=False)
    population.to_parquet(output / "opening_screen_population.parquet", index=False)
    effects.to_parquet(output / "opening_screen_familywise.parquet", index=False)
    source_frequency.to_parquet(output / "opening_screen_selected_sources.parquet", index=False)
    plot_top_models(population, output / "opening_screen_recovery_coverage.png")
    report = {
        "checkpoint": "10", "status": "complete", "systems": int(system.system_id.nunique()),
        "gate": gate, "selected_source_frequency": source_frequency.to_dict(orient="records"),
        "selection_modes": selection_modes.to_dict(orient="records"),
        "decision": "stage_3_supported" if not gate["passed"] else "opening_representation_gate_passed",
    }
    atomic_yaml(output / "checkpoint10_report.yaml", report)
    print("Gate:", gate)
    print("\nTop W1 selections:\n", source_frequency[source_frequency.target == "w1"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
