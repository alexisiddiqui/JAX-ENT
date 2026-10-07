"""Checkpoint 13 report: does strict per-system BV refitting rescue the tail?"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_comparison import aggregate
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


BASELINE = "logpf_ridge_gaussian"
REFIT = "system_refit_logpf_ridge_gaussian"


def gate(refit_system: pd.DataFrame, baseline_system: pd.DataFrame) -> dict:
    endpoint = dict(target="w1", band="q5", stratum="common_support")
    arm_m = refit_system[(refit_system.target == "w1") & (refit_system.band == "q5")
                         & (refit_system.stratum == "common_support")
                         & (refit_system.calibration == "marginal")]
    arm_c = refit_system[(refit_system.target == "w1") & (refit_system.band == "q5")
                         & (refit_system.stratum == "common_support")
                         & (refit_system.calibration == "mondrian")]
    base_m = baseline_system[(baseline_system.target == "w1") & (baseline_system.band == "q5")
                             & (baseline_system.stratum == "common_support")
                             & (baseline_system.model == BASELINE)
                             & (baseline_system.calibration == "marginal")]
    base_c = baseline_system[(baseline_system.target == "w1") & (baseline_system.band == "q5")
                             & (baseline_system.stratum == "common_support")
                             & (baseline_system.model == BASELINE)
                             & (baseline_system.calibration == "mondrian")]
    paired = arm_m[["system_id", "distribution_recovery"]].merge(
        base_m[["system_id", "distribution_recovery"]], on="system_id",
        suffixes=("_refit", "_fixed"), validate="one_to_one",
    )
    delta = 100 * (paired.distribution_recovery_refit - paired.distribution_recovery_fixed)
    ci = bootstrap_median_ci(delta.to_numpy(), 10000, 1313)
    score_pair = arm_c[["system_id", "mean_interval_score"]].merge(
        base_c[["system_id", "mean_interval_score"]], on="system_id",
        suffixes=("_refit", "_fixed"), validate="one_to_one",
    )
    recovery = float(100 * arm_m.distribution_recovery.median())
    coverage = float(100 * arm_c.coverage_90.median())
    score_delta = float((score_pair.mean_interval_score_refit - score_pair.mean_interval_score_fixed).median())
    passed = recovery >= 85 and 85 <= coverage <= 95 and ci[0] > 0 and score_delta <= 0
    return {
        **endpoint, "systems": int(len(paired)), "recovery_percent": recovery,
        "mondrian_coverage_percent": coverage,
        "recovery_improvement_pp": float(np.median(delta)),
        "recovery_improvement_ci95": [float(ci[0]), float(ci[1])],
        "mondrian_interval_score_delta": score_delta, "passed": bool(passed),
    }


def coefficient_stability(hyper: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    choices = hyper.drop_duplicates([
        "system_id", "fit_replica", "calibration_replica", "test_replica"
    ]).copy()
    choices["boundary"] = choices.bc_multiplier.isin([0.5, 2.0]) | choices.bh_multiplier.isin([0.5, 2.0])
    choices["default"] = (choices.bc_multiplier == 1.0) & (choices.bh_multiplier == 1.0)
    per_system = choices.groupby("system_id", as_index=False).agg(
        assignments=("fit_replica", "size"), unique_pairs=("bc_multiplier", lambda _: 0),
        bc_log_sd=("bc_multiplier", lambda x: float(np.std(np.log(x)))),
        bh_log_sd=("bh_multiplier", lambda x: float(np.std(np.log(x)))),
        boundary_fraction=("boundary", "mean"), default_fraction=("default", "mean"),
    )
    unique = choices.groupby("system_id").apply(
        lambda x: len(x[["bc_multiplier", "bh_multiplier"]].drop_duplicates()),
        include_groups=False,
    )
    per_system["unique_pairs"] = per_system.system_id.map(unique)
    summary = {
        "assignments": int(len(choices)),
        "boundary_selection_percent": float(100 * choices.boundary.mean()),
        "default_selection_percent": float(100 * choices.default.mean()),
        "systems_same_pair_all_assignments_percent": float(100 * (per_system.unique_pairs == 1).mean()),
        "median_unique_pairs_per_system": float(per_system.unique_pairs.median()),
        "median_bc_log_sd": float(per_system.bc_log_sd.median()),
        "median_bh_log_sd": float(per_system.bh_log_sd.median()),
    }
    return per_system, summary


def plot_results(refit_population: pd.DataFrame, baseline_population: pd.DataFrame,
                 hyper: pd.DataFrame, path) -> None:
    order = [f"q{i}" for i in range(6)]; x = np.arange(6)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for model, population, label in (
        (BASELINE, baseline_population, "Fixed BV"), (REFIT, refit_population, "Per-system refit BV")
    ):
        fit = population[(population.target == "w1") & (population.model == model)
                         & (population.calibration == "marginal")
                         & (population.stratum == "common_support")].set_index("band")
        cal = population[(population.target == "w1") & (population.model == model)
                         & (population.calibration == "mondrian")
                         & (population.stratum == "common_support")].set_index("band")
        axes[0].plot(x, [100 * fit.distribution_recovery.get(b, np.nan) for b in order], marker="o", label=label)
        axes[1].plot(x, [100 * cal.coverage_90.get(b, np.nan) for b in order], marker="o", label=label)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].set_ylabel("Empirical 90% coverage (%)"); axes[1].axhline(90, color="black", linestyle=":")
    for axis in axes[:2]: axis.set_xticks(x, order); axis.grid(alpha=.25); axis.legend(fontsize=8)
    choices = hyper.drop_duplicates(["system_id", "fit_replica", "calibration_replica", "test_replica"])
    grid = pd.crosstab(choices.bh_multiplier, choices.bc_multiplier).reindex(
        index=[0.5, .75, 1., 1.5, 2.], columns=[0.5, .75, 1., 1.5, 2.], fill_value=0
    )
    image = axes[2].imshow(grid.to_numpy(), origin="lower", cmap="viridis")
    axes[2].set_xticks(range(5), grid.columns); axes[2].set_yticks(range(5), grid.index)
    axes[2].set_xlabel("contact multiplier"); axes[2].set_ylabel("acceptor multiplier")
    for i in range(5):
        for j in range(5): axes[2].text(j, i, int(grid.iloc[i, j]), ha="center", va="center", color="white")
    fig.colorbar(image, ax=axes[2], label="ordered assignments")
    axes[0].set_title("Distribution fit across W1 bands"); axes[1].set_title("Strict calibration")
    axes[2].set_title("Selected BV coefficients"); fig.suptitle("Final contingent per-system BV refit")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint13_bv_refit"
    summary = pd.read_parquet(output / "bv_refit_assignment_summary.parquet")
    hyper = pd.read_parquet(output / "bv_refit_hyperparameters.parquet")
    refit_system, refit_population = aggregate(summary)
    previous = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint11_conditional_likelihood"
    baseline_system = pd.read_parquet(previous / "conditional_likelihood_system_summary.parquet")
    baseline_population = pd.read_parquet(previous / "conditional_likelihood_population.parquet")
    decision = gate(refit_system, baseline_system)
    stability, stability_summary = coefficient_stability(hyper)
    refit_system.to_parquet(output / "bv_refit_system_summary.parquet", index=False)
    refit_population.to_parquet(output / "bv_refit_population.parquet", index=False)
    stability.to_parquet(output / "coefficient_stability_by_system.parquet", index=False)
    plot_results(refit_population, baseline_population, hyper, output / "bv_refit_recovery_coverage.png")
    conclusion = ("per_system_bv_refit_passes_final_gate" if decision["passed"] else
                  "fixed_bv_contact_acceptor_representation_not_rescued_by_refit")
    report = {"checkpoint": "13", "status": "complete",
              "systems": int(refit_system.system_id.nunique()), "gate": decision,
              "coefficient_stability": stability_summary, "conclusion": conclusion}
    atomic_yaml(output / "checkpoint13_report.yaml", report)
    print(report)


if __name__ == "__main__":
    main()
