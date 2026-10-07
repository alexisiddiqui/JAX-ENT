"""Population plots and paired inference for strict conformal Checkpoint 8."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import aggregate_population
from jaxent.examples.ATLAS_BV.analysis.vector_final_comparison import bootstrap_median_ci


def weighted_marginal(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["system_id", "fit_replica", "calibration_replica", "test_replica", "target", "model", "stratum"]
    rows = []
    for values, group in summary.groupby(keys, sort=False):
        weights = group.pairs.to_numpy()
        rows.append({
            **dict(zip(keys, values)), "pairs": int(weights.sum()),
            "coverage_90": float(np.average(group.coverage_90, weights=weights)),
            "mondrian_coverage_90": float(np.average(group.mondrian_coverage_90, weights=weights)),
            "historical_coverage_90": float(np.average(group.historical_coverage_90, weights=weights)),
        })
    return pd.DataFrame(rows)


def paired_model_effects(summary: pd.DataFrame) -> pd.DataFrame:
    block = summary[(summary.band.isin(["global", "q5"])) & (summary.stratum == "common_support")]
    keys = ["system_id", "fit_replica", "calibration_replica", "test_replica", "target", "band", "stratum"]
    rows = []
    for target, band in (("rmsd", "global"), ("w1", "q5")):
        selected = block[(block.target == target) & (block.band == band)]
        scalar = selected[selected.model == "absolute_l1_isotonic"]
        ridge = selected[selected.model == "raw_per_residue_ridge"]
        merged = ridge.merge(scalar, on=keys, suffixes=("_ridge", "_scalar"), validate="one_to_one")
        for metric, multiplier in (("coverage_90", 100), ("mondrian_coverage_90", 100),
                                   ("distribution_recovery", 100), ("median_interval_width", 1)):
            delta = multiplier * (merged[f"{metric}_ridge"] - merged[f"{metric}_scalar"])
            system_delta = delta.groupby(merged.system_id).mean().to_numpy()
            low, high = bootstrap_median_ci(system_delta, 10000, 811 + len(rows))
            rows.append({
                "target": target, "band": band, "metric": metric, "systems": len(system_delta),
                "median_ridge_minus_scalar": float(np.median(system_delta)),
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                "systems_positive_percent": float(100 * np.mean(system_delta > 0)),
            })
    return pd.DataFrame(rows)


def plot_results(population: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    for column, target in enumerate(("rmsd", "w1")):
        order = ["hyperlocal", "local", "global"] if target == "rmsd" else [f"q{i}" for i in range(6)]
        x = np.arange(len(order))
        for model, label, marker in (
            ("absolute_l1_isotonic", "Absolute-L1 isotonic", "o"),
            ("raw_per_residue_ridge", "Per-residue ridge", "s"),
        ):
            for stratum, linestyle in (("all", "--"), ("common_support", "-")):
                block = population[(population.target == target) & (population.model == model)
                                   & (population.stratum == stratum)].set_index("band")
                axes[0, column].plot(
                    x, [100 * block.coverage_90.get(b, np.nan) for b in order],
                    marker=marker, linestyle=linestyle, label=f"{label} / {stratum}",
                )
                axes[1, column].plot(
                    x, [100 * block.distribution_recovery.get(b, np.nan) for b in order],
                    marker=marker, linestyle=linestyle, label=f"{label} / {stratum}",
                )
        axes[0, column].set_title(target.upper()); axes[0, column].set_xticks(x, order)
        axes[1, column].set_xticks(x, order); axes[0, column].axhline(90, color="black", linestyle=":")
        axes[0, column].grid(alpha=.25); axes[1, column].grid(alpha=.25)
    axes[0, 0].set_ylabel("Strict conformal 90% coverage (%)")
    axes[1, 0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[0, 1].legend(fontsize=7); fig.suptitle("Strict A-fit / B-calibrate / C-test result")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_support(population: pd.DataFrame, path) -> None:
    strata = ["common_support", "pf_extrapolation", "pf_vector_oos", "structurally_novel"]
    labels = ["Common", "PF extrap.", "PF-vector OOS", "Structural novel"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(strata)); width = .36
    block = population[(population.target == "w1") & (population.band == "q5")]
    for offset, model, label in (
        (-width / 2, "absolute_l1_isotonic", "Absolute-L1"),
        (width / 2, "raw_per_residue_ridge", "Per-residue ridge"),
    ):
        arm = block[block.model == model].set_index("stratum")
        axes[0].bar(x + offset, [100 * arm.coverage_90.get(s, np.nan) for s in strata], width, label=label)
    ridge = block[block.model == "raw_per_residue_ridge"].set_index("stratum")
    axes[1].bar(x, [ridge.effective_frames.get(s, np.nan) for s in strata], color="#4477aa")
    axes[0].axhline(90, color="black", linestyle=":"); axes[0].set_ylabel("W1 q5 coverage (%)")
    axes[1].set_ylabel("Median effective frames"); axes[0].legend()
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("W1 q5 support-stratified strict conformal audit")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal"
    summary = pd.read_parquet(output / "strict_conformal_assignment_summary.parquet")
    population, base_report = aggregate_population(summary, load_config())
    marginal = weighted_marginal(summary)
    marginal_system = marginal.groupby(["system_id", "target", "model", "stratum"], as_index=False).agg(
        coverage_90=("coverage_90", "mean"), mondrian_coverage_90=("mondrian_coverage_90", "mean"),
        historical_coverage_90=("historical_coverage_90", "mean"),
    )
    marginal_population = marginal_system.groupby(["target", "model", "stratum"], as_index=False).median(numeric_only=True)
    effects = paired_model_effects(summary)
    marginal.to_parquet(output / "strict_conformal_marginal_assignments.parquet", index=False)
    marginal_population.to_parquet(output / "strict_conformal_marginal_population.parquet", index=False)
    effects.to_parquet(output / "strict_conformal_paired_model_effects.parquet", index=False)
    plot_results(population, output / "strict_conformal_coverage_recovery.png")
    plot_support(population, output / "strict_conformal_w1q5_support.png")
    report = {
        **base_report,
        "marginal_population": marginal_population.to_dict(orient="records"),
        "paired_ridge_minus_scalar": effects.to_dict(orient="records"),
    }
    atomic_yaml(output / "checkpoint8_final_report.yaml", report)
    print(pd.DataFrame(report["marginal_population"]).to_string(index=False))
    print("\n", effects.to_string(index=False))
    print("\nGate:", report["gate"])


if __name__ == "__main__":
    main()
