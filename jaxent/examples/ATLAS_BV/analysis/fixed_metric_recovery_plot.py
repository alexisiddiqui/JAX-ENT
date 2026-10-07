"""Plot fixed-coefficient recovery across W1 quantiles for key PF distances."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE


def recovery_table(population: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    selected = population[
        (population.target == "w1") & (population.calibration == "marginal")
        & (population.stratum == "common_support") & population.model.isin(models)
    ][["model", "band", "distribution_recovery"]].copy()
    selected["label"] = selected.model.map(models)
    selected["recovery_percent"] = 100 * selected.distribution_recovery
    return selected[["model", "label", "band", "recovery_percent"]]


def add_system_sd(table: pd.DataFrame, system: pd.DataFrame) -> pd.DataFrame:
    sd = (
        system[(system.target == "w1") & (system.calibration == "marginal")
               & (system.stratum == "common_support") & system.model.isin(table.model.unique())]
        .groupby(["model", "band"], as_index=False).distribution_recovery.std()
        .rename(columns={"distribution_recovery": "system_sd_percent"})
    )
    sd.system_sd_percent *= 100
    return table.merge(sd, on=["model", "band"], how="left", validate="one_to_one")


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint14_fixed_metrics"
    scalar_summary = pd.read_parquet(output / "fixed_metric_assignment_summary.parquet")
    from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_comparison import aggregate
    scalar_system, scalar_population = aggregate(scalar_summary)
    baseline_path = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint11_conditional_likelihood"
    ridge_population = pd.read_parquet(baseline_path / "conditional_likelihood_population.parquet")
    ridge_system = pd.read_parquet(baseline_path / "conditional_likelihood_system_summary.parquet")
    scalar_names = {f"fixed_logpf_scalar_{name}_gaussian": label for name, label in (
        ("l1", "Absolute-L1 (scalar)"), ("l2", "L2 (scalar)"),
        ("cosine", "Cosine (scalar)"), ("correlation", "Correlation (scalar)"),
    )}
    raw = add_system_sd(recovery_table(scalar_population, scalar_names), scalar_system)
    ridge = add_system_sd(
        recovery_table(ridge_population, {"logpf_ridge_gaussian": "Per-residue ridge"}), ridge_system
    )
    raw = pd.concat([raw, ridge], ignore_index=True)
    raw_models = {**scalar_names, "logpf_ridge_gaussian": "Per-residue ridge"}
    comparison = raw.assign(representation="raw_logpf")
    comparison.to_parquet(output / "fixed_metric_w1_recovery.parquet", index=False)

    order = [f"q{i}" for i in range(6)]; x = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(9.5, 5.6))
    for model, label in raw_models.items():
        block = raw[raw.model == model].set_index("band")
        values = np.array([block.recovery_percent.get(band, np.nan) for band in order])
        sd = np.array([block.system_sd_percent.get(band, np.nan) for band in order])
        line = axis.plot(x, values, marker="o", linewidth=2, label=label)[0]
        axis.fill_between(x, np.maximum(0, values - sd), np.minimum(100, values + sd),
                          color=line.get_color(), alpha=.12, linewidth=0)
    axis.set_xticks(x, order); axis.set_xlabel("Held-out coordinate-W1 quantile band")
    axis.set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axis.set_title("Fixed BV log-PF: distribution fit across W1 scale\nShading: ±1 SD across systems")
    axis.grid(alpha=.25); axis.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output / "fixed_metric_w1_recovery.png", dpi=180); plt.close(fig)
    print(comparison.pivot(index=["representation", "label"], columns="band", values="recovery_percent").round(2))


if __name__ == "__main__":
    main()
