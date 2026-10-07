"""Report checkpoint-17 MD KDE-density versus fixed-BV population-ratio results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml


OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint17_kde_population"
PRIMARY_RANK = 10
ORDER = [f"q{i}" for i in range(6)]
MODELS = {
    "signed_sum_alpha1": "Signed BV sum, α=1",
    "signed_sum_local_alpha": "Signed BV sum, local α",
    "signed_mean_global_alpha": "Signed BV mean, LOSO α",
    "absolute_l1_local_alpha": "Absolute L1, local α",
    "l2_local_alpha": "L2, local α",
    "cosine_local_alpha": "Cosine, local α",
    "correlation_local_alpha": "Correlation, local α",
    "per_residue_ridge": "Per-residue ridge",
}


def system_then_population(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["system_id", "rank", "model", "target_kind", "structural_axis", "band", "stratum",
            "band_low_angstrom", "band_high_angstrom"]
    metrics = ["distribution_recovery", "distribution_sqrt_jsd", "mae", "spearman", "coverage_90",
               "mean_interval_score", "pairs"]
    system = summary.groupby(keys, as_index=False, observed=True)[metrics].mean()
    population_keys = [key for key in keys if key != "system_id"]
    population = system.groupby(population_keys, as_index=False, observed=True)[metrics].median()
    spread = (system.groupby(population_keys, as_index=False, observed=True).distribution_recovery.std()
              .rename(columns={"distribution_recovery": "recovery_system_sd"}))
    return system, population.merge(spread, on=population_keys, how="left", validate="one_to_one")


def plot_recovery(population: pd.DataFrame) -> None:
    selected = population[(population["rank"] == PRIMARY_RANK)
                          & (population.structural_axis == "w1")
                          & (population.stratum == "common_support")
                          & population.model.isin(MODELS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for axis, kind, title in zip(axes, ("signed", "magnitude"),
                                 ("Signed pairwise population change", "Magnitude of population change")):
        block = selected[selected.target_kind == kind]
        for model in MODELS:
            rows = block[block.model == model].set_index("band")
            if rows.empty:
                continue
            y = 100 * np.array([rows.distribution_recovery.get(q, np.nan) for q in ORDER])
            sd = 100 * np.array([rows.recovery_system_sd.get(q, np.nan) for q in ORDER])
            line = axis.plot(np.arange(6), y, marker="o", lw=2, label=MODELS[model])[0]
            axis.fill_between(np.arange(6), np.maximum(0, y - sd), np.minimum(100, y + sd),
                              alpha=.12, color=line.get_color(), linewidth=0)
        labels = []
        for index, q in enumerate(ORDER):
            row = block[block.band == q]
            if row.empty:
                labels.append(q)
            else:
                item = row.iloc[0]
                labels.append(f"{q}\n{item.band_low_angstrom:.3f}–{item.band_high_angstrom:.3f} Å")
        axis.set_xticks(np.arange(6), labels)
        axis.set_title(title); axis.set_xlabel("Global frame-pair W1 band")
        axis.grid(alpha=.25); axis.legend(fontsize=8)
    axes[0].set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    fig.suptitle("MD W1-kernel target density versus fixed-BV pairwise predictions\n"
                 "System medians; shading ±1 system SD; KDE neighbour rank 10")
    fig.tight_layout(); fig.savefig(OUTPUT / "kde_population_recovery_global_w1.png", dpi=180); plt.close(fig)


def plot_scale_diagnostic(system: pd.DataFrame) -> None:
    selected = system[(system["rank"] == PRIMARY_RANK) & (system.structural_axis == "w1")
                      & (system.stratum == "common_support")]
    pairs = [
        ("signed_sum_alpha1", "signed_sum_local_alpha", "Signed BV sum"),
        ("absolute_l1_alpha1", "absolute_l1_local_alpha", "Absolute L1"),
        ("l2_alpha1", "l2_local_alpha", "L2"),
    ]
    records = []
    for fixed, fitted, label in pairs:
        left = (selected[selected.model == fixed].groupby("system_id").distribution_recovery.mean()
                .rename("alpha1"))
        right = (selected[selected.model == fitted].groupby("system_id").distribution_recovery.mean()
                 .rename("fitted"))
        joined = pd.concat([left, right], axis=1).dropna()
        for system_id, row in joined.iterrows():
            records.append({"system_id": system_id, "metric": label,
                            "alpha1_recovery": row.alpha1, "fitted_recovery": row.fitted})
    frame = pd.DataFrame(records); frame.to_parquet(OUTPUT / "kde_population_scale_diagnostic.parquet", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 6.6))
    for label, block in frame.groupby("metric"):
        axis.scatter(100 * block.alpha1_recovery, 100 * block.fitted_recovery, s=35, alpha=.75, label=label)
    axis.axline((0, 0), slope=1, color="black", ls="--", lw=1)
    axis.set(xlabel="Recovery with α=1 (%)", ylabel="Recovery with fitted per-system α (%)",
             title="Does scale fitting improve distribution recovery?")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(OUTPUT / "kde_population_fixed_vs_fitted_scale.png", dpi=180); plt.close(fig)


def plot_coverage(population: pd.DataFrame) -> None:
    selected = population[(population["rank"] == PRIMARY_RANK) & (population.structural_axis == "w1")
                          & (population.stratum == "common_support") & population.model.isin(MODELS)]
    fig, axis = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        rows = selected[selected.model == model].set_index("band")
        if rows.empty: continue
        axis.plot(range(6), 100 * np.array([rows.coverage_90.get(q, np.nan) for q in ORDER]),
                  marker="o", lw=2, label=MODELS[model])
    axis.axhline(90, color="black", ls="--", lw=1.5, label="Nominal 90%")
    axis.set(xticks=range(6), xticklabels=ORDER, xlabel="Global frame-pair W1 band",
             ylabel="Strict A-fit/B-calibrate/C-test coverage (%)", ylim=(0, 101),
             title="Population-change interval coverage across W1 scale")
    axis.grid(alpha=.25); axis.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(OUTPUT / "kde_population_strict_coverage.png", dpi=180); plt.close(fig)


def plot_alphas(fits: pd.DataFrame, global_fits: pd.DataFrame) -> None:
    local = fits[(fits["rank"] == PRIMARY_RANK) & fits.model.isin(["signed_sum", *(
        "absolute_l1", "l2", "cosine", "correlation")])].copy()
    local["source"] = "Per-system"
    global_block = global_fits[(global_fits["rank"] == PRIMARY_RANK)].copy()
    global_block["alpha"] = global_block.global_alpha_loso; global_block["source"] = "LOSO global"
    data = pd.concat([local[["model", "alpha", "source"]], global_block[["model", "alpha", "source"]]])
    models = list(dict.fromkeys(data.model)); fig, axis = plt.subplots(figsize=(10, 5.8))
    for offset, (source, block) in zip((-.16, .16), data.groupby("source")):
        values = [block.loc[block.model == model, "alpha"].dropna().to_numpy() for model in models]
        positions = np.arange(len(models)) + offset
        axis.boxplot(values, positions=positions, widths=.28, patch_artist=True,
                     boxprops={"alpha": .45}, medianprops={"color": "black"})
    axis.set_xticks(range(len(models)), models, rotation=25, ha="right")
    axis.set_yscale("symlog", linthresh=1e-3)
    axis.set_ylabel("Fitted α (symlog)"); axis.set_title("Scale coefficients at KDE neighbour rank 10")
    axis.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUTPUT / "kde_population_alpha_distributions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    summary = pd.read_parquet(OUTPUT / "kde_population_assignment_summary.parquet")
    fits = pd.read_parquet(OUTPUT / "kde_population_fits.parquet")
    global_fits = pd.read_parquet(OUTPUT / "kde_population_global_scales.parquet")
    system, population = system_then_population(summary)
    system.to_parquet(OUTPUT / "kde_population_system_summary.parquet", index=False)
    population.to_parquet(OUTPUT / "kde_population_population_summary.parquet", index=False)
    plot_recovery(population); plot_scale_diagnostic(system); plot_coverage(population); plot_alphas(fits, global_fits)
    headline = population[(population["rank"] == PRIMARY_RANK) & (population.structural_axis == "w1")
                          & (population.stratum == "common_support") & population.model.isin(MODELS)]
    table = headline.pivot(index="model", columns="band", values="distribution_recovery") * 100
    table.round(2).to_csv(OUTPUT / "kde_population_recovery_percent.csv")
    atomic_yaml(OUTPUT / "checkpoint17_report.yaml", {"checkpoint": 17, "status": "report_complete",
                "primary_neighbour_rank": PRIMARY_RANK, "recovery_definition": "100*(1-sqrt(JSD))"})
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
