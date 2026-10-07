"""Aggregate and plot checkpoint-18 thermodynamic population metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml
from jaxent.examples.ATLAS_BV.analysis.kde_population_report_checkpoint17 import system_then_population
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import THERMODYNAMIC_METRICS


OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint18_thermodynamic_population"
BASELINE_OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint17_kde_population"
ORDER = [f"q{i}" for i in range(6)]
PRIMARY_RANK = 10
THERMO_LABELS = {
    "work_shape": r"Work Shape, $\delta H_{opt}$",
    "work_scale": r"Work Scale, $\delta H_{abs}$",
    "work_density_legacy_zq": r"Work Density, legacy $Zq$",
    "work_density_unnormalized_q": r"Work Density, unnormalized $q$",
    "work_density_normalized_q_over_z": r"Work Density, normalized $q/Z$",
}
BASELINE_LABELS = {
    "absolute_l1_local_alpha": "Absolute L1, local α",
    "l2_local_alpha": "L2, local α",
    "cosine_local_alpha": "Cosine, local α",
    "correlation_local_alpha": "Correlation, local α",
}


def selected(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["rank"] == PRIMARY_RANK) & (frame.structural_axis == "w1")
                 & (frame.stratum == "common_support")]


def curve(axis, frame: pd.DataFrame, model: str, label: str) -> None:
    rows = frame[frame.model == model].set_index("band")
    y = 100 * np.array([rows.distribution_recovery.get(q, np.nan) for q in ORDER])
    sd = 100 * np.array([rows.recovery_system_sd.get(q, np.nan) for q in ORDER])
    line = axis.plot(range(6), y, marker="o", lw=2, label=label)[0]
    axis.fill_between(range(6), np.maximum(0, y - sd), np.minimum(100, y + sd),
                      color=line.get_color(), alpha=.11, linewidth=0)


def plot_headline(thermo: pd.DataFrame, baseline: pd.DataFrame) -> None:
    thermo = selected(thermo); baseline = selected(baseline)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), sharey=True)
    for metric, label in THERMO_LABELS.items():
        curve(axes[0], thermo, f"{metric}_local_alpha", f"{label}, local α")
    for model, label in BASELINE_LABELS.items():
        curve(axes[1], baseline, model, label)
    labels = []
    for index, band in enumerate(ORDER):
        candidates = thermo[thermo.band == band]
        if candidates.empty:
            candidates = baseline[baseline.band == band]
        if candidates.empty:
            labels.append(band)
        else:
            row = candidates.iloc[0]
            labels.append(f"{band}\n{row.band_low_angstrom:.3f}–{row.band_high_angstrom:.3f} Å")
    for axis, title in zip(axes, ("Thermodynamic BV metrics", "Checkpoint-17 magnitude references")):
        axis.set_xticks(range(6), labels); axis.set_xlabel("Global frame-pair W1 band")
        axis.set_title(title); axis.grid(alpha=.25); axis.legend(fontsize=8)
    axes[0].set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    fig.suptitle("MD W1-kernel population magnitude: thermodynamic third arm\n"
                 "System medians; shading ±1 system SD; KDE neighbour rank 10")
    fig.tight_layout(); fig.savefig(OUTPUT / "thermodynamic_recovery_global_w1.png", dpi=180); plt.close(fig)


def plot_entropy_variants(thermo: pd.DataFrame) -> None:
    frame = selected(thermo); fig, axis = plt.subplots(figsize=(10, 6.2))
    for metric in THERMODYNAMIC_METRICS[2:]:
        curve(axis, frame, f"{metric}_local_alpha", f"{THERMO_LABELS[metric]}, local α")
        curve(axis, frame, f"{metric}_global_alpha", f"{THERMO_LABELS[metric]}, LOSO α")
    axis.set(xticks=range(6), xticklabels=ORDER, xlabel="Global frame-pair W1 band",
             ylabel=r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)",
             title="Effect of entropy normalization and coefficient transfer")
    axis.grid(alpha=.25); axis.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(OUTPUT / "entropy_normalization_recovery.png", dpi=180); plt.close(fig)


def plot_scale_diagnostic(system: pd.DataFrame) -> None:
    frame = selected(system); records = []
    for metric, label in THERMO_LABELS.items():
        raw = frame[frame.model == f"{metric}_alpha1"].groupby("system_id").distribution_recovery.mean()
        fitted = frame[frame.model == f"{metric}_local_alpha"].groupby("system_id").distribution_recovery.mean()
        joined = pd.concat([raw.rename("raw"), fitted.rename("fitted")], axis=1).dropna()
        for system_id, row in joined.iterrows():
            records.append({"system_id": system_id, "metric": metric, "label": label,
                            "alpha1_recovery": row.raw, "local_alpha_recovery": row.fitted})
    result = pd.DataFrame(records)
    result.to_parquet(OUTPUT / "thermodynamic_scale_diagnostic.parquet", index=False)
    fig, axis = plt.subplots(figsize=(7.5, 6.7))
    for label, block in result.groupby("label"):
        axis.scatter(100 * block.alpha1_recovery, 100 * block.local_alpha_recovery,
                     s=28, alpha=.65, label=label)
    axis.axline((0, 0), slope=1, color="black", ls="--", lw=1)
    axis.set(xlabel="Recovery with α=1 (%)", ylabel="Recovery with fitted local α (%)",
             title="Thermodynamic metrics: parameter-free versus fitted scale")
    axis.grid(alpha=.25); axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTPUT / "thermodynamic_fixed_vs_fitted_scale.png", dpi=180); plt.close(fig)


def plot_coverage(thermo: pd.DataFrame) -> None:
    frame = selected(thermo); fig, axis = plt.subplots(figsize=(10.5, 6))
    for metric, label in THERMO_LABELS.items():
        rows = frame[frame.model == f"{metric}_local_alpha"].set_index("band")
        axis.plot(range(6), 100 * np.array([rows.coverage_90.get(q, np.nan) for q in ORDER]),
                  marker="o", lw=2, label=label)
    axis.axhline(90, color="black", ls="--", lw=1.5, label="Nominal 90%")
    axis.set(xticks=range(6), xticklabels=ORDER, xlabel="Global frame-pair W1 band",
             ylabel="Strict A-fit/B-calibrate/C-test coverage (%)", ylim=(0, 101),
             title="Thermodynamic population-change interval coverage")
    axis.grid(alpha=.25); axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTPUT / "thermodynamic_strict_coverage.png", dpi=180); plt.close(fig)


def plot_alpha_distributions(fits: pd.DataFrame, global_fits: pd.DataFrame) -> None:
    local = fits[fits["rank"] == PRIMARY_RANK][["model", "alpha"]].copy()
    local["source"] = "Per-system α"
    loso = global_fits[global_fits["rank"] == PRIMARY_RANK][
        ["model", "global_alpha_loso"]
    ].rename(columns={"global_alpha_loso": "alpha"})
    loso["source"] = "LOSO α"
    data = pd.concat([local, loso], ignore_index=True)
    positions = np.arange(len(THERMODYNAMIC_METRICS), dtype=float)
    colors = {"Per-system α": "#4c78a8", "LOSO α": "#f58518"}
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for offset, source in ((-.18, "Per-system α"), (.18, "LOSO α")):
        block = data[data.source == source]
        values = [block.loc[block.model == metric, "alpha"].dropna().to_numpy()
                  for metric in THERMODYNAMIC_METRICS]
        boxes = axis.boxplot(values, positions=positions + offset, widths=.30,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.4})
        for box in boxes["boxes"]:
            box.set_facecolor(colors[source]); box.set_alpha(.55)
        axis.scatter([], [], marker="s", s=80, color=colors[source], alpha=.55, label=source)
    axis.set_xticks(positions, [THERMO_LABELS[metric] for metric in THERMODYNAMIC_METRICS],
                    rotation=20, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel(r"Fitted coefficient $\alpha$ (log scale)")
    axis.set_title("Thermodynamic population-metric scale coefficients\nKDE neighbour rank 10")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(OUTPUT / "thermodynamic_alpha_distributions.png", dpi=180); plt.close(fig)


def main() -> None:
    summary = pd.read_parquet(OUTPUT / "thermodynamic_population_assignment_summary.parquet")
    fits = pd.read_parquet(OUTPUT / "thermodynamic_population_fits.parquet")
    global_fits = pd.read_parquet(OUTPUT / "thermodynamic_population_global_scales.parquet")
    system, population = system_then_population(summary)
    baseline = pd.read_parquet(BASELINE_OUTPUT / "kde_population_population_summary.parquet")
    system.to_parquet(OUTPUT / "thermodynamic_population_system_summary.parquet", index=False)
    population.to_parquet(OUTPUT / "thermodynamic_population_population_summary.parquet", index=False)
    plot_headline(population, baseline); plot_entropy_variants(population)
    plot_scale_diagnostic(system); plot_coverage(population); plot_alpha_distributions(fits, global_fits)
    headline = selected(population)
    models = [f"{metric}_local_alpha" for metric in THERMODYNAMIC_METRICS]
    table = headline[headline.model.isin(models)].pivot(index="model", columns="band", values="distribution_recovery") * 100
    table.round(2).to_csv(OUTPUT / "thermodynamic_recovery_percent.csv")
    sensitivity = (population[(population.structural_axis == "w1")
                              & (population.stratum == "common_support")
                              & population.model.isin(models)]
                   .groupby(["rank", "model"], as_index=False).distribution_recovery.mean())
    sensitivity.to_parquet(OUTPUT / "thermodynamic_bandwidth_sensitivity.parquet", index=False)
    atomic_yaml(OUTPUT / "checkpoint18_report.yaml", {
        "checkpoint": 18, "status": "report_complete", "recovery_definition": "100*(1-sqrt(JSD))",
        "entropy_variants": ["legacy Zq", "unnormalized q", "normalized q/Z"],
    })
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
