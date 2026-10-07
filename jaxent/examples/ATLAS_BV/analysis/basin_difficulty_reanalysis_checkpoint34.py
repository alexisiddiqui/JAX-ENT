"""Checkpoint 34: reanalyse basin recovery conditional on MaxEnt difficulty."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_systems
from jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 import (
    LOCKED_CANDIDATE,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import stable_seed
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)


CP32 = HERE / "outputs/analysis/pairwise_geometry/checkpoint32_laplacian_topology"
OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint34_basin_difficulty"
DIFFICULTY_LABELS = ("D1", "D2", "D3", "D4")
SIZE_LABELS = ("Q1", "Q2", "Q3", "Q4")


def load_existing_basin_results() -> pd.DataFrame:
    sources = (
        (CP32 / "development/heldout_reweighting.parquet", "development_24"),
        (CP32 / "confirmation/confirmation_reweighting.parquet", "confirmation_87"),
    )
    blocks = []
    for path, cohort in sources:
        frame = pd.read_parquet(path)
        frame = frame[
            (frame.replica == 3)
            & (frame.candidate == LOCKED_CANDIDATE)
            & (frame.bias_kind == "basin")
        ].copy()
        frame["cohort"] = cohort
        blocks.append(frame)
    combined = pd.concat(blocks, ignore_index=True)
    pivot = combined.pivot_table(
        index=[
            "system_id",
            "cohort",
            "bias_level",
            "target_basin_mass",
            "achieved_prior_ess_fraction",
        ],
        columns="arm",
        values=["weight_tv", "cluster_population_mae", "test_mse"],
    ).reset_index()
    pivot.columns = [
        "_".join(str(value) for value in column if str(value))
        for column in pivot.columns
    ]
    pivot = pivot.rename(
        columns={
            "weight_tv_maxent": "maxent_tv",
            "weight_tv_laplacian": "laplacian_tv",
            "cluster_population_mae_maxent": "maxent_population_mae",
            "cluster_population_mae_laplacian": "laplacian_population_mae",
            "test_mse_maxent": "maxent_test_mse",
            "test_mse_laplacian": "laplacian_test_mse",
        }
    )
    pivot["tv_gain"] = pivot.maxent_tv - pivot.laplacian_tv
    pivot["fractional_tv_recovery"] = pivot.tv_gain / np.maximum(
        pivot.maxent_tv, np.finfo(float).eps
    )
    pivot["population_mae_gain"] = (
        pivot.maxent_population_mae - pivot.laplacian_population_mae
    )
    metadata = pd.DataFrame(load_systems())[["system_id", "length"]]
    metadata["n_residues"] = pd.to_numeric(metadata.pop("length"), errors="raise")
    pivot = pivot.merge(metadata, on="system_id", validate="many_to_one")
    return add_bins(pivot)


def add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["difficulty_bin"] = pd.qcut(
        result.maxent_tv,
        4,
        labels=DIFFICULTY_LABELS,
    )
    lengths = result[["system_id", "n_residues"]].drop_duplicates()
    lengths["size_quartile"] = pd.qcut(
        lengths.n_residues,
        4,
        labels=SIZE_LABELS,
    )
    return result.merge(lengths, on=["system_id", "n_residues"], validate="many_to_one")


def clustered_interval(
    frame: pd.DataFrame, column: str, *seed_parts: object
) -> tuple[float, float]:
    systems = frame.system_id.unique()
    rng = np.random.default_rng(stable_seed("checkpoint34", *seed_parts))
    draws = np.empty(10_000)
    by_system = {name: values[column].to_numpy() for name, values in frame.groupby("system_id")}
    for index in range(len(draws)):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        draws[index] = np.mean(np.concatenate([by_system[name] for name in sampled]))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in frame.groupby(groups, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(groups, key, strict=True))
        for column in ("maxent_tv", "tv_gain", "fractional_tv_recovery", "population_mae_gain"):
            low, high = clustered_interval(block, column, *key, column)
            record.update(
                {
                    f"{column}_mean": float(block[column].mean()),
                    f"{column}_median": float(block[column].median()),
                    f"{column}_ci_low": low,
                    f"{column}_ci_high": high,
                }
            )
        record["systems"] = block.system_id.nunique()
        record["challenges"] = len(block)
        record["positive_gain_fraction"] = float((block.tv_gain > 0).mean())
        rows.append(record)
    return pd.DataFrame(rows)


def bin_labels(frame: pd.DataFrame) -> dict[str, str]:
    return {
        str(name): f"{name}\n{block.maxent_tv.min():.3f}–{block.maxent_tv.max():.3f}"
        for name, block in frame.groupby("difficulty_bin", observed=True)
    }


def plot_binned_recovery(frame: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    labels = bin_labels(frame)
    ordered = summary.set_index("difficulty_bin").reindex(DIFFICULTY_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for axis, column, title in (
        (axes[0], "tv_gain", "Absolute recovery"),
        (axes[1], "fractional_tv_recovery", "Recovery relative to MaxEnt error"),
    ):
        mean = ordered[f"{column}_mean"]
        axis.errorbar(
            range(4),
            mean,
            yerr=[mean - ordered[f"{column}_ci_low"], ordered[f"{column}_ci_high"] - mean],
            marker="o",
            capsize=4,
            lw=2,
        )
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(range(4), [labels[name] for name in DIFFICULTY_LABELS])
        axis.set_xlabel("MaxEnt basin-error bin (TV range)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean TV gain")
    axes[1].set_ylabel("Mean fractional TV recovery")
    fig.suptitle("Basin recovery increases with recoverable MaxEnt error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_conditioned_length(interaction: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharex=True)
    for axis, column, title in (
        (axes[0], "tv_gain", "Absolute recovery"),
        (axes[1], "fractional_tv_recovery", "Fractional recovery"),
    ):
        for quartile, block in interaction.groupby("size_quartile", observed=True):
            values = block.set_index("difficulty_bin").reindex(DIFFICULTY_LABELS)
            mean = values[f"{column}_mean"]
            axis.errorbar(
                range(4),
                mean,
                yerr=[mean - values[f"{column}_ci_low"], values[f"{column}_ci_high"] - mean],
                marker="o",
                capsize=3,
                label=str(quartile),
            )
        axis.axhline(0, color="black", lw=1)
        axis.set_xticks(range(4), DIFFICULTY_LABELS)
        axis.set_xlabel("MaxEnt-error quartile")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean recovery")
    axes[1].legend(title="Length quartile")
    fig.suptitle("Protein length after conditioning on basin-challenge difficulty")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def regression_diagnostics(frame: pd.DataFrame) -> dict:
    length = (frame.n_residues - frame.n_residues.mean()) / frame.n_residues.std()
    difficulty = (frame.maxent_tv - frame.maxent_tv.mean()) / frame.maxent_tv.std()
    design = np.column_stack((np.ones(len(frame)), difficulty, length))
    coefficients = np.linalg.lstsq(design, frame.tv_gain, rcond=None)[0]
    rho_error, p_error = spearmanr(frame.maxent_tv, frame.tv_gain)
    rho_length, p_length = spearmanr(frame.n_residues, frame.tv_gain)
    rho_length_error, p_length_error = spearmanr(frame.n_residues, frame.maxent_tv)
    return {
        "observations": len(frame),
        "systems": frame.system_id.nunique(),
        "spearman_maxent_error_vs_gain": [float(rho_error), float(p_error)],
        "spearman_length_vs_gain": [float(rho_length), float(p_length)],
        "spearman_length_vs_maxent_error": [
            float(rho_length_error),
            float(p_length_error),
        ],
        "standardized_ols": {
            "intercept": float(coefficients[0]),
            "maxent_error_coefficient": float(coefficients[1]),
            "residue_count_coefficient": float(coefficients[2]),
        },
        "interpretation": (
            "descriptive post-hoc RBF reanalysis; no weights were refitted. "
            "Difficulty composition explains part, but not all, of the length trend."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_existing_basin_results()
    bins = summarize(rows, ["difficulty_bin"])
    interaction = summarize(rows, ["size_quartile", "difficulty_bin"])
    atomic_parquet(rows, OUTPUT / "basin_difficulty_rows.parquet")
    bins.to_csv(OUTPUT / "basin_difficulty_bins.csv", index=False)
    interaction.to_csv(OUTPUT / "basin_length_difficulty_interaction.csv", index=False)
    plot_binned_recovery(rows, bins, OUTPUT / "basin_recovery_by_maxent_error_bin.png")
    plot_conditioned_length(interaction, OUTPUT / "basin_recovery_conditioned_on_difficulty.png")
    atomic_yaml(OUTPUT / "checkpoint34_report.yaml", regression_diagnostics(rows))


if __name__ == "__main__":
    main()
