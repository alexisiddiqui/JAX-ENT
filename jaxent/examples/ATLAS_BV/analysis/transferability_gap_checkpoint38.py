"""Checkpoint 38: quantify and explain the global-alpha transferability gap.

This analysis never refits checkpoint 36 or 37. It pairs their held-out replica-C
results, derives structural descriptors from the same post-equilibration trajectories,
and tests whether cohort-atypical systems have larger alpha mismatches and recovery gaps.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-cp38-matplotlib")

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import TheilSenRegressor
import yaml

from jaxent.examples.ATLAS_BV.analysis import corrected_variance_recovery_checkpoint36 as cp36
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    load_config,
    post_equilibration_indices,
    replica_paths,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


CP36 = cp36.OUTPUT / "pilot"
CP37 = HERE / "outputs/analysis/pairwise_geometry/checkpoint37_global_alpha"
OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint38_transferability_gap"
KEYS = ["system_id", "metric", "model", "band"]
PRIMARY_METRICS = (*cp36.CLEAN_WORK, *cp36.DERIVED_WORK, *cp36.PF_METRICS, *cp36.COORDINATES)
DESCRIPTOR_COLUMNS = (
    "length_atypicality",
    "compactness_atypicality",
    "secondary_structure_atypicality",
    "heterogeneity_atypicality",
    "rg_cv",
    "ss_temporal_js_mean",
)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def paired_gap_cells(per_system: pd.DataFrame, global_alpha: pd.DataFrame) -> pd.DataFrame:
    """Pair held-out results exactly and calculate signed/absolute recovery gaps."""
    required = {*KEYS, "pairs", "distribution_recovery"}
    for name, frame in (("per-system", per_system), ("global", global_alpha)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} results missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} results contain duplicate result keys")
    old_keys = pd.MultiIndex.from_frame(per_system[KEYS])
    new_keys = pd.MultiIndex.from_frame(global_alpha[KEYS])
    if set(old_keys) != set(new_keys):
        raise ValueError("checkpoint 36/37 result keys do not match")
    cells = per_system[KEYS + ["pairs", "distribution_recovery"]].merge(
        global_alpha[KEYS + ["pairs", "distribution_recovery"]],
        on=KEYS,
        suffixes=("_system", "_global"),
        validate="one_to_one",
    )
    if not np.array_equal(cells.pairs_system, cells.pairs_global):
        raise ValueError("checkpoint 36/37 pair counts differ")
    cells["pairs"] = cells.pop("pairs_global")
    cells = cells.drop(columns="pairs_system")
    cells["signed_gap"] = (
        cells.distribution_recovery_global - cells.distribution_recovery_system
    )
    cells["absolute_gap"] = cells.signed_gap.abs()
    return cells


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(), weights=weights.to_numpy()))


def aggregate_system_predictor_gaps(cells: pd.DataFrame) -> pd.DataFrame:
    """Collapse eligible bands equally, retaining planned sensitivity summaries."""
    rows = []
    for (system, metric, model), block in cells.groupby(
        ["system_id", "metric", "model"], sort=False
    ):
        common = block[block.band.isin(("q2", "q3"))]
        has_common = set(common.band) == {"q2", "q3"}
        rows.append(
            {
                "system_id": system,
                "metric": metric,
                "model": model,
                "absolute_gap": float(block.absolute_gap.mean()),
                "signed_gap": float(block.signed_gap.mean()),
                "absolute_gap_pair_weighted": weighted_mean(block.absolute_gap, block.pairs),
                "signed_gap_pair_weighted": weighted_mean(block.signed_gap, block.pairs),
                "absolute_gap_q23": float(common.absolute_gap.mean()) if has_common else np.nan,
                "signed_gap_q23": float(common.signed_gap.mean()) if has_common else np.nan,
                "bands": int(block.band.nunique()),
                "pairs": int(block.pairs.sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    block: pd.DataFrame, column: str, samples: int, seed: int
) -> tuple[float, float]:
    by_system = block.set_index("system_id")[column]
    systems = by_system.index.to_numpy()
    values = by_system.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(systems), size=(samples, len(systems)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_predictor_gaps(
    gaps: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    rows = []
    for (metric, model), block in gaps.groupby(["metric", "model"], sort=False):
        low, high = bootstrap_mean_interval(
            block, "absolute_gap", bootstrap_samples, stable_seed(seed, metric, model)
        )
        rows.append(
            {
                "metric": metric,
                "model": model,
                "mean_absolute_gap": float(block.absolute_gap.mean()),
                "absolute_gap_ci_low": low,
                "absolute_gap_ci_high": high,
                "mean_signed_gap": float(block.signed_gap.mean()),
                "median_absolute_gap": float(block.absolute_gap.median()),
                "mean_pair_weighted_gap": float(block.absolute_gap_pair_weighted.mean()),
                "systems": int(block.system_id.nunique()),
                "mean_bands": float(block.bands.mean()),
                "primary": bool(model == "direct" and metric in PRIMARY_METRICS),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_absolute_gap", ascending=False)


def descriptor_identity(row: dict[str, str], config: dict) -> dict:
    markers = []
    for replica in (1, 2, 3):
        marker = cp36.feature_folder(cp36.OUTPUT, row["system_id"], replica) / "complete.json"
        payload = json.loads(marker.read_text())
        markers.append(payload["identity"]["inputs"])
    return {
        "system_id": row["system_id"],
        "length": int(row["length"]),
        "equilibration_ns": config["analysis"]["equilibration_ns"],
        "frame_interval_ns": config["analysis"]["frame_interval_ns"],
        "inputs": markers,
        "analysis_sha256": cp36.digest(Path(__file__)),
    }


def trajectory_descriptor(row: dict[str, str], config: dict) -> dict:
    """Calculate post-equilibration C-alpha Rg and trajectory DSSP summaries."""
    rg_parts = []
    composition_parts = []
    pdb = HERE / row["pdb_path"]
    for replica, trajectory in enumerate(replica_paths(row), 1):
        universe = mda.Universe(str(pdb), str(trajectory))
        ca = universe.select_atoms("protein and name CA")
        if len(ca) != int(row["length"]):
            raise ValueError(
                f"{row['system_id']} R{replica}: expected {row['length']} CAs, found {len(ca)}"
            )
        keep = post_equilibration_indices(
            universe.trajectory.n_frames,
            config["analysis"]["equilibration_ns"],
            config["analysis"]["frame_interval_ns"],
        )
        rg = np.empty(len(keep), dtype=float)
        for index, frame in enumerate(keep):
            universe.trajectory[int(frame)]
            positions = ca.positions.astype(float, copy=False)
            centered = positions - positions.mean(axis=0)
            rg[index] = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
        assignment = np.asarray(DSSP(universe).run(frames=keep).results.dssp)
        if assignment.shape != (len(keep), int(row["length"])):
            raise ValueError(
                f"{row['system_id']} R{replica}: unexpected DSSP shape {assignment.shape}"
            )
        composition = np.column_stack(
            [(assignment == state).mean(axis=1) for state in ("H", "E", "-")]
        )
        if not np.allclose(composition.sum(axis=1), 1.0):
            raise ValueError(f"{row['system_id']} R{replica}: invalid DSSP fractions")
        rg_parts.append(rg)
        composition_parts.append(composition)
    rg = np.concatenate(rg_parts)
    composition = np.concatenate(composition_parts)
    mean_composition = composition.mean(axis=0)
    temporal_js = np.asarray(
        [jensenshannon(values, mean_composition, base=2.0) for values in composition]
    )
    q25, q75 = np.quantile(rg, [0.25, 0.75])
    return {
        "system_id": row["system_id"],
        "n_residues": int(row["length"]),
        "frames": int(len(rg)),
        "rg_mean": float(rg.mean()),
        "rg_median": float(np.median(rg)),
        "rg_sd": float(rg.std(ddof=1)),
        "rg_iqr": float(q75 - q25),
        "rg_cv": float(rg.std(ddof=1) / rg.mean()),
        "helix_fraction": float(mean_composition[0]),
        "sheet_fraction": float(mean_composition[1]),
        "coil_fraction": float(mean_composition[2]),
        "helix_fraction_sd": float(composition[:, 0].std(ddof=1)),
        "sheet_fraction_sd": float(composition[:, 1].std(ddof=1)),
        "coil_fraction_sd": float(composition[:, 2].std(ddof=1)),
        "ss_temporal_js_mean": float(temporal_js.mean()),
        "ss_temporal_js_sd": float(temporal_js.std(ddof=1)),
    }


def descriptor_worker(payload):
    row, config = payload
    return descriptor_identity(row, config), trajectory_descriptor(row, config)


def load_or_build_descriptors(
    rows: list[dict[str, str]],
    config: dict,
    destination: Path,
    workers: int,
    force: bool,
) -> pd.DataFrame:
    parts = destination / "descriptor_parts"
    parts.mkdir(parents=True, exist_ok=True)
    records = []
    pending = []
    for row in rows:
        path = parts / f"{row['system_id']}.json"
        identity = descriptor_identity(row, config)
        if path.exists() and not force:
            cached = json.loads(path.read_text())
            if cached.get("identity") == identity:
                records.append(cached["descriptor"])
                continue
        pending.append(row)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(descriptor_worker, (row, config)): row for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            identity, descriptor = future.result()
            path = parts / f"{descriptor['system_id']}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2)
                + "\n"
            )
            temporary.replace(path)
            records.append(descriptor)
            print(
                f"descriptors [{index}/{len(pending)}] {descriptor['system_id']}",
                flush=True,
            )
    result = pd.DataFrame(records).sort_values("system_id").reset_index(drop=True)
    if len(result) != len(rows) or result.system_id.nunique() != len(rows):
        raise ValueError("descriptor cache did not yield one row per system")
    return result


def rms_scale(values: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.square(values))))
    return scale if scale > np.finfo(float).eps else 1.0


def add_cohort_atypicality(descriptors: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = descriptors.copy()
    result["log_length"] = np.log(result.n_residues)
    result["log_rg_mean"] = np.log(result.rg_mean)
    regression = TheilSenRegressor(random_state=seed).fit(
        result[["log_length"]], result.log_rg_mean
    )
    result["compactness_residual"] = (
        result.log_rg_mean - regression.predict(result[["log_length"]])
    )
    log_length_centered = result.log_length - result.log_length.median()
    compactness_centered = (
        result.compactness_residual - result.compactness_residual.median()
    )
    result["length_atypicality"] = (
        log_length_centered.abs() / rms_scale(log_length_centered.to_numpy())
    )
    result["compactness_atypicality"] = (
        compactness_centered.abs() / rms_scale(compactness_centered.to_numpy())
    )
    composition = result[["helix_fraction", "sheet_fraction", "coil_fraction"]].to_numpy()
    centroid = composition.mean(axis=0)
    ss_distance = np.asarray(
        [jensenshannon(row, centroid, base=2.0) for row in composition]
    )
    result["secondary_structure_js"] = ss_distance
    result["secondary_structure_atypicality"] = ss_distance / rms_scale(ss_distance)
    result["heterogeneity_atypicality"] = np.sqrt(
        (
            np.square(result.length_atypicality)
            + np.square(result.compactness_atypicality)
            + np.square(result.secondary_structure_atypicality)
        )
        / 3.0
    )
    return result


def attach_alpha_mismatch(
    gaps: pd.DataFrame, per_system_fits: pd.DataFrame, global_parameters: pd.DataFrame
) -> pd.DataFrame:
    old = per_system_fits[["system_id", "metric", "model", "alpha"]].rename(
        columns={"alpha": "system_alpha"}
    )
    new = global_parameters[["metric", "model", "global_alpha"]]
    result = gaps.merge(old, on=["system_id", "metric", "model"], validate="one_to_one")
    result = result.merge(new, on=["metric", "model"], validate="many_to_one")
    if (result[["system_alpha", "global_alpha"]] <= 0).any().any():
        raise ValueError("alpha mismatch requires strictly positive alphas")
    result["alpha_mismatch"] = np.abs(
        np.log(result.system_alpha / result.global_alpha)
    )
    result["primary"] = result.model.eq("direct") & result.metric.isin(PRIMARY_METRICS)
    return result


def within_predictor_standardize(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = np.empty(len(frame), dtype=float)
    for _, positions in frame.groupby(["metric", "model"], sort=False).indices.items():
        positions = np.asarray(positions)
        block = frame.iloc[positions][column].to_numpy(dtype=float)
        scale = block.std(ddof=1)
        values[positions] = (block - block.mean()) / (scale if scale > 0 else 1.0)
    return values


def design_matrix(
    frame: pd.DataFrame, include_mismatch: bool, gap_column: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = frame.reset_index(drop=True).copy()
    ordered["gap_z"] = within_predictor_standardize(ordered, gap_column)
    ordered["mismatch_z"] = within_predictor_standardize(ordered, "alpha_mismatch")
    heterogeneity = ordered.heterogeneity_atypicality.to_numpy(dtype=float)
    heterogeneity = (heterogeneity - heterogeneity.mean()) / heterogeneity.std(ddof=1)
    band_count = ordered.bands.to_numpy(dtype=float)
    band_count = (band_count - band_count.mean()) / band_count.std(ddof=1)
    predictors = ordered.metric + ":" + ordered.model
    dummy = pd.get_dummies(predictors, drop_first=True, dtype=float)
    columns = [np.ones(len(ordered)), heterogeneity]
    names = ["intercept", "heterogeneity_atypicality"]
    if include_mismatch:
        columns.append(ordered.mismatch_z.to_numpy())
        names.append("alpha_mismatch")
    columns.append(band_count)
    names.append("band_count")
    columns.extend(dummy[column].to_numpy() for column in dummy)
    names.extend(f"predictor[{column}]" for column in dummy)
    return np.column_stack(columns), ordered.gap_z.to_numpy(), names


def alpha_design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = frame.reset_index(drop=True).copy()
    ordered["mismatch_z"] = within_predictor_standardize(ordered, "alpha_mismatch")
    heterogeneity = ordered.heterogeneity_atypicality.to_numpy(dtype=float)
    heterogeneity = (heterogeneity - heterogeneity.mean()) / heterogeneity.std(ddof=1)
    predictors = ordered.metric + ":" + ordered.model
    dummy = pd.get_dummies(predictors, drop_first=True, dtype=float)
    columns = [np.ones(len(ordered)), heterogeneity]
    names = ["intercept", "heterogeneity_atypicality"]
    columns.extend(dummy[column].to_numpy() for column in dummy)
    names.extend(f"predictor[{column}]" for column in dummy)
    return np.column_stack(columns), ordered.mismatch_z.to_numpy(), names


def coefficient(x: np.ndarray, y: np.ndarray, index: int) -> float:
    return float(np.linalg.lstsq(x, y, rcond=None)[0][index])


def resampled_rows(frame: pd.DataFrame, sampled_systems: np.ndarray) -> np.ndarray:
    groups = {system: np.flatnonzero(frame.system_id.to_numpy() == system) for system in frame.system_id.unique()}
    return np.concatenate([groups[system] for system in sampled_systems])


def infer_coefficient(
    frame: pd.DataFrame,
    matrix_builder,
    term: str,
    bootstrap_samples: int,
    permutations: int,
    seed: int,
    permute_column: str,
) -> dict:
    frame = frame.reset_index(drop=True)
    x, y, names = matrix_builder(frame)
    index = names.index(term)
    estimate = coefficient(x, y, index)
    systems = frame.system_id.unique()
    system_rows = {
        system: np.flatnonzero(frame.system_id.to_numpy() == system)
        for system in systems
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples)
    for draw in range(bootstrap_samples):
        sampled = rng.choice(systems, size=len(systems), replace=True)
        rows = resampled_rows(frame, sampled)
        boot[draw] = coefficient(x[rows], y[rows], index)
    low, high = np.quantile(boot, [0.025, 0.975])
    null = np.empty(permutations)
    if permute_column == "alpha_mismatch":
        # The standardized mismatch column varies by predictor; permute whole
        # system blocks while preserving predictor alignment within each block.
        block_values = {}
        for system in systems:
            rows = system_rows[system]
            keys = list(
                zip(frame.iloc[rows].metric, frame.iloc[rows].model, strict=True)
            )
            block_values[system] = dict(zip(keys, x[rows, index], strict=True))
        for draw in range(permutations):
            shuffled = rng.permutation(systems)
            xp = x.copy()
            for destination_system, source_system in zip(systems, shuffled, strict=True):
                rows = system_rows[destination_system]
                keys = zip(frame.iloc[rows].metric, frame.iloc[rows].model, strict=True)
                xp[rows, index] = [block_values[source_system][key] for key in keys]
            null[draw] = coefficient(xp, y, index)
    else:
        values = {
            system: x[system_rows[system][0], index]
            for system in systems
        }
        for draw in range(permutations):
            shuffled = rng.permutation(systems)
            xp = x.copy()
            for destination_system, source_system in zip(systems, shuffled, strict=True):
                rows = system_rows[destination_system]
                xp[rows, index] = values[source_system]
            null[draw] = coefficient(xp, y, index)
    p_value = float((1 + np.sum(np.abs(null) >= abs(estimate))) / (permutations + 1))
    return {
        "term": term,
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "permutation_p": p_value,
        "bootstrap_samples": bootstrap_samples,
        "permutations": permutations,
    }


def primary_mechanism_models(
    analysis: pd.DataFrame, bootstrap_samples: int, permutations: int, seed: int
) -> pd.DataFrame:
    primary = analysis[analysis.primary].reset_index(drop=True)
    specifications = [
        (
            "A_atypicality_to_alpha_mismatch",
            alpha_design_matrix,
            "heterogeneity_atypicality",
            "heterogeneity_atypicality",
        ),
        (
            "B_atypicality_to_recovery_gap",
            lambda frame: design_matrix(frame, False, "absolute_gap"),
            "heterogeneity_atypicality",
            "heterogeneity_atypicality",
        ),
        (
            "C_joint_gap_model_heterogeneity",
            lambda frame: design_matrix(frame, True, "absolute_gap"),
            "heterogeneity_atypicality",
            "heterogeneity_atypicality",
        ),
        (
            "C_joint_gap_model_alpha_mismatch",
            lambda frame: design_matrix(frame, True, "absolute_gap"),
            "alpha_mismatch",
            "alpha_mismatch",
        ),
    ]
    rows = []
    for name, builder, term, permute in specifications:
        result = infer_coefficient(
            primary,
            builder,
            term,
            bootstrap_samples,
            permutations,
            stable_seed(seed, name),
            permute,
        )
        result["model"] = name
        rows.append(result)
    table = pd.DataFrame(rows)
    beta_b = table.loc[
        table.model == "B_atypicality_to_recovery_gap", "estimate"
    ].iloc[0]
    beta_c = table.loc[
        table.model == "C_joint_gap_model_heterogeneity", "estimate"
    ].iloc[0]
    table["heterogeneity_attenuation_fraction"] = np.nan
    table.loc[
        table.model == "C_joint_gap_model_heterogeneity",
        "heterogeneity_attenuation_fraction",
    ] = (beta_b - beta_c) / beta_b if beta_b else np.nan
    return table


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def exploratory_correlations(analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, model), block in analysis.groupby(["metric", "model"]):
        for response in ("absolute_gap", "alpha_mismatch"):
            for descriptor in DESCRIPTOR_COLUMNS:
                rho, p_value = spearmanr(block[response], block[descriptor])
                rows.append(
                    {
                        "metric": metric,
                        "model": model,
                        "response": response,
                        "descriptor": descriptor,
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                        "systems": int(block.system_id.nunique()),
                        "primary": bool(block.primary.all()),
                    }
                )
    table = pd.DataFrame(rows)
    table["fdr_q"] = np.nan
    for (_, _), positions in table.groupby(["response", "primary"]).indices.items():
        table.loc[positions, "fdr_q"] = bh_adjust(table.loc[positions, "p_value"].to_numpy())
    return table


def sensitivity_correlations(analysis: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "primary_equal_band": (analysis.primary, "absolute_gap"),
        "primary_pair_weighted": (analysis.primary, "absolute_gap_pair_weighted"),
        "primary_common_q23": (analysis.primary & analysis.absolute_gap_q23.notna(), "absolute_gap_q23"),
        "work_pf_direct": (
            analysis.model.eq("direct")
            & analysis.metric.isin((*cp36.CLEAN_WORK, *cp36.DERIVED_WORK, *cp36.PF_METRICS)),
            "absolute_gap",
        ),
        "primary_without_w1": (analysis.primary & analysis.metric.ne("w1"), "absolute_gap"),
        "energy_direct": (analysis.model.eq("direct") & analysis.metric.isin(cp36.ENERGY_METRICS), "absolute_gap"),
        "variance_magnitude": (analysis.model.eq("variance_magnitude"), "absolute_gap"),
        "local_dispersion": (analysis.model.eq("local_dispersion"), "absolute_gap"),
    }
    rows = []
    for scope, (mask, column) in scopes.items():
        block = analysis[mask]
        system = block.groupby("system_id").agg(
            gap=(column, "mean"),
            alpha_mismatch=("alpha_mismatch", "mean"),
            heterogeneity=("heterogeneity_atypicality", "first"),
        )
        for response in ("gap", "alpha_mismatch"):
            rho, p_value = spearmanr(system[response], system.heterogeneity)
            rows.append(
                {
                    "scope": scope,
                    "response": response,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "systems": len(system),
                    "predictors": int(block[["metric", "model"]].drop_duplicates().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def system_primary_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    primary = analysis[analysis.primary]
    columns = ["system_id", *DESCRIPTOR_COLUMNS, "n_residues", "rg_mean", "helix_fraction", "sheet_fraction", "coil_fraction"]
    descriptors = primary[columns].drop_duplicates("system_id")
    aggregate = primary.groupby("system_id").agg(
        mean_absolute_gap=("absolute_gap", "mean"),
        mean_signed_gap=("signed_gap", "mean"),
        mean_alpha_mismatch=("alpha_mismatch", "mean"),
        mean_bands=("bands", "mean"),
    ).reset_index()
    return aggregate.merge(descriptors, on="system_id", validate="one_to_one")


def plot_predictor_gaps(summary: pd.DataFrame, destination: Path) -> None:
    ordered = summary.sort_values("mean_absolute_gap")
    y = np.arange(len(ordered))
    colors = np.where(ordered.primary, "#4C78A8", "#B9B9B9")
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True, constrained_layout=True)
    axes[0].barh(y, ordered.mean_absolute_gap, color=colors)
    axes[0].errorbar(
        ordered.mean_absolute_gap,
        y,
        xerr=[
            ordered.mean_absolute_gap - ordered.absolute_gap_ci_low,
            ordered.absolute_gap_ci_high - ordered.mean_absolute_gap,
        ],
        fmt="none",
        ecolor="black",
        capsize=2,
    )
    axes[0].set_xlabel("Mean |global − per-system recovery|")
    axes[0].set_title("Absolute transferability gap")
    axes[1].barh(y, ordered.mean_signed_gap, color=np.where(ordered.mean_signed_gap >= 0, "#59A14F", "#E15759"))
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Mean global − per-system recovery")
    axes[1].set_title("Signed change")
    labels = ordered.metric + ":" + ordered.model.replace(
        {"variance_magnitude": "VM", "local_dispersion": "disp"}
    )
    axes[0].set_yticks(y, labels, fontsize=8)
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("ATLAS BV checkpoint 38: global-alpha transferability gap")
    fig.savefig(destination / "transferability_gap_by_predictor.png", dpi=180)
    plt.close(fig)


def add_fit_line(axis, x: pd.Series, y: pd.Series) -> None:
    coefficient_values = np.polyfit(x, y, 1)
    domain = np.linspace(x.min(), x.max(), 100)
    axis.plot(domain, np.polyval(coefficient_values, domain), color="black", linestyle="--")


def plot_mechanism_scatter(system: pd.DataFrame, destination: Path) -> None:
    panels = (
        ("heterogeneity_atypicality", "mean_alpha_mismatch", "Atypicality → alpha mismatch"),
        ("heterogeneity_atypicality", "mean_absolute_gap", "Atypicality → recovery gap"),
        ("mean_alpha_mismatch", "mean_absolute_gap", "Alpha mismatch → recovery gap"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), constrained_layout=True)
    for axis, (x_name, y_name, title) in zip(axes, panels, strict=True):
        axis.scatter(system[x_name], system[y_name], c=system.n_residues, cmap="viridis", s=48)
        add_fit_line(axis, system[x_name], system[y_name])
        for row in system.itertuples():
            axis.annotate(row.system_id, (getattr(row, x_name), getattr(row, y_name)), fontsize=6, alpha=0.75)
        rho, p_value = spearmanr(system[x_name], system[y_name])
        axis.set_title(f"{title}\nSpearman ρ={rho:.2f}, p={p_value:.3g}")
        axis.set_xlabel(x_name.replace("_", " "))
        axis.set_ylabel(y_name.replace("_", " "))
        axis.grid(alpha=0.2)
    fig.savefig(destination / "alpha_mismatch_mechanism.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(correlations: pd.DataFrame, destination: Path) -> None:
    block = correlations[
        (correlations.response == "absolute_gap") & correlations.primary
    ].copy()
    block["predictor"] = block.metric + ":" + block.model
    matrix = block.pivot(index="predictor", columns="descriptor", values="spearman_rho")
    matrix = matrix.reindex(columns=DESCRIPTOR_COLUMNS)
    fig, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), [name.replace("_", " ") for name in matrix.columns], rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    for row in range(len(matrix)):
        for column in range(len(matrix.columns)):
            axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title("Per-predictor Spearman correlation: transfer gap vs heterogeneity")
    fig.colorbar(image, ax=axis, label="Spearman ρ")
    fig.savefig(destination / "predictor_heterogeneity_correlations.png", dpi=180)
    plt.close(fig)


def plot_system_summary(system: pd.DataFrame, destination: Path) -> None:
    ordered = system.sort_values("heterogeneity_atypicality")
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    axes[0].bar(x, ordered.heterogeneity_atypicality, color="#F28E2B")
    axes[0].set_ylabel("Cohort atypicality")
    axes[1].plot(x, ordered.mean_absolute_gap, marker="o", label="Recovery gap")
    scaled = ordered.mean_alpha_mismatch / ordered.mean_alpha_mismatch.max() * ordered.mean_absolute_gap.max()
    axes[1].plot(x, scaled, marker="s", label="Alpha mismatch (scaled)")
    axes[1].set_ylabel("Mean primary-predictor gap")
    axes[1].set_xticks(x, ordered.system_id, rotation=55, ha="right")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("System atypicality and global-alpha transfer burden")
    fig.savefig(destination / "system_gap_and_atypicality.png", dpi=180)
    plt.close(fig)


def pythonify(value):
    if isinstance(value, dict):
        return {key: pythonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pythonify(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--force-descriptors", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.workers, args.bootstrap_samples, args.permutations) < 1:
        parser.error("workers, bootstrap samples, and permutations must be positive")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    per_system_results = pd.read_parquet(CP36 / "corrected_variance_results.parquet")
    global_results = pd.read_parquet(CP37 / "global_alpha_results.parquet")
    cells = paired_gap_cells(per_system_results, global_results)
    gaps = aggregate_system_predictor_gaps(cells)
    per_system_fits = pd.read_parquet(CP36 / "corrected_variance_fits.parquet")
    global_parameters = pd.read_csv(CP37 / "global_alpha_parameters.csv")
    gaps = attach_alpha_mismatch(gaps, per_system_fits, global_parameters)

    rows = cp36.selected_rows("pilot", cp36.SINGLE_SYSTEM)
    config = load_config()
    descriptors = load_or_build_descriptors(
        rows, config, destination, args.workers, args.force_descriptors
    )
    descriptors = add_cohort_atypicality(descriptors, args.seed)
    analysis = gaps.merge(descriptors, on="system_id", validate="many_to_one")

    predictor_summary = summarize_predictor_gaps(
        analysis, args.bootstrap_samples, args.seed
    )
    mechanism = primary_mechanism_models(
        analysis, args.bootstrap_samples, args.permutations, args.seed
    )
    correlations = exploratory_correlations(analysis)
    sensitivity = sensitivity_correlations(analysis)
    system_summary = system_primary_summary(analysis)

    atomic_parquet(cells, destination / "transfer_gap_cells.parquet")
    atomic_parquet(analysis, destination / "system_predictor_gaps.parquet")
    atomic_parquet(descriptors, destination / "system_descriptors.parquet")
    predictor_summary.to_csv(destination / "predictor_gap_summary.csv", index=False)
    mechanism.to_csv(destination / "mechanism_models.csv", index=False)
    correlations.to_csv(destination / "predictor_correlations.csv", index=False)
    sensitivity.to_csv(destination / "sensitivity_correlations.csv", index=False)
    system_summary.to_csv(destination / "system_primary_summary.csv", index=False)

    plot_predictor_gaps(predictor_summary, destination)
    plot_mechanism_scatter(system_summary, destination)
    plot_correlation_heatmap(correlations, destination)
    plot_system_summary(system_summary, destination)

    model_lookup = mechanism.set_index("model")
    beta_a = model_lookup.loc["A_atypicality_to_alpha_mismatch"]
    beta_b = model_lookup.loc["B_atypicality_to_recovery_gap"]
    beta_c_h = model_lookup.loc["C_joint_gap_model_heterogeneity"]
    beta_c_m = model_lookup.loc["C_joint_gap_model_alpha_mismatch"]
    attenuation = beta_c_h.heterogeneity_attenuation_fraction
    evidence = bool(
        beta_a.estimate > 0
        and beta_a.permutation_p < 0.05
        and beta_b.estimate > 0
        and beta_b.permutation_p < 0.05
        and beta_c_m.estimate > 0
        and beta_c_m.permutation_p < 0.05
        and attenuation > 0
    )
    report = {
        "checkpoint": 38,
        "systems": int(analysis.system_id.nunique()),
        "predictor_models": int(analysis[["metric", "model"]].drop_duplicates().shape[0]),
        "primary_predictors": list(PRIMARY_METRICS),
        "paired_result_cells": len(cells),
        "system_predictor_rows": len(analysis),
        "gap_definition": "within-system equal mean over eligible bands of abs(global recovery - per-system recovery)",
        "heterogeneity_definition": "RMS of normalized log-length, length-adjusted Rg, and DSSP composition distances from the 24-system cohort",
        "mechanism_models": mechanism.to_dict(orient="records"),
        "mechanism_pattern_supported": evidence,
        "interpretation": (
            "Observational evidence consistent with alpha absorbing system size/shape heterogeneity; "
            "the checkpoint cannot establish a causal mechanism."
            if evidence
            else "The pre-specified joint evidence pattern was not met; individual associations remain descriptive."
        ),
        "provenance": {
            "checkpoint36_results_sha256": cp36.digest(CP36 / "corrected_variance_results.parquet"),
            "checkpoint37_results_sha256": cp36.digest(CP37 / "global_alpha_results.parquet"),
            "analysis_sha256": cp36.digest(Path(__file__)),
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "permutations": args.permutations,
        },
    }
    path = destination / "checkpoint38_report.yaml"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(pythonify(report), sort_keys=False))
    temporary.replace(path)
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    main()
