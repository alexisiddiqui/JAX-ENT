#!/usr/bin/env python3
"""Physics-first dual-track analysis of the real MoPrP HDX dataset.

Track A fits residue log protection factors with the source-specific EX2
forward model.  Track B compresses peptide curves with a shared-rate mixture.
The two parameterizations are reported side by side but are never equated:
mixture weights are not residue coordinates or protection factors.

Raw triplicate/control measurements are not present in the validation bundle,
so this runner reports multistart/profile sensitivity and curve errors only. It
does not report calibrated experimental covariance or use the production
``Sigma.npz`` curve construction as observation noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr

from jaxent.src.analysis.hdx_ex2 import (
    EX2SolutionSet,
    compare_trajectory_hdx,
    convolve_isotope_and_deuteron_distributions,
    fit_ex2_solution_set,
    load_expfact_dataset,
    load_intrinsic_rate_file,
    peptide_deuteron_count_distribution,
    thin_deuteron_count_distribution,
)
from jaxent.src.analysis.hdx_rate_mixture import (
    fit_shared_rate_mixture,
    rate_distribution_summaries,
    select_shared_rate_model,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[4]
BASE = PACKAGE_ROOT / "examples/2_CrossValidation"
MOPRP = BASE / "data/_MoPrP"
FEATURES = BASE / "fitting/jaxENT/_featurise"
BV_BC = 0.35
BV_BH = 2.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def _source_assignments(filename: str = "moprp.list") -> np.ndarray:
    rows = []
    for line in (MOPRP / filename).read_text().splitlines():
        fields = line.split()
        if fields:
            rows.append([int(fields[0]), int(fields[1]), int(fields[2])])
    return np.asarray(rows, dtype=int)


def _feature_bundle(
    name: str, feature_dir: Path = FEATURES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topology_path = feature_dir / f"topology_{name}.json"
    feature_path = feature_dir / f"features_{name}.npz"
    topology = json.loads(topology_path.read_text())["topologies"]
    residue_ids = np.asarray([item["residues"][0] for item in topology], dtype=int)
    with np.load(feature_path) as data:
        log_pf = BV_BC * np.asarray(data["heavy_contacts"], dtype=float) + BV_BH * np.asarray(
            data["acceptor_contacts"], dtype=float
        )
        stored_rates = np.asarray(data["k_ints"], dtype=float)
    if log_pf.shape[0] != len(residue_ids) or stored_rates.shape != (len(residue_ids),):
        raise ValueError(f"{name} feature and residue coordinates are not aligned")
    return residue_ids, log_pf, stored_rates


def _complete_trajectory_peptides(peptide_map, feature_residue_ids: np.ndarray):
    represented = set(feature_residue_ids.tolist())
    complete = []
    missing_by_peptide: dict[int, list[int]] = {}
    for row, peptide_id in enumerate(peptide_map.peptide_ids):
        active = set(peptide_map.residue_ids[peptide_map.matrix[row] > 0].tolist())
        missing = sorted(active - represented)
        missing_by_peptide[int(peptide_id)] = missing
        if not missing:
            complete.append(row)
    return np.asarray(complete, dtype=int), missing_by_peptide


def _trim_counts(sequence: str, assignments: np.ndarray, trim: int) -> np.ndarray:
    excluded = {1, *(index + 1 for index, code in enumerate(sequence) if code in {"P", "B"})}
    return np.asarray(
        [
            sum(residue not in excluded for residue in range(start + trim, end + 1))
            for _, start, end in assignments
        ],
        dtype=int,
    )


def _correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(pearsonr(x[valid], y[valid]).statistic),
        float(spearmanr(x[valid], y[valid]).statistic),
        float(np.sqrt(np.mean((x[valid] - y[valid]) ** 2))),
    )


def construction_audit(dataset, canonical_rates, feature_ids, stored_rates) -> pd.DataFrame:
    assignments = _source_assignments()
    clustering_assignments = np.loadtxt(MOPRP / "moprp.ass", dtype=int)
    rounded = np.asarray(
        [0.08, 0.33, 0.67, 1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 160.0, 240.0, 390.0, 750.0, 1440.0]
    )
    trim_one = dataset.peptide_map.active_amide_counts
    trim_two = _trim_counts(dataset.sequence, assignments, 2)
    _, missing = _complete_trajectory_peptides(dataset.peptide_map, feature_ids)
    canonical_feature_rates = canonical_rates.aligned(feature_ids)
    rate_mask = (canonical_feature_rates > 0) & (stored_rates > 0)
    ratios = stored_rates[rate_mask] / canonical_feature_rates[rate_mask]
    rows = [
        {
            "check": "exact_timepoints_replace_rounded_values",
            "value": float(np.max(np.abs(rounded - dataset.protocol.timepoints_min))),
            "details": f"max_abs_min; first exact={dataset.protocol.timepoints_min[0]:.6g}, rounded={rounded[0]:.6g}",
        },
        {
            "check": "trim_two_changes_active_amide_denominators",
            "value": int(np.count_nonzero(trim_one != trim_two)),
            "details": f"trim1={trim_one.tolist()}; trim2={trim_two.tolist()}",
        },
        {
            "check": "moprp_ass_differs_from_fitted_moprp_list",
            "value": int(np.count_nonzero(assignments[:, :3] != clustering_assignments[:, :3])),
            "details": "peptide 8 C terminus is 74 in fitted moprp.list and 73 in moprp.ass",
        },
        {
            "check": "trajectory_peptides_missing_active_amides",
            "value": int(sum(bool(values) for values in missing.values())),
            "details": json.dumps(missing, sort_keys=True),
        },
        {
            "check": "stored_hdxer_vs_canonical_expfact_rate_ratio_median",
            "value": float(np.median(ratios)),
            "details": f"min={ratios.min():.6g}; max={ratios.max():.6g}; n={len(ratios)}",
        },
    ]
    return pd.DataFrame(rows)


def _fit_ex2_tracks(dataset, rates: np.ndarray, args: argparse.Namespace):
    conditions = {"unregularized": 0.0, "source_harmonic": 1e-8}
    sets: dict[str, EX2SolutionSet] = {}
    summaries = []
    pf_rows = []
    arrays: dict[str, np.ndarray] = {}
    published_median = np.loadtxt(MOPRP / "median.pfact", dtype=float)[:, 1]
    for offset, (condition, harmonic) in enumerate(conditions.items()):
        fitted = fit_ex2_solution_set(
            dataset.observed_uptake,
            rates,
            dataset.protocol.timepoints_min,
            dataset.peptide_map,
            starts=args.ex2_starts,
            seed=args.seed + offset * 100_000,
            harmonic_strength=harmonic,
            random_search_steps=args.ex2_random_search,
            initial_log_pf_vectors=[published_median],
            maxiter=args.maxiter,
        )
        sets[condition] = fitted
        arrays[f"ex2__{condition}__predicted"] = np.stack(
            [solution.predicted for solution in fitted.solutions]
        )
        arrays[f"ex2__{condition}__log_pf"] = np.stack(
            [solution.log_pf for solution in fitted.solutions]
        )
        for rank, solution in enumerate(fitted.solutions):
            summaries.append(
                {
                    "condition": condition,
                    "solution_rank": rank,
                    "objective": solution.objective,
                    "curve_rmse": solution.rmse,
                    "optimizer_success": solution.success,
                    "iterations": solution.iterations,
                    "gradient_norm": solution.gradient_norm,
                    "initialization": solution.initialization,
                    "lower_bound_residues": int(np.nansum(solution.log_pf < 1e-4)),
                    "upper_bound_residues": int(np.nansum(solution.log_pf > 29.999)),
                }
            )
            for residue_id, value in zip(fitted.residue_ids, solution.log_pf):
                if np.isfinite(value):
                    pf_rows.append(
                        {
                            "condition": condition,
                            "solution_rank": rank,
                            "residue_id": int(residue_id),
                            "log_pf": float(value),
                        }
                    )
    return sets, pd.DataFrame(summaries), pd.DataFrame(pf_rows), arrays


def _trajectory_comparisons(dataset, canonical_rates, ex2_sets, args: argparse.Namespace):
    comparison_rows = []
    residual_rows = []
    distance_rows = []
    nmr_rows = []
    arrays: dict[str, np.ndarray] = {}
    nmr = pd.read_csv(MOPRP / "nmr.csv", encoding="utf-8-sig")
    nmr_ids = nmr["Ord_res"].to_numpy(int)
    nmr_values = nmr["ln(Pfact)"].to_numpy(float)
    nmr_kex = nmr["kex (min-1)"].to_numpy(float)
    canonical_nmr_rates = canonical_rates.aligned(nmr_ids)
    nmr_targets = {
        "published_intrinsic_rate_convention": nmr_values,
        "rebased_to_expfact_pH4p4_298K": np.log(canonical_nmr_rates / nmr_kex),
    }
    nmr_subsets = {
        "all_reported_residues": np.ones(len(nmr_ids), dtype=bool),
        "exclude_published_outliers_91_94": ~np.isin(nmr_ids, [91, 94]),
    }

    legacy_rate_table = np.loadtxt(
        MOPRP / "_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat"
    )
    legacy_rate_by_id = {
        int(residue): float(rate) for residue, rate in legacy_rate_table
    }
    feature_specs = [
        (name, mode, args.features_dir, f"{name}_{mode}")
        for mode in args.feature_modes
        for name in ("AF2_filtered", "AF2_MSAss")
    ]
    if args.include_legacy_features:
        feature_specs.extend(
            (name, "legacy_switched_missing_c", FEATURES, name)
            for name in ("AF2_filtered", "AF2_MSAss")
        )

    for name, contact_mode, feature_dir, feature_stem in feature_specs:
        feature_ids, frame_log_pf, _ = _feature_bundle(feature_stem, feature_dir)
        rate_cases = [
            (
                "canonical_expfact_pH4p4_298K",
                feature_ids,
                frame_log_pf,
                canonical_rates.aligned(feature_ids),
            )
        ]
        if contact_mode == "hard":
            legacy_ids = np.asarray(
                [residue for residue in feature_ids if int(residue) in legacy_rate_by_id],
                dtype=int,
            )
            legacy_columns = np.asarray(
                [int(np.flatnonzero(feature_ids == residue)[0]) for residue in legacy_ids]
            )
            rate_cases.append(
                (
                    "legacy_hdxer_96_residue_sensitivity",
                    legacy_ids,
                    frame_log_pf[legacy_columns],
                    np.asarray([legacy_rate_by_id[int(residue)] for residue in legacy_ids]),
                )
            )

        for rate_basis, case_ids, case_log_pf, rates in rate_cases:
            complete, missing = _complete_trajectory_peptides(dataset.peptide_map, case_ids)
            complete_map = dataset.peptide_map.subset_peptides(complete).aligned_to(case_ids)
            observed = dataset.observed_uptake[complete]
            comparison = compare_trajectory_hdx(
                case_log_pf,
                rates,
                dataset.protocol,
                complete_map,
                observed_uptake=observed,
            )
            comparison_rows.extend(
                [
                    {
                        "ensemble": name,
                        "contact_mode": contact_mode,
                        "intrinsic_rate_basis": rate_basis,
                        "semantics": "average_first",
                        "curve_rmse": comparison.average_first_rmse,
                        "n_complete_peptides": len(complete),
                        "excluded_peptide_ids": json.dumps(
                            [peptide for peptide, values in missing.items() if values]
                        ),
                    },
                    {
                        "ensemble": name,
                        "contact_mode": contact_mode,
                        "intrinsic_rate_basis": rate_basis,
                        "semantics": "frame_mixture_sensitivity",
                        "curve_rmse": comparison.frame_mixture_rmse,
                        "n_complete_peptides": len(complete),
                        "excluded_peptide_ids": json.dumps(
                            [peptide for peptide, values in missing.items() if values]
                        ),
                    },
                ]
            )
            key = rate_basis.replace("_pH4p4_298K", "")
            array_prefix = f"trajectory__{name}__{contact_mode}__{key}"
            arrays[f"{array_prefix}__average_first"] = (
                comparison.average_first_curves
            )
            arrays[f"{array_prefix}__frame_mixture"] = (
                comparison.frame_mixture_curves
            )
            for semantics, predicted in (
                ("average_first", comparison.average_first_curves),
                ("frame_mixture_sensitivity", comparison.frame_mixture_curves),
            ):
                for local_peptide, peptide_index in enumerate(complete):
                    for time_index, time_min in enumerate(dataset.protocol.timepoints_min):
                        residual = float(predicted[local_peptide, time_index] - observed[local_peptide, time_index])
                        residual_rows.append(
                            {
                                "ensemble": name,
                                "contact_mode": contact_mode,
                                "intrinsic_rate_basis": rate_basis,
                                "semantics": semantics,
                                "peptide_id": int(dataset.peptide_map.peptide_ids[peptide_index]),
                                "time_min": float(time_min),
                                "predicted_uptake": float(predicted[local_peptide, time_index]),
                                "observed_uptake": float(observed[local_peptide, time_index]),
                                "residual": residual,
                                "squared_error": residual**2,
                            }
                        )
            arrays[f"{array_prefix}__peptide_indices"] = complete

        mean_log_pf = np.mean(frame_log_pf, axis=1)
        feature_lookup = {int(residue): index for index, residue in enumerate(feature_ids)}
        common_nmr = [index for index, residue in enumerate(nmr_ids) if int(residue) in feature_lookup]
        x = np.asarray([mean_log_pf[feature_lookup[int(nmr_ids[index])]] for index in common_nmr])
        for subset_name, subset_mask in nmr_subsets.items():
            keep = np.asarray([subset_mask[index] for index in common_nmr])
            for target_name, target_values in nmr_targets.items():
                y = target_values[common_nmr]
                pearson, spearman, rmse = _correlations(x[keep], y[keep])
                nmr_rows.append(
                    {
                        "source": name,
                        "contact_mode": contact_mode,
                        "condition": "BV_average_log_pf",
                        "solution_rank": -1,
                        "nmr_target": target_name,
                        "nmr_subset": subset_name,
                        "n_residues": int(np.count_nonzero(keep)),
                        "pearson": pearson,
                        "spearman": spearman,
                        "rmse_log_pf": rmse,
                    }
                )

        for condition, fitted in ex2_sets.items():
            active = np.any(dataset.peptide_map.matrix > 0, axis=0)
            common_ids = np.intersect1d(feature_ids, dataset.peptide_map.residue_ids[active])
            feature_columns = np.asarray([feature_lookup[int(residue)] for residue in common_ids])
            fit_lookup = {int(residue): index for index, residue in enumerate(fitted.residue_ids)}
            fit_columns = np.asarray([fit_lookup[int(residue)] for residue in common_ids])
            distances = np.asarray(
                [
                    np.sqrt(
                        np.mean(
                            (mean_log_pf[feature_columns] - solution.log_pf[fit_columns]) ** 2
                        )
                    )
                    for solution in fitted.solutions
                ]
            )
            distance_rows.append(
                {
                    "ensemble": name,
                    "contact_mode": contact_mode,
                    "condition": condition,
                    "n_residues": len(common_ids),
                    "minimum_solution_rmse": float(distances.min()),
                    "median_solution_rmse": float(np.median(distances)),
                    "maximum_solution_rmse": float(distances.max()),
                }
            )

    for condition, fitted in ex2_sets.items():
        fit_lookup = {int(residue): index for index, residue in enumerate(fitted.residue_ids)}
        common = [index for index, residue in enumerate(nmr_ids) if int(residue) in fit_lookup]
        columns = np.asarray([fit_lookup[int(nmr_ids[index])] for index in common])
        for rank, solution in enumerate(fitted.solutions):
            x = solution.log_pf[columns]
            for subset_name, subset_mask in nmr_subsets.items():
                keep = np.asarray([subset_mask[index] for index in common])
                for target_name, target_values in nmr_targets.items():
                    y = target_values[common]
                    pearson, spearman, rmse = _correlations(x[keep], y[keep])
                    nmr_rows.append(
                        {
                            "source": "experimental_EX2_fit",
                            "condition": condition,
                            "solution_rank": rank,
                            "nmr_target": target_name,
                            "nmr_subset": subset_name,
                            "n_residues": int(np.count_nonzero(np.isfinite(x[keep]))),
                            "pearson": pearson,
                            "spearman": spearman,
                            "rmse_log_pf": rmse,
                        }
                    )
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(residual_rows),
        pd.DataFrame(distance_rows),
        pd.DataFrame(nmr_rows),
        arrays,
    )


def _deuteron_count_distributions(
    dataset, canonical_rates, ex2_sets, args: argparse.Namespace
) -> pd.DataFrame:
    """Pre-quench peptide-1 exchange-count predictions for envelope follow-up."""

    rows = []
    times = (1.0, 60.0, 1440.0)
    full_rates = canonical_rates.aligned(dataset.peptide_map.residue_ids)
    peptide_index = 0

    def append_distribution(
        source, condition, rank, rate_basis, semantics, time, distribution
    ):
        for exchanged_amides, probability in enumerate(distribution):
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "solution_rank": rank,
                    "intrinsic_rate_basis": rate_basis,
                    "semantics": semantics,
                    "time_min": time,
                    "exchanged_amides": exchanged_amides,
                    "probability": float(probability),
                    "stage": "pre_quench; no natural-isotope convolution or instrument response",
                }
            )

    for condition, fitted in ex2_sets.items():
        for rank, solution in enumerate(fitted.solutions):
            for time in times:
                distribution = peptide_deuteron_count_distribution(
                    solution.log_pf,
                    full_rates,
                    time,
                    dataset.peptide_map,
                    peptide_index,
                )
                append_distribution(
                    "experimental_EX2_fit",
                    condition,
                    rank,
                    "canonical_expfact_pH4p4_298K",
                    "residue_EX2",
                    time,
                    distribution,
                )

    feature_specs = [
        (name, mode, args.features_dir, f"{name}_{mode}")
        for mode in args.feature_modes
        for name in ("AF2_filtered", "AF2_MSAss")
    ]
    if args.include_legacy_features:
        feature_specs.extend(
            (name, "legacy_switched_missing_c", FEATURES, name)
            for name in ("AF2_filtered", "AF2_MSAss")
        )

    for name, contact_mode, feature_dir, feature_stem in feature_specs:
        feature_ids, frame_log_pf, _ = _feature_bundle(feature_stem, feature_dir)
        peptide_map = dataset.peptide_map.subset_peptides([peptide_index]).aligned_to(feature_ids)
        mean_log_pf = np.mean(frame_log_pf, axis=1)
        rate_sets = {"canonical_expfact_pH4p4_298K": canonical_rates.aligned(feature_ids)}
        for rate_basis, rates in rate_sets.items():
            for time in times:
                average_first = peptide_deuteron_count_distribution(
                    mean_log_pf, rates, time, peptide_map, 0
                )
                frame_distributions = np.stack(
                    [
                        peptide_deuteron_count_distribution(
                            frame_log_pf[:, frame], rates, time, peptide_map, 0
                        )
                        for frame in range(frame_log_pf.shape[1])
                    ]
                )
                append_distribution(
                    name,
                    f"BV_{contact_mode}",
                    -1,
                    rate_basis,
                    "average_first",
                    time,
                    average_first,
                )
                append_distribution(
                    name,
                    f"BV_{contact_mode}",
                    -1,
                    rate_basis,
                    "frame_mixture_sensitivity",
                    time,
                    np.mean(frame_distributions, axis=0),
                )
    return pd.DataFrame(rows)


def _trajectory_attribution(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Express construction sensitivities relative to canonical hard contacts."""

    canonical = comparisons[
        comparisons["intrinsic_rate_basis"] == "canonical_expfact_pH4p4_298K"
    ].copy()
    reference = canonical[canonical["contact_mode"] == "hard"][
        ["ensemble", "semantics", "curve_rmse"]
    ].rename(columns={"curve_rmse": "hard_contact_rmse"})
    attributed = canonical.merge(reference, on=["ensemble", "semantics"], how="left")
    attributed["rmse_delta_vs_hard"] = (
        attributed["curve_rmse"] - attributed["hard_contact_rmse"]
    )
    return attributed.sort_values(["ensemble", "semantics", "contact_mode"])


def _published_cluster_comparisons(
    ex2_sets, validation_dir: Path | None
) -> pd.DataFrame:
    """Compare fits with published regional exPfact cluster centers.

    Published clusters are region-wise summaries and cannot be combined into a
    coherent global PF vector.  Raw log-PF RMSE is descriptive only.
    """

    columns = [
        "condition",
        "solution_rank",
        "objective",
        "region",
        "n_residues",
        "nearest_published_cluster",
        "nearest_cluster_proportion",
        "log_pf_rmse_to_center",
    ]
    if validation_dir is None:
        return pd.DataFrame(columns=columns)
    mclust = validation_dir / "mclust"
    if not mclust.is_dir():
        raise FileNotFoundError(f"published mclust directory not found: {mclust}")
    rows = []
    for model_path in sorted(mclust.glob("*.mod")):
        models = pd.read_csv(model_path, sep=r"\s+", index_col=0)
        residue_ids = np.asarray([int(label.removeprefix("V")) for label in models.index])
        centers = models.to_numpy(dtype=float).T
        proportion_path = model_path.with_suffix(".pro")
        proportions = pd.read_csv(proportion_path, sep=r"\s+", index_col=0).iloc[:, 0].to_numpy(float)
        for condition, fitted in ex2_sets.items():
            lookup = {int(residue): index for index, residue in enumerate(fitted.residue_ids)}
            available = np.asarray([int(residue) in lookup for residue in residue_ids])
            columns_in_fit = np.asarray([lookup[int(residue)] for residue in residue_ids[available]])
            for rank, solution in enumerate(fitted.solutions):
                values = solution.log_pf[columns_in_fit]
                finite = np.isfinite(values)
                distances = np.sqrt(
                    np.mean(
                        (centers[:, available][:, finite] - values[finite][None, :]) ** 2,
                        axis=1,
                    )
                )
                nearest = int(np.argmin(distances))
                rows.append(
                    {
                        "condition": condition,
                        "solution_rank": rank,
                        "objective": solution.objective,
                        "region": model_path.stem,
                        "n_residues": int(np.count_nonzero(finite)),
                        "nearest_published_cluster": nearest + 1,
                        "nearest_cluster_proportion": float(proportions[nearest]),
                        "log_pf_rmse_to_center": float(distances[nearest]),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _bin_raw_mass_envelope(
    path: Path,
    *,
    base_mass: float,
    spacing: float,
    n_bins: int,
) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    nominal = np.rint((values[:, 0] - base_mass) / spacing).astype(int)
    valid = (nominal >= 0) & (nominal < n_bins) & (values[:, 1] > 0)
    binned = np.bincount(
        nominal[valid], weights=values[valid, 1], minlength=n_bins
    ).astype(float)
    if binned.sum() <= 0:
        raise ValueError(f"no positive envelope intensity mapped to nominal bins: {path}")
    return binned / binned.sum()


def _match_envelope_length(distribution: np.ndarray, n_bins: int) -> np.ndarray:
    result = np.zeros(n_bins, dtype=float)
    width = min(n_bins, len(distribution))
    result[:width] = distribution[:width]
    if result.sum() <= 0:
        raise ValueError("predicted envelope has no mass in the observed bin range")
    return result / result.sum()


def _peptide1_envelope_scores(
    deuteron_counts: pd.DataFrame,
    validation_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score peptide-1 shapes with control-calibrated effective back-exchange."""

    columns = [
        "source",
        "condition",
        "solution_rank",
        "intrinsic_rate_basis",
        "semantics",
        "time_min",
        "effective_survival_probability",
        "envelope_rmse",
        "envelope_r2",
    ]
    if validation_dir is None:
        return pd.DataFrame(columns=columns), {"status": "official envelope files not supplied"}
    paths = {index: validation_dir / f"pep1.{index}.txt" for index in range(1, 6)}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing official peptide-1 envelope files: {missing}")

    spacing = 1.00627
    n_bins = 10
    protonated_raw = np.loadtxt(paths[1], dtype=float)
    base_mass = float(protonated_raw[np.argmax(protonated_raw[:, 1]), 0])
    protonated = _bin_raw_mass_envelope(
        paths[1], base_mass=base_mass, spacing=spacing, n_bins=n_bins
    )
    fully_deuterated = _bin_raw_mass_envelope(
        paths[5], base_mass=base_mass, spacing=spacing, n_bins=n_bins
    )
    fully_labelled_counts = np.zeros(6, dtype=float)
    fully_labelled_counts[-1] = 1.0

    def full_control_loss(survival: float) -> float:
        retained = thin_deuteron_count_distribution(fully_labelled_counts, survival)
        predicted = _match_envelope_length(
            convolve_isotope_and_deuteron_distributions(protonated, retained), n_bins
        )
        return float(np.mean((predicted - fully_deuterated) ** 2))

    optimized = minimize_scalar(
        full_control_loss,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    survival = float(optimized.x)
    retained_full = thin_deuteron_count_distribution(fully_labelled_counts, survival)
    predicted_full = _match_envelope_length(
        convolve_isotope_and_deuteron_distributions(protonated, retained_full), n_bins
    )
    full_tss = float(np.sum((fully_deuterated - fully_deuterated.mean()) ** 2))
    calibration = {
        "status": "effective survival calibrated only from protonated and fully deuterated controls",
        "base_mass": base_mass,
        "nominal_spacing": spacing,
        "n_bins": n_bins,
        "effective_survival_probability": survival,
        "fully_deuterated_control_rmse": float(
            np.sqrt(np.mean((predicted_full - fully_deuterated) ** 2))
        ),
        "fully_deuterated_control_r2": float(
            1.0 - np.sum((predicted_full - fully_deuterated) ** 2) / full_tss
        ),
        "caveat": (
            "independent effective survival is a control-calibrated approximation; "
            "no residue-specific quench kinetics or instrument line-shape model is fitted"
        ),
    }
    observed_by_time = {
        1.0: _bin_raw_mass_envelope(
            paths[2], base_mass=base_mass, spacing=spacing, n_bins=n_bins
        ),
        60.0: _bin_raw_mass_envelope(
            paths[3], base_mass=base_mass, spacing=spacing, n_bins=n_bins
        ),
        1440.0: _bin_raw_mass_envelope(
            paths[4], base_mass=base_mass, spacing=spacing, n_bins=n_bins
        ),
    }
    rows = []
    group_columns = [
        "source",
        "condition",
        "solution_rank",
        "intrinsic_rate_basis",
        "semantics",
        "time_min",
    ]
    for keys, group in deuteron_counts.groupby(group_columns, dropna=False):
        source, condition, rank, rate_basis, semantics, time = keys
        pre_quench = group.sort_values("exchanged_amides")["probability"].to_numpy(float)
        retained = thin_deuteron_count_distribution(pre_quench, survival)
        predicted = _match_envelope_length(
            convolve_isotope_and_deuteron_distributions(protonated, retained), n_bins
        )
        observed = observed_by_time[float(time)]
        sse = float(np.sum((predicted - observed) ** 2))
        tss = float(np.sum((observed - observed.mean()) ** 2))
        rows.append(
            {
                "source": source,
                "condition": condition,
                "solution_rank": int(rank),
                "intrinsic_rate_basis": rate_basis,
                "semantics": semantics,
                "time_min": float(time),
                "effective_survival_probability": survival,
                "envelope_rmse": float(np.sqrt(sse / n_bins)),
                "envelope_r2": float(1.0 - sse / tss),
            }
        )
    return pd.DataFrame(rows, columns=columns), calibration


def _kinetic_track(dataset, args: argparse.Namespace):
    selection = select_shared_rate_model(
        dataset.observed_uptake,
        dataset.protocol.timepoints_min,
        components=tuple(args.components),
        shrinkages=tuple(args.shrinkages),
        starts=args.mixture_starts,
        seed=args.seed + 700_000,
    )
    fitted = fit_shared_rate_mixture(
        dataset.observed_uptake,
        dataset.protocol.timepoints_min,
        selection.selected_components,
        selection.selected_shrinkage,
        starts=args.mixture_starts,
        seed=args.seed + 800_000,
        maxiter=args.maxiter,
    )
    summaries = rate_distribution_summaries(fitted)
    rows = []
    for peptide_index, peptide_id in enumerate(dataset.peptide_map.peptide_ids):
        rows.append(
            {
                "peptide_id": int(peptide_id),
                **{name: float(values[peptide_index]) for name, values in summaries.items()},
            }
        )
    fit_row = pd.DataFrame(
        [
            {
                "selected_components": selection.selected_components,
                "selected_shrinkage": selection.selected_shrinkage,
                "curve_rmse": fitted.rmse,
                "optimizer_success": fitted.success,
                "gradient_norm": fitted.gradient_norm,
                "interpretation": "peptide kinetic embedding; not residue PF or conformational covariance",
            }
        ]
    )
    arrays = {
        "kinetic__rates": fitted.rates,
        "kinetic__weights": fitted.weights,
        "kinetic__predicted": fitted.predicted,
    }
    return fit_row, pd.DataFrame(selection.rows), pd.DataFrame(rows), arrays


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_expfact_dataset(MOPRP)
    canonical_rates = load_intrinsic_rate_file(
        MOPRP / "expfact_kint_pH4p4_298K_min.dat",
        provider="exPfact-3Ala-numeric-reference",
        temperature_k=298.0,
        ph=4.4,
    )
    rates = canonical_rates.aligned(dataset.peptide_map.residue_ids)
    # Preserve the original construction audit against the legacy HDXer-rate
    # bundle while scoring corrected trajectory features separately below.
    feature_ids, _, stored_rates = _feature_bundle("AF2_MSAss", FEATURES)

    audit = construction_audit(dataset, canonical_rates, feature_ids, stored_rates)
    ex2_sets, ex2_summary, ex2_pf, arrays = _fit_ex2_tracks(dataset, rates, args)
    trajectory, trajectory_residuals, pf_distance, nmr, trajectory_arrays = _trajectory_comparisons(
        dataset, canonical_rates, ex2_sets, args
    )
    trajectory_attribution = _trajectory_attribution(trajectory)
    deuteron_counts = _deuteron_count_distributions(dataset, canonical_rates, ex2_sets, args)
    cluster_comparisons = _published_cluster_comparisons(
        ex2_sets, args.expfact_validation_dir
    )
    envelope_scores, envelope_calibration = _peptide1_envelope_scores(
        deuteron_counts, args.expfact_validation_dir
    )
    arrays.update(trajectory_arrays)

    audit.to_csv(args.output_dir / "construction_audit.csv", index=False)
    ex2_summary.to_csv(args.output_dir / "ex2_solutions.csv", index=False)
    ex2_pf.to_csv(args.output_dir / "ex2_pf_solutions.csv", index=False)
    trajectory.to_csv(args.output_dir / "trajectory_curve_comparisons.csv", index=False)
    trajectory_residuals.to_csv(
        args.output_dir / "trajectory_peptide_time_residuals.csv", index=False
    )
    trajectory_attribution.to_csv(
        args.output_dir / "trajectory_construction_attribution.csv", index=False
    )
    pf_distance.to_csv(args.output_dir / "trajectory_pf_solution_distances.csv", index=False)
    nmr.to_csv(args.output_dir / "nmr_holdout_comparisons.csv", index=False)
    deuteron_counts.to_csv(
        args.output_dir / "peptide1_deuteron_count_distributions.csv", index=False
    )
    cluster_comparisons.to_csv(
        args.output_dir / "published_cluster_comparisons.csv", index=False
    )
    envelope_scores.to_csv(
        args.output_dir / "peptide1_envelope_scores.csv", index=False
    )
    (args.output_dir / "peptide1_envelope_calibration.json").write_text(
        json.dumps(envelope_calibration, indent=2, default=_json_default) + "\n"
    )

    if not args.skip_mixture:
        kinetic_fit, kinetic_selection, kinetic_summaries, kinetic_arrays = _kinetic_track(
            dataset, args
        )
        kinetic_fit.to_csv(args.output_dir / "kinetic_fit.csv", index=False)
        kinetic_selection.to_csv(args.output_dir / "kinetic_selection.csv", index=False)
        kinetic_summaries.to_csv(args.output_dir / "kinetic_rate_summaries.csv", index=False)
        arrays.update(kinetic_arrays)

    np.savez_compressed(args.output_dir / "arrays.npz", **arrays)
    feature_manifest_path = args.features_dir / "manifest.json"
    manifest = {
        "status": "mean_only_physics_audit; experimental covariance not calibrated",
        "protocol": {
            "timepoints_min": dataset.protocol.timepoints_min,
            "temperature_k": dataset.protocol.temperature_k,
            "experimental_pd": dataset.protocol.experimental_pd,
            "intrinsic_rate_ph": dataset.protocol.intrinsic_rate_ph,
            "exchange_regime": dataset.protocol.exchange_regime,
            "normalization": dataset.protocol.normalization,
            "replicate_count_reported_by_source": dataset.protocol.replicate_count,
            "replicate_values_available_locally": False,
        },
        "peptide_map": {
            "convention": dataset.peptide_map.convention,
            "active_amide_counts": dataset.peptide_map.active_amide_counts,
            "fitted_assignment_file": str(MOPRP / "moprp.list"),
            "clustering_assignment_file_not_used_for_fit": str(MOPRP / "moprp.ass"),
        },
        "inputs": {
            path.name: {"path": path, "sha256": _sha256(path)}
            for path in (
                MOPRP / "moprp.seq",
                MOPRP / "moprp.times",
                MOPRP / "moprp.dexp",
                MOPRP / "moprp.list",
                MOPRP / "expfact_kint_pH4p4_298K_min.dat",
            )
        },
        "trajectory_features": {
            "directory": args.features_dir,
            "manifest": feature_manifest_path,
            "manifest_sha256": (
                _sha256(feature_manifest_path) if feature_manifest_path.exists() else None
            ),
            "canonical_contact_mode": "hard",
            "switched_contact_role": "legacy sensitivity only",
        },
        "configuration": vars(args),
        "interpretation": {
            "EX2": "residue PF solution set; multistart range is not a confidence interval",
            "shared_rate_mixture": "peptide kinetic embedding only",
            "trajectory": "comparison only; no reweighting or BV coefficient fitting",
            "BV_contacts": (
                "binary protein contact counts are canonical; rational switched contacts "
                "are a labelled construction sensitivity"
            ),
            "frame_mixture": "static-subpopulation sensitivity, not primary HDXer semantics",
            "peptide1_deuteron_counts": "pre-quench Poisson-binomial prediction",
            "peptide1_envelope": (
                "protonated-control convolution with full-deuterated-control effective survival"
                if args.expfact_validation_dir is not None
                else "not scored because official control and exchange spectra were not supplied"
            ),
            "published_clusters": (
                "regional reference summaries; not coherent global solutions"
                if args.expfact_validation_dir is not None
                else "not evaluated because no official validation directory was supplied"
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_moprp_ex2_physics_bv_v2",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_featurise_physics_v2",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=("hard", "switched"),
        default=("hard", "switched"),
    )
    parser.add_argument(
        "--no-legacy-features",
        action="store_false",
        dest="include_legacy_features",
        help="Do not include the original switched/missing-C-terminal feature baseline",
    )
    parser.set_defaults(include_legacy_features=True)
    parser.add_argument("--ex2-starts", type=int, default=20)
    parser.add_argument(
        "--ex2-random-search",
        type=int,
        default=10000,
        help="Source exPfact random candidates evaluated before each local minimization",
    )
    parser.add_argument("--mixture-starts", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--shrinkages", type=float, nargs="+", default=[0.0, 1e-4, 1e-2, 1.0]
    )
    parser.add_argument("--skip-mixture", action="store_true")
    parser.add_argument(
        "--expfact-validation-dir",
        type=Path,
        default=None,
        help="Optional official exPfact validation directory containing mclust outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
