#!/usr/bin/env python3
"""Train and qualify transferable joint covariance/variance priors on ISO.

This is the development side of the ISO -> MoPrP boundary.  Known ISO population
covariances supervise point and low-rank family models.  Artifacts written here are frozen
before the separate MoPrP runner loads them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import MDAnalysis as mda
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.src.analysis.elastic_network import anm_covariance
from jaxent.src.analysis.joint_covariance_geometry import (
    build_residue_pair_features,
    fit_linear_geometry_model,
    load_linear_geometry_model,
    peptide_tangent_prior,
    relative_log_variance,
    reference_population_geometry,
    save_linear_geometry_model,
    scale_free_log_spd,
    upper_triangle_features,
)
from jaxent.src.analysis.joint_geometry_reweighting import (
    FrozenPeptidePrior,
    ReweightingCell,
    optimize_joint_reweighting,
    peptide_logpf_covariance,
    predict_uptake,
)
from jaxent.src.analysis.pf_variance import (
    covariance_profiles_from_covariance,
    inverse_overlap_degree_weights,
    jensen_shannon_recovery_percent,
    overlap_projection,
    uptake_from_log_pf,
    weighted_population_covariance,
)
from jaxent.src.analysis.state_population import correlation_of

import investigate_pf_peptides as peptide_base


REFERENCE_BC, REFERENCE_BH = 0.35, 2.0
POPULATIONS = (0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
               0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
RIDGES = (0.01, 0.1, 1.0, 10.0)
RANKS = (0, 1, 2, 3, 4)
MARGINAL_STRENGTHS = (0.1, 1.0, 10.0)
METHODS = (
    "mean_only",
    "shape",
    "marginal_oracle",
    "conditional_oracle",
    "unlearned",
    "point",
    "family",
    "oracle",
)
ANM_CUTOFF = 24.0
SEED = 20260722

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[1]
FEATURE_ROOT = HERE / "_featurise"
CLUSTER_ROOT = EXAMPLE / "data/_clustering_results"
REFERENCE_STRUCTURE = (
    EXAMPLE
    / "data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_feature_bundle(stem: str) -> dict[str, np.ndarray]:
    topology_path = FEATURE_ROOT / f"topology_{stem}.json"
    topology = json.loads(topology_path.read_text())["topologies"]
    residue_ids = np.asarray([entry["residues"][0] for entry in topology], dtype=int)
    with np.load(FEATURE_ROOT / f"features_{stem}.npz") as arrays:
        heavy = np.asarray(arrays["heavy_contacts"], dtype=float)
        acceptor = np.asarray(arrays["acceptor_contacts"], dtype=float)
        k_ints = np.asarray(arrays["k_ints"], dtype=float)
    if heavy.shape[0] != residue_ids.size or acceptor.shape != heavy.shape:
        raise ValueError(f"{stem}: features and topology are not aligned")
    return {
        "residue_ids": residue_ids,
        "heavy": heavy,
        "acceptor": acceptor,
        "k_ints": k_ints,
    }


def _assignments(name: str) -> np.ndarray:
    path = CLUSTER_ROOT / f"cluster_assignments_{name}.csv"
    return pd.read_csv(path)["cluster_assignment"].to_numpy(dtype=int)


def population_weights(assignments: np.ndarray, open_population: float) -> np.ndarray:
    """Uniform-within-state ISO weights with strict zero decoy mass."""

    assignments = np.asarray(assignments)
    weights = np.zeros(assignments.size, dtype=float)
    open_mask, closed_mask = assignments == 0, assignments == 1
    if not open_mask.any() or not closed_mask.any():
        raise ValueError("ISO population requires both open and closed frames")
    weights[open_mask] = float(open_population) / open_mask.sum()
    weights[closed_mask] = (1.0 - float(open_population)) / closed_mask.sum()
    return weights


def _structure(residue_ids: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    universe = mda.Universe(str(REFERENCE_STRUCTURE))
    ca = universe.select_atoms("name CA")
    lookup = {int(resid): index for index, resid in enumerate(ca.resids)}
    indices = np.asarray([lookup[int(resid)] for resid in residue_ids], dtype=int)
    coordinates = ca.positions[indices].astype(float)
    residue_names = [str(ca.resnames[index]) for index in indices]
    return coordinates, residue_names, anm_covariance(coordinates, cutoff=ANM_CUTOFF)


def load_iso_training_data(populations: tuple[float, ...] = POPULATIONS) -> dict:
    bi, tri = _load_feature_bundle("iso_bi"), _load_feature_bundle("iso_tri")
    bi_assignments, tri_assignments = _assignments("ISO_BI"), _assignments("ISO_TRI")
    if not np.array_equal(bi_assignments, tri_assignments[tri_assignments != -1]):
        raise ValueError("ISO_TRI non-decoy state order does not match ISO_BI")
    for key in ("heavy", "acceptor"):
        if not np.array_equal(bi[key], tri[key][:, tri_assignments != -1]):
            raise ValueError(f"ISO_TRI non-decoy {key} features do not match ISO_BI")
    coordinates, residue_names, anm = _structure(bi["residue_ids"])
    feature_views, feature_names = [], None
    for bundle in (bi, tri):
        features, names = build_residue_pair_features(
            residue_ids=bundle["residue_ids"],
            k_ints=bundle["k_ints"],
            heavy_contacts=bundle["heavy"],
            acceptor_contacts=bundle["acceptor"],
            coordinates=coordinates,
            residue_names=residue_names,
            anm_covariance=anm,
            reference_bc=REFERENCE_BC,
            reference_bh=REFERENCE_BH,
        )
        if feature_names is not None and names != feature_names:
            raise ValueError("ISO feature views produced different schemas")
        feature_names = names
        feature_views.append(features)
    geometries = np.stack(
        [
            reference_population_geometry(
                bi["heavy"],
                bi["acceptor"],
                population_weights(bi_assignments, population),
                bc=REFERENCE_BC,
                bh=REFERENCE_BH,
            )
            for population in populations
        ]
    )
    return {
        "bi": bi,
        "tri": tri,
        "bi_assignments": bi_assignments,
        "tri_assignments": tri_assignments,
        "feature_views": feature_views,
        "feature_names": feature_names,
        "target_geometries": geometries,
        "populations": np.asarray(populations),
    }


def population_folds(n_populations: int, n_folds: int = 5) -> tuple[np.ndarray, ...]:
    """Return deterministic contiguous held-out population bands."""

    return tuple(np.asarray(part, dtype=int) for part in np.array_split(np.arange(n_populations), n_folds))


def _shared_scores(
    target: np.ndarray,
    centers: list[np.ndarray],
    modes: list[np.ndarray],
    penalty: float,
) -> np.ndarray:
    if modes[0].shape[0] == 0:
        return np.zeros(0)
    response = np.concatenate(
        [upper_triangle_features(target - center) for center in centers]
    )
    design = np.concatenate(
        [np.stack([upper_triangle_features(mode) for mode in view_modes], axis=1) for view_modes in modes],
        axis=0,
    )
    return np.linalg.solve(
        design.T @ design + float(penalty) * np.eye(design.shape[1]),
        design.T @ response,
    )


def geometry_cross_validation(
    data: dict,
    *,
    ridges: tuple[float, ...] = RIDGES,
    ranks: tuple[int, ...] = RANKS,
    score_penalties: tuple[float, ...] = RIDGES,
) -> pd.DataFrame:
    rows = []
    folds = population_folds(len(data["populations"]))
    all_indices = np.arange(len(data["populations"]))
    dimension = data["target_geometries"].shape[1]
    for fold_index, held_out in enumerate(folds):
        train = np.setdiff1d(all_indices, held_out)
        for ridge in ridges:
            for rank in ranks:
                if rank >= train.size:
                    continue
                model, _ = fit_linear_geometry_model(
                    data["feature_views"],
                    data["feature_names"],
                    data["target_geometries"][train],
                    rank=rank,
                    ridge=ridge,
                )
                predictions = [model.predict(view, dimension) for view in data["feature_views"]]
                penalties = (0.0,) if rank == 0 else score_penalties
                for score_penalty in penalties:
                    for population_index in held_out:
                        target = data["target_geometries"][population_index]
                        centers = [prediction[0] for prediction in predictions]
                        modes = [prediction[1] for prediction in predictions]
                        scores = _shared_scores(target, centers, modes, score_penalty)
                        errors = []
                        for center, view_modes in zip(centers, modes, strict=True):
                            reconstructed = center + (
                                np.einsum("k,kij->ij", scores, view_modes) if rank else 0.0
                            )
                            errors.append(float(np.mean(np.square(reconstructed - target))))
                        rows.append(
                            {
                                "fold": fold_index,
                                "population_index": int(population_index),
                                "open_population": float(data["populations"][population_index]),
                                "method": "point" if rank == 0 else "family",
                                "rank": rank,
                                "ridge": ridge,
                                "score_penalty": score_penalty,
                                "geometry_mse": float(np.mean(errors)),
                                "score_norm": float(np.linalg.norm(scores)),
                            }
                        )
    return pd.DataFrame(rows)


def select_geometry_settings(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    selected = {}
    for method in ("point", "family"):
        subset = frame[frame.method == method]
        grouped = (
            subset.groupby(["rank", "ridge", "score_penalty"], as_index=False)
            .agg(mean_geometry_mse=("geometry_mse", "mean"), max_geometry_mse=("geometry_mse", "max"))
            .sort_values(["mean_geometry_mse", "max_geometry_mse", "rank", "ridge"])
        )
        row = grouped.iloc[0]
        selected[method] = {
            "rank": int(row["rank"]),
            "ridge": float(row["ridge"]),
            "score_penalty": float(row["score_penalty"]),
            "mean_geometry_mse": float(row["mean_geometry_mse"]),
        }
    return selected


def _population_metrics(
    weights: np.ndarray, assignments: np.ndarray, open_population: float
) -> dict[str, float]:
    populations = np.asarray(
        [weights[assignments == label].sum() for label in (0, 1, -1)], dtype=float
    )
    target = jnp.asarray([open_population, 1.0 - open_population, 0.0])
    return {
        "open_mass": float(populations[0]),
        "closed_mass": float(populations[1]),
        "decoy_mass": float(populations[2]),
        "recovery_pct": float(jensen_shannon_recovery_percent(jnp.asarray(populations), target)),
        "ess": float(1.0 / np.square(weights).sum()),
    }


def _panel(panel_name: str) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    _, mapping, _ = peptide_base._build_panel(
        panel_name,
        peptide_base.generate_panel_bounds()[panel_name],
        FEATURE_ROOT / "features_iso_tri.npz",
        FEATURE_ROOT / "topology_iso_tri.json",
    )
    splits = {
        index: tuple(np.asarray(values, dtype=int) for values in peptide_base.LOCKED_SPLITS[(panel_name, index)])
        for index in range(3)
    }
    return mapping, splits


def _fixed_prior(
    method: str,
    *,
    bundle: dict,
    assignments: np.ndarray,
    mapping: np.ndarray,
    projection: np.ndarray,
    marginal_weights: np.ndarray,
    model=None,
    pair_features: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> FrozenPeptidePrior:
    reference_logpf = REFERENCE_BC * bundle["heavy"] + REFERENCE_BH * bundle["acceptor"]
    uniform = jnp.full(reference_logpf.shape[1], 1.0 / reference_logpf.shape[1])
    unweighted_covariance = peptide_logpf_covariance(reference_logpf, mapping, uniform)
    if method == "mean_only":
        return FrozenPeptidePrior(kind="mean_only")
    if method == "shape":
        return FrozenPeptidePrior(kind="shape", correlation=correlation_of(unweighted_covariance))
    if method == "unlearned":
        return FrozenPeptidePrior(
            kind="unlearned",
            geometry=scale_free_log_spd(unweighted_covariance, projection),
            relative_variance=relative_log_variance(unweighted_covariance, marginal_weights),
        )
    if method in {"marginal_oracle", "conditional_oracle", "oracle"}:
        if target_weights is None:
            raise ValueError(f"{method} prior requires target weights")
        residue_logpf = jnp.asarray(reference_logpf)
        residue_covariance = weighted_population_covariance(residue_logpf, target_weights)
        peptide_covariance = jnp.asarray(mapping) @ residue_covariance @ jnp.asarray(mapping).T
        if method in {"marginal_oracle", "conditional_oracle"}:
            profile_index = 1 if method == "marginal_oracle" else 3
            profile = covariance_profiles_from_covariance(peptide_covariance)[profile_index]
            return FrozenPeptidePrior(
                kind=method,
                profile=profile,
                profile_weights=marginal_weights,
            )
        return FrozenPeptidePrior(
            kind="oracle",
            geometry=scale_free_log_spd(peptide_covariance, projection),
            relative_variance=relative_log_variance(peptide_covariance, marginal_weights),
        )
    if method not in {"point", "family"} or model is None or pair_features is None:
        raise ValueError(f"cannot build prior {method!r}")
    residue_center, residue_modes = model.predict(pair_features, bundle["heavy"].shape[0])
    center, modes, marginal, marginal_modes = peptide_tangent_prior(
        residue_center,
        residue_modes,
        mapping,
        projection,
        marginal_weights,
    )
    return FrozenPeptidePrior(
        kind=method,
        geometry=center,
        geometry_modes=modes,
        relative_variance=marginal,
        marginal_modes=marginal_modes,
        score_precision=model.score_precision,
    )


def _mean_mse(
    bundle: dict,
    weights: np.ndarray,
    bc: float,
    bh: float,
    mapping: np.ndarray,
    target: np.ndarray,
) -> float:
    log_pf = bc * bundle["heavy"] + bh * bundle["acceptor"]
    prediction = np.asarray(
        predict_uptake(log_pf, bundle["k_ints"], peptide_base.TIMEPOINTS, mapping, weights)
    ).T
    return float(np.mean(np.square(prediction - target)))


def run_reweighting_qualification(
    data: dict,
    models: dict,
    selected_geometry: dict,
    *,
    smoke: bool,
    steps: int,
    starts: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[pd.DataFrame, dict | None]:
    populations = data["populations"] if not smoke else np.asarray([0.50])
    panels = ("equal", "random_fixed", "random_variable") if not smoke else ("equal",)
    split_indices = range(3) if not smoke else range(1)
    marginal_strengths = MARGINAL_STRENGTHS if not smoke else (1.0,)
    methods = METHODS
    rows = []
    prior_cache = {}
    bundles = {"ISO_BI": data["bi"], "ISO_TRI": data["tri"]}
    assignments = {"ISO_BI": data["bi_assignments"], "ISO_TRI": data["tri_assignments"]}
    feature_views = dict(zip(bundles, data["feature_views"], strict=True))
    job_index = 0

    for panel_name in panels:
        full_mapping, splits = _panel(panel_name)
        for split_index in split_indices:
            train_indices, val_indices = splits[split_index]
            mappings = {"train": full_mapping[train_indices], "val": full_mapping[val_indices]}
            for population in populations:
                target_weights = population_weights(data["bi_assignments"], float(population))
                mean_logpf = (
                    REFERENCE_BC * data["bi"]["heavy"] + REFERENCE_BH * data["bi"]["acceptor"]
                ) @ target_weights
                residue_target = np.asarray(
                    uptake_from_log_pf(mean_logpf, data["bi"]["k_ints"], peptide_base.TIMEPOINTS)
                )
                target_uptake = {
                    subset: np.asarray(residue_target @ mapping.T).T
                    for subset, mapping in mappings.items()
                }
                for method in methods:
                    strengths = marginal_strengths if method in {"unlearned", "point", "family", "oracle"} else (0.0,)
                    for marginal_strength in strengths:
                        selected_for_shard = job_index % num_shards == shard_index
                        job_index += 1
                        if not selected_for_shard:
                            continue
                        cells = {}
                        for ensemble, bundle in bundles.items():
                            mapping = mappings["train"]
                            projection = np.asarray(overlap_projection(mapping))
                            overlap_weights = np.asarray(inverse_overlap_degree_weights(mapping))
                            cache_key = (panel_name, split_index, method, ensemble)
                            if method in {"marginal_oracle", "conditional_oracle", "oracle"}:
                                prior = _fixed_prior(
                                    method,
                                    bundle=bundle,
                                    assignments=assignments[ensemble],
                                    mapping=mapping,
                                    projection=projection,
                                    marginal_weights=overlap_weights,
                                    target_weights=population_weights(assignments[ensemble], float(population)),
                                )
                            else:
                                if cache_key not in prior_cache:
                                    prior_cache[cache_key] = _fixed_prior(
                                        method,
                                        bundle=bundle,
                                        assignments=assignments[ensemble],
                                        mapping=mapping,
                                        projection=projection,
                                        marginal_weights=overlap_weights,
                                        model=models.get(method),
                                        pair_features=feature_views[ensemble],
                                    )
                                prior = prior_cache[cache_key]
                            uniform = np.full(bundle["heavy"].shape[1], 1.0 / bundle["heavy"].shape[1])
                            mean_reference = _mean_mse(
                                bundle,
                                uniform,
                                REFERENCE_BC,
                                REFERENCE_BH,
                                mapping,
                                target_uptake["train"],
                            )
                            cells[ensemble] = ReweightingCell(
                                name=ensemble,
                                heavy_contacts=bundle["heavy"],
                                acceptor_contacts=bundle["acceptor"],
                                k_ints=bundle["k_ints"],
                                timepoints=peptide_base.TIMEPOINTS,
                                mapping=mapping,
                                observed_uptake=target_uptake["train"],
                                train_time_indices=np.arange(peptide_base.TIMEPOINTS.size),
                                projection=projection,
                                marginal_weights=overlap_weights,
                                mean_reference=mean_reference,
                                prior=prior,
                            )
                        score_strength = float(selected_geometry.get("family", {}).get("score_penalty", 1.0))
                        result = optimize_joint_reweighting(
                            cells,
                            marginal_strength=marginal_strength,
                            score_strength=score_strength,
                            steps=steps,
                            starts=starts,
                            seed=SEED + int(round(float(population) * 1000)) + split_index,
                        )
                        for ensemble, bundle in bundles.items():
                            weights = result["weights"][ensemble]
                            metrics = _population_metrics(
                                weights, assignments[ensemble], float(population)
                            )
                            validation_mse = _mean_mse(
                                bundle,
                                weights,
                                result["bc"],
                                result["bh"],
                                mappings["val"],
                                target_uptake["val"],
                            )
                            rows.append(
                                {
                                    "panel": panel_name,
                                    "split": split_index,
                                    "open_population_target": float(population),
                                    "method": method,
                                    "marginal_strength": float(marginal_strength),
                                    "ensemble": ensemble,
                                    "objective": result["objective"],
                                    "best_start": result["best_start"],
                                    "bc": result["bc"],
                                    "bh": result["bh"],
                                    "val_mean_mse": validation_mse,
                                    **metrics,
                                }
                            )
    frame = pd.DataFrame(rows)
    selection = select_reweighting_settings(frame) if num_shards == 1 else None
    return frame, selection


def select_reweighting_settings(frame: pd.DataFrame) -> dict:
    baselines = frame[frame.method == "mean_only"][[
        "panel", "split", "open_population_target", "ensemble", "val_mean_mse"
    ]].rename(columns={"val_mean_mse": "baseline_mse"})
    candidates = frame[frame.method.isin(["unlearned", "point", "family"])].merge(
        baselines,
        on=["panel", "split", "open_population_target", "ensemble"],
        how="left",
    )
    candidates["mean_ratio"] = candidates.val_mean_mse / candidates.baseline_mse.clip(lower=1e-12)
    # Rank each setting by the worse of ISO_BI and ISO_TRI within a matched
    # population/panel/split cell.  This prevents an easy ensemble from hiding a
    # failure to reject the TRI decoy or recover the BI population.
    paired = (
        candidates.groupby(
            ["method", "marginal_strength", "panel", "split", "open_population_target"],
            as_index=False,
        )
        .agg(
            recovery_pct=("recovery_pct", "min"),
            decoy_mass=("decoy_mass", "max"),
            mean_ratio=("mean_ratio", "max"),
        )
    )
    grouped = (
        paired.groupby(["method", "marginal_strength"], as_index=False)
        .agg(
            median_recovery=("recovery_pct", "median"),
            median_decoy=("decoy_mass", "median"),
            median_mean_ratio=("mean_ratio", "median"),
        )
    )
    eligible = grouped[grouped.median_mean_ratio <= 1.05]
    pool = eligible if not eligible.empty else grouped
    winners = (
        pool.sort_values(
            ["method", "median_recovery", "median_decoy"], ascending=[True, False, True]
        )
        .groupby("method", as_index=False)
        .first()
    )
    primary = winners.sort_values(
        ["median_recovery", "median_decoy"], ascending=[False, True]
    ).iloc[0]
    return {
        "mean_gate_passed_by_any": bool(not eligible.empty),
        "per_method": {
            row.method: {
                "marginal_strength": float(row.marginal_strength),
                "median_recovery": float(row.median_recovery),
                "median_decoy": float(row.median_decoy),
                "median_mean_ratio": float(row.median_mean_ratio),
            }
            for row in winners.itertuples(index=False)
        },
        "primary_method": str(primary.method),
    }


def write_qualification_summary(
    frame: pd.DataFrame, selection: dict, output_dir: Path
) -> None:
    """Write a compact human-readable summary and an effect-size diagnostic."""

    lines = [
        "# ISO joint-geometry qualification",
        "",
        f"Primary transferable method: `{selection['primary_method']}`.",
        f"Mean-fit gate passed: `{selection['mean_gate_passed_by_any']}`.",
        "",
        "The learned point/family priors and unlearned prior are selectable. "
        "Marginal, conditional, and full-covariance oracle methods are reference ceilings only.",
        "",
        "| Method | Marginal strength | Median worse-ensemble recovery (%) | "
        "Median worse-ensemble decoy mass | Median worse-ensemble mean ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in selection["per_method"].items():
        lines.append(
            f"| {method} | {values['marginal_strength']:.3g} | "
            f"{values['median_recovery']:.3f} | {values['median_decoy']:.4f} | "
            f"{values['median_mean_ratio']:.4f} |"
        )
    (output_dir / "iso_qualification_summary.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    plot_frame = frame[frame.method.isin(["mean_only", "shape", "unlearned", "point", "family"])]
    summary = plot_frame.groupby(["method", "ensemble"], as_index=False).recovery_pct.median()
    ensembles = tuple(summary.ensemble.unique())
    methods = ("mean_only", "shape", "unlearned", "point", "family")
    x = np.arange(len(methods))
    width = 0.8 / max(len(ensembles), 1)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, ensemble in enumerate(ensembles):
        values = summary[summary.ensemble == ensemble].set_index("method").recovery_pct
        axis.bar(
            x + (index - (len(ensembles) - 1) / 2) * width,
            [values.get(method, np.nan) for method in methods],
            width,
            label=ensemble,
        )
    axis.set_xticks(x, methods, rotation=20, ha="right")
    axis.set_ylabel("Median ISO population recovery (%)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "iso_recovery_effect_sizes.png", dpi=180)
    plt.close(figure)


def _finalize_iso_qualification(
    frame: pd.DataFrame, output_dir: Path, *, write_raw: bool = True
) -> dict:
    selection = select_reweighting_settings(frame)
    if write_raw:
        frame.to_csv(output_dir / "iso_reweighting_raw.csv", index=False)
    (output_dir / "selected_reweighting_settings.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True)
    )
    for method in ("point", "family"):
        manifest_path = output_dir / f"{method}_artifact/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["selected_marginal_strength"] = selection["per_method"][method][
            "marginal_strength"
        ]
        manifest["iso_primary_method"] = selection["primary_method"]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    write_qualification_summary(frame, selection, output_dir)
    return selection


def merge_iso_shards(output_dir: Path, num_shards: int) -> dict:
    paths = [
        output_dir / f"iso_reweighting_raw.shard_{index:04d}_of_{num_shards:04d}.csv"
        for index in range(num_shards)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing ISO qualification shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["panel", "split", "open_population_target", "method", "marginal_strength", "ensemble"]
    if frame.duplicated(keys).any():
        raise ValueError("ISO qualification shards contain duplicate jobs")
    return _finalize_iso_qualification(frame, output_dir)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.merge_shards:
        selection = merge_iso_shards(args.output_dir, args.num_shards)
        print(json.dumps(selection, indent=2))
        return
    if args.train_only and args.qualification_only:
        raise ValueError("--train-only and --qualification-only are mutually exclusive")
    populations = POPULATIONS if not args.smoke else (0.05, 0.50, 0.95)
    ridges = RIDGES if not args.smoke else (0.1, 1.0)
    ranks = RANKS if not args.smoke else (0, 1)
    data = load_iso_training_data(populations)
    if args.qualification_only:
        selected = json.loads((args.output_dir / "selected_geometry_settings.json").read_text())
        models = {
            method: load_linear_geometry_model(args.output_dir / f"{method}_artifact")[0]
            for method in ("point", "family")
        }
    else:
        cv = geometry_cross_validation(data, ridges=ridges, ranks=ranks, score_penalties=ridges)
        cv.to_csv(args.output_dir / "geometry_model_cv.csv", index=False)
        selected = select_geometry_settings(cv)
        selected["candidate_marginal_strengths"] = list(MARGINAL_STRENGTHS)
        (args.output_dir / "selected_geometry_settings.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True)
        )

    inputs = {
        "features_iso_bi": _sha256(FEATURE_ROOT / "features_iso_bi.npz"),
        "features_iso_tri": _sha256(FEATURE_ROOT / "features_iso_tri.npz"),
        "clusters_iso_bi": _sha256(CLUSTER_ROOT / "cluster_assignments_ISO_BI.csv"),
        "clusters_iso_tri": _sha256(CLUSTER_ROOT / "cluster_assignments_ISO_TRI.csv"),
        "reference_structure": _sha256(REFERENCE_STRUCTURE),
    }
    if not args.qualification_only:
        models = {}
        for method in ("point", "family"):
            setting = selected[method]
            model, scores = fit_linear_geometry_model(
                data["feature_views"],
                data["feature_names"],
                data["target_geometries"],
                rank=int(setting["rank"]),
                ridge=float(setting["ridge"]),
            )
            models[method] = model
            save_linear_geometry_model(
                model,
                args.output_dir / f"{method}_artifact",
                {
                    "training_populations": list(populations),
                    "reference_bc": REFERENCE_BC,
                    "reference_bh": REFERENCE_BH,
                    "anm_cutoff": ANM_CUTOFF,
                    "score_penalty": setting["score_penalty"],
                    "candidate_marginal_strengths": list(MARGINAL_STRENGTHS),
                    "input_hashes": inputs,
                    "seed": SEED,
                    "smoke": bool(args.smoke),
                },
            )
            np.savez_compressed(
                args.output_dir / f"{method}_population_scores.npz", scores=scores
            )

    if not args.qualification_only:
        pd.DataFrame(
            {
                "open_population": populations,
                "closed_population": [1.0 - value for value in populations],
                "tri_decoy_population": np.zeros(len(populations)),
            }
        ).to_csv(args.output_dir / "population_grid.csv", index=False)
    if not args.train_only:
        raw, reweighting_selection = run_reweighting_qualification(
            data,
            models,
            selected,
            smoke=bool(args.smoke),
            steps=args.steps,
            starts=args.starts,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        if args.num_shards == 1:
            reweighting_selection = _finalize_iso_qualification(raw, args.output_dir)
            print(json.dumps(reweighting_selection, indent=2))
        else:
            shard_path = args.output_dir / (
                f"iso_reweighting_raw.shard_{args.shard_index:04d}_of_{args.num_shards:04d}.csv"
            )
            raw.to_csv(shard_path, index=False)
            print(f"wrote ISO qualification shard to {shard_path}")
    print(json.dumps(selected, indent=2))
    if args.qualification_only:
        print(f"used frozen ISO artifacts under {args.output_dir}")
    else:
        print(f"wrote frozen ISO artifacts under {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "_iso_joint_geometry_prior")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--starts", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
