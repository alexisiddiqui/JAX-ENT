#!/usr/bin/env python3
"""Archived external validator for the failed former Stage-J experiment.

The ISO qualification did not pass its preregistered held-out mean-fit gate.  The
implementation remains available as provenance, but execution is prohibited so its
fallback ``primary_method`` cannot accidentally launch MoPrP validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import jax.numpy as jnp
import numpy as np
import pandas as pd

import _moprp_recovery_common as common
from jaxent.src.analysis.elastic_network import anm_covariance
from jaxent.src.analysis.joint_covariance_geometry import (
    build_residue_pair_features,
    feature_out_of_distribution,
    load_linear_geometry_model,
    peptide_tangent_prior,
    relative_log_variance,
    scale_free_log_spd,
)
from jaxent.src.analysis.joint_geometry_reweighting import (
    FrozenPeptidePrior,
    ReweightingCell,
    optimize_joint_reweighting,
    peptide_logpf_covariance,
    predict_uptake,
)
from jaxent.src.analysis.pf_variance import (
    conditional_subset_effective_sample_size,
    covariance_profiles_from_covariance,
    inverse_overlap_degree_weights,
    overlap_projection,
    weighted_population_covariance,
)
from jaxent.src.analysis.state_population import (
    correlation_of,
    state_populations,
    strict_recovery_percent,
)


REFERENCE_BC, REFERENCE_BH = 0.35, 2.0
ANM_CUTOFF = 24.0
ALPHA = 0.05
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
TARGET_STATES = {"Folded", "PUF1", "PUF2"}
HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = (
    common.PACKAGE_ROOT
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/_iso_joint_geometry_prior"
)
STRUCTURE = common.BASE / "data/_MoPrP/MoPrP_max_plddt_4334.pdb"


def prohibit_failed_stage_j_moprp_launch() -> None:
    """Fail closed: former Stage-J artifacts are not qualified for MoPrP."""

    raise RuntimeError(
        "former Stage J failed its ISO held-out mean gate; MoPrP execution is prohibited. "
        "Use only a qualified geometry-regularised HDX target-variance artifact."
    )


def time_folds(n_timepoints: int, block: int = 3) -> list[np.ndarray]:
    return [np.arange(start, min(start + block, n_timepoints)) for start in range(0, n_timepoints, block)]


def _structure(residue_ids: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    universe = mda.Universe(str(STRUCTURE))
    ca = universe.select_atoms("name CA")
    lookup = {int(resid): index for index, resid in enumerate(ca.resids)}
    indices = np.asarray([lookup[int(resid)] for resid in residue_ids], dtype=int)
    coordinates = ca.positions[indices].astype(float)
    names = [str(ca.resnames[index]) for index in indices]
    return coordinates, names, anm_covariance(coordinates, cutoff=ANM_CUTOFF)


def _features(inputs: common.EnsembleInputs) -> tuple[np.ndarray, tuple[str, ...]]:
    coordinates, residue_names, anm = _structure(inputs.feature_residue_ids)
    return build_residue_pair_features(
        residue_ids=inputs.feature_residue_ids,
        k_ints=inputs.k_ints,
        heavy_contacts=inputs.heavy_contacts,
        acceptor_contacts=inputs.acceptor_contacts,
        coordinates=coordinates,
        residue_names=residue_names,
        anm_covariance=anm,
        reference_bc=REFERENCE_BC,
        reference_bh=REFERENCE_BH,
    )


def _build_prior(
    method: str,
    inputs: common.EnsembleInputs,
    mapping: np.ndarray,
    projection: np.ndarray,
    marginal_weights: np.ndarray,
    *,
    model=None,
    pair_features: np.ndarray | None = None,
) -> FrozenPeptidePrior:
    reference_logpf = inputs.log_pf_by_frame(REFERENCE_BC, REFERENCE_BH)
    uniform = jnp.full(inputs.n_frames, 1.0 / inputs.n_frames)
    unweighted_covariance = peptide_logpf_covariance(reference_logpf, mapping, uniform)
    if method == "mean_only":
        return FrozenPeptidePrior(kind="mean_only")
    if method == "shape":
        return FrozenPeptidePrior(kind="shape", correlation=correlation_of(unweighted_covariance))
    if method == "unlearned":
        return FrozenPeptidePrior(
            kind="unlearned",
            geometry=scale_free_log_spd(unweighted_covariance, projection, ALPHA),
            relative_variance=relative_log_variance(unweighted_covariance, marginal_weights, ALPHA),
        )
    if method in {"marginal_oracle", "conditional_oracle", "oracle"}:
        residue_covariance = weighted_population_covariance(
            reference_logpf, inputs.reference_weights
        )
        peptide_covariance = jnp.asarray(mapping) @ residue_covariance @ jnp.asarray(mapping).T
        if method in {"marginal_oracle", "conditional_oracle"}:
            profile_index = 1 if method == "marginal_oracle" else 3
            profile = covariance_profiles_from_covariance(peptide_covariance, ALPHA)[
                profile_index
            ]
            return FrozenPeptidePrior(
                kind=method,
                profile=profile,
                profile_weights=marginal_weights,
            )
        return FrozenPeptidePrior(
            kind="oracle",
            geometry=scale_free_log_spd(peptide_covariance, projection, ALPHA),
            relative_variance=relative_log_variance(peptide_covariance, marginal_weights, ALPHA),
        )
    if method not in {"point", "family"} or model is None or pair_features is None:
        raise ValueError(f"cannot build MoPrP prior {method!r}")
    center, modes = model.predict(pair_features, inputs.feature_residue_ids.size)
    geometry, geometry_modes, marginal, marginal_modes = peptide_tangent_prior(
        center, modes, mapping, projection, marginal_weights, ALPHA
    )
    return FrozenPeptidePrior(
        kind=method,
        geometry=geometry,
        geometry_modes=geometry_modes,
        relative_variance=marginal,
        marginal_modes=marginal_modes,
        score_precision=model.score_precision,
    )


def _mse(
    inputs: common.EnsembleInputs,
    weights: np.ndarray,
    bc: float,
    bh: float,
    mapping: np.ndarray,
    observed: np.ndarray,
    time_indices: np.ndarray,
) -> float:
    prediction = np.asarray(
        predict_uptake(
            inputs.log_pf_by_frame(bc, bh),
            inputs.k_ints,
            inputs.timepoints,
            mapping,
            weights,
        )
    ).T
    return float(np.mean(np.square(prediction[:, time_indices] - observed[:, time_indices])))


def _state_metrics(weights: np.ndarray, inputs: common.EnsembleInputs) -> dict[str, float]:
    populations = np.asarray(state_populations(weights, inputs.states, inputs.support))
    result = {
        "recovery_pct": float(
            strict_recovery_percent(weights, inputs.states, inputs.support, inputs.targets)
        ),
        "decoy_mass": float(
            sum(populations[index] for index, state in enumerate(inputs.support) if state not in TARGET_STATES)
        ),
        "ess": float(1.0 / np.square(weights).sum()),
    }
    for index, state in enumerate(inputs.support):
        result[f"population_{state}"] = float(populations[index])
        mask = inputs.states == state
        mass = float(weights[mask].sum())
        result[f"within_state_ess_{state}"] = (
            float(conditional_subset_effective_sample_size(weights, mask)) if mass > 1e-12 else 0.0
        )
    return result


def _load_artifacts(directory: Path):
    models, manifests = {}, {}
    for method in ("point", "family"):
        models[method], manifests[method] = load_linear_geometry_model(
            directory / f"{method}_artifact", require_iso_provenance=True
        )
    primary = manifests["point"].get("iso_primary_method")
    if primary != manifests["family"].get("iso_primary_method"):
        raise ValueError("point and family artifacts disagree on the ISO primary method")
    return models, manifests, primary


def evaluate_transfer(frame: pd.DataFrame, primary_method: str) -> dict:
    folds = frame[frame.fold != "full"].copy()
    baselines = folds[folds.method == "mean_only"][["fold", "ensemble", "val_mean_mse"]].rename(
        columns={"val_mean_mse": "mean_only_mse"}
    )
    shape = folds[folds.method == "shape"][["fold", "ensemble", "recovery_pct", "decoy_mass"]].rename(
        columns={"recovery_pct": "shape_recovery", "decoy_mass": "shape_decoy"}
    )
    primary = folds[folds.method == primary_method].merge(
        baselines, on=["fold", "ensemble"]
    ).merge(shape, on=["fold", "ensemble"])
    primary["recovery_gain"] = primary.recovery_pct - primary.shape_recovery
    primary["decoy_change"] = primary.decoy_mass - primary.shape_decoy
    primary["mean_ratio"] = primary.val_mean_mse / primary.mean_only_mse.clip(lower=1e-12)
    ensemble_rows = []
    for ensemble, group in primary.groupby("ensemble"):
        ensemble_rows.append(
            {
                "ensemble": ensemble,
                "median_recovery_gain": float(group.recovery_gain.median()),
                "median_decoy_change": float(group.decoy_change.median()),
                "median_mean_ratio": float(group.mean_ratio.median()),
                "positive_folds": int((group.recovery_gain > 0).sum()),
            }
        )
    supported = bool(
        len(ensemble_rows) == 2
        and all(row["median_recovery_gain"] > 0 for row in ensemble_rows)
        and all(row["median_decoy_change"] < 0 for row in ensemble_rows)
        and all(row["median_mean_ratio"] <= 1.05 for row in ensemble_rows)
        and all(row["positive_folds"] >= 4 for row in ensemble_rows)
    )
    learning_adds_value = False
    if primary_method in {"point", "family"}:
        learned = folds[folds.method == primary_method]
        unlearned = folds[folds.method == "unlearned"]
        merged = learned.merge(
            unlearned[["fold", "ensemble", "recovery_pct"]],
            on=["fold", "ensemble"],
            suffixes=("_learned", "_unlearned"),
        )
        learning_adds_value = bool(
            all(
                group.recovery_pct_learned.median() > group.recovery_pct_unlearned.median()
                for _, group in merged.groupby("ensemble")
            )
        )
    return {
        "primary_method": primary_method,
        "transfer_supported": supported,
        "learning_adds_value": learning_adds_value,
        "ensemble_results": ensemble_rows,
    }


def write_validation_summary(frame: pd.DataFrame, decision: dict, output_dir: Path) -> None:
    """Write compact transfer results and a method-level recovery plot."""

    lines = [
        "# MoPrP external validation of the ISO joint-geometry prior",
        "",
        f"ISO-selected primary method: `{decision['primary_method']}`.",
        f"Transfer supported: `{decision.get('transfer_supported', 'smoke only')}`.",
        f"Learning adds value over the unlearned prior: "
        f"`{decision.get('learning_adds_value', 'smoke only')}`.",
        "",
        "MoPrP labels are used only for evaluation. The marginal, conditional, and full "
        "oracle methods are non-transferable reference ceilings.",
    ]
    if decision.get("ensemble_results"):
        lines.extend(
            [
                "",
                "| Ensemble | Median recovery gain (pp) | Median decoy change | "
                "Median mean-MSE ratio | Positive folds |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in decision["ensemble_results"]:
            lines.append(
                f"| {row['ensemble']} | {row['median_recovery_gain']:.3f} | "
                f"{row['median_decoy_change']:.4f} | {row['median_mean_ratio']:.4f} | "
                f"{row['positive_folds']} |"
            )
    (output_dir / "moprp_validation_summary.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    plot_frame = frame[
        (frame.fold != "full")
        & frame.method.isin(["mean_only", "shape", "unlearned", "point", "family"])
    ]
    if plot_frame.empty:
        plot_frame = frame[
            frame.method.isin(["mean_only", "shape", "unlearned", "point", "family"])
        ]
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
    axis.set_ylabel("Median MoPrP recovery (%)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "moprp_recovery_effect_sizes.png", dpi=180)
    plt.close(figure)


def _finalize_validation(
    frame: pd.DataFrame,
    weight_arrays: dict[str, np.ndarray],
    primary_method: str,
    args: argparse.Namespace,
) -> dict:
    frame.to_csv(args.output_dir / "moprp_validation_raw.csv", index=False)
    np.savez_compressed(args.output_dir / "moprp_fitted_weights.npz", **weight_arrays)
    decision = (
        evaluate_transfer(frame, primary_method)
        if not args.smoke
        else {"primary_method": primary_method, "status": "smoke_only"}
    )
    decision["iso_artifact_dir"] = str(args.artifact_dir)
    decision["moprp_input_hashes"] = common.input_hashes()
    (args.output_dir / "transfer_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True)
    )
    write_validation_summary(frame, decision, args.output_dir)
    return decision


def merge_validation_shards(
    primary_method: str, args: argparse.Namespace
) -> dict:
    csv_paths = [
        args.output_dir
        / f"moprp_validation_raw.shard_{index:04d}_of_{args.num_shards:04d}.csv"
        for index in range(args.num_shards)
    ]
    npz_paths = [
        args.output_dir
        / f"moprp_fitted_weights.shard_{index:04d}_of_{args.num_shards:04d}.npz"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in (*csv_paths, *npz_paths) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing MoPrP validation shards: {missing}")
    frame = pd.concat([pd.read_csv(path, dtype={"fold": str}) for path in csv_paths], ignore_index=True)
    if frame.duplicated(["fold", "method", "ensemble"]).any():
        raise ValueError("MoPrP validation shards contain duplicate jobs")
    arrays: dict[str, np.ndarray] = {}
    for path in npz_paths:
        with np.load(path) as shard:
            duplicates = set(arrays).intersection(shard.files)
            if duplicates:
                raise ValueError(f"duplicate fitted-weight keys across shards: {sorted(duplicates)}")
            arrays.update({key: np.asarray(shard[key]) for key in shard.files})
    return _finalize_validation(frame, arrays, primary_method, args)


def run(args: argparse.Namespace) -> None:
    prohibit_failed_stage_j_moprp_launch()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_index < num_shards")
    models, manifests, primary_method = _load_artifacts(args.artifact_dir)
    if not args.smoke and any(bool(manifest.get("smoke")) for manifest in manifests.values()):
        raise ValueError(
            "production MoPrP validation refuses ISO smoke artifacts; rerun ISO without --smoke"
        )
    if primary_method not in {"unlearned", "point", "family"}:
        raise ValueError(f"invalid ISO primary method {primary_method!r}")
    if args.merge_shards:
        decision = merge_validation_shards(primary_method, args)
        print(json.dumps(decision, indent=2))
        return
    inputs = {name: common.load_ensemble_inputs(name) for name in common.ENSEMBLES}
    keep = np.ones(next(iter(inputs.values())).mapping.shape[0], dtype=bool)
    keep[common.PEPTIDE1_INDEX] = False
    features, ood = {}, {}
    for ensemble, item in inputs.items():
        features[ensemble], names = _features(item)
        for method, model in models.items():
            if tuple(names) != model.feature_names:
                raise ValueError(f"{ensemble}: MoPrP and ISO feature schemas differ")
            ood[f"{ensemble}:{method}"] = feature_out_of_distribution(features[ensemble], model)
    if args.shard_index == 0:
        (args.output_dir / "feature_ood.json").write_text(
            json.dumps(ood, indent=2, sort_keys=True)
        )

    folds = time_folds(next(iter(inputs.values())).timepoints.size)
    fold_specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        (str(index), np.setdiff1d(np.arange(15), held), held) for index, held in enumerate(folds)
    ]
    fold_specs.append(("full", np.arange(15), np.arange(15)))
    if args.smoke:
        fold_specs = fold_specs[-1:]
    steps = min(args.steps, 20) if args.smoke else args.steps
    starts = min(args.starts, 1) if args.smoke else args.starts
    rows, weight_arrays = [], {}
    prior_cache = {}
    job_index = 0
    for fold_name, train_times, val_times in fold_specs:
        for method in METHODS:
            selected_for_shard = job_index % args.num_shards == args.shard_index
            job_index += 1
            if not selected_for_shard:
                continue
            cells = {}
            for ensemble, item in inputs.items():
                mapping = item.mapping[keep]
                observed = item.observed_uptake[keep]
                projection = np.asarray(overlap_projection(mapping))
                overlap_weights = np.asarray(inverse_overlap_degree_weights(mapping))
                cache_key = (method, ensemble)
                if cache_key not in prior_cache:
                    prior_cache[cache_key] = _build_prior(
                        method,
                        item,
                        mapping,
                        projection,
                        overlap_weights,
                        model=models.get(method),
                        pair_features=features[ensemble],
                    )
                uniform = np.full(item.n_frames, 1.0 / item.n_frames)
                mean_reference = _mse(
                    item, uniform, REFERENCE_BC, REFERENCE_BH, mapping, observed, train_times
                )
                cells[ensemble] = ReweightingCell(
                    name=ensemble,
                    heavy_contacts=item.heavy_contacts,
                    acceptor_contacts=item.acceptor_contacts,
                    k_ints=item.k_ints,
                    timepoints=item.timepoints,
                    mapping=mapping,
                    observed_uptake=observed,
                    train_time_indices=train_times,
                    projection=projection,
                    marginal_weights=overlap_weights,
                    mean_reference=mean_reference,
                    prior=prior_cache[cache_key],
                )
            marginal_strength = (
                float(manifests[method]["selected_marginal_strength"])
                if method in manifests and "selected_marginal_strength" in manifests[method]
                else 1.0
            )
            if method == "unlearned":
                selection_path = args.artifact_dir / "selected_reweighting_settings.json"
                selection = json.loads(selection_path.read_text())
                marginal_strength = float(selection["per_method"]["unlearned"]["marginal_strength"])
            score_strength = float(manifests["family"].get("score_penalty", 1.0))
            result = optimize_joint_reweighting(
                cells,
                marginal_strength=marginal_strength,
                score_strength=score_strength,
                steps=steps,
                starts=starts,
                seed=20260722 + (0 if fold_name == "full" else int(fold_name)),
            )
            for ensemble, item in inputs.items():
                weights = result["weights"][ensemble]
                mapping = item.mapping[keep]
                observed = item.observed_uptake[keep]
                val_mean = _mse(
                    item, weights, result["bc"], result["bh"], mapping, observed, val_times
                )
                peptide1_mean = _mse(
                    item,
                    weights,
                    result["bc"],
                    result["bh"],
                    item.mapping[[common.PEPTIDE1_INDEX]],
                    item.observed_uptake[[common.PEPTIDE1_INDEX]],
                    val_times,
                )
                rows.append(
                    {
                        "fold": fold_name,
                        "method": method,
                        "ensemble": ensemble,
                        "marginal_strength": marginal_strength,
                        "objective": result["objective"],
                        "best_start": result["best_start"],
                        "bc": result["bc"],
                        "bh": result["bh"],
                        "val_mean_mse": val_mean,
                        "peptide1_mse": peptide1_mean,
                        **result["components"],
                        **_state_metrics(weights, item),
                    }
                )
                weight_arrays[f"{method}__fold_{fold_name}__{ensemble}"] = weights
    frame = pd.DataFrame(rows)
    if args.num_shards == 1:
        decision = _finalize_validation(frame, weight_arrays, primary_method, args)
        print(json.dumps(decision, indent=2))
        print(f"wrote MoPrP external validation under {args.output_dir}")
    else:
        csv_path = args.output_dir / (
            f"moprp_validation_raw.shard_{args.shard_index:04d}_of_{args.num_shards:04d}.csv"
        )
        npz_path = args.output_dir / (
            f"moprp_fitted_weights.shard_{args.shard_index:04d}_of_{args.num_shards:04d}.npz"
        )
        frame.to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, **weight_arrays)
        print(f"wrote MoPrP validation shard to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "_moprp_joint_geometry_validation")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
