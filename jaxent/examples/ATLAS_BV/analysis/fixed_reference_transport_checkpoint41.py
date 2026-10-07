"""Checkpoint 41: fixed-reference and self-distance population transport."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-cp41-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis import (
    corrected_variance_recovery_checkpoint36 as cp36,
)
from jaxent.examples.ATLAS_BV.analysis import (
    replica_sampling_mechanism_checkpoint40 as cp40,
)
from jaxent.examples.ATLAS_BV.analysis import (
    sparse_population_extrapolation_checkpoint39 as cp39,
)
from jaxent.examples.ATLAS_BV.analysis.common import HERE
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    entropy_contributions,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)


OUTPUT = (
    HERE / "outputs/analysis/pairwise_geometry/checkpoint41_fixed_reference_transport"
)
LABEL_COUNTS = (2, 4, 8, 16, 32)
TRANSPORTS = ("cross", "target_self", "source_common", "joint_common")
PF_RAW = "pf_l1"
PF_FIXED = "pf_l1_fixed_origin"
WORK_BASE = "work_opt"
WORK_PRIMARY = "work_opt_fixed_mean_pool_then_transform"
REPRESENTATIONS = (
    PF_RAW,
    PF_FIXED,
    WORK_BASE,
    "work_opt_dynamic_pool_then_transform",
    "work_opt_fixed_mean_frame_then_pool",
    WORK_PRIMARY,
    "work_opt_fixed_profile_frame_then_pool",
    "work_opt_fixed_profile_pool_then_transform",
)
SEED = 20260927
INVARIANCE_TOLERANCE = 1.0e-10


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def fixed_work_opt(z: np.ndarray, center: float | np.ndarray) -> np.ndarray:
    """Work Opt using a fixed scalar or residue-profile reference."""
    values = np.atleast_2d(np.asarray(z, dtype=float))
    reference = np.asarray(center, dtype=float)
    if reference.ndim == 1:
        reference = reference[:, None]
    shape = np.abs(values - reference)
    return shape - entropy_contributions(shape, "legacy_zq")


def dynamic_work_opt(z: np.ndarray) -> np.ndarray:
    return np.asarray(cp36.work_representations(z)["work_opt"][0], dtype=float)


def reference_profile(system: str, analysed_frames: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load the shared t=0 BV profile and prove it is external to analysis."""
    profiles = []
    paths = []
    for replica in cp39.REPLICAS:
        path = cp36.feature_folder(cp36.OUTPUT, system, replica) / "features.npz"
        with np.load(path, allow_pickle=False) as archive:
            profile = (
                cp36.PROTOCOL["bv_bc"] * np.asarray(archive["heavy_contacts"][:, 0])
                + cp36.PROTOCOL["bv_bh"]
                * np.asarray(archive["acceptor_contacts"][:, 0])
            )
        profiles.append(np.asarray(profile, dtype=float))
        paths.append(path)
    if not all(np.array_equal(profiles[0], profile) for profile in profiles[1:]):
        raise ValueError(f"{system}: replica t=0 BV profiles are not identical")
    if np.any(np.asarray(analysed_frames, dtype=int) <= 0):
        raise ValueError(f"{system}: the external frame entered an analysed subset")
    return profiles[0], {
        "system_id": system,
        "reference_frame": 0,
        "minimum_analysed_frame": int(np.min(analysed_frames)),
        "replica_profiles_exact": True,
        "reference_profile_sha256": hashlib.sha256(profiles[0].tobytes()).hexdigest(),
        "reference_inputs_json": json.dumps(
            {str(path): cp36.digest(path) for path in paths}, sort_keys=True
        ),
    }


def descriptor_bundle(
    z: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    z0: np.ndarray,
) -> dict[str, np.ndarray]:
    """All preregistered representations for one frame subset."""
    z = np.asarray(z, dtype=float)
    indices = np.asarray(indices, dtype=int)
    zbar = cp39.region_descriptor(z, weights, indices)
    mu0 = float(np.mean(z0))
    frame_dynamic = dynamic_work_opt(z)
    frame_fixed_mean = fixed_work_opt(z, mu0)
    frame_fixed_profile = fixed_work_opt(z, z0)
    return {
        "__zbar": zbar,
        PF_RAW: zbar,
        PF_FIXED: zbar - z0[:, None],
        WORK_BASE: cp39.region_descriptor(frame_dynamic, weights, indices),
        "work_opt_dynamic_pool_then_transform": dynamic_work_opt(zbar),
        "work_opt_fixed_mean_frame_then_pool": cp39.region_descriptor(
            frame_fixed_mean, weights, indices
        ),
        WORK_PRIMARY: fixed_work_opt(zbar, mu0),
        "work_opt_fixed_profile_frame_then_pool": cp39.region_descriptor(
            frame_fixed_profile, weights, indices
        ),
        "work_opt_fixed_profile_pool_then_transform": fixed_work_opt(zbar, z0),
    }


def pooled_bundle(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    left_log_mass: np.ndarray,
    right_log_mass: np.ndarray,
    z0: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pool A/B frames before nonlinear region-level Work transformations."""
    zbar = cp39.pooled_descriptor(
        left["__zbar"], right["__zbar"], left_log_mass, right_log_mass
    )
    result = {
        "__zbar": zbar,
        PF_RAW: zbar,
        PF_FIXED: zbar - z0[:, None],
    }
    for name in (
        WORK_BASE,
        "work_opt_fixed_mean_frame_then_pool",
        "work_opt_fixed_profile_frame_then_pool",
    ):
        result[name] = cp39.pooled_descriptor(
            left[name], right[name], left_log_mass, right_log_mass
        )
    result["work_opt_dynamic_pool_then_transform"] = dynamic_work_opt(zbar)
    result[WORK_PRIMARY] = fixed_work_opt(zbar, float(np.mean(z0)))
    result["work_opt_fixed_profile_pool_then_transform"] = fixed_work_opt(zbar, z0)
    return result


def distance_modes(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    source_common: dict[str, np.ndarray],
    joint_common: dict[str, np.ndarray],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        transport: {} for transport in TRANSPORTS
    }
    for representation in REPRESENTATIONS:
        source_fit = cp39.pair_distance(source[representation])
        output["cross"][representation] = (
            source_fit,
            cp39.pair_distance(target[representation], source[representation]),
        )
        output["target_self"][representation] = (
            source_fit,
            cp39.pair_distance(target[representation]),
        )
        common = cp39.pair_distance(source_common[representation])
        output["source_common"][representation] = (common, common)
        joint = cp39.pair_distance(joint_common[representation])
        output["joint_common"][representation] = (joint, joint)
    return output


def assert_pf_invariance(
    system: str,
    transfer: str,
    matrices: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> float:
    maximum = 0.0
    for transport in TRANSPORTS:
        for raw, fixed in zip(
            matrices[transport][PF_RAW], matrices[transport][PF_FIXED], strict=True
        ):
            maximum = max(maximum, float(np.max(np.abs(raw - fixed))))
    if maximum > INVARIANCE_TOLERANCE:
        raise ValueError(
            f"{system} {transfer}: PF fixed-origin invariance failed ({maximum})"
        )
    # The reconstruction deliberately has deterministic tie-breaking.  Sub-ulp
    # translation noise can otherwise choose a different tied solution despite
    # mathematically identical L1 geometry.  Once equivalence is audited, use
    # exact copies so the negative control also remains identical downstream.
    for transport in TRANSPORTS:
        matrices[transport][PF_FIXED] = tuple(
            value.copy() for value in matrices[transport][PF_RAW]
        )
    return maximum


def evaluate_transfer(
    system: str,
    transfer: str,
    matrices: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    source_values: np.ndarray,
    target_values: np.ndarray,
    label_counts: tuple[int, ...],
    repeats: int,
) -> pd.DataFrame:
    rows = []
    orders = cp39.labels_for_repeat(len(source_values), repeats, system)
    for transport, representations in matrices.items():
        for representation, (fit_distance, query_distance) in representations.items():
            for repeat, order in enumerate(orders):
                for known in label_counts:
                    if not 1 < known < len(source_values):
                        continue
                    labels = np.sort(order[:known])
                    held = np.setdiff1d(
                        np.arange(len(target_values)), labels, assume_unique=True
                    )
                    prediction, alpha, degenerate = cp39.anchored_prediction(
                        fit_distance, query_distance, source_values, labels
                    )
                    rows.append(
                        {
                            "system_id": system,
                            "transfer": transfer,
                            "transport": transport,
                            "representation": representation,
                            "known": known,
                            "repeat": repeat,
                            "heldout": len(held),
                            "alpha": alpha,
                            "degenerate_alpha": degenerate,
                            "spearman": cp39.finite_spearman(
                                target_values[held], prediction[held]
                            ),
                            "distribution_recovery": cp39.md_distribution_recovery(
                                target_values[held], prediction[held], source_values
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    return frame.groupby(
        ["system_id", "transfer", "transport", "representation", "known"],
        as_index=False,
    ).agg(
        repetitions=("repeat", "nunique"),
        heldout=("heldout", "mean"),
        alpha=("alpha", "mean"),
        degenerate_alpha_fraction=("degenerate_alpha", "mean"),
        spearman=("spearman", "mean"),
        distribution_recovery=("distribution_recovery", "mean"),
    )


def temporal_analysis(
    row: dict,
    data: dict,
    z0: np.ndarray,
    landmarks: int,
    repeats: int,
    label_counts: tuple[int, ...],
) -> tuple[pd.DataFrame, list[dict]]:
    c = np.flatnonzero(data["replicas"] == 3)
    c = c[np.argsort(data["frames"][c], kind="stable")]
    midpoint = len(c) // 2
    halves = {"C1": c[:midpoint], "C2": c[midpoint:]}
    rows, audits = [], []
    for source_name, target_name in (("C1", "C2"), ("C2", "C1")):
        source, target = halves[source_name], halves[target_name]
        order = cp39.maximin_order(data["structural"][np.ix_(source, source)])
        centers = source[order[: min(landmarks, len(source))]]
        bandwidth = cp39.neighbour_bandwidth(
            data["structural"][np.ix_(source, source)], 10
        )
        source_values, source_weights, _ = cp40.local_estimates(
            data["structural"], centers, source, bandwidth, True
        )
        target_values, target_weights, _ = cp40.local_estimates(
            data["structural"], centers, target, bandwidth, False
        )
        _, joint_weights, _ = cp40.local_estimates(
            data["structural"], centers, np.concatenate((source, target)), bandwidth, True
        )
        source_desc = descriptor_bundle(data["logpf"], source, source_weights, z0)
        target_desc = descriptor_bundle(data["logpf"], target, target_weights, z0)
        joint_desc = descriptor_bundle(
            data["logpf"], np.concatenate((source, target)), joint_weights, z0
        )
        matrices = distance_modes(source_desc, target_desc, source_desc, joint_desc)
        transfer = f"{source_name}_to_{target_name}"
        maximum = assert_pf_invariance(row["system_id"], transfer, matrices)
        audits.append(
            {
                "system_id": row["system_id"],
                "transfer": transfer,
                "pf_invariance_max_abs_distance": maximum,
            }
        )
        rows.append(
            evaluate_transfer(
                row["system_id"],
                transfer,
                matrices,
                source_values,
                target_values,
                label_counts,
                repeats,
            )
        )
    return pd.concat(rows, ignore_index=True), audits


def ab_to_c_analysis(
    row: dict,
    data: dict,
    z0: np.ndarray,
    landmarks: int,
    repeats: int,
    label_counts: tuple[int, ...],
) -> tuple[pd.DataFrame, dict]:
    replicas = data["replicas"]
    structural = data["structural"]
    centers = cp39.maximin_order(structural)[: min(landmarks, len(structural))]
    a = np.flatnonzero(replicas == 1)
    ab = np.flatnonzero(np.isin(replicas, (1, 2)))
    abc = np.arange(len(replicas))
    bandwidth = cp39.neighbour_bandwidth(structural[np.ix_(a, a)], 10)
    targets, weights, _ = cp39.local_targets_and_weights(
        structural, centers, replicas, bandwidth
    )
    replica_desc = {}
    for replica in cp39.REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        replica_desc[replica] = descriptor_bundle(
            data["logpf"], indices, weights[replica], z0
        )
    source = pooled_bundle(
        replica_desc[1], replica_desc[2], targets[1], targets[2], z0
    )
    target = replica_desc[3]
    target_ab = np.logaddexp(targets[1], targets[2]) - np.log(2.0)
    _, ab_weights, _ = cp40.local_estimates(
        structural, centers, ab, bandwidth, True
    )
    _, abc_weights, _ = cp40.local_estimates(
        structural, centers, abc, bandwidth, True
    )
    ab_common = descriptor_bundle(data["logpf"], ab, ab_weights, z0)
    abc_common = descriptor_bundle(data["logpf"], abc, abc_weights, z0)
    matrices = distance_modes(source, target, ab_common, abc_common)
    maximum = assert_pf_invariance(row["system_id"], "AB_to_C", matrices)
    result = evaluate_transfer(
        row["system_id"],
        "AB_to_C",
        matrices,
        target_ab,
        targets[3],
        label_counts,
        repeats,
    )
    return result, {
        "system_id": row["system_id"],
        "transfer": "AB_to_C",
        "pf_invariance_max_abs_distance": maximum,
    }


def analyse_system(task: tuple) -> str:
    row, phases, landmarks, repeats, label_counts, parts = task
    system = row["system_id"]
    data = cp40.system_arrays(row)
    z0, reference_audit = reference_profile(system, data["frames"])
    audits = []
    if "temporal" in phases:
        temporal, temporal_audits = temporal_analysis(
            row, data, z0, landmarks, repeats, label_counts
        )
        atomic_parquet(temporal, parts / f"{system}.temporal.parquet")
        audits.extend(temporal_audits)
    if "transfer" in phases:
        transfer, transfer_audit = ab_to_c_analysis(
            row, data, z0, landmarks, repeats, label_counts
        )
        atomic_parquet(transfer, parts / f"{system}.transfer.parquet")
        audits.append(transfer_audit)
    audit = pd.DataFrame([{**reference_audit, **item} for item in audits])
    atomic_parquet(audit, parts / f"{system}.audit.parquet")
    return system


def paired_interval(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(
        ["transfer", "representation", "transport", "known"], as_index=False
    ).agg(
        median_spearman=("spearman", "median"),
        median_recovery=("distribution_recovery", "median"),
        median_alpha=("alpha", "median"),
        systems=("system_id", "nunique"),
    )


def contrast(
    frame: pd.DataFrame,
    test_representation: str,
    test_transport: str,
    reference_representation: str,
    reference_transport: str,
    comparison: str,
    bootstrap: int,
) -> pd.DataFrame:
    selected = frame[
        (
            (frame.representation == test_representation)
            & (frame.transport == test_transport)
        )
        | (
            (frame.representation == reference_representation)
            & (frame.transport == reference_transport)
        )
    ].copy()
    selected["arm"] = np.where(
        (selected.representation == test_representation)
        & (selected.transport == test_transport),
        "test",
        "reference",
    )
    rows = []
    for (transfer, known), block in selected.groupby(["transfer", "known"]):
        pivot = block.pivot_table(
            index="system_id", columns="arm", values=["spearman", "distribution_recovery"]
        ).dropna()
        if not len(pivot) or not {"test", "reference"}.issubset(pivot.columns.levels[1]):
            continue
        delta = pivot[("spearman", "test")] - pivot[("spearman", "reference")]
        low, high = paired_interval(
            delta,
            stable_seed(comparison, transfer, known),
            bootstrap,
        )
        rows.append(
            {
                "comparison": comparison,
                "transfer": transfer,
                "known": known,
                "test_representation": test_representation,
                "test_transport": test_transport,
                "reference_representation": reference_representation,
                "reference_transport": reference_transport,
                "median_test_spearman": float(pivot[("spearman", "test")].median()),
                "median_reference_spearman": float(
                    pivot[("spearman", "reference")].median()
                ),
                "median_spearman_delta": float(np.median(delta)),
                "delta_ci_low": low,
                "delta_ci_high": high,
                "median_test_recovery": float(
                    pivot[("distribution_recovery", "test")].median()
                ),
                "systems": len(pivot),
            }
        )
    return pd.DataFrame(rows)


def contrast_summary(frame: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    specifications = []
    for representation in REPRESENTATIONS:
        specifications.append(
            (
                representation,
                "target_self",
                representation,
                "cross",
                f"target_self_vs_cross:{representation}",
            )
        )
    for representation in REPRESENTATIONS[3:]:
        specifications.append(
            (
                representation,
                "cross",
                WORK_BASE,
                "cross",
                f"representation_vs_original:{representation}",
            )
        )
    specifications.extend(
        [
            (
                WORK_PRIMARY,
                "cross",
                "work_opt_dynamic_pool_then_transform",
                "cross",
                "fixed_reference_increment_same_order",
            ),
            (
                "work_opt_dynamic_pool_then_transform",
                "cross",
                WORK_BASE,
                "cross",
                "pool_then_transform_vs_frame_then_pool",
            ),
            (
                WORK_PRIMARY,
                "cross",
                "work_opt_fixed_mean_frame_then_pool",
                "cross",
                "pooling_order_with_fixed_mean",
            ),
            (
                "work_opt_fixed_profile_pool_then_transform",
                "cross",
                "work_opt_fixed_profile_frame_then_pool",
                "cross",
                "pooling_order_with_fixed_profile",
            ),
            (
                WORK_PRIMARY,
                "cross",
                WORK_BASE,
                "source_common",
                "primary_fixed_vs_source_common",
            ),
            (
                WORK_PRIMARY,
                "target_self",
                WORK_BASE,
                "cross",
                "primary_fixed_self_vs_original",
            ),
            (
                PF_FIXED,
                "cross",
                PF_RAW,
                "cross",
                "pf_fixed_origin_invariance",
            ),
        ]
    )
    return pd.concat(
        [contrast(frame, *specification, bootstrap) for specification in specifications],
        ignore_index=True,
    )


def mechanism_decisions(
    summary: pd.DataFrame, contrasts: pd.DataFrame, primary_known: int
) -> pd.DataFrame:
    rows = []
    for transfer in sorted(summary.transfer.unique()):
        primary = summary[
            (summary.transfer == transfer)
            & (summary.known == primary_known)
            & (summary.representation == WORK_PRIMARY)
            & (summary.transport == "cross")
        ]
        reference_increment = contrasts[
            (contrasts.transfer == transfer)
            & (contrasts.known == primary_known)
            & (contrasts.comparison == "fixed_reference_increment_same_order")
        ]
        pooling_order = contrasts[
            (contrasts.transfer == transfer)
            & (contrasts.known == primary_known)
            & (contrasts.comparison == "pool_then_transform_vs_frame_then_pool")
        ]
        common = contrasts[
            (contrasts.transfer == transfer)
            & (contrasts.known == primary_known)
            & (contrasts.comparison == "primary_fixed_vs_source_common")
        ]
        if primary.empty or reference_increment.empty or pooling_order.empty or common.empty:
            continue
        rho = float(primary.iloc[0].median_spearman)
        reference_increment_low = float(reference_increment.iloc[0].delta_ci_low)
        pooling_order_low = float(pooling_order.iloc[0].delta_ci_low)
        versus_common = float(common.iloc[0].delta_ci_low)
        if rho >= 0.5 and reference_increment_low > 0 and versus_common > 0:
            reference_result = "strong_support"
        elif reference_increment_low > 0:
            reference_result = "partial_support"
        else:
            reference_result = "no_support"
        for representation in (PF_RAW, WORK_BASE):
            self_row = contrasts[
                (contrasts.transfer == transfer)
                & (contrasts.known == primary_known)
                & (
                    contrasts.comparison
                    == f"target_self_vs_cross:{representation}"
                )
            ]
            self_supported = bool(
                len(self_row) and float(self_row.iloc[0].delta_ci_low) > 0
            )
            rows.append(
                {
                    "transfer": transfer,
                    "representation": representation,
                    "external_reference_result": reference_result
                    if representation == WORK_BASE
                    else "not_applicable_translation_invariant",
                    "primary_fixed_work_rho": rho
                    if representation == WORK_BASE
                    else np.nan,
                    "external_reference_increment_ci_low": reference_increment_low
                    if representation == WORK_BASE
                    else np.nan,
                    "pool_then_transform_ci_low": pooling_order_low
                    if representation == WORK_BASE
                    else np.nan,
                    "pool_then_transform_supported": pooling_order_low > 0
                    if representation == WORK_BASE
                    else False,
                    "target_self_supported": self_supported,
                    "target_self_delta_ci_low": float(self_row.iloc[0].delta_ci_low)
                    if len(self_row)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(summary: pd.DataFrame, destination: Path) -> None:
    transfers = ("C1_to_C2", "C2_to_C1", "AB_to_C")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    specifications = {
        "PF L1": (
            (PF_RAW, "cross", "original cross", "#1f77b4"),
            (PF_RAW, "target_self", "target self-distance", "#d62728"),
            (PF_RAW, "source_common", "source-common control", "#9467bd"),
        ),
        "Work Opt": (
            (WORK_BASE, "cross", "original cross", "#1f77b4"),
            (
                "work_opt_dynamic_pool_then_transform",
                "cross",
                "pool then transform",
                "#ff7f0e",
            ),
            (
                WORK_PRIMARY,
                "cross",
                "pool then transform + fixed mean",
                "#2ca02c",
            ),
            (WORK_BASE, "target_self", "target self-distance", "#d62728"),
            (WORK_BASE, "source_common", "source-common control", "#9467bd"),
        ),
    }
    for row, (title, arms) in enumerate(specifications.items()):
        for column, transfer in enumerate(transfers):
            axis = axes[row, column]
            for representation, transport, label, color in arms:
                block = summary[
                    (summary.transfer == transfer)
                    & (summary.representation == representation)
                    & (summary.transport == transport)
                ].sort_values("known")
                axis.plot(
                    block.known,
                    block.median_spearman,
                    marker="o",
                    label=label,
                    color=color,
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
            axis.set_title(f"{title}: {transfer.replace('_', ' ')}")
            axis.set_xticks(LABEL_COUNTS)
            if row == 1:
                axis.set_xlabel("Known neighbourhood populations")
            if column == 0:
                axis.set_ylabel("Held-out Spearman rho")
    handles, labels = axes[1, 2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center right", bbox_to_anchor=(1.14, 0.5))
    figure.suptitle("Checkpoint 41: fixed-reference and self-distance transfer")
    figure.tight_layout()
    figure.savefig(destination / "fixed_reference_learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_primary_summary(summary: pd.DataFrame, destination: Path) -> None:
    primary_known = 32 if 32 in set(summary.known) else int(summary.known.max())
    arms = (
        (PF_RAW, "cross", "PF original cross"),
        (PF_RAW, "target_self", "PF target self"),
        (WORK_BASE, "cross", "Work original cross"),
        (
            "work_opt_dynamic_pool_then_transform",
            "cross",
            "Work pool then transform",
        ),
        (WORK_PRIMARY, "cross", "Work pool + fixed mean"),
        (WORK_BASE, "source_common", "Work source-common"),
    )
    transfers = ("C1_to_C2", "C2_to_C1", "AB_to_C")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, transfer in zip(axes, transfers, strict=True):
        values = []
        labels = []
        for representation, transport, label in arms:
            block = summary[
                (summary.transfer == transfer)
                & (summary.known == primary_known)
                & (summary.representation == representation)
                & (summary.transport == transport)
            ]
            values.append(float(block.iloc[0].median_spearman))
            labels.append(label)
        positions = np.arange(len(values))
        axis.barh(positions, values, color=["#4c78a8", "#72b7b2", "#e45756", "#f2cf5b", "#54a24b", "#b279a2"])
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        axis.set_title(transfer.replace("_", " "))
        axis.set_yticks(positions)
        axis.set_xlabel("Median held-out rho")
        axis.invert_yaxis()
    axes[0].set_yticklabels(labels)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    figure.suptitle(f"Checkpoint 41: {primary_known}-label mechanism comparison")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination / "fixed_reference_32label_summary.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def report(destination: Path, systems: list[str], bootstrap: int) -> None:
    temporal = pd.concat(
        [pd.read_parquet(destination / "parts" / f"{system}.temporal.parquet") for system in systems],
        ignore_index=True,
    )
    transfer = pd.concat(
        [pd.read_parquet(destination / "parts" / f"{system}.transfer.parquet") for system in systems],
        ignore_index=True,
    )
    audits = pd.concat(
        [pd.read_parquet(destination / "parts" / f"{system}.audit.parquet") for system in systems],
        ignore_index=True,
    )
    combined = pd.concat([temporal, transfer], ignore_index=True)
    atomic_parquet(temporal, destination / "temporal_transfer.parquet")
    atomic_parquet(transfer, destination / "ab_to_c_transfer.parquet")
    atomic_parquet(audits, destination / "reference_audit.parquet")
    summary = cohort_summary(combined)
    contrasts = contrast_summary(combined, bootstrap)
    primary_known = 32 if 32 in set(summary.known) else int(summary.known.max())
    decisions = mechanism_decisions(summary, contrasts, primary_known)
    summary.to_csv(destination / "cohort_summary.csv", index=False)
    contrasts.to_csv(destination / "paired_contrasts.csv", index=False)
    decisions.to_csv(destination / "mechanism_decisions.csv", index=False)
    plot_learning_curves(summary, destination)
    plot_primary_summary(summary, destination)
    pf_rows = combined[
        combined.representation.isin((PF_RAW, PF_FIXED))
    ].pivot_table(
        index=["system_id", "transfer", "transport", "known"],
        columns="representation",
        values=["alpha", "spearman", "distribution_recovery"],
    )
    pf_result_max = float(
        np.nanmax(np.abs(pf_rows.xs(PF_RAW, axis=1, level=1) - pf_rows.xs(PF_FIXED, axis=1, level=1)))
    )
    if pf_result_max > INVARIANCE_TOLERANCE:
        raise ValueError(f"PF result invariance failed ({pf_result_max})")
    payload = {
        "checkpoint": 41,
        "systems": len(systems),
        "primary_known_populations": primary_known,
        "external_reference": "shared t=0 starting-structure BV profile",
        "pf_fixed_origin_max_abs_result_difference": pf_result_max,
        "reference_audit_passed": bool(
            audits.replica_profiles_exact.all()
            and audits.pf_invariance_max_abs_distance.max() <= INVARIANCE_TOLERANCE
        ),
        "decisions": decisions.to_dict(orient="records"),
        "variance_order_audit": {
            "checkpoint40_operation": "pool frames, find pooled structural neighbours, then compute local variance",
            "compute_then_pool_explanation_rejected": True,
        },
        "interpretation": [
            "A common origin cannot alter PF L1; the fixed-origin arm is an exact negative control.",
            "Target-self uses target features but no target population values.",
            "Source-common predicts in source geometry and is therefore a source-copy control, not direct alignment of target descriptors.",
            "Alpha is least-squares fitted; conclusions use held-out Spearman rho and MD distribution recovery, not MAE.",
        ],
        "provenance": {
            "analysis_sha256": cp36.digest(Path(__file__)),
            "bootstrap_samples": bootstrap,
            "seed": SEED,
        },
    }
    temporary = destination / "checkpoint41_report.yaml.tmp"
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(destination / "checkpoint41_report.yaml")


def valid_part(path: Path, required: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return bool(len(frame) and required.issubset(frame.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("temporal", "transfer", "report", "all"), default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--label-counts", default="2,4,8,16,32")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if min(args.workers, args.repeats, args.landmarks, args.bootstrap_samples) < 1:
        parser.error("all numeric settings must be positive")
    label_counts = tuple(sorted({int(value) for value in args.label_counts.split(",")}))
    rows = cp36.selected_rows("pilot", cp36.SINGLE_SYSTEM)
    if args.limit is not None:
        rows = rows[: args.limit]
    destination = args.output.resolve()
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    requested = (
        {"temporal", "transfer"}
        if args.phase == "all"
        else {args.phase}
        if args.phase != "report"
        else set()
    )
    required = {
        "temporal": {"system_id", "transfer", "transport", "representation", "spearman"},
        "transfer": {"system_id", "transfer", "transport", "representation", "spearman"},
    }
    tasks = []
    for row in rows:
        missing = {
            phase
            for phase in requested
            if not valid_part(parts / f"{row['system_id']}.{phase}.parquet", required[phase])
        }
        if missing:
            tasks.append((row, missing, args.landmarks, args.repeats, label_counts, parts))
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(analyse_system, task): task[0]["system_id"]
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), 1):
                print(f"[{index}/{len(tasks)}] {future.result()} complete", flush=True)
    if args.phase in ("all", "report"):
        systems = [row["system_id"] for row in rows]
        report(destination, systems, args.bootstrap_samples)
        print(f"report: {destination / 'checkpoint41_report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
