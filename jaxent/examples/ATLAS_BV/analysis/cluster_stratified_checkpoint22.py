"""Checkpoint 22: A-only structural clusters and within/between population recovery."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist, squareform
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.basin_census import (
    align_to_reference,
    load_ca_coordinates,
)
from jaxent.examples.ATLAS_BV.analysis.contact_difference_pilot_checkpoint20 import (
    contact_pair_features,
    safe_distance,
    tune_ridge,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    absolute_change_vectors,
    density_targets,
    frame_w1_signatures,
    mass_metrics,
    scalar_scale,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import (
    fixed_pair_features,
    load_backbone_dihedrals,
    pair_endpoints,
    periodic_dihedral_pair_distance,
    regularized_residue_variance,
    select_dihedral_zscore_models,
    select_information_models,
    training_circular_variance,
    variance_scaled_dihedral_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    OUTPUT as CP19_OUTPUT,
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    THERMODYNAMIC_METRICS,
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint22_cluster_stratified"
FIT, TUNE, TEST = 1, 2, 3
HDBSCAN_MIN = (30, 60, 120)
HDBSCAN_SAMPLES = (5, 10, 20)
K_VALUES = range(2, 7)


def pair_matrix(n: int, fn) -> np.ndarray:
    left, right = np.triu_indices(n, 1)
    values = fn(left, right)
    return squareform(values)


def select_hdbscan(distance: np.ndarray, space: str, system: str) -> dict:
    best = None
    for size in HDBSCAN_MIN:
        for samples in HDBSCAN_SAMPLES:
            candidate_distance = distance.copy()
            np.fill_diagonal(candidate_distance, 0.0)
            labels = HDBSCAN(
                min_cluster_size=size, min_samples=samples, metric="precomputed"
            ).fit_predict(candidate_distance)
            keep = labels >= 0
            clusters = np.unique(labels[keep])
            noise = float(np.mean(~keep))
            valid = (
                len(clusters) >= 2
                and all(np.sum(labels == c) >= 30 for c in clusters)
                and noise <= 0.5
            )
            if valid:
                retained_distance = distance[np.ix_(keep, keep)].copy()
                np.fill_diagonal(retained_distance, 0.0)
                sil = float(
                    silhouette_score(
                        retained_distance, labels[keep], metric="precomputed"
                    )
                )
            else:
                sil = np.nan
            score = (1 - noise) * max(sil, 0) if valid else -np.inf
            row = {
                "system_id": system,
                "algorithm": "hdbscan",
                "space": space,
                "min_cluster_size": size,
                "min_samples": samples,
                "clusters": len(clusters),
                "noise_fraction": noise,
                "silhouette": sil,
                "selection_score": score,
                "valid": valid,
                "labels": labels,
            }
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    return best


def select_kmeans(features: np.ndarray, space: str, system: str) -> dict:
    pca = PCA(n_components=0.95, svd_solver="full").fit(features)
    scores = pca.transform(features)
    best = None
    for k in K_VALUES:
        fitted = KMeans(n_clusters=k, n_init=20, random_state=20260904).fit(scores)
        counts = np.bincount(fitted.labels_)
        valid = counts.min() >= 0.1 * len(scores)
        sil = (
            float(
                silhouette_score(
                    scores,
                    fitted.labels_,
                    sample_size=min(500, len(scores)),
                    random_state=1,
                )
            )
            if valid
            else -np.inf
        )
        row = {
            "system_id": system,
            "algorithm": "kmeans",
            "space": space,
            "k": k,
            "clusters": k,
            "silhouette": sil,
            "valid": valid,
            "pca": pca,
            "model": fitted,
            "labels": fitted.labels_,
        }
        if best is None or row["silhouette"] > best["silhouette"]:
            best = row
    if not best["valid"]:
        fitted = KMeans(n_clusters=2, n_init=20, random_state=20260904).fit(scores)
        best = {
            "system_id": system,
            "algorithm": "kmeans",
            "space": space,
            "k": 2,
            "clusters": 2,
            "silhouette": float(silhouette_score(scores, fitted.labels_)),
            "valid": False,
            "pca": pca,
            "model": fitted,
            "labels": fitted.labels_,
        }
    return best


def structural_clusters(
    row: dict, config: dict, data: dict, angles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    coords, reps, _ = load_ca_coordinates(row, config)
    aligned = align_to_reference(coords)
    signatures = frame_w1_signatures(coords, 256)
    fit = np.flatnonzero(reps == FIT)
    k_candidates = [
        select_kmeans(aligned[fit].reshape(len(fit), -1), "ca_pca", data["system"]),
        select_kmeans(signatures[fit], "w1_pca", data["system"]),
    ]
    selected = max(k_candidates, key=lambda x: x["silhouette"])
    base = (
        aligned.reshape(len(aligned), -1)
        if selected["space"] == "ca_pca"
        else signatures
    )
    labels = selected["model"].predict(selected["pca"].transform(base))
    train_score = selected["pca"].transform(base[fit])
    centers = selected["model"].cluster_centers_
    radii = np.array(
        [
            np.quantile(
                np.linalg.norm(
                    train_score[selected["labels"] == c] - centers[c], axis=1
                ),
                0.99,
            )
            for c in range(selected["clusters"])
        ]
    )
    all_score = selected["pca"].transform(base)
    assigned_distance = np.linalg.norm(all_score - centers[labels], axis=1)
    supported = assigned_distance <= radii[labels]
    zvar, _ = regularized_residue_variance(
        training_circular_variance(angles, reps, FIT), 0.1
    )
    acoords = aligned[fit].reshape(len(fit), -1)
    asig = signatures[fit]
    aang = angles[:, fit]
    distances = {
        "ca_rms": cdist(acoords, acoords) / np.sqrt(aligned.shape[1]),
        "structural_w1": cdist(asig, asig, metric="cityblock") / asig.shape[1],
        "drmsd": pair_matrix(
            len(fit),
            lambda left, right: periodic_dihedral_pair_distance(aang, left, right),
        ),
        "z_drmsd": pair_matrix(
            len(fit),
            lambda left, right: variance_scaled_dihedral_pair_distance(
                aang, left, right, zvar, True
            ),
        ),
        "z_quadratic": pair_matrix(
            len(fit),
            lambda left, right: variance_scaled_dihedral_pair_distance(
                aang, left, right, zvar, False
            ),
        ),
    }
    audits = []
    for space, distance in distances.items():
        np.fill_diagonal(distance, 0.0)
        audits.append(select_hdbscan(distance, space, data["system"]))
    for candidate in k_candidates:
        audits.append(
            {k: v for k, v in candidate.items() if k not in {"pca", "model", "labels"}}
            | {"noise_fraction": 0.0, "selection_score": candidate["silhouette"]}
        )
    meta = {
        "selected_space": selected["space"],
        "clusters": selected["clusters"],
        "silhouette": selected["silhouette"],
        "supported_fraction": float(np.mean(supported)),
        "forced": not selected["valid"],
    }
    return labels, supported, aligned, audits, meta


def feature_set(
    data: dict,
    indices: dict,
    angles: np.ndarray,
    aligned: np.ndarray,
    target: dict,
) -> dict:
    features = fixed_pair_features(data, indices)
    for replica, pairs in data["pairs"].items():
        left, right = pair_endpoints(pairs, indices[replica])
        features["backbone_drmsd_control"][replica] = periodic_dihedral_pair_distance(
            angles, left, right
        )
        features.setdefault("fixed_pf_cosine", {})[replica] = safe_distance(
            data["z"], left, right, "cosine"
        )
        features.setdefault("fixed_pf_correlation", {})[replica] = safe_distance(
            data["z"], left, right, "correlation"
        )
        features.setdefault("pf_per_residue_ridge", {})[replica] = (
            absolute_change_vectors(data["z"], left, right)
        )
        delta_ca = aligned[left] - aligned[right]
        features.setdefault("ca_rms_control", {})[replica] = np.sqrt(
            np.mean(np.square(delta_ca), axis=(1, 2))
        )
    info, _ = select_information_models(data, indices, target)
    features.update(info)
    dihedral, _ = select_dihedral_zscore_models(data, angles, indices, target)
    features.update(dihedral)
    thermo = thermodynamic_pair_features(data["z"], data["pairs"])
    for name in THERMODYNAMIC_METRICS:
        features[name] = {r: thermo[name][r][indices[r]] for r in (1, 2, 3)}
    features.update(contact_pair_features(data, indices))
    return features


def relation(
    labels: np.ndarray,
    supported: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, np.ndarray]:
    common = supported[left] & supported[right]
    return {
        "within": labels[left] == labels[right],
        "between": labels[left] != labels[right],
        "common_within": common & (labels[left] == labels[right]),
        "common_between": common & (labels[left] != labels[right]),
        "novel": ~common,
    }


def analyse(row: dict, config: dict, edges: np.ndarray):
    data = system_data(row, config)
    targets, _ = density_targets(data, FIT, PRIMARY_RANK)
    indices = {
        r: sampled_indices(data["system"], r, len(v), PAIR_CAP)
        for r, v in targets.items()
    }
    y = {r: np.abs(v[indices[r]]) for r, v in targets.items()}
    angles, meta = load_backbone_dihedrals(row, config)
    labels, supported, aligned, audits, cluster_meta = structural_clusters(
        row, config, data, angles
    )
    features = feature_set(data, indices, angles, aligned, y)
    results = []
    fits = []
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    for name, x in features.items():
        xa = np.asarray(x[FIT])
        xb = np.asarray(x[TUNE])
        xc = np.asarray(x[TEST])
        if xa.ndim == 1:
            alpha = scalar_scale(xa, y[FIT], True)[0]
            pred = alpha * xc
            hp = alpha
        else:
            model, hp, scale = tune_ridge(xa, y[FIT], xb, y[TUNE], positive=True)
            pred = np.maximum(0, model.predict(xc / scale))
        fits.append(
            {
                "system_id": data["system"],
                "model": name,
                "parameter": hp,
                **cluster_meta,
                **meta,
            }
        )
        pairs = data["pairs"][TEST]
        left, right = pair_endpoints(pairs, indices[TEST])
        rel = relation(labels, supported, left, right)
        w1 = pairs.w1.to_numpy()[indices[TEST]]
        for kind, rmask in rel.items():
            for band in range(6):
                bmask = (w1 >= edges[band]) & (
                    (w1 < edges[band + 1]) if band < 5 else True
                )
                mask = rmask & bmask
                if (
                    mask.sum() < 30
                    or len(np.unique(np.r_[left[mask], right[mask]])) < 20
                ):
                    continue
                results.append(
                    {
                        "system_id": data["system"],
                        "model": name,
                        "relation": kind,
                        "band": f"q{band}",
                        "pairs": int(mask.sum()),
                        "unique_frames": len(np.unique(np.r_[left[mask], right[mask]])),
                        "mae": float(np.mean(np.abs(y[TEST][mask] - pred[mask]))),
                        **mass_metrics(
                            y[TEST][mask],
                            pred[mask],
                            y[FIT],
                            settings["distribution_bins"],
                            settings["distribution_smoothing"],
                        ),
                    }
                )
    return (
        results,
        [{k: v for k, v in a.items() if k != "labels"} for a in audits],
        fits,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true", help="reuse persisted pilot tables"
    )
    parser.add_argument(
        "--full", action="store_true", help="run all systems with the 1/2/3 assignment"
    )
    args = parser.parse_args()
    config = load_config()
    by = {r["system_id"]: r for r in load_systems()}
    selected = (
        pd.Series(list(by), name="system_id")
        if args.full
        else pd.read_parquet(CP19_OUTPUT / "pilot_systems.parquet").system_id
    )
    with open(
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
    ) as f:
        edges = np.asarray(yaml.safe_load(f)["edges_angstrom"])
    run_output = OUTPUT / "full_single_assignment" if args.full else OUTPUT
    run_output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        results = pd.read_parquet(run_output / "cluster_stratified_results.parquet")
        audits = pd.read_parquet(run_output / "cluster_audit.parquet")
    else:
        rr = []
        aa = []
        ff = []
        for i, s in enumerate(selected, 1):
            r, a, f = analyse(by[s], config, edges)
            rr += r
            aa += a
            ff += f
            print(f"[{i}/{len(selected)}] {s}", flush=True)
        results = pd.DataFrame(rr)
        audits = pd.DataFrame(aa)
        fits = pd.DataFrame(ff)
        atomic_parquet(results, run_output / "cluster_stratified_results.parquet")
        atomic_parquet(audits, run_output / "cluster_audit.parquet")
        atomic_parquet(fits, run_output / "cluster_metric_fits.parquet")
    summary = results.groupby(["model", "relation", "band"], as_index=False).agg(
        recovery=("distribution_recovery", "median"),
        recovery_sd=("distribution_recovery", "std"),
        systems=("system_id", "nunique"),
    )
    atomic_parquet(summary, run_output / "cluster_stratified_summary.parquet")
    top = [
        "work_scale",
        "work_shape",
        "work_density_legacy_zq",
        "absolute_l1",
        "l2",
        "fixed_pf_cosine",
        "fixed_pf_correlation",
        "pf_w1_raw",
        "information_quadratic",
        "contact_correlation",
        "structural_w1_control",
        "backbone_zquadratic_control",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, rel in zip(axes, ["within", "between"]):
        for name in top:
            selected_rows = summary[
                (summary.model == name) & (summary.relation == rel)
            ].set_index("band")
            x = np.array(
                [selected_rows.recovery.get(f"q{i}", np.nan) for i in range(6)]
            )
            sd = np.array(
                [selected_rows.recovery_sd.get(f"q{i}", np.nan) for i in range(6)]
            )
            ax.plot(
                range(6),
                100 * x,
                marker="o",
                label=name,
            )
            ax.fill_between(
                range(6),
                np.clip(100 * (x - sd), 0, 100),
                np.clip(100 * (x + sd), 0, 100),
                alpha=0.06,
            )
        ax.set_title(rel.title() + " cluster")
        band_counts = (
            results[results.relation == rel].groupby("band").system_id.nunique()
        )
        ax.set_xticks(
            range(6),
            [
                f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å\nn={band_counts.get(f'q{i}', 0)}"
                for i in range(6)
            ],
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run_output / "within_between_recovery.png", dpi=180)
    plt.close(fig)

    paired_rows = []
    rng = np.random.default_rng(20260904)
    for name in sorted(results.model.unique()):
        paired = (
            results[(results.model == name) & (results.band == "q5")]
            .pivot(
                index="system_id", columns="relation", values="distribution_recovery"
            )
            .dropna(subset=["within", "between"])
        )
        delta = paired.within.to_numpy() - paired.between.to_numpy()
        if len(delta):
            bootstrap = np.median(
                rng.choice(delta, size=(5000, len(delta)), replace=True), axis=1
            )
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            paired_rows.append(
                {
                    "model": name,
                    "systems": len(delta),
                    "within_median": float(paired.within.median()),
                    "between_median": float(paired.between.median()),
                    "median_paired_delta": float(np.median(delta)),
                    "bootstrap_95_low": float(low),
                    "bootstrap_95_high": float(high),
                }
            )
    paired_q5 = pd.DataFrame(paired_rows).sort_values("median_paired_delta")
    atomic_parquet(paired_q5, run_output / "q5_paired_cluster_contrasts.parquet")
    fig, ax = plt.subplots(figsize=(9, 8))
    y_position = np.arange(len(paired_q5))
    center = 100 * paired_q5.median_paired_delta.to_numpy()
    low = 100 * paired_q5.bootstrap_95_low.to_numpy()
    high = 100 * paired_q5.bootstrap_95_high.to_numpy()
    ax.errorbar(
        center,
        y_position,
        xerr=np.vstack([center - low, high - center]),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(
        y_position,
        [
            f"{name} (n={count})"
            for name, count in zip(paired_q5.model, paired_q5.systems)
        ],
    )
    ax.set_xlabel("Paired q5 recovery change: within − between (percentage points)")
    ax.set_title("Same-system q5 structural-cluster contrast\n95% system bootstrap CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_output / "q5_paired_cluster_contrasts.png", dpi=180)
    plt.close(fig)
    valid = audits[(audits.algorithm == "hdbscan") & audits.valid].system_id.nunique()
    overall_feasible = (
        (results.groupby(["system_id", "relation"]).size().unstack(fill_value=0) > 0)
        .all(axis=1)
        .sum()
    )
    q5_rows = results[results.band == "q5"]
    q5_within = set(q5_rows[q5_rows.relation == "within"].system_id)
    q5_between = set(q5_rows[q5_rows.relation == "between"].system_id)
    tail_feasible = len(q5_within & q5_between)
    tail_counts = {
        relation_name: int(
            results[
                (results.band == "q5") & (results.relation == relation_name)
            ].system_id.nunique()
        )
        for relation_name in ("within", "between", "novel")
    }
    atomic_yaml(
        run_output / "checkpoint22_report.yaml",
        {
            "checkpoint": 22,
            "status": "full_single_assignment_complete"
            if args.full
            else "pilot_complete",
            "systems": len(selected),
            "hdbscan_viable_systems": int(valid),
            "within_between_feasible_systems_any_band": int(overall_feasible),
            "q5_systems_by_relation": tail_counts,
            "q5_within_and_between_feasible_systems": tail_feasible,
            "tail_support_criterion": "at least 8 systems evaluable in both within and between q5",
            "full_run_gate_passed": bool(tail_feasible >= 8),
            "work_scale_q5_paired_systems": int(
                paired_q5.loc[paired_q5.model == "work_scale", "systems"].iloc[0]
            ),
            "work_scale_q5_median_paired_delta_points": float(
                100
                * paired_q5.loc[
                    paired_q5.model == "work_scale", "median_paired_delta"
                ].iloc[0]
            ),
        },
    )


if __name__ == "__main__":
    main()
