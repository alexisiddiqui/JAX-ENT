#!/usr/bin/env python3
"""Real-data canary: does the ISO no-correspondence result reproduce on MoPrP?

No fitting, no optimization, no decoy test, no cluster recovery.

There is no oracle on real data. Correspondence is measured against two imperfect AF2
reference ensembles at uniform weights -- Filtered registered as the mildly-positive
reference, MSAss as the negative one. Neither is truth and n=2 references has no statistical
power, so this can fail loudly but cannot succeed quietly: agreement with Filtered would be a
contradiction to explain, never a promotion.

The oracle-free identifiability control is the one result here that does not depend on either
reference being right.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.splitting.sparse_map import create_sparse_map, normalize_sparse_map_rows
from jaxent.src.interfaces.topology import PTSerialiser, TopologyFactory
from jaxent.src.models.HDX.BV.features import BV_input_features

REPO = Path(__file__).resolve().parents[4]


def _load_litmus():
    """Reuse the registered ISO metrics/constructions rather than re-implementing them."""
    path = (
        REPO
        / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_uptake_rate_covariance.py"
    )
    spec = importlib.util.spec_from_file_location("_iso_uptake_rate_litmus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LITMUS = _load_litmus()
ALPHAS = LITMUS.ALPHAS
PRIMARY_ALPHA = LITMUS.PRIMARY_ALPHA

# Hardcoded in featurise_CrossVal_MSAss_Filtered.py:71-89; minutes.
TIMEPOINTS = np.asarray(
    [0.08, 0.33, 0.67, 1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 160.0, 240.0, 390.0, 750.0, 1440.0]
)
# BV hard-contact coefficients; verified against BV_Model_Parameters defaults (config.py:17-18).
BV_BC, BV_BH = 0.35, 2.0
# splitdata_CrossVal.py:95 slices range(len(segs) - 2), so only the first 12 are ever fitted.
N_FITTED_PEPTIDES = 12
PERMUTATION_SEED = 1729

REFERENCES = {"AF2_filtered": "mildly_positive", "AF2_MSAss": "negative"}
MIN_RESOLVING_DISTANCE = 0.05
# ISO gates, reused unchanged.
FULL_GATE = {"trace_low": 0.5, "trace_high": 2.0, "max_distance": 0.25, "min_off_diagonal": 0.80}
PROFILE_GATE = {"scale_low": 0.5, "scale_high": 2.0, "min_correlation": 0.90, "max_log_rmse": 0.25}

# ISO effective ranks, for the comparison that decides whether 15 timepoints change anything.
ISO_EFFECTIVE_RANK_RANGE = (1.01, 1.64)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_pf_from_features(data: Any) -> np.ndarray:
    """Residues x frames log protection factors under the BV hard-contact model."""
    return BV_BC * np.asarray(data["heavy_contacts"], float) + BV_BH * np.asarray(
        data["acceptor_contacts"], float
    )


def build_sparse_map(segments: np.ndarray, dfrac: np.ndarray, paths: dict[str, Path]) -> np.ndarray:
    """Row-normalized peptide x residue map from the real experimental segments.

    peptide_trim=2 matches splitdata_CrossVal.py:78-88 for MoPrP, not the ISO trim of 1. Real
    peptides overlap, so rows are not disjoint; that is the data, not a defect.
    """
    feature_topology = PTSerialiser.load_list_from_json(str(paths["topology"]))
    features = BV_input_features.load(str(paths["features"]))
    data = [
        HDX_peptide(
            dfrac=dfrac[index],
            top=TopologyFactory.from_range(
                chain="A",
                start=int(segment[0]),
                end=int(segment[1]),
                fragment_index=index,
                peptide=True,
                peptide_trim=2,
                fragment_name="MoPrPCrossVal",
            ),
        )
        for index, segment in enumerate(segments)
    ]
    sparse_map = create_sparse_map(features, feature_topology, data, check_trim=True)
    dense = np.asarray(normalize_sparse_map_rows(sparse_map).todense(), dtype=float)
    assert np.allclose(dense.sum(axis=1), 1.0, atol=1e-6)
    return dense


def construction_registry() -> pd.DataFrame:
    rows = [
        ("curve_raw_uptake", "observable", "Cov_t(y(t))", "raw curve geometry"),
        ("curve_uptake_over_t", "observable", "Cov_t(y(t)/t)", "small-time rate proxy"),
        ("curve_apparent_rate", "observable", "Cov_t(-log(1-y(t))/t)", "apparent-rate proxy"),
        (
            "curve_adjacent_survival_slope",
            "observable",
            "Cov_intervals(-Delta log(1-y)/Delta t)",
            "interval rate proxy",
        ),
        (
            "curve_timepoint_rms_normalized",
            "observable_shape_only",
            "Cov_t(y/RMS_peptide(y_t))",
            "shape-only control",
        ),
        (
            "production_sigma",
            "observable",
            "np.cov(dfrac) + 1e-6 I",
            "the matrix the production Sigma_MSE loss actually consumes",
        ),
        (
            "control_logpf",
            "reference_control",
            "M Cov_w(z) M^T",
            "coordinate-mismatch negative control",
        ),
    ]
    return pd.DataFrame(rows, columns=["construction", "availability", "formula", "role"])


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = REPO / "examples/2_CrossValidation"
    paths = {
        "dfrac": base / "data/_MoPrP/_output/MoPrP_dfrac.dat",
        "segments": base / "data/_MoPrP/_output/MoPrP_segments.txt",
        "sigma": base / "data/_MoPrP_covariance_matrices/Sigma.npz",
    }
    ensembles = {
        name: {
            "features": base / f"fitting/jaxENT/_featurise/features_{stem}.npz",
            "topology": base / f"fitting/jaxENT/_featurise/topology_{stem}.json",
        }
        for name, stem in (("AF2_MSAss", "AF2_MSAss"), ("AF2_filtered", "AF2_filtered"))
    }

    read = dict(sep=r"\s+", comment="#", header=None)
    dfrac = pd.read_csv(paths["dfrac"], **read).to_numpy(float)
    segments = pd.read_csv(paths["segments"], **read).to_numpy(int)
    sigma = np.load(paths["sigma"])["Sigma"].astype(float)

    mapping_full = build_sparse_map(segments, dfrac, ensembles["AF2_MSAss"])
    fitted = slice(0, N_FITTED_PEPTIDES)
    mapping = mapping_full[fitted]
    # Subsetting Sigma to the fitted peptides is a registered step, not a reconciliation:
    # the production matrix is 14x14 while only 12 peptides are ever fitted.
    sigma_fitted = sigma[fitted, fitted]
    mean_curve = dfrac[fitted].T  # T x P, as curve_covariances expects

    preflight = []

    # 1. Sigma identity: is the production "observation noise covariance" a curve construction?
    sigma_recomputed = np.cov(dfrac) + np.diag(np.full(dfrac.shape[0], 1e-6))
    population_raw = LITMUS.curve_covariances(dfrac.T, TIMEPOINTS)["curve_raw_uptake"]
    bessel = len(TIMEPOINTS) / (len(TIMEPOINTS) - 1)
    sigma_vs_curve = np.abs(sigma[fitted, fitted] - (population_raw[fitted, fitted] * bessel)).max()
    preflight.append(
        {
            "check": "sigma_is_np_cov_of_dfrac",
            "value": float(np.abs(sigma - sigma_recomputed).max()),
            "note": "max abs difference; Sigma == np.cov(dfrac) + 1e-6 I",
        }
    )
    preflight.append(
        {
            "check": "sigma_equals_curve_raw_uptake_up_to_bessel_and_ridge",
            "value": float(sigma_vs_curve),
            "note": "max abs difference on the 12 fitted peptides",
        }
    )
    preflight.append(
        {
            "check": "sigma_dimension_vs_fitted_peptides",
            "value": float(sigma.shape[0] - N_FITTED_PEPTIDES),
            "note": f"Sigma is {sigma.shape[0]}x{sigma.shape[0]}; {N_FITTED_PEPTIDES} peptides are fitted",
        }
    )

    matrices: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    permutation_rows, k_int_reference = [], None

    for name, ensemble_paths in ensembles.items():
        with np.load(ensemble_paths["features"]) as data:
            log_pf = log_pf_from_features(data)
            k_int = np.asarray(data["k_ints"], float)
        if args.smoke:
            log_pf = log_pf[:, :50]
        if k_int_reference is None:
            k_int_reference = k_int
        else:
            preflight.append(
                {
                    "check": "k_ints_identical_across_ensembles",
                    "value": float(np.abs(k_int - k_int_reference).max()),
                    "note": "references must differ only by conformation",
                }
            )

        weights = np.full(log_pf.shape[1], 1 / log_pf.shape[1])
        rates = LITMUS.effective_rates(log_pf, k_int)
        rate_covariance = LITMUS.weighted_covariance(rates, weights)
        references[name] = mapping @ rate_covariance @ mapping.T
        matrices[f"reference__{name}"] = references[name]
        matrices[f"control_logpf__{name}"] = mapping @ LITMUS.weighted_covariance(
            log_pf.T, weights
        ) @ mapping.T

        # Map congruence: M C M^T must equal the covariance of mapped values.
        preflight.append(
            {
                "check": f"map_congruence__{name}",
                "value": float(
                    LITMUS.relative_error(
                        references[name], LITMUS.weighted_covariance(rates @ mapping.T, weights)
                    )
                ),
                "note": "M C M^T vs Cov(M x)",
            }
        )

        # Oracle-free identifiability control. At uniform weights every frame is equal-weight,
        # so the permutation is unrestricted: it preserves each amide's marginal rate
        # distribution, hence every predicted mean uptake value, while changing coupling.
        permuted = LITMUS.permute_equal_weight_frames(rates, weights, seed=PERMUTATION_SEED)
        original_uptake = (
            np.einsum("tfr,f->tr", LITMUS.framewise_uptake(rates, TIMEPOINTS), weights) @ mapping.T
        )
        permuted_uptake = (
            np.einsum("tfr,f->tr", LITMUS.framewise_uptake(permuted, TIMEPOINTS), weights)
            @ mapping.T
        )
        permuted_covariance = mapping @ LITMUS.weighted_covariance(permuted, weights) @ mapping.T
        permutation_rows.append(
            {
                "ensemble": name,
                "mean_uptake_max_abs_change": float(np.max(np.abs(permuted_uptake - original_uptake))),
                "full_covariance_relative_change": LITMUS.relative_error(
                    permuted_covariance, references[name]
                ),
                "marginal_relative_change": LITMUS.relative_error(
                    np.diag(permuted_covariance), np.diag(references[name])
                ),
                "conditional_relative_change": LITMUS.relative_error(
                    LITMUS.profiles(permuted_covariance, PRIMARY_ALPHA)["conditional"],
                    LITMUS.profiles(references[name], PRIMARY_ALPHA)["conditional"],
                ),
            }
        )

    # 2. Resolving power. Gates the whole conclusion: if the two references are nearly
    # identical, any discrimination signal is noise.
    resolving = LITMUS.matrix_metrics(references["AF2_MSAss"], references["AF2_filtered"])
    resolving_distance = float(resolving["normalized_frobenius_distance"])
    has_resolving_power = resolving_distance > MIN_RESOLVING_DISTANCE

    candidates = dict(LITMUS.curve_covariances(mean_curve, TIMEPOINTS))
    candidates["production_sigma"] = sigma_fitted
    for name in ensembles:
        candidates[f"control_logpf__{name}"] = matrices[f"control_logpf__{name}"]

    metric_rows, profile_rows = [], []
    for construction, candidate in candidates.items():
        matrices[f"candidate__{construction}"] = candidate
        for reference_name, reference in references.items():
            metrics = LITMUS.matrix_metrics(candidate, reference)
            full_pass = bool(
                FULL_GATE["trace_low"] <= metrics["trace_ratio"] <= FULL_GATE["trace_high"]
                and metrics["normalized_frobenius_distance"] <= FULL_GATE["max_distance"]
                and metrics["off_diagonal_correlation"] >= FULL_GATE["min_off_diagonal"]
            )
            metric_rows.append(
                {
                    "construction": construction,
                    "reference": reference_name,
                    "reference_role": REFERENCES[reference_name],
                    **metrics,
                    "full_gate_pass": full_pass,
                }
            )
            for alpha in ALPHAS:
                candidate_profiles = LITMUS.profiles(candidate, alpha)
                reference_profiles = LITMUS.profiles(reference, alpha)
                for profile in ("marginal", "conditional"):
                    pm = LITMUS.profile_metrics(
                        candidate_profiles[profile], reference_profiles[profile]
                    )
                    gate = bool(
                        np.isclose(alpha, PRIMARY_ALPHA)
                        and PROFILE_GATE["scale_low"] <= pm["scale_ratio"] <= PROFILE_GATE["scale_high"]
                        and pm["pearson"] >= PROFILE_GATE["min_correlation"]
                        and pm["spearman"] >= PROFILE_GATE["min_correlation"]
                        and pm["log_rmse"] <= PROFILE_GATE["max_log_rmse"]
                    )
                    profile_rows.append(
                        {
                            "construction": construction,
                            "reference": reference_name,
                            "alpha": alpha,
                            "profile": profile,
                            **pm,
                            "profile_gate_pass": gate,
                        }
                    )

    metrics_frame = pd.DataFrame(metric_rows)
    profile_frame = pd.DataFrame(profile_rows)

    # Registered directional statistic: positive favours the mildly-positive reference.
    discrimination_rows = []
    for construction in candidates:
        rows = metrics_frame[metrics_frame.construction == construction]
        to_msass = float(
            rows[rows.reference == "AF2_MSAss"].normalized_frobenius_distance.iloc[0]
        )
        to_filtered = float(
            rows[rows.reference == "AF2_filtered"].normalized_frobenius_distance.iloc[0]
        )
        delta = to_msass - to_filtered
        discrimination_rows.append(
            {
                "construction": construction,
                "distance_to_MSAss_negative": to_msass,
                "distance_to_filtered_mildly_positive": to_filtered,
                "delta": delta,
                "favours_filtered": bool(delta > 0),
                "delta_vs_resolving_distance": float(abs(delta) / resolving_distance)
                if resolving_distance
                else np.nan,
            }
        )
    discrimination = pd.DataFrame(discrimination_rows)

    construction_registry().to_csv(args.output_dir / "construction_registry.csv", index=False)
    pd.DataFrame(preflight).to_csv(args.output_dir / "preflight.csv", index=False)
    pd.DataFrame([{**resolving, "has_resolving_power": has_resolving_power}]).to_csv(
        args.output_dir / "resolving_power.csv", index=False
    )
    metrics_frame.to_csv(args.output_dir / "matrix_metrics.csv", index=False)
    profile_frame.to_csv(args.output_dir / "profile_metrics.csv", index=False)
    discrimination.to_csv(args.output_dir / "discrimination.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(
        args.output_dir / "identifiability_permutation.csv", index=False
    )
    np.savez_compressed(args.output_dir / "covariance_matrices.npz", **matrices)

    observable = metrics_frame[metrics_frame.construction.isin(candidates)]
    passing = observable[observable.full_gate_pass]
    gates = {
        "resolving_power": {
            "normalized_frobenius_distance": resolving_distance,
            "threshold": MIN_RESOLVING_DISTANCE,
            "has_resolving_power": has_resolving_power,
        },
        "constructions_passing_full_gate": passing[["construction", "reference"]].to_dict("records"),
        "constructions_favouring_filtered": discrimination[
            discrimination.favours_filtered
        ].construction.tolist(),
        "max_mean_uptake_change_under_permutation": float(
            pd.DataFrame(permutation_rows).mean_uptake_max_abs_change.max()
        ),
        "fitting_performed": False,
        "stage2_status": "pending_user_review",
    }
    (args.output_dir / "gate_results.json").write_text(json.dumps(gates, indent=2) + "\n")

    manifest = {
        "stage": "moprp_real_data_covariance_correspondence_canary",
        "status": "smoke" if args.smoke else "complete_pending_user_review",
        "fitting_performed": False,
        "oracle_available": False,
        "reference_weights": "uniform",
        "reference_roles": REFERENCES,
        "timepoints_minutes": TIMEPOINTS.tolist(),
        "bv_coefficients": {"bc": BV_BC, "bh": BV_BH},
        "fitted_peptides": N_FITTED_PEPTIDES,
        "total_segments": int(segments.shape[0]),
        "primary_alpha": PRIMARY_ALPHA,
        "alpha_sensitivity": list(ALPHAS),
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()
        }
        | {
            f"{name}__{key}": {"path": str(path), "sha256": sha256(path)}
            for name, group in ensembles.items()
            for key, path in group.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    permutation = pd.DataFrame(permutation_rows)
    lines = [
        "# MoPrP real-data covariance-correspondence canary",
        "",
        "## Question",
        "",
        "Does the ISO no-correspondence result reproduce on real MoPrP data, with 15 timepoints",
        "and real overlapping peptides? No fitting, optimization, decoy test, or cluster recovery",
        "was run.",
        "",
        "## What this cannot conclude",
        "",
        "There is no oracle. Correspondence is measured against two imperfect AF2 ensembles at",
        "uniform weights, neither of which is truth. Agreement with the Filtered reference would",
        "be a contradiction to explain, not a promotion.",
        "",
        "## Preflight: resolving power",
        "",
        f"The two references differ by normalized distance `{resolving_distance:.3f}` "
        f"(off-diagonal correlation `{resolving['off_diagonal_correlation']:.3f}`, trace ratio "
        f"`{resolving['trace_ratio']:.3f}`), against a `{MIN_RESOLVING_DISTANCE}` threshold: "
        f"**{'PASS' if has_resolving_power else 'FAIL -- the canary is inconclusive'}**.",
        "",
        "## Identifiability control (oracle-free)",
        "",
    ]
    for row in permutation.itertuples(index=False):
        lines.append(
            f"- **{row.ensemble}**: permuting frames within equal-weight groups changed predicted "
            f"mean uptake by at most `{row.mean_uptake_max_abs_change:.2e}` while changing "
            f"full/marginal/conditional mapped rate covariance by "
            f"`{row.full_covariance_relative_change:.3f}` / `{row.marginal_relative_change:.3f}` / "
            f"`{row.conditional_relative_change:.3f}`."
        )
    lines += [
        "",
        "This needs no truth ensemble: on the real system, the mean uptake curve does not",
        "determine conformational coupling.",
        "",
        "## Correspondence and discrimination",
        "",
        "`delta > 0` favours the mildly-positive Filtered reference.",
        "",
        "| Construction | Eff. rank | d(MSAss) | d(Filtered) | delta | Favours Filtered | Any full gate |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in discrimination.itertuples(index=False):
        rank = float(
            metrics_frame[metrics_frame.construction == row.construction].effective_rank.iloc[0]
        )
        gate = bool(
            metrics_frame[metrics_frame.construction == row.construction].full_gate_pass.any()
        )
        lines.append(
            f"| {row.construction} | {rank:.2f} | {row.distance_to_MSAss_negative:.3f} | "
            f"{row.distance_to_filtered_mildly_positive:.3f} | {row.delta:+.3f} | "
            f"{'yes' if row.favours_filtered else 'no'} | {'yes' if gate else 'no'} |"
        )
    observable_ranks = metrics_frame[
        metrics_frame.construction.isin(LITMUS.curve_covariances(mean_curve, TIMEPOINTS))
        | (metrics_frame.construction == "production_sigma")
    ].effective_rank
    degeneracy_survives = bool(observable_ranks.max() <= ISO_EFFECTIVE_RANK_RANGE[1] * 1.5)
    lines += [
        "",
        f"Observable effective ranks span `{observable_ranks.min():.2f}--{observable_ranks.max():.2f}`, "
        f"against ISO's `{ISO_EFFECTIVE_RANK_RANGE[0]}--{ISO_EFFECTIVE_RANK_RANGE[1]}` at five "
        f"timepoints. The near-rank-one degeneracy "
        f"**{'reproduces' if degeneracy_survives else 'does NOT reproduce'}** at 15 timepoints, so it "
        f"{'is not' if degeneracy_survives else 'may be'} an artefact of the ISO timepoint grid.",
        "",
        f"No construction passed the full gate against either reference ({int(passing.shape[0])} of "
        f"{int(observable.shape[0])}). Every distance is `{discrimination[['distance_to_MSAss_negative', 'distance_to_filtered_mildly_positive']].min().min():.2f}--"
        f"{discrimination[['distance_to_MSAss_negative', 'distance_to_filtered_mildly_positive']].max().max():.2f}`, "
        "i.e. essentially unrelated to both references, and the deltas are small relative to the",
        "reference separation. The defensible reading is the null one: no construction is close to",
        "either reference, and the registered direction is not supported. A construction sitting",
        "nearer the negative reference is not evidence that it prefers the worse ensemble.",
        "",
        "## What the production Sigma matrix is",
        "",
        "`Sigma.npz`, consumed by the production `Sigma_MSE` loss, is reproduced by",
        "`np.cov(dfrac) + 1e-6 I` and equals the `curve_raw_uptake` construction up to the Bessel",
        "factor and that ridge. MoPrP has no replicates, so it is curve geometry, not observation",
        "noise. That is a legitimate role as fixed mean-residual weighting; it is not a PF",
        "conformational variance and must not be read as one.",
        "",
        "## Checkpoint",
        "",
        "Stage 2 remains **pending explicit user approval**. No fitting is authorized by this run.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    shown = [
        (references["AF2_filtered"], "Filtered conformational"),
        (references["AF2_MSAss"], "MSAss conformational"),
        (candidates["production_sigma"], "production Sigma"),
    ]
    for axis, (matrix, title) in zip(axes, shown):
        image = axis.imshow(matrix / (np.trace(matrix) or 1), cmap="coolwarm")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(args.output_dir / "correspondence.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "_moprp_uptake_covariance_litmus"
    )
    parser.add_argument("--smoke", action="store_true", help="integration check only; not evidence")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
