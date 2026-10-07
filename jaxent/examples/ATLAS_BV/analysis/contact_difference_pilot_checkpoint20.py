"""Checkpoint 20: contact-coordinate differences versus fixed-BV population metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_systems
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK, density_targets, mass_metrics, scalar_scale, system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    OUTPUT as CHECKPOINT19_OUTPUT, PAIR_CAP, RIDGE_ALPHAS, sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint20_contact_difference_pilot"
FIT_REPLICA, TUNE_REPLICA, TEST_REPLICA = 1, 2, 3
MAGNITUDE_MODELS = (
    "work_scale",
    "legacy_density",
    "fixed_pf_cosine",
    "fixed_pf_correlation",
    "contact_l1",
    "contact_l2",
    "contact_cosine",
    "contact_correlation",
    "contact_channel_ridge",
    "contact_per_residue_ridge",
)
CONTACT_MODELS = tuple(name for name in MAGNITUDE_MODELS if name.startswith("contact_"))


def safe_distance(profile: np.ndarray, left: np.ndarray, right: np.ndarray, metric: str) -> np.ndarray:
    values = pf_pair_distance(profile, left, right, metric)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def channel_change_scales(heavy: np.ndarray, acceptor: np.ndarray, pair_frame: pd.DataFrame,
                          indices: np.ndarray) -> tuple[float, float]:
    left = pair_frame.left_frame.to_numpy()[indices]; right = pair_frame.right_frame.to_numpy()[indices]
    scales = []
    for channel in (heavy, acceptor):
        rms = float(np.sqrt(np.mean(np.square(channel[:, left] - channel[:, right]))))
        scales.append(rms if rms > np.finfo(float).eps else 1.0)
    return scales[0], scales[1]


def contact_pair_features(data: dict, sample_indices: dict[int, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    heavy = data["heavy"]; acceptor = data["acceptor"]
    fit_pairs = data["pairs"][FIT_REPLICA]
    heavy_scale, acceptor_scale = channel_change_scales(
        heavy, acceptor, fit_pairs, sample_indices[FIT_REPLICA]
    )
    stacked = np.concatenate([heavy / heavy_scale, acceptor / acceptor_scale], axis=0)
    output: dict[str, dict[int, np.ndarray]] = {name: {} for name in CONTACT_MODELS}
    for replica, pair_frame in data["pairs"].items():
        selected = sample_indices[replica]
        left = pair_frame.left_frame.to_numpy()[selected]; right = pair_frame.right_frame.to_numpy()[selected]
        delta_heavy = heavy[:, left] - heavy[:, right]
        delta_acceptor = acceptor[:, left] - acceptor[:, right]
        l1_channels = np.column_stack([
            np.mean(np.abs(delta_heavy), axis=0) / heavy_scale,
            np.mean(np.abs(delta_acceptor), axis=0) / acceptor_scale,
        ])
        l2_channels = np.column_stack([
            np.sqrt(np.mean(np.square(delta_heavy), axis=0)) / heavy_scale,
            np.sqrt(np.mean(np.square(delta_acceptor), axis=0)) / acceptor_scale,
        ])
        output["contact_l1"][replica] = l1_channels.sum(axis=1, keepdims=True)
        output["contact_l2"][replica] = np.sqrt(np.square(l2_channels).sum(axis=1, keepdims=True))
        output["contact_cosine"][replica] = safe_distance(stacked, left, right, "cosine")[:, None]
        output["contact_correlation"][replica] = safe_distance(stacked, left, right, "correlation")[:, None]
        output["contact_channel_ridge"][replica] = l1_channels
        output["contact_per_residue_ridge"][replica] = np.concatenate([
            np.abs(delta_heavy).T / heavy_scale,
            np.abs(delta_acceptor).T / acceptor_scale,
        ], axis=1)
    return output


def baseline_pair_features(data: dict, sample_indices: dict[int, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    thermo = thermodynamic_pair_features(data["z"], data["pairs"])
    output = {name: {} for name in ("work_scale", "legacy_density", "fixed_pf_cosine", "fixed_pf_correlation")}
    for replica, pair_frame in data["pairs"].items():
        selected = sample_indices[replica]
        left = pair_frame.left_frame.to_numpy()[selected]; right = pair_frame.right_frame.to_numpy()[selected]
        output["work_scale"][replica] = thermo["work_scale"][replica][selected, None]
        output["legacy_density"][replica] = thermo["work_density_legacy_zq"][replica][selected, None]
        output["fixed_pf_cosine"][replica] = safe_distance(data["z"], left, right, "cosine")[:, None]
        output["fixed_pf_correlation"][replica] = safe_distance(data["z"], left, right, "correlation")[:, None]
    return output


def tune_ridge(x_fit: np.ndarray, y_fit: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray,
               positive: bool) -> tuple[Ridge, float, np.ndarray]:
    rms = np.sqrt(np.mean(np.square(x_fit), axis=0)); rms = np.where(rms > 1e-12, rms, 1.0)
    scored = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=False, positive=positive).fit(x_fit / rms, y_fit)
        loss = float(np.mean(np.abs(y_tune - model.predict(x_tune / rms))))
        scored.append((loss, -alpha, float(alpha), model))
    _, _, alpha, model = min(scored, key=lambda item: (item[0], item[1]))
    return model, alpha, rms


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    value = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return value if np.isfinite(value) else 0.0


def regions(w1: np.ndarray, edges: np.ndarray) -> dict[str, np.ndarray]:
    result = {"all": np.ones(len(w1), bool), "q4_q5": w1 >= edges[4]}
    for band in range(6):
        result[f"q{band}"] = ((w1 >= edges[band]) & (w1 < edges[band + 1]))
    result["q5"] = w1 >= edges[5]
    return result


def evaluate_system(row: dict[str, str], config: dict, edges: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    data = system_data(row, config); system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT_REPLICA, PRIMARY_RANK)
    indices = {replica: sampled_indices(system, replica, len(values), PAIR_CAP)
               for replica, values in signed_targets.items()}
    magnitude_targets = {replica: np.abs(values[indices[replica]]) for replica, values in signed_targets.items()}
    contact = contact_pair_features(data, indices); baseline = baseline_pair_features(data, indices)
    features = {**baseline, **contact}; result_rows = []; coefficient_rows = []; diagnostic_rows = []
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    predictions = {}
    for name in MAGNITUDE_MODELS:
        model, alpha, rms = tune_ridge(features[name][1], magnitude_targets[1], features[name][2],
                                       magnitude_targets[2], positive=True)
        prediction = np.maximum(0.0, model.predict(features[name][3] / rms)); predictions[name] = prediction
        for index, coefficient in enumerate(model.coef_):
            coefficient_rows.append({"system_id": system, "model": name, "feature_index": index,
                                     "ridge_alpha": alpha, "standardized_coefficient": float(coefficient),
                                     "original_scale_coefficient": float(coefficient / rms[index])})
        w1 = data["pairs"][3].w1.to_numpy()[indices[3]]
        for region, mask in regions(w1, edges).items():
            if not mask.any(): continue
            distribution = mass_metrics(magnitude_targets[3][mask], prediction[mask], magnitude_targets[1],
                                        settings["distribution_bins"], settings["distribution_smoothing"])
            result_rows.append({"system_id": system, "model": name, "target_kind": "magnitude",
                                "region": region, "pairs": int(mask.sum()), "bandwidth_angstrom": bandwidth,
                                "mae": float(np.mean(np.abs(magnitude_targets[3][mask] - prediction[mask]))),
                                "spearman": spearman(magnitude_targets[3][mask], prediction[mask]), **distribution})

    # Signed two-channel diagnostic: directly test whether freeing the fixed BV projection helps ordering.
    signed_features = {}
    for replica, pair_frame in data["pairs"].items():
        selected = indices[replica]
        left = pair_frame.left_frame.to_numpy()[selected]; right = pair_frame.right_frame.to_numpy()[selected]
        signed_features[replica] = np.column_stack([
            data["heavy"][:, left].mean(axis=0) - data["heavy"][:, right].mean(axis=0),
            data["acceptor"][:, left].mean(axis=0) - data["acceptor"][:, right].mean(axis=0),
        ])
    signed_y = {replica: values[indices[replica]] for replica, values in signed_targets.items()}
    free_model, alpha, rms = tune_ridge(signed_features[1], signed_y[1], signed_features[2], signed_y[2], positive=False)
    free_prediction = free_model.predict(signed_features[3] / rms)
    fixed_x = .35 * signed_features[1][:, 0] + 2.0 * signed_features[1][:, 1]
    fixed_alpha, _, _ = scalar_scale(fixed_x, signed_y[1], False)
    signed_predictions = {
        "fixed_signed_bv": fixed_alpha * (.35 * signed_features[3][:, 0] + 2.0 * signed_features[3][:, 1]),
        "free_signed_contacts": free_prediction,
    }
    w1 = data["pairs"][3].w1.to_numpy()[indices[3]]
    for name, prediction in signed_predictions.items():
        for region, mask in regions(w1, edges).items():
            if not mask.any(): continue
            distribution = mass_metrics(signed_y[3][mask], prediction[mask], signed_y[1],
                                        settings["distribution_bins"], settings["distribution_smoothing"])
            result_rows.append({"system_id": system, "model": name, "target_kind": "signed",
                                "region": region, "pairs": int(mask.sum()), "bandwidth_angstrom": bandwidth,
                                "mae": float(np.mean(np.abs(signed_y[3][mask] - prediction[mask]))),
                                "spearman": spearman(signed_y[3][mask], prediction[mask]), **distribution})

    scale_residual = magnitude_targets[2] - Ridge(alpha=0, fit_intercept=False, positive=True).fit(
        features["work_scale"][1], magnitude_targets[1]).predict(features["work_scale"][2])
    for name in CONTACT_MODELS:
        if features[name][2].shape[1] <= 2:
            for column in range(features[name][2].shape[1]):
                diagnostic_rows.append({"system_id": system, "model": name, "feature_index": column,
                                        "residual_spearman": spearman(features[name][2][:, column], scale_residual)})
    return result_rows, coefficient_rows, diagnostic_rows


def make_report(results: pd.DataFrame, coefficients: pd.DataFrame, diagnostics: pd.DataFrame,
                edges: np.ndarray) -> None:
    magnitude = results[results.target_kind == "magnitude"]
    baseline = magnitude[magnitude.model == "work_scale"][["system_id", "region", "distribution_recovery", "mae"]]
    baseline = baseline.rename(columns={"distribution_recovery": "baseline_recovery", "mae": "baseline_mae"})
    paired = magnitude.merge(baseline, on=["system_id", "region"], validate="many_to_one")
    paired["recovery_improvement"] = paired.distribution_recovery - paired.baseline_recovery
    paired["mae_improvement"] = paired.baseline_mae - paired.mae
    atomic_parquet(paired, OUTPUT / "contact_paired_results.parquet")
    summary = (paired.groupby(["model", "region"], as_index=False)
               .agg(median_recovery=("distribution_recovery", "median"),
                    median_improvement=("recovery_improvement", "median"),
                    contributing_systems=("system_id", "nunique"),
                    systems_improved=("recovery_improvement", lambda x: int((x >= .03).sum())),
                    median_mae_improvement=("mae_improvement", "median")))
    atomic_parquet(summary, OUTPUT / "contact_pilot_summary.parquet")
    labels = {name: name.replace("_", " ").title() for name in MAGNITUDE_MODELS}
    bands = [f"q{i}" for i in range(6)]
    xticks = [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 12), sharex=True)
    selected_models = MAGNITUDE_MODELS
    for name in selected_models:
        block = paired[(paired.model == name) & paired.region.isin(bands)]
        grouped = block.groupby("region").distribution_recovery
        median = grouped.median(); sd = grouped.std()
        y = 100 * np.array([median.get(b, np.nan) for b in bands])
        spread = 100 * np.array([sd.get(b, np.nan) for b in bands])
        line = axes[0].plot(range(6), y, marker="o", lw=2, label=labels[name])[0]
        axes[0].fill_between(range(6), np.maximum(0, y-spread), np.minimum(100, y+spread),
                             color=line.get_color(), alpha=.08, linewidth=0)
        if name != "work_scale":
            grouped = block.groupby("region").recovery_improvement
            median = grouped.median(); lo = grouped.quantile(.25); hi = grouped.quantile(.75)
            y = 100 * np.array([median.get(b, np.nan) for b in bands])
            axes[1].plot(range(6), y, marker="o", lw=2, label=labels[name])
            axes[1].fill_between(range(6), 100*np.array([lo.get(b,np.nan) for b in bands]),
                                 100*np.array([hi.get(b,np.nan) for b in bands]), alpha=.08)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    axes[0].set_title("Contact-coordinate magnitude recovery across W1 distance")
    axes[1].axhline(3, color="black", ls="--", label="+3 pp gate"); axes[1].axhline(0, color="grey")
    axes[1].set_ylabel("Paired improvement over Work Scale (pp)"); axes[1].set_xlabel("Global W1 band")
    axes[1].set_xticks(range(6), xticks)
    for axis in axes: axis.grid(alpha=.25); axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Contact-difference pilot: 12 systems, replica 1→2→3, KDE rank 10")
    fig.tight_layout(); fig.savefig(OUTPUT / "contact_recovery_across_global_w1.png", dpi=180); plt.close(fig)

    signed = results[results.target_kind == "signed"]
    fig, axis = plt.subplots(figsize=(10.5, 6))
    for name in ("fixed_signed_bv", "free_signed_contacts"):
        block = signed[(signed.model == name) & signed.region.isin(bands)]
        median = block.groupby("region").distribution_recovery.median()
        axis.plot(range(6), 100*np.array([median.get(b,np.nan) for b in bands]), marker="o", lw=2,
                  label=labels.get(name, name.replace("_", " ").title()))
    axis.set_xticks(range(6), xticks); axis.set_xlabel("Global W1 band")
    axis.set_ylabel(r"Signed recovery, $100(1-\sqrt{JSD})$ (%)")
    axis.set_title("Signed contact-coordinate diagnostic"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(OUTPUT / "signed_contact_recovery_across_global_w1.png", dpi=180); plt.close(fig)

    tail = summary[(summary.region.isin(["q4", "q5"])) & summary.model.isin(CONTACT_MODELS)]
    required_systems = np.ceil(2 * tail.contributing_systems / 3).astype(int)
    passing = tail[(tail.median_improvement >= .03) & (tail.systems_improved >= required_systems)
                   & (tail.median_mae_improvement >= 0)]
    atomic_yaml(OUTPUT / "checkpoint20_report.yaml", {
        "checkpoint": 20, "status": "pilot_complete", "full_run_gate_passed": bool(len(passing)),
        "passing_model_regions": passing[["model", "region"]].to_dict("records"),
        "gate": ">=3 pp median q4 or q5 improvement, >=2/3 contributing systems improve >=3 pp, MAE not worse",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.parse_args()
    config = load_config(); by_id = {row["system_id"]: row for row in load_systems()}
    selection = pd.read_parquet(CHECKPOINT19_OUTPUT / "pilot_systems.parquet")
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    OUTPUT.mkdir(parents=True, exist_ok=True); result_rows=[]; coefficient_rows=[]; diagnostic_rows=[]
    for index, system in enumerate(selection.system_id, 1):
        results, coefficients, diagnostics = evaluate_system(by_id[system], config, edges)
        result_rows.extend(results); coefficient_rows.extend(coefficients); diagnostic_rows.extend(diagnostics)
        print(f"[{index}/{len(selection)}] {system}", flush=True)
    results = pd.DataFrame(result_rows); coefficients = pd.DataFrame(coefficient_rows); diagnostics = pd.DataFrame(diagnostic_rows)
    atomic_parquet(results, OUTPUT / "contact_pilot_results.parquet")
    atomic_parquet(coefficients, OUTPUT / "contact_pilot_coefficients.parquet")
    atomic_parquet(diagnostics, OUTPUT / "contact_pilot_diagnostics.parquet")
    make_report(results, coefficients, diagnostics, edges)


if __name__ == "__main__":
    main()
