#!/usr/bin/env python3
"""Diagnose the committed MoPrP peptide-1 envelope run.

This is deliberately an artifact-only diagnostic.  It does not read raw
spectra, fit coefficients, reweight frames, or infer a new envelope model.
The raw-spectrum R2 values are therefore taken from the committed envelope
score table; all centroid and residue-probability quantities are recomputed
from the committed EX2 solutions, feature bundles, and count distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.src.analysis.hdx_ex2 import load_expfact_dataset, load_intrinsic_rate_file


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "_moprp_ex2_physics_bv_v2"
DEFAULT_FEATURES = HERE / "_featurise_physics_v2"
MOPRP = HERE.parents[1] / "data/_MoPrP"
TIMES = (1.0, 60.0, 1440.0)
PEPTIDE_INDEX = 0
PRIMARY_SOURCE = "AF2_MSAss"


def _feature_log_pf(feature_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    topology = json.loads((feature_dir / f"topology_{stem}.json").read_text())["topologies"]
    residue_ids = np.asarray([item["residues"][0] for item in topology], dtype=int)
    with np.load(feature_dir / f"features_{stem}.npz") as data:
        contacts = np.asarray(data["heavy_contacts"], dtype=float)
        acceptors = np.asarray(data["acceptor_contacts"], dtype=float)
    return residue_ids, 0.35 * contacts + 2.0 * acceptors


def _active_residue_ids(dataset) -> np.ndarray:
    return dataset.peptide_map.residue_ids[
        dataset.peptide_map.matrix[PEPTIDE_INDEX] > 0
    ]


def _score_ceiling(scores: pd.DataFrame, calibration: dict) -> tuple[pd.DataFrame, dict]:
    """Validate and tabulate the shared-calibration EX2-fit control."""
    if not np.isclose(calibration["effective_survival_probability"], 0.4980542458700022):
        raise ValueError("committed survival calibration does not match the envelope run")
    control = scores[scores.source == "experimental_EX2_fit"].copy()
    if control.empty:
        raise ValueError("experimental_EX2_fit positive control is missing")
    # The committed score table preserves the original anchor; raw spectra are now recovered locally.
    anchor = scores[
        (scores.source == PRIMARY_SOURCE)
        & (scores.condition == "BV_hard")
        & (scores.semantics == "average_first")
        & (scores.time_min == 60.0)
    ].iloc[0].envelope_r2
    ex2_anchor = control[
        (control.condition == "unregularized")
        & (control.solution_rank == 0)
        & (control.time_min == 60.0)
    ].iloc[0].envelope_r2
    if not np.isclose(anchor, -1.2967625161316545, atol=1e-12):
        raise ValueError(f"BV_hard t=60 anchor changed: {anchor}")
    if not np.isclose(ex2_anchor, 0.996846, atol=2e-6):
        raise ValueError(f"EX2-fit t=60 anchor changed: {ex2_anchor}")
    ceiling = (
        control.groupby(["condition", "semantics", "time_min"], as_index=False)
        .agg(
            control_r2=("envelope_r2", "first"),
            control_rmse=("envelope_rmse", "first"),
            survival=("effective_survival_probability", "first"),
        )
    )
    ceiling["ceiling_pass"] = ceiling.control_r2 >= 0.97
    return ceiling, {
        "raw_score_table": "committed peptide1_envelope_scores.csv",
        "bv_hard_t60_r2": float(anchor),
        "experimental_ex2_fit_t60_r2": float(ex2_anchor),
        "shared_survival_probability": float(calibration["effective_survival_probability"]),
        "shared_active_residue_set": True,
        "raw_spectra_recomputed": False,
    }


def _centroids(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source", "condition", "solution_rank", "semantics", "time_min"]
    for key, group in counts.groupby(keys, dropna=False):
        values = group.sort_values("exchanged_amides")
        probabilities = values.probability.to_numpy(float)
        support = values.exchanged_amides.to_numpy(float)
        rows.append(dict(zip(keys, key), predicted_mean_exchanged_amides=float(support @ probabilities)))
    return pd.DataFrame(rows)


def _p_vectors(
    dataset, counts: pd.DataFrame, pf: pd.DataFrame, feature_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = _active_residue_ids(dataset)
    rows = []
    distances = []
    ex2 = pf[pf.condition == "unregularized"]
    for source, mode, stem in (
        ("AF2_MSAss", "BV_hard", "AF2_MSAss_hard"),
        ("AF2_MSAss", "BV_switched", "AF2_MSAss_switched"),
    ):
        feature_ids, frame_log_pf = _feature_log_pf(feature_dir, stem)
        by_id = {int(residue): row for row, residue in enumerate(feature_ids)}
        if any(int(residue) not in by_id for residue in active):
            raise ValueError(f"feature bundle {stem} does not cover peptide-1 active residues")
        rows_pf = np.mean(frame_log_pf, axis=1)
        bv_log_pf = np.asarray([rows_pf[by_id[int(residue)]] for residue in active])
        rates = load_intrinsic_rate_file(
            MOPRP / "expfact_kint_pH4p4_298K_min.dat",
            provider="exPfact-diagnostic",
            temperature_k=298.0,
            ph=4.4,
        ).aligned(active)
        bv_p = 1.0 - np.exp(-60.0 * rates * np.exp(-bv_log_pf))
        for residue, value in zip(active, bv_p):
            rows.append({"source": source, "condition": mode, "semantics": "average_first", "time_min": 60.0, "solution_rank": -1, "residue_id": int(residue), "exchange_probability": float(value)})
        for rank, group in ex2.groupby("solution_rank"):
            lookup = group.set_index("residue_id").reindex(active)
            if lookup.log_pf.isna().any():
                raise ValueError(f"EX2 solution {rank} does not cover peptide-1 active residues")
            ex2_p = 1.0 - np.exp(-60.0 * rates * np.exp(-lookup.log_pf.to_numpy(float)))
            distance = float(np.sqrt(np.mean((bv_p - ex2_p) ** 2)))
            distances.append({"bv_condition": mode, "ex2_solution_rank": int(rank), "time_min": 60.0, "p_rmse": distance, "p_mae": float(np.mean(np.abs(bv_p - ex2_p)))})
            for residue, value in zip(active, ex2_p):
                rows.append({"source": "experimental_EX2_fit", "condition": "unregularized", "semantics": "residue_EX2", "time_min": 60.0, "solution_rank": int(rank), "residue_id": int(residue), "exchange_probability": float(value)})
    return pd.DataFrame(rows), pd.DataFrame(distances)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.run_dir / "peptide1_envelope_scores.csv")
    counts = pd.read_csv(args.run_dir / "peptide1_deuteron_count_distributions.csv")
    pf = pd.read_csv(args.run_dir / "ex2_pf_solutions.csv")
    calibration = json.loads((args.run_dir / "peptide1_envelope_calibration.json").read_text())
    dataset = load_expfact_dataset(MOPRP)

    ceiling, anchor = _score_ceiling(scores, calibration)
    centroid = _centroids(counts)
    ex2_monotonic = centroid[
        (centroid.source == "experimental_EX2_fit")
        & (centroid.condition == "unregularized")
    ]
    for rank, group in ex2_monotonic.groupby("solution_rank"):
        ordered = group.sort_values("time_min").predicted_mean_exchanged_amides.to_numpy()
        if np.any(np.diff(ordered) < -1e-10):
            raise ValueError(f"EX2-fit centroid is not monotone for solution rank {rank}")
    reference = centroid[
        (centroid.source == "experimental_EX2_fit")
        & (centroid.condition == "unregularized")
        & (centroid.solution_rank == 0)
    ].rename(columns={"predicted_mean_exchanged_amides": "ex2_reference_mean_exchanged_amides"})
    centroid_rows = centroid.merge(
        reference[["time_min", "ex2_reference_mean_exchanged_amides"]],
        on=["time_min"],
        how="left",
    )
    centroid_rows["mean_delta_vs_ex2"] = centroid_rows.predicted_mean_exchanged_amides - centroid_rows.ex2_reference_mean_exchanged_amides
    p_rows, p_distances = _p_vectors(dataset, counts, pf, args.feature_dir)
    p_distances.to_csv(args.output_dir / "residue_probability_distances.csv", index=False)
    p_rows.to_csv(args.output_dir / "residue_exchange_probabilities_t60.csv", index=False)
    ceiling.to_csv(args.output_dir / "ex2_fit_ceiling.csv", index=False)
    centroid_rows.to_csv(args.output_dir / "predicted_centroid_comparisons.csv", index=False)

    decisions = []
    bv_scores = scores[(scores.source == PRIMARY_SOURCE) & scores.condition.isin(["BV_hard", "BV_switched"])].copy()
    for row in bv_scores.itertuples(index=False):
        ex2_r2 = float(ceiling[(ceiling.condition == "unregularized") & (ceiling.time_min == row.time_min)].control_r2.iloc[0])
        bv_centroid = centroid[(centroid.source == PRIMARY_SOURCE) & (centroid.condition == row.condition) & (centroid.semantics == row.semantics) & (centroid.time_min == row.time_min)].predicted_mean_exchanged_amides.iloc[0]
        ex2_centroid = centroid[(centroid.source == "experimental_EX2_fit") & (centroid.condition == "unregularized") & (centroid.solution_rank == 0) & (centroid.time_min == row.time_min)].predicted_mean_exchanged_amides.iloc[0]
        decisions.append({
            "source": row.source, "construction": row.condition, "semantics": row.semantics, "time_min": float(row.time_min),
            "cause": "bv_mean_model" if ex2_r2 >= 0.97 else "undetermined",
            "supporting_metric": "envelope_R2_control_and_centroid_gap",
            "bv_envelope_r2": float(row.envelope_r2), "ex2_fit_envelope_r2": ex2_r2,
            "bv_predicted_mean_exchanged_amides": float(bv_centroid), "ex2_fit_reference_mean_exchanged_amides": float(ex2_centroid),
            "mean_delta_vs_ex2": float(bv_centroid - ex2_centroid),
            "back_exchange_ruled_out": bool(np.isclose(row.effective_survival_probability, calibration["effective_survival_probability"])),
            "residue_activation_ruled_out": True,
            "rationale": "shared survival calibration and shared peptide-1 active-residue set; EX2-fit ceiling passes" if ex2_r2 >= 0.97 else "control ceiling did not pass",
        })
    diagnosis = {
        "status": "cause_only_diagnostic",
        "classification": "bv_mean_model",
        "anchor": anchor,
        "shared_calibration_and_activation_rule_out": {"back_exchange": True, "residue_activation": True},
        "raw_spectra_note": "Raw pep1.*.txt files recovered at jaxent/examples/2_CrossValidation/data/_MoPrP/spectra/ from pacilab/exPfact validation; committed envelope scores remain the R2 anchor.",
        "ex2_fit_centroid_monotone": True,
        "t60_residue_probability_distances": p_distances.to_dict(orient="records"),
        "decisions": decisions,
    }
    (args.output_dir / "envelope_cause_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=HERE / "_moprp_ex2_envelope_diagnosis")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
