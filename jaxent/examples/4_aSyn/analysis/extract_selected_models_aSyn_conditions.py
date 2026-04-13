#!/usr/bin/env python3
"""Extract selected per-replicate model arrays for 4_aSyn condition workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from jaxent.examples.common import loading
from jaxent.examples.common.config import ExperimentConfig

ENSEMBLE_PATTERN = (
    r"(.+?)_(mcMSE|MSE|Sigma_MSE)_(.+?)_split(\d+)"
    r"_maxent([\d.]+)_bvreg([\d.]+)_bvregfn([A-Za-z0-9]+)"
)


def _resolve_path(script_dir: Path, value: str, absolute_paths: bool) -> Path:
    return Path(value) if absolute_paths else (script_dir / value).resolve()


def _resolve_effective_split_type(run_split_type: str, datasplit_dir: Path) -> str:
    """Return the split-type name that actually exists under datasplit_dir."""
    if "_cluster" in run_split_type:
        full_path = datasplit_dir / run_split_type
        stripped = run_split_type.replace("_cluster", "")
        stripped_path = datasplit_dir / stripped
        if full_path.exists():
            return run_split_type
        if stripped_path.exists():
            return stripped
        return stripped
    return run_split_type


def _float_key(value: float, digits: int = 12) -> float:
    return round(float(value), digits)


def _load_convergence_index(convergence_path: Path, target_value: float) -> int | None:
    if not convergence_path.exists():
        return None

    with open(convergence_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        try:
            value = float(line.split("_", 1)[1]) if line.startswith("state_") else float(line)
        except ValueError:
            continue
        if abs(value - target_value) < 1e-9:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected model arrays from processed 4_aSyn analysis outputs."
    )
    parser.add_argument(
        "--processed-data-dir",
        required=True,
        help="Directory containing processed run directories from process_optimisation_results.",
    )
    parser.add_argument(
        "--scores-csv",
        required=True,
        help="Path to detailed model_scores.csv from score_models_aSyn_conditions.py.",
    )
    parser.add_argument(
        "--selection-csv",
        required=True,
        help="Path to model_selection_performance_summary.csv from mixed linear model analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <processed-data-dir>/_extracted_<processed-basename>.",
    )
    parser.add_argument(
        "--datasplit-dir",
        default="../fitting/_datasplits",
        help="Datasplit base directory (used for split-type alias resolution).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to experiment config YAML. Defaults to <script_dir>/../config.yaml",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided paths as absolute.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent
    
    processed_data_dir = _resolve_path(script_dir, args.processed_data_dir, args.absolute_paths)
    datasplit_dir = _resolve_path(script_dir, args.datasplit_dir, args.absolute_paths)
    scores_csv_path = _resolve_path(script_dir, args.scores_csv, args.absolute_paths)
    selection_csv_path = _resolve_path(script_dir, args.selection_csv, args.absolute_paths)
    
    if args.config:
        config_path = _resolve_path(script_dir, args.config, args.absolute_paths)
    else:
        config_path = exp_dir / "config.yaml"

    if args.output_dir:
        output_dir = _resolve_path(script_dir, args.output_dir, args.absolute_paths)
    else:
        output_dir = processed_data_dir / f"_extracted_{processed_data_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig.from_yaml(config_path)

    print(f"processed_data_dir: {processed_data_dir}")
    print(f"scores_csv_path:    {scores_csv_path}")
    print(f"selection_csv_path: {selection_csv_path}")
    print(f"output_dir:         {output_dir}")
    print("-" * 60)

    if not scores_csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")
    if not selection_csv_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {selection_csv_path}")
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_data_dir}")

    scores_df = pd.read_csv(scores_csv_path)
    selection_df = pd.read_csv(selection_csv_path)

    required_cols = {
        "ensemble",
        "split_type",
        "loss_function",
        "bv_reg_function",
        "split_idx",
        "maxent_value",
        "bv_reg_value",
        "convergence_value",
    }
    missing = sorted(required_cols - set(scores_df.columns))
    if missing:
        raise ValueError(f"Scores CSV missing required columns: {missing}")
    if "score_metric" not in selection_df.columns or "direction" not in selection_df.columns:
        raise ValueError("Selection CSV must include 'score_metric' and 'direction' columns")

    scores_df = scores_df[scores_df["ensemble"].isin(cfg.ensembles)].copy()
    scores_df["split_idx"] = pd.to_numeric(scores_df["split_idx"], errors="coerce")
    scores_df["maxent_value"] = pd.to_numeric(scores_df["maxent_value"], errors="coerce")
    scores_df["bv_reg_value"] = pd.to_numeric(scores_df["bv_reg_value"], errors="coerce")
    scores_df["convergence_value"] = pd.to_numeric(scores_df["convergence_value"], errors="coerce")
    scores_df = scores_df.dropna(
        subset=["split_idx", "maxent_value", "bv_reg_value", "convergence_value"]
    )
    scores_df["split_idx"] = scores_df["split_idx"].astype(int)

    print("Cataloging processed runs...")
    all_run_info, _ = loading.load_processed_run_info(
        str(processed_data_dir),
        ENSEMBLE_PATTERN,
        extra_group_names=["bv_reg_str", "bv_reg_fn"],
    )

    run_lookup: dict[tuple, dict] = {}
    for run in all_run_info:
        eff_split_type = _resolve_effective_split_type(
            run["run_split_type"], datasplit_dir / run["ensemble"]
        )
        key = (
            run["ensemble"],
            run["loss_name"],
            eff_split_type,
            int(run["split_idx"]),
            _float_key(run["maxent_value"]),
            _float_key(float(run["bv_reg_str"])),
            run["bv_reg_fn"],
        )
        run_lookup[key] = run

    if not run_lookup:
        raise RuntimeError(f"No processed runs matched pattern in: {processed_data_dir}")

    group_cols = ["ensemble", "split_type", "loss_function", "bv_reg_function"]
    print("Extracting selected arrays...")

    selection_rows = selection_df.drop_duplicates(subset=["score_metric", "direction"])
    for _, sel_row in tqdm(selection_rows.iterrows(), total=len(selection_rows)):
        metric = str(sel_row["score_metric"])
        direction = str(sel_row["direction"]).lower()

        if metric not in scores_df.columns:
            print(f"WARNING: Metric '{metric}' not found in scores CSV. Skipping.")
            continue
        if direction not in {"min", "max"}:
            print(f"WARNING: Unsupported direction '{direction}' for metric '{metric}'. Skipping.")
            continue

        metric_dir = output_dir / f"{metric}_{direction}"
        metric_dir.mkdir(parents=True, exist_ok=True)

        for group_keys, group_df in scores_df.groupby(group_cols):
            group_dict = dict(zip(group_cols, group_keys))
            ensemble = group_dict["ensemble"]
            loss_fn = group_dict["loss_function"]
            split_type = group_dict["split_type"]
            bv_reg_fn = group_dict["bv_reg_function"]

            max_idx = group_df["split_idx"].max()
            if pd.isna(max_idx):
                continue
            num_replicates = int(max_idx) + 1

            agg_frame_weights: list[np.ndarray | None] = []
            agg_pred_ln_pf: list[np.ndarray | None] = []

            for split_idx in range(num_replicates):
                rep_df = group_df[group_df["split_idx"] == split_idx].dropna(subset=[metric])
                if rep_df.empty:
                    agg_frame_weights.append(None)
                    agg_pred_ln_pf.append(None)
                    continue

                best_row_idx = rep_df[metric].idxmin() if direction == "min" else rep_df[metric].idxmax()
                best_row = rep_df.loc[best_row_idx]

                lookup_key = (
                    ensemble,
                    loss_fn,
                    split_type,
                    int(split_idx),
                    _float_key(best_row["maxent_value"]),
                    _float_key(best_row["bv_reg_value"]),
                    str(best_row["bv_reg_function"]),
                )

                run = run_lookup.get(lookup_key)
                if run is None:
                    print(f"WARNING: Processed run not found for key: {lookup_key}")
                    agg_frame_weights.append(None)
                    agg_pred_ln_pf.append(None)
                    continue

                run_path = Path(run["full_run_path"])
                weights_path = run_path / "frame_weights.npy"
                pf_path = run_path / "pred_ln_pf.npy"
                convergence_path = run_path / "convergence_thresholds.txt"

                conv_idx = _load_convergence_index(
                    convergence_path, float(best_row["convergence_value"])
                )
                if conv_idx is None:
                    print(
                        f"WARNING: convergence_value {best_row['convergence_value']} not found in "
                        f"{convergence_path}"
                    )
                    agg_frame_weights.append(None)
                    agg_pred_ln_pf.append(None)
                    continue

                if weights_path.exists():
                    weights = np.load(weights_path)
                    agg_frame_weights.append(
                        weights[conv_idx] if conv_idx < weights.shape[0] else None
                    )
                else:
                    agg_frame_weights.append(None)

                if pf_path.exists():
                    pred_ln_pf = np.load(pf_path)
                    agg_pred_ln_pf.append(
                        pred_ln_pf[conv_idx] if conv_idx < pred_ln_pf.shape[0] else None
                    )
                else:
                    agg_pred_ln_pf.append(None)

            first_w = next((x for x in agg_frame_weights if x is not None), None)
            first_pf = next((x for x in agg_pred_ln_pf if x is not None), None)

            final_weights = (
                np.stack(
                    [x if x is not None else np.full_like(first_w, np.nan) for x in agg_frame_weights]
                )
                if first_w is not None
                else np.array([])
            )
            final_pfs = (
                np.stack(
                    [x if x is not None else np.full_like(first_pf, np.nan) for x in agg_pred_ln_pf]
                )
                if first_pf is not None
                else np.array([])
            )

            save_name = f"{ensemble}_{loss_fn}_{split_type}_{bv_reg_fn}"
            out_path = metric_dir / f"{save_name}_selected.npz"
            np.savez(out_path, frame_weights=final_weights, pred_ln_pf=final_pfs)

    print("Extraction complete.")


if __name__ == "__main__":
    main()
