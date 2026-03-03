#!/usr/bin/env python3
"""Thin wrapper for mixed linear model score analysis (Experiment 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jaxent.examples.common.analysis import run_analysis_on_subset
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.plotting import setup_publication_style

MARKER_LIST = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X", "d"]


def _resolve_path(script_dir: Path, path_value: str, absolute_paths: bool) -> Path:
    return Path(path_value) if absolute_paths else (script_dir / path_value).resolve()


def _resolve_base_output_dir(
    script_dir: Path,
    scores_csv_path: Path,
    output_dir_arg: str | None,
    absolute_paths: bool,
) -> Path:
    if output_dir_arg:
        return _resolve_path(script_dir, output_dir_arg, absolute_paths)

    scores_parent_dir = scores_csv_path.parent
    scores_parent_basename = scores_parent_dir.name
    if scores_parent_basename.startswith("_scores_"):
        return scores_parent_dir.parent / f"_analysis_{scores_parent_basename}"
    return scores_parent_dir / f"_analysis_{scores_csv_path.stem}"


def _filter_best_convergence(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ensemble", "split_type", "split_idx", "maxent_value"]
    for col in ("loss_function", "bv_reg_value", "bv_reg_function"):
        if col in df.columns:
            group_cols.append(col)

    missing_cols = [c for c in group_cols + ["val_loss"] if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Cannot filter. Missing columns: {missing_cols}")
        return df

    out = df.copy()
    out["val_loss"] = pd.to_numeric(out["val_loss"], errors="coerce")
    out = out.sort_values("val_loss", ascending=True, na_position="last")
    out = out.drop_duplicates(subset=group_cols, keep="first")
    return out.sort_index()


def _build_subset_map(df: pd.DataFrame, analyze_subsets: bool) -> dict[str, pd.DataFrame]:
    subsets: dict[str, pd.DataFrame] = {"whole_dataset": df}
    if not analyze_subsets:
        return subsets

    split_cols: list[str] = []
    if "loss_function" in df.columns:
        split_cols.append("loss_function")
    if "bv_reg_function" in df.columns:
        split_cols.append("bv_reg_function")

    if not split_cols:
        return subsets

    for _, row in df[split_cols].drop_duplicates().iterrows():
        mask = pd.Series(True, index=df.index)
        name_parts: list[str] = []
        for col in split_cols:
            val = row[col]
            mask &= df[col] == val
            name_parts.append(str(val))
        subsets["_".join(name_parts)] = df[mask]

    return subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse mixed linear model score metrics (Experiment 1).")
    parser.add_argument(
        "--scores-csv-path",
        default="../fitting/jaxENT/_processed__optimise_test_SIGMA_500__20260216_224925/_scores__processed__optimise_test_SIGMA_500__20260216_224925/model_scores.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-metric", default="recovery_percent")
    parser.add_argument("--absolute-paths", action="store_true", default=False)
    parser.add_argument("--filter-mode", choices=["both", "unfiltered", "filtered"], default="both")
    parser.add_argument("--analyze-subsets", action="store_true", default=False)
    parser.add_argument("--config", default=None, help="Path to experiment config yaml (default: ../config.yaml).")
    args = parser.parse_args()

    setup_publication_style()

    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent
    config_path = Path(args.config).resolve() if args.config else exp_dir / "config.yaml"
    config = ExperimentConfig.from_yaml(config_path)

    scores_csv_path = _resolve_path(script_dir, args.scores_csv_path, args.absolute_paths)
    if not scores_csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")

    base_output_dir = _resolve_base_output_dir(
        script_dir=script_dir,
        scores_csv_path=scores_csv_path,
        output_dir_arg=args.output_dir,
        absolute_paths=args.absolute_paths,
    )

    print("=" * 80)
    print("JAX-ENT Linear Effects Modelling Analysis")
    print("=" * 80)
    print(f"Resolved scores_csv_path: {scores_csv_path}")
    print(f"Base output dir: {base_output_dir}")
    print(f"Target metric: {args.target_metric}")
    print(f"Filter mode: {args.filter_mode}")
    print("-" * 80)

    df_master = pd.read_csv(scores_csv_path)
    print(f"\nLoaded {len(df_master)} rows from {scores_csv_path}")
    print("\nDataFrame info:")
    df_master.info()
    print("\nDataFrame head:")
    print(df_master.head())
    print("-" * 80)

    modes = [False, True] if args.filter_mode == "both" else [args.filter_mode == "filtered"]

    for filter_best in modes:
        run_output_dir = Path(f"{base_output_dir}_filtered") if filter_best else base_output_dir
        run_output_dir.mkdir(parents=True, exist_ok=True)

        df = _filter_best_convergence(df_master) if filter_best else df_master.copy()
        df_clean = df.dropna(subset=[args.target_metric]).copy()
        if df_clean.empty:
            print(f"No data remaining after dropping NaNs in '{args.target_metric}'")
            continue

        for subset_name, subset_df in _build_subset_map(df_clean, args.analyze_subsets).items():
            if subset_df.empty:
                continue
            safe_name = "".join(c for c in subset_name if c.isalnum() or c in ("_", "-")).rstrip()
            subset_output_dir = run_output_dir / safe_name
            subset_output_dir.mkdir(parents=True, exist_ok=True)
            run_analysis_on_subset(
                subset_df.copy(),
                target_metric=args.target_metric,
                output_dir=str(subset_output_dir),
                style=config.style,
                marker_list=MARKER_LIST,
            )


if __name__ == "__main__":
    main()
