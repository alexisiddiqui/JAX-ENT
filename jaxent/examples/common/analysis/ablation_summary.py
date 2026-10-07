"""Manifest-driven optimizer-ablation selection and gate reporting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FIXED_SETTINGS = (
    "execution_mode",
    "step_chunk_size",
    "reset_threshold_cooldown_on_oscillation",
)
_RUN_RE = re.compile(
    r"^(?P<ensemble>.+?)_(?P<loss>mcMSE|MSE|Sigma_MSE)_(?P<split>.+?)"
    r"_split(?P<idx>\d+)_maxent(?P<maxent>[-+0-9.eE]+)"
    r"(?:_bvreg(?P<bv>[-+0-9.eE]+))?(?:_bvregfn(?P<reg>[A-Za-z0-9]+))?"
    r"_(?:results|config)$"
)


def _manifest_paths(roots: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        path = Path(root)
        paths.extend(path.rglob("prelaunch_manifest.json") if path.is_dir() else [path])
    return sorted(set(paths))


def _load_manifests(paths: list[Path]) -> list[dict]:
    manifests = []
    for path in paths:
        data = json.loads(path.read_text())
        data["_manifest_path"] = str(path)
        manifests.append(data)
    if not manifests:
        raise ValueError("No prelaunch_manifest.json files were found")

    for key in ("commit", "lockfile_sha256"):
        values = {m.get("runtime", {}).get(key) for m in manifests}
        if len(values) != 1:
            raise ValueError(f"Inconsistent manifest runtime setting: {key}: {values}")
    for key in FIXED_SETTINGS:
        values = {m.get("factors", {}).get(key) for m in manifests}
        if len(values) != 1:
            raise ValueError(f"Inconsistent fixed setting: {key}: {values}")
    return manifests


def _score_paths(manifest: dict) -> list[Path]:
    root = Path(manifest["_manifest_path"]).parent
    explicit = manifest.get("output_paths", {}).get("scores")
    paths = [Path(explicit)] if explicit else []
    paths.extend(root.rglob("model_scores.csv"))
    processed = root.parent / f"_processed_{root.name}"
    paths.extend(processed.rglob("model_scores.csv"))
    return sorted({path for path in paths if path.exists()})


def _read_scores(manifests: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for manifest in manifests:
        paths = _score_paths(manifest)
        if not paths:
            diagnostics.append({"kind": "missing_scores", "manifest": manifest["_manifest_path"]})
            continue
        for path in paths:
            frame = pd.read_csv(path)
            frame["cell"] = Path(manifest["_manifest_path"]).parent.name
            frame["manifest_path"] = manifest["_manifest_path"]
            for key, value in manifest.get("factors", {}).items():
                if key not in frame:
                    frame[key] = value
            frame["example"] = manifest.get("example")
            frame["commit"] = manifest.get("runtime", {}).get("commit")
            frame["lockfile_sha256"] = manifest.get("runtime", {}).get("lockfile_sha256")
            frames.append(frame)
    if not frames:
        raise ValueError("No model_scores.csv files were found")
    return pd.concat(frames, ignore_index=True), diagnostics


def _selection_key_columns(df: pd.DataFrame) -> list[str]:
    columns = ["cell", "example", "ensemble", "split_type", "split_idx", "loss_function", "maxent_value"]
    for column in ("bv_reg_value", "bv_reg_function", "bv_value"):
        if column in df.columns:
            columns.append(column)
    return [column for column in columns if column in df.columns]


def select_by_validation_mse(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select finite convergence minima, with deterministic audit tie-breaks."""
    work = df.copy()
    work["val_mse"] = pd.to_numeric(work["val_mse"], errors="coerce")
    work = work[np.isfinite(work["val_mse"])].copy()
    if work.empty:
        return work, []
    group = _selection_key_columns(work)
    tie_warnings: list[str] = []
    for keys, rows in work.groupby(group, dropna=False, sort=False):
        if rows["val_mse"].eq(rows["val_mse"].min()).sum() > 1:
            tie_warnings.append(f"exact validation-MSE tie for {keys}")

    def numeric_column(*names: str) -> pd.Series:
        for name in names:
            if name in work:
                return pd.to_numeric(work[name], errors="coerce")
        return pd.Series(np.nan, index=work.index)

    work["_conv_sort"] = numeric_column("convergence_threshold", "convergence_value")
    work["_maxent_sort"] = numeric_column("maxent_value")
    work["_bv_sort"] = numeric_column("bv_reg_value", "bv_value")
    work = work.sort_values(
        group + ["val_mse", "_conv_sort", "_maxent_sort", "_bv_sort"],
        ascending=[True] * len(group) + [True, False, True, True],
        kind="stable",
    )
    return (
        work.drop_duplicates(group, keep="first")
        .drop(columns=["_conv_sort", "_maxent_sort", "_bv_sort"], errors="ignore")
        .reset_index(drop=True),
        tie_warnings,
    )


def _summary(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "n": int(numeric.size),
        "mean": float(np.mean(numeric)) if numeric.size else np.nan,
        "sample_variance": float(np.var(numeric, ddof=1)) if numeric.size > 1 else np.nan,
        "sd": float(np.std(numeric, ddof=1)) if numeric.size > 1 else np.nan,
    }


def _family_key(row: pd.Series) -> tuple:
    reg = row.get("bv_reg_function")
    if pd.isna(reg):
        reg = None
    return (
        row.get("ensemble"), row.get("loss_function"), row.get("split_type"),
        int(row.get("split_idx")) if pd.notna(row.get("split_idx")) else None,
        float(row.get("maxent_value")) if pd.notna(row.get("maxent_value")) else None,
        float(row.get("bv_reg_value")) if pd.notna(row.get("bv_reg_value")) else None,
        reg,
    )


def _parse_run_name(path: Path) -> tuple | None:
    match = _RUN_RE.match(path.stem)
    if not match:
        return None
    values = match.groupdict()
    return (
        values["ensemble"], values["loss"], values["split"], int(values["idx"]),
        float(values["maxent"]), float(values["bv"]) if values["bv"] is not None else None,
        values["reg"],
    )


def _configured_run_key(path: Path) -> tuple | None:
    """Parse a run name, replacing one-decimal BV labels with config values."""
    key = _parse_run_name(path)
    if key is None:
        return None
    config_path = path if path.name.endswith("_config.json") else path.with_name(path.name.replace("_results.hdf5", "_config.json"))
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            value = config.get("loss_config", {}).get("bv_reg_scaling")
            if value is not None and key[5] is not None:
                key = (*key[:5], float(value), key[6])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return key


def _grid_diagnostics(manifest: dict, scores: pd.DataFrame) -> dict:
    root = Path(manifest["_manifest_path"]).parent
    grid = manifest.get("grid", {})
    bv_values = grid.get("bv_values") or [None]
    regs = grid.get("bv_reg_functions") or [None]
    expected = {
        (ensemble, loss, split, idx, float(maxent),
         None if bv is None else float(bv), reg)
        for ensemble in grid.get("ensembles", [])
        for loss in grid.get("losses", [])
        for split in grid.get("split_types", [])
        for idx in range(int(grid.get("num_splits", 0)))
        for maxent in grid.get("maxent_values", [])
        for bv in bv_values
        for reg in regs
    }
    hdf_keys = {_configured_run_key(path) for path in root.rglob("*_results.hdf5")}
    cfg_keys = {_configured_run_key(path) for path in root.rglob("*_config.json")}
    hdf_keys.discard(None)
    cfg_keys.discard(None)

    score_keys = set()
    for _, row in scores.iterrows():
        score_keys.add(_family_key(row))
    expected_family = {
        (a, b, c, d, e, f, g) for a, b, c, d, e, f, g in expected
    }
    finite_family = set()
    if not scores.empty:
        finite = scores[np.isfinite(pd.to_numeric(scores["val_mse"], errors="coerce"))]
        finite_family = {_family_key(row) for _, row in finite.iterrows()}
    missing_hdf = sorted(expected - hdf_keys, key=str)
    missing_cfg = sorted(expected - cfg_keys, key=str)
    missing_score = sorted(expected_family - score_keys, key=str)
    nonfinite = sorted((expected_family & hdf_keys & cfg_keys) - score_keys, key=str)
    return {
        "manifest": manifest["_manifest_path"],
        "expected_fit_count": int(manifest.get("expected_fit_count", len(expected))),
        "hdf_count": len(list(root.rglob("*_results.hdf5"))),
        "config_count": len(list(root.rglob("*_config.json"))),
        "score_rows": int(len(scores)),
        "missing_hdf_count": len(missing_hdf),
        "missing_config_count": len(missing_cfg),
        "missing_score_family_count": len(missing_score),
        "nonfinite_family_count": len(nonfinite),
        "missing_hdf_families": json.dumps(missing_hdf),
        "missing_config_families": json.dumps(missing_cfg),
        "missing_score_families": json.dumps(missing_score),
        "nonfinite_families": json.dumps(nonfinite),
        "complete": not (missing_hdf or missing_cfg or missing_score or nonfinite),
        "maxent_count": len(grid.get("maxent_values", [])),
        "bv_count": len(grid.get("bv_values", [])),
    }


def _paired_rows(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    specs = {
        1: [("loss", "Sigma_MSE", "MSE", True), ("ensemble", "ISO_BI", "ISO_TRI", True)],
        2: [("loss", "Sigma_MSE", "MSE", True), ("ensemble", "AF2_filtered", "AF2_MSAss", True)],
        3: [("loss", "MSE", "Sigma_MSE", True), ("ensemble", "AF2_filtered", "AF2_MSAss", False)],
    }
    for (cell, example), cell_rows in selected.groupby(["cell", "example"], dropna=False):
        example = int(example)
        for gate_type, left, right, hard_gate in specs.get(example, []):
            if gate_type == "loss":
                group_cols = ["cell", "ensemble", "split_type", "split_idx"]
                pivot_index = group_cols
                pivot = cell_rows.pivot_table(index=pivot_index, columns="loss_function", values="recovery_percent", aggfunc="first")
            else:
                group_cols = ["cell", "loss_function", "split_type", "split_idx"]
                pivot_index = group_cols
                pivot = cell_rows.pivot_table(index=pivot_index, columns="ensemble", values="recovery_percent", aggfunc="first")
            if left not in pivot or right not in pivot:
                continue
            for index, value in (pivot[left] - pivot[right]).dropna().items():
                item = dict(zip(pivot_index, index if isinstance(index, tuple) else (index,)))
                item.update({
                    "cell": cell,
                    "example": example,
                    "gate_type": gate_type,
                    "gate": f"{left} - {right}",
                    "hard_gate": hard_gate,
                    "left": left,
                    "right": right,
                    "difference": float(value),
                })
                rows.append(item)
    return pd.DataFrame(rows)


def _gate_statistics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if pairs.empty:
        return pd.DataFrame()
    for keys, group in pairs.groupby(["cell", "example", "gate_type", "gate", "hard_gate"], dropna=False):
        cell, example, gate_type, gate, hard_gate = keys
        summary = _summary(group["difference"])
        split_means = group.groupby("split_type")["difference"].mean().to_dict()
        pooled_positive = bool(summary["mean"] > 0)
        split_positive = bool(split_means) and all(value > 0 for value in split_means.values())
        status = "clean pass" if pooled_positive and split_positive else ("fragile" if pooled_positive else "fail")
        loo = {}
        for split_type, split_idx in group[["split_type", "split_idx"]].drop_duplicates().itertuples(index=False):
            keep = ~((group["split_type"] == split_type) & (group["split_idx"] == split_idx))
            loo[f"loo_{split_type}_{split_idx}_mean"] = float(group.loc[keep, "difference"].mean()) if keep.any() else np.nan
        depends = any(value <= 0 for value in loo.values() if np.isfinite(value))
        rows.append({
            "cell": cell, "example": example, "gate_type": gate_type, "gate": gate,
            "hard_gate": hard_gate, "status": status, "pooled_mean": summary["mean"],
            "pooled_sample_variance": summary["sample_variance"], "pooled_sd": summary["sd"],
            "count": summary["n"], "depends_on_one_split": depends,
            **{f"{key}_mean": value for key, value in split_means.items()}, **loo,
        })
    return pd.DataFrame(rows)


def _cell_code(name: str) -> str | None:
    match = re.search(r"(?:^|_)([AB]\d+)(?:_|$)", name)
    return match.group(1) if match else None


def _matched_pair_lines(gate_statistics: pd.DataFrame) -> list[str]:
    """Render the requested matched-cell gate comparisons for the audit report."""
    lines: list[str] = []
    pair_specs = (("A0", "A1"), ("B1", "A2"), ("B2", "A3"), ("B3", "A4"))
    if gate_statistics.empty:
        return ["- No matched-cell gate statistics were available."]
    stats = gate_statistics.copy()
    stats["cell_code"] = stats["cell"].map(_cell_code)
    for left, right in pair_specs:
        left_rows = stats[stats["cell_code"] == left]
        right_rows = stats[stats["cell_code"] == right]
        if left_rows.empty or right_rows.empty:
            lines.append(f"- {left}/{right}: unavailable (one or both cells were not supplied).")
            continue
        joined = left_rows.merge(
            right_rows,
            on=["example", "gate_type", "gate"],
            suffixes=("_left", "_right"),
        )
        if joined.empty:
            lines.append(f"- {left}/{right}: no common gates were available.")
            continue
        comparisons = "; ".join(
            f"Ex{int(row.example)} {row.gate}: {row.status_left} ({row.pooled_mean_left:.6g}) vs {row.status_right} ({row.pooled_mean_right:.6g})"
            for row in joined.itertuples()
        )
        lines.append(f"- {left}/{right}: {comparisons}.")
    return lines


def _wall_time_summary(manifest: dict) -> str:
    values: list[float] = []
    root = Path(manifest["_manifest_path"]).parent
    for path in root.rglob("*_config.json"):
        try:
            value = json.loads(path.read_text()).get("runtime", {}).get("wall_time_seconds")
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not values:
        return "wall time unavailable"
    return f"wall time mean/max {np.mean(values):.3f}/{np.max(values):.3f}s"


def summarize_ablation(
    roots: list[str | Path], output_dir: str | Path, *, example: int | None = None
) -> dict[str, pd.DataFrame | str]:
    manifests = _load_manifests(_manifest_paths(roots))
    if example is not None:
        manifests = [m for m in manifests if m.get("example") == example]
    scores, diagnostics = _read_scores(manifests)
    selected_convergence, tie_warnings = select_by_validation_mse(scores)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # First select the convergence checkpoint for each fitted hyperparameter
    # model, then select MaxEnt/BV strictly by finite validation MSE per cell.
    across = [c for c in ["cell", "example", "ensemble", "split_type", "split_idx", "loss_function"] if c in selected_convergence]
    work = selected_convergence.copy()
    work = work.sort_values(across + ["val_mse"], kind="stable")
    selected_models = work.drop_duplicates(across, keep="first").reset_index(drop=True)
    selected_models["convergence_value"] = selected_models.get("convergence_value", selected_models.get("convergence_threshold"))
    for col in ("bv_reg_value", "bv_value"):
        if col in selected_models and "bv_reg_value" not in selected_models:
            selected_models["bv_reg_value"] = selected_models[col]

    group_rows: list[dict] = []
    for keys, group in selected_models.groupby([c for c in ["cell", "example", "ensemble", "loss_function", "split_type"] if c in selected_models], dropna=False):
        names = [c for c in ["cell", "example", "ensemble", "loss_function", "split_type"] if c in selected_models]
        row = dict(zip(names, keys if isinstance(keys, tuple) else (keys,)))
        rec = _summary(group["recovery_percent"])
        val = _summary(group["val_mse"])
        row.update({"n": rec["n"], "recovery_mean": rec["mean"], "recovery_sample_variance": rec["sample_variance"], "recovery_sd": rec["sd"], "val_mse_mean": val["mean"], "val_mse_sample_variance": val["sample_variance"], "candidate_count": int(len(scores[(scores["cell"] == row.get("cell")) & (scores["ensemble"] == row.get("ensemble")) & (scores["loss_function"] == row.get("loss_function")) & (scores["split_type"] == row.get("split_type"))]))})
        group_rows.append(row)
    group_statistics = pd.DataFrame(group_rows)
    gate_pairs = _paired_rows(selected_models)
    gate_statistics = _gate_statistics(gate_pairs)
    completeness = []
    for manifest in manifests:
        completeness.append(_grid_diagnostics(manifest, scores[scores["manifest_path"] == manifest["_manifest_path"]]))
    grid_completeness = pd.DataFrame(completeness)

    selected_models.to_csv(output / "selected_models.csv", index=False)
    group_statistics.to_csv(output / "group_statistics.csv", index=False)
    gate_pairs.to_csv(output / "gate_pairs.csv", index=False)
    gate_statistics.to_csv(output / "gate_statistics.csv", index=False)
    grid_completeness.to_csv(output / "grid_completeness.csv", index=False)

    report = [
        "# Optimizer ablation report", "",
        f"Manifests: {len(manifests)}; finite convergence selections: {len(selected_convergence)}; selected models: {len(selected_models)}.",
        "Selection criterion: minimum finite validation MSE first at convergence, then across MaxEnt/BV per fitted model family. Validation loss, recovery, terminal state, and optimizer best state were not used.",
        "", "## Cell and runtime summary", "",
    ]
    for manifest in manifests:
        runtime = manifest.get("runtime", {})
        factors = manifest.get("factors", {})
        report.append(f"- {Path(manifest['_manifest_path']).parent.name}: example {manifest.get('example')}, commit {runtime.get('commit')}, backend {runtime.get('backend')}, jobs {factors.get('parallel_jobs')}; expected fits {manifest.get('expected_fit_count')}; {_wall_time_summary(manifest)}.")
    report += ["", "## Gate results", ""]
    if gate_statistics.empty:
        report.append("- No complete gate pairs were available.")
    else:
        for _, row in gate_statistics.sort_values(["cell", "gate"]).iterrows():
            report.append(f"- {row['cell']} {row['gate']}: {row['status']}; pooled mean {row['pooled_mean']:.6g}; leave-one-block dependence={bool(row['depends_on_one_split'])}.")
    report += ["", "## Matched cell pairs", ""]
    report.extend(_matched_pair_lines(gate_statistics))
    report += ["", "## Selected distributions", ""]
    for column in ("convergence_value", "maxent_value", "bv_reg_value"):
        if column in selected_models:
            report.append(f"- {column}: {selected_models[column].value_counts(dropna=False).to_dict()}")
    report += ["", "## Diagnostics", ""]
    for item in diagnostics:
        report.append(f"- {item['kind']}: {item['manifest']}")
    for warning in tie_warnings:
        report.append(f"- WARNING: {warning}")
    report += ["", "No claims of statistical significance are made."]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return {"selected_models": selected_models, "group_statistics": group_statistics, "gate_pairs": gate_pairs, "gate_statistics": gate_statistics, "grid_completeness": grid_completeness, "report": str(output / "report.md")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--example", type=int)
    args = parser.parse_args()
    summarize_ablation(args.root, args.output_dir, example=args.example)


if __name__ == "__main__":
    main()
