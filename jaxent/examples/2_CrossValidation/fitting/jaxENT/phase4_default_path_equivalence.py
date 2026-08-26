#!/usr/bin/env python3
"""Null-calibrated numerical-equivalence gate for the Phase-4 default path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
DEFAULT_OUTPUT = HERE / "_moprp_sigma_phase4_equivalence"
RESULT_NAME = "population_pivot_results.json"
MARGIN = 10.0
CRITERION = (
    "Pass iff structure and all non-numeric leaves match across every run; changed-run "
    "max_abs and max_rel spreads are each no greater than 10 times the corresponding "
    "baseline-run (null) spread; and baseline-vs-changed max_abs and max_rel are each no "
    "greater than 10 times the corresponding null spread. A zero null requires exact equality."
)
SUPERSEDED_BYTE_CHECK = {
    "reason": "Cross-process JAX/XLA float nondeterminism makes byte identity inapplicable.",
    "pre_existing_sha256": "17971aa7c77a7f6ff5abc3ed9833d8ab39e3879137d49fca23a32ceb351e5323",
    "no_flag_rerun_sha256": [
        "da2f7e45629869d33a0db634869977ad8c95c6e74b658e1b9ff8f31716f7ea9a",
        "663531fa8763e91e56aa465bdd80940d63d2243be24f315e9328c529f4f10733",
    ],
    "passed": False,
}


def _path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def flatten_numeric(payload: Any) -> dict[str, float]:
    """Return numeric JSON leaves keyed by dotted paths (booleans are non-numeric)."""
    flattened: dict[str, float] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, _path(path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, _path(path, str(index)))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[path] = float(value)

    walk(payload, "")
    return flattened


def flatten_nonnumeric(payload: Any) -> dict[str, Any]:
    """Capture JSON structure and every string, boolean, and null leaf."""
    flattened: dict[str, Any] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            flattened[path] = {"container": "dict", "keys": sorted(value)}
            for key, child in value.items():
                walk(child, _path(path, str(key)))
        elif isinstance(value, list):
            flattened[path] = {"container": "list", "length": len(value)}
            for index, child in enumerate(value):
                walk(child, _path(path, str(index)))
        elif not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            flattened[path] = {"leaf_type": type(value).__name__, "value": value}

    walk(payload, "")
    return flattened


def compare(a: Any, b: Any) -> dict[str, Any]:
    """Compare two JSON payloads, with structural mismatch outside tolerance."""
    a_numeric, b_numeric = flatten_numeric(a), flatten_numeric(b)
    structure_identical = (
        set(a_numeric) == set(b_numeric) and flatten_nonnumeric(a) == flatten_nonnumeric(b)
    )
    if set(a_numeric) != set(b_numeric) or not a_numeric:
        return {
            "max_abs": float("inf") if set(a_numeric) != set(b_numeric) else 0.0,
            "max_rel": float("inf") if set(a_numeric) != set(b_numeric) else 0.0,
            "argmax_path": None,
            "structure_identical": structure_identical,
        }
    differences = {path: abs(a_numeric[path] - b_numeric[path]) for path in a_numeric}
    argmax = max(differences, key=differences.get)
    relative = {
        path: difference / max(abs(a_numeric[path]), abs(b_numeric[path]))
        if max(abs(a_numeric[path]), abs(b_numeric[path])) > 0.0
        else 0.0
        for path, difference in differences.items()
    }
    return {
        "max_abs": differences[argmax],
        "max_rel": max(relative.values()),
        "argmax_path": argmax,
        "structure_identical": structure_identical,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_pairwise(payloads: list[Any]) -> dict[str, Any]:
    comparisons = [compare(payloads[i], payloads[j]) for i, j in combinations(range(len(payloads)), 2)]
    if not comparisons:
        raise ValueError("at least two runs are required per arm")
    result = max(comparisons, key=lambda item: item["max_abs"]).copy()
    result["max_abs"] = max(item["max_abs"] for item in comparisons)
    result["max_rel"] = max(item["max_rel"] for item in comparisons)
    result["structure_identical"] = all(item["structure_identical"] for item in comparisons)
    return result


def _within(value: float, null: float) -> bool:
    return value == 0.0 if null == 0.0 else value <= MARGIN * null


def _run(script: Path, output: Path, steps: int) -> float:
    started = time.monotonic()
    subprocess.run(
        ["uv", "run", "python", str(script), "--output-dir", str(output), "--steps", str(steps)],
        cwd=script.parent,
        check=True,
    )
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=int, default=2)
    parser.add_argument("--changed-runs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--baseline-ref", default="HEAD^")
    args = parser.parse_args()
    if args.baseline_runs < 2 or args.changed_runs < 2:
        parser.error("both run counts must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "equivalence_report.json"
    preregistration = {
        "status": "preregistered; changed runs have not been inspected",
        "criterion": CRITERION,
        "margin_factor": MARGIN,
        "steps": args.steps,
        "baseline_ref": args.baseline_ref,
    }
    report_path.write_text(json.dumps(preregistration, indent=2) + "\n")

    scratch = Path(tempfile.mkdtemp(prefix="phase4-equivalence-", dir=args.output_dir))
    worktree = Path(tempfile.mkdtemp(prefix="phase4-pristine-"))
    worktree.rmdir()
    added = False
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(worktree), args.baseline_ref], cwd=REPO_ROOT, check=True)
        added = True
        # Scientific inputs are intentionally gitignored. Overlay them without changing the
        # checked-out source revision whose default path is the baseline under test.
        shutil.copytree(
            REPO_ROOT / "jaxent/examples/2_CrossValidation/data",
            worktree / "jaxent/examples/2_CrossValidation/data",
            dirs_exist_ok=True,
        )
        for input_dir in (
            "_featurise_physics_v2",
            "_moprp_recovery_coefficient_lock",
            "_moprp_pivot_litmus",
            "_moprp_kint_sensitivity/expfact_recomputed",
        ):
            shutil.copytree(
                HERE / input_dir,
                worktree / "jaxent/examples/2_CrossValidation/fitting/jaxENT" / input_dir,
                dirs_exist_ok=True,
            )
        cluster_relative = Path(
            "jaxent/examples/2_CrossValidation/analysis/"
            "_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters/"
            "global_frame_to_cluster_ensemble.csv"
        )
        (worktree / cluster_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / cluster_relative, worktree / cluster_relative)
        ratios_relative = Path("jaxent/examples/2_CrossValidation/analysis/state_ratios.json")
        (worktree / ratios_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / ratios_relative, worktree / ratios_relative)
        baseline_script = worktree / "jaxent/examples/2_CrossValidation/fitting/jaxENT/moprp_population_pivot.py"
        changed_script = HERE / "moprp_population_pivot.py"
        paths: dict[str, list[Path]] = {"baseline": [], "changed": []}
        durations: dict[str, list[float]] = {"baseline": [], "changed": []}
        for arm, count, script in (
            ("baseline", args.baseline_runs, baseline_script),
            ("changed", args.changed_runs, changed_script),
        ):
            for index in range(count):
                output = scratch / f"{arm}-{index + 1}"
                durations[arm].append(_run(script, output, args.steps))
                paths[arm].append(output / RESULT_NAME)

        payloads = {arm: [json.loads(path.read_text()) for path in arm_paths] for arm, arm_paths in paths.items()}
        baseline_spread = _max_pairwise(payloads["baseline"])
        changed_spread = _max_pairwise(payloads["changed"])
        cross = compare(payloads["baseline"][0], payloads["changed"][0])
        all_structures = baseline_spread["structure_identical"] and changed_spread["structure_identical"] and all(
            compare(a, b)["structure_identical"] for a in payloads["baseline"] for b in payloads["changed"]
        )
        passed = all_structures and all(
            _within(value, null)
            for value, null in (
                (changed_spread["max_abs"], baseline_spread["max_abs"]),
                (changed_spread["max_rel"], baseline_spread["max_rel"]),
                (cross["max_abs"], baseline_spread["max_abs"]),
                (cross["max_rel"], baseline_spread["max_rel"]),
            )
        )
        report = {
            "criterion": CRITERION,
            "margin_factor": MARGIN,
            "baseline_ref": args.baseline_ref,
            "changed_ref": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "steps": args.steps,
            "run_sha256": {arm: [_sha256(path) for path in arm_paths] for arm, arm_paths in paths.items()},
            "run_seconds": durations,
            "baseline_spread": baseline_spread,
            "changed_spread": changed_spread,
            "baseline_vs_changed": cross,
            "structure_identical_across_all_runs": all_structures,
            "superseded_byte_check": SUPERSEDED_BYTE_CHECK,
            "passed": passed,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        if added:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO_ROOT, check=True)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
