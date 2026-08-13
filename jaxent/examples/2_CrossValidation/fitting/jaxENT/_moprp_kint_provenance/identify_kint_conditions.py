"""Identify MoPrP rate-file conditions with upstream exPfact's own kint code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


TEMPERATURES_K = (278.0, 288.0, 293.0, 298.0, 310.0)
PHS = (2.5, 4.0, 4.4, 6.5, 7.0, 7.4)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_upstream_kint(expfact_root: Path):
    python_dir = expfact_root / "python"
    spec = importlib.util.spec_from_file_location("upstream_expfact_kint", python_dir / "kint.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load upstream exPfact kint.py")
    import sys

    sys.path.insert(0, str(python_dir))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.calculate_kint_for_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expfact-root", type=Path, required=True)
    parser.add_argument("--expfact-revision", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = args.data_dir / "moprp.seq"
    shipped_path = args.data_dir / "moprp.kint"
    canonical_path = args.data_dir / "expfact_kint_pH4p4_298K_min.dat"
    sequence = sequence_path.read_text().strip()
    shipped = np.loadtxt(shipped_path, dtype=float)[:, 1]
    canonical = np.loadtxt(canonical_path, dtype=float)[:, 1]
    calculate = load_upstream_kint(args.expfact_root)

    rows: list[dict[str, float | int]] = []
    for temperature_k in TEMPERATURES_K:
        for ph in PHS:
            rates_hr, _ = calculate(1, len(sequence), sequence, temperature_k, ph)
            rates_min = np.asarray(rates_hr, dtype=float) / 60.0
            valid = (rates_min > 0) & (shipped > 0)
            log_ratio = np.log(rates_min[valid] / shipped[valid])
            rows.append(
                {
                    "temperature_k": temperature_k,
                    "ph": ph,
                    "n_compared": int(np.sum(valid)),
                    "mean_ln_ratio_to_moprp_kint": float(np.mean(log_ratio)),
                    "sd_ln_ratio_to_moprp_kint": float(np.std(log_ratio)),
                    "rms_ln_ratio_to_moprp_kint": float(np.sqrt(np.mean(log_ratio**2))),
                    "max_abs_ln_ratio_to_moprp_kint": float(np.max(np.abs(log_ratio))),
                }
            )

    csv_path = args.output_dir / "condition_grid.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    control_hr, _ = calculate(1, len(sequence), sequence, 298.0, 4.4)
    control_min = np.asarray(control_hr, dtype=float) / 60.0
    control_valid = (control_min > 0) & (canonical > 0)
    best = min(rows, key=lambda row: float(row["rms_ln_ratio_to_moprp_kint"]))
    script_path = Path(__file__).resolve()
    manifest = {
        "analysis": "MoPrP intrinsic-rate condition grid",
        "expfact_root": str(args.expfact_root.resolve()),
        "expfact_revision": args.expfact_revision,
        "rate_convention": "upstream calculate_kint_for_sequence output divided by 60 (hr^-1 to min^-1)",
        "temperature_grid_k": TEMPERATURES_K,
        "ph_grid": PHS,
        "canonical_control": {
            "temperature_k": 298.0,
            "ph": 4.4,
            "n_compared": int(np.sum(control_valid)),
            "max_abs_rate_difference_min_inverse": float(
                np.max(np.abs(control_min[control_valid] - canonical[control_valid]))
            ),
            "max_abs_ln_ratio": float(
                np.max(np.abs(np.log(control_min[control_valid] / canonical[control_valid])))
            ),
        },
        "best_grid_point_for_moprp_kint": best,
        "conclusion": "no grid match" if float(best["rms_ln_ratio_to_moprp_kint"]) > 1e-6 else "grid match",
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in (sequence_path, shipped_path, canonical_path, args.expfact_root / "python/kint.py", args.expfact_root / "python/constants_HD.py")
        },
        "outputs": {str(csv_path.resolve()): sha256(csv_path)},
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
