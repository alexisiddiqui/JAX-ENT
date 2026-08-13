#!/usr/bin/env python3
"""Materialise and validate the MoPrP HDXrate/PDLA intrinsic-rate vector."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
from hdxrate import k_int_from_sequence

from jaxent.src.models.func.uptake import calculate_HDXrate_from_sequence


TEMPERATURE_K = 298.0
EFFECTIVE_PD = 4.4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vector(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path, dtype=float)
    return values[:, 0].astype(int), values[:, 1]


def comparison(reference_ids, reference, path: Path) -> dict[str, object]:
    ids, candidate = load_vector(path)
    if not np.array_equal(ids, reference_ids):
        raise ValueError(f"residue-ID mismatch in {path}")
    valid = (reference > 0) & (candidate > 0)
    delta = np.log(reference[valid] / candidate[valid])
    differing_sentinels = reference_ids[(reference < 0) != (candidate < 0)].tolist()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "n_positive_overlap": int(valid.sum()),
        "mean_ln_ratio": float(delta.mean()),
        "rms_ln_ratio": float(np.sqrt(np.mean(delta**2))),
        "max_abs_ln_ratio": float(np.max(np.abs(delta))),
        "sentinel_disagreement_residue_ids": differing_sentinels,
        "c_terminal_note": (
            "Residue 101 is compared without patching; any terminal-reference difference is retained."
        ),
    }


def array_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(reference) & np.isfinite(candidate) & (reference > 0) & (candidate > 0)
    delta = np.log(reference[valid] / candidate[valid])
    return {
        "n_positive_overlap": int(valid.sum()),
        "max_abs_rate_difference": float(np.max(np.abs(reference[valid] - candidate[valid]))),
        "max_abs_ln_ratio": float(np.max(np.abs(delta))),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = next(parent for parent in here.parents if (parent / ".git").exists())
    data = repo / "jaxent/examples/2_CrossValidation/data/_MoPrP"
    output_dir = here / "validated_hdxrate_pdla"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "moprp_hdxrate_pdla_pD4p4_298K_min.dat"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "validation_report.json"
    for path in (output, manifest_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    sequence_path = data / "moprp.seq"
    sequence = sequence_path.read_text().strip()
    wrapper_s = calculate_HDXrate_from_sequence(sequence, TEMPERATURE_K, EFFECTIVE_PD)
    direct_s = np.asarray(
        k_int_from_sequence(
            sequence,
            TEMPERATURE_K,
            EFFECTIVE_PD,
            reference="poly",
            exchange_type="HD",
            d_percentage=100.0,
            ph_correction=False,
        ),
        dtype=float,
    )
    np.testing.assert_array_equal(wrapper_s, direct_s)
    wrapper_min = calculate_HDXrate_from_sequence(
        sequence, TEMPERATURE_K, EFFECTIVE_PD, unit="min^-1"
    )
    np.testing.assert_array_equal(wrapper_min, wrapper_s * 60.0)

    residue_ids = np.arange(1, len(sequence) + 1, dtype=int)
    nonexchangeable = (
        ~np.isfinite(wrapper_s)
        | (wrapper_s <= 0)
        | np.asarray([aa == "P" for aa in sequence])
    )
    nonexchangeable[0] = True  # protein N terminus has no preceding peptide amide
    materialised = wrapper_min.copy()
    materialised[nonexchangeable] = -1.0
    with output.open("x") as handle:
        handle.write("# HDXrate 0.2.2 PDLA/poly rates; 298 K; effective pD 4.4; ph_correction=False.\n")
        handle.write("# Columns: one-based residue_id, intrinsic_rate_min^-1; -1 is non-exchangeable.\n")
        for residue_id, rate in zip(residue_ids, materialised, strict=True):
            handle.write(f"{residue_id} {rate:.17g}\n")

    comparisons = {
        "historical_expfact_equivalent": {
            **array_comparison(materialised, np.where(nonexchangeable, -1.0, direct_s * 60.0)),
            "source": "direct HDXrate reference='poly' (the historical PDLA-equivalent model)",
            "interpretation": "exact executable model comparison before any file formatting",
        },
        "moprp_shipped": comparison(residue_ids, materialised, data / "moprp.kint"),
        "current_3ala": comparison(
            residue_ids, materialised, data / "expfact_kint_pH4p4_298K_min.dat"
        ),
    }
    report = {
        "wrapper_equals_direct_before_conversion": True,
        "minute_values_equal_seconds_times_60": True,
        "residue_ids": {"first": 1, "last": len(sequence), "count": len(sequence)},
        "proline_residue_ids": residue_ids[np.asarray([aa == "P" for aa in sequence])].tolist(),
        "sentinel_residue_ids": residue_ids[materialised < 0].tolist(),
        "comparisons": comparisons,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "source": "JAX-ENT HDXrate/PDLA validated arm",
        "hdxrate_version": importlib.metadata.version("HDXrate"),
        "jaxent_commit": commit,
        "sequence": {"path": str(sequence_path.resolve()), "sha256": sha256(sequence_path)},
        "temperature_k": TEMPERATURE_K,
        "pD": EFFECTIVE_PD,
        "pD_convention": "effective pD supplied unchanged; no automatic +0.4 correction",
        "reference": "poly",
        "exchange_type": "HD",
        "d_percentage": 100.0,
        "ph_correction": False,
        "native_units": "s^-1",
        "output_units": "min^-1",
        "conversion": "multiply by 60 exactly",
        "residue_rule": "one-based IDs matching moprp.seq",
        "sentinel_rule": "-1 for HDXrate nonpositive residues and proline; values never logged",
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
        "validation_report": {"path": str(report_path.resolve()), "sha256": sha256(report_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
