#!/usr/bin/env python3
"""Materialise HDXrate poly (PDLA) intrinsic rates for MoPrP at 298 K, pD 4.0, in min^-1.

Rates are computed on the MoPrP109 construct sequence (the protein in solution, so the
true termini) and written in ``moprp.seq`` numbering (MoPrP109 resid - 1).  pD 4.0 is
supplied unchanged (no +0.4 glass-electrode correction).  -1 marks prolines.

Evidence for this choice is in ``../_moprp_kint_scale_check``: with ``median.pfact`` these
rates reproduce ``moprp.dexp`` with no global correction (best ln shift +0.04), and they
match the shipped ``moprp.kint`` and the ``nmr.csv`` kint scale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jaxent.src.models.func.uptake import calculate_HDXrate_from_sequence

TEMPERATURE_K = 298.0
PD = 4.0
C_TERMINAL_EXTENSION_109 = "YYDGRRS"


def main() -> None:
    here = Path(__file__).resolve().parent
    data = here.parents[2] / "data" / "_MoPrP"
    output = data / "hdxrate_poly_pD4p0_298K_min.dat"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")

    sequence = (data / "moprp.seq").read_text().strip()
    construct = "G" + sequence + C_TERMINAL_EXTENSION_109
    rates = calculate_HDXrate_from_sequence(construct, TEMPERATURE_K, PD, unit="min^-1")
    rates = np.asarray(rates, dtype=float)[1 : len(sequence) + 1]
    proline = np.array([aa == "P" for aa in sequence])
    rates[proline | ~np.isfinite(rates) | (rates <= 0)] = -1.0

    with output.open("x") as handle:
        handle.write("# HDXrate poly (PDLA) intrinsic rates; 298 K; pD 4.0 (ph_correction=False).\n")
        handle.write("# Computed on the MoPrP109 construct sequence; numbered as moprp.seq (MoPrP109 resid - 1).\n")
        handle.write("# Columns: residue_id, intrinsic_rate_min^-1; -1 is non-exchangeable (proline).\n")
        for residue_id, rate in enumerate(rates, start=1):
            handle.write(f"{residue_id} {rate:.17g}\n")

    manifest = {
        "provider": "HDXrate",
        "reference": "poly",
        "temperature_k": TEMPERATURE_K,
        "pD": PD,
        "ph_correction": False,
        "units": "min^-1",
        "construct_sequence": construct,
        "numbering": "moprp.seq one-based (MoPrP109 resid - 1)",
        "output": {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
        "median_rate_min^-1": float(np.median(rates[rates > 0])),
    }
    (here / "hdxrate_poly_pD4p0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
