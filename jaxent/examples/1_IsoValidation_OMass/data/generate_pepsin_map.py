#!/usr/bin/env python3
"""Build a deterministic, sequence-only synthetic pepsin map for TeaA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TOP = (
    HERE
    / "_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H",
    "HIP": "H", "GLH": "E", "ASH": "D", "LYN": "K", "CYX": "C",
    "CYM": "C", "MSE": "M",
}
W_P1 = {
    "F": 1.00, "L": 1.00, "W": 0.90, "Y": 0.80, "M": 0.60,
    "A": 0.40, "V": 0.35, "I": 0.35, "E": 0.30, "Q": 0.25,
    "C": 0.20, "T": 0.15, "S": 0.12, "N": 0.12, "G": 0.10,
    "H": 0.12, "D": 0.08, "K": 0.06, "R": 0.05, "P": 0.00,
}
W_P1P = {
    "F": 0.90, "L": 0.90, "W": 0.80, "Y": 0.70, "M": 0.50,
    "A": 0.40, "V": 0.35, "I": 0.35, "E": 0.30, "Q": 0.25,
    "C": 0.20, "T": 0.18, "S": 0.15, "N": 0.15, "G": 0.15,
    "H": 0.15, "D": 0.10, "K": 0.08, "R": 0.08, "P": 0.00,
}
PRO_NEIGHBOUR_FACTOR = 0.15


def sequence(topology: Path) -> tuple[str, np.ndarray]:
    alpha = mda.Universe(str(topology)).select_atoms("protein and name CA")
    return "".join(THREE_TO_ONE.get(name, "X") for name in alpha.resnames), alpha.resids.astype(int)


def bond_propensity(seq: str) -> np.ndarray:
    weights = np.zeros(len(seq) - 1)
    for i, (p1, p1p) in enumerate(zip(seq[:-1], seq[1:], strict=True)):
        value = W_P1.get(p1, 0.10) * W_P1P.get(p1p, 0.15)
        if p1 == "P" or p1p == "P":
            value = 0.0
        else:
            if i and seq[i - 1] == "P":
                value *= PRO_NEIGHBOUR_FACTOR
            if i + 2 < len(seq) and seq[i + 2] == "P":
                value *= PRO_NEIGHBOUR_FACTOR
        weights[i] = value
    return weights


def calibrate(weights: np.ndarray, target_length: float) -> float:
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        spacing = len(weights) / max(np.clip(weights * mid, 0, 1).sum(), 1e-9)
        if abs(spacing - target_length) < 1e-4:
            break
        if spacing > target_length:
            lo = mid
        else:
            hi = mid
    return mid


def digest(probabilities, rng, min_length, max_length):
    cuts = np.flatnonzero(rng.random(len(probabilities)) < probabilities)
    boundaries = np.concatenate(([-1], cuts, [len(probabilities)]))
    return [
        (start + 1, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if min_length <= stop - start <= max_length
    ]


def observable_indices(seq: str, start: int, stop: int) -> np.ndarray:
    """Indices represented after N-terminal trim and proline exclusion."""
    return np.asarray([i for i in range(start + 1, stop + 1) if seq[i] != "P"], dtype=int)


def build_map(seq: str, args):
    weights = bond_propensity(seq)
    scale = calibrate(weights, args.mean_length)
    probabilities = np.clip(weights * scale, 0, 1)
    rng = np.random.default_rng(args.seed)
    coverage = np.zeros(len(seq), dtype=int)
    peptides, seen, digests = [], set(), 0
    observable_mask = np.asarray([letter != "P" for letter in seq])
    observable_mask[0] = False
    while coverage[observable_mask].mean() < args.coverage and digests < args.max_digests:
        digests += 1
        for peptide in digest(probabilities, rng, args.min_length, args.max_length):
            if peptide in seen:
                continue
            observed = observable_indices(seq, *peptide)
            if not len(observed):
                continue
            seen.add(peptide)
            peptides.append(peptide)
            coverage[observed] += 1
    return sorted(peptides), coverage, probabilities, scale, digests, observable_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--mean-length", type=float, default=10.0)
    parser.add_argument("--coverage", type=float, default=2.5)
    parser.add_argument("--max-digests", type=int, default=500)
    parser.add_argument("--out", type=Path, default=HERE / "_peptides")
    args = parser.parse_args()
    if not 2 <= args.min_length <= args.max_length:
        parser.error("require 2 <= min-length <= max-length")
    if args.mean_length <= 0 or args.coverage <= 0 or args.max_digests < 1:
        parser.error("mean-length, coverage and max-digests must be positive")

    seq, resids = sequence(args.topology)
    peptides, coverage, probabilities, scale, digests, observable = build_map(seq, args)
    if coverage[observable].mean() < args.coverage:
        raise RuntimeError("maximum digests reached before observable coverage target")
    segments = np.asarray([(int(resids[a]), int(resids[b])) for a, b in peptides])
    physical_lengths = np.asarray([b - a + 1 for a, b in peptides])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out / "segs_teaa_pepsin.dat", segments, fmt="%d")
    provenance = {
        "topology": str(args.topology.resolve()), "sequence_length": len(seq),
        "seed": args.seed, "digests_drawn": digests, "propensity_scale": scale,
        "length_bounds": [args.min_length, args.max_length],
        "target_mean_length": args.mean_length, "target_observable_coverage": args.coverage,
        "n_peptides": len(peptides), "peptide_length_mean": float(physical_lengths.mean()),
        "peptide_length_min": int(physical_lengths.min()),
        "peptide_length_max": int(physical_lengths.max()),
        "observable_coverage_mean": float(coverage[observable].mean()),
        "observable_residues_uncovered": int(np.sum(coverage[observable] == 0)),
        "model": {
            "form": "p_cut(i) = clip(scale * W_P1[P1] * W_P1_prime[P1_prime] * proline_mask, 0, 1)",
            "W_P1": W_P1, "W_P1_prime": W_P1P,
            "proline_rule": "zero at P1/P1-prime; x0.15 per Pro at P2/P2-prime",
            "pooling": "independent Bernoulli digests, deduplicated",
            "selection_information": "sequence only; no target uptake, kinetics, structure, or state labels",
        },
    }
    (args.out / "pepsin_map_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    np.savez_compressed(args.out / "pepsin_map.npz", segments=segments, coverage=coverage,
                        bond_probability=probabilities, resids=resids)
    print(json.dumps({k: v for k, v in provenance.items() if k != "model"}, indent=2))


if __name__ == "__main__":
    main()
