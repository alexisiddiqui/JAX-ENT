"""Extract and structure-number the MoPrP ValDX data.

The residue indices in ``moprp.list`` and ``median.pfact`` are one-based
positions in ``moprp.seq``. This script locates that complete HDX sequence in
a PDB chain and writes the corresponding PDB residue IDs to the generated
segments and protection-factor files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import MDAnalysis as mda
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "_MoPrP"
DEFAULT_STRUCTURE = (
    SCRIPT_DIR
    / "MoPrP109_s20_r1_msa1-127_n12700_do1_20260904_191954_protonated_max_plddt_1627.pdb"
)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class StructureResidue:
    chain_id: str
    residue_id: int
    insertion_code: str
    amino_acid: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MoPrP ValDX data using PDB residue numbering."
    )
    parser.add_argument(
        "--structure", type=Path, default=DEFAULT_STRUCTURE,
        help="PDB whose residue IDs should be used in generated outputs.",
    )
    parser.add_argument(
        "--sequence", type=Path, default=DEFAULT_DATA_DIR / "moprp.seq",
        help="File containing the one-letter HDX sequence.",
    )
    parser.add_argument(
        "--chain",
        help="PDB chain to search. By default, require a unique match across all chains.",
    )
    return parser.parse_args(argv)


def _residue_chain_id(residue: object) -> str:
    chain_ids = {str(value).strip() for value in residue.atoms.chainIDs}
    if len(chain_ids) != 1:
        raise ValueError(
            f"Residue {residue.resid} has ambiguous chain IDs: {sorted(chain_ids)}"
        )
    return next(iter(chain_ids))


def load_structure_residues(structure_path: Path) -> dict[str, list[StructureResidue]]:
    """Return ordered canonical protein residues grouped by PDB chain."""
    universe = mda.Universe(str(structure_path))
    chains: dict[str, list[StructureResidue]] = defaultdict(list)
    for residue in universe.select_atoms("protein").residues:
        residue_name = str(residue.resname).upper()
        if residue_name not in THREE_TO_ONE:
            raise ValueError(
                f"Unsupported protein residue {residue_name!r} at residue {residue.resid}"
            )
        chain_id = _residue_chain_id(residue)
        chains[chain_id].append(
            StructureResidue(
                chain_id=chain_id,
                residue_id=int(residue.resid),
                insertion_code=str(residue.icode).strip(),
                amino_acid=THREE_TO_ONE[residue_name],
            )
        )
    if not chains:
        raise ValueError(f"No protein residues found in structure: {structure_path}")
    return dict(chains)


def _all_occurrences(sequence: str, query: str) -> list[int]:
    starts: list[int] = []
    start = sequence.find(query)
    while start != -1:
        starts.append(start)
        start = sequence.find(query, start + 1)
    return starts


def build_sequence_to_structure_map(
    structure_path: Path, hdx_sequence: str, chain: str | None = None
) -> tuple[dict[int, int], str, tuple[int, int]]:
    """Map one-based HDX positions to integer residue IDs in one PDB chain."""
    sequence = "".join(hdx_sequence.split()).upper()
    if not sequence:
        raise ValueError("HDX sequence is empty")
    invalid = sorted(set(sequence) - set(THREE_TO_ONE.values()))
    if invalid:
        raise ValueError(f"HDX sequence contains unsupported residue codes: {invalid}")

    chains = load_structure_residues(structure_path)
    if chain is not None:
        if chain not in chains:
            raise ValueError(
                f"Chain {chain!r} not found; available chains: {sorted(chains)}"
            )
        candidates = {chain: chains[chain]}
    else:
        candidates = chains

    matches: list[tuple[str, int, list[StructureResidue]]] = []
    for chain_id, residues in candidates.items():
        structure_sequence = "".join(residue.amino_acid for residue in residues)
        for start in _all_occurrences(structure_sequence, sequence):
            matches.append((chain_id, start, residues))

    if not matches:
        scope = f"chain {chain!r}" if chain is not None else "any chain"
        raise ValueError(f"HDX sequence was not found exactly and contiguously in {scope}")
    if len(matches) != 1:
        locations = [f"chain {item[0]!r} offset {item[1]}" for item in matches]
        raise ValueError(
            "HDX sequence match is ambiguous; specify --chain if possible. "
            f"Matches: {locations}"
        )

    chain_id, start, residues = matches[0]
    matched = residues[start : start + len(sequence)]
    insertion_codes = [residue for residue in matched if residue.insertion_code]
    if insertion_codes:
        first = insertion_codes[0]
        raise ValueError(
            "Matched residues contain insertion codes, which cannot be represented in "
            f"the integer output format (first: {first.residue_id}{first.insertion_code})"
        )
    residue_ids = [residue.residue_id for residue in matched]
    if len(set(residue_ids)) != len(residue_ids):
        raise ValueError("Matched sequence contains duplicate integer residue IDs")

    mapping = {position: residue_id for position, residue_id in enumerate(residue_ids, 1)}
    return mapping, chain_id, (residue_ids[0], residue_ids[-1])


def map_hdx_position(position: object, mapping: dict[int, int], source: str) -> int:
    numeric = float(position)
    if not numeric.is_integer():
        raise ValueError(f"{source} contains a non-integer HDX position: {position!r}")
    position_id = int(numeric)
    if position_id not in mapping:
        raise ValueError(
            f"{source} HDX position {position_id} is outside 1-{len(mapping)}"
        )
    return mapping[position_id]


def extract_data(
    structure_path: Path,
    sequence_path: Path,
    chain: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    raw_dfrac_path = data_dir / "moprp.dexp"
    raw_segs_path = data_dir / "moprp.list"
    pf_path = data_dir / "median.pfact"
    output_dir = data_dir / "_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dfrac_path = output_dir / "MoPrP_dfrac.dat"
    output_segs_path = output_dir / "MoPrP_segments.txt"
    output_pfact_path = output_dir / "MoPrP_pfactors.dat"

    hdx_sequence = sequence_path.read_text().strip()
    mapping, matched_chain, residue_span = build_sequence_to_structure_map(
        structure_path, hdx_sequence, chain
    )
    print(
        f"Mapped {len(mapping)} HDX residues to chain {matched_chain!r}, "
        f"PDB residues {residue_span[0]}-{residue_span[1]}"
    )

    print("Reading data files...")
    dfrac_df = pd.read_csv(raw_dfrac_path, header=None, sep=r"\s+")
    segs_df = pd.read_csv(raw_segs_path, header=None, sep=r"\s+")
    pfact_df = pd.read_csv(pf_path, header=None, sep=r"\s+")
    times_hours = dfrac_df.iloc[:, 0].values
    times_minutes = times_hours * 60
    dfrac_data = dfrac_df.iloc[:, 1:].values
    if len(segs_df) != dfrac_data.shape[1]:
        raise ValueError(
            f"Segment count ({len(segs_df)}) does not match uptake columns "
            f"({dfrac_data.shape[1]})"
        )

    print(f"Time points (hours): {times_hours}")
    print(f"Time points (minutes): {times_minutes}")
    print(f"Number of time points: {len(times_minutes)}")
    print(f"Number of segments: {dfrac_data.shape[1]}")

    print("Creating dfrac file...")
    with output_dfrac_path.open("w") as handle:
        header = "#\t" + "\t".join(f"{time:.2f}" for time in times_minutes)
        handle.write(header + "\t times/min\n")
        for segment_idx in range(dfrac_data.shape[1]):
            segment_data = dfrac_data[:, segment_idx]
            handle.write("\t".join(f"{value:.5f}" for value in segment_data) + "\n")

    print("Creating structure-numbered segments file...")
    with output_segs_path.open("w") as handle:
        for _, row in segs_df.iterrows():
            res_start = map_hdx_position(row.iloc[1], mapping, "moprp.list start")
            res_end = map_hdx_position(row.iloc[2], mapping, "moprp.list end")
            handle.write(f"{res_start} {res_end}\n")

    print("Creating structure-numbered protection factors file...")
    filtered_pfact_df = pfact_df[pfact_df.iloc[:, 1] > 0]
    with output_pfact_path.open("w") as handle:
        for _, row in filtered_pfact_df.iterrows():
            residue_num = map_hdx_position(row.iloc[0], mapping, "median.pfact")
            handle.write(f"{residue_num}\t{row.iloc[1]:.5f}\n")

    print("Files created successfully:")
    print(f"Deuteration fractions: {output_dfrac_path}")
    print(f"Segments: {output_segs_path}")
    print(f"Protection factors: {output_pfact_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    extract_data(args.structure, args.sequence, args.chain)


if __name__ == "__main__":
    main()
