#!/usr/bin/env python3
"""Re-extract the raw peptide/ETD-fragment HDX uptake from the SI spreadsheet (sheet "Figure S11").

Source: data/_aSyn/ac2c03183_si_002.xlsx (openpyxl is not a dependency; the sheet XML is parsed
directly with zipfile + regex, resolving shared strings).

Outputs (data/_aSyn/uptake/):
  S11_fragment_uptake_raw.csv        one row per (condition, peptide, fragment): q, cfit, rmse, x_plateau, k_exp,
                                     u0..u8 (u9 for Lysosomal) in Da and sd0..sd8 (sd9) in Da.
  <condition>_peptide_uptake.csv     tidy long table of whole peptides only (fragment == 'peptide'):
                                     time index/time [s], uptake/SD [Da], fraction = u/q, fraction_sd = sd/q,
                                     contributing residues (start+1..end without prolines), tail flag, use flag.
  S11_time_grids.json                time grids per condition, normalisation factors of the reported k_exp,
                                     pD used for intrinsic rates, notes.

Conventions (see data/CONTROL_MD_PROVENANCE.md, "Experimental data re-read from the SI spreadsheet"):
- Tris_only / Extracellular / Intracellular: t = 0, 0.05, 0.1, 0.25, 0.5, 1, 10, 30 s, then an ambiguous last
  point (60 or 300 s). Only the first 8 points are flagged use=True.
- Lysosomal: 10 points, absolute time scale not identifiable from the curves -> time_s left empty, use=False.
- Reported k_exp are ms^-1 and Tris-normalised: k [s^-1] = 1000 * k_exp / norm[condition], with
  norm = k_bradykinin(Tris) / k_bradykinin(condition) from sheet "Figure S5-middle panel".
- q = number of exchanging amides = len - 1 - prolines (first residue excluded); checked against the residue list.

Usage (from repo root):  python jaxent/examples/4_aSyn/data/extract_uptake_S11.py
"""
from __future__ import annotations

import argparse
import html
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
XLSX = HERE / "_aSyn" / "ac2c03183_si_002.xlsx"
OUT = HERE / "_aSyn" / "uptake"

SECTION_TO_CONDITION = {
    "Tris only": "Tris_only",
    "Extracellular": "Extracellular",
    "Intracellular": "Intracellular",
    "Lysosomal": "Lysosomal",
}
PH = {"Tris_only": 7.4, "Extracellular": 7.4, "Intracellular": 7.2, "Lysosomal": 4.9}  # data/_aSyn/aSyn_pH.csv
TIMES_S = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 10.0, 30.0]  # first 8 points (ms grid converted to s)
N_USE = 8
LAST_POINT_CANDIDATES_S = [60.0, 300.0]
TAIL_START = 114  # electrostatic acidic tail (excluded from training in the pilot)


# ----------------------------------------------------------------------------- xlsx
def _col_index(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def read_sheet(xlsx: Path, sheet_name: str) -> dict[int, dict[str, str]]:
    """Return {row_number: {column_letter: value_string}} for one sheet (shared strings resolved)."""
    z = zipfile.ZipFile(xlsx)
    wb = z.read("xl/workbook.xml").decode()
    rels = z.read("xl/_rels/workbook.xml.rels").decode()
    sheets = {html.unescape(n): rid for n, rid in re.findall(r'<sheet name="([^"]+)"[^>]*r:id="([^"]+)"', wb)}
    if sheet_name not in sheets:
        raise KeyError(f"sheet {sheet_name!r} not found; have {list(sheets)}")
    target = dict(re.findall(r'<Relationship Id="([^"]+)"[^>]*Target="([^"]+)"', rels))[sheets[sheet_name]]
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S):
            shared.append(html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))))
    xml = z.read("xl/" + target.lstrip("/").removeprefix("xl/")).decode()
    rows: dict[int, dict[str, str]] = {}
    for r in re.finditer(r'<row [^>]*r="(\d+)"[^>]*>(.*?)</row>', xml, re.S):
        cells = {}
        for c in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', r.group(2), re.S):
            ref, attr, body = c.group(1), c.group(2), c.group(3) or ""
            v = re.search(r"<v>(.*?)</v>", body)
            if v is not None:
                val = v.group(1)
                if 't="s"' in attr:
                    val = shared[int(val)]
            else:
                t = re.search(r"<t[^>]*>(.*?)</t>", body, re.S)
                val = html.unescape(t.group(1)) if t else None
            if val is not None and val != "":
                cells[ref] = val
        rows[int(r.group(1))] = cells
    return rows


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------------------- parse
def parse_s11(xlsx: Path) -> pd.DataFrame:
    rows = read_sheet(xlsx, "Figure S11")
    out = []
    cond, header = None, None
    for rn in sorted(rows):
        cells = rows[rn]
        a = cells.get("A", "")
        if len(cells) == 1 and a in SECTION_TO_CONDITION:
            cond, header = SECTION_TO_CONDITION[a], None
            continue
        if a == "Sequence":
            header = {v: k for k, v in cells.items()}  # name -> column letter
            continue
        if cond is None or header is None or "B" not in cells:
            continue
        start, end = int(float(cells["B"])), int(float(cells["C"]))
        npts = max(int(m.group(1)) for m in (re.match(r"Uptake_1_(\d+)$", h) for h in header) if m)
        for j in range(1, 15):
            q = _num(cells.get(header.get(f"q_{j}", ""), None))
            if np.isnan(q):
                continue
            name = cells.get(header.get(f"Fragment_{j}", ""), None)
            frag = "peptide" if j == 1 and not name else (name or f"frag{j}")
            rec = dict(condition=cond, start=start, end=end, sequence=a, fragment=frag, frag_idx=j, q=q,
                       cfit=_num(cells.get(header.get(f"Cfit_{j}", ""))),
                       rmse=_num(cells.get(header.get(f"rmse_{j}", ""))),
                       x_plateau=_num(cells.get(header.get(f"X_plateau_{j}", ""))),
                       k_exp=_num(cells.get(header.get(f"k_exp_{j}", ""))))
            for k in range(npts):
                rec[f"u{k}"] = _num(cells.get(header.get(f"Uptake_{j}_{k + 1}", "")))
            for k in range(npts):
                rec[f"sd{k}"] = _num(cells.get(header.get(f"Uptake_SD_{j}_{k + 1}", "")))
            out.append(rec)
    df = pd.DataFrame(out)
    cols = ["condition", "start", "end", "sequence", "fragment", "frag_idx", "q", "cfit", "rmse", "x_plateau", "k_exp"]
    cols += [f"u{k}" for k in range(9)] + [f"sd{k}" for k in range(9)] + ["u9", "sd9"]
    return df[cols]


def normalisation_factors(xlsx: Path) -> dict[str, float]:
    """k_bradykinin(Tris) / k_bradykinin(condition) from sheet 'Figure S5-middle panel'."""
    rows = read_sheet(xlsx, "Figure S5-middle panel")
    k = {}
    for cells in rows.values():
        if cells.get("A") in PH and "B" in cells:
            k[cells["A"]] = float(cells["B"])
    return {c: k["Tris_only"] / k[c] for c in k}


def contributing_residues(start: int, end: int, sequence: str) -> list[int]:
    """Residues start+1..end excluding prolines (sequence is the peptide's own sequence, start..end)."""
    assert len(sequence) == end - start + 1, (start, end, sequence)
    return [start + i for i in range(1, len(sequence)) if sequence[i] != "P"]


def tidy_condition(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    p = df[(df.condition == cond) & (df.fragment == "peptide")].sort_values(["start", "end"]).reset_index(drop=True)
    npts = 10 if cond == "Lysosomal" else 9
    rows = []
    for pi, r in p.iterrows():
        res = contributing_residues(int(r.start), int(r.end), r.sequence)
        if len(res) != int(r.q):
            raise ValueError(f"{cond} {r.start}-{r.end}: q={r.q} but {len(res)} contributing residues")
        for k in range(npts):
            known = cond != "Lysosomal" and k < N_USE
            rows.append(dict(condition=cond, peptide_index=pi, label=f"{int(r.start)}-{int(r.end)}",
                             start=int(r.start), end=int(r.end), sequence=r.sequence, q=int(r.q),
                             residues=" ".join(map(str, res)), tail=bool(r.start >= TAIL_START),
                             t_index=k, time_s=TIMES_S[k] if known else np.nan,
                             uptake_Da=r[f"u{k}"], sd_Da=r[f"sd{k}"],
                             fraction=r[f"u{k}"] / r.q, fraction_sd=r[f"sd{k}"] / r.q,
                             use=bool(known and np.isfinite(r[f"u{k}"]))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xlsx", default=str(XLSX))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check-existing", action="store_true",
                    help="compare with an existing S11_fragment_uptake_raw.csv before overwriting it")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = parse_s11(Path(a.xlsx))
    raw = out / "S11_fragment_uptake_raw.csv"
    if a.check_existing and raw.exists():
        old = pd.read_csv(raw)
        num = [c for c in df.columns if c not in ("condition", "sequence", "fragment")]
        same_shape = old.shape == df.shape
        maxdiff = np.nanmax(np.abs(old[num].to_numpy(float) - df[num].to_numpy(float))) if same_shape else np.inf
        nan_same = same_shape and bool((old[num].isna().to_numpy() == df[num].isna().to_numpy()).all())
        print(f"existing CSV: shape {old.shape} vs {df.shape}; max |diff| {maxdiff:.3g}; NaN pattern equal {nan_same}; "
              f"labels equal {same_shape and (old[['condition', 'sequence', 'fragment']].values == df[['condition', 'sequence', 'fragment']].values).all()}")
    df.to_csv(raw, index=False)
    print(f"wrote {raw} ({len(df)} rows; whole peptides per condition: "
          f"{df[df.fragment == 'peptide'].groupby('condition').size().to_dict()})")

    for cond in SECTION_TO_CONDITION.values():
        t = tidy_condition(df, cond)
        path = out / f"{cond}_peptide_uptake.csv"
        t.to_csv(path, index=False)
        print(f"wrote {path} ({t.peptide_index.nunique()} peptides, {int(t.use.sum())} usable points)")

    norm = normalisation_factors(Path(a.xlsx))
    meta = {
        "source": str(Path(a.xlsx).name) + " sheet 'Figure S11'",
        "time_grid_s": {c: TIMES_S + [None] for c in ("Tris_only", "Extracellular", "Intracellular")},
        "last_point_candidates_s": LAST_POINT_CANDIDATES_S,
        "n_points_used": N_USE,
        "Lysosomal": {"n_points": 10, "time_grid_s": None,
                      "note": "same relative spacing, absolute scale not identifiable from the curves (needs SI PDF)"},
        "k_exp_units": "ms^-1, Tris-normalised: k[s^-1] = 1000 * k_exp / norm[condition]",
        "norm_factor": norm,
        "norm_source": "k_bradykinin(Tris_only)/k_bradykinin(condition), sheet 'Figure S5-middle panel'",
        "pD_for_kint": PH,
        "kint_convention": "calculate_HDXrate(T=293 K, pD = measured pH, no +0.4), s^-1",
        "fraction": "u/q, q = len - 1 - prolines; residues start+1..end without prolines",
        "tail_start": TAIL_START,
    }
    with open(out / "S11_time_grids.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out / 'S11_time_grids.json'}; norm factors {({k: round(v, 4) for k, v in norm.items()})}")


if __name__ == "__main__":
    main()
