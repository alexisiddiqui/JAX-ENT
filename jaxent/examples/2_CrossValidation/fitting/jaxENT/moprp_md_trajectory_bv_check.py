#!/usr/bin/env python3
"""BV contact / coefficient-fit check on an explicit-solvent MD trajectory of MoPrP109.

The solvated xtc is reduced to the protein (first N atoms, order of the protein-only PDB), each
residue is made whole around its CA (minimum image), and the frames are featurised exactly like the
single-structure check with uniform frame weights.  Requires MOPRP_STRUCTURE=109.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

import _moprp_recovery_common as common
import moprp_acceptor_channel_diagnostic as D
import moprp_bv_offset_fit as O
import moprp_single_structure_bv_check as S
from moprp_pivot_litmus import _peptide_map


def protein_trajectory(pdb: Path, xtc: Path, out_xtc: Path) -> dict:
    """Write a protein-only, residue-whole copy of ``xtc`` and return QC numbers."""

    prot = mda.Universe(str(pdb))
    n = len(prot.atoms)
    src = mda.Universe(str(pdb))  # coordinates replaced per frame below
    reader = mda.coordinates.XTC.XTCReader(str(xtc))
    ca = prot.select_atoms("name CA").indices
    res_ca = {r.ix: r.atoms.select_atoms("name CA").indices for r in prot.residues}
    atom_ca = np.asarray(
        [res_ca[a.resindex][0] if len(res_ca[a.resindex]) else -1 for a in prot.atoms]
    )
    has_ca = atom_ca >= 0
    qc = {"n_frames": len(reader), "dt_ps": float(reader.dt), "max_ca_ca": 0.0, "extent_max": 0.0}
    with mda.Writer(str(out_xtc), n) as writer:
        for ts in reader:
            pos = ts.positions[:n].astype(np.float64).copy()
            # 1) chain the CA atoms with minimum image so the backbone is whole
            ca_raw = pos[ca].copy()
            ca_whole = ca_raw.copy()
            steps = minimize_vectors(np.diff(ca_raw, axis=0), ts.dimensions)
            ca_whole[1:] = ca_raw[0] + np.cumsum(steps, axis=0)
            ca_slot = {int(i): k for k, i in enumerate(ca)}
            # 2) place every atom of a residue next to its (whole) CA
            ca_idx = atom_ca[has_ca]
            vec = minimize_vectors(pos[has_ca] - pos[ca_idx], ts.dimensions)
            slots = np.asarray([ca_slot[int(i)] for i in ca_idx])
            pos[has_ca] = ca_whole[slots] + vec
            src.atoms.positions = pos
            src.dimensions = None
            writer.write(src.atoms)
            d = np.linalg.norm(np.diff(pos[ca], axis=0), axis=1)
            qc["max_ca_ca"] = max(qc["max_ca_ca"], float(d.max()))
            qc["extent_max"] = max(qc["extent_max"], float(np.ptp(pos, axis=0).max()))
    return qc


def run(args) -> None:
    if common.STRUCTURE != "109":
        raise SystemExit("set MOPRP_STRUCTURE=109")
    template = common.load_ensemble_inputs("AF2_MSAss", args.rate_source)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        xtc = tmp / "protein.xtc"
        qc = protein_trajectory(args.pdb, args.xtc, xtc)
        print("QC", qc, flush=True)
        for wname, window in S.WINDOWS.items():
            ids, heavy, acc = S._featurise(args.pdb, window, tmp, f"md_{wname}", traj=xtc)
            inputs = S._inputs_for(ids, heavy, acc, template)
            ens = [(inputs, _peptide_map(inputs))]
            results[wname] = {
                "n_frames": inputs.n_frames,
                "contacts": D._contact_stats(inputs),
                "residue_regression": D._residue_regression(inputs),
                "published_mse": O._mse(common.PUBLISHED_BC, common.PUBLISHED_BH, 0.0, ens),
                "no_offset": O._fit(ens, fix_c=0.0),
                "free_offset": O._fit(ens),
                "bh0_offset": O._fit(ens, fix_bh=0.0),
                "bh_profile_offset": [{"bh": b, **O._fit(ens, fix_bh=b)} for b in O.BH_GRID],
                "bh_profile_no_offset": [{"bh": b, **O._fit(ens, fix_bh=b, fix_c=0.0)} for b in O.BH_GRID],
            }
            print(wname, "done", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"qc": qc, "rate_source": args.rate_source, "results": results}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, required=True, help="protein-only PDB (chain A) matching the xtc's leading atoms")
    p.add_argument("--xtc", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    run(p.parse_args())
