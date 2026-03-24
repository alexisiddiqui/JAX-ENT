"""
Loads a directory of NeuralPLexer outputs (protein PDBs + ligand SDFs),
applies a conservative constrained MMFF/UFF minimisation to the CDZ ligand
atoms (Ca²⁺ ions and other monoatomic species are left untouched), and writes
two sets of merged protein-ligand PDB files for easy comparison:

  <input_dir>/combined_outputs/            – original (unminimised) complexes
  <input_dir>/combined_outputs_minimised/  – after constrained cleanup

If all complexes in a set share the same atom count, a combined multiframe
combined_all.pdb is also written for that set.

Minimisation settings
---------------------
  Per-atom restraints are weighted by each heavy atom's minimum distance
  to any protein heavy atom (dmin):

    dmin ≤ 3.0 Å  →  maxDispl 0.10 Å, force 300 kcal/mol/Å²
    3.0–4.0 Å     →  maxDispl 0.15 Å, force 200
    4.0–5.0 Å     →  maxDispl 0.22 Å, force 100
    > 5.0 Å       →  maxDispl 0.30 Å, force  60

  max_iters = 200
  ignoreInterfragInteractions = False  (inter-fragment clashes are resolved)

Loading strategy
----------------
  Protein PDB  -> MDAnalysis Universe
  Ligand SDF   -> RDKit Mol -> MonomerInfo annotated -> MDAnalysis Universe
                 (bond topology preserved for CONECT records)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

# ---------------------------------------------------------------------------
# Constrained MMFF/UFF cleanup (CDZ ligands only, Ca²⁺ ions untouched)
# ---------------------------------------------------------------------------

MAX_ITERS: int = 300
MMFF_VARIANT: str = "MMFF94s"

# Per-atom restraint parameters keyed by minimum protein distance (Å).
# Bins are (upper_bound_exclusive, max_displ, force_constant).
# Atoms at or below the first threshold use the tightest restraint.
_DISTANCE_BINS: tuple[tuple[float, float, float], ...] = (
    (2.0, 0.10, 300.0),  # clashing  – hold firmly
    (3.0, 0.15, 200.0),  # first shell            – moderate
    (4.0, 0.22, 100.0),  # second shell           – loose
    (float("inf"), 0.30, 60.0),  # solvent-exposed – very loose
)


def _restraint_params(dmin: float) -> tuple[float, float]:
    """Return (max_displ, force_constant) for a given min protein distance."""
    for upper, max_displ, force_k in _DISTANCE_BINS:
        if dmin < upper:
            return max_displ, force_k
    # Should never reach here, but satisfy type-checker
    return _DISTANCE_BINS[-1][1], _DISTANCE_BINS[-1][2]


def _prot_heavy_xyz(prot_u: mda.Universe) -> np.ndarray:
    """Return (N, 3) float32 array of all protein heavy-atom positions."""
    heavy = prot_u.select_atoms("not name H*")
    return heavy.positions.astype(np.float32)


def _min_protein_distances(lig_xyz: np.ndarray, prot_xyz: np.ndarray) -> np.ndarray:
    """
    Return per-ligand-atom minimum distance to any protein heavy atom.

    Parameters
    ----------
    lig_xyz  : (n_lig,  3) array of ligand heavy-atom positions
    prot_xyz : (n_prot, 3) array of protein heavy-atom positions

    Returns
    -------
    dmins : (n_lig,) float32 array
    """
    # Compute squared distances without building the full (n_lig, n_prot) matrix
    # for large proteins — iterate over ligand atoms (always small).
    dmins = np.empty(len(lig_xyz), dtype=np.float32)
    for i, p in enumerate(lig_xyz):
        d2 = np.sum((prot_xyz - p) ** 2, axis=1)
        dmins[i] = np.sqrt(d2.min())
    return dmins


def _sanitize_fragment_for_ff(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize an organic fragment conservatively, without touching valence of ions."""
    mol = Chem.Mol(mol)
    rdmolops.Cleanup(mol)

    sanitize_ops = (
        rdmolops.SanitizeFlags.SANITIZE_CLEANUP
        | rdmolops.SanitizeFlags.SANITIZE_PROPERTIES
        | rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS
        | rdmolops.SanitizeFlags.SANITIZE_KEKULIZE
        | rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
        | rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
        | rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION
        | rdmolops.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | rdmolops.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY
        | rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS
    )

    fail = rdmolops.SanitizeMol(mol, sanitizeOps=sanitize_ops, catchErrors=True)
    if fail:
        raise ValueError(f"RDKit sanitization failed with flag: {fail}")
    return mol


def _build_forcefield(mol: Chem.Mol, conf_id: int = -1):
    """
    Return (ff, ff_name, add_position_constraint_fn).

    ignoreInterfragInteractions=False ensures that clashes between
    disconnected CDZ fragments (e.g. two non-covalent pieces) are
    penalised and resolved during minimisation.
    """
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=MMFF_VARIANT)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol,
            props,
            confId=conf_id,
            ignoreInterfragInteractions=False,  # ← resolve inter-fragment clashes
        )
        return ff, "MMFF", ff.MMFFAddPositionConstraint

    if AllChem.UFFHasAllMoleculeParams(mol):
        ff = AllChem.UFFGetMoleculeForceField(
            mol,
            confId=conf_id,
            ignoreInterfragInteractions=False,  # ← same for UFF fallback
        )
        return ff, "UFF", ff.UFFAddPositionConstraint

    raise ValueError("No MMFF or UFF parameters available for this molecule.")


def constrained_cleanup_ligand_mol(
    mol_in: Chem.Mol,
    prot_xyz: np.ndarray | None = None,
) -> Chem.Mol:
    """
    Conservative in-memory constrained cleanup of a NeuralPLexer ligand Mol.

    Strategy
    --------
    - Split into disconnected fragments.
    - Skip monoatomic fragments (Ca²⁺, Cl⁻, Zn²⁺, …) entirely.
    - For each multi-atom fragment (CDZ ligand pieces):
        * Strip Hs, sanitize conservatively.
        * Re-add Hs with 3-D coordinates for a sensible local geometry.
        * Build MMFF94s force field (UFF fallback).
        * Per-atom restraints weighted by minimum distance to protein:
            dmin ≤ 3.0 Å  →  maxDispl 0.10, force 300
            3.0–4.0 Å     →  maxDispl 0.15, force 200
            4.0–5.0 Å     →  maxDispl 0.22, force 100
            > 5.0 Å       →  maxDispl 0.30, force  60
          Falls back to the > 5.0 Å defaults if prot_xyz is None.
        * Minimise for up to MAX_ITERS steps.
    - Copy optimised heavy-atom coordinates back into the output molecule.
    - Heavy atoms from ignored fragments keep their original coordinates.

    Parameters
    ----------
    mol_in   : Chem.Mol
        Molecule parsed from SDF without sanitization (removeHs=False).
    prot_xyz : (N, 3) float32 ndarray or None
        Protein heavy-atom positions used to assign per-atom restraints.
        If None, the loosest restraint tier is applied to all atoms.

    Returns
    -------
    Chem.Mol
        Copy of mol_in with updated heavy-atom coordinates for CDZ fragments.
    """
    if mol_in.GetNumConformers() == 0:
        raise ValueError("Input molecule has no conformer; cannot minimise.")

    out = Chem.Mol(mol_in)
    out_conf = out.GetConformer()

    frag_maps: list[tuple[int, ...]] = []
    frag_mols = Chem.GetMolFrags(
        mol_in,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=frag_maps,
    )

    for frag_mol, global_map in zip(frag_mols, frag_maps):
        # Identify heavy atoms in this fragment
        heavy_global = [
            gidx for gidx in global_map if mol_in.GetAtomWithIdx(gidx).GetAtomicNum() > 1
        ]

        # Leave monoatomic fragments (Ca²⁺, Cl⁻, etc.) completely untouched
        if len(heavy_global) <= 1:
            continue

        # ---------- prepare the working copy ----------
        frag_heavy = Chem.RemoveHs(frag_mol, sanitize=False)
        frag_heavy = _sanitize_fragment_for_ff(frag_heavy)

        addhs_params = rdmolops.AddHsParameters()
        addhs_params.addCoords = True
        work = rdmolops.AddHs(frag_heavy, addhs_params)

        # ---------- build force field ----------
        ff, ff_name, add_position_constraint = _build_forcefield(work)

        # Per-atom restraints: tighter near protein, looser in solvent
        n_heavy_work = frag_heavy.GetNumAtoms()
        if prot_xyz is not None and len(prot_xyz) > 0:
            # Collect current heavy-atom positions for this fragment
            in_conf = mol_in.GetConformer()
            frag_heavy_xyz = np.array(
                [list(in_conf.GetAtomPosition(gidx)) for gidx in heavy_global],
                dtype=np.float32,
            )
            dmins = _min_protein_distances(frag_heavy_xyz, prot_xyz)
        else:
            # No protein context: apply the loosest tier uniformly
            dmins = np.full(n_heavy_work, float("inf"), dtype=np.float32)

        for idx in range(n_heavy_work):
            max_displ, force_k = _restraint_params(float(dmins[idx]))
            add_position_constraint(idx, max_displ, force_k)

        ff.Initialize()
        status = ff.Minimize(maxIts=MAX_ITERS)

        if status not in (0, 1):
            raise RuntimeError(
                f"{ff_name} minimization failed for fragment "
                f"(global atom indices {heavy_global}). Status={status}"
            )

        # ---------- copy optimised heavy-atom coords back ----------
        optimised_heavy = Chem.RemoveHs(work, sanitize=False)
        opt_conf = optimised_heavy.GetConformer()

        for local_idx, global_idx in enumerate(heavy_global):
            pos = opt_conf.GetAtomPosition(local_idx)
            out_conf.SetAtomPosition(global_idx, pos)

    return out


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_rank_plddt(filename: str) -> tuple[int, str] | None:
    """Extract (rank_int, plddt_str) or None from a NeuralPLexer filename."""
    m = re.search(r"rank(\d+)_plddt(\d+(?:\.\d+)?)", filename)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


# ---------------------------------------------------------------------------
# Ligand SDF → annotated RDKit Mol → MDAnalysis Universe
# ---------------------------------------------------------------------------


def _assign_lig_monomer_info(mol: Chem.Mol) -> Chem.Mol:
    """
    Annotate every atom with PDB MonomerInfo so the MDAnalysis RDKit converter
    produces correct resname / resid / chain assignments.

    Fragment classification
    -----------------------
    Ca single-atom frags  → resname 'CA ', chain Z, one residue per ion
    Multi-atom frags      → resname 'CDZ', chain Z, one residue per fragment
    Other single-atom     → element-based resname ('CL', 'ZN', …), chain Z
    """
    frags = Chem.GetMolFrags(mol)

    ca_frags, other_frags = [], []
    for frag in frags:
        atom = mol.GetAtomWithIdx(frag[0])
        if len(frag) == 1 and atom.GetSymbol() == "Ca":
            ca_frags.append(frag)
        else:
            other_frags.append(frag)

    ca_frags = sorted(ca_frags, key=lambda f: f[0])
    large_frags = sorted([f for f in other_frags if len(f) > 1], key=lambda f: f[0])
    small_frags = sorted([f for f in other_frags if len(f) == 1], key=lambda f: f[0])

    atom_resinfo: dict[int, tuple[str, int]] = {}
    resid = 1

    for frag in ca_frags:
        atom_resinfo[frag[0]] = ("CA ", resid)
        resid += 1

    for frag in large_frags:
        for idx in frag:
            atom_resinfo[idx] = ("CDZ", resid)
        resid += 1

    for frag in small_frags:
        sym = mol.GetAtomWithIdx(frag[0]).GetSymbol().upper()
        atom_resinfo[frag[0]] = (sym[:3], resid)
        resid += 1

    em = Chem.RWMol(mol)
    for idx in range(em.GetNumAtoms()):
        atom = em.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        res_name, res_seq = atom_resinfo[idx]

        pdb_atom_name = f"{sym.upper():<4s}" if len(sym) >= 2 else f" {sym.upper():<3s}"

        mi = Chem.AtomPDBResidueInfo()
        mi.SetName(pdb_atom_name)
        mi.SetResidueName(res_name)
        mi.SetResidueNumber(res_seq)
        mi.SetChainId("Z")
        mi.SetIsHeteroAtom(True)
        mi.SetOccupancy(1.0)
        mi.SetTempFactor(0.0)
        atom.SetMonomerInfo(mi)

    return em.GetMol()


def _mol_to_universe(mol: Chem.Mol) -> mda.Universe:
    """Convert an annotated RDKit Mol to an MDAnalysis Universe via the RDKit converter."""
    return mda.Universe(mol)


def load_lig_universe(
    sdf_path: str | Path,
    minimise: bool = False,
    prot_xyz: np.ndarray | None = None,
) -> mda.Universe:
    """
    Load a ligand SDF, optionally minimise CDZ fragments, annotate MonomerInfo,
    and return an MDAnalysis Universe with full bond connectivity.

    Parameters
    ----------
    sdf_path  : path to the .sdf file
    minimise  : if True, run constrained_cleanup_ligand_mol before annotation
    prot_xyz  : (N, 3) protein heavy-atom positions for distance-weighted
                restraints; only used when minimise=True
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"RDKit could not parse SDF: {sdf_path}")

    if minimise:
        mol = constrained_cleanup_ligand_mol(mol, prot_xyz=prot_xyz)

    mol = _assign_lig_monomer_info(mol)
    return _mol_to_universe(mol)


# ---------------------------------------------------------------------------
# Build and write complexes for one output directory
# ---------------------------------------------------------------------------


def build_and_write_complexes(
    pairs: list[tuple[int, str, Path, Path]],
    out_dir: Path,
    *,
    minimise: bool,
) -> None:
    """
    For each (rank, plddt, prot_path, lig_path) pair:
      - Optionally minimise the ligand (CDZ fragments only).
      - Merge protein + ligand into a single MDAnalysis Universe.
      - Write individual complex_rank{rank}_plddt{plddt}.pdb files.
      - If all complexes have equal atom counts, write combined_all.pdb.
    """
    out_dir.mkdir(exist_ok=True)
    tag = "minimised" if minimise else "original"
    print(f"\n[INFO] Writing {tag} complexes → {out_dir}")

    merged_list: list[mda.Universe] = []

    for rank, plddt, prot_path, lig_path in pairs:
        try:
            prot_u = mda.Universe(str(prot_path))
            prot_xyz = _prot_heavy_xyz(prot_u) if minimise else None
            lig_u = load_lig_universe(lig_path, minimise=minimise, prot_xyz=prot_xyz)
            merged = mda.Merge(prot_u.atoms, lig_u.atoms)
        except Exception as exc:
            print(f"  [ERROR] rank{rank}: {exc}", file=sys.stderr)
            continue

        merged_list.append(merged)
        out_path = out_dir / f"complex_rank{rank}_plddt{plddt}.pdb"

        with mda.Writer(str(out_path), n_atoms=merged.atoms.n_atoms, bonds="conect") as W:
            W.write(merged.atoms)

        print(f"  [OK] {out_path.name}  ({lig_u.atoms.n_atoms} ligand atoms)")

    if not merged_list:
        print(f"  [WARNING] No complexes written to {out_dir}.")
        return

    # Combined multiframe PDB when atom counts are uniform
    atom_counts = [u.atoms.n_atoms for u in merged_list]
    if len(set(atom_counts)) == 1:
        all_out = out_dir / "combined_all.pdb"
        print(f"[INFO] Uniform atom count ({atom_counts[0]}). Writing {all_out.name} …")
        with mda.Writer(
            str(all_out),
            n_atoms=atom_counts[0],
            multiframe=True,
            bonds="conect",
        ) as W:
            for merged in merged_list:
                W.write(merged.atoms)
        print(f"  [OK] combined_all.pdb  ({len(merged_list)} models)")
    else:
        print(f"[INFO] Atom counts differ {atom_counts}. Skipping combined_all.pdb.")


# ---------------------------------------------------------------------------
# Protein-only (APO) output writer
# ---------------------------------------------------------------------------


def build_and_write_protein_only(
    pairs: list[tuple[int, str, Path, None]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(exist_ok=True)
    print(f"\n[INFO] Protein-only mode — writing to {out_dir}")
    universes: list[mda.Universe] = []
    for rank, plddt, prot_path, _ in pairs:
        try:
            u = mda.Universe(str(prot_path))
        except Exception as exc:
            print(f"  [ERROR] rank{rank}: {exc}", file=sys.stderr)
            continue
        universes.append(u)
        out_path = out_dir / f"prot_rank{rank}_plddt{plddt}.pdb"
        with mda.Writer(str(out_path), n_atoms=u.atoms.n_atoms) as W:
            W.write(u.atoms)
        print(f"  [OK] {out_path.name}  ({u.atoms.n_atoms} atoms)")

    if not universes:
        print(f"  [WARNING] No structures written to {out_dir}.")
        return

    atom_counts = [u.atoms.n_atoms for u in universes]
    if len(set(atom_counts)) == 1:
        all_out = out_dir / "combined_all.pdb"
        print(f"[INFO] Uniform atom count ({atom_counts[0]}). Writing {all_out.name} …")
        with mda.Writer(str(all_out), n_atoms=atom_counts[0], multiframe=True) as W:
            for u in universes:
                W.write(u.atoms)
        print(f"  [OK] combined_all.pdb  ({len(universes)} models)")
    else:
        print(f"[INFO] Atom counts differ {atom_counts}. Skipping combined_all.pdb.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine NeuralPLexer protein PDB + ligand SDF outputs into "
            "complex PDBs.  Writes two sets for comparison:\n"
            "  combined_outputs/           – original poses\n"
            "  combined_outputs_minimised/ – after constrained MMFF cleanup"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory containing prot_rank* and lig_rank* files.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        metavar="N",
        help="Only process the top N structures (by rank/pLDDT). Default: all.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] --input_dir does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover and match protein/ligand pairs
    prot_files = sorted(
        p for p in input_dir.glob("prot_rank*.pdb") if parse_rank_plddt(p.name) is not None
    )
    if not prot_files:
        print(f"[WARNING] No prot_rank*.pdb files found in {input_dir}.")
        sys.exit(0)

    lig_files_exist = any(input_dir.glob("lig_rank*.sdf"))

    if not lig_files_exist:
        # APO / protein-only mode
        pairs_apo = []
        for prot_path in prot_files:
            rank, plddt = parse_rank_plddt(prot_path.name)
            pairs_apo.append((rank, plddt, prot_path, None))
        pairs_apo.sort(key=lambda t: t[0])
        if args.top_n is not None:
            pairs_apo = pairs_apo[: args.top_n]
            print(f"[INFO] Filtering to top {args.top_n} structure(s).")
        build_and_write_protein_only(pairs_apo, out_dir=input_dir / "combined_outputs")
        print("\n[DONE]")
        return

    # --- existing HOLO path below ---
    pairs: list[tuple[int, str, Path, Path]] = []
    for prot_path in prot_files:
        rank, plddt = parse_rank_plddt(prot_path.name)
        lig_path = input_dir / f"lig_rank{rank}_plddt{plddt}.sdf"
        if not lig_path.exists():
            print(f"[WARNING] No matching SDF for {prot_path.name} — skipping.")
            continue
        pairs.append((rank, plddt, prot_path, lig_path))

    if not pairs:
        print("[ERROR] No valid protein+ligand pairs found.", file=sys.stderr)
        sys.exit(1)

    pairs.sort(key=lambda t: t[0])
    print(f"[INFO] Found {len(pairs)} matched pair(s).")
    if args.top_n is not None:
        pairs = pairs[: args.top_n]
        print(f"[INFO] Filtering to top {args.top_n} structure(s) by rank/pLDDT.")
    print(
        "[INFO] Minimisation settings: distance-weighted per-atom restraints, "
        f"max_iters={MAX_ITERS}, ignoreInterfragInteractions=False\n"
        "       dmin ≤ 2.0 Å → maxDispl 0.10, force 300\n"
        "       2.0–3.0 Å   → maxDispl 0.15, force 200\n"
        "       3.0–4.0 Å   → maxDispl 0.22, force 100\n"
        "       > 4.0 Å     → maxDispl 0.30, force  60"
    )

    # ── Original (unminimised) ──────────────────────────────────────────────
    build_and_write_complexes(
        pairs,
        out_dir=input_dir / "combined_outputs",
        minimise=False,
    )

    # ── Minimised ──────────────────────────────────────────────────────────
    build_and_write_complexes(
        pairs,
        out_dir=input_dir / "combined_outputs_minimised",
        minimise=True,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
