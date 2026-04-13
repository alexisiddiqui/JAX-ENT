#!/usr/bin/env python3
"""
minimise_complexes.py

OpenMM TIP3P explicit-solvent minimisation for NeuralPLexer protein-ligand
complexes from a multiframe PDB produced by combine_neuralplexer.py.

Each MODEL block is minimised independently. Protein Cα atoms are harmonically
restrained so the backbone cannot drift far from the NeuralPLexer pose.

Force-field stack
-----------------
  Protein        Amber ff14SB
  CDZ ligand     OpenFF 2.1.0 (SMIRNOFF) — bond orders via rdDetermineBonds
  Water / ions   amber14/tip3p.xml  (TIP3P water, Na⁺, Cl⁻, Ca²⁺)
  Solvent model  TIP3P explicit, rectangular box, PME long-range electrostatics

Outputs (in --output_dir)
--------------------------
  frame_000_minimised.pdb, frame_001_minimised.pdb, …
  combined_minimised.pdb    — all frames in one multiframe PDB
  energies.csv              — potential energy per iteration per frame
  energy_curves.png         — three-panel figure

Usage
-----
  python minimise_complexes.py \\
      --input  combined_outputs_minimised/combined_all.pdb \\
      --output_dir openmm_minimised/

  # Charged ligand, custom box padding:
  python minimise_complexes.py --input ... --ligand_charge -1 --box_padding 1.5

Requirements
------------
  openmm >= 8.0
  openmmforcefields >= 0.11
  openff-toolkit >= 0.14
  pdbfixer
  MDAnalysis
  rdkit >= 2022.03   (rdDetermineBonds)
  matplotlib  numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

try:
    from rdkit.Chem import rdDetermineBonds as _rddb

    _HAS_RDDB = True
except ImportError:
    _HAS_RDDB = False

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    import pdbfixer
    from openff.toolkit import Molecule as OFFMolecule
    from openmmforcefields.generators import SystemGenerator
except ImportError as exc:
    sys.exit(
        f"[ERROR] Missing dependency: {exc}\n"
        "  pip install openmm openmmforcefields 'openff-toolkit>=0.14' pdbfixer"
    )

# ── Defaults ───────────────────────────────────────────────────────────────────
_FF_PROTEIN = "amber/ff14SB.xml"
_FF_WATER = "amber14/tip3p.xml"  # TIP3P water + Na⁺/Cl⁻/Ca²⁺ ion parameters
_FF_LIGAND = "openff-2.1.0"
_RESTRAINT_K = 100.0  # kcal/mol/Å² on protein Cα atoms
_MAX_ITERS = 2000
_BLOCK_SIZE = 50  # minimisation iterations between energy recordings
_CONV_TOL = 0.01  # kcal/mol — stop when |ΔE_block| < this
_BOX_PADDING_NM = 1.2  # nm of water padding around the solute
_NONBONDED_CUTOFF_NM = 1.0  # nm PME real-space cutoff
_SALT_CONC_M = 0.15  # mol/L — NaCl concentration added to solvent box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 0.  SDF template loader ───────────────────────────────────────────────────


def _load_cdz_template(sdf_path: Path) -> Chem.Mol:
    """
    Read the first mol from *sdf_path*, strip Ca²⁺ ions (atomic number 20),
    and return a sanitized heavy-atom RDKit mol whose bond topology will be
    re-used for every PDB frame (positions are replaced per-frame).
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"Could not parse SDF template: {sdf_path}")
    edit = Chem.RWMol(mol)
    ca_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 20]
    for idx in sorted(ca_idxs, reverse=True):
        edit.RemoveAtom(idx)
    cdz = edit.GetMol()
    log.info(f"SDF template loaded: {cdz.GetNumAtoms()} heavy atoms from {sdf_path.name}")
    return cdz


# ── 1.  Frame loading and splitting ───────────────────────────────────────────


def load_frames(pdb_path: Path) -> mda.Universe:
    """Load all MODEL frames into memory."""
    u = mda.Universe(str(pdb_path), in_memory=True)
    log.info(
        f"Loaded {u.trajectory.n_frames} frame(s) from {pdb_path.name}  "
        f"({u.atoms.n_atoms} atoms/frame)"
    )
    return u


def _ligand_atomgroup(u: mda.Universe) -> mda.AtomGroup:
    """
    Return chain-Z atoms regardless of whether MDAnalysis stores the chain
    in segid or chainID.
    """
    for sel in ("segid Z", "chainID Z"):
        try:
            ag = u.select_atoms(sel)
            if ag.n_atoms > 0:
                return ag
        except Exception:
            pass
    return u.atoms[[]]  # empty


def split_frame(
    u: mda.Universe,
) -> tuple[mda.AtomGroup, mda.AtomGroup, mda.AtomGroup]:
    """
    Split the current trajectory frame into:
      prot_ag  — all protein atoms (not chain Z)
      cdz_ag   — multi-atom CDZ ligand on chain Z
      ca_ag    — Ca²⁺ ions on chain Z
    """
    lig = _ligand_atomgroup(u)
    prot = u.atoms - lig
    cdz = lig.select_atoms("resname CDZ")
    ca = lig.select_atoms("resname CA")
    return prot, cdz, ca


# ── 2.  CDZ: MDAnalysis → RDKit mol with bond orders + explicit H ─────────────


def cdz_to_rdkit(
    cdz_ag: mda.AtomGroup,
    charge: int = 0,
    template: Chem.Mol | None = None,
) -> Chem.Mol | None:
    """
    Convert a CDZ AtomGroup (heavy atoms only) to a sanitized RDKit Mol with
    explicit H placed in 3-D.

    When *template* is supplied (a mol loaded from the NeuralPLexer SDF with
    correct bond orders), its connectivity is re-used and only the conformer
    coordinates are replaced from *cdz_ag*.  This avoids the MDA bond-guesser
    which cannot handle Cl/Br without custom vdW radii.

    Without a template, bond orders are assigned via rdDetermineBonds
    (requires CONECT records in the PDB).

    Returns None when the AtomGroup is empty.
    Raises RuntimeError when bond-order assignment cannot be completed.
    """
    if cdz_ag.n_atoms == 0:
        return None

    if template is not None:
        # ── SDF-template path: bond topology from SDF, positions from PDB frame ─
        if template.GetNumAtoms() != cdz_ag.n_atoms:
            raise RuntimeError(
                f"SDF template has {template.GetNumAtoms()} heavy atoms but "
                f"CDZ AtomGroup has {cdz_ag.n_atoms}; check --ligand_sdf."
            )
        rwmol = Chem.RWMol(template)
        conf = Chem.Conformer(cdz_ag.n_atoms)
        for i in range(cdz_ag.n_atoms):
            pos = cdz_ag.positions[i]
            conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
        rwmol.RemoveAllConformers()
        rwmol.AddConformer(conf, assignId=True)
        mol = rwmol.GetMol()
    else:
        # ── CONECT path: bond orders via rdDetermineBonds ─────────────────────
        if not _HAS_RDDB:
            raise RuntimeError("rdDetermineBonds not available — upgrade RDKit to >= 2022.03.")

        mol = cdz_ag.convert_to("RDKIT", NoImplicit=False)
        rwmol = Chem.RWMol(mol)

        try:
            _rddb.DetermineBondOrders(rwmol, charge=charge)
        except Exception as exc:
            log.warning(f"DetermineBondOrders failed ({exc}); trying DetermineBonds")
            try:
                _rddb.DetermineBonds(rwmol, charge=charge)
            except Exception as exc2:
                raise RuntimeError(
                    f"Bond-order assignment failed: {exc2}\n"
                    "Try --ligand_charge with the correct net formal charge."
                ) from exc2

        mol = rwmol.GetMol()

    # Place H atoms in 3-D relative to the existing heavy-atom skeleton
    try:
        params = rdmolops.AddHsParameters()
        params.addCoords = True
        mol = rdmolops.AddHs(mol, params)
    except Exception:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    return mol


# ── 3.  Protein: MDA AtomGroup → PDBFixer → OpenMM topology ──────────────────


def fix_protein(
    prot_ag: mda.AtomGroup,
) -> tuple[app.Topology, "openmm.unit.Quantity"]:
    """
    Write protein atoms to a temp PDB, run PDBFixer to add missing heavy atoms
    and hydrogens (pH 7.0), return (topology, positions) in OpenMM units.

    Solvent is NOT added here; it is added to the fully-assembled Modeller
    (protein + ligand + structural ions) inside build_system() so that the
    water box tightly wraps the whole complex.

    A NamedTemporaryFile is used because MDA Writers require a real path.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(tmp_fd)
    try:
        with mda.Writer(tmp_path, n_atoms=prot_ag.n_atoms) as W:
            W.write(prot_ag)

        fixer = pdbfixer.PDBFixer(filename=tmp_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    log.info(f"  PDBFixer: {fixer.topology.getNumAtoms()} protein atoms after fixing")
    return fixer.topology, fixer.positions


# ── 4.  Assemble OpenMM system ────────────────────────────────────────────────


def _positions_to_nm(positions) -> np.ndarray:
    """
    Convert OpenMM Quantity positions (or a plain list of Vec3) to a
    (N, 3) float64 array in nanometres.
    """
    try:
        return np.asarray(positions.value_in_unit(unit.nanometers))
    except AttributeError:
        return np.asarray([[v[0], v[1], v[2]] for v in positions])


def _ca_ion_topology(n_ions: int) -> app.Topology:
    """Minimal OpenMM Topology for n Ca²⁺ ions, used for adding to the Modeller."""
    top = app.Topology()
    chain = top.addChain("I")
    for _ in range(n_ions):
        res = top.addResidue("CA", chain)
        top.addAtom("CA", app.Element.getBySymbol("Ca"), res)
    return top


def _best_platform() -> "openmm.Platform":
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return openmm.Platform.getPlatformByName(name)
        except Exception as exc:
            print(f"Platform {name} not available; trying next option... ({exc})")
            continue
    return openmm.Platform.getPlatformByName("Reference")


def build_system(
    prot_top: app.Topology,
    prot_pos,
    cdz_mol: Chem.Mol | None,
    ca_ag: mda.AtomGroup,
    *,
    restraint_k: float = _RESTRAINT_K,
    box_padding_nm: float = _BOX_PADDING_NM,
    salt_conc_m: float = _SALT_CONC_M,
    nonbonded_cutoff_nm: float = _NONBONDED_CUTOFF_NM,
    cached_off_mols: list[OFFMolecule] | None = None,
) -> tuple[openmm.System, app.Topology, app.Simulation, list[OFFMolecule]]:
    """
    Combine protein + CDZ fragment(s) + Ca²⁺ ions into one OpenMM system
    with TIP3P explicit solvent, PME electrostatics, and harmonic Cα restraints.

    Solvation workflow
    ------------------
    1. Assemble a Modeller with protein, ligand fragments, and structural Ca²⁺.
    2. Call modeller.addSolvent() using the SystemGenerator's ForceField object
       so that TIP3P water and NaCl counter-ions are placed around the full
       complex in one rectangular box.
    3. Pass the solvated topology to generator.create_system() which applies
       ff14SB + TIP3P + OpenFF parameters in a single step.

    This ordering (assemble → solvate → parametrise) ensures:
      • the box dimensions wrap the entire solute,
      • no solvent molecules overlap with the ligand or ions, and
      • the SystemGenerator sees the complete, solvated topology.

    CDZ handling
    ------------
    Disconnected fragments are added individually so that SystemGenerator can
    parametrise each piece with OpenFF.

    Returns (system, combined_topology, simulation, off_mols).
    """
    modeller = app.Modeller(prot_top, prot_pos)
    off_mols: list[OFFMolecule] = cached_off_mols[:] if cached_off_mols else []

    # ── CDZ fragments ─────────────────────────────────────────────────────────
    if cdz_mol is not None and not off_mols:
        frag_mols = list(Chem.GetMolFrags(cdz_mol, asMols=True, sanitizeFrags=False))

        for f_idx, frag_mol in enumerate(frag_mols):
            try:
                off_mol = OFFMolecule.from_rdkit(frag_mol, allow_undefined_stereo=True)
            except Exception as exc:
                raise RuntimeError(
                    f"OpenFF parametrisation failed for CDZ fragment {f_idx}: {exc}\n"
                    "Check the ligand connectivity or try --ligand_charge."
                ) from exc

            off_mols.append(off_mol)

            frag_top_omm = off_mol.to_topology().to_openmm()
            conf = frag_mol.GetConformer()
            frag_pos = [
                openmm.Vec3(
                    conf.GetAtomPosition(j).x * 0.1,  # Å → nm
                    conf.GetAtomPosition(j).y * 0.1,
                    conf.GetAtomPosition(j).z * 0.1,
                )
                for j in range(frag_mol.GetNumAtoms())
            ] * unit.nanometers

            modeller.add(frag_top_omm, frag_pos)

        log.info(f"  CDZ: {cdz_mol.GetNumAtoms()} atoms, {len(frag_mols)} fragment(s)")

    # ── Ca²⁺ structural ions ──────────────────────────────────────────────────
    if ca_ag.n_atoms > 0:
        ca_top = _ca_ion_topology(ca_ag.n_atoms)
        ca_pos = [
            openmm.Vec3(
                float(ca_ag.positions[i, 0]) * 0.1,
                float(ca_ag.positions[i, 1]) * 0.1,
                float(ca_ag.positions[i, 2]) * 0.1,
            )
            for i in range(ca_ag.n_atoms)
        ] * unit.nanometers
        modeller.add(ca_top, ca_pos)
        log.info(f"  Ca²⁺: {ca_ag.n_atoms} structural ion(s)")

    # ── SystemGenerator — ff14SB + TIP3P + OpenFF ligand, PME ────────────────
    # forcefield_kwargs configure the nonbonded method for the explicit-solvent
    # system.  PME handles long-range electrostatics; HBonds constraints allow
    # a 2 fs timestep.  No implicit-solvent dielectric parameters are used.
    generator = SystemGenerator(
        forcefields=[_FF_PROTEIN, _FF_WATER],
        small_molecule_forcefield=_FF_LIGAND,
        molecules=off_mols,
        periodic_forcefield_kwargs={
            "nonbondedMethod": app.PME,
            "nonbondedCutoff": nonbonded_cutoff_nm * unit.nanometers,
            "constraints": app.HBonds,
        },
    )

    # ── Add TIP3P water box + NaCl around the fully-assembled solute ──────────
    # addSolvent() uses the SystemGenerator's internal ForceField so that water
    # and ion templates are recognised.  Structural Ca²⁺ ions already present
    # in the Modeller are counted when neutralising the system.
    n_atoms_before_solvent = modeller.topology.getNumAtoms()
    modeller.addSolvent(
        generator.forcefield,
        model="tip3p",
        padding=box_padding_nm * unit.nanometers,
        ionicStrength=salt_conc_m * unit.molar,
        neutralize=True,
    )
    n_solvent = modeller.topology.getNumAtoms() - n_atoms_before_solvent
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box_nm = [v[i].value_in_unit(unit.nanometers) for i, v in enumerate(box_vecs)]
    log.info(
        f"  Solvent: {n_solvent} atoms added  "
        f"(box ≈ {box_nm[0]:.2f} × {box_nm[1]:.2f} × {box_nm[2]:.2f} nm)"
    )

    # ── Create the parametrised system ───────────────────────────────────────
    system = generator.create_system(modeller.topology)

    # ── Harmonic position restraints on protein Cα ────────────────────────────
    _add_backbone_restraints(system, modeller.topology, modeller.positions, restraint_k)

    # ── Simulation object ─────────────────────────────────────────────────────
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )
    platform = _best_platform()
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    log.info(f"  Platform: {platform.getName()}")
    log.info(f"  Total atoms in system: {modeller.topology.getNumAtoms()}")

    return system, modeller.topology, simulation, off_mols


def _add_backbone_restraints(
    system: openmm.System,
    topology: app.Topology,
    positions,
    k_kcal: float,
) -> None:
    """
    Harmonic position restraints on every protein Cα atom.

    k_kcal is in kcal/mol/Å²; converted to kJ/mol/nm² internally.
    Ca²⁺ ions (residue name "CA") are explicitly excluded — they share the
    atom name "CA" with protein alpha-carbons.
    """
    k_kj_nm2 = k_kcal * 4.184 * 100.0  # kcal/mol/Å² → kJ/mol/nm²

    force = openmm.CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    force.addGlobalParameter("k", k_kj_nm2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    pos_nm = _positions_to_nm(positions)
    n_restrained = 0

    for atom in topology.atoms():
        if atom.name == "CA" and atom.residue.name not in {"CA"}:
            i = atom.index
            force.addParticle(i, [pos_nm[i, 0], pos_nm[i, 1], pos_nm[i, 2]])
            n_restrained += 1

    system.addForce(force)
    log.info(f"  Cα restraints on {n_restrained} atoms  (k = {k_kcal} kcal/mol/Å²)")


# ── 5.  Chunked minimisation with energy recording ───────────────────────────


@dataclass
class FrameRecord:
    frame_idx: int
    steps: list[int] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)  # kcal/mol
    converged: bool = False

    @property
    def e0(self) -> float:
        return self.energies[0] if self.energies else float("nan")

    @property
    def e_final(self) -> float:
        return self.energies[-1] if self.energies else float("nan")

    @property
    def delta_e(self) -> float:
        return self.e_final - self.e0


def _current_energy(simulation: app.Simulation) -> float:
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


def chunked_minimise(
    simulation: app.Simulation,
    frame_idx: int,
    max_iters: int = _MAX_ITERS,
    block_size: int = _BLOCK_SIZE,
    conv_tol: float = _CONV_TOL,
) -> FrameRecord:
    """
    Run energy minimisation in blocks of `block_size` iterations, recording
    the potential energy after each block.

    Convergence is declared when the energy drop across a single block falls
    below `conv_tol` kcal/mol.  Each `simulation.minimizeEnergy(maxIterations=N)`
    call continues from the current context positions, so the energy trace is
    monotonically non-increasing.

    Returns a FrameRecord with steps[] and energies[] aligned by index.
    """
    rec = FrameRecord(frame_idx=frame_idx)
    rec.steps.append(0)
    rec.energies.append(_current_energy(simulation))
    log.info(f"  E_start = {rec.e0:>14,.2f} kcal/mol")

    iters_done = 0
    while iters_done < max_iters:
        block = min(block_size, max_iters - iters_done)
        simulation.minimizeEnergy(maxIterations=block)
        iters_done += block

        e = _current_energy(simulation)
        delta = abs(rec.energies[-1] - e)

        rec.steps.append(iters_done)
        rec.energies.append(e)

        log.debug(f"    iter {iters_done:5d}  E = {e:>14,.2f} kcal/mol  |ΔE| = {delta:.5f}")

        if delta < conv_tol:
            rec.converged = True
            log.info(f"  Converged at iter {iters_done}  |ΔE| = {delta:.5f} kcal/mol")
            break

    if not rec.converged:
        log.info(f"  Reached max_iters={max_iters} without convergence")

    log.info(
        f"  E_final = {rec.e_final:>14,.2f} kcal/mol  ΔE_total = {rec.delta_e:>+,.2f} kcal/mol"
    )
    return rec


# ── 6.  Output helpers ────────────────────────────────────────────────────────


def write_pdb(
    simulation: app.Simulation,
    topology: app.Topology,
    path: Path,
) -> None:
    """Write the current context positions to a PDB file (includes CRYST1 record)."""
    state = simulation.context.getState(getPositions=True)
    with open(path, "w") as fh:
        app.PDBFile.writeFile(topology, state.getPositions(), fh)


def write_combined_pdb(pdb_paths: list[Path], out_path: Path) -> None:
    """Concatenate individual PDB files as MODEL/ENDMDL blocks."""
    with open(out_path, "w") as out:
        for model_num, p in enumerate(pdb_paths, start=1):
            out.write(f"MODEL     {model_num:4d}\n")
            for line in p.read_text().splitlines():
                if line.startswith(("END", "MODEL", "ENDMDL")):
                    continue
                out.write(line + "\n")
            out.write("ENDMDL\n")
    log.info(f"Combined PDB: {out_path.name}  ({len(pdb_paths)} models)")


def write_csv(records: list[FrameRecord], path: Path) -> None:
    import csv

    rows = [
        {
            "frame": r.frame_idx,
            "iteration": s,
            "energy_kcal_mol": round(e, 4),
            "delta_e_kcal_mol": round(e - r.e0, 4),
            "converged": r.converged,
        }
        for r in records
        for s, e in zip(r.steps, r.energies)
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame",
                "iteration",
                "energy_kcal_mol",
                "delta_e_kcal_mol",
                "converged",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"CSV: {path.name}  ({len(rows)} rows)")


def plot_energies(records: list[FrameRecord], path: Path) -> None:
    """
    Three-panel figure:
      Left   – absolute potential energy vs iteration (E in kcal/mol)
      Centre – ΔE = E − E₀  (convergence rate, all frames aligned at 0)
      Right  – horizontal bar chart of total ΔE per frame (quick summary)
    """
    n = len(records)
    if n == 0:
        log.warning("No records to plot.")
        return

    cmap = plt.cm.viridis
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    labels = [f"F{r.frame_idx}" + (" ✓" if r.converged else "") for r in records]

    fig, (ax_abs, ax_rel, ax_bar) = plt.subplots(
        1,
        3,
        figsize=(19, 5),
        gridspec_kw={"width_ratios": [5, 5, 3]},
    )

    # ── Left: absolute energy ─────────────────────────────────────────────────
    for rec, col, lbl in zip(records, colors, labels):
        ax_abs.plot(rec.steps, rec.energies, color=col, lw=1.5, label=lbl)

    ax_abs.set_xlabel("Minimisation iteration", fontsize=11)
    ax_abs.set_ylabel("Potential energy (kcal/mol)", fontsize=11)
    ax_abs.set_title("Absolute energy", fontsize=12)
    ax_abs.grid(True, alpha=0.22, linestyle="--")
    ax_abs.legend(fontsize=7, ncol=max(1, n // 8), loc="upper right", framealpha=0.7)

    # ── Centre: energy drop ───────────────────────────────────────────────────
    for rec, col, lbl in zip(records, colors, labels):
        delta = [e - rec.e0 for e in rec.energies]
        ax_rel.plot(rec.steps, delta, color=col, lw=1.5, label=lbl)
        ax_rel.annotate(
            f"{rec.delta_e:+.0f}",
            xy=(rec.steps[-1], rec.delta_e),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=6,
            color=col,
            va="center",
        )

    ax_rel.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax_rel.set_xlabel("Minimisation iteration", fontsize=11)
    ax_rel.set_ylabel("ΔE from starting pose (kcal/mol)", fontsize=11)
    ax_rel.set_title("Energy drop relative to starting pose", fontsize=12)
    ax_rel.grid(True, alpha=0.22, linestyle="--")
    ax_rel.legend(fontsize=7, ncol=max(1, n // 8), loc="lower right", framealpha=0.7)

    # ── Right: total ΔE bar chart ─────────────────────────────────────────────
    delta_totals = [r.delta_e for r in records]
    bar_colors = ["#27ae60" if r.converged else col for r, col in zip(records, colors)]
    ax_bar.barh(
        range(n),
        delta_totals,
        color=bar_colors,
        edgecolor="k",
        linewidth=0.4,
        height=0.7,
    )
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(labels, fontsize=8)
    ax_bar.set_xlabel("Total ΔE (kcal/mol)", fontsize=11)
    ax_bar.set_title("Total energy change", fontsize=12)
    ax_bar.axvline(0, color="k", lw=0.8)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, axis="x", alpha=0.2, linestyle="--")

    for i, dE in enumerate(delta_totals):
        ax_bar.text(
            dE / 2,
            i,
            f"{dE:+.0f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_abs, ax_rel], shrink=0.82, pad=0.01)
    cbar.set_label("Frame index", fontsize=9)

    from matplotlib.patches import Patch

    ax_bar.legend(
        handles=[Patch(facecolor="#27ae60", label="Converged")],
        fontsize=8,
        loc="lower right",
    )

    plt.suptitle(
        "OpenMM TIP3P explicit-solvent minimisation — energy trajectories",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot: {path.name}")


# ── 7.  CLI ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Multiframe PDB from combine_neuralplexer.py (combined_all.pdb)",
    )
    p.add_argument(
        "--output_dir",
        required=False,
        type=Path,
        help="Output directory (created if absent)",
    )
    p.add_argument(
        "--ligand_sdf",
        type=Path,
        default=None,
        metavar="SDF",
        help=(
            "SDF file containing the CDZ ligand (e.g. lig_ref.sdf from the "
            "NeuralPLexer ensemble output).  Bond topology is read once from "
            "this file and re-used for every frame, bypassing the MDA "
            "bond-guesser (required when the ligand contains Cl/Br)."
        ),
    )
    p.add_argument(
        "--ligand_charge",
        default=1,
        type=int,
        help="Net formal charge of the CDZ ligand (default: 1)",
    )
    p.add_argument(
        "--restraint_k",
        default=_RESTRAINT_K,
        type=float,
        metavar="KCAL",
        help=f"Cα harmonic restraint k in kcal/mol/Å² (default: {_RESTRAINT_K})",
    )
    p.add_argument(
        "--max_iters",
        default=_MAX_ITERS,
        type=int,
        help=f"Max minimisation iterations per frame (default: {_MAX_ITERS})",
    )
    p.add_argument(
        "--block_size",
        default=_BLOCK_SIZE,
        type=int,
        help=f"Iterations per energy-recording block (default: {_BLOCK_SIZE})",
    )
    p.add_argument(
        "--conv_tol",
        default=_CONV_TOL,
        type=float,
        help=f"Convergence |ΔE| threshold kcal/mol (default: {_CONV_TOL})",
    )
    p.add_argument(
        "--box_padding",
        default=_BOX_PADDING_NM,
        type=float,
        metavar="NM",
        help=f"Water-box padding around solute in nm (default: {_BOX_PADDING_NM})",
    )
    p.add_argument(
        "--salt_conc",
        default=_SALT_CONC_M,
        type=float,
        metavar="M",
        help=f"NaCl ionic strength added to solvent box in mol/L (default: {_SALT_CONC_M})",
    )
    p.add_argument(
        "--nonbonded_cutoff",
        default=_NONBONDED_CUTOFF_NM,
        type=float,
        metavar="NM",
        help=f"PME real-space cutoff in nm (default: {_NONBONDED_CUTOFF_NM})",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = args.input.parent / f"{args.input.stem}_minimised"
        log.info(f"No --output_dir provided; using {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    u = load_frames(args.input)
    n_frames = u.trajectory.n_frames

    cdz_template: Chem.Mol | None = None
    cached_off_mols: list[OFFMolecule] = []
    if args.ligand_sdf is not None:
        if not args.ligand_sdf.exists():
            log.error(f"--ligand_sdf not found: {args.ligand_sdf}")
            sys.exit(1)
        cdz_template = _load_cdz_template(args.ligand_sdf)

        cache_path = args.ligand_sdf.with_suffix(".off.sdf")
        if cache_path.exists():
            try:
                cached_off_mols = [OFFMolecule.from_file(str(cache_path))]
                log.info(f"Loaded cached OpenFF molecules from {cache_path.name}")
            except Exception as exc:
                log.warning(f"Could not load cache from {cache_path}: {exc}")
                cached_off_mols = []
        else:
            cached_off_mols = []

    records: list[FrameRecord] = []
    written_pdbs: list[Path] = []

    for frame_idx, _ts in enumerate(u.trajectory):
        log.info(f"\n{'─' * 60}")
        log.info(f"Frame {frame_idx + 1}/{n_frames}")
        log.info(f"{'─' * 60}")

        prot_ag, cdz_ag, ca_ag = split_frame(u)
        log.info(
            f"  Split: {prot_ag.n_atoms} protein  |  {cdz_ag.n_atoms} CDZ  |  {ca_ag.n_atoms} Ca²⁺"
        )

        try:
            cdz_mol = cdz_to_rdkit(cdz_ag, charge=args.ligand_charge, template=cdz_template)
        except RuntimeError as exc:
            log.error(f"Ligand preparation failed: {exc}")
            continue

        try:
            prot_top, prot_pos = fix_protein(prot_ag)
        except Exception as exc:
            log.error(f"PDBFixer failed: {exc}")
            continue

        try:
            _sys, topology, simulation, generated_off_mols = build_system(
                prot_top,
                prot_pos,
                cdz_mol,
                ca_ag,
                restraint_k=args.restraint_k,
                box_padding_nm=args.box_padding,
                salt_conc_m=args.salt_conc,
                nonbonded_cutoff_nm=args.nonbonded_cutoff,
                cached_off_mols=cached_off_mols,
            )
        except Exception as exc:
            log.error(f"System build failed: {exc}")
            continue

        if args.ligand_sdf is not None and not cached_off_mols and generated_off_mols:
            cache_path = args.ligand_sdf.with_suffix(".off.sdf")
            try:
                generated_off_mols[0].to_file(str(cache_path), file_format="sdf")
                log.info(f"Cached OpenFF molecules to {cache_path.name}")
                cached_off_mols = generated_off_mols
            except Exception as exc:
                log.warning(f"Could not save OpenFF cache: {exc}")

        rec = chunked_minimise(
            simulation,
            frame_idx,
            max_iters=args.max_iters,
            block_size=args.block_size,
            conv_tol=args.conv_tol,
        )
        records.append(rec)

        out_pdb = args.output_dir / f"frame_{frame_idx:03d}_minimised.pdb"
        write_pdb(simulation, topology, out_pdb)
        written_pdbs.append(out_pdb)
        log.info(f"  → {out_pdb.name}")

    if not records:
        log.error("No frames were successfully minimised.")
        sys.exit(1)

    write_combined_pdb(written_pdbs, args.output_dir / "combined_minimised.pdb")
    write_csv(records, args.output_dir / "energies.csv")
    plot_energies(records, args.output_dir / "energy_curves.png")

    # ── Final summary table ───────────────────────────────────────────────────
    n_conv = sum(r.converged for r in records)
    sep = "═" * 65
    log.info(f"\n{sep}")
    log.info(f"DONE — {len(records)}/{n_frames} frames minimised, {n_conv} converged")
    log.info(sep)
    log.info(f"  {'Frame':>5}  {'E_start':>14}  {'E_final':>14}  {'ΔE':>12}  Status")
    log.info(f"  {'─' * 5}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 9}")
    for r in records:
        log.info(
            f"  {r.frame_idx:>5}  "
            f"{r.e0:>14,.2f}  "
            f"{r.e_final:>14,.2f}  "
            f"{r.delta_e:>+12,.2f}  "
            f"{'converged' if r.converged else 'max_iters'}"
        )
    log.info(sep)


if __name__ == "__main__":
    main()
