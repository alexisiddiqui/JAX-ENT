#!/usr/bin/env python3
"""Score the unchanged ISO TRI frames in the Python 3.11 PyRosetta environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import MDAnalysis as mda
import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "jaxent/examples/ATLAS_BV/analysis"))
from pyrosetta_energy_score_checkpoint24 import (
    guessed_adjacency, set_pose_coordinates, pose_atom_records,
)
from pyrosetta_energy_common import build_atom_mapping
from openmm_vacuum_common import unwrap_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-limit", type=int)
    args = parser.parse_args()
    sys.path.append("/home/alexi/anaconda3/lib/python3.11/site-packages")
    import pyrosetta as pyro
    pyro.init("-mute all -ignore_unrecognized_res true -pH_mode true")
    universe = mda.Universe(str(args.topology), topology_format="PDB")
    # ISO uses GLH/HSE/HSP protonation aliases and an NHT cap, with the
    # terminal carbonyl O stored in the cap residue. Preserve that chemistry
    # using native Rosetta residue types rather than dropping unknown residues.
    aliases = {"GLH": "GLU", "HSE": "HIS", "HSP": "HIS"}
    one_letter = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
                          "ARNDCQEGHILKMFPSTWYV"))
    residues = [r for r in universe.residues if r.resname != "NHT"]
    sequence = "".join(one_letter[aliases.get(r.resname, r.resname)] for r in residues)
    pose = pyro.pose_from_sequence(sequence, "fa_standard")
    types = pose.residue_type_set_for_pose()
    for index, residue in enumerate(residues, 1):
        name = {"GLH": "GLU_P1", "HSP": "HIS_P"}.get(residue.resname)
        if name:
            replacement = pyro.rosetta.core.conformation.ResidueFactory.create_residue(types.name_map(name))
            pose.replace_residue(index, replacement, True)
    caps = [r for r in universe.residues if r.resname == "NHT"]
    if len(caps) != 1 or caps[0].ix != universe.residues[-1].ix:
        raise ValueError("Expected a single terminal ISO NHT amidation cap")
    pyro.rosetta.core.pose.add_variant_type_to_pose_residue(
        pose, pyro.rosetta.core.chemical.CTERM_AMIDATION, pose.total_residue())
    atom_names = list(universe.atoms.names)
    atom_resnames = [aliases.get(n, n) for n in universe.atoms.resnames]
    atom_resindices = np.asarray(universe.atoms.resindices, dtype=int)+1
    for atom in caps[0].atoms:
        atom_resindices[atom.index] = len(residues)
        atom_resnames[atom.index] = residues[-1].resname
        atom_names[atom.index] = {"HT1": "1HN", "HT2": "2HN"}.get(atom.name, atom.name)
    mapping, metadata = build_atom_mapping(atom_names, atom_resnames, atom_resindices,
        list(universe.atoms.elements), universe.atoms.positions, pose_atom_records(pose))
    adjacency = guessed_adjacency(universe)
    # The initial pose has ideal coordinates. Resolve any hydrogen naming
    # fallbacks by bonded heavy parent, rather than its initial Cartesian position.
    pose_hydrogens = {}
    for record in pose_atom_records(pose):
        if record["element"] == "H" and not record["virtual"]:
            parent = pose.residue(record["resindex"]).atom_base(record["atomno"])
            pose_hydrogens.setdefault((record["resindex"], parent), []).append(record["atomno"])
    raw_hydrogens = {}
    for atom in universe.atoms:
        if atom.element == "H":
            parents = [i for i in adjacency[atom.index] if universe.atoms[i].element != "H"]
            if len(parents) != 1:
                raise ValueError(f"Hydrogen has {len(parents)} heavy parents: {atom.index}")
            key = tuple(map(int, mapping[parents[0]]))
            raw_hydrogens.setdefault(key, []).append(atom.index)
    remapped = 0
    for (resindex, parent), indices in raw_hydrogens.items():
        targets = set(pose_hydrogens[resindex, parent])
        if len(indices) != len(targets):
            raise ValueError(f"Hydrogen count mismatch at {resindex}/{parent}")
        pending = []
        for index in indices:
            current = tuple(map(int, mapping[index]))
            if current[0] == resindex and current[1] in targets:
                targets.remove(current[1])
            else:
                pending.append(index)
        for index, target in zip(pending, sorted(targets), strict=True):
            mapping[index] = (resindex, target)
            remapped += 1
    if len(set(map(tuple, mapping))) != len(mapping):
        raise ValueError("Non-bijective atom mapping")
    metadata.update(hydrogen_parent_remapped=remapped,
        mapping_sha256=hashlib.sha256(mapping.tobytes()).hexdigest(),
        hydrogen_fallback_policy="bonded heavy parent; equivalent hydrogen permutations only")
    metadata.update(residues=len(residues), pose_residues=pose.total_residue(),
        protonation_types={str(r.resid): r.resname for r in residues if r.resname in aliases},
        terminal_cap="NHT mapped to Rosetta CTERM_AMIDATION; original atom coordinates retained")
    trajectory = mda.Universe(str(args.topology), str(args.trajectory),
                              topology_format="PDB", format="XTC")
    n = min(len(trajectory.trajectory), args.frame_limit or len(trajectory.trajectory))
    scores = np.empty(n)
    virtual = [(r["resindex"], r["atomno"]) for r in pose_atom_records(pose) if r["virtual"]]
    scorefxn = pyro.create_score_function("ref2015")
    start = time.monotonic()
    first_frame_terms = {}
    max_coordinate_delta = 0.
    for i, frame in enumerate(trajectory.trajectory[:n]):
        xyz = unwrap_positions(frame.positions, np.asarray(frame.triclinic_dimensions), adjacency)
        set_pose_coordinates(pyro, pose, mapping, xyz)
        # Proline NV closure atoms are virtual, absent from the trajectory, and
        # otherwise remain at the ideal extended pose positions. Rebuild only
        # those virtual atoms from the unchanged real-atom coordinates.
        for resindex, atomno in virtual:
            residue = pose.residue(resindex)
            position = residue.icoor(atomno).build(residue, pose.conformation())
            pose.set_xyz(pyro.rosetta.core.id.AtomID(atomno, resindex), position)
        scores[i] = float(scorefxn(pose))
        if i == 0:
            records = {(r["resindex"], r["atomno"]): r["position"] for r in pose_atom_records(pose)}
            mapped_xyz = np.asarray([records[tuple(map(int, pair))] for pair in mapping])
            max_coordinate_delta = float(np.max(np.abs(mapped_xyz-xyz)))
            if max_coordinate_delta > 1e-6:
                raise ValueError("Rosetta coordinates changed during assignment/scoring")
            weights = scorefxn.weights()
            terms = pose.energies().total_energies()
            first_frame_terms = {pyro.rosetta.core.scoring.name_from_score_type(s):
                float(weights[s]*terms[s]) for s in scorefxn.get_nonzero_weighted_scoretypes()}
        if i % 100 == 0:
            print(f"PyRosetta: {i+1}/{n} frames", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite PyRosetta score")
    fingerprint = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.topology, args.trajectory)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, frame=np.arange(n), ref2015_total=scores)
    temporary.replace(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        **metadata, frames=n, scorefunction="ref2015", relaxation="none",
        periodic_boundary_handling="bond-connected unwrapping only",
        first_frame_weighted_terms=first_frame_terms,
        coordinate_assignment_max_absolute_delta=max_coordinate_delta,
        virtual_atom_count=len(virtual), virtual_atom_policy="rebuild internal coordinates; real atoms unchanged",
        input_sha256=fingerprint, pyrosetta_version=pyro._version_string(),
        scoring_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, elapsed_seconds=time.monotonic()-start), indent=2))


if __name__ == "__main__":
    main()
