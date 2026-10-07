"""PyRosetta-independent helpers for checkpoint-24 trajectory rescoring."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def canonical_pdb_atom_name(name: str, residue_name: str) -> str:
    """Translate the GROMACS/PDB atom names used by ATLAS to Rosetta names."""
    atom = str(name).strip().upper()
    residue = str(residue_name).strip().upper()
    aliases = {"HN": "H", "OT1": "O", "OT2": "OXT"}
    if atom in aliases:
        return aliases[atom]
    if residue == "ILE" and atom == "CD":
        return "CD1"
    if residue == "ILE" and re.fullmatch(r"HD[123]", atom):
        return atom[-1] + "HD1"
    if residue in {"SER", "CYS"} and atom == "HG1":
        return "HG"
    match = re.fullmatch(r"(H[A-Z]*\d*)([123])", atom)
    if match:
        return match.group(2) + match.group(1)
    return atom


def build_atom_mapping(
    pdb_names: list[str],
    pdb_resnames: list[str],
    pdb_resindices: np.ndarray,
    pdb_elements: list[str],
    pdb_positions: np.ndarray,
    pose_atoms: list[dict],
    heavy_tolerance_angstrom: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Map every PDB atom to a real Pose atom, residue by residue.

    Canonical names are authoritative. Coordinate assignment is only a fallback for
    otherwise unmatched atoms of the same element; heavy-atom fallbacks must remain
    essentially coincident with the PDB used to create the Pose.
    """
    count = len(pdb_names)
    mapping = np.full((count, 2), -1, dtype=np.int32)
    fallback_distances: list[float] = []
    ignored_hydrogens: list[str] = []
    pose_by_residue: dict[int, list[dict]] = {}
    for atom in pose_atoms:
        if not atom["virtual"]:
            pose_by_residue.setdefault(int(atom["resindex"]), []).append(atom)
    residues = np.unique(np.asarray(pdb_resindices, dtype=int))
    for resindex in residues:
        pdb_indices = np.where(np.asarray(pdb_resindices) == resindex)[0]
        candidates = pose_by_residue.get(int(resindex), [])
        if len(pdb_indices) < len(candidates):
            raise ValueError(
                f"residue {resindex}: {len(pdb_indices)} PDB atoms but "
                f"{len(candidates)} real Pose atoms"
            )
        remaining_pdb = set(map(int, pdb_indices))
        remaining_pose = set(range(len(candidates)))
        by_name: dict[str, list[int]] = {}
        for candidate_index, atom in enumerate(candidates):
            by_name.setdefault(str(atom["name"]).strip().upper(), []).append(
                candidate_index
            )
        for pdb_index in pdb_indices:
            canonical = canonical_pdb_atom_name(
                pdb_names[pdb_index], pdb_resnames[pdb_index]
            )
            matches = [i for i in by_name.get(canonical, []) if i in remaining_pose]
            if len(matches) == 1:
                candidate_index = matches[0]
                atom = candidates[candidate_index]
                mapping[pdb_index] = (atom["resindex"], atom["atomno"])
                remaining_pdb.remove(int(pdb_index))
                remaining_pose.remove(candidate_index)
        for element in sorted(
            {str(pdb_elements[index]).strip().upper() for index in remaining_pdb}
        ):
            pdb_group = [
                index
                for index in remaining_pdb
                if str(pdb_elements[index]).strip().upper() == element
            ]
            pose_group = [
                index
                for index in remaining_pose
                if str(candidates[index]["element"]).strip().upper() == element
            ]
            if len(pdb_group) < len(pose_group):
                raise ValueError(
                    f"residue {resindex} element {element}: unmatched counts differ"
                )
            if not pose_group:
                continue
            left = np.asarray(pdb_positions, dtype=float)[pdb_group]
            right = np.asarray(
                [candidates[index]["position"] for index in pose_group], dtype=float
            )
            costs = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                distance = float(costs[row, column])
                if element != "H" and distance > heavy_tolerance_angstrom:
                    raise ValueError(
                        f"residue {resindex}: heavy-atom fallback is {distance:.3f} A"
                    )
                pdb_index = pdb_group[row]
                candidate_index = pose_group[column]
                atom = candidates[candidate_index]
                mapping[pdb_index] = (atom["resindex"], atom["atomno"])
                fallback_distances.append(distance)
                remaining_pdb.remove(pdb_index)
                remaining_pose.remove(candidate_index)
        if remaining_pose:
            raise ValueError(f"residue {resindex}: incomplete atom mapping")
        for pdb_index in sorted(remaining_pdb):
            if str(pdb_elements[pdb_index]).strip().upper() != "H":
                raise ValueError(f"residue {resindex}: unmatched non-hydrogen PDB atom")
            mapping[pdb_index] = (0, 0)
            ignored_hydrogens.append(
                f"{resindex}:{str(pdb_names[pdb_index]).strip()}"
            )
    if np.any(mapping < 0):
        raise ValueError("atom mapping contains unresolved entries")
    checksum = hashlib.sha256(mapping.tobytes()).hexdigest()
    return mapping, {
        "atoms": count,
        "fallback_atoms": len(fallback_distances),
        "maximum_fallback_distance_angstrom": max(fallback_distances, default=0.0),
        "ignored_pdb_hydrogens": ignored_hydrogens,
        "mapping_sha256": checksum,
    }


def score_term_group(term: str) -> str:
    """Assign a weighted Rosetta score term to a coarse diagnostic family."""
    if term in {"fa_atr", "fa_rep", "fa_intra_rep", "fa_intra_atr_xover4"}:
        return "packing"
    if term in {"fa_sol", "fa_intra_sol_xover4", "lk_ball", "lk_ball_wtd"}:
        return "solvation"
    if term == "fa_elec" or term.startswith("hbond_"):
        return "electrostatic_hbond"
    if term in {
        "fa_dun",
        "rama_prepro",
        "omega",
        "p_aa_pp",
        "pro_close",
        "yhh_planarity",
    }:
        return "torsional_rotamer"
    if term.startswith("cart_bonded"):
        return "cartesian_bonded"
    if term in {"ref", "dslf_fa13"}:
        return "reference_disulfide"
    return "other"
