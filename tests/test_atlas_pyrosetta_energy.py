import numpy as np

from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_common import (
    build_atom_mapping,
    canonical_pdb_atom_name,
    score_term_group,
)


def test_canonical_pdb_atom_names_cover_gromacs_aliases():
    assert canonical_pdb_atom_name("HN", "ALA") == "H"
    assert canonical_pdb_atom_name("OT2", "ALA") == "OXT"
    assert canonical_pdb_atom_name("HB1", "ALA") == "1HB"
    assert canonical_pdb_atom_name("HG11", "VAL") == "1HG1"
    assert canonical_pdb_atom_name("HD2", "ILE") == "2HD1"
    assert canonical_pdb_atom_name("HG1", "SER") == "HG"


def test_atom_mapping_uses_names_then_element_constrained_fallback():
    pose_atoms = [
        {"resindex": 1, "atomno": 1, "name": "N", "element": "N", "virtual": False, "position": [0, 0, 0]},
        {"resindex": 1, "atomno": 2, "name": "CA", "element": "C", "virtual": False, "position": [1, 0, 0]},
        {"resindex": 1, "atomno": 3, "name": "1HB", "element": "H", "virtual": False, "position": [2, 0, 0]},
        {"resindex": 1, "atomno": 4, "name": "NV", "element": "X", "virtual": True, "position": [0, 0, 0]},
    ]
    mapping, metadata = build_atom_mapping(
        ["N", "CA", "HX"],
        ["ALA", "ALA", "ALA"],
        np.array([1, 1, 1]),
        ["N", "C", "H"],
        np.array([[0, 0, 0], [1, 0, 0], [2.2, 0, 0]]),
        pose_atoms,
    )
    assert mapping.tolist() == [[1, 1], [1, 2], [1, 3]]
    assert metadata["fallback_atoms"] == 1


def test_score_term_groups_are_explicit():
    assert score_term_group("fa_atr") == "packing"
    assert score_term_group("fa_sol") == "solvation"
    assert score_term_group("hbond_sc") == "electrostatic_hbond"
    assert score_term_group("fa_dun") == "torsional_rotamer"
    assert score_term_group("cart_bonded") == "cartesian_bonded"
    assert score_term_group("ref") == "reference_disulfide"


def test_atom_mapping_may_ignore_only_unrepresented_hydrogen():
    pose_atoms = [
        {"resindex": 1, "atomno": 1, "name": "N", "element": "N", "virtual": False, "position": [0, 0, 0]},
        {"resindex": 1, "atomno": 2, "name": "H", "element": "H", "virtual": False, "position": [1, 0, 0]},
    ]
    mapping, metadata = build_atom_mapping(
        ["N", "HN", "HD1"],
        ["HIS", "HIS", "HIS"],
        np.array([1, 1, 1]),
        ["N", "H", "H"],
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        pose_atoms,
    )
    assert mapping.tolist() == [[1, 1], [1, 2], [0, 0]]
    assert metadata["ignored_pdb_hydrogens"] == ["1:HD1"]
