import numpy as np

from jaxent.examples.ATLAS_BV.analysis.openmm_energy_population_checkpoint23 import (
    GAS_CONSTANT_KJ_MOL_K,
    direct_boltzmann_log_ratio,
    pair_energy_difference,
)
from jaxent.examples.ATLAS_BV.analysis.openmm_vacuum_common import (
    canonical_atom_name,
    unwrap_positions,
)


def test_canonical_atom_name_matches_pdb_and_gromacs_hydrogen_conventions():
    assert canonical_atom_name("1HG2") == canonical_atom_name("HG21")
    assert canonical_atom_name("CA") == "CA"


def test_boltzmann_log_ratio_has_physical_sign_and_scale():
    rt = GAS_CONSTANT_KJ_MOL_K * 300.0
    observed = direct_boltzmann_log_ratio(np.array([rt, -rt, 0.0]), 300.0)
    np.testing.assert_allclose(observed, [-1.0, 1.0, 0.0])


def test_pair_energy_difference_preserves_direction():
    energy = np.array([10.0, 13.0, 4.0])
    observed = pair_energy_difference(energy, np.array([0, 1]), np.array([1, 2]))
    np.testing.assert_allclose(observed, [-3.0, 9.0])


def test_unwrap_positions_makes_bond_crossing_box_whole():
    positions = np.array([[9.9, 5.0, 5.0], [0.1, 5.0, 5.0], [0.3, 5.0, 5.0]])
    box = np.eye(3) * 10.0
    adjacency = [[1], [0, 2], [1]]
    observed = unwrap_positions(positions, box, adjacency)

    np.testing.assert_allclose(
        np.linalg.norm(np.diff(observed, axis=0), axis=1), [0.2, 0.2]
    )
