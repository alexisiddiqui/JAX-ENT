import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.analysis import distances

from jaxent.src.models.func.common import (
    compute_trajectory_average_com_distances,
    get_residue_atom_pairs,
)


@pytest.fixture(scope="module")
def bpti_universe():
    """Fixture to load the BPTI universe, available to all tests in this module."""
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    return mda.Universe(topology_path, trajectory_path)


class TestGetResidueAtomPairs:
    """Test suite for the get_residue_atom_pairs function."""

    def test_get_amide_nitrogens(self, bpti_universe):
        """Test retrieving amide nitrogen atoms."""
        # Select some residues, excluding the first and any prolines
        protein = bpti_universe.select_atoms("protein")
        common_residues = {
            (r.resname, r.resid) for r in protein.residues if r.resid > 1 and r.resname != "PRO"
        }

        nh_pairs = get_residue_atom_pairs(bpti_universe, common_residues, "N")

        assert isinstance(nh_pairs, list)
        assert len(nh_pairs) > 0

        # Check a specific, known residue (e.g., ARG 20)
        arg20_n_atom = protein.select_atoms("resid 20 and name N")
        assert len(arg20_n_atom) == 1
        expected_pair = (20, arg20_n_atom.indices[0])
        assert expected_pair in nh_pairs

    def test_skips_proline_and_first_residue(self, bpti_universe):
        """Test that PRO residues and the first residue are correctly skipped."""
        protein = bpti_universe.select_atoms("protein")
        all_residues = {(r.resname, r.resid) for r in protein.residues}

        nh_pairs = get_residue_atom_pairs(bpti_universe, all_residues, "N")

        # Extract just the residue IDs from the results
        found_resids = {resid for resid, atom_idx in nh_pairs}

        # Check that the first residue (resid 1) is not in the results
        assert 1 not in found_resids

        # Check that no Proline residues are in the results
        proline_resids = {r.resid for r in protein.residues if r.resname == "PRO"}
        assert len(found_resids.intersection(proline_resids)) == 0

    def test_empty_common_residues(self, bpti_universe):
        """Test that an empty set of common residues returns an empty list."""
        result = get_residue_atom_pairs(bpti_universe, set(), "N")
        assert result == []


class TestComputeTrajectoryAverageComDistances:
    """Test suite for compute_trajectory_average_com_distances."""

    def test_basic_calculation(self, bpti_universe):
        """Test basic distance calculation between a few residues."""
        # Select a few residues that are stable and far apart
        res10 = bpti_universe.select_atoms("resid 10")
        res20 = bpti_universe.select_atoms("resid 20")
        res30 = bpti_universe.select_atoms("resid 30")
        group_list = [res10, res20, res30]

        dist_matrix, dist_std = compute_trajectory_average_com_distances(
            bpti_universe, group_list, verbose=False
        )

        assert dist_matrix.shape == (3, 3)
        assert dist_std.shape == (3, 3)

        # Diagonal elements should be zero
        assert np.all(np.diag(dist_matrix) == 0)
        assert np.all(np.diag(dist_std) == 0)

        # Off-diagonal elements should be positive
        assert np.all(dist_matrix[np.triu_indices(3, k=1)] > 0)

        # Symmetry check
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_with_single_frame(self, bpti_universe):
        """Test calculation over a single frame."""
        group_list = [
            bpti_universe.select_atoms("resid 10"),
            bpti_universe.select_atoms("resid 20"),
        ]

        # Manually calculate for the first frame
        bpti_universe.trajectory[0]
        coms = np.array([g.center_of_mass() for g in group_list])
        manual_dist = distances.distance_array(coms, coms, box=bpti_universe.dimensions)

        # Run function for the first frame
        dist_matrix, dist_std = compute_trajectory_average_com_distances(
            bpti_universe, group_list, start=0, stop=1, verbose=False
        )

        assert np.allclose(dist_matrix, manual_dist)
        # Standard deviation for a single frame should be zero
        assert np.allclose(dist_std, 0)

    def test_empty_group_list_raises_error(self, bpti_universe):
        """Test that an empty group list raises a ValueError."""
        with pytest.raises(ValueError, match="group_list cannot be empty"):
            compute_trajectory_average_com_distances(bpti_universe, [], verbose=False)

    def test_no_frames_raises_error(self, bpti_universe):
        """Test that a frame range with no frames raises a ValueError."""
        group_list = [bpti_universe.select_atoms("resid 10")]
        with pytest.raises(ValueError, match="No frames selected for analysis"):
            compute_trajectory_average_com_distances(
                bpti_universe, group_list, start=10, stop=5, verbose=False
            )
