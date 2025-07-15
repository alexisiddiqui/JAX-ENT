import unittest

import MDAnalysis as mda
import numpy as np

from jaxent.src.models.func.contacts import calc_BV_contacts_universe


def create_test_universe(n_frames=2):
    """Creates a simple MDAnalysis Universe for testing."""
    n_residues = 5
    n_atoms_per_residue = 3  # N, H, O
    n_atoms = n_residues * n_atoms_per_residue

    # Create topology attributes
    # Per-atom attributes
    atom_resindices = np.repeat(np.arange(n_residues), n_atoms_per_residue)
    atom_names = ["N", "H", "O"] * n_residues
    atom_types = ["N", "H", "O"] * n_residues

    # Per-residue attributes
    resids = np.arange(1, n_residues + 1)
    # Fix: segids should have one entry per segment, not per residue
    # Since all residues are in segment 0, we need only one segment ID
    segids = ["A"]  # Only one segment

    # Set coordinates for multiple frames
    # Frame 0: Simple grid
    coords_frame0 = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        coords_frame0[i] = [i * 2.0, 0, 0]

    # Frame 1: Atoms move closer
    coords_frame1 = np.copy(coords_frame0)
    # Move atom 10 (residue 4, 'H') closer to atom 0 (residue 1, 'N')
    coords_frame1[10] = [0.5, 0, 0]
    # Move atom 13 (residue 5, 'H') closer to atom 3 (residue 2, 'N')
    coords_frame1[13] = [6.5, 0, 0]

    # Create coordinate array for all frames
    coordinates = np.array([coords_frame0, coords_frame1])

    # Create a universe with the coordinate data
    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=atom_resindices,
        residue_segindex=[0] * n_residues,
        trajectory=False,
    )  # Don't create trajectory yet

    u.add_TopologyAttr("names", atom_names)
    u.add_TopologyAttr("types", atom_types)
    u.add_TopologyAttr("resids", resids)
    u.add_TopologyAttr("segids", segids)

    # Create MemoryReader with coordinate data
    from MDAnalysis.coordinates.memory import MemoryReader

    u.trajectory = MemoryReader(
        coordinates, dimensions=np.array([[50, 50, 50, 90, 90, 90]] * n_frames), dt=1.0
    )

    return u


class TestCalcBVContacts(unittest.TestCase):
    def setUp(self):
        """Set up a test universe before each test."""
        self.universe = create_test_universe()
        # Disable icecream logging for cleaner test output
        try:
            from icecream import ic

            ic.disable()
        except ImportError:
            pass

    def tearDown(self):
        """Enable icecream logging after tests."""
        try:
            from icecream import ic

            ic.enable()
        except ImportError:
            pass

    def test_heavy_contacts_hard_cutoff(self):
        """Test heavy atom contact counting with a hard cutoff."""
        target_atoms = self.universe.select_atoms("resid 1 and name N")
        radius = 2.1  # Should only include atom at 2.0A away

        contacts = calc_BV_contacts_universe(
            universe=self.universe,
            target_atoms=target_atoms,
            contact_selection="heavy",
            radius=radius,
            residue_ignore=(-1, 1),  # Ignore resid 1 and 2
        )

        # Frame 0:
        # Target is atom 0 (res 1, N) at [0,0,0]
        # Ignored residues: 1, 2 (atoms 0-5)
        # Heavy atoms: all N and O
        # Atom 6 (res 3, N) is at [12,0,0] -> too far
        # Atom 8 (res 3, O) is at [16,0,0] -> too far
        # Expected contacts: 0

        # Frame 1:
        # Same as frame 0.
        # Expected contacts: 0

        expected_contacts = [[0.0, 0.0]]  # (targets, frames) - 1 target, 2 frames
        self.assertEqual(np.array(contacts).shape, (1, 2))
        np.testing.assert_allclose(contacts, expected_contacts, atol=1e-5)

    def test_oxygen_contacts_hard_cutoff(self):
        """Test oxygen atom contact counting."""
        target_atoms = self.universe.select_atoms("resid 1 and name N")
        radius = 4.1  # Should include atom 2 (O) at 4.0A

        contacts = calc_BV_contacts_universe(
            universe=self.universe,
            target_atoms=target_atoms,
            contact_selection="oxygen",
            radius=radius,
            residue_ignore=(0, 0),  # Only ignore self
        )

        # Frame 0:
        # Target is atom 0 (res 1, N) at [0,0,0]
        # Ignored residues: 1 (atoms 0-2)
        # Atom 2 (res 1, O) is at [4,0,0] -> ignored
        # Atom 5 (res 2, O) is at [10,0,0] -> too far
        # Expected contacts: 0

        # Frame 1:
        # Same as frame 0
        # Expected contacts: 0

        expected_contacts = [[0.0, 0.0]]  # (targets, frames) - 1 target, 2 frames
        self.assertEqual(np.array(contacts).shape, (1, 2))
        np.testing.assert_allclose(contacts, expected_contacts, atol=1e-5)

    def test_residue_ignore_logic(self):
        """Test that residue_ignore correctly excludes residues."""
        target_atoms = self.universe.select_atoms("resid 3 and name N")  # Atom at [12,0,0]
        radius = 2.1

        # Case 1: Ignore residues 1 to 5 -> should find no contacts
        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(-2, 2)
        )
        # Expected: 0 contacts in both frames
        np.testing.assert_allclose(contacts, [[0.0, 0.0]], atol=1e-5)

        # Case 2: Ignore only self -> should find contacts
        contacts_no_ignore = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(0, 0)
        )
        # Frame 0: Target atom 6 at [12,0,0]. Atom 5 (res 2, O) at [10,0,0] is a contact.
        # Frame 1: Same.
        expected = [[1.0, 1.0]]  # 1 target, 2 frames
        np.testing.assert_allclose(contacts_no_ignore, expected, atol=1e-5)

    def test_switching_function(self):
        """Test contact counting with the switching function."""
        target_atoms = self.universe.select_atoms("resid 1 and name N")
        radius = 2.0

        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(-1, 1), switch=True
        )

        # The switch function is (1 - x) / (1 - x^2) where x = (r/r0)^6
        # Frame 0: No atoms are within radius * 1.2, so contacts should be 0
        # Frame 1: Atom 10 (res 4, H) is at 0.5A. This is a H, so not counted.
        # Let's re-run with a different target
        target_atoms = self.universe.select_atoms("resid 3 and name N")  # Atom 6 at [12,0,0]
        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(0, 0), switch=True
        )

        # Frame 0: Atom 5 (res 2, O) is at [10,0,0]. dist=2.0. r/r0 = 1. x=1. switch is undefined but limit is 0.5
        # Atom 7 (res 3, H) is at [14,0,0]. dist=2.0. H, so ignored.
        # Atom 4 (res 2, H) is at [8,0,0]. dist=4.0. H, ignored.
        # Atom 8 (res 3, O) is at [16,0,0]. dist=4.0. too far.

        # Let's manually calculate for atom 5
        r = 2.0
        r0 = 2.0
        x = (r / r0) ** 6
        # The function is not well-behaved at r=r0. Let's assume it should be 0 at the boundary.
        # The implementation has `distances <= radius * 1`, so it should be 0.
        # Let's test a distance inside the cutoff
        self.universe.atoms[5].position = [11.0, 0, 0]  # dist = 1.0
        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(0, 0), switch=True
        )
        r = 1.0
        x = (r / r0) ** 6
        expected_val = (1 - x) / (1 - x * x)

        # Only frame 0 is changed
        np.testing.assert_allclose(contacts[0][0], expected_val, atol=1e-5)

    def test_no_contacts_found(self):
        """Test case where no contacts should be found."""
        target_atoms = self.universe.select_atoms("resid 1 and name N")
        radius = 0.1  # Very small radius

        contacts = calc_BV_contacts_universe(self.universe, target_atoms, "heavy", radius)
        expected_contacts = [[0.0, 0.0]]  # 1 target, 2 frames
        np.testing.assert_allclose(contacts, expected_contacts, atol=1e-5)

    def test_invalid_selection_raises_error(self):
        """Test that an invalid contact_selection raises a ValueError."""
        with self.assertRaises(ValueError):
            calc_BV_contacts_universe(
                self.universe, self.universe.select_atoms("name N"), "invalid_selection", 5.0
            )

    def test_multiple_frames_behavior(self):
        """Test that contacts are calculated correctly across frames."""
        # Target atom 0 (res 1, N) at [0,0,0]
        # In frame 1, atom 10 (res 4, H) moves to [0.5, 0, 0]
        target_atoms = self.universe.select_atoms("resid 1 and name N")
        radius = 1.0

        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(-2, 2)
        )

        # Frame 0: No heavy atoms within 1.0A of atom 0 (excluding resids -1 to 3)
        # Frame 1: Atom 10 is H, so not a heavy atom.
        # Let's check atom 12 (res 5, N) at [24,0,0]
        # Move atom 12 to [0.8, 0, 0] in frame 1
        self.universe.trajectory[1]
        self.universe.atoms[12].position = [0.8, 0, 0]
        self.universe.trajectory[0]  # Reset to first frame

        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(-2, 2)
        )

        # Frame 0: Expected 0 contacts
        # Frame 1: Expected 1 contact (atom 12)
        expected = [[0.0, 1.0]]
        self.assertEqual(np.array(contacts).shape, (1, 2))
        np.testing.assert_allclose(contacts, expected, atol=1e-5)

    def test_multiple_target_atoms(self):
        """Test calculation with multiple target atoms."""
        # Target 1: atom 0 (res 1, N) at [0,0,0]
        # Target 2: atom 3 (res 2, N) at [6,0,0]
        target_atoms = self.universe.select_atoms("(resid 1 or resid 2) and name N")
        self.assertEqual(len(target_atoms), 2)
        radius = 1.0

        # In frame 1, atom 10 (res 4, H) moves to [0.5, 0, 0] -> ignored (H)
        # In frame 1, atom 13 (res 5, H) moves to [6.5, 0, 0] -> ignored (H)
        # Let's add heavy atom contacts
        self.universe.trajectory[1]
        self.universe.atoms[9].position = [0.8, 0, 0]  # res 4, N
        self.universe.atoms[12].position = [5.5, 0, 0]  # res 5, N
        self.universe.trajectory[0]

        contacts = calc_BV_contacts_universe(
            self.universe, target_atoms, "heavy", radius, residue_ignore=(-1, 1)
        )

        # Expected shape: (2 targets, 2 frames)
        self.assertEqual(np.array(contacts).shape, (2, 2))

        # Target 1 (atom 0):
        # Frame 0: 0 contacts
        # Frame 1: 1 contact (atom 9 at 0.8A)

        # Target 2 (atom 3):
        # Frame 0: 0 contacts
        # Frame 1: 1 contact (atom 12 at 5.5A, dist=0.5)

        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        np.testing.assert_allclose(contacts, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
