from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np
from MDAnalysis import Universe

from jaxent.forwardmodels.base import Input_Features


@dataclass(frozen=True)
class HBondNetworkFeatures(Input_Features):
    """Features representing the hydrogen bond network for each frame"""

    contact_matrices: List[np.ndarray]  # Shape: (n_frames, n_residues, n_residues)
    residue_ids: List[int]  # Shape: (n_residues,)

    @property
    def features_shape(self) -> Tuple[int, ...]:
        return (len(self.contact_matrices), len(self.residue_ids), len(self.residue_ids))


class HBondNetwork:
    """
    Calculate hydrogen bond networks from molecular structures
    This uses the Baker Hubbard method to calculate hbonds
    """

    def __init__(
        self,
        distance_cutoff: float = 2.5,
        angle_cutoff: float = 120.0,
        residue_ignore: Tuple[int, int] = (0, 0),
    ):
        """
        Args:
            distance_cutoff: Maximum H...A distance in Angstroms (default 2.5Å)
            angle_cutoff: Minimum D-H...A angle in degrees (default 120°)
            residue_ignore: Range of residues to ignore relative to donor
        """
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff
        self.residue_ignore = residue_ignore

    def _identify_hbond_atoms(
        self, universe: Universe
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Identify potential hydrogen bond donors and acceptors"""
        # Donors: N-H groups (backbone and side chain)
        donors = []
        for residue in universe.residues:
            # Backbone N-H
            try:
                n_atom = residue.atoms.select_atoms("name N")[0]
                h_atom = residue.atoms.select_atoms("name H or name HN")[0]
                donors.append((n_atom.index, h_atom.index))
            except IndexError:
                continue

        # Acceptors: O atoms (backbone and side chain)
        acceptors = []
        for residue in universe.residues:
            # Backbone O
            try:
                o_atom = residue.atoms.select_atoms("name O or name OXT")[0]
                acceptors.append((residue.resid, o_atom.index))
            except IndexError:
                continue

            # Side chain O (e.g., Asp, Glu, Ser, Thr)
            try:
                side_o = residue.atoms.select_atoms("name OG* or name OE* or name OD*")
                for o in side_o:
                    acceptors.append((residue.resid, o.index))
            except IndexError:
                continue

        return donors, acceptors

    def calculate_hbonds(self, universe: Universe) -> np.ndarray:
        """Calculate hydrogen bonds for all frames in the trajectory"""
        donors, acceptors = self._identify_hbond_atoms(universe)
        n_residues = len(universe.residues)
        n_frames = universe.trajectory.n_frames
        contact_matrices = np.zeros((n_frames, n_residues, n_residues))

        for ts in universe.trajectory:
            positions = universe.atoms.positions
            contact_matrix = np.zeros((n_residues, n_residues))

            for donor_n_idx, donor_h_idx in donors:
                donor_res = universe.atoms[donor_n_idx].residue

                for acceptor_res_id, acceptor_idx in acceptors:
                    # Skip if residue is in ignore range
                    res_diff = acceptor_res_id - donor_res.resid
                    if self.residue_ignore[0] <= res_diff <= self.residue_ignore[1]:
                        continue

                    # Get positions
                    d_pos = positions[donor_n_idx]
                    h_pos = positions[donor_h_idx]
                    a_pos = positions[acceptor_idx]

                    # Check H...A distance
                    ha_dist = np.linalg.norm(h_pos - a_pos)
                    if ha_dist > self.distance_cutoff:
                        continue

                    # Calculate D-H...A angle
                    hd = d_pos - h_pos
                    ha = a_pos - h_pos

                    # Normalize vectors
                    hd_norm = np.linalg.norm(hd)
                    ha_norm = np.linalg.norm(ha)

                    if hd_norm == 0 or ha_norm == 0:
                        continue

                    hd = hd / hd_norm
                    ha = ha / ha_norm

                    # Calculate angle
                    cos_angle = np.dot(hd, ha)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

                    if angle > self.angle_cutoff:
                        contact_matrix[donor_res.resid - 1, acceptor_res_id - 1] += 1

            contact_matrices[ts.frame] = contact_matrix

        return contact_matrices

    def build_network(self, ensemble: List[Universe]) -> HBondNetworkFeatures:
        """Build hydrogen bond network features from an ensemble"""
        all_contact_matrices = []

        # Calculate contact matrices for each universe in ensemble
        for universe in ensemble:
            contact_matrices = self.calculate_hbonds(universe)
            all_contact_matrices.extend(contact_matrices)

        # Get residue IDs (assuming same across ensemble)
        residue_ids = [r.resid for r in ensemble[0].residues]

        return HBondNetworkFeatures(contact_matrices=all_contact_matrices, residue_ids=residue_ids)

    def get_average_network(self, universe: Universe) -> nx.Graph:
        """
        Create a weighted network representing average H-bond frequencies across all frames
        by averaging over individual frame networks.

        Args:
            universe: MDAnalysis Universe object with trajectory

        Returns:
            NetworkX graph with nodes as residues and edges weighted by H-bond frequency
            normalized by the number of frames
        """
        # Calculate contact matrices for all frames
        contact_matrices = self.calculate_hbonds(universe)
        n_frames = len(contact_matrices)

        # Create the average graph
        G_avg = nx.Graph()
        residue_ids = [r.resid for r in universe.residues]
        G_avg.add_nodes_from(residue_ids)

        # Dictionary to store cumulative weights
        edge_weights = {}

        # Process each frame
        for frame_idx, contact_matrix in enumerate(contact_matrices):
            # Create graph for this frame
            n_residues = len(residue_ids)
            for i in range(n_residues):
                for j in range(i + 1, n_residues):
                    if contact_matrix[i][j] > 0:
                        edge = (residue_ids[i], residue_ids[j])
                        # Add to cumulative weights
                        if edge in edge_weights:
                            edge_weights[edge] += contact_matrix[i][j]
                        else:
                            edge_weights[edge] = contact_matrix[i][j]

        # Add averaged edges to final graph
        for (res1, res2), cumulative_weight in edge_weights.items():
            avg_weight = cumulative_weight / n_frames
            if avg_weight > 0:  # Only add edges with non-zero average weight
                G_avg.add_edge(res1, res2, weight=float(avg_weight))

        return G_avg
