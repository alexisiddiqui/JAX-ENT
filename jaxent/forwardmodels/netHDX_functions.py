from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from MDAnalysis import Universe
from tqdm import tqdm

from jaxent.config.base import BaseConfig
from jaxent.forwardmodels.base import ForwardPass, Input_Features, Output_Features


@dataclass(frozen=True)
class NetHDXConfig(BaseConfig):
    """Configuration for netHDX calculations"""

    distance_cutoff: float = 5  # Maximum H...A distance in Angstroms
    angle_cutoff: float = 150.0  # Minimum D-H...A angle in degrees
    residue_ignore: Tuple[int, int] = (0, 0)  # Range of residues to ignore relative to donor
    switch: bool = True  # uses rational 6-12 switch
    angle_switch: bool = True  # to be implemented
    switch_scale: float = 3.0  # same as HDXer
    angle_switch_scale: float = 3.0  # to be tested


@dataclass(frozen=True)
class NetworkMetrics:
    """Per-residue network metrics for a single frame"""

    degrees: dict[int, float]
    clustering_coeffs: dict[int, float]
    betweenness: dict[int, float]
    kcore_numbers: dict[int, float]
    min_path_lengths: dict[int, float]
    mean_path_lengths: dict[int, float]
    max_path_lengths: dict[int, float]


@dataclass(frozen=True)
class NetHDX_input_features(Input_Features):
    """Features representing the hydrogen bond network for each frame"""

    contact_matrices: List[np.ndarray]  # Shape: (n_frames, n_residues, n_residues)
    residue_ids: List[int]  # Shape: (n_residues,)
    network_metrics: Optional[List[NetworkMetrics]] = None  # Shape: (n_frames,)

    @property
    def features_shape(self) -> Tuple[int, ...]:
        return (len(self.contact_matrices), len(self.residue_ids), len(self.residue_ids))


@dataclass(frozen=True)
class NetHDX_output_features(Output_Features):
    """Output features for netHDX model"""

    log_Pf: list  # (1, residues)
    k_ints: Optional[list]

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (1, len(self.log_Pf))


def contact_matrix_to_graph(contact_matrix: np.ndarray, residue_ids: List[int]) -> nx.Graph:
    """Convert a contact matrix to a NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(residue_ids)

    n_residues = len(residue_ids)
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if contact_matrix[i][j] > 0:
                G.add_edge(residue_ids[i], residue_ids[j], weight=float(contact_matrix[i][j]))

    return G


def calculate_switch(x: float, x_cutoff: float, scale: float = 1.0) -> float:
    """
    Calculate smooth rational switch function value that transitions from 1 to 0.
    Args:
        x: Input value (distance or angle)
        x_cutoff: Value where function begins decreasing
        scale: Scaling factor for switch steepness
    Returns:
        Switch function value: 1.0 when x <= x_cutoff, smoothly decreases to 0 when x > x_cutoff
    """
    return rational_6_12(x=x, k=scale, d0=x_cutoff)


def rational_6_12(x, k, d0=2.4, n=6, m=12):
    if x < d0:
        return 1
    else:
        num = 1 - ((x - d0) / k) ** n
        denom = 1 - ((x - d0) / k) ** m
        return num / denom


def rational_6_12_angle(x, k, d0=120, n=6, m=12):
    """
    Angle-compatible rational switch function.
    Args:
        x: Input angle in degrees
        k: Switch width parameter in degrees
        d0: Cutoff angle in degrees (default: 120°)
        n: Power for numerator (default: 6)
        m: Power for denominator (default: 12)
    Returns:
        Switch function value between 0 and 1, returns 1 when angle > cutoff
    """
    # Convert angle difference to smallest equivalent angle in [-180, 180]
    dx = x - d0
    dx = ((dx + 180) % 360) - 180

    if dx < 0:  # Function is 0 when angle is less than cutoff
        num = 1 - (-dx / k) ** n
        denom = 1 - (-dx / k) ** m
        return num / denom
    else:
        return 1


def calculate_angle_switch(angle: float, cutoff: float, scale: float = 1.0) -> float:
    """
    Calculates a smoothed switch function for angles that handles periodicity.
    Uses scale parameter to control the steepness of the transition.

    Args:
        angle: D-H...A angle in degrees
        cutoff: Minimum angle cutoff in degrees
        scale: Scaling factor for switch steepness (higher values = sharper transition)

    Returns:
        Switch value between 0 and 1
    """
    return rational_6_12_angle(x=angle, k=scale * 30, d0=cutoff)


def calculate_hbond_weight(
    ha_dist: float,
    angle: float,
    distance_cutoff: float,
    angle_cutoff: float,
    switch: bool = True,
    switch_scale: float = 2.0,
) -> float:
    """Calculate H-bond weight with separate distance and angle components"""
    if not switch:
        return 1.0 if ha_dist <= distance_cutoff and angle >= angle_cutoff else 0.0

    # Calculate components with different scales
    dist_switch = calculate_switch(ha_dist, distance_cutoff, 0.5)
    angle_switch = calculate_angle_switch(angle, angle_cutoff, 0.5)

    # Geometric mean for smoother combination
    return np.sqrt(dist_switch * angle_switch)


def identify_hbond_donors(universe: Universe) -> List[Tuple[int, int]]:
    """
    Identify potential hydrogen bond donors (N-H groups) in the universe.

    Args:
        universe: MDAnalysis Universe containing structure

    Returns:
        List of (N_atom_index, H_atom_index) tuples for potential donors
    """
    donors = []
    for residue in universe.residues:
        try:
            n_atom = residue.atoms.select_atoms("name N")[0]
            h_atom = residue.atoms.select_atoms("name H or name HN")[0]
            donors.append((n_atom.index, h_atom.index))
        except IndexError:
            continue
    return donors


def identify_hbond_acceptors(universe: Universe) -> List[Tuple[int, int]]:
    """
    Identify potential hydrogen bond acceptors (O atoms) in the universe.

    Args:
        universe: MDAnalysis Universe containing structure

    Returns:
        List of (residue_id, O_atom_index) tuples for potential acceptors
    """
    acceptors = []
    for residue in universe.residues:
        # Backbone oxygen
        try:
            o_atom = residue.atoms.select_atoms("name O or name OXT")[0]
            acceptors.append((residue.resid, o_atom.index))
        except IndexError:
            continue

        # Side chain oxygens
        try:
            side_o = residue.atoms.select_atoms("name OG* or name OE* or name OD*")
            for o in side_o:
                acceptors.append((residue.resid, o.index))
        except IndexError:
            continue
    return acceptors


def calculate_hbond_angle(d_pos: np.ndarray, h_pos: np.ndarray, a_pos: np.ndarray) -> float:
    """
    Calculate the D-H...A angle for a potential hydrogen bond.

    Args:
        d_pos: Position of donor atom
        h_pos: Position of hydrogen atom
        a_pos: Position of acceptor atom

    Returns:
        Angle in degrees
    """
    hd = d_pos - h_pos
    ha = a_pos - h_pos

    # Normalize vectors
    hd_norm = np.linalg.norm(hd)
    ha_norm = np.linalg.norm(ha)

    if hd_norm == 0 or ha_norm == 0:
        return 0.0

    hd = hd / hd_norm
    ha = ha / ha_norm

    # Calculate angle
    cos_angle = np.dot(hd, ha)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi


def calculate_frame_hbonds(
    universe: Universe,
    donors: List[Tuple[int, int]],
    acceptors: List[Tuple[int, int]],
    config: NetHDXConfig,
) -> np.ndarray:
    """
    Calculate hydrogen bonds for a single frame with switch function.

    Args:
        universe: MDAnalysis Universe containing structure
        donors: List of (N_atom_index, H_atom_index) tuples
        acceptors: List of (residue_id, O_atom_index) tuples
        config: Configuration parameters

    Returns:
        Contact matrix for the frame with H-bond weights
    """
    positions = universe.atoms.positions
    n_residues = len(universe.residues)
    contact_matrix = np.zeros((n_residues, n_residues))

    for donor_n_idx, donor_h_idx in donors:
        donor_res = universe.atoms[donor_n_idx].residue

        for acceptor_res_id, acceptor_idx in acceptors:
            # Skip if residue is in ignore range
            res_diff = acceptor_res_id - donor_res.resid
            if config.residue_ignore[0] <= res_diff <= config.residue_ignore[1]:
                continue

            # Get positions
            d_pos = positions[donor_n_idx]
            h_pos = positions[donor_h_idx]
            a_pos = positions[acceptor_idx]

            # Calculate H...A distance
            ha_dist = np.linalg.norm(h_pos - a_pos)

            # Early exit if beyond cutoff to save computation
            if ha_dist > config.distance_cutoff and config.switch:
                continue

            # Calculate D-H...A angle
            angle = calculate_hbond_angle(d_pos, h_pos, a_pos)

            # Calculate H-bond weight using switch function
            weight = calculate_hbond_weight(
                ha_dist=ha_dist,
                angle=angle,
                distance_cutoff=config.distance_cutoff,
                angle_cutoff=config.angle_cutoff,
                switch=config.switch,
                switch_scale=config.switch_scale,
            )

            if weight > 0:
                contact_matrix[donor_res.resid - 1, acceptor_res_id - 1] = weight

    return contact_matrix


def calculate_trajectory_hbonds(
    universe: Universe,
    config: NetHDXConfig,
) -> np.ndarray:
    """
    Calculate hydrogen bonds for all frames in a trajectory.

    Args:
        universe: MDAnalysis Universe with trajectory
        config: Configuration parameters

    Returns:
        Array of contact matrices for each frame
    """
    donors = identify_hbond_donors(universe)
    acceptors = identify_hbond_acceptors(universe)

    n_residues = len(universe.residues)
    n_frames = universe.trajectory.n_frames
    contact_matrices = np.zeros((n_frames, n_residues, n_residues))

    for ts in universe.trajectory:
        contact_matrices[ts.frame] = calculate_frame_hbonds(universe, donors, acceptors, config)

    return contact_matrices


def compute_path_lengths(G: nx.Graph, node: int) -> Tuple[float, float, float]:
    """Compute min, mean, and max path lengths for a node"""
    path_lengths = []
    for target in G.nodes():
        if target != node:
            try:
                length = nx.shortest_path_length(G, node, target, weight="weight")
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                continue

    if not path_lengths:
        return float("inf"), float("inf"), float("inf")

    return min(path_lengths), np.mean(path_lengths), max(path_lengths)


def compute_weighted_clustering(G: nx.Graph) -> dict:
    """Custom weighted clustering coefficient implementation"""
    clustering = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            clustering[node] = 0.0
            continue

        # Calculate weighted triangles
        triangle_weight = 0.0
        max_triangle_weight = 0.0
        for i, n1 in enumerate(neighbors):
            w1 = G[node][n1]["weight"]
            for n2 in neighbors[i + 1 :]:
                w2 = G[node][n2]["weight"]
                if G.has_edge(n1, n2):
                    w3 = G[n1][n2]["weight"]
                    triangle_weight += (w1 * w2 * w3) ** (1 / 3)
                max_triangle_weight += (w1 * w2) ** (1 / 2)

        clustering[node] = triangle_weight / max_triangle_weight if max_triangle_weight > 0 else 0.0
    return clustering


def compute_local_clustering(G: nx.Graph, node: int) -> float:
    """Compute weighted clustering coefficient for a single node"""
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0.0

    possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
    if possible_triangles == 0:
        return 0.0

    triangles = 0
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i + 1 :]:
            if G.has_edge(n1, n2):
                w1 = G[node][n1]["weight"]
                w2 = G[node][n2]["weight"]
                w3 = G[n1][n2]["weight"]
                triangles += (w1 * w2 * w3) ** (1 / 3)

    return triangles / possible_triangles


# def compute_frame_metrics(contact_matrix: np.ndarray, residue_ids: List[int]) -> NetworkMetrics:
#     """Compute network metrics with revised normalization"""
#     G = contact_matrix_to_graph(contact_matrix, residue_ids)

#     # Normalize weights to [0,1]
#     max_weight = max((w for _, _, w in G.edges(data="weight")), default=1.0)
#     if max_weight > 0:
#         for u, v, w in G.edges(data="weight"):
#             G[u][v]["weight"] = w / max_weight

#     # Weighted degree
#     degrees = {node: sum(G[node][neighbor]["weight"] for neighbor in G[node]) for node in G.nodes()}

#     # Local clustering coefficients
#     clustering = {node: compute_local_clustering(G, node) for node in G.nodes()}

#     # Betweenness with distance transformation
#     dist_weights = {(u, v): 1 / (w + 1e-6) for u, v, w in G.edges(data="weight")}
#     nx.set_edge_attributes(G, dist_weights, "distance")
#     betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

#     # K-core (topological)
#     k_core = nx.core_number(G)

#     # Path lengths using distance weights
#     path_metrics = {}
#     for node in G.nodes():
#         try:
#             distances = [
#                 d
#                 for d in dict(
#                     nx.single_source_dijkstra_path_length(G, node, weight="distance")
#                 ).values()
#                 if d > 0
#             ]
#             if distances:
#                 path_metrics[node] = (min(distances), np.mean(distances), max(distances))
#             else:
#                 path_metrics[node] = (float("inf"), float("inf"), float("inf"))
#         except nx.NetworkXNoPath:
#             path_metrics[node] = (float("inf"), float("inf"), float("inf"))

#     min_paths = {node: metrics[0] for node, metrics in path_metrics.items()}
#     mean_paths = {node: metrics[1] for node, metrics in path_metrics.items()}
#     max_paths = {node: metrics[2] for node, metrics in path_metrics.items()}

#     return NetworkMetrics(
#         degrees=degrees,
#         clustering_coeffs=clustering,
#         betweenness=betweenness,
#         kcore_numbers=k_core,
#         min_path_lengths=min_paths,
#         mean_path_lengths=mean_paths,
#         max_path_lengths=max_paths,
#     )


def compute_frame_metrics(contact_matrix: np.ndarray, residue_ids: List[int]) -> NetworkMetrics:
    """Compute network metrics using standard NetworkX implementations"""
    G = contact_matrix_to_graph(contact_matrix, residue_ids)

    max_weight = max((w for _, _, w in G.edges(data="weight")), default=1.0)
    if max_weight > 0:
        for u, v, w in G.edges(data="weight"):
            G[u][v]["weight"] = w / max_weight

    degrees = {node: sum(G[node][neighbor]["weight"] for neighbor in G[node]) for node in G.nodes()}

    # Standard NetworkX implementations
    clustering = nx.clustering(G, weight="weight")
    dist_weights = {(u, v): 1 / (w + 1e-6) for u, v, w in G.edges(data="weight")}
    nx.set_edge_attributes(G, dist_weights, "distance")
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    k_core = nx.core_number(G)

    # Path lengths
    path_metrics = {}
    for node in G.nodes():
        try:
            distances = [
                d
                for d in dict(
                    nx.single_source_dijkstra_path_length(G, node, weight="distance")
                ).values()
                if d > 0
            ]
            if distances:
                path_metrics[node] = (min(distances), np.mean(distances), max(distances))
            else:
                path_metrics[node] = (float("inf"), float("inf"), float("inf"))
        except nx.NetworkXNoPath:
            path_metrics[node] = (float("inf"), float("inf"), float("inf"))

    min_paths = {node: metrics[0] for node, metrics in path_metrics.items()}
    mean_paths = {node: metrics[1] for node, metrics in path_metrics.items()}
    max_paths = {node: metrics[2] for node, metrics in path_metrics.items()}

    return NetworkMetrics(
        degrees=degrees,
        clustering_coeffs=clustering,
        betweenness=betweenness,
        kcore_numbers=k_core,
        min_path_lengths=min_paths,
        mean_path_lengths=mean_paths,
        max_path_lengths=max_paths,
    )


def compute_trajectory_metrics(
    contact_matrices: List[np.ndarray], residue_ids: List[int]
) -> List[NetworkMetrics]:
    """Compute network metrics for all frames in a trajectory"""
    compute_metrics = partial(compute_frame_metrics, residue_ids=residue_ids)
    return list(map(compute_metrics, contact_matrices))


def parallel_compute_trajectory_metrics(
    contact_matrices: List[np.ndarray], residue_ids: List[int], n_processes: int = None
) -> List[NetworkMetrics]:
    """Parallel version of compute_trajectory_metrics"""
    with Pool(processes=n_processes) as pool:
        compute_metrics = partial(compute_frame_metrics, residue_ids=residue_ids)
        return list(
            tqdm(
                pool.imap(compute_metrics, contact_matrices),
                total=len(contact_matrices),
                desc="Computing metrics",
            )
        )


def calculate_frame_wrapper(args):
    universe, frame_idx, donors, acceptors, config = args
    universe.trajectory[frame_idx]
    return calculate_frame_hbonds(universe, donors, acceptors, config)


def parallel_calculate_trajectory_hbonds(
    universe: Universe, config: NetHDXConfig, n_processes: int = 10
) -> np.ndarray:
    """Parallel version of calculate_trajectory_hbonds"""
    donors = identify_hbond_donors(universe)
    acceptors = identify_hbond_acceptors(universe)
    n_residues = len(universe.residues)
    n_frames = universe.trajectory.n_frames

    # Create frame data tuples
    frame_data = [(universe, i, donors, acceptors, config) for i in range(n_frames)]

    # Process frames in parallel with progress bar
    with Pool(processes=n_processes) as pool:
        contact_matrices = list(
            tqdm(
                pool.imap(calculate_frame_wrapper, frame_data),
                total=n_frames,
                desc="Processing frames",
            )
        )

    return np.array(contact_matrices)


# Updated version of build_hbond_network to use parallel processing
def parallel_build_hbond_network(
    ensemble: List[Universe], config: Optional[NetHDXConfig] = None, n_processes: int = 10
) -> NetHDX_input_features:
    """Parallel version of build_hbond_network"""
    if config is None:
        config = NetHDXConfig()

    all_contact_matrices = []

    # Calculate contact matrices for each universe in parallel
    for universe in ensemble:
        contact_matrices = parallel_calculate_trajectory_hbonds(
            universe, config, n_processes=n_processes
        )
        all_contact_matrices.extend(contact_matrices)

    residue_ids = [r.resid for r in ensemble[0].residues]

    # Compute network metrics in parallel
    network_metrics = parallel_compute_trajectory_metrics(
        all_contact_matrices, residue_ids, n_processes=n_processes
    )

    return NetHDX_input_features(
        contact_matrices=all_contact_matrices,
        residue_ids=residue_ids,
        network_metrics=network_metrics,
    )


def build_hbond_network(
    ensemble: List[Universe], config: Optional[NetHDXConfig] = None, parallel: bool = True
) -> NetHDX_input_features:
    """
    Build hydrogen bond network features from an ensemble of structures.
    Now includes network metrics computation.
    """
    if parallel:
        return parallel_build_hbond_network(ensemble=ensemble, config=config)

    if config is None:
        config = NetHDXConfig()

    all_contact_matrices = []

    # Calculate contact matrices for each universe in ensemble
    for universe in ensemble:
        contact_matrices = calculate_trajectory_hbonds(universe, config)
        all_contact_matrices.extend(contact_matrices)

    # Get residue IDs (assuming same across ensemble)
    residue_ids = [r.resid for r in ensemble[0].residues]

    # Compute network metrics
    network_metrics = compute_trajectory_metrics(all_contact_matrices, residue_ids)

    return NetHDX_input_features(
        contact_matrices=all_contact_matrices,
        residue_ids=residue_ids,
        network_metrics=network_metrics,
    )


def create_average_network(
    universe: Universe,
    config: Optional[NetHDXConfig] = None,
) -> nx.Graph:
    """
    Create a weighted network representing average H-bond frequencies across all frames.

    Args:
        universe: MDAnalysis Universe with trajectory
        config: Configuration parameters (optional)

    Returns:
        NetworkX graph with nodes as residues and edges weighted by H-bond frequency
    """
    if config is None:
        config = NetHDXConfig()

    # Calculate contact matrices for all frames
    contact_matrices = calculate_trajectory_hbonds(universe, config)
    n_frames = len(contact_matrices)

    # Create the average graph
    G_avg = nx.Graph()
    residue_ids = [r.resid for r in universe.residues]
    G_avg.add_nodes_from(residue_ids)

    # dictionary to store cumulative weights
    edge_weights = {}

    # Process each frame
    for contact_matrix in contact_matrices:
        n_residues = len(residue_ids)
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                if contact_matrix[i][j] > 0:
                    edge = (residue_ids[i], residue_ids[j])
                    # Add to cumulative weights
                    edge_weights[edge] = edge_weights.get(edge, 0) + contact_matrix[i][j]

    # Add averaged edges to final graph
    for (res1, res2), cumulative_weight in edge_weights.items():
        avg_weight = cumulative_weight / n_frames
        if avg_weight > 0:  # Only add edges with non-zero average weight
            G_avg.add_edge(res1, res2, weight=float(avg_weight))

    return G_avg


class NetHDX_ForwardPass(ForwardPass):
    def __call__(
        self, features: NetHDX_input_features, parameters: NetHDXConfig
    ) -> NetHDX_output_features:
        # Process contact matrices to calculate protection factors
        # This is a placeholder - implement actual netHDX calculation here
        avg_contacts = np.mean(features.contact_matrices, axis=0)
        log_pf = np.log10(np.sum(avg_contacts, axis=1)).tolist()

        return NetHDX_output_features(log_Pf=log_pf, k_ints=None)
