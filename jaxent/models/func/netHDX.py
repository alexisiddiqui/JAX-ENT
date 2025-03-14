from functools import partial
from multiprocessing import Pool
from typing import Optional, Sequence, cast

import MDAnalysis as mda
import networkx as nx
import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis  # type: ignore
from MDAnalysis.core.groups import ResidueGroup  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from jaxent.models.config import NetHDXConfig
from jaxent.models.HDX.netHDX.features import NetHDX_input_features, NetworkMetrics


def identify_hbond_donors(universe: mda.Universe) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Identify potential hydrogen bond donors (N-H groups) in the universe.

    Args:
        universe: MDAnalysis Universe containing structure

    Returns:
        list of (N_atom_index, H_atom_index) tuples for potential donors
    """
    donors = []
    residue_ids = []
    for residue in cast(ResidueGroup, universe.residues):
        try:
            n_atom = residue.atoms.select_atoms("name CA")[0]
            h_atom = residue.atoms.select_atoms("name H or name HN")[0]

            # throw an error if both atoms are not found
            assert n_atom and h_atom

            donors.append((n_atom.index, h_atom.index))
            residue_ids.append(residue.resid)
        except IndexError:
            continue
    return (donors, residue_ids)


def identify_hbond_acceptors(universe: mda.Universe) -> list[tuple[int, int]]:
    """
    Identify potential hydrogen bond acceptors (O atoms) in the universe.

    Args:
        universe: MDAnalysis Universe containing structure

    Returns:
        list of (residue_id, O_atom_index) tuples for potential acceptors
    """
    acceptors = []
    for residue in cast(ResidueGroup, universe.residues):
        # oxygen
        try:
            o_atom = residue.atoms.select_atoms("element O or name O or name OXT")[0]
            acceptors.append((residue.resid, o_atom.index))
        except IndexError:
            continue

        # # Side chain oxygens
        # try:
        #     side_o = residue.atoms.select_atoms("element O or name OG* or name OE* or name OD*")
        #     for o in side_o:
        #         acceptors.append((residue.resid, o.index))
        # except IndexError:
        #     continue
    return acceptors


def analyze_hbonds_for_shell(
    universe: mda.Universe, donor_selection: str, distance: float, angle: float
) -> HydrogenBondAnalysis:
    """
    Analyze hydrogen bonds for a specific shell defined by distance and angle cutoffs.

    Args:
        universe: MDAnalysis Universe containing structure
        donor_selection: MDAnalysis selection string for the donor
        distance: Distance cutoff for this shell
        angle: Angle cutoff for this shell

    Returns:
        HydrogenBondAnalysis results for this shell
    """
    hbonds = HydrogenBondAnalysis(
        universe=universe,
        donors_sel=donor_selection,
        hydrogens_sel="element H",
        acceptors_sel="name O",
        d_a_cutoff=distance,  # donor-acceptor cutoff
        d_h_a_angle_cutoff=int(angle),  # donor-hydrogen-acceptor angle cutoff
        update_selections=False,  # Update selections at each frame
    )
    hbonds.run()
    return hbonds


def create_contact_matrix_for_frame(
    hbonds: HydrogenBondAnalysis, frame: int, n_residues: int, residue_mapping: dict
) -> np.ndarray:
    """
    Create a contact matrix for a specific frame from hydrogen bond analysis results.
    Only includes residues specified in the residue_mapping dictionary.

    Args:
        hbonds: HydrogenBondAnalysis results
        frame: Frame number to analyze
        n_residues: Total number of residues in the filtered set
        residue_mapping: Mapping from residue IDs to matrix indices

    Returns:
        Contact matrix for the frame with only the mapped residues
    """
    contact_matrix = np.zeros((n_residues, n_residues))

    # Get bonds for this frame
    # Results format is [frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle]
    frame_bonds = hbonds.results.hbonds[hbonds.results.hbonds[:, 0] == frame]

    for bond in frame_bonds:
        # Get donor and acceptor residue IDs from the atom indices
        donor_atom = hbonds.u.atoms[int(bond[1])]
        acceptor_atom = hbonds.u.atoms[int(bond[3])]

        donor_resid = donor_atom.residue.resid
        acceptor_resid = acceptor_atom.residue.resid

        # Skip if residues not in mapping
        if donor_resid not in residue_mapping or acceptor_resid not in residue_mapping:
            continue

        donor_idx = residue_mapping[donor_resid]
        acceptor_idx = residue_mapping[acceptor_resid]

        # Increment the contact count
        contact_matrix[donor_idx, acceptor_idx] += 1

    return contact_matrix


# def _process_residue(args):
#     residue, universe, config, n_frames, n_residues, residue_mapping = args
#     matrix = np.zeros((n_frames, n_residues, n_residues))
#     try:
#         donor_sel = f"resid {residue.resid} and name N"
#         for dist, angle in zip(config.distance_cutoff, config.angle_cutoff):
#             hbonds = analyze_hbonds_for_shell(
#                 universe=universe, donor_selection=donor_sel, distance=dist, angle=angle
#             )
#             unique_frames = np.unique(hbonds.results.hbonds[:, 0])
#             for frame in unique_frames:
#                 frame_contacts = create_contact_matrix_for_frame(
#                     hbonds=hbonds,
#                     frame=int(frame),
#                     n_residues=n_residues,
#                     residue_mapping=residue_mapping,
#                 )
#                 matrix[int(frame)] += frame_contacts
#     except Exception as e:
#         print(f"Warning: processing residue {residue.resid}: {str(e)}")
#     return matrix


# Then, update calculate_trajectory_hbonds as follows:


# def calculate_trajectory_hbonds(
#     universe: mda.Universe, config: NetHDXConfig, n_workers=10
# ) -> np.ndarray:
#     """
#     Calculate hydrogen bonds for all frames across all shells defined in config,
#     parallelizing the computation across residues.
#     """

#     n_residues = len(universe.residues)
#     n_frames = len(universe.trajectory)
#     residue_mapping = {res.resid: i for i, res in enumerate(universe.residues)}

#     # Prepare arguments for each residue
#     args = [
#         (residue, universe, config, n_frames, n_residues, residue_mapping)
#         for residue in universe.residues
#     ]

#     with multiprocessing.Pool(processes=n_workers) as pool:
#         results = list(
#             tqdm(
#                 pool.imap(_process_residue, args),
#                 total=len(args),
#                 desc="Analyzing residues",
#             )
#         )

#     # Sum contributions of all residues.
#     contact_matrices = np.zeros((n_frames, n_residues, n_residues))
#     for mat in results:
#         contact_matrices += mat


# #     return contact_matrices
# def calculate_trajectory_hbonds(universe: mda.Universe, config: NetHDXConfig) -> np.ndarray:
#     """
#     Calculate hydrogen bonds for all frames across all shells defined in config.
#     """
#     residues = cast(ResidueGroup, universe.residues)
#     n_residues = len(residues)
#     n_frames = len(universe.trajectory)
#     residue_list = [r for r in residues]
#     residue_mapping = {res.resid: i for i, res in enumerate(residue_list)}
#     contact_matrices = np.zeros((n_frames, n_residues, n_residues))

#     for residue in tqdm(residue_list, desc="Analyzing residues"):
#         try:
#             donor_sel = f"resid {residue.resid} and name N"

#             for dist, angle in zip(config.distance_cutoff, config.angle_cutoff):
#                 hbonds = analyze_hbonds_for_shell(
#                     universe=universe, donor_selection=donor_sel, distance=dist, angle=angle
#                 )

#                 # Now correctly handle each frame
#                 unique_frames = np.unique(hbonds.results.hbonds[:, 0])
#                 for frame in unique_frames:
#                     frame_contacts = create_contact_matrix_for_frame(
#                         hbonds=hbonds,
#                         frame=frame,
#                         n_residues=n_residues,
#                         residue_mapping=residue_mapping,
#                     )
#                     contact_matrices[int(frame)] += frame_contacts

#         except Exception as e:
#             print(f"Warning: processing residue {residue.resid}: {str(e)}")
#             continue

#     return contact_matrices


def build_hbond_network(
    ensemble: list[mda.Universe],
    config: Optional[NetHDXConfig] = None,
    common_residue_ids: Optional[list[int]] = None,
) -> NetHDX_input_features:
    """
    Build hydrogen bond network features from an ensemble of structures using
    MDAnalysis HydrogenBondAnalysis.

    Args:
        ensemble: List of MDAnalysis Universe objects
        config: Configuration parameters for hydrogen bond analysis
        common_residue_ids: Optional list of residue IDs to include in the network
                           (if None, all residues are included)

    Returns:
        NetHDX_input_features object containing contact matrices and network metrics
    """
    if config is None:
        config = NetHDXConfig()

    all_contact_matrices = []

    # If common_residue_ids is provided, create a mapping from residue IDs to matrix indices
    residue_mapping = None
    common_indices = [None]
    if common_residue_ids:
        # Get mapping of all residues in first universe
        all_residues = [r.resid for r in cast(ResidueGroup, ensemble[0].residues)]

        # Create a mapping from original indices to filtered indices

        common_indices = [
            all_residues.index(res_id) for res_id in common_residue_ids if res_id in all_residues
        ]
        residue_mapping = {i: idx for idx, i in enumerate(common_indices)}

    # Calculate contact matrices for each universe in ensemble
    for universe in ensemble:
        # Calculate full contact matrices first
        contact_matrices = calculate_trajectory_hbonds(
            universe, config, common_residue_ids=common_residue_ids
        )

        # If common_residue_ids is provided, filter the contact matrices
        if common_residue_ids and residue_mapping:
            filtered_matrices = []
            for matrix in contact_matrices:
                # Create a smaller matrix with only the common residues
                n_common = len(common_residue_ids)
                filtered_matrix = np.zeros((n_common, n_common))

                # Map values from original matrix to filtered matrix
                for i, orig_i in enumerate(common_indices):
                    for j, orig_j in enumerate(common_indices):
                        filtered_matrix[i, j] = matrix[orig_i, orig_j]

                filtered_matrices.append(filtered_matrix)

            all_contact_matrices.extend(filtered_matrices)
        else:
            all_contact_matrices.extend(contact_matrices)

    # Get residue IDs (either common residues or all residues)
    residue_ids = (
        common_residue_ids
        if common_residue_ids
        else [r.resid for r in cast(ResidueGroup, ensemble[0].residues)]
    )

    # Compute network metrics
    try:
        network_metrics = parallel_compute_trajectory_metrics(
            all_contact_matrices, residue_ids, n_processes=None
        )
    except Exception as e:
        print(f"Warning: failed to compute network metrics: {str(e)}")
        network_metrics = compute_trajectory_metrics(all_contact_matrices, residue_ids)

    return NetHDX_input_features(
        contact_matrices=all_contact_matrices,
        residue_ids=residue_ids,
        network_metrics=network_metrics,
    )


def calculate_trajectory_hbonds(
    universe: mda.Universe, config: NetHDXConfig, common_residue_ids: Optional[list[int]] = None
) -> np.ndarray:
    """
    Calculate hydrogen bonds for all frames across all shells defined in config.

    Args:
        universe: MDAnalysis Universe containing structure
        config: Configuration parameters for hydrogen bond analysis
        common_residue_ids: Optional list of residue IDs to include in the analysis

    Returns:
        Array of contact matrices for all frames
    """
    residues = cast(ResidueGroup, universe.residues)

    # Filter residues if common_residue_ids is provided
    if common_residue_ids:
        residue_list = [r for r in residues if r.resid in common_residue_ids]
    else:
        residue_list = [r for r in residues]

    n_residues = len(residue_list)
    n_frames = len(universe.trajectory)

    # Create mapping from residue IDs to matrix indices
    residue_mapping = {res.resid: i for i, res in enumerate(residue_list)}

    contact_matrices = np.zeros((n_frames, n_residues, n_residues))

    for residue in tqdm(residue_list, desc="Analyzing residues"):
        try:
            donor_sel = f"resid {residue.resid} and name N"

            for dist, angle in zip(config.distance_cutoff, config.angle_cutoff):
                hbonds = analyze_hbonds_for_shell(
                    universe=universe, donor_selection=donor_sel, distance=dist, angle=angle
                )

                # Process each frame
                unique_frames = np.unique(hbonds.results.hbonds[:, 0])
                for frame in unique_frames:
                    frame_contacts = create_contact_matrix_for_frame(
                        hbonds=hbonds,
                        frame=frame,
                        n_residues=n_residues,
                        residue_mapping=residue_mapping,
                    )
                    contact_matrices[int(frame)] += frame_contacts

        except Exception as e:
            print(f"Warning: processing residue {residue.resid}: {str(e)}")
            continue

    return contact_matrices


def compute_path_lengths(G: nx.Graph, node: int) -> tuple[float, float, float]:
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

    return min(path_lengths), float(np.mean(path_lengths)), max(path_lengths)


# def compute_weighted_clustering(G: nx.Graph) -> dict:
#     """Custom weighted clustering coefficient implementation"""
#     clustering = {}
#     for node in G.nodes():
#         neighbors = list(G.neighbors(node))
#         if len(neighbors) < 2:
#             clustering[node] = 0.0
#             continue

#         # Calculate weighted triangles
#         triangle_weight = 0.0
#         max_triangle_weight = 0.0
#         for i, n1 in enumerate(neighbors):
#             w1 = G[node][n1]["weight"]
#             for n2 in neighbors[i + 1 :]:
#                 w2 = G[node][n2]["weight"]
#                 if G.has_edge(n1, n2):
#                     w3 = G[n1][n2]["weight"]
#                     triangle_weight += (w1 * w2 * w3) ** (1 / 3)
#                 max_triangle_weight += (w1 * w2) ** (1 / 2)

#         clustering[node] = triangle_weight / max_triangle_weight if max_triangle_weight > 0 else 0.0
#     return clustering


# def compute_local_clustering(G: nx.Graph, node: int) -> float:
#     """Compute weighted clustering coefficient for a single node"""
#     neighbors = list(G.neighbors(node))
#     if len(neighbors) < 2:
#         return 0.0

#     possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
#     if possible_triangles == 0:
#         return 0.0

#     triangles = 0
#     for i, n1 in enumerate(neighbors):
#         for n2 in neighbors[i + 1 :]:
#             if G.has_edge(n1, n2):
#                 w1 = G[node][n1]["weight"]
#                 w2 = G[node][n2]["weight"]
#                 w3 = G[n1][n2]["weight"]
#                 triangles += (w1 * w2 * w3) ** (1 / 3)

#     return triangles / possible_triangles


def compute_graph_metrics(G: nx.Graph) -> NetworkMetrics:
    """Compute network metrics from a given NetworkX graph."""
    max_weight = max((w for _, _, w in G.edges(data="weight")), default=1.0)  # type: ignore
    if max_weight > 0:
        for u, v, w in G.edges(data="weight"):  # type: ignore
            G[u][v]["weight"] = w / max_weight

    degrees = {node: sum(G[node][neighbor]["weight"] for neighbor in G[node]) for node in G.nodes()}

    # Standard NetworkX implementations
    clustering = nx.clustering(G, weight="weight")

    assert isinstance(clustering, dict), (
        "Clustering output is not a dictionary - did something go wrong?"
    )
    dist_weights = {(u, v): 1 / (w + 1e-6) for u, v, w in G.edges(data="weight")}  # type: ignore
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


def compute_frame_metrics(contact_matrix: np.ndarray, residue_ids: Sequence[int]) -> NetworkMetrics:
    """Compute network metrics using a contact matrix by creating a graph first."""
    G = contact_matrix_to_graph(contact_matrix, residue_ids)
    return compute_graph_metrics(G)


def compute_trajectory_metrics(
    contact_matrices: list[np.ndarray], residue_ids: Sequence[int]
) -> list[NetworkMetrics]:
    """Compute network metrics for all frames in a trajectory"""
    compute_metrics = partial(compute_frame_metrics, residue_ids=residue_ids)
    return list(map(compute_metrics, contact_matrices))


def parallel_compute_trajectory_metrics(
    contact_matrices: list[np.ndarray], residue_ids: Sequence[int], n_processes: int | None = 10
) -> list[NetworkMetrics]:
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


def create_average_network(
    universe: mda.Universe,
    config: Optional[NetHDXConfig] = None,
    weight_threshold: float = 0,  # Added weight threshold parameter
) -> nx.Graph:
    """
    Create a weighted network representing average H-bond frequencies across all frames.

    Args:
        universe: MDAnalysis Universe with trajectory
        config: Configuration parameters (optional)
        weight_threshold: minimum average weight for including an edge (default: 0.1)

    Returns:
        NetworkX graph with nodes as residues and edges weighted by H-bond frequency above threshold
    """
    if config is None:
        config = NetHDXConfig()

    # Calculate contact matrices for all frames using the new HydrogenBondAnalysis approach
    contact_matrices = calculate_trajectory_hbonds(universe, config)
    n_frames = len(contact_matrices)

    # Calculate average contact matrix
    # Since our contact matrices now contain integer counts, we'll normalize by frames
    avg_contact_matrix = np.mean(contact_matrices, axis=0)

    # Normalize the average matrix by the number of shells to get frequencies
    n_shells = len(config.distance_cutoff)  # Number of distance/angle shell combinations
    # avg_contact_matrix = avg_contact_matrix / n_shells

    # Create graph using the weight threshold

    residue_ids = [r.resid for r in cast(ResidueGroup, universe.residues)]
    G_avg = contact_matrix_to_graph(
        avg_contact_matrix, residue_ids, weight_threshold=weight_threshold
    )

    # Print summary statistics
    print("\nAverage network statistics:")
    print(f"Total frames averaged: {n_frames}")
    print(f"Number of shells analyzed: {n_shells}")
    print(f"Weight threshold used: {weight_threshold}")
    print(f"Number of nodes: {G_avg.number_of_nodes()}")
    print(f"Number of edges: {G_avg.number_of_edges()}")

    # Calculate and print edge weight statistics
    weights = [d["weight"] for _, _, d in G_avg.edges(data=True)]
    if weights:
        weights_np = np.array(weights)
        print("\nEdge weight statistics:")
        print(f"  Min: {weights_np.min():.3f}")
        print(f"  Max: {weights_np.max():.3f}")
        print(f"  Mean: {weights_np.mean():.3f}")
        print(f"  Median: {np.median(weights_np):.3f}")
    else:
        print("\nNo edges present in the network.")

    return G_avg


def contact_matrix_to_graph(
    contact_matrix: np.ndarray, residue_ids: Sequence[int], weight_threshold: float = 0.1
) -> nx.Graph:
    """
    Convert a contact matrix to a NetworkX graph with filtered edges.

    Args:
        contact_matrix: numpy array containing contact weights (normalized frequencies)
        residue_ids: list of residue IDs corresponding to matrix indices
        weight_threshold: minimum weight value for including an edge

    Returns:
        NetworkX graph with filtered edges above threshold
    """
    G = nx.Graph()
    G.add_nodes_from(residue_ids)

    n_residues = len(residue_ids)
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            weight = contact_matrix[i, j]
            if weight > weight_threshold:
                G.add_edge(residue_ids[i], residue_ids[j], weight=float(weight))

    return G
