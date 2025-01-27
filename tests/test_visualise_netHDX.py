import os
from itertools import product

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from MDAnalysis import Universe

from jaxent.forwardmodels.models import HBondNetwork


def plot_hbond_network_enhanced_PCA(
    G: nx.Graph,
    universe: Universe,  # Add Universe parameter to access coordinates
    title: str = "H-Bond Network",
    layout: str = "spring",
    color_scheme: str = "single",
    edge_style: str = "weight",
    show_labels: bool = True,
) -> Figure:
    """Enhanced helper function to plot hydrogen bond networks with spatial regularization"""
    from scipy.spatial.distance import pdist, squareform
    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize list to store distance matrices for each frame
    distance_matrices = []

    # Get list of residues and their IDs once
    resids = []
    ca_selection = []
    for residue in universe.residues:
        try:
            ca_atoms = residue.atoms.select_atoms("name CA")
            if len(ca_atoms) > 0:
                ca_selection.append(ca_atoms)
                resids.append(residue.resid)
        except:
            continue

    # Iterate through all frames in trajectory
    for ts in universe.trajectory:
        # Get CA positions for current frame
        ca_positions = np.array([ca.positions[0] for ca in ca_selection])

        # Calculate pairwise distances for this frame
        pairwise_distances = pdist(ca_positions)
        distance_matrix = squareform(pairwise_distances)
        distance_matrices.append(distance_matrix)

    # Average distance matrices across all frames
    avg_distance_matrix = np.mean(distance_matrices, axis=0)

    # Perform PCA on the averaged distance matrix
    pca = PCA(n_components=2)
    ca_positions_2d = pca.fit_transform(avg_distance_matrix)

    # Create dictionary mapping resids to their 2D PCA-projected coordinates
    pos_init = {resid: pos for resid, pos in zip(resids, ca_positions_2d)}
    # Normalize initial positions to fit in plot
    if pos_init:
        positions = np.array(list(pos_init.values()))
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        scale = max_pos - min_pos
        for resid in pos_init:
            pos_init[resid] = (pos_init[resid] - min_pos) / scale

    # Layout options with appropriate parameters
    if layout == "spring":
        # Use initial positions as starting point for spring layout
        pos = nx.spring_layout(G, k=10.0, iterations=50, seed=42, pos=pos_init)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        # Use initial positions for Kamada-Kawai
        pos = nx.kamada_kawai_layout(G, pos=pos_init)
    elif layout == "spectral":
        pos = nx.spectral_layout(G, scale=10)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    # Node coloring
    if color_scheme == "community":
        communities = nx.community.greedy_modularity_communities(G)
        colors = [plt.cm.tab20(i) for i in range(len(communities))]
        node_colors = []
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(colors[i])
                    break
    elif color_scheme == "degree":
        degrees = dict(G.degree())
        node_colors = [plt.cm.viridis(d / max(degrees.values())) for d in degrees.values()]
    else:
        node_colors = "lightblue"

    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [max(500, v * 100) for v in degrees.values()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6)

    # Edge styling
    if G.number_of_edges() > 0:
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        max_weight = max(weights)

        if edge_style == "gradient":
            edge_widths = [3.0 for _ in weights]
            edge_colors = [plt.cm.YlOrRd(w / max_weight) for w in weights]
        else:  # weight-based
            edge_widths = [max(1.0, 5.0 * w / max_weight) for w in weights]
            edge_colors = "gray"

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

        if show_labels:
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # Node labels
    if show_labels:
        nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.margins(0.2)

    return fig


def plot_hbond_network_enhanced(
    G: nx.Graph,
    title: str = "H-Bond Network",
    layout: str = "spring",
    color_scheme: str = "single",
    edge_style: str = "weight",
    show_labels: bool = True,
) -> Figure:
    """Enhanced helper function to plot hydrogen bond networks"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Layout options with appropriate parameters
    if layout == "spring":
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42, weight="weight")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G, scale=2)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    # Node coloring
    if color_scheme == "community":
        communities = nx.community.greedy_modularity_communities(G)
        colors = [plt.cm.tab20(i) for i in range(len(communities))]
        node_colors = []
        for node in G.nodes():
            for i, comm in enumerate(communities):
                if node in comm:
                    node_colors.append(colors[i])
                    break
    elif color_scheme == "degree":
        degrees = dict(G.degree())
        node_colors = [plt.cm.viridis(d / max(degrees.values())) for d in degrees.values()]
    else:
        node_colors = "lightblue"

    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [max(500, v * 100) for v in degrees.values()]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6)

    # Edge styling
    if G.number_of_edges() > 0:
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        max_weight = max(weights)

        if edge_style == "gradient":
            edge_widths = [3.0 for _ in weights]
            edge_colors = [plt.cm.YlOrRd(w / max_weight) for w in weights]
        else:  # weight-based
            edge_widths = [max(1.0, 5.0 * w / max_weight) for w in weights]
            edge_colors = "gray"

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

        if show_labels:
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # Node labels
    if show_labels:
        nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.margins(0.2)

    return fig


def test_hbond_network_class():
    """Test the HBondNetwork class implementation with multiple visualization options"""
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

    # Create test output directory
    test_dir = "tests/hbond_network_viz"
    os.makedirs(test_dir, exist_ok=True)

    print("\nTesting HBondNetwork class...")
    print("-" * 80)

    # H-bond parameter sets
    hbond_params = [
        {"distance_cutoff": 2.5, "angle_cutoff": 120.0},
        # {"distance_cutoff": 2.5, "angle_cutoff": 150.0},
        # {"distance_cutoff": 2.5, "angle_cutoff": 170.0},
    ]

    # Visualization parameter sets
    viz_params = {
        "layout": ["spring", "circular", "kamada_kawai"],
        "color_scheme": ["community", "degree"],
        "edge_style": ["weight", "gradient"],
        "show_labels": [True],
    }

    universe = mda.Universe(topology_path, trajectory_path)

    for i, params in enumerate(hbond_params, 1):
        print(f"\nParameter set {i}: {params}")
        network = HBondNetwork(**params)

        try:
            # Get the averaged network directly
            G = network.get_average_network(universe)

            # Print network statistics
            print("\nNetwork Analysis:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")

            # Get edge weight statistics
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            if weights:
                print(f"  Mean H-bond frequency: {np.mean(weights):.3f}")
                print(f"  Max H-bond frequency: {np.max(weights):.3f}")
                print(f"  Min H-bond frequency: {np.min(weights):.3f}")

            # Generate visualizations for different parameter combinations
            for layout, color_scheme, edge_style, show_labels in product(*viz_params.values()):
                viz_name = f"d{params['distance_cutoff']}_a{params['angle_cutoff']}"
                viz_name += f"_{layout}_{color_scheme}_{edge_style}"
                viz_name += "_labeled" if show_labels else "_unlabeled"

                title = (
                    f"H-Bond Network\nD={params['distance_cutoff']}Å, A={params['angle_cutoff']}°\n"
                    f"{universe.trajectory.n_frames} frames\n"
                    f"{layout.replace('_', ' ').title()} layout, {color_scheme} coloring"
                )

                fig = plot_hbond_network_enhanced(
                    G,
                    title=title,
                    layout=layout,
                    color_scheme=color_scheme,
                    edge_style=edge_style,
                    show_labels=show_labels,
                )

                output_path = os.path.join(test_dir, f"{viz_name}.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved visualization: {output_path}")

                fig = plot_hbond_network_enhanced_PCA(
                    G,
                    universe=universe,
                    title=title,
                    layout=layout,
                    color_scheme=color_scheme,
                    edge_style=edge_style,
                    show_labels=show_labels,
                )

                output_path = os.path.join(test_dir, f"{viz_name}_PCA.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved visualization: {output_path}")

        except Exception as e:
            print(f"Analysis failed for parameter set {i}: {str(e)}")
            continue

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_hbond_network_class()
