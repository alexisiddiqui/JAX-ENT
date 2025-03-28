import os
from itertools import product

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from MDAnalysis import Universe
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from jaxent.models.config import BV_model_Config, NetHDXConfig
from jaxent.models.func.contacts import calc_BV_contacts_universe
from jaxent.models.func.netHDX import create_average_network


def plot_hbond_network_enhanced_PCA(
    G: nx.Graph,
    universe: Universe,
    title: str = "H-Bond Network",
    layout: str = "spring",
    color_scheme: str = "single",
    edge_style: str = "weight",
    show_labels: bool = True,
) -> Figure:
    """
    Enhanced helper function to plot hydrogen bond networks with spatial regularization using PCA.
    Modified to work with N-O distances and proper residue indexing.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create mapping from residue ID to index
    resid_to_idx = {res.resid: i for i, res in enumerate(universe.residues)}
    idx_to_resid = {i: res.resid for i, res in enumerate(universe.residues)}

    # Initialize list to store distance matrices for each frame
    distance_matrices = []

    # Get list of residues and their CA atoms
    ca_selection = []
    for residue in universe.residues:
        try:
            ca_atoms = residue.atoms.select_atoms("name CA")
            if len(ca_atoms) > 0:
                ca_selection.append(ca_atoms)
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

    # Set avg_distance_matrix as avg_dist between all nodes in G using the mapping
    for u, v in G.edges():
        idx_u = resid_to_idx[u]
        idx_v = resid_to_idx[v]
        G[u][v]["avg_dist"] = avg_distance_matrix[idx_u, idx_v]

    # Perform PCA on the averaged distance matrix
    pca = PCA(n_components=2)
    ca_positions_2d = pca.fit_transform(avg_distance_matrix)

    # Create dictionary mapping residue IDs to their 2D PCA-projected coordinates
    pos_init = {idx_to_resid[i]: pos for i, pos in enumerate(ca_positions_2d)}

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
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42, pos=pos_init, weight="avg_dist")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G, pos=pos_init, weight="avg_dist")
    elif layout == "spectral":
        pos = nx.spectral_layout(G, scale=10, weight="avg_dist")
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
    elif color_scheme == "index":
        # Use a colormap based on node index
        node_colors = [plt.cm.inferno(i / (len(G.nodes()) - 1)) for i, _ in enumerate(G.nodes())]
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


def create_pml_script(
    G: nx.Graph, universe: mda.Universe, heavy_contacts: dict, output_pml_path: str
):
    """
    Create a PyMOL .pml script to visualize the average network with bond thickness based on edge weights.

    Args:
        G: NetworkX graph representing the average hydrogen bond network.
        universe: MDAnalysis Universe containing the structure.
        heavy_contacts: Dictionary mapping residue IDs to their contact values
        output_pml_path: Path to save the generated .pml script.
    """
    with open(output_pml_path, "w") as pml_file:
        # Initial setup
        pml_file.write("# Create a copy of loaded structure\n")
        pml_file.write("create 4nx4, all\n")
        pml_file.write("hide all\n")
        pml_file.write("show cartoon\n\n")

        # Iterate over edges to create bonds with thickness based on weight
        pml_file.write("# Create bonds with thickness based on edge weights\n")
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)

            # Get residue numbers and names from the universe
            # Ensure u and v are integers
            u_idx = int(u) - 1  # Convert numpy.int64 to Python int
            v_idx = int(v) - 1

            res_u = universe.residues[u_idx]
            res_v = universe.residues[v_idx]

            # Create atom selections using proper PyMOL syntax
            atom1 = f"/4nx4//A/{res_u.resname}`{res_u.resnum}/CA"
            atom2 = f"/4nx4//A/{res_v.resname}`{res_v.resnum}/CA"

            pml_file.write(f"distance bond_{u}_{v}, {atom1}, {atom2}\n")
            pml_file.write(f"set dash_width, {weight}, bond_{u}_{v}\n")

        pml_file.write("\n# Set B-factors based on heavy atom contacts\n")
        for resid, contact in heavy_contacts.items():
            # Convert numpy.int64 to Python int for the resid
            resid = int(resid)
            # Convert contact value to float if it's numpy type
            contact = float(contact)
            pml_file.write(f"alter 4nx4 and resid {resid} and name CA, b={contact}\n")

        pml_file.write("sort\n")
        pml_file.write("rebuild\n")

        # Visualization settings
        pml_file.write("\n# Set background color\n")
        pml_file.write("bg_color white\n")

    print(f"PyMOL script saved to: {output_pml_path}")


def plot_hbond_network_enhanced(
    G: nx.Graph,
    title: str = "H-Bond Network",
    layout: str = "spring",
    color_scheme: str = "single",
    edge_style: str = "weight",
    show_labels: bool = True,
) -> Figure:
    """
    Enhanced helper function to plot hydrogen bond networks.

    Args:
        G: NetworkX graph of H-bond network
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        color_scheme: Node coloring scheme ('single', 'community', 'degree', 'index')
        edge_style: Edge styling ('weight', 'gradient')
        show_labels: Whether to show node and edge labels

    Returns:
        Matplotlib Figure object
    """

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
    elif color_scheme == "index":
        # use a colormap based on node index
        node_colors = [plt.cm.inferno(i / (len(G.nodes()) - 1)) for i, _ in enumerate(G.nodes())]
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


def plot_hbond_heatmap(
    G: nx.Graph,
    title: str = "H-Bond Contact Heatmap",
    colormap: str = "viridis",
    sort_method: str = "cluster",
) -> Figure:
    """
    Create a heatmap visualization of the contact weights between nodes in the H-bond network.

    Args:
        G: NetworkX graph of H-bond network
        title: Plot title
        colormap: Matplotlib colormap to use for the heatmap
        sort_method: Method to sort nodes ('cluster', 'id', 'weight')

    Returns:
        Matplotlib Figure object
    """
    # Create adjacency matrix from the graph with weights
    nodes = list(G.nodes())
    n = len(nodes)

    # Create a mapping from node IDs to indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize the adjacency matrix with zeros
    adj_matrix = np.zeros((n, n))

    # Fill in the adjacency matrix with edge weights
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        weight = data.get("weight", 1.0)
        adj_matrix[i, j] = weight
        adj_matrix[j, i] = weight  # Make it symmetric for undirected graph

    # Sort the matrix if requested
    sorted_nodes = nodes
    if sort_method == "cluster" and n > 1:
        # Sort nodes by hierarchical clustering
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        # Check if there are any weights to avoid division by zero
        if np.max(adj_matrix) > 0:
            # Normalize the adjacency matrix
            norm_adj_matrix = adj_matrix / np.max(adj_matrix)

            # Convert weights to distances (ensuring no negative values)
            # Higher weight means closer nodes (smaller distance)
            dist_matrix = np.ones_like(norm_adj_matrix) - norm_adj_matrix
            np.fill_diagonal(dist_matrix, 0)  # Set diagonal to 0

            # Convert to condensed form for linkage
            try:
                condensed_dist = squareform(dist_matrix)

                # Perform hierarchical clustering
                Z = hierarchy.linkage(condensed_dist, method="average")
                idx = hierarchy.leaves_list(Z)

                # Reorder the matrix
                adj_matrix = adj_matrix[idx, :][:, idx]
                sorted_nodes = [nodes[i] for i in idx]
            except Exception as e:
                print(f"Clustering failed, using original order: {str(e)}")
                # Fall back to original order
                sorted_nodes = nodes
        else:
            print("No edge weights found, using original order")

    elif sort_method == "weight":
        # Sort by total connection weight
        row_sums = adj_matrix.sum(axis=1)
        idx = np.argsort(-row_sums)  # Sort in descending order
        adj_matrix = adj_matrix[idx, :][:, idx]
        sorted_nodes = [nodes[i] for i in idx]

    elif sort_method == "id":
        # Sort by node ID (assuming IDs are numeric)
        try:
            idx = np.argsort([int(node) for node in nodes])
            adj_matrix = adj_matrix[idx, :][:, idx]
            sorted_nodes = [nodes[i] for i in idx]
        except (ValueError, TypeError):
            print("Could not convert node IDs to integers, using original order")

    # Create the figure and plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(adj_matrix, cmap=colormap, interpolation="nearest")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Contact Weight")

    # Set up axis labels and title
    ax.set_title(title)
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")

    # Add tick labels
    tick_spacing = max(1, n // 20)  # Limit the number of ticks for readability
    ax.set_xticks(np.arange(0, n, tick_spacing))
    ax.set_yticks(np.arange(0, n, tick_spacing))
    ax.set_xticklabels([sorted_nodes[i] for i in range(0, n, tick_spacing)], rotation=90)
    ax.set_yticklabels([sorted_nodes[i] for i in range(0, n, tick_spacing)])

    plt.tight_layout()
    return fig


def test_hbond_network():
    """Test the H-bond network implementation with multiple visualization options"""
    topology_path = "./tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "./tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    # topology_path = "./tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    # topology_path = "./tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    # trajectory_path = topology_path.replace(".pdb", "_small.xtc")

    # Create test output directory
    test_dir = "tests/hbond_network_viz"
    os.system(f"rm -rf {test_dir}")
    os.makedirs(test_dir, exist_ok=True)

    print("\nTesting H-bond network functionality...")
    print("-" * 80)

    # H-bond parameter configurations
    configs = {
        "netHDX_standard": NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
            angle_cutoff=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        "BV_standard": BV_model_Config(),
    }

    # Visualization parameter sets
    viz_params = {
        "layout": ["spring", "kamada_kawai"],
        # "color_scheme": ["community", "degree", "index"],
        "color_scheme": ["index"],
        "edge_style": ["weight", "gradient"],
        "show_labels": [True],
    }

    # Heatmap parameter sets
    heatmap_params = {
        "colormap": ["viridis", "hot"],
        "sort_method": ["cluster", "weight", "id"],
    }

    universe = mda.Universe(topology_path, trajectory_path)
    for config_name, config in configs.items():
        print(f"\nConfiguration: {config_name}")

        try:
            # Create config-specific directory
            config_dir = os.path.join(test_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)

            # Get the averaged network
            G = create_average_network(universe, config)

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

            # Generate PyMOL script
            output_pml_path = os.path.join(config_dir, f"{config_name}.pml")

            NH_residue_atom_index: list[tuple[int, int]] = []
            for residue in universe.residues:
                if residue.resname != "PRO":
                    try:
                        N_atom = residue.atoms.select_atoms("name N")[0]
                        NH_residue_atom_index.append((residue.resid, N_atom.index))
                    except IndexError:
                        continue

            heavy_contacts = calc_BV_contacts_universe(
                universe=universe,
                contact_selection="heavy",
                radius=6.5,
                residue_ignore=(-2, 2),
                residue_atom_index=NH_residue_atom_index,
            )

            # average contacts across all frames: heavy_contacts ((n_targets, n_frames))
            heavy_contacts = np.mean(heavy_contacts, axis=1)
            heavy_contacts = {idx + 1: contact for idx, contact in enumerate(heavy_contacts)}

            create_pml_script(G, universe, heavy_contacts, output_pml_path)

            # Generate visualizations for different parameter combinations
            for layout, color_scheme, edge_style, show_labels in product(*viz_params.values()):
                viz_name = f"{layout}_{color_scheme}_{edge_style}"
                viz_name += "_labeled" if show_labels else "_unlabeled"

                title = (
                    f"H-Bond Network: {config_name}\n"
                    f"{universe.trajectory.n_frames} frames\n"
                    f"{layout.replace('_', ' ').title()} layout, {color_scheme} coloring"
                )

                # Regular network visualization
                fig = plot_hbond_network_enhanced(
                    G,
                    title=title,
                    layout=layout,
                    color_scheme=color_scheme,
                    edge_style=edge_style,
                    show_labels=show_labels,
                )

                output_path = os.path.join(config_dir, f"{viz_name}.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved visualization: {output_path}")

                # PCA-based visualization
                fig = plot_hbond_network_enhanced_PCA(
                    G,
                    universe=universe,
                    title=title,
                    layout=layout,
                    color_scheme=color_scheme,
                    edge_style=edge_style,
                    show_labels=show_labels,
                )

                output_path = os.path.join(config_dir, f"{viz_name}_PCA.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved visualization: {output_path}")

            # Generate heatmap visualizations for different parameter combinations
            for colormap, sort_method in product(*heatmap_params.values()):
                heatmap_name = f"heatmap_{colormap}_{sort_method}"

                heatmap_title = (
                    f"H-Bond Network Heatmap: {config_name}\n"
                    f"{universe.trajectory.n_frames} frames\n"
                    f"Colormap: {colormap}, Sorting: {sort_method}"
                )

                # Generate heatmap
                fig = plot_hbond_heatmap(
                    G,
                    title=heatmap_title,
                    colormap=colormap,
                    sort_method=sort_method,
                )

                output_path = os.path.join(config_dir, f"{heatmap_name}.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved heatmap visualization: {output_path}")

        except Exception as e:
            print(f"Analysis failed for configuration {config_name}: {str(e)}")
            continue

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_hbond_network()
