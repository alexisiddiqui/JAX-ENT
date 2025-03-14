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

from jaxent.forwardmodels.functions import calc_BV_contacts_universe
from jaxent.forwardmodels.netHDX_functions import NetHDXConfig, create_average_network


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


def test_hbond_network():
    """Test the H-bond network implementation with multiple visualization options"""
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    # topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    trajectory_path = topology_path.replace(".pdb", "_small.xtc")

    # Create test output directory
    test_dir = "tests/hbond_network_viz"
    os.makedirs(test_dir, exist_ok=True)

    print("\nTesting H-bond network functionality...")
    print("-" * 80)

    # H-bond parameter configurations
    configs = [
        # NetHDXConfig(distance_cutoff=5, angle_cutoff=150.0),
        NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
            angle_cutoff=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        # NetHDXConfig(distance_cutoff=2.5, angle_cutoff=170.0),
    ]

    # Visualization parameter sets
    viz_params = {
        "layout": ["spring", "circular", "kamada_kawai"],
        # "color_scheme": ["community", "degree", "index"],
        "color_scheme": ["index"],
        "edge_style": ["weight", "gradient"],
        "show_labels": [True],
    }

    universe = mda.Universe(topology_path)
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}:")
        print(f"  Distance cutoff: {config.distance_cutoff}Å")
        print(f"  Angle cutoff: {config.angle_cutoff}°")

        try:
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
            str_dist_list = "_".join([str(i) for i in config.distance_cutoff])
            str_angle_list = "_".join([str(i) for i in config.angle_cutoff])

            viz_name = f"d{str_dist_list}_a{str_angle_list}"
            output_pml_path = os.path.join(test_dir, f"average_network_{viz_name}.pml")

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
                viz_name += f"_{layout}_{color_scheme}_{edge_style}"
                viz_name += "_labeled" if show_labels else "_unlabeled"

                title = (
                    f"H-Bond Network\nD={config.distance_cutoff}Å, A={config.angle_cutoff}°\n"
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

                output_path = os.path.join(test_dir, f"{viz_name}.png")
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

                output_path = os.path.join(test_dir, f"{viz_name}_PCA.png")
                fig.savefig(output_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"Saved visualization: {output_path}")

                # PyMOL script for visualization

            # Add this inside the main try block where other visualizations are created

        except Exception as e:
            print(f"Analysis failed for configuration {i}: {str(e)}")
            continue

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_hbond_network()
