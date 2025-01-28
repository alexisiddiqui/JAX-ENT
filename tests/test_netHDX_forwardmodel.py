import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from jaxent.forwardmodels.models import netHDX_model
from jaxent.forwardmodels.netHDX_functions import (
    NetHDXConfig,
    build_hbond_network,
    create_average_network,
    identify_hbond_acceptors,
    identify_hbond_donors,
)


def plot_hbond_network(G: nx.Graph, title: str = "H-Bond Network") -> Figure:
    """Helper function to plot a hydrogen bond network

    Args:
        G: NetworkX graph representing the H-bond network
        title: Title for the plot

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    print("\nDebug - Network properties:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Edge data:")
    for e in G.edges(data=True):
        print(f"  Edge {e[0]}-{e[1]}: weight={e[2]['weight']}")

    # Calculate node positions using spring layout with adjusted parameters
    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Draw nodes with size based on degree
    degrees = dict(G.degree())
    node_sizes = [max(500, v * 100) for v in degrees.values()]
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=node_sizes, alpha=0.6)

    # Draw edges with width based on weight
    if G.number_of_edges() > 0:
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        max_weight = max(weights)
        edge_widths = [max(1.0, 5.0 * w / max_weight) for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.7)

        # Add weight labels on edges
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis("off")
    plt.margins(0.2)  # Add more space around the plot

    return fig


def test_nethdx_model():
    """Test the netHDX model implementation"""
    # Test data paths
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

    print("\nInitializing netHDX model test...")
    print("-" * 80)

    # Load test system
    universe = mda.Universe(topology_path, trajectory_path)
    ensemble = [universe for _ in range(2)]  # Small test ensemble

    # Initialize model
    model = netHDX_model()

    # Test initialization
    try:
        init_success = model.initialise(ensemble)
        print(f"Model initialization: {'✓' if init_success else '✗'}")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    # Test featurization
    try:
        feature_data = build_hbond_network(ensemble)
        print("Featurization successful:")
        print(f"  Number of frames: {len(feature_data.contact_matrices)}")
        print(f"  Matrix shape: {feature_data.contact_matrices[0].shape}")
    except Exception as e:
        print(f"Featurization failed: {str(e)}")
        return

    print("\nTest completed successfully!")


def test_hbond_network():
    """Test the H-bond network implementation with visualization"""
    topology_path = (
        "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

    print("\nTesting H-bond network functionality...")
    print("-" * 80)

    # Test different parameter sets
    test_configs = [
        NetHDXConfig(distance_cutoff=3.5, angle_cutoff=150.0),  # Standard H-bond parameters
        NetHDXConfig(distance_cutoff=4.5, angle_cutoff=150.0),  # Slightly stricter
        NetHDXConfig(distance_cutoff=5, angle_cutoff=150.0),  # The strictest for testing only
    ]

    # Create test output directory
    output_dir = "tests/hbond_network_viz"
    os.makedirs(output_dir, exist_ok=True)

    universe = mda.Universe(topology_path, trajectory_path)

    for i, config in enumerate(test_configs, 1):
        print(f"\nConfiguration {i}:")
        print(f"  Distance cutoff: {config.distance_cutoff}Å")
        print(f"  Angle cutoff: {config.angle_cutoff}°")

        try:
            # Test donor/acceptor identification
            donors = identify_hbond_donors(universe)
            acceptors = identify_hbond_acceptors(universe)
            print(f"Found {len(donors)} donors and {len(acceptors)} acceptors")

            # Get average network
            G = create_average_network(universe, config)

            print("\nNetwork analysis:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")

            # Calculate network metrics
            degrees = dict(G.degree())
            avg_degree = sum(degrees.values()) / len(degrees)
            print(f"  Average degree: {avg_degree:.2f}")

            # Edge weight statistics
            weights = [d["weight"] for (u, v, d) in G.edges(data=True)]
            if weights:
                print(f"  Average H-bond frequency: {np.mean(weights):.3f}")
                print(f"  Max H-bond frequency: {np.max(weights):.3f}")
                print(f"  Min H-bond frequency: {np.min(weights):.3f}")

            # Identify hubs (top 10% by degree)
            hub_threshold = np.percentile(list(degrees.values()), 90)
            hubs = [node for node, degree in degrees.items() if degree > hub_threshold]
            print(f"  Hub residues (top 10%): {sorted(hubs)}")

            # Plot and save network
            title = (
                f"H-Bond Network\n"
                f"Distance cutoff: {config.distance_cutoff}Å, "
                f"Angle cutoff: {config.angle_cutoff}°\n"
                f"({universe.trajectory.n_frames} frames)"
            )

            fig = plot_hbond_network(G, title)
            output_path = os.path.join(
                output_dir,
                f"hbond_network_d{config.distance_cutoff}_a{config.angle_cutoff}.png",
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            print(f"\nSaved visualization to: {output_path}")

        except Exception as e:
            print(f"Analysis failed for configuration {i}: {str(e)}")
            continue

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_nethdx_model()
    test_hbond_network()
