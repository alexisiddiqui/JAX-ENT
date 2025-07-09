import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from jaxent.src.forwardmodels.models import netHDX_model
from jaxent.src.forwardmodels.netHDX_functions import (
    NetHDXConfig,
    build_hbond_network,
    create_average_network,
    identify_hbond_acceptors,
    identify_hbond_donors,
)


def plot_hbond_network(
    G: nx.Graph, title: str = "H-Bond Network", weight_threshold: float = 0.01
) -> Figure:
    """Helper function to plot a hydrogen bond network

    Args:
        G: NetworkX graph representing the H-bond network
        title: Title for the plot
        weight_threshold: Minimum edge weight to include in visualization (default: 0.01)

    Returns:
        Matplotlib Figure object
    """
    # Create a copy of the graph to avoid modifying the original
    G_filtered = G.copy()

    # Remove edges with weight below threshold
    edges_to_remove = [
        (u, v) for u, v, d in G_filtered.edges(data=True) if d.get("weight", 0) < weight_threshold
    ]
    G_filtered.remove_edges_from(edges_to_remove)

    # Remove isolated nodes (nodes with no remaining edges)
    G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))

    fig, ax = plt.subplots(figsize=(12, 8))

    print("\nDebug - Network properties after filtering:")
    print(f"Number of nodes: {G_filtered.number_of_nodes()}")
    print(f"Number of edges: {G_filtered.number_of_edges()}")
    print(f"Edges removed: {len(edges_to_remove)}")

    # Calculate node positions using spring layout with adjusted parameters
    pos = nx.spring_layout(G_filtered, k=2.0, iterations=50, seed=42)

    # Draw nodes with size based on degree
    degrees = dict(G_filtered.degree())
    node_sizes = [max(500, v * 100) for v in degrees.values()]
    nx.draw_networkx_nodes(G_filtered, pos, node_color="lightblue", node_size=node_sizes, alpha=0.6)

    # Draw edges with width based on weight
    if G_filtered.number_of_edges() > 0:
        edges = G_filtered.edges()
        weights = [G_filtered[u][v]["weight"] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [max(1.0, 5.0 * w / max_weight) for w in weights]
        nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, edge_color="gray", alpha=0.7)

        # Add weight labels on edges
        edge_labels = {(u, v): f"{G_filtered[u][v]['weight']:.2f}" for u, v in G_filtered.edges()}
        nx.draw_networkx_edge_labels(G_filtered, pos, edge_labels, font_size=8)

    # Draw node labels
    nx.draw_networkx_labels(G_filtered, pos)

    plt.title(f"{title}\n(Edges with weight < {weight_threshold:.2f} filtered out)")
    plt.axis("off")
    plt.margins(0.2)  # Add more space around the plot

    return fig


def plot_distributions(G: nx.Graph, title: str) -> Figure:
    """Plot distributions of edge weights and node contact weights with percentiles"""
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 5))
    percentiles = [1, 10, 25, 50]
    colors = ["black", "blue", "green", "red"]  # Distinct colors for each percentile

    edge_weights = [G[u][v].get("distance", 0) for u, v in G.edges()]
    if edge_weights:
        # Main histogram
        n, bins, patches = ax1.hist(
            edge_weights, bins=50, color="skyblue", edgecolor="black", alpha=0.7
        )

        # Calculate percentiles
        perc_values = np.percentile(edge_weights, percentiles)

        # Add percentile lines and labels
        for p, val, color in zip(percentiles, perc_values, colors):
            ax0.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{p}th: {val:.2f}")

        ax0.set_title("Edge Distances Distribution", fontsize=12)
        ax0.set_xlabel("Hydrogen Bond Frequency", fontsize=10)
        ax0.set_ylabel("Count", fontsize=10)
        ax0.grid(True, linestyle="--", alpha=0.6)
        # ax0.set_yscale("log")
        # Add legend outside plot
        ax0.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax0.text(0.5, 0.5, "No Edges Found", ha="center", va="center")
        ax0.set_title("Edge Distances Distribution (No Data)")

    # Edge weights distribution
    edge_weights = [G[u][v].get("weight", 0) for u, v in G.edges()]
    if edge_weights:
        # Main histogram
        n, bins, patches = ax1.hist(
            edge_weights, bins=50, color="skyblue", edgecolor="black", alpha=0.7
        )

        # Calculate percentiles
        perc_values = np.percentile(edge_weights, percentiles)

        # Add percentile lines and labels
        for p, val, color in zip(percentiles, perc_values, colors):
            ax1.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{p}th: {val:.2f}")

        ax1.set_title("Edge Weights Distribution", fontsize=12)
        ax1.set_xlabel("Hydrogen Bond Frequency", fontsize=10)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.set_yscale("log")
        # Add legend outside plot
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax1.text(0.5, 0.5, "No Edges Found", ha="center", va="center")
        ax1.set_title("Edge Weights Distribution (No Data)")

    # Node contact weights distribution (weighted degree)
    weighted_degrees = dict(nx.degree(G, weight="weight"))
    node_weights = list(weighted_degrees.values())
    if node_weights:
        # Main histogram
        n, bins, patches = ax2.hist(
            node_weights, bins=50, color="lightgreen", edgecolor="black", alpha=0.7
        )

        # Calculate percentiles
        perc_values = np.percentile(node_weights, percentiles)

        # Add percentile lines and labels
        for p, val, color in zip(percentiles, perc_values, colors):
            ax2.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{p}th: {val:.2f}")

        ax2.set_title("Node Contact Weights Distribution", fontsize=12)
        ax2.set_xlabel("Total Contact Frequency", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)

        # Add legend outside plot
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax2.text(0.5, 0.5, "No Node Weights", ha="center", va="center")
        ax2.set_title("Node Weights Distribution (No Data)")

    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
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
        NetHDXConfig(distance_cutoff=2.5, angle_cutoff=120.0),  # Standard H-bond parameters
        NetHDXConfig(distance_cutoff=3.0, angle_cutoff=120.0),  # Slightly stricter
        NetHDXConfig(distance_cutoff=3.0, angle_cutoff=150.0),  # The strictest for testing only
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

            # Plot and save distributions
            dist_title = (
                f"H-Bond Frequency Distributions\n"
                f"Distance cutoff: {config.distance_cutoff}Å, "
                f"Angle cutoff: {config.angle_cutoff}°\n"
                f"({universe.trajectory.n_frames} frames)"
            )
            dist_fig = plot_distributions(G, dist_title)
            dist_output_path = os.path.join(
                output_dir,
                f"hbond_distributions_d{config.distance_cutoff}_a{config.angle_cutoff}.png",
            )
            dist_fig.savefig(dist_output_path, bbox_inches="tight", dpi=300)
            plt.close(dist_fig)

            print("\nSaved visualizations to:")
            print(f"  Network: {output_path}")
            print(f"  Distributions: {dist_output_path}")

        except Exception as e:
            print(f"Analysis failed for configuration {i}: {str(e)}")
            continue

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_nethdx_model()
    test_hbond_network()
