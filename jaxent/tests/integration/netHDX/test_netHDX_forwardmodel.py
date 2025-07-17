import os
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import numpy as np
from matplotlib.figure import Figure

from jaxent.src.models.func.contacts import calc_BV_contacts_universe
from jaxent.src.models.func.netHDX import (
    NetHDXConfig,
    build_hbond_network,
    create_average_network,
    identify_hbond_acceptors,
    identify_hbond_donors,
)
from jaxent.src.models.HDX.netHDX.forwardmodel import netHDX_model
from jaxent.tests.test_utils import get_inst_path


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
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)

    # Test data paths
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    trajectory_path = inst_path / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

    print("\nInitializing netHDX model test...")
    print("-" * 80)

    # Load test system
    universe = mda.Universe(str(topology_path), str(trajectory_path))
    ensemble = [universe for _ in range(2)]  # Small test ensemble

    # Initialize model
    model = netHDX_model(config=NetHDXConfig())

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
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)

    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    trajectory_path = inst_path / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

    print("\nTesting H-bond network functionality...")
    print("-" * 80)

    # Test different parameter sets
    test_configs = [
        NetHDXConfig(distance_cutoff=2.5, angle_cutoff=120.0),  # Standard H-bond parameters
        NetHDXConfig(distance_cutoff=3.0, angle_cutoff=120.0),  # Slightly stricter
        NetHDXConfig(distance_cutoff=3.0, angle_cutoff=150.0),  # The strictest for testing only
    ]

    # Create test output directory
    output_dir = base_dir / "tests" / "hbond_network_viz"
    os.makedirs(output_dir, exist_ok=True)

    universe = mda.Universe(str(topology_path), str(trajectory_path))

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


def load_contact_data(data_dir: Path, file_prefix: str = "Contacts") -> dict:
    """Load contact data from .tmp files.

    Args:
        data_dir: Directory containing the .tmp files
        file_prefix: Prefix of files to load ("Contacts" or "Hbonds")

    Returns:
        Dictionary mapping residue IDs to their contact values
    """
    import glob

    # Get all .tmp files matching the prefix
    pattern = str(data_dir / f"{file_prefix}_chain_0_res_*.tmp")
    tmp_files = glob.glob(pattern)

    contact_data = {}
    for file_path in tmp_files:
        # Extract residue number from filename
        filename = Path(file_path).name
        try:
            resid = int(filename.split("res_")[1].split(".tmp")[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not extract residue ID from filename {file_path}")
            continue

        # Read first value from file
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:  # Check if line is not empty
                    contact_value = float(first_line.split()[0])
                    contact_data[resid] = contact_value
        except (ValueError, IndexError):
            print(f"Warning: Could not read contact value from {file_path}")
            continue

    return contact_data


def test_calc_contacts_universe():
    """Test the calculation of contacts against reference data with detailed comparison."""
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)

    topology_path = inst_path / "clean" / "HOIP" / "train_HOIP_high_rank_1" / "HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    data_dir = inst_path / "clean" / "HOIP" / "train_HOIP_high_rank_1"

    universe = mda.Universe(str(topology_path))

    # Get N atoms (for heavy atom contacts)
    NH_residue_atom_index: List[Tuple[int, int]] = []
    for residue in universe.residues:
        if residue.resname != "PRO":
            try:
                N_atom = residue.atoms.select_atoms("name N")[0]
                NH_residue_atom_index.append((residue.resid, N_atom.index))
            except IndexError:
                continue

    HN_residue_atom_index: List[Tuple[int, int]] = []
    for residue in universe.residues:
        if residue.resname != "PRO":
            try:
                H_atom = residue.atoms.select_atoms("name H")[0]
                HN_residue_atom_index.append((residue.resid, H_atom.index))
            except IndexError:
                continue
    NH_residues = set(res_num for res_num, _ in NH_residue_atom_index)
    HN_residues = set(res_num for res_num, _ in HN_residue_atom_index)

    # Find the intersection of residue numbers
    common_residues = NH_residues.intersection(HN_residues)

    # Filter both lists to keep only the entries where residue number is in the intersection
    NH_residue_atom_index = [
        (res_num, atom_idx)
        for res_num, atom_idx in NH_residue_atom_index
        if res_num in common_residues
    ]
    HN_residue_atom_index = [
        (res_num, atom_idx)
        for res_num, atom_idx in HN_residue_atom_index
        if res_num in common_residues
    ]

    # Sort both lists by residue number to ensure they're in the same order
    NH_residue_atom_index.sort()
    HN_residue_atom_index.sort()

    # Convert atom indices to actual Atom objects
    NH_atoms = [universe.atoms[atom_idx] for _, atom_idx in NH_residue_atom_index]
    HN_atoms = [universe.atoms[atom_idx] for _, atom_idx in HN_residue_atom_index]

    # Calculate contacts - pass Atom objects instead of indices
    heavy_contacts = calc_BV_contacts_universe(
        universe=universe,
        target_atoms=NH_atoms,
        contact_selection="heavy",
        radius=6.5,
        switch=False,
    )

    oxygen_contacts = calc_BV_contacts_universe(
        universe=universe,
        target_atoms=HN_atoms,
        contact_selection="oxygen",
        radius=2.4,
        switch=False,
    )

    # Load reference data
    ref_contacts = load_contact_data(data_dir, "Contacts")
    ref_hbonds = load_contact_data(data_dir, "Hbonds")

    # Print comparison header
    print("\nDetailed Contacts Comparison:")
    print("-" * 120)
    print(
        f"{'Residue ID':^10} {'Calc Heavy':^12} {'Ref Heavy':^12} {'Heavy Δ':^12} {'Heavy %Δ':^12} "
        f"{'Calc O':^12} {'Ref O':^12} {'O Δ':^12} {'O %Δ':^12} {'Match?':^8}"
    )
    print("-" * 120)

    matches = []
    mismatches = []

    # Compare calculated vs reference values
    for i, ((resid, _), heavy, oxygen) in enumerate(
        zip(NH_residue_atom_index, heavy_contacts, oxygen_contacts)
    ):
        calc_heavy = np.mean(heavy)
        calc_oxygen = np.mean(oxygen)

        ref_heavy = ref_contacts.get(resid, None)
        ref_oxygen = ref_hbonds.get(resid, None)

        if ref_heavy is not None and ref_oxygen is not None:
            heavy_diff = calc_heavy - ref_heavy  # Note: Changed to show direction
            oxygen_diff = calc_oxygen - ref_oxygen

            heavy_pct = (heavy_diff / ref_heavy * 100) if ref_heavy != 0 else float("inf")
            oxygen_pct = (oxygen_diff / ref_oxygen * 100) if ref_oxygen != 0 else float("inf")

            # Consider a match if both absolute differences are within tolerance
            matches_within_tol = (abs(heavy_diff) < 1.0) and (abs(oxygen_diff) < 1.0)

            comparison = {
                "resid": resid,
                "calc_heavy": calc_heavy,
                "ref_heavy": ref_heavy,
                "heavy_diff": heavy_diff,
                "heavy_pct": heavy_pct,
                "calc_oxygen": calc_oxygen,
                "ref_oxygen": ref_oxygen,
                "oxygen_diff": oxygen_diff,
                "oxygen_pct": oxygen_pct,
            }

            if matches_within_tol:
                matches.append(comparison)
            else:
                mismatches.append(comparison)

            # Add +/- signs to differences
            heavy_diff_str = f"{'+' if heavy_diff > 0 else ''}{heavy_diff:.2f}"
            oxygen_diff_str = f"{'+' if oxygen_diff > 0 else ''}{oxygen_diff:.2f}"
            heavy_pct_str = f"{'+' if heavy_diff > 0 else ''}{heavy_pct:.1f}%"
            oxygen_pct_str = f"{'+' if oxygen_diff > 0 else ''}{oxygen_pct:.1f}%"

            print(
                f"{resid:^10d} {calc_heavy:^12.2f} {ref_heavy:^12.2f} {heavy_diff_str:^12} {heavy_pct_str:^12} "
                f"{calc_oxygen:^12.2f} {ref_oxygen:^12.2f} {oxygen_diff_str:^12} {oxygen_pct_str:^12} "
                f"{'✓' if matches_within_tol else '✗':^8}"
            )

    # Print summary with direction analysis
    print("\nSummary:")
    print(f"Total residues compared: {len(matches) + len(mismatches)}")
    print(f"Matching contacts: {len(matches)}")
    print(f"Mismatching contacts: {len(mismatches)}")

    if mismatches:
        print("\nMismatch Analysis:")
        print("-" * 80)

        # Analyze trends in mismatches
        heavy_higher = sum(1 for m in mismatches if m["heavy_diff"] > 0)
        heavy_lower = sum(1 for m in mismatches if m["heavy_diff"] < 0)
        oxygen_higher = sum(1 for m in mismatches if m["oxygen_diff"] > 0)
        oxygen_lower = sum(1 for m in mismatches if m["oxygen_diff"] < 0)

        print("\nHeavy Contact Trends:")
        print(f"  Higher than reference: {heavy_higher} residues")
        print(f"  Lower than reference:  {heavy_lower} residues")
        print(f"  Average deviation: {np.mean([m['heavy_diff'] for m in mismatches]):.2f}")
        print(f"  Average % change: {np.mean([m['heavy_pct'] for m in mismatches]):.1f}%")

        print("\nOxygen Contact Trends:")
        print(f"  Higher than reference: {oxygen_higher} residues")
        print(f"  Lower than reference:  {oxygen_lower} residues")
        print(f"  Average deviation: {np.mean([m['oxygen_diff'] for m in mismatches]):.2f}")
        print(f"  Average % change: {np.mean([m['oxygen_pct'] for m in mismatches]):.1f}%")

        print("\nLargest Mismatches:")
        sorted_by_heavy = sorted(mismatches, key=lambda x: abs(x["heavy_pct"]), reverse=True)[:5]
        sorted_by_oxygen = sorted(mismatches, key=lambda x: abs(x["oxygen_pct"]), reverse=True)[:5]

        print("\nTop 5 Heavy Contact Mismatches:")
        for m in sorted_by_heavy:
            print(
                f"  Residue {m['resid']}: {m['calc_heavy']:.2f} vs {m['ref_heavy']:.2f} "
                f"(Δ: {m['heavy_diff']:+.2f}, {m['heavy_pct']:+.1f}%)"
            )

        print("\nTop 5 Oxygen Contact Mismatches:")
        for m in sorted_by_oxygen:
            print(
                f"  Residue {m['resid']}: {m['calc_oxygen']:.2f} vs {m['ref_oxygen']:.2f} "
                f"(Δ: {m['oxygen_diff']:+.2f}, {m['oxygen_pct']:+.1f}%)"
            )

    # Assert that most contacts match within tolerance
    match_ratio = len(matches) / (len(matches) + len(mismatches))
    assert match_ratio > 0.9, (
        f"Only {match_ratio:.1%} of contacts match reference values (threshold: 90%)"
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_nethdx_model()
    test_hbond_network()
