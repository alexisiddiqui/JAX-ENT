#!/usr/bin/env python3
"""
Clustering script for molecular dynamics trajectory analysis.

This script performs PCA and clustering (K-means or rules-based) on previously calculated
structural features from the analyse_LocalFeatures_PUFs.py script.

It reads in:
- Feature data (.npy files)
- Ensemble information
- Feature specifications (JSON)
- Rules specifications (JSON, optional)

And performs:
- PCA on pairwise distances
- K-means or rules-based clustering
- Visualization and plotting
- Cluster trajectory saving

Input directory structure expected:
analysis_output/
├── data/
│   ├── all_features_data.npy
│   ├── ensemble_labels.npy
│   ├── feature_names.txt
│   ├── ensemble_names.txt
│   └── {ensemble_name}/
│       ├── {ensemble_name}_features.npy
│       ├── {ensemble_name}_info.json
│       └── ...
└── ...

This script can be run after analyse_LocalFeatures_PUFs.py to perform clustering
on the calculated features.

python jaxent/examples/2_CrossValidation/analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --input_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    

python jaxent/examples/2_CrossValidation/analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_plddt_ordered_all_filtered.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_plddt_ordered.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --input_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_complete \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_complete_cluster \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --json_rules_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_rules_spec.json
    


"""

import argparse
import csv  # added to write feature CSVs
import datetime
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from MDAnalysis.analysis import align
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Color mapping for ensembles - customize as needed
ENSEMBLE_COLORS = {
    "AF2-MSAss": "#1f77b4",  # Blue
    "AF2-Filtered": "#ff7f0e",  # Orange
    "pH4": "#2ca02c",  # Green
    "pH7": "#d62728",  # Red
    "Folded": "#9467bd",  # Purple
    "Unfolded": "#8c564b",  # Brown
    "Control": "#e377c2",  # Pink
    "Treatment": "#7f7f7f",  # Gray
    "NMR-20C": "#bcbd22",  # Olive
    "NMR-37C": "#17becf",  # Cyan
}

# Default colors for ensembles not in the mapping
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def setup_logger(log_file):
    """Set up the logger to output to both file and console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    return logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Perform PCA and clustering on previously calculated structural features"
    )
    parser.add_argument(
        "--ensembles",
        nargs="+",
        type=str,
        required=True,
        help="List of 'topology,trajectory' pairs for each ensemble (must match analysis input)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        required=True,
        help="Names for each ensemble (must match analysis input)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing analysis results from analyse_LocalFeatures_PUFs.py",
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=False,
        help="Path to JSON file containing structural region definitions",
    )
    parser.add_argument(
        "--json_feature_spec",
        type=str,
        default=None,
        help="Path to JSON file describing region-specific features (for metadata)",
    )
    parser.add_argument(
        "--json_rules_spec",
        type=str,
        default=None,
        help="Path to JSON file describing rules for clustering",
    )
    parser.add_argument(
        "--atom_selection",
        type=str,
        default="name CA",
        help='Atom selection string for PCA (default: "name CA")',
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of clusters for k-means (default: 4, ignored if rules-based)",
    )
    parser.add_argument(
        "--num_components_pca_coords",
        type=int,
        default=10,
        help="Number of PCA components for pairwise coordinates (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: input_dir + '_clustered')",
    )
    parser.add_argument(
        "--save_pdbs",
        action="store_true",
        help="Save individual PDB files for each cluster (default: False)",
    )
    parser.add_argument(
        "--log", action="store_true", default=True, help="Enable logging (default: True)"
    )

    return parser.parse_args()


def validate_and_parse_ensembles(ensembles_list, names_list):
    """Validate and parse ensemble inputs"""
    ensembles = []
    ensemble_names = []

    for i, ensemble_str in enumerate(ensembles_list):
        try:
            topology, trajectory = ensemble_str.split(",", 1)
            topology = topology.strip()
            trajectory = trajectory.strip()

            # Check if files exist
            if not os.path.exists(topology):
                raise FileNotFoundError(f"Topology file not found: {topology}")
            if not os.path.exists(trajectory):
                raise FileNotFoundError(f"Trajectory file not found: {trajectory}")

            ensembles.append((topology, trajectory))
            ensemble_names.append(names_list[i])

        except ValueError:
            raise ValueError(f"Invalid ensemble format: {ensemble_str}. Use 'topology,trajectory'")

    if len(names_list) != len(ensembles):
        raise ValueError(
            f"Number of names ({len(names_list)}) must match number of ensembles ({len(ensembles)})"
        )

    return ensembles, ensemble_names


def get_ensemble_color(ensemble_name, ensemble_idx):
    """Get color for ensemble based on name or default colors"""
    if ensemble_name in ENSEMBLE_COLORS:
        return ENSEMBLE_COLORS[ensemble_name]
    else:
        return DEFAULT_COLORS[ensemble_idx % len(DEFAULT_COLORS)]


def load_feature_spec(json_path):
    """Load feature specifications from JSON file"""
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def load_rules_spec(json_path):
    """Load rules specifications from JSON file"""
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def load_analysis_data(input_dir):
    """Load previously calculated analysis data"""
    data_dir = os.path.join(input_dir, "data")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    logger.info("Loading analysis data...")

    # Load main data files
    all_features_data = np.load(os.path.join(data_dir, "all_features_data.npy"))
    ensemble_labels = np.load(os.path.join(data_dir, "ensemble_labels.npy"))

    # Load feature names
    with open(os.path.join(data_dir, "feature_names.txt"), "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Load ensemble names
    with open(os.path.join(data_dir, "ensemble_names.txt"), "r") as f:
        stored_ensemble_names = [line.strip() for line in f.readlines()]

    logger.info(f"Loaded features data: {all_features_data.shape}")
    logger.info(f"Features: {', '.join(feature_names)}")
    logger.info(f"Stored ensemble names: {stored_ensemble_names}")

    return all_features_data, ensemble_labels, feature_names, stored_ensemble_names


def evaluate_simple_rule(feature_values, rule):
    """Evaluate a simple threshold-based rule on feature values"""
    min_thresh = rule.get("min_threshold")
    max_thresh = rule.get("max_threshold")
    inclusive = rule.get("inclusive", True)

    if inclusive:
        # Inclusive: value >= min and value <= max
        return (feature_values >= min_thresh) & (feature_values <= max_thresh)
    else:
        # Exclusive: value < min or value > max
        return (feature_values < min_thresh) | (feature_values > max_thresh)


def evaluate_rule_expression(all_features_data, feature_names, rule_expr, logger=None):
    """
    Recursively evaluate a rule expression with boolean logic.

    Args:
        all_features_data: numpy array of shape (n_frames, n_features)
        feature_names: list of feature names
        rule_expr: rule expression (dict or list)
        logger: logger object for logging

    Returns:
        numpy array of booleans indicating which frames satisfy the rule
    """
    n_frames = all_features_data.shape[0]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    # If it's a simple rule (has spec_name), evaluate it directly
    if isinstance(rule_expr, dict) and "spec_name" in rule_expr:
        spec_name = rule_expr["spec_name"]

        if spec_name not in feature_to_idx:
            if logger:
                logger.warning(
                    f"Feature '{spec_name}' not found in calculated features. Skipping rule."
                )
            return np.zeros(n_frames, dtype=bool)

        feature_idx = feature_to_idx[spec_name]
        feature_values = all_features_data[:, feature_idx]
        return evaluate_simple_rule(feature_values, rule_expr)

    # If it's a logical operation
    elif isinstance(rule_expr, dict) and "operator" in rule_expr:
        operator = rule_expr["operator"].upper()

        if operator == "NOT":
            # NOT operator expects a single operand
            operand = rule_expr.get("operand") or rule_expr.get("operands")
            if isinstance(operand, list) and len(operand) > 0:
                operand = operand[0]
            result = evaluate_rule_expression(all_features_data, feature_names, operand, logger)
            return ~result

        elif operator in ["AND", "OR"]:
            operands = rule_expr.get("operands", [])
            if not operands:
                if logger:
                    logger.warning(f"{operator} operator requires operands")
                return (
                    np.zeros(n_frames, dtype=bool)
                    if operator == "AND"
                    else np.ones(n_frames, dtype=bool)
                )

            # Evaluate all operands
            results = []
            for operand in operands:
                result = evaluate_rule_expression(all_features_data, feature_names, operand, logger)
                results.append(result)

            # Combine results based on operator
            if operator == "AND":
                combined = results[0]
                for result in results[1:]:
                    combined = combined & result
                return combined
            else:  # OR
                combined = results[0]
                for result in results[1:]:
                    combined = combined | result
                return combined

        else:
            if logger:
                logger.warning(f"Unknown operator: {operator}")
            return np.zeros(n_frames, dtype=bool)

    # If it's a list (implicit AND for backward compatibility)
    elif isinstance(rule_expr, list):
        if not rule_expr:
            return np.ones(n_frames, dtype=bool)

        # Treat as AND of all rules in the list
        results = []
        for rule in rule_expr:
            result = evaluate_rule_expression(all_features_data, feature_names, rule, logger)
            results.append(result)

        combined = results[0]
        for result in results[1:]:
            combined = combined & result
        return combined

    else:
        if logger:
            logger.warning(f"Invalid rule expression format: {type(rule_expr)}")
        return np.zeros(n_frames, dtype=bool)


def format_rule_expression(rule_expr, indent=0):
    """Format a rule expression as a human-readable string for logging"""
    indent_str = "  " * indent

    # Simple rule
    if isinstance(rule_expr, dict) and "spec_name" in rule_expr:
        spec_name = rule_expr["spec_name"]
        min_thresh = rule_expr.get("min_threshold")
        max_thresh = rule_expr.get("max_threshold")
        inclusive = rule_expr.get("inclusive", True)

        if inclusive:
            return f"{indent_str}{min_thresh} <= {spec_name} <= {max_thresh}"
        else:
            return f"{indent_str}{spec_name} < {min_thresh} OR {spec_name} > {max_thresh}"

    # Logical operation
    elif isinstance(rule_expr, dict) and "operator" in rule_expr:
        operator = rule_expr["operator"].upper()

        if operator == "NOT":
            operand = rule_expr.get("operand") or rule_expr.get("operands", [None])[0]
            return (
                f"{indent_str}NOT (\n{format_rule_expression(operand, indent + 1)}\n{indent_str})"
            )

        elif operator in ["AND", "OR"]:
            operands = rule_expr.get("operands", [])
            if not operands:
                return f"{indent_str}{operator} (empty)"

            formatted_operands = [format_rule_expression(op, indent + 1) for op in operands]
            return (
                f"{indent_str}{operator} (\n"
                + f"\n{indent_str}{operator}\n".join(formatted_operands)
                + f"\n{indent_str})"
            )

        else:
            return f"{indent_str}Unknown operator: {operator}"

    # List (implicit AND)
    elif isinstance(rule_expr, list):
        if not rule_expr:
            return f"{indent_str}(empty list)"
        formatted_rules = [format_rule_expression(rule, indent + 1) for rule in rule_expr]
        return f"{indent_str}AND (implicit) (\n" + "\n".join(formatted_rules) + f"\n{indent_str})"

    else:
        return f"{indent_str}Invalid expression"


def perform_rules_based_clustering(all_features_data, feature_names, rules_spec):
    """
    Perform clustering based on rules with support for complex boolean logic.

    The rules_spec can now contain:
    1. Simple rules (backward compatible): list of rules with implicit AND
    2. Complex rules: dictionary with "operator" key for boolean logic
    """
    logger.info("Performing rules-based clustering with boolean logic support...")
    logger.info("=" * 60)
    logger.info("RULES SPECIFICATION SUMMARY:")
    logger.info("=" * 60)

    n_frames = all_features_data.shape[0]
    cluster_labels = np.full(n_frames, -1, dtype=int)

    # Sort cluster ID's for priority processing
    sorted_cluster_ids = sorted(rules_spec.keys(), key=int)

    # Log the rules specification
    for cluster_id_str in sorted_cluster_ids:
        cluster_id = int(cluster_id_str)
        rules = rules_spec[cluster_id_str]
        logger.info(f"Cluster {cluster_id}:")
        logger.info(format_rule_expression(rules, indent=1))
        logger.info("")

    logger.info("Cluster assignment priority: Lower-numbered clusters have higher priority")
    logger.info("Frames not matching any cluster rules will be assigned to Cluster -1")
    logger.info("=" * 60)

    # Process each cluster
    for cluster_id_str in sorted_cluster_ids:
        cluster_id = int(cluster_id_str)
        rules = rules_spec[cluster_id_str]

        logger.info(f"Processing Cluster {cluster_id} rules...")

        # Evaluate the rule expression
        cluster_mask = evaluate_rule_expression(all_features_data, feature_names, rules, logger)

        # Only assign to frames not yet assigned (priority system)
        unassigned_mask = cluster_labels == -1
        assign_mask = cluster_mask & unassigned_mask
        frames_to_assign = np.sum(assign_mask)

        cluster_labels[assign_mask] = cluster_id

        logger.info(f"  Frames satisfying rules: {np.sum(cluster_mask)}")
        logger.info(f"  Frames assigned to Cluster {cluster_id}: {frames_to_assign}")
        logger.info(f"  Remaining unassigned frames: {np.sum(cluster_labels == -1)}")
        logger.info("")

    # Log final results
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    logger.info("=" * 60)
    logger.info("FINAL CLUSTERING RESULTS:")
    logger.info("=" * 60)
    for label, count in zip(unique_labels, counts):
        percentage = (count / n_frames) * 100
        logger.info(f"Cluster {label}: {count} frames ({percentage:.1f}%)")
    logger.info("=" * 60)

    return cluster_labels


def calculate_distances_and_perform_pca(universe, selection, num_components, chunk_size):
    """Calculate pairwise distances and perform incremental PCA in chunks"""
    logger.info("Calculating pairwise distances and performing incremental PCA...")

    # Select atoms for distance calculation
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2

    logger.info(f"Selected {n_atoms} atoms, processing {n_frames} frames")
    logger.info(f"Each frame will generate {n_distances} pairwise distances")

    # Initialize IncrementalPCA
    ipca_batch_size = max(num_components * 10, 100)
    pca = IncrementalPCA(n_components=num_components, batch_size=ipca_batch_size)

    # Process frames in chunks
    frame_indices = np.arange(n_frames)

    # Store PCA coordinates for all frames
    pca_coords = np.zeros((n_frames, num_components))

    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="Processing frame chunks"):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_indices = frame_indices[chunk_start:chunk_end]
        chunk_size_actual = len(chunk_indices)

        # Pre-allocate array just for this chunk of frames
        chunk_distances = np.zeros((chunk_size_actual, n_distances))

        # Process each frame in the chunk
        for i, frame_idx in enumerate(chunk_indices):
            # Go to the specific frame
            universe.trajectory[frame_idx]

            # Get atom positions for this frame
            positions = atoms.positions

            # Calculate pairwise distances for this frame
            frame_distances = pdist(positions, metric="euclidean")

            # Store the distances for this frame in the chunk array
            chunk_distances[i] = frame_distances

        # Partial fit PCA with this chunk
        if chunk_start == 0:
            # For the first chunk, we need to fit and transform
            chunk_pca_coords = pca.fit_transform(chunk_distances)
        else:
            # For subsequent chunks, we partial_fit and transform
            pca.partial_fit(chunk_distances)
            chunk_pca_coords = pca.transform(chunk_distances)

        # Store the PCA coordinates for this chunk
        pca_coords[chunk_start:chunk_end] = chunk_pca_coords

        # Free memory by deleting the chunk distances
        del chunk_distances

    logger.info(f"Completed incremental PCA with {num_components} components")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    return pca_coords, pca


def perform_kmeans_clustering(features_data, n_clusters):
    """Perform k-means clustering on feature data"""
    logger.info(f"Performing k-means clustering with {n_clusters} clusters...")

    # Use MiniBatchKMeans for large datasets
    if len(features_data) > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=1000, n_init="auto"
        )
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    cluster_labels = kmeans.fit_predict(features_data)
    cluster_centers = kmeans.cluster_centers_

    # Count frames per cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    logger.info(f"Clustering complete: {len(unique_labels)} clusters")
    logger.info(f"Average frames per cluster: {np.mean(counts):.1f}")
    logger.info(f"Min frames per cluster: {np.min(counts)}")
    logger.info(f"Max frames per cluster: {np.max(counts)}")

    return cluster_labels, cluster_centers, kmeans


def create_publication_plots(
    all_pca_coords,
    ensemble_names,
    ensemble_labels,
    cluster_labels,
    cluster_centers,
    pca,
    output_dir,
):
    """Create publication-quality plots of the PCA results and clustering"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set style for publication-quality plots
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.edgecolor"] = "black"

    logger.info("Creating PCA projection plots with clusters...")

    n_ensembles = len(ensemble_names)
    fig, axes = plt.subplots(2, n_ensembles, figsize=(8 * n_ensembles, 12))
    if n_ensembles == 1:
        axes = axes.reshape(-1, 1)

    variance_pc1 = pca.explained_variance_ratio_[0] * 100
    variance_pc2 = pca.explained_variance_ratio_[1] * 100

    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        ensemble_pca_coords = all_pca_coords[ensemble_mask]
        ensemble_clusters = cluster_labels[ensemble_mask]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        x, y = ensemble_pca_coords[:, 0], ensemble_pca_coords[:, 1]

        # Top row: Colored by clusters
        ax_main = axes[0, j]
        sns.kdeplot(x=x, y=y, ax=ax_main, levels=10, cmap="Blues", fill=True, alpha=0.3)
        scatter = ax_main.scatter(x, y, c=ensemble_clusters, cmap="viridis", s=20, alpha=0.7)

        ax_main.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=12)
        ax_main.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=12)
        ax_main.set_title(f"PCA with Clusters - {ensemble_name}", fontsize=14)

        if j == n_ensembles - 1:  # Only add colorbar to rightmost plot
            plt.colorbar(scatter, ax=ax_main, label="Cluster Label")

        # Bottom row: Colored by ensemble
        ax_ensemble = axes[1, j]
        sns.kdeplot(x=x, y=y, ax=ax_ensemble, levels=10, cmap="Blues", fill=True, alpha=0.3)
        ax_ensemble.scatter(x, y, color=ensemble_color, s=20, alpha=0.7, label=ensemble_name)

        ax_ensemble.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=12)
        ax_ensemble.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=12)
        ax_ensemble.set_title(f"PCA by Ensemble - {ensemble_name}", fontsize=14)
        ax_ensemble.legend()

    plt.tight_layout()
    pca_plot_path = os.path.join(plots_dir, "pca_clusters_ensembles.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create combined ensemble plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Combined plot colored by clusters
    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        x = all_pca_coords[ensemble_mask, 0]
        y = all_pca_coords[ensemble_mask, 1]
        ensemble_clusters = cluster_labels[ensemble_mask]

        scatter1 = ax1.scatter(
            x, y, c=ensemble_clusters, cmap="viridis", s=20, alpha=0.7, label=ensemble_name
        )

    ax1.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
    ax1.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
    ax1.set_title("All Ensembles - Colored by Clusters", fontsize=16)
    plt.colorbar(scatter1, ax=ax1, label="Cluster Label")

    # Combined plot colored by ensemble
    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        x = all_pca_coords[ensemble_mask, 0]
        y = all_pca_coords[ensemble_mask, 1]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        ax2.scatter(x, y, color=ensemble_color, s=20, alpha=0.7, label=ensemble_name)

    ax2.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
    ax2.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
    ax2.set_title("All Ensembles - Colored by Ensemble", fontsize=16)
    ax2.legend()

    plt.tight_layout()
    combined_plot_path = os.path.join(plots_dir, "pca_combined_ensembles.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "pca_plot": pca_plot_path,
        "combined_plot": combined_plot_path,
    }


def create_pca_feature_plot(
    all_features_data,
    feature_names,
    ensemble_names,
    ensemble_labels,
    output_dir,
    cluster_labels=None,
):
    """Perform PCA on features and create scatter plots with separate panels per ensemble."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Performing PCA on features...")

    # Handle NaN values by replacing them with the mean of the column
    features_data_cleaned = np.nan_to_num(
        all_features_data, nan=np.nanmean(all_features_data, axis=0)
    )

    # Standardize features for PCA
    scaler = StandardScaler()
    features_data_scaled = scaler.fit_transform(features_data_cleaned)

    # Perform PCA on all features together
    n_components = min(len(feature_names), 10)
    logger.info(f"Performing PCA on {len(feature_names)} features with {n_components} components")

    pca_features = IncrementalPCA(n_components=n_components)
    pca_features_coords = pca_features.fit_transform(features_data_scaled)

    logger.info(f"PCA explained variance ratio: {pca_features.explained_variance_ratio_[:2]}")

    n_ensembles = len(ensemble_names)
    fig, axes = plt.subplots(1, n_ensembles, figsize=(8 * n_ensembles, 6))
    if n_ensembles == 1:
        axes = [axes]

    variance_pc1 = pca_features.explained_variance_ratio_[0] * 100
    variance_pc2 = pca_features.explained_variance_ratio_[1] * 100

    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        x = pca_features_coords[ensemble_mask, 0]
        y = pca_features_coords[ensemble_mask, 1]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        ax = axes[j]

        # Color by cluster if available, otherwise use ensemble color
        if cluster_labels is not None:
            ensemble_clusters = cluster_labels[ensemble_mask]
            scatter = ax.scatter(x, y, c=ensemble_clusters, cmap="viridis", s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="Cluster Label")
        else:
            ax.scatter(x, y, color=ensemble_color, s=20, alpha=0.7)

        ax.set_xlabel(f"Feature PC1 ({variance_pc1:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"Feature PC2 ({variance_pc2:.1f}% variance)", fontsize=12)
        ax.set_title(f"PCA of Features - {ensemble_name}", fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "pca_features.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"PCA of features plots created and saved to: {save_path}")
    return save_path


def create_cluster_ratio_plot(cluster_labels, ensemble_names, ensemble_labels, output_dir):
    """Create plots showing the distribution ratio of frames across clusters"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating cluster ratio plots...")

    # Count frames per cluster for each ensemble
    unique_clusters = np.unique(cluster_labels)
    n_ensembles = len(ensemble_names)

    fig, axes = plt.subplots(1, n_ensembles + 1, figsize=(6 * (n_ensembles + 1), 8))
    if n_ensembles == 0:
        axes = [axes]

    # Plot for each ensemble
    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        ensemble_clusters = cluster_labels[ensemble_mask]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        unique_ensemble_clusters, counts = np.unique(ensemble_clusters, return_counts=True)
        total_frames = len(ensemble_clusters)
        percentages = (counts / total_frames) * 100

        ax = axes[j]
        bars = ax.bar(unique_ensemble_clusters, counts, color=ensemble_color, alpha=0.8)

        # Add count labels above bars
        for bar, count, percentage in zip(bars, counts, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (total_frames * 0.01),
                f"{count}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Number of Frames", fontsize=12)
        ax.set_title(f"Frames per Cluster - {ensemble_name}", fontsize=14)
        ax.set_xticks(unique_clusters)
        ax.set_xticklabels([f"C{c}" for c in unique_clusters])

    # Combined plot
    ax_combined = axes[-1]
    width = 0.8 / n_ensembles
    x_offset = np.arange(len(unique_clusters))

    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        ensemble_clusters = cluster_labels[ensemble_mask]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        # Count clusters, ensuring all clusters are represented
        counts = np.zeros(len(unique_clusters))
        unique_ensemble_clusters, ensemble_counts = np.unique(ensemble_clusters, return_counts=True)

        for idx, cluster in enumerate(unique_clusters):
            if cluster in unique_ensemble_clusters:
                counts[idx] = ensemble_counts[unique_ensemble_clusters == cluster][0]

        ax_combined.bar(
            x_offset + j * width,
            counts,
            width,
            label=ensemble_name,
            color=ensemble_color,
            alpha=0.8,
        )

    ax_combined.set_xlabel("Cluster", fontsize=12)
    ax_combined.set_ylabel("Number of Frames", fontsize=12)
    ax_combined.set_title("Frames per Cluster - All Ensembles", fontsize=14)
    ax_combined.set_xticks(x_offset + width * (n_ensembles - 1) / 2)
    ax_combined.set_xticklabels([f"C{c}" for c in unique_clusters])

    ax_combined.legend()

    plt.tight_layout()
    ratio_plot_path = os.path.join(plots_dir, "cluster_ratios_ensembles.png")
    plt.savefig(ratio_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Cluster ratio plots created: {ratio_plot_path}")
    return {"ratio_plot": ratio_plot_path}


def save_cluster_trajectories(
    ensemble_data,
    cluster_labels,
    pca_coords,
    cluster_centers,
    ensemble_names,
    ensemble_labels,
    output_dir,
    save_pdbs=False,
    all_features_data=None,
    feature_names=None,
):
    """Save cluster trajectories and optionally save representative PDB files.
    Additional optional args:
      - all_features_data: numpy array (n_frames, n_features) used to write CSVs next to PDBs
      - feature_names: list of feature column names
    """
    logger.info("Saving cluster trajectories...")

    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Find representative frames (medoids) for each cluster
    representative_frames = {}

    for cluster_idx in unique_clusters:
        cluster_mask = cluster_labels == cluster_idx
        if not np.any(cluster_mask):
            continue

        cluster_frame_indices = np.where(cluster_mask)[0]
        cluster_pca_coords = pca_coords[cluster_mask]

        if cluster_centers is not None and len(cluster_centers) > cluster_idx and cluster_idx >= 0:
            center = cluster_centers[cluster_idx]
        else:
            center = np.mean(cluster_pca_coords, axis=0)

        distances = np.sqrt(np.sum((cluster_pca_coords - center) ** 2, axis=1))
        min_dist_idx = np.argmin(distances)
        representative_frame_idx = cluster_frame_indices[min_dist_idx]

        ensemble_idx = ensemble_labels[representative_frame_idx]
        ensemble_frame_indices = np.where(ensemble_labels == ensemble_idx)[0]
        local_frame_idx = np.where(ensemble_frame_indices == representative_frame_idx)[0][0]

        representative_frames[cluster_idx] = {
            "global_idx": representative_frame_idx,
            "ensemble_idx": ensemble_idx,
            "local_frame_idx": local_frame_idx,
        }

    # Save ensemble-specific cluster data
    for ensemble_idx, ensemble_info in enumerate(ensemble_data):
        ensemble_name = ensemble_info["name"]
        ensemble_safe_name = ensemble_name.replace(" ", "_").replace("-", "_")

        ensemble_clusters_dir = os.path.join(clusters_dir, ensemble_safe_name)
        os.makedirs(ensemble_clusters_dir, exist_ok=True)

        ensemble_mask = ensemble_labels == ensemble_idx
        ensemble_cluster_labels = cluster_labels[ensemble_mask]
        ensemble_frame_indices = np.where(ensemble_mask)[0]

        ensemble_centers_file = os.path.join(
            ensemble_clusters_dir, f"{ensemble_safe_name}_cluster_centers.xtc"
        )
        universe = ensemble_info["universe"]

        with mda.Writer(ensemble_centers_file, universe.atoms.n_atoms) as writer:
            for cluster_idx in unique_clusters:
                if cluster_idx in representative_frames:
                    rep_info = representative_frames[cluster_idx]
                    if rep_info["ensemble_idx"] == ensemble_idx:
                        universe.trajectory[rep_info["local_frame_idx"]]
                        writer.write(universe.atoms)

        logger.info(f"Saved cluster centers for {ensemble_name}: {ensemble_centers_file}")

        mapping_file = os.path.join(
            ensemble_clusters_dir, f"{ensemble_safe_name}_frame_to_cluster.csv"
        )
        with open(mapping_file, "w") as f:
            f.write("local_frame_index,global_frame_index,cluster_label\n")
            for local_idx, (global_idx, cluster_label) in enumerate(
                zip(ensemble_frame_indices, ensemble_cluster_labels)
            ):
                f.write(f"{local_idx},{global_idx},{cluster_label}\n")

        # --- Add: write per-ensemble features CSV for all frames (parent ensemble) ---
        if all_features_data is not None and feature_names is not None:
            ensemble_features_csv = os.path.join(
                ensemble_clusters_dir, f"{ensemble_safe_name}_all_frames_features.csv"
            )
            header = [
                "local_frame_index",
                "global_frame_index",
                "cluster_label",
                "ensemble_index",
            ] + feature_names
            with open(ensemble_features_csv, "w", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(header)
                for local_idx, global_idx in enumerate(ensemble_frame_indices):
                    cluster_label = int(ensemble_cluster_labels[local_idx])
                    feats = all_features_data[int(global_idx)].tolist()
                    writer.writerow(
                        [local_idx, int(global_idx), cluster_label, ensemble_idx] + feats
                    )
            logger.info(f"Saved per-ensemble features CSV: {ensemble_features_csv}")
        # --- end addition ---

        # Save PDB files and CSVs if requested
        if save_pdbs:
            logger.info(f"Saving multi-frame PDB files for ensemble: {ensemble_name}")

            pdb_dir = os.path.join(ensemble_clusters_dir, "pdbs")
            os.makedirs(pdb_dir, exist_ok=True)

            # Save medoid/centroid frames for each cluster (multiframe) and medoids CSV
            medoids_file = os.path.join(pdb_dir, f"{ensemble_safe_name}_cluster_medoids.pdb")
            medoids_features_csv = medoids_file.replace(".pdb", "_features.csv")
            medoids_rows = []

            with mda.Writer(medoids_file, multiframe=True) as writer:
                for cluster_idx in unique_clusters:
                    if cluster_idx in representative_frames:
                        rep_info = representative_frames[cluster_idx]
                        if rep_info["ensemble_idx"] == ensemble_idx:
                            universe.trajectory[rep_info["local_frame_idx"]]
                            writer.write(universe.atoms)
                            # Gather feature row if features available
                            if all_features_data is not None:
                                global_idx = rep_info["global_idx"]
                                feats = all_features_data[global_idx].tolist()
                                medoids_rows.append(
                                    [
                                        int(cluster_idx),
                                        int(global_idx),
                                        int(rep_info["ensemble_idx"]),
                                        int(rep_info["local_frame_idx"]),
                                    ]
                                    + feats
                                )

            logger.info(f"Saved cluster medoids PDB: {medoids_file}")

            # Write medoids features CSV if data available
            if (
                all_features_data is not None
                and feature_names is not None
                and len(medoids_rows) > 0
            ):
                header = [
                    "cluster_idx",
                    "global_idx",
                    "ensemble_idx",
                    "local_frame_idx",
                ] + feature_names
                with open(medoids_features_csv, "w", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(header)
                    writer.writerows(medoids_rows)
                logger.info(f"Saved medoids features CSV: {medoids_features_csv}")

            # Save multi-frame PDB for each cluster and per-cluster features CSV
            for cluster_idx in unique_clusters:
                cluster_frames_mask = ensemble_cluster_labels == cluster_idx
                if not np.any(cluster_frames_mask):
                    continue

                cluster_local_indices = np.where(cluster_frames_mask)[0]
                n_cluster_frames = len(cluster_local_indices)
                if n_cluster_frames == 0:
                    continue

                cluster_pdb_file = os.path.join(
                    pdb_dir, f"{ensemble_safe_name}_cluster_{cluster_idx}_frames.pdb"
                )
                cluster_features_csv = cluster_pdb_file.replace(".pdb", "_features.csv")

                cluster_csv_rows = []

                with mda.Writer(cluster_pdb_file, multiframe=True) as writer:
                    for local_frame_idx in tqdm(
                        cluster_local_indices,
                        desc=f"Saving cluster {cluster_idx} frames for {ensemble_name}",
                    ):
                        universe.trajectory[local_frame_idx]
                        writer.write(universe.atoms)

                        # global index corresponding to this local frame within ensemble
                        global_idx = ensemble_frame_indices[local_frame_idx]
                        if all_features_data is not None:
                            feats = all_features_data[global_idx].tolist()
                            cluster_csv_rows.append(
                                [
                                    int(cluster_idx),
                                    int(global_idx),
                                    int(ensemble_idx),
                                    int(local_frame_idx),
                                ]
                                + feats
                            )

                logger.info(
                    f"Saved {n_cluster_frames} frames for cluster {cluster_idx}: {cluster_pdb_file}"
                )

                # Write cluster features CSV
                if (
                    all_features_data is not None
                    and feature_names is not None
                    and len(cluster_csv_rows) > 0
                ):
                    header = [
                        "cluster_idx",
                        "global_idx",
                        "ensemble_idx",
                        "local_frame_idx",
                    ] + feature_names
                    with open(cluster_features_csv, "w", newline="") as csvf:
                        writer = csv.writer(csvf)
                        writer.writerow(header)
                        writer.writerows(cluster_csv_rows)
                    logger.info(f"Saved cluster features CSV: {cluster_features_csv}")

            # Save individual medoid PDB files and single-row CSVs
            for cluster_idx in unique_clusters:
                if cluster_idx in representative_frames:
                    rep_info = representative_frames[cluster_idx]
                    if rep_info["ensemble_idx"] == ensemble_idx:
                        medoid_pdb_file = os.path.join(
                            pdb_dir, f"{ensemble_safe_name}_cluster_{cluster_idx}_medoid.pdb"
                        )
                        universe.trajectory[rep_info["local_frame_idx"]]
                        universe.atoms.write(medoid_pdb_file)
                        logger.info(
                            f"Saved medoid PDB for cluster {cluster_idx}: {medoid_pdb_file}"
                        )

                        # Write single-row feature CSV next to medoid PDB
                        if all_features_data is not None and feature_names is not None:
                            global_idx = rep_info["global_idx"]
                            feats = all_features_data[global_idx].tolist()
                            medoid_feature_csv = medoid_pdb_file.replace(".pdb", "_feature.csv")
                            header = [
                                "cluster_idx",
                                "global_idx",
                                "ensemble_idx",
                                "local_frame_idx",
                            ] + feature_names
                            with open(medoid_feature_csv, "w", newline="") as csvf:
                                writer = csv.writer(csvf)
                                writer.writerow(header)
                                writer.writerow(
                                    [
                                        int(cluster_idx),
                                        int(global_idx),
                                        int(rep_info["ensemble_idx"]),
                                        int(rep_info["local_frame_idx"]),
                                    ]
                                    + feats
                                )
                            logger.info(f"Saved medoid feature CSV: {medoid_feature_csv}")

    # Save global frame-to-cluster mapping with ensemble information
    global_mapping_file = os.path.join(clusters_dir, "global_frame_to_cluster_ensemble.csv")
    with open(global_mapping_file, "w") as f:
        f.write("global_frame_index,cluster_label,ensemble_index,ensemble_name\n")
        for global_idx, (cluster_label, ensemble_label) in enumerate(
            zip(cluster_labels, ensemble_labels)
        ):
            ensemble_name = ensemble_names[ensemble_label]
            f.write(f"{global_idx},{cluster_label},{ensemble_label},{ensemble_name}\n")

    logger.info(f"Saved {n_clusters} clusters across all ensembles")
    logger.info("Saved global frame-to-cluster-ensemble mapping")


def create_feature_histograms_by_cluster(
    all_features_data, feature_names, cluster_labels, ensemble_names, ensemble_labels, output_dir
):
    """Create histograms for each feature, colored by cluster ID"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating feature histograms colored by cluster...")

    # Get unique clusters and create color map
    unique_clusters = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)

    # Use a colormap for clusters
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}

    # Special color for unassigned frames (cluster -1)
    if -1 in unique_clusters:
        cluster_color_map[-1] = "lightgray"

    n_features = len(feature_names)

    # Create subplots - arrange in a grid
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

    # Flatten axes array for easier indexing
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_features > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        feature_data = all_features_data[:, i]

        # Create histogram for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = feature_data[cluster_mask]

            if len(cluster_data) > 0:
                cluster_label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unassigned"
                ax.hist(
                    cluster_data,
                    bins=30,
                    alpha=0.7,
                    color=cluster_color_map[cluster_id],
                    label=f"{cluster_label} (n={len(cluster_data)})",
                    density=True,
                )

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Distribution of {feature_name} by Cluster", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(plots_dir, "feature_histograms_by_cluster.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature histograms by cluster saved to: {save_path}")

    # Also create individual plots for each feature for better readability
    for i, feature_name in enumerate(feature_names):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        feature_data = all_features_data[:, i]

        # Create histogram for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = feature_data[cluster_mask]

            if len(cluster_data) > 0:
                cluster_label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unassigned"

                # Calculate statistics
                mean_val = np.mean(cluster_data)
                std_val = np.std(cluster_data)

                ax.hist(
                    cluster_data,
                    bins=30,
                    alpha=0.7,
                    color=cluster_color_map[cluster_id],
                    label=f"{cluster_label} (n={len(cluster_data)}, μ={mean_val:.2f}, σ={std_val:.2f})",
                    density=True,
                )

        ax.set_xlabel(feature_name, fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"Distribution of {feature_name} by Cluster", fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual feature plot
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        individual_save_path = os.path.join(
            plots_dir, f"histogram_{safe_feature_name}_by_cluster.png"
        )
        plt.savefig(individual_save_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Individual feature histograms by cluster created")
    return save_path


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Start timer
    start_time = datetime.datetime.now()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir + "_clustered"

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    global logger
    log_file = os.path.join(args.output_dir, "clustering_analysis.log")
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("Starting clustering analysis on previously calculated features")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")

    # Validate and parse ensemble inputs
    try:
        ensembles, ensemble_names = validate_and_parse_ensembles(args.ensembles, args.names)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    # Load analysis data
    try:
        all_features_data, ensemble_labels, feature_names, stored_ensemble_names = (
            load_analysis_data(args.input_dir)
        )
    except FileNotFoundError as e:
        logger.error(f"Error loading analysis data: {e}")
        sys.exit(1)

    # Verify ensemble names match
    if ensemble_names != stored_ensemble_names:
        logger.warning(
            f"Input ensemble names {ensemble_names} don't match stored names {stored_ensemble_names}"
        )
        logger.warning("Proceeding with stored names from analysis data")
        ensemble_names = stored_ensemble_names

    # Load feature specification (for metadata)
    feature_spec = load_feature_spec(args.json_feature_spec)
    rules_spec = load_rules_spec(args.json_rules_spec)

    # Load universes for PCA calculation
    ensemble_data = []
    all_pca_coords = []
    global_frame_idx = 0

    for i, ((topology, trajectory), name) in enumerate(zip(ensembles, ensemble_names)):
        logger.info(f"Loading ensemble {i + 1}: {name}")
        logger.info(f"  Topology: {topology}")
        logger.info(f"  Trajectory: {trajectory}")

        universe = mda.Universe(topology, trajectory)
        n_frames = len(universe.trajectory)

        logger.info(f"  Loaded {len(universe.atoms)} atoms and {n_frames} frames")

        # Store ensemble information
        ensemble_info = {
            "universe": universe,
            "name": name,
            "n_frames": n_frames,
            "start_frame_idx": global_frame_idx,
            "end_frame_idx": global_frame_idx + n_frames,
        }
        ensemble_data.append(ensemble_info)
        global_frame_idx += n_frames

    # --- Added: align trajectories to the first ensemble for easier visual comparison ---
    logger.info("Aligning trajectories to reference ensemble for easier comparison...")
    if len(ensemble_data) > 0:
        ref_universe = ensemble_data[0]["universe"]
        try:
            ref_atoms = ref_universe.select_atoms(args.atom_selection)
        except Exception as e:
            logger.warning(f"Reference selection failed: {e}. Skipping alignment.")
            ref_atoms = None

        if ref_atoms is not None:
            for i, ensemble_info in enumerate(ensemble_data):
                uni = ensemble_info["universe"]
                name = ensemble_info["name"]
                try:
                    sel_atoms = uni.select_atoms(args.atom_selection)
                except Exception as e:
                    logger.warning(
                        f"Selection '{args.atom_selection}' failed for {name}: {e}. Skipping alignment."
                    )
                    continue

                # Ensure the selection has the same number of atoms as the reference
                if sel_atoms.n_atoms != ref_atoms.n_atoms:
                    logger.warning(
                        f"Skipping alignment for {name}: selection atom count {sel_atoms.n_atoms} != reference {ref_atoms.n_atoms}"
                    )
                    continue

                try:
                    aligner = align.AlignTraj(
                        uni, ref_universe, select=args.atom_selection, in_memory=True
                    )
                    aligner.run()
                    logger.info(
                        f"Aligned ensemble '{name}' to reference '{ensemble_data[0]['name']}'"
                    )
                except Exception as e:
                    logger.warning(f"Alignment failed for {name}: {e}")
    # --- end addition ---

    # Calculate PCA coordinates for each ensemble
    logger.info("Calculating PCA coordinates for each ensemble...")
    for i, ensemble_info in enumerate(ensemble_data):
        universe = ensemble_info["universe"]
        name = ensemble_info["name"]

        logger.info(f"Processing PCA for ensemble: {name}")
        chunk_size = len(universe.trajectory)  # Process each ensemble in one chunk

        pca_coords, pca = calculate_distances_and_perform_pca(
            universe, args.atom_selection, args.num_components_pca_coords, chunk_size
        )

        all_pca_coords.append(pca_coords)
        ensemble_info["pca"] = pca

    # Combine all PCA coordinates
    all_pca_coords = np.vstack(all_pca_coords)

    # Normalize features before clustering
    features_data_cleaned = np.nan_to_num(
        all_features_data, nan=np.nanmean(all_features_data, axis=0)
    )
    features_scaled = StandardScaler().fit_transform(features_data_cleaned)

    # Perform clustering
    if rules_spec:
        logger.info("Performing rules-based clustering with boolean logic support...")
        cluster_labels = perform_rules_based_clustering(
            all_features_data, feature_names, rules_spec
        )
        # Rules-based clustering doesn't have centers in the same way k-means does.
        # We can calculate them as the mean of the features for each cluster.
        unique_clusters = np.unique(cluster_labels)
        cluster_centers = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Exclude unclassified points
                cluster_mask = cluster_labels == cluster_id
                if np.any(cluster_mask):
                    cluster_center = features_scaled[cluster_mask].mean(axis=0)
                    cluster_centers.append(cluster_center)
        cluster_centers = np.array(cluster_centers) if cluster_centers else None
    else:
        # Perform k-means clustering on the features
        logger.info(f"Performing k-means clustering with {args.n_clusters} clusters...")
        cluster_labels, cluster_centers, kmeans_features = perform_kmeans_clustering(
            features_scaled, args.n_clusters
        )

    # Create plots
    logger.info("Creating plots...")

    # Feature histograms by cluster (NEW)
    create_feature_histograms_by_cluster(
        all_features_data,
        feature_names,
        cluster_labels,
        ensemble_names,
        ensemble_labels,
        args.output_dir,
    )

    # PCA of features (separate panels per ensemble)
    if len(feature_names) >= 2:
        feature_pca_plot = create_pca_feature_plot(
            all_features_data,
            feature_names,
            ensemble_names,
            ensemble_labels,
            args.output_dir,
            cluster_labels=cluster_labels,
        )

    # PCA plots with clusters (pairwise distance coordinates)
    pca_for_plotting = ensemble_data[0]["pca"]
    plot_paths = create_publication_plots(
        all_pca_coords,
        ensemble_names,
        ensemble_labels,
        cluster_labels,
        cluster_centers,
        pca_for_plotting,
        args.output_dir,
    )

    # Cluster ratio plots
    cluster_ratio_plots = create_cluster_ratio_plot(
        cluster_labels, ensemble_names, ensemble_labels, args.output_dir
    )

    # Save cluster trajectories and mapping
    save_cluster_trajectories(
        ensemble_data,
        cluster_labels,
        features_scaled,
        cluster_centers,
        ensemble_names,
        ensemble_labels,
        args.output_dir,
        args.save_pdbs,
        all_features_data=all_features_data,
        feature_names=feature_names,
    )

    # Save clustering results
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save global clustering data
    np.save(os.path.join(data_dir, "cluster_labels.npy"), cluster_labels)
    np.save(os.path.join(data_dir, "all_pca_coords.npy"), all_pca_coords)
    if cluster_centers is not None:
        np.save(os.path.join(data_dir, "cluster_centers.npy"), cluster_centers)

    # Save ensemble-specific clustering data
    for ensemble_idx, ensemble_info in enumerate(ensemble_data):
        ensemble_name = ensemble_info["name"]
        ensemble_safe_name = ensemble_name.replace(" ", "_").replace("-", "_")

        # Create ensemble-specific data directory
        ensemble_data_dir = os.path.join(data_dir, ensemble_safe_name)
        os.makedirs(ensemble_data_dir, exist_ok=True)

        # Get data for this ensemble
        ensemble_mask = ensemble_labels == ensemble_idx
        ensemble_pca_coords = all_pca_coords[ensemble_mask]
        ensemble_cluster_labels = cluster_labels[ensemble_mask]

        # Save ensemble-specific data
        np.save(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_pca_coords.npy"),
            ensemble_pca_coords,
        )
        np.save(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_cluster_labels.npy"),
            ensemble_cluster_labels,
        )

        # Update ensemble info
        ensemble_info_dict = {
            "name": ensemble_name,
            "n_frames": ensemble_info["n_frames"],
            "topology": ensembles[ensemble_idx][0],
            "trajectory": ensembles[ensemble_idx][1],
            "feature_names": feature_names,
            "n_clusters": len(np.unique(ensemble_cluster_labels)),
            "cluster_distribution": {
                int(k): int(v)
                for k, v in zip(*np.unique(ensemble_cluster_labels, return_counts=True))
            },
            "clustering_method": "rules-based" if rules_spec else "k-means",
            "n_pca_components": args.num_components_pca_coords,
        }

        with open(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_clustering_info.json"), "w"
        ) as f:
            json.dump(ensemble_info_dict, f, indent=2)

        logger.info(f"Saved clustering data for {ensemble_name} in {ensemble_data_dir}")

    # End timer and report
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    total_frames = len(ensemble_labels)
    logger.info("=" * 80)
    logger.info(f"Clustering analysis complete. Results saved to {args.output_dir}")
    logger.info(f"Processed {len(ensembles)} ensembles with {total_frames} total frames")
    logger.info(f"Found {len(np.unique(cluster_labels))} clusters")
    if rules_spec:
        logger.info("Used rules-based clustering with boolean logic support")
    else:
        logger.info(f"Used k-means clustering with {args.n_clusters} clusters")
    logger.info(f"Total execution time: {elapsed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
