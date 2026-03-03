"""
[Script Name] analyse_LocalFeatures_PUFs.py

[Brief Description of Functionality]
Analyzes structural features (contacts, RMSD, SASA, etc.) of MoPrP trajectories to characterize
and distinguish between different conformational states (Folded, PUF1, PUF2, Unfolded).
It processes multiple ensembles and computes feature distributions.

Requirements:
    - Trajectory and Topology files for each ensemble/state.
    - `jaxent` library for feature calculation.

Usage:
    # Typically run as part of the analysis pipeline to generate data for clustering
    python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py

Output:
    - Plots and data files containing feature distributions for each state.
into folded, PUF1, PUF2 and unfolded states.

Multi-ensemble support: Each ensemble should have its own topology and trajectory file pair.
The script will analyze all ensembles together and plot them with different colors.

| Structural Element | Original Numbering | PDB Numbering | Key Features |
|-------------------|-------------------|---------------|--------------|
| **N-terminal unstructured** | 23-130 | Not in PDB | Flexible, fast exchange |
| **β1 strand** | 127-132 | N/A (partial) | Part of PUF1, mop = 0.4 kcal/mol/M |
| **Loop β1-α1** | 133-143 | 3-13 | Part of PUF1, Ile138 slow exchange |
| **α1 helix** | 144-153 | 14-23 | C-terminal more stable than N-terminal |
| **Loop α1-β2** | 154-156 | 24-26 | Forms 310 helix at pH 4 |
| **β2 strand** | 160-163 | 30-33 | Part of PUF2, mop = 0.8 kcal/mol/M |
| **Loop β2-α2** | 164-170 | 34-40 | Flexible region |
| **α2 helix** | 171-193 | 41-63 | C-terminal destabilized at pH 4 |
| **Disulfide bond** | Cys178-Cys213 | 48-83 | High stability region, mop = 1.1 kcal/mol/M |
| **Loop α2-α3** | 194-198 | 64-68 | Fast exchange, negligible mop |
| **α3 helix** | 199-223 | 69-93 | Three distinct stability regions |
| **C-terminal tail** | 224-231 | 94-101 | Partially structured |

Features analyzed:
- Internal dihedral angles change from reference (ensure periodic angle handling)
- COM displacement from reference structure
- Region-specific dihedral angle changes for helices and strands
- Pairwise signed COM RMSD differences between regions

# Regions analyzed:
- α1, α2, α3 helices
- β1, β2 strands

Each per frame feature is saved in a separate .npy file.

It generates the following plots:
- Distribution of each feature (all ensembles in same panel)
- Pairwise scatter plots of features (separate panels per ensemble)
- Correlation matrix of features (separate panels per ensemble)
- PCA of features (separate panels per ensemble)
- PCA of pairwise coordinates, annotated by clusters (separate panels per ensemble)

The features are then clustered to 4 clusters using KMeans clustering and the clusters are visualized in PCA space.

Input Args:
- ensembles: List of "topology,trajectory" pairs for each ensemble
- names: Names for each ensemble (for plotting and output)
- output_dir: Directory to save the output plots and feature files
- json_data_path: Path to json file containing metadata about system
- n_clusters: Number of clusters to use for KMeans clustering on the features (default: 4)


python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc --n_clusters 4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters4 --save_pdbs --region_features

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" --names "AF2-Filtered" --n_clusters 4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_4 --save_pdbs --region_features


python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" --names "AF2-Filtered" "AF2-MSAss" --n_clusters 4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_4 --save_pdbs --region_features


python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_4 \
    --save_pdbs \
    --region_features \
    --internal_rmsd \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

    

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "NMR-20C" "NMR-37C" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_4 \
    --save_pdbs \
    --region_features \
    --internal_rmsd \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

    

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "NMR-20C" "NMR-37C" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" \
    --names "AF2-Filtered" "AF2-MSAss" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --n_clusters 4 \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_blah \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb 

python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_plddt_ordered_all_filtered.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,/home/alexi/Documents/xFold_Sampling/af_sample/MoPrP_plddt_ordered.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_complete \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb 



python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_pH4_production_analysis/concatenated_downsampled.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_pH4_production_analysis/concatenated_downsampled_100.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_pH7_production_analysis/concatenated_downsampled.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_pH7_production_analysis/concatenated_downsampled_100.xtc" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb" "/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb,/home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb"  \
    --names "MD-pH4" "MD-pH7" "NMR-20C" "NMR-37C" \
    --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues_MD.json \
    --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_MD_complete \
    --save_pdbs \
    --json_feature_spec /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb


"""

import argparse
import datetime
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rms
import numpy as np
import seaborn as sns
from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
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
        description="Analyze and cluster molecular dynamics trajectories from multiple ensembles"
    )
    parser.add_argument(
        "--ensembles",
        nargs="+",
        type=str,
        required=True,
        help="List of 'topology,trajectory' pairs for each ensemble",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        required=False,
        help="Names for each ensemble (must match number of ensembles)",
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=False,
        help="Path to JSON file containing structural region definitions",
    )
    parser.add_argument(
        "--reference_pdb",
        type=str,
        required=False,
        help="Path to reference PDB file for calculating differences (optional)",
    )
    parser.add_argument(
        "--atom_selection",
        type=str,
        default="name CA",
        help='Atom selection string for PCA (default: "name CA")',
    )
    parser.add_argument(
        "--n_clusters", type=int, default=4, help="Number of clusters for k-means (default: 4)"
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
        help="Output directory (default: auto-generated based on ensemble names and time)",
    )
    parser.add_argument(
        "--save_pdbs",
        action="store_true",
        help="Save individual PDB files for each cluster (default: False)",
    )
    parser.add_argument(
        "--log", action="store_true", default=True, help="Enable logging (default: True)"
    )

    # Add feature selection arguments
    feature_group = parser.add_argument_group("Feature Selection")
    feature_group.add_argument(
        "--all_features",
        action="store_true",
        help="Calculate all available features (default behavior if no features specified)",
    )
    feature_group.add_argument("--rmsd", action="store_true", help="Calculate RMSD to reference")
    feature_group.add_argument("--rog", action="store_true", help="Calculate radius of gyration")
    feature_group.add_argument(
        "--sasa", action="store_true", help="Calculate solvent accessible surface area"
    )
    feature_group.add_argument(
        "--native_contacts", action="store_true", help="Calculate native contacts"
    )
    feature_group.add_argument(
        "--secondary_structure", action="store_true", help="Calculate secondary structure content"
    )
    feature_group.add_argument(
        "--json_feature_spec",
        type=str,
        default=None,
        help="Path to JSON file describing region-specific features to calculate.",
    )
    feature_group.add_argument(
        "--json_rules_spec",
        type=str,
        default=None,
        help="Path to JSON file describing rules for clustering.",
    )

    return parser.parse_args()


def validate_and_parse_ensembles(ensembles_list, names_list=None):
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

            # Handle ensemble names
            if names_list and i < len(names_list):
                ensemble_names.append(names_list[i])
            else:
                # Generate default name from topology filename
                base_name = os.path.splitext(os.path.basename(topology))[0]
                ensemble_names.append(f"Ensemble_{i + 1}_{base_name}")

        except ValueError:
            raise ValueError(f"Invalid ensemble format: {ensemble_str}. Use 'topology,trajectory'")

    if names_list and len(names_list) != len(ensembles):
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


def load_structural_regions(json_path):
    """Load structural region definitions from JSON file"""
    if not json_path or not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    return data.get("moPrP_data", {})


def parse_residue_range(residue_string):
    """Parse residue range string like '144-153' or '144-153,160-163' into list of (start, end) tuples"""
    if not residue_string or residue_string == "null":
        return []

    # Handle cases like "1-2 (partial)" or "Cys178-Cys213"
    if "(" in residue_string:
        residue_string = residue_string.split("(")[0].strip()

    ranges = []
    # Split by comma to handle multiple ranges
    range_parts = [part.strip() for part in residue_string.split(",")]

    for range_part in range_parts:
        if range_part.startswith("Cys"):
            # Handle disulfide bonds like "Cys178-Cys213"
            parts = range_part.split("-")
            if len(parts) == 2:
                start = int(parts[0][3:])  # Remove 'Cys' prefix
                end = int(parts[1][3:])  # Remove 'Cys' prefix
                ranges.append((start, end))
        else:
            try:
                if "-" in range_part:
                    start, end = map(int, range_part.split("-"))
                    ranges.append((start, end))
                else:
                    # Single residue
                    residue = int(range_part)
                    ranges.append((residue, residue))
            except ValueError:
                continue

    return ranges


def get_structural_regions(json_data, parse_all_elements=False):
    """Extract helix and strand regions from JSON data, handling multiple ranges per element"""
    if not json_data:
        return {}

    regions = {}
    structural_elements = json_data.get("structural_elements", [])
    mapping_offset = json_data.get("sequence_info", {}).get("mapping_offset", 122)

    for element in structural_elements:
        name = element["name"]

        # Focus on helices and strands, unless all elements are requested
        if parse_all_elements or (
            "helix" in name or "strand" in name or "Core" in name or "tail" in name.lower()
        ):
            original_range = element.get("original_residues")
            pdb_range = element.get("pdb_residues")

            # Parse original numbering (can be multiple ranges)
            orig_ranges = parse_residue_range(original_range)

            if orig_ranges:
                # Convert to PDB numbering
                pdb_ranges = []
                for orig_start, orig_end in orig_ranges:
                    pdb_start = orig_start - mapping_offset
                    pdb_end = orig_end - mapping_offset

                    # Ensure PDB residues are within valid range (1-101 based on structure)
                    pdb_ranges.append((pdb_start, pdb_end))

                if pdb_ranges:
                    regions[name] = {
                        "original_ranges": orig_ranges,
                        "pdb_ranges": pdb_ranges,
                        "type": "helix" if "helix" in name else "strand",
                    }

    return regions


def circular_distance_degrees(angle1, angle2):
    """Calculate the minimum angular distance between two angles in degrees.

    Handles the periodicity of angles correctly, assuming angles are in [-180, 180].
    Returns the absolute minimum angular distance.
    """
    diff = angle1 - angle2
    # Normalize to [-180, 180] range
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return abs(diff)


def calculate_dihedral_angle(pos1, pos2, pos3, pos4):
    """Calculate dihedral angle between four points"""
    # Vectors
    v1 = pos2 - pos1
    v2 = pos3 - pos2
    v3 = pos4 - pos3

    # Normal vectors to planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Calculate angle
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1, 1)

    # Check sign using scalar triple product
    sign = np.sign(np.dot(np.cross(n1, n2), v2))

    angle = sign * np.arccos(cos_angle)
    return np.degrees(angle)


def calculate_region_dihedral_features(universe, ref_universe, regions, use_reference=True):
    """Calculate mean phi and psi angle changes or absolute values for each region (separately), handling multiple ranges per region"""
    if use_reference:
        logger.info("Calculating region-specific phi and psi angle changes from reference...")
        feature_suffix_phi = "_phi_change"
        feature_suffix_psi = "_psi_change"
        units = "°"
    else:
        logger.info("Calculating region-specific absolute phi and psi angles...")
        feature_suffix_phi = "_phi_abs"
        feature_suffix_psi = "_psi_abs"
        units = "°"

    features = {}
    n_frames = len(universe.trajectory)

    for region_name, region_info in regions.items():
        pdb_ranges = region_info["pdb_ranges"]

        # Create selection for all ranges in this region
        selection_parts = []
        for pdb_start, pdb_end in pdb_ranges:
            selection_parts.append(f"resid {pdb_start}:{pdb_end}")

        selection = f"({' or '.join(selection_parts)}) and protein"

        try:
            region_atoms = universe.select_atoms(selection)
            if use_reference:
                ref_region_atoms = ref_universe.select_atoms(selection)

            if len(region_atoms) == 0:
                logger.warning(
                    f"No atoms found for region {region_name} with selection: {selection}"
                )
                continue

            # Get residues in region
            residues = region_atoms.residues
            if use_reference:
                ref_residues = ref_region_atoms.residues
                if len(residues) != len(ref_residues):
                    logger.warning(f"Residue count mismatch for region {region_name}")
                    continue

            region_phi_values = []
            region_psi_values = []

            for ts in tqdm(universe.trajectory, desc=f"Calculating dihedrals for {region_name}"):
                frame_phi_values = []
                frame_psi_values = []

                for i, res in enumerate(residues):
                    try:
                        # Calculate phi angle (need previous residue)
                        if i > 0:
                            prev_res = residues[i - 1]

                            # Current frame phi
                            phi_atoms = [
                                prev_res.atoms.select_atoms("name C")[0].position,
                                res.atoms.select_atoms("name N")[0].position,
                                res.atoms.select_atoms("name CA")[0].position,
                                res.atoms.select_atoms("name C")[0].position,
                            ]
                            phi_current = calculate_dihedral_angle(*phi_atoms)

                            if use_reference:
                                ref_res = ref_residues[i]
                                ref_prev_res = ref_residues[i - 1]
                                # Reference frame phi
                                phi_ref_atoms = [
                                    ref_prev_res.atoms.select_atoms("name C")[0].position,
                                    ref_res.atoms.select_atoms("name N")[0].position,
                                    ref_res.atoms.select_atoms("name CA")[0].position,
                                    ref_res.atoms.select_atoms("name C")[0].position,
                                ]
                                phi_ref = calculate_dihedral_angle(*phi_ref_atoms)
                                # Calculate circular distance for phi angle
                                phi_value = circular_distance_degrees(phi_current, phi_ref)
                            else:
                                phi_value = abs(phi_current)

                            frame_phi_values.append(phi_value)

                        # Calculate psi angle (need next residue)
                        if i < len(residues) - 1:
                            next_res = residues[i + 1]

                            # Current frame psi
                            psi_atoms = [
                                res.atoms.select_atoms("name N")[0].position,
                                res.atoms.select_atoms("name CA")[0].position,
                                res.atoms.select_atoms("name C")[0].position,
                                next_res.atoms.select_atoms("name N")[0].position,
                            ]
                            psi_current = calculate_dihedral_angle(*psi_atoms)

                            if use_reference:
                                ref_res = ref_residues[i]
                                ref_next_res = ref_residues[i + 1]
                                # Reference frame psi
                                psi_ref_atoms = [
                                    ref_res.atoms.select_atoms("name N")[0].position,
                                    ref_res.atoms.select_atoms("name CA")[0].position,
                                    ref_res.atoms.select_atoms("name C")[0].position,
                                    ref_next_res.atoms.select_atoms("name N")[0].position,
                                ]
                                psi_ref = calculate_dihedral_angle(*psi_ref_atoms)
                                # Calculate circular distance for psi angle
                                psi_value = circular_distance_degrees(psi_current, psi_ref)
                            else:
                                psi_value = abs(psi_current)

                            frame_psi_values.append(psi_value)

                    except (IndexError, AttributeError):
                        # Skip if backbone atoms are missing
                        continue

                # Calculate mean phi and psi values for this frame (separately)
                if frame_phi_values:
                    region_phi_values.append(np.mean(frame_phi_values))
                else:
                    region_phi_values.append(0.0)

                if frame_psi_values:
                    region_psi_values.append(np.mean(frame_psi_values))
                else:
                    region_psi_values.append(0.0)

            # Store phi and psi as separate features
            phi_feature_name = f"{region_name}{feature_suffix_phi}"
            psi_feature_name = f"{region_name}{feature_suffix_psi}"

            features[phi_feature_name] = np.array(region_phi_values)
            features[psi_feature_name] = np.array(region_psi_values)

            if use_reference:
                logger.info(
                    f"Calculated phi changes for {region_name}: mean = {np.mean(region_phi_values):.2f}{units}"
                )
                logger.info(
                    f"Calculated psi changes for {region_name}: mean = {np.mean(region_psi_values):.2f}{units}"
                )
            else:
                logger.info(
                    f"Calculated absolute phi for {region_name}: mean = {np.mean(region_phi_values):.2f}{units}"
                )
                logger.info(
                    f"Calculated absolute psi for {region_name}: mean = {np.mean(region_psi_values):.2f}{units}"
                )

        except Exception as e:
            logger.warning(f"Failed to calculate dihedral features for {region_name}: {e}")
            continue

    return features


def calculate_region_com_features(universe, ref_universe, regions, use_reference=True):
    """Calculate pairwise RMSD between CA atoms of different regions or absolute values, handling multiple ranges per region"""
    if use_reference:
        logger.info("Calculating region-specific pairwise CA RMSD differences from reference...")
        feature_prefix = "CA_RMSD_diff"
        units = "Å"
    else:
        logger.info("Calculating region-specific absolute pairwise CA RMSD...")
        feature_prefix = "CA_RMSD_abs"
        units = "Å"

    features = {}
    region_names = list(regions.keys())
    n_regions = len(region_names)

    if n_regions < 2:
        logger.warning("Need at least 2 regions for pairwise RMSD calculations")
        return features

    # Calculate CA positions for all regions and frames
    region_ca_data = {}
    ref_region_ca_data = {}

    for region_name, region_info in regions.items():
        pdb_ranges = region_info["pdb_ranges"]

        # Create selection for all ranges in this region
        selection_parts = []
        for pdb_start, pdb_end in pdb_ranges:
            selection_parts.append(f"resid {pdb_start}:{pdb_end}")

        selection = f"({' or '.join(selection_parts)}) and protein and name CA"

        try:
            region_atoms = universe.select_atoms(selection)
            if use_reference:
                ref_region_atoms = ref_universe.select_atoms(selection)

            if len(region_atoms) == 0:
                continue

            # Store reference positions if using reference
            if use_reference:
                ref_region_ca_data[region_name] = ref_region_atoms.positions.copy()

            # Calculate CA positions for each frame
            ca_positions_frames = []
            for ts in universe.trajectory:
                ca_positions_frames.append(region_atoms.positions.copy())
            region_ca_data[region_name] = np.array(ca_positions_frames)

        except Exception as e:
            logger.warning(f"Failed to get CA positions for region {region_name}: {e}")
            continue

    # Calculate pairwise RMSD between regions
    valid_regions = list(region_ca_data.keys())

    for i, region1 in enumerate(valid_regions):
        for j, region2 in enumerate(valid_regions):
            if i >= j:  # Only calculate upper triangle to avoid duplicates
                continue

            # Get CA positions for both regions
            region1_ca_frames = region_ca_data[region1]  # Shape: (n_frames, n_atoms1, 3)
            region2_ca_frames = region_ca_data[region2]  # Shape: (n_frames, n_atoms2, 3)

            if use_reference:
                ref_region1_ca = ref_region_ca_data[region1]  # Shape: (n_atoms1, 3)
                ref_region2_ca = ref_region_ca_data[region2]  # Shape: (n_atoms2, 3)

                # Calculate reference pairwise RMSD (mean distance between all CA pairs)
                ref_pairwise_distances = []
                for atom1_pos in ref_region1_ca:
                    for atom2_pos in ref_region2_ca:
                        ref_pairwise_distances.append(np.linalg.norm(atom1_pos - atom2_pos))
                ref_mean_pairwise_rmsd = np.mean(ref_pairwise_distances)

            # Calculate frame-by-frame pairwise RMSD
            frame_rmsd_values = []
            n_frames = len(region1_ca_frames)

            for frame_idx in range(n_frames):
                frame_region1_ca = region1_ca_frames[frame_idx]  # Shape: (n_atoms1, 3)
                frame_region2_ca = region2_ca_frames[frame_idx]  # Shape: (n_atoms2, 3)

                # Calculate pairwise distances for this frame
                frame_pairwise_distances = []
                for atom1_pos in frame_region1_ca:
                    for atom2_pos in frame_region2_ca:
                        frame_pairwise_distances.append(np.linalg.norm(atom1_pos - atom2_pos))
                frame_mean_pairwise_rmsd = np.mean(frame_pairwise_distances)

                if use_reference:
                    # Calculate difference from reference
                    rmsd_value = frame_mean_pairwise_rmsd - ref_mean_pairwise_rmsd
                else:
                    # Use absolute value
                    rmsd_value = frame_mean_pairwise_rmsd

                frame_rmsd_values.append(rmsd_value)

            feature_name = f"{feature_prefix}_{region1}_vs_{region2}"
            features[feature_name] = np.array(frame_rmsd_values)

            if use_reference:
                logger.info(
                    f"Calculated pairwise CA RMSD differences for {region1} vs {region2}: "
                    f"mean = {np.mean(frame_rmsd_values):.2f} {units}, "
                    f"ref_rmsd = {ref_mean_pairwise_rmsd:.2f} {units}"
                )
            else:
                logger.info(
                    f"Calculated absolute pairwise CA RMSD for {region1} vs {region2}: "
                    f"mean = {np.mean(frame_rmsd_values):.2f} {units}"
                )

    return features


def calculate_mean_internal_pairwise_rmsd(universe, ref_universe, regions, use_reference=True):
    """Calculate the mean internal pairwise RMSD of CA atoms within each region, handling multiple ranges per region."""
    if use_reference:
        logger.info("Calculating mean internal pairwise CA RMSD differences from reference...")
        feature_prefix = "Internal_CA_RMSD_diff"
        units = "Å"
    else:
        logger.info("Calculating absolute mean internal pairwise CA RMSD...")
        feature_prefix = "Internal_CA_RMSD_abs"
        units = "Å"

    features = {}
    n_frames = len(universe.trajectory)

    for region_name, region_info in regions.items():
        pdb_ranges = region_info["pdb_ranges"]

        # Create selection for all ranges in this region
        selection_parts = []
        for pdb_start, pdb_end in pdb_ranges:
            selection_parts.append(f"resid {pdb_start}:{pdb_end}")

        selection = f"({' or '.join(selection_parts)}) and protein and name CA"

        try:
            region_atoms = universe.select_atoms(selection)
            if len(region_atoms) < 2:  # Need at least 2 atoms for pairwise distances
                logger.warning(
                    f"Region {region_name} has fewer than 2 CA atoms. Skipping internal RMSD."
                )
                continue

            if use_reference:
                ref_region_atoms = ref_universe.select_atoms(selection)
                if len(region_atoms) != len(ref_region_atoms):
                    logger.warning(
                        f"Atom count mismatch for region {region_name} between universe and reference. Skipping."
                    )
                    continue

                # Calculate reference internal pairwise RMSD
                ref_positions = ref_region_atoms.positions
                ref_pairwise_distances = pdist(ref_positions, metric="euclidean")
                ref_mean_internal_pairwise_rmsd = np.mean(ref_pairwise_distances)

            frame_internal_rmsd_values = []
            for ts in tqdm(universe.trajectory, desc=f"Internal RMSD for {region_name}"):
                current_positions = region_atoms.positions
                current_pairwise_distances = pdist(current_positions, metric="euclidean")
                current_mean_internal_pairwise_rmsd = np.mean(current_pairwise_distances)

                if use_reference:
                    rmsd_value = (
                        current_mean_internal_pairwise_rmsd - ref_mean_internal_pairwise_rmsd
                    )
                else:
                    rmsd_value = current_mean_internal_pairwise_rmsd

                frame_internal_rmsd_values.append(rmsd_value)

            feature_name = f"{feature_prefix}_{region_name}"
            features[feature_name] = np.array(frame_internal_rmsd_values)

            if use_reference:
                logger.info(
                    f"Calculated internal CA RMSD differences for {region_name}: "
                    f"mean = {np.mean(frame_internal_rmsd_values):.2f} {units}, "
                    f"ref_rmsd = {ref_mean_internal_pairwise_rmsd:.2f} {units}"
                )
            else:
                logger.info(
                    f"Calculated absolute internal CA RMSD for {region_name}: "
                    f"mean = {np.mean(frame_internal_rmsd_values):.2f} {units}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to calculate internal pairwise RMSD for region {region_name}: {e}"
            )
            continue

    return features


def calculate_region_specific_features(universe, ref_universe, json_data, use_reference=True):
    """Calculate all region-specific features"""
    if not json_data:
        logger.warning("No JSON data provided for region-specific features")
        return {}, []

    # Get structural regions
    regions = get_structural_regions(json_data, parse_all_elements=True)

    if not regions:
        logger.warning("No valid regions found in JSON data")
        return {}, []

    logger.info(f"Found {len(regions)} structural regions: {list(regions.keys())}")

    # Calculate features
    all_features = {}

    # Dihedral angle features
    dihedral_features = calculate_region_dihedral_features(
        universe, ref_universe, regions, use_reference
    )
    all_features.update(dihedral_features)

    # CA RMSD features
    ca_rmsd_features = calculate_region_com_features(universe, ref_universe, regions, use_reference)
    all_features.update(ca_rmsd_features)

    # Internal pairwise RMSD features
    internal_rmsd_features = calculate_mean_internal_pairwise_rmsd(
        universe, ref_universe, regions, use_reference
    )
    all_features.update(internal_rmsd_features)

    # Prepare feature data and names
    if all_features:
        feature_names = list(all_features.keys())
        feature_data_list = [all_features[name] for name in feature_names]
        return feature_data_list, feature_names
    else:
        return [], []


def calculate_rmsd(universe, ref_universe, selection="name CA", use_reference=True):
    """Calculate RMSD for each frame relative to a reference structure or absolute values."""
    if use_reference:
        logger.info(f"Calculating RMSD differences from reference using selection: {selection}")
        ref_atoms = ref_universe.select_atoms(selection)
    else:
        logger.info(f"Calculating absolute RMSD using selection: {selection}")

    rmsd_values = []

    for ts in tqdm(universe.trajectory, desc="Calculating RMSD"):
        atoms = universe.select_atoms(selection)

        if use_reference:
            # Ensure both selections have the same number of atoms
            if len(atoms) != len(ref_atoms):
                logger.warning(
                    f"Frame {ts.frame}: Atom count mismatch for RMSD calculation. Skipping frame."
                )
                rmsd_values.append(np.nan)
                continue
            rmsd_values.append(mda.analysis.rms.rmsd(atoms.positions, ref_atoms.positions))
        else:
            # Calculate RMSD from first frame as reference
            if ts.frame == 0:
                first_frame_positions = atoms.positions.copy()
            rmsd_values.append(mda.analysis.rms.rmsd(atoms.positions, first_frame_positions))

    return np.array(rmsd_values)


def calculate_radius_of_gyration(universe, selection="all"):
    """Calculate Radius of Gyration for each frame."""
    logger.info(f"Calculating Radius of Gyration using selection: {selection}")
    rog_values = []
    for ts in tqdm(universe.trajectory, desc="Calculating RoG"):
        atoms = universe.select_atoms(selection)
        rog_values.append(atoms.radius_of_gyration())
    return np.array(rog_values)


def calculate_sasa(universe, selection="all"):
    """Calculate Solvent Accessible Surface Area (SASA) for each frame."""
    logger.info(f"Calculating SASA using selection: {selection}")
    R = SASAAnalysis(universe, select=selection, VdWradii=None, n_points=960, probe_radius=1.4)
    R.run()
    sasa_values = R.results.total_area
    return np.array(sasa_values)


def calculate_native_contacts(universe, ref_universe, selection="name CA", cutoff=4.5):
    """
    Calculate the fraction of native contacts for each frame using MDAnalysis.analysis.contacts.
    """
    from MDAnalysis.analysis import contacts

    logger.info(f"Calculating native contacts using selection: {selection} and cutoff: {cutoff} Å")

    # Get reference atoms from the first frame of ref_universe
    ref_atoms = ref_universe.select_atoms(selection)
    n_atoms = len(ref_atoms)

    logger.info(f"Selected {n_atoms} atoms for native contacts analysis")

    # Set up the contacts analysis
    ca = contacts.Contacts(
        universe, select=(selection, selection), refgroup=(ref_atoms, ref_atoms), radius=cutoff
    )

    # Run the analysis
    logger.info("Running native contacts analysis...")
    ca.run()

    # Extract the fraction of native contacts (Q values)
    native_contacts_fractions = ca.results.timeseries[:, 1]

    logger.info("Native contacts analysis complete.")
    logger.info(f"Average fraction of native contacts: {np.mean(native_contacts_fractions):.3f}")

    return native_contacts_fractions


def calculate_secondary_structure(universe, selection="protein"):
    """
    Calculate alpha helical and beta sheet content for each frame using DSSP.
    """
    logger.info(f"Calculating secondary structure content using selection: {selection}")
    from MDAnalysis.analysis.dssp import DSSP

    protein = universe.select_atoms(selection)
    n_residues = len(protein.residues)
    if n_residues == 0:
        logger.warning("No protein residues found for secondary structure calculation.")
        return np.zeros(len(universe.trajectory)), np.zeros(len(universe.trajectory))

    try:
        # Run DSSP analysis once for all frames
        dssp_analysis = DSSP(universe)
        dssp_analysis.run()

        # Get the DSSP results for all frames
        dssp_results = dssp_analysis.results.dssp

        alpha_helix_content = []
        beta_sheet_content = []

        # Process each frame's results
        for frame_idx, ss_string in enumerate(dssp_results):
            helix_count = sum(c in ("H", "G", "I") for c in ss_string)
            sheet_count = sum(c in ("E", "B") for c in ss_string)
            alpha_helix_content.append(helix_count / n_residues)
            beta_sheet_content.append(sheet_count / n_residues)

    except Exception as e:
        logger.warning(f"Secondary structure calculation failed: {e}")
        # Return zeros if DSSP fails
        n_frames = len(universe.trajectory)
        alpha_helix_content = [0.0] * n_frames
        beta_sheet_content = [0.0] * n_frames

    return np.array(alpha_helix_content), np.array(beta_sheet_content)


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


def calculate_features_from_spec(universe, ref_universe, regions, feature_spec, use_reference):
    """Calculate features based on a JSON specification."""
    all_features = {}
    feature_metadata = {}

    if not feature_spec or not regions:
        return {}, {}

    # Region Dihedrals
    for spec in feature_spec.get("region_dihedrals", []):
        region_name = spec.get("region_name")
        spec_name = spec.get("spec_name")
        if not region_name or not spec_name or region_name not in regions:
            logger.warning(f"Skipping invalid dihedral spec: {spec}")
            continue

        sub_regions = {region_name: regions[region_name]}
        use_ref = spec.get("reference", use_reference)

        dihedral_features = calculate_region_dihedral_features(
            universe, ref_universe, sub_regions, use_reference=use_ref
        )

        feature_suffix_phi = "_phi_change" if use_ref else "_phi_abs"
        feature_suffix_psi = "_psi_change" if use_ref else "_psi_abs"

        phi_key = f"{region_name}{feature_suffix_phi}"
        psi_key = f"{region_name}{feature_suffix_psi}"

        if phi_key in dihedral_features:
            phi_feature_name = f"{spec_name}_phi"
            all_features[phi_feature_name] = dihedral_features[phi_key]
            feature_metadata[phi_feature_name] = spec

        if psi_key in dihedral_features:
            psi_feature_name = f"{spec_name}_psi"
            all_features[psi_feature_name] = dihedral_features[psi_key]
            feature_metadata[psi_feature_name] = spec

    # Internal RMSD
    for spec in feature_spec.get("internal_rmsd", []):
        region_name = spec.get("region_name")
        spec_name = spec.get("spec_name")
        if not region_name or not spec_name or region_name not in regions:
            logger.warning(f"Skipping invalid internal RMSD spec: {spec}")
            continue

        sub_regions = {region_name: regions[region_name]}
        use_ref = spec.get("reference", use_reference)

        internal_rmsd_features = calculate_mean_internal_pairwise_rmsd(
            universe, ref_universe, sub_regions, use_reference=use_ref
        )

        if internal_rmsd_features:
            # Should be only one feature calculated
            all_features[spec_name] = list(internal_rmsd_features.values())[0]
            feature_metadata[spec_name] = spec

    # Region RMSD (pairwise)
    for spec in feature_spec.get("region_rmsd", []):
        region_names_str = spec.get("region_names")
        spec_name = spec.get("spec_name")
        if not region_names_str or not spec_name:
            logger.warning(f"Skipping invalid region RMSD spec: {spec}")
            continue

        region_names = [s.strip() for s in region_names_str.split(",")]
        if any(r not in regions for r in region_names):
            logger.warning(f"Invalid region name in region RMSD spec: {spec}")
            continue

        sub_regions = {r: regions[r] for r in region_names}
        use_ref = spec.get("reference", use_reference)

        com_features = calculate_region_com_features(
            universe, ref_universe, sub_regions, use_reference=use_ref
        )

        if com_features:
            # Should be only one feature calculated
            all_features[spec_name] = list(com_features.values())[0]
            feature_metadata[spec_name] = spec

    return all_features, feature_metadata


def perform_kmeans_clustering(pca_coords, n_clusters):
    """Perform k-means clustering on PCA coordinates"""
    logger.info(f"Performing k-means clustering with {n_clusters} clusters...")

    # Use MiniBatchKMeans for large datasets
    if len(pca_coords) > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=1000, n_init="auto"
        )
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    cluster_labels = kmeans.fit_predict(pca_coords)
    cluster_centers = kmeans.cluster_centers_

    # Count frames per cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    logger.info(f"Clustering complete: {len(unique_labels)} clusters")
    logger.info(f"Average frames per cluster: {np.mean(counts):.1f}")
    logger.info(f"Min frames per cluster: {np.min(counts)}")
    logger.info(f"Max frames per cluster: {np.max(counts)}")

    return cluster_labels, cluster_centers, kmeans


def perform_rules_based_clustering(all_features_data, feature_names, rules_spec):
    """Perform clustering based on a set of rules defined in a JSON file."""
    logger.info("Performing rules-based clustering...")
    n_frames = all_features_data.shape[0]
    cluster_labels = np.full(n_frames, -1, dtype=int)

    # Create a mapping from feature name to its column index
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Sort cluster IDs to ensure priority (e.g., "0" before "1")
    sorted_cluster_ids = sorted(rules_spec.keys(), key=int)

    for cluster_id_str in sorted_cluster_ids:
        cluster_id = int(cluster_id_str)
        rules = rules_spec[cluster_id_str]

        # Start with a mask where all frames are potential candidates
        cluster_mask = np.ones(n_frames, dtype=bool)

        for rule in rules:
            spec_name = rule.get("spec_name")
            min_thresh = rule.get("min_threshold")
            max_thresh = rule.get("max_threshold")
            inclusive = rule.get("inclusive", True)  # Default to inclusive

            if spec_name not in feature_to_idx:
                logger.warning(
                    f"Feature '{spec_name}' in rules not found in calculated features. Skipping rule."
                )
                continue

            feature_idx = feature_to_idx[spec_name]
            feature_values = all_features_data[:, feature_idx]

            if inclusive:
                # Inclusive: value >= min and value <= max
                rule_mask = (feature_values >= min_thresh) & (feature_values <= max_thresh)
            else:
                # Exclusive: value < min or value > max
                rule_mask = (feature_values < min_thresh) | (feature_values > max_thresh)

            # A frame must satisfy ALL rules for a cluster, so AND the masks
            cluster_mask &= rule_mask

        # Assign cluster ID to frames that satisfy all rules for this cluster
        # and have not yet been assigned a cluster from a higher-priority rule
        assign_mask = cluster_mask & (cluster_labels == -1)
        cluster_labels[assign_mask] = cluster_id

    # Log the results
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    logger.info("Rules-based clustering complete.")
    for label, count in zip(unique_labels, counts):
        logger.info(f"Cluster {label}: {count} frames")

    return cluster_labels


def create_feature_distribution_plots(
    all_features_data,
    feature_names,
    ensemble_names,
    ensemble_labels,
    output_dir,
    feature_metadata=None,
):
    """Create distribution plots for each feature. Dihedral (phi/psi) plots for the same region are combined into a single figure with two panels."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating feature distribution plots with mean, median, variance, and range...")

    if feature_metadata is None:
        feature_metadata = {}

    processed_features = set()

    for i, name in enumerate(feature_names):
        if name in processed_features:
            continue

        # Check for dihedral pairs
        is_phi = "_phi" in name
        is_psi = "_psi" in name
        phi_name, psi_name = None, None

        if is_phi:
            phi_name = name
            psi_name = name.replace("_phi", "_psi")
            if psi_name not in feature_names:
                psi_name = None
        elif is_psi:
            phi_name_candidate = name.replace("_psi", "_phi")
            if phi_name_candidate in feature_names:
                continue  # Will be handled by phi
            # Orphan psi
            phi_name = None
            psi_name = name

        if phi_name and psi_name:
            # Paired dihedral plot
            processed_features.add(phi_name)
            processed_features.add(psi_name)
            phi_idx = feature_names.index(phi_name)
            psi_idx = feature_names.index(psi_name)

            fig, axes = plt.subplots(1, 2, figsize=(24, 8))

            # Plot Phi
            for j, ensemble_name in enumerate(ensemble_names):
                ensemble_mask = ensemble_labels == j
                ensemble_data = all_features_data[ensemble_mask, phi_idx]
                ensemble_color = get_ensemble_color(ensemble_name, j)
                mean_val, median_val, std_val = (
                    np.mean(ensemble_data),
                    np.median(ensemble_data),
                    np.std(ensemble_data),
                )
                var_val = np.var(ensemble_data)
                range_val = np.max(ensemble_data) - np.min(ensemble_data)
                sns.histplot(
                    ensemble_data,
                    alpha=0.4,
                    label=f"{ensemble_name} (μ={mean_val:.2f}, M={median_val:.2f}, σ²={var_val:.2f}, R={range_val:.2f})",
                    color=ensemble_color,
                    stat="density",
                    ax=axes[0],
                )
                # Add highlighting
                spec = feature_metadata.get(phi_name)
                if spec:
                    highlight_range = spec.get("plot_highlight_range")
                    if highlight_range is not None:
                        use_median = spec.get("use_median", False)
                        center = np.median(ensemble_data) if use_median else np.mean(ensemble_data)
                        low = center - highlight_range
                        high = center + highlight_range
                        axes[0].axvspan(low, high, color=ensemble_color, alpha=0.1)

            axes[0].set_title(f"Distribution of {phi_name}", fontsize=16)
            axes[0].set_xlabel(phi_name, fontsize=14)
            axes[0].set_ylabel("Density", fontsize=14)
            axes[0].legend()

            # Plot Psi
            for j, ensemble_name in enumerate(ensemble_names):
                ensemble_mask = ensemble_labels == j
                ensemble_data = all_features_data[ensemble_mask, psi_idx]
                ensemble_color = get_ensemble_color(ensemble_name, j)
                mean_val, median_val, std_val = (
                    np.mean(ensemble_data),
                    np.median(ensemble_data),
                    np.std(ensemble_data),
                )
                var_val = np.var(ensemble_data)
                range_val = np.max(ensemble_data) - np.min(ensemble_data)
                sns.histplot(
                    ensemble_data,
                    alpha=0.4,
                    label=f"{ensemble_name} (μ={mean_val:.2f}, M={median_val:.2f}, σ²={var_val:.2f}, R={range_val:.2f})",
                    color=ensemble_color,
                    stat="density",
                    ax=axes[1],
                )
                # Add highlighting
                spec = feature_metadata.get(psi_name)
                if spec:
                    highlight_range = spec.get("plot_highlight_range")
                    if highlight_range is not None:
                        use_median = spec.get("use_median", False)
                        center = np.median(ensemble_data) if use_median else np.mean(ensemble_data)
                        low = center - highlight_range
                        high = center + highlight_range
                        axes[1].axvspan(low, high, color=ensemble_color, alpha=0.1)

            axes[1].set_title(f"Distribution of {psi_name}", fontsize=16)
            axes[1].set_xlabel(psi_name, fontsize=14)
            axes[1].set_ylabel("Density", fontsize=14)
            axes[1].legend()

            base_name = phi_name.split("_phi")[0]
            fig.suptitle(f"Distribution of Dihedrals for {base_name}", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            save_path = os.path.join(
                plots_dir, f"distribution_dihedrals_{base_name.replace(' ', '_')}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        else:
            # Single feature plot
            processed_features.add(name)
            fig = plt.figure(figsize=(16, 8))
            ax = plt.gca()
            ensemble_stats = []
            for j, ensemble_name in enumerate(ensemble_names):
                ensemble_mask = ensemble_labels == j
                ensemble_data = all_features_data[ensemble_mask, i]
                ensemble_color = get_ensemble_color(ensemble_name, j)
                mean_val, median_val, std_val = (
                    np.mean(ensemble_data),
                    np.median(ensemble_data),
                    np.std(ensemble_data),
                )
                var_val = np.var(ensemble_data)
                range_val = np.max(ensemble_data) - np.min(ensemble_data)
                ensemble_stats.append((ensemble_name, mean_val, median_val, var_val, range_val))
                sns.histplot(
                    ensemble_data,
                    alpha=0.4,
                    label=f"{ensemble_name} (μ={mean_val:.2f}, M={median_val:.2f}, σ²={var_val:.2f}, R={range_val:.2f})",
                    color=ensemble_color,
                    stat="density",
                    ax=ax,
                )
                # Add highlighting
                spec = feature_metadata.get(name)
                if spec:
                    highlight_range = spec.get("plot_highlight_range")
                    if highlight_range is not None:
                        use_median = spec.get("use_median", False)
                        center = np.median(ensemble_data) if use_median else np.mean(ensemble_data)
                        low = center - highlight_range
                        high = center + highlight_range
                        ax.axvspan(low, high, color=ensemble_color, alpha=0.1)

            plt.title(f"Distribution of {name} - All Ensembles", fontsize=16)
            plt.xlabel(name, fontsize=14)
            plt.ylabel("Density", fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            stats_text = "Statistics:\n" + "\n".join(
                [
                    f"{ens}: μ={mean:.3f}, M={median:.3f}, σ²={var:.3f}, R={range:.3f}"
                    for ens, mean, median, var, range in ensemble_stats
                ]
            )
            plt.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            plt.tight_layout()
            save_path = os.path.join(plots_dir, f"distribution_{name.replace(' ', '_')}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    logger.info("Feature distribution plots with mean, median, variance, and range created.")


def create_pairwise_scatter_plots(
    all_features_data, feature_names, ensemble_names, ensemble_labels, output_dir
):
    """Create pairwise scatter plots of features with separate panels per ensemble."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating pairwise scatter plots...")

    import pandas as pd

    n_ensembles = len(ensemble_names)

    # Create subplots for each ensemble
    fig, axes = plt.subplots(1, n_ensembles, figsize=(6 * n_ensembles, 6))
    if n_ensembles == 1:
        axes = [axes]

    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        ensemble_data = all_features_data[ensemble_mask]
        ensemble_color = get_ensemble_color(ensemble_name, j)

        df = pd.DataFrame(ensemble_data, columns=feature_names)

        # Use the specific subplot
        ax = axes[j]
        pair_plot = sns.pairplot(
            df, diag_kind="kde", plot_kws={"color": ensemble_color, "alpha": 0.6}
        )
        pair_plot.fig.suptitle(
            f"Pairwise Feature Scatter Plots - {ensemble_name}", y=1.02, fontsize=16
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plots_dir, f"pairwise_scatter_plots_{ensemble_name.replace(' ', '_')}.png"
            ),
            dpi=300,
        )
        plt.close()

    logger.info("Pairwise scatter plots created.")


def create_correlation_matrix_plot(
    all_features_data, feature_names, ensemble_names, ensemble_labels, output_dir
):
    """Create correlation matrix plots with separate panels per ensemble."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating correlation matrix plots...")

    import pandas as pd

    n_ensembles = len(ensemble_names)
    fig, axes = plt.subplots(1, n_ensembles, figsize=(10 * n_ensembles, 10))
    if n_ensembles == 1:
        axes = [axes]

    for j, ensemble_name in enumerate(ensemble_names):
        ensemble_mask = ensemble_labels == j
        ensemble_data = all_features_data[ensemble_mask]

        df = pd.DataFrame(ensemble_data, columns=feature_names)
        corr_matrix = df.corr()

        ax = axes[j]
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title(f"Feature Correlation Matrix - {ensemble_name}", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrices.png"), dpi=300)
    plt.close()
    logger.info("Correlation matrix plots created.")


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

    # Perform PCA on all features together
    pca_features = IncrementalPCA(n_components=min(len(feature_names), 10))
    pca_features_coords = pca_features.fit_transform(features_data_cleaned)

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
    plt.savefig(os.path.join(plots_dir, "pca_features.png"), dpi=300)
    plt.close()

    logger.info("PCA of features plots created.")
    return os.path.join(plots_dir, "pca_features.png")


def create_publication_plots(
    all_pca_coords,
    ensemble_names,
    ensemble_labels,
    cluster_labels,
    cluster_centers,
    pca,
    output_dir,
):
    """Create publication-quality plots of the PCA results and clustering with separate panels per ensemble"""
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


def create_cluster_ratio_plot(cluster_labels, ensemble_names, ensemble_labels, output_dir):
    """Create plots showing the distribution ratio of frames across clusters, split by ensemble."""
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
):
    """Save cluster trajectories and optionally save representative PDB files"""
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
        center = cluster_centers[cluster_idx]

        # Find frame closest to cluster center (medoid)
        distances = np.sqrt(np.sum((cluster_pca_coords - center) ** 2, axis=1))
        min_dist_idx = np.argmin(distances)
        representative_frame_idx = cluster_frame_indices[min_dist_idx]

        # Determine which ensemble and local frame index
        ensemble_idx = ensemble_labels[representative_frame_idx]

        # Calculate local frame index within the ensemble
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

        # Create ensemble-specific directory
        ensemble_clusters_dir = os.path.join(clusters_dir, ensemble_safe_name)
        os.makedirs(ensemble_clusters_dir, exist_ok=True)

        # Get frames belonging to this ensemble
        ensemble_mask = ensemble_labels == ensemble_idx
        ensemble_cluster_labels = cluster_labels[ensemble_mask]
        ensemble_frame_indices = np.where(ensemble_mask)[0]

        # Save ensemble-specific trajectory with all cluster centers from this ensemble
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

        # Save frame-to-cluster mapping for this ensemble
        mapping_file = os.path.join(
            ensemble_clusters_dir, f"{ensemble_safe_name}_frame_to_cluster.csv"
        )
        with open(mapping_file, "w") as f:
            f.write("local_frame_index,global_frame_index,cluster_label\n")
            for local_idx, (global_idx, cluster_label) in enumerate(
                zip(ensemble_frame_indices, ensemble_cluster_labels)
            ):
                f.write(f"{local_idx},{global_idx},{cluster_label}\n")

        # Save PDB files if requested
        if save_pdbs:
            logger.info(f"Saving multi-frame PDB files for ensemble: {ensemble_name}")

            # Create PDB directory for this ensemble
            pdb_dir = os.path.join(ensemble_clusters_dir, "pdbs")
            os.makedirs(pdb_dir, exist_ok=True)

            # Save medoid/centroid frames for each cluster
            medoids_file = os.path.join(pdb_dir, f"{ensemble_safe_name}_cluster_medoids.pdb")
            with mda.Writer(medoids_file, multiframe=True) as writer:
                for cluster_idx in unique_clusters:
                    if cluster_idx in representative_frames:
                        rep_info = representative_frames[cluster_idx]
                        if rep_info["ensemble_idx"] == ensemble_idx:
                            universe.trajectory[rep_info["local_frame_idx"]]
                            # Add cluster info to remarks
                            writer.write(universe.atoms)

            logger.info(f"Saved cluster medoids PDB: {medoids_file}")

            # Save multi-frame PDB for each cluster containing frames from this ensemble
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

                with mda.Writer(cluster_pdb_file, multiframe=True) as writer:
                    for local_frame_idx in tqdm(
                        cluster_local_indices,
                        desc=f"Saving cluster {cluster_idx} frames for {ensemble_name}",
                    ):
                        universe.trajectory[local_frame_idx]
                        writer.write(universe.atoms)

                logger.info(
                    f"Saved {n_cluster_frames} frames for cluster {cluster_idx}: {cluster_pdb_file}"
                )

            # Save individual medoid PDB files for each cluster
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


def get_feature_type(feature_name):
    """Categorize feature name into a type."""
    name_lower = feature_name.lower()
    if "rmsd" in name_lower:
        if "internal" in name_lower:
            return "Internal RMSD"
        elif "diff" in name_lower or "abs" in name_lower:
            return "COM RMSD"
        else:
            return "RMSD"
    elif "phi" in name_lower or "psi" in name_lower:
        return "Dihedral"
    elif "gyration" in name_lower:
        return "Radius of Gyration"
    elif "sasa" in name_lower:
        return "SASA"
    elif "native_contacts" in name_lower:
        return "Native Contacts"
    elif "helix" in name_lower or "sheet" in name_lower:
        return "Secondary Structure"
    else:
        return "Other"


def save_feature_statistics(
    all_features_data, feature_names, ensemble_names, ensemble_labels, output_dir
):
    """Calculate and save summary statistics of features to a CSV file."""
    import pandas as pd

    logger.info("Calculating and saving feature statistics...")
    stats_list = []
    for i, name in enumerate(feature_names):
        feature_type = get_feature_type(name)
        for j, ensemble_name in enumerate(ensemble_names):
            ensemble_mask = ensemble_labels == j
            ensemble_data = all_features_data[ensemble_mask, i]

            mean_val = np.mean(ensemble_data)
            median_val = np.median(ensemble_data)
            var_val = np.var(ensemble_data)
            range_val = np.max(ensemble_data) - np.min(ensemble_data)

            stats_list.append(
                {
                    "Ensemble": ensemble_name,
                    "Feature": name,
                    "Feature Type": feature_type,
                    "Mean": mean_val,
                    "Median": median_val,
                    "Variance": var_val,
                    "Range": range_val,
                }
            )

    stats_df = pd.DataFrame(stats_list)

    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_summary_statistics.csv")
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Feature statistics saved to {csv_path}")

    return stats_df


def plot_feature_statistics_tables(stats_df, output_dir):
    """Create and save plots of feature statistics tables, with each feature in its own table,
    rows colored by ensemble, and features of the same type grouped into panels."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating feature statistics table plots...")

    # Group by feature type first
    grouped_by_type = stats_df.groupby("Feature Type")

    for feature_type, type_group in grouped_by_type:
        # Get unique features within this type
        unique_features = type_group["Feature"].unique()
        n_features = len(unique_features)

        if n_features == 0:
            continue

        # Calculate subplot layout (prefer more columns than rows)
        n_cols = min(3, n_features)  # Max 3 columns
        n_rows = (n_features + n_cols - 1) // n_cols

        # Create figure with subplots for each feature
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))

        # Flatten axes array for easier indexing
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_features > 1 else [axes]
        else:
            axes = axes.flatten()

        # Create table for each feature
        for idx, feature_name in enumerate(unique_features):
            feature_data = type_group[type_group["Feature"] == feature_name].copy()

            # Prepare data for the table (drop Feature and Feature Type columns)
            table_data = feature_data[["Ensemble", "Mean", "Median", "Variance", "Range"]].round(3)

            ax = axes[idx]
            ax.axis("tight")
            ax.axis("off")

            # Create the table
            table = ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc="center",
                loc="center",
            )

            # Style the table
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Color rows by ensemble
            for row_idx in range(len(table_data)):
                ensemble_name = table_data.iloc[row_idx]["Ensemble"]
                ensemble_idx = list(stats_df["Ensemble"].unique()).index(ensemble_name)
                row_color = get_ensemble_color(ensemble_name, ensemble_idx)

                # Color all cells in this row (including header row adjustment)
                for col_idx in range(len(table_data.columns)):
                    table[(row_idx + 1, col_idx)].set_facecolor(row_color)
                    table[(row_idx + 1, col_idx)].set_alpha(0.3)

            # Style header row
            for col_idx in range(len(table_data.columns)):
                table[(0, col_idx)].set_facecolor("#E8E8E8")
                table[(0, col_idx)].set_text_props(weight="bold")

            # Add title for this feature
            ax.set_title(f"{feature_name}", fontsize=12, fontweight="bold", pad=20)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis("off")

        # Add overall title for the feature type
        fig.suptitle(f"Summary Statistics for {feature_type} Features", fontsize=16, y=0.98)

        # Add legend for ensemble colors
        unique_ensembles = stats_df["Ensemble"].unique()
        legend_elements = []
        for ens_idx, ensemble_name in enumerate(unique_ensembles):
            color = get_ensemble_color(ensemble_name, ens_idx)
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, label=ensemble_name)
            )

        # Add legend to the figure
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
            title="Ensembles",
            title_fontsize=12,
        )

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])

        save_path = os.path.join(plots_dir, f"stats_tables_{feature_type.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Created statistics tables for {feature_type}: {save_path}")

    logger.info("Feature statistics table plots created.")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Start timer
    start_time = datetime.datetime.now()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    global logger
    log_file = os.path.join(args.output_dir, "multi_ensemble_analysis.log")
    logger = setup_logger(log_file)

    # Validate and parse ensemble inputs
    try:
        ensembles, ensemble_names = validate_and_parse_ensembles(args.ensembles, args.names)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Check if reference PDB is provided
    use_reference = args.reference_pdb is not None
    if use_reference and not os.path.exists(args.reference_pdb):
        logger.error(f"Reference PDB file not found: {args.reference_pdb}")
        sys.exit(1)

    # Set up output directory
    if args.output_dir is None:
        ensemble_names_str = "_".join(ensemble_names[:3])  # Use first 3 names
        if len(ensemble_names) > 3:
            ensemble_names_str += "_etc"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            f"multi_ensemble_analysis_{ensemble_names_str}_clusters{args.n_clusters}_{timestamp}"
        )

    logger.info("=" * 80)
    logger.info("Starting multi-ensemble trajectory analysis and clustering")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Processing {len(ensembles)} ensembles: {ensemble_names}")
    logger.info(
        f"Ensembles: {[(name, top, traj) for (top, traj), name in zip(ensembles, ensemble_names)]}"
    )

    # Load structural data from JSON if provided
    json_data = load_structural_regions(args.json_data_path) if args.json_data_path else None
    feature_spec = load_feature_spec(args.json_feature_spec)
    rules_spec = load_rules_spec(args.json_rules_spec)

    # Load universes for all ensembles
    ensemble_data = []
    all_pca_coords = []
    ensemble_labels = []
    global_frame_idx = 0

    for i, ((topology, trajectory), name) in enumerate(zip(ensembles, ensemble_names)):
        logger.info(f"Loading ensemble {i + 1}: {name}")
        logger.info(f"  Topology: {topology}")
        logger.info(f"  Trajectory: {trajectory}")

        universe = mda.Universe(topology, trajectory)

        # Use reference PDB if provided, otherwise use first frame
        if use_reference:
            ref_universe = mda.Universe(args.reference_pdb)
            logger.info(f"  Using reference PDB: {args.reference_pdb}")
        else:
            ref_universe = mda.Universe(topology)  # Reference structure (first frame)
            logger.info("  Using first frame as reference")

        n_frames = len(universe.trajectory)

        logger.info(f"  Loaded {len(universe.atoms)} atoms and {n_frames} frames")

        # Store ensemble information
        ensemble_info = {
            "universe": universe,
            "ref_universe": ref_universe,
            "name": name,
            "n_frames": n_frames,
            "start_frame_idx": global_frame_idx,
            "end_frame_idx": global_frame_idx + n_frames,
        }
        ensemble_data.append(ensemble_info)

        # Create ensemble labels for this ensemble
        ensemble_labels.extend([i] * n_frames)
        global_frame_idx += n_frames

    # Convert ensemble labels to numpy array
    ensemble_labels = np.array(ensemble_labels)
    total_frames = len(ensemble_labels)
    logger.info(f"Total frames across all ensembles: {total_frames}")
    if use_reference:
        logger.info("Using reference structure for difference calculations")
    else:
        logger.info("Calculating absolute values (no reference structure)")

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

    # --- Feature Selection ---
    feature_flags = {
        "rmsd": args.rmsd,
        "rog": args.rog,
        "sasa": args.sasa,
        "native_contacts": args.native_contacts,
        "secondary_structure": args.secondary_structure,
    }

    if not any(feature_flags.values()) and not args.json_feature_spec or args.all_features:
        for key in feature_flags:
            feature_flags[key] = True
        logger.info("Calculating all available simple features for all ensembles.")

    # Calculate features for each ensemble
    logger.info("Calculating features for all ensembles...")
    all_features_data = []
    feature_names = []
    feature_metadata = {}

    for i, ensemble_info in enumerate(ensemble_data):
        universe = ensemble_info["universe"]
        ref_universe = ensemble_info["ref_universe"]
        name = ensemble_info["name"]

        logger.info(f"Calculating features for ensemble: {name}")

        features_dict = {}

        if feature_flags["rmsd"]:
            rmsd_values = calculate_rmsd(
                universe, ref_universe, selection="protein and name CA", use_reference=use_reference
            )
            features_dict["RMSD"] = rmsd_values

        if feature_flags["rog"]:
            rog_values = calculate_radius_of_gyration(universe, selection="protein")
            features_dict["Radius of Gyration"] = rog_values

        if feature_flags["sasa"]:
            sasa_values = calculate_sasa(universe, selection="protein")
            features_dict["SASA"] = sasa_values

        if feature_flags["native_contacts"]:
            native_contacts_values = calculate_native_contacts(
                universe, ref_universe, selection="protein and name CA"
            )
            features_dict["Native Contacts"] = native_contacts_values

        if feature_flags["secondary_structure"]:
            alpha_helix_content, beta_sheet_content = calculate_secondary_structure(
                universe, selection="protein"
            )
            features_dict["Alpha Helix Content"] = alpha_helix_content
            features_dict["Beta Sheet Content"] = beta_sheet_content

        if args.json_feature_spec:
            if not json_data:
                logger.warning(
                    "--json-feature-spec provided but no --json_data_path for region definitions. Skipping."
                )
            else:
                regions = get_structural_regions(json_data, parse_all_elements=True)
            features_dict, feature_metadata = calculate_features_from_spec(
                universe, ref_universe, regions, feature_spec, use_reference
            )

            # Center features according to specification (per-ensemble centering)
            for feature_name, spec in feature_metadata.items():
                centre_method = spec.get("centre")
                if centre_method and feature_name in features_dict:
                    feature_values = features_dict[feature_name]
                    if centre_method == "mean":
                        center_val = np.mean(feature_values)
                        features_dict[feature_name] -= center_val
                        logger.info(
                            f"  Ensemble {name}: Centered feature {feature_name} by mean: {center_val:.4f}"
                        )
                    elif centre_method == "median":
                        center_val = np.median(feature_values)
                        features_dict[feature_name] -= center_val
                        logger.info(
                            f"  Ensemble {name}: Centered feature {feature_name} by median: {center_val:.4f}"
                        )

            feature_names = list(features_dict.keys())
            features_data_list = list(features_dict.values())

        if not feature_names:
            feature_names = list(features_dict.keys())
        elif feature_names != list(features_dict.keys()):
            logger.warning(
                f"Feature names mismatch for ensemble {name}. Using first ensemble's names."
            )

        if features_dict:
            # Ensure consistent feature order
            ensemble_features = np.array([features_dict[fname] for fname in feature_names]).T
            all_features_data.append(ensemble_features)
        else:
            logger.warning(f"No features calculated for ensemble {name}")

    if not all_features_data:
        logger.error("No features were calculated for any ensemble.")
        sys.exit(1)

    # Combine all features
    all_features_data = np.vstack(all_features_data)
    logger.info(f"Combined features shape: {all_features_data.shape}")
    logger.info(f"Features: {', '.join(feature_names)}")

    # Save and plot feature statistics
    if len(feature_names) > 0:
        stats_df = save_feature_statistics(
            all_features_data, feature_names, ensemble_names, ensemble_labels, args.output_dir
        )
        plot_feature_statistics_tables(stats_df, args.output_dir)

    # Normalize features before clustering
    features_data_cleaned = np.nan_to_num(
        all_features_data, nan=np.nanmean(all_features_data, axis=0)
    )
    features_scaled = StandardScaler().fit_transform(features_data_cleaned)

    # Perform clustering
    if rules_spec:
        cluster_labels = perform_rules_based_clustering(
            all_features_data, feature_names, rules_spec
        )
        # Rules-based clustering doesn't have centers in the same way k-means does.
        # We can calculate them as the mean of the features for each cluster.
        n_clusters = len(np.unique(cluster_labels))
        cluster_centers = np.array(
            [
                features_scaled[cluster_labels == i].mean(axis=0)
                for i in sorted(np.unique(cluster_labels))
                if i != -1  # Exclude unclassified points if any
            ]
        )
    else:
        # Perform k-means clustering on the features
        logger.info(
            f"Performing k-means clustering on combined features with {args.n_clusters} clusters..."
        )
        cluster_labels, cluster_centers, kmeans_features = perform_kmeans_clustering(
            features_scaled, args.n_clusters
        )

    # Create plots
    logger.info("Creating plots...")

    # Feature distribution plots (all ensembles in same panel)
    if len(feature_names) > 0:
        create_feature_distribution_plots(
            all_features_data,
            feature_names,
            ensemble_names,
            ensemble_labels,
            args.output_dir,
            feature_metadata=feature_metadata,
        )

        # Pairwise and correlation plots (separate panels per ensemble)
        if len(feature_names) > 1:
            create_pairwise_scatter_plots(
                all_features_data, feature_names, ensemble_names, ensemble_labels, args.output_dir
            )
            create_correlation_matrix_plot(
                all_features_data, feature_names, ensemble_names, ensemble_labels, args.output_dir
            )

        # PCA of features (separate panels per ensemble)
        if len(feature_names) >= 2:
            create_pca_feature_plot(
                all_features_data,
                feature_names,
                ensemble_names,
                ensemble_labels,
                args.output_dir,
                cluster_labels=cluster_labels,
            )

    # PCA of pairwise coordinates (separate panels per ensemble)
    # Use the PCA from the first ensemble for variance explanation
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
    )

    # Save feature data with ensemble-specific names
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save global data
    np.save(os.path.join(data_dir, "all_features_data.npy"), all_features_data)
    np.save(os.path.join(data_dir, "ensemble_labels.npy"), ensemble_labels)
    np.save(os.path.join(data_dir, "cluster_labels.npy"), cluster_labels)
    np.save(os.path.join(data_dir, "all_pca_coords.npy"), all_pca_coords)

    with open(os.path.join(data_dir, "feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    with open(os.path.join(data_dir, "ensemble_names.txt"), "w") as f:
        for name in ensemble_names:
            f.write(f"{name}\n")

    # Save ensemble-specific feature data
    for ensemble_idx, ensemble_info in enumerate(ensemble_data):
        ensemble_name = ensemble_info["name"]
        ensemble_safe_name = ensemble_name.replace(" ", "_").replace("-", "_")

        # Create ensemble-specific data directory
        ensemble_data_dir = os.path.join(data_dir, ensemble_safe_name)
        os.makedirs(ensemble_data_dir, exist_ok=True)

        # Get data for this ensemble
        ensemble_mask = ensemble_labels == ensemble_idx
        ensemble_features = all_features_data[ensemble_mask]
        ensemble_pca_coords = all_pca_coords[ensemble_mask]
        ensemble_cluster_labels = cluster_labels[ensemble_mask]

        # Save ensemble-specific data
        np.save(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_features.npy"), ensemble_features
        )
        np.save(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_pca_coords.npy"),
            ensemble_pca_coords,
        )
        np.save(
            os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_cluster_labels.npy"),
            ensemble_cluster_labels,
        )

        # Save ensemble info
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
        }

        with open(os.path.join(ensemble_data_dir, f"{ensemble_safe_name}_info.json"), "w") as f:
            json.dump(ensemble_info_dict, f, indent=2)

        logger.info(f"Saved ensemble-specific data for {ensemble_name} in {ensemble_data_dir}")

    # End timer and report
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    logger.info("=" * 80)
    logger.info(f"Multi-ensemble analysis complete. Results saved to {args.output_dir}")
    logger.info(f"Processed {len(ensembles)} ensembles with {total_frames} total frames")
    logger.info(f"Found {len(np.unique(cluster_labels))} clusters")
    logger.info(f"Total execution time: {elapsed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
