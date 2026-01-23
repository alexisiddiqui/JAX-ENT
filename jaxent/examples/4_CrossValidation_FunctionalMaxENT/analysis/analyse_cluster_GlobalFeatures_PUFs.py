"""
This script analyzes the features distributions relevant for clustering the trajectory provided into folded, PUF1, PUF2 and unfolded states.

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

# Regions analyzed:
- α1, α2, α3 helices
- β1, β2 strands

Each per frame feature is saved in a separate .npy file.

python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_GlobalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc --n_clusters 4 --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_Globalfeatures_clusters4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --region_features --save_pdbs --rmsd --secondary_structure

python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_GlobalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc --n_clusters 4 --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_Globalfeatures_clusters4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --save_pdbs --secondary_structure


python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_GlobalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc --n_clusters 4 --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_Globalfeatures_clusters4 --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --save_pdbs --region_features

####

python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_GlobalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc --n_clusters 4 --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_Globalfeatures_clusters4_MSAss --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --region_features --save_pdbs --rmsd --secondary_structure

python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_GlobalFeatures_PUFs.py --topology_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb --trajectory_paths /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc --n_clusters 4 --output_dir /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_Globalfeatures_clusters4_MSAss --json_data_path /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json --save_pdbs --secondary_structure

"""

import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.rms
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        description="Analyze and cluster molecular dynamics trajectories"
    )
    parser.add_argument("--topology_path", type=str, required=True, help="Path to topology file")
    parser.add_argument(
        "--trajectory_paths", nargs="+", type=str, required=True, help="Paths to trajectory files"
    )
    parser.add_argument(
        "--json_data_path",
        type=str,
        required=False,
        help="Path to JSON file containing structural region definitions",
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
        help="Output directory (default: auto-generated based on topology name and time)",
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


    return parser.parse_args()





def calculate_rmsd(universe, ref_universe, selection="name CA"):
    """Calculate RMSD for each frame relative to a reference structure."""
    logger.info(f"Calculating RMSD using selection: {selection}")
    rmsd_values = []
    ref_atoms = ref_universe.select_atoms(selection)
    for ts in tqdm(universe.trajectory, desc="Calculating RMSD"):
        atoms = universe.select_atoms(selection)
        # Ensure both selections have the same number of atoms
        if len(atoms) != len(ref_atoms):
            logger.warning(
                f"Frame {ts.frame}: Atom count mismatch for RMSD calculation. Skipping frame."
            )
            rmsd_values.append(np.nan)
            continue
        rmsd_values.append(mda.analysis.rms.rmsd(atoms.positions, ref_atoms.positions))
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
    A native contact is defined as a pair of atoms within a certain cutoff distance
    in the reference structure that are also within that cutoff in the current frame.
    """
    from MDAnalysis.analysis import contacts

    logger.info(f"Calculating native contacts using selection: {selection} and cutoff: {cutoff} Å")

    # Get reference atoms from the first frame of ref_universe
    ref_atoms = ref_universe.select_atoms(selection)
    n_atoms = len(ref_atoms)

    logger.info(f"Selected {n_atoms} atoms for native contacts analysis")

    # Set up the contacts analysis
    # We use the same selection for both groups to get all pairwise contacts
    ca = contacts.Contacts(
        universe, select=(selection, selection), refgroup=(ref_atoms, ref_atoms), radius=cutoff
    )

    # Run the analysis
    logger.info("Running native contacts analysis...")
    ca.run()

    # Extract the fraction of native contacts (Q values)
    # ca.results.timeseries has shape (n_frames, 2) where:
    # column 0 = frame number, column 1 = fraction of native contacts
    native_contacts_fractions = ca.results.timeseries[:, 1]

    # Get the total number of native contacts from the first calculation
    # Since we don't have direct access to the reference contacts count,
    # we'll calculate it from the first frame's data
    first_frame_contacts = ca.results.timeseries[0, 1]  # This is the fraction for first frame

    # For native contacts, we want to return the fraction (Q values) directly
    # since that's the standard metric. If absolute numbers are needed,
    # they can be calculated later by multiplying with the total possible contacts
    native_contacts_values = native_contacts_fractions

    logger.info("Native contacts analysis complete.")
    logger.info(f"Average fraction of native contacts: {np.mean(native_contacts_fractions):.3f}")
    logger.info(f"Min fraction of native contacts: {np.min(native_contacts_fractions):.3f}")
    logger.info(f"Max fraction of native contacts: {np.max(native_contacts_fractions):.3f}")

    return native_contacts_values


def calculate_secondary_structure(universe, selection="protein"):
    """
    Calculate alpha helical and beta sheet content for each frame using DSSP.
    Returns fraction of residues in helix and sheet per frame.
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


def create_feature_distribution_plots(features_data, feature_names, output_dir):
    """Create distribution plots for each feature."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating feature distribution plots...")

    for i, name in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        sns.histplot(features_data[:, i], kde=True, color="steelblue")
        plt.title(f"Distribution of {name}", fontsize=16)
        plt.xlabel(name, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"distribution_{name.replace(' ', '_')}.png"), dpi=300)
        plt.close()
    logger.info("Feature distribution plots created.")


def create_pairwise_scatter_plots(features_data, feature_names, output_dir):
    """Create pairwise scatter plots of features."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating pairwise scatter plots...")

    # Convert to DataFrame for easier plotting with seaborn.pairplot
    import pandas as pd

    df = pd.DataFrame(features_data, columns=feature_names)
    pair_plot = sns.pairplot(df, diag_kind="kde")
    pair_plot.fig.suptitle("Pairwise Feature Scatter Plots", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pairwise_scatter_plots.png"), dpi=300)
    plt.close()
    logger.info("Pairwise scatter plots created.")


def create_correlation_matrix_plot(features_data, feature_names, output_dir):
    """Create a correlation matrix plot of features."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating correlation matrix plot...")

    import pandas as pd

    df = pd.DataFrame(features_data, columns=feature_names)
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmax=1, vmin=-1
    )
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    logger.info("Correlation matrix plot created.")


def create_pca_feature_plot(
    features_data, feature_names, output_dir, num_components=2, cluster_labels=None
):
    """Perform PCA on features and create a publication-quality scatter plot of the components."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Performing PCA on features with {num_components} components...")

    # Handle NaN values by replacing them with the mean of the column
    features_data_cleaned = np.nan_to_num(features_data, nan=np.nanmean(features_data, axis=0))

    # Perform PCA on the features
    pca_features = IncrementalPCA(n_components=num_components)
    pca_features_coords = pca_features.fit_transform(features_data_cleaned)

    # Set style for publication-quality plots
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.edgecolor"] = "black"

    # Create figure with GridSpec for complex layout
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 3, 1])

    # Main PCA scatter plot
    ax_main = fig.add_subplot(gs[1, :2])

    # Get x and y coordinates for the first two components
    x, y = pca_features_coords[:, 0], pca_features_coords[:, 1]

    # Calculate point density for contour plot
    sns.kdeplot(x=x, y=y, ax=ax_main, levels=20, cmap="Blues", fill=True, alpha=0.5, zorder=0)

    # Color by cluster if available, otherwise by first feature
    if cluster_labels is not None:
        scatter = ax_main.scatter(
            x, y, c=cluster_labels, cmap="viridis", s=20, alpha=0.7, zorder=10, edgecolor="none"
        )
        cbar_label = "Cluster Label"
    else:
        # Use the first feature (often RMSD) for coloring
        scatter = ax_main.scatter(
            x,
            y,
            c=features_data[:, 0],
            cmap="viridis",
            s=20,
            alpha=0.7,
            zorder=10,
            edgecolor="none",
        )
        cbar_label = feature_names[0]

    # Labels and title
    variance_pc1 = pca_features.explained_variance_ratio_[0] * 100
    variance_pc2 = pca_features.explained_variance_ratio_[1] * 100
    ax_main.set_xlabel(f"Feature PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
    ax_main.set_ylabel(f"Feature PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
    ax_main.set_title("PCA of Structural Features", fontsize=16, pad=20)

    # Add histograms for PC1 distribution
    ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
    sns.histplot(x, kde=True, ax=ax_top, color="darkblue", alpha=0.6)
    ax_top.set_ylabel("Density", fontsize=12)
    ax_top.set_title("Feature PC1 Distribution", fontsize=14)
    ax_top.tick_params(labelbottom=False)

    # Add histograms for PC2 distribution
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    sns.histplot(y=y, kde=True, ax=ax_right, color="darkblue", alpha=0.6, orientation="horizontal")
    ax_right.set_xlabel("Density", fontsize=12)
    ax_right.set_title("Feature PC2 Distribution", fontsize=14)
    ax_right.tick_params(labelleft=False)

    # Create explained variance ratio plot
    ax_var = fig.add_subplot(gs[2, :])
    components = range(1, len(pca_features.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca_features.explained_variance_ratio_)

    # Plot individual and cumulative explained variance
    bars = ax_var.bar(
        components,
        pca_features.explained_variance_ratio_,
        color="steelblue",
        alpha=0.7,
        label="Individual",
    )

    ax_var2 = ax_var.twinx()
    line = ax_var2.plot(
        components,
        cumulative,
        "o-",
        color="firebrick",
        linewidth=2.5,
        markersize=8,
        label="Cumulative",
    )

    # Add explained variance labels
    ax_var.set_xlabel("Principal Component", fontsize=14)
    ax_var.set_ylabel("Explained Variance Ratio", fontsize=14)
    ax_var2.set_ylabel("Cumulative Explained Variance", fontsize=14)
    ax_var.set_title("Explained Variance by Principal Components", fontsize=16, pad=20)

    # Set x-axis to integers
    ax_var.set_xticks(components)
    ax_var2.set_ylim([0, 1.05])

    # Combine legends
    lines, labels = ax_var.get_legend_handles_labels()
    lines2, labels2 = ax_var2.get_legend_handles_labels()
    ax_var.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=12)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=14, labelpad=15)

    # Save the figure
    plt.tight_layout()
    pca_plot_path = os.path.join(plots_dir, "pca_features.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("PCA of features plot created.")
    return pca_plot_path


def create_publication_plots(pca_coords, cluster_labels, cluster_centers, pca, output_dir):
    """Create publication-quality plots of the PCA results and clustering"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set style for publication-quality plots
    plt.style.use("default")  # Use the default style or choose another valid style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.edgecolor"] = "black"

    # 1. Create main PCA plot with clusters
    logger.info("Creating PCA projection plot with clusters...")

    # Figure setup with gridspec for complex layout
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 3, 1])

    # Main PCA scatter plot
    ax_main = fig.add_subplot(gs[1, :2])

    # Calculate point density for contour plot
    x, y = pca_coords[:, 0], pca_coords[:, 1]
    sns.kdeplot(x=x, y=y, ax=ax_main, levels=20, cmap="Blues", fill=True, alpha=0.5, zorder=0)

    # Scatter plot of frames colored by cluster
    scatter = ax_main.scatter(
        x, y, c=cluster_labels, cmap="viridis", s=20, alpha=0.7, zorder=10, edgecolor="none"
    )

    # Plot cluster centers if available
    if cluster_centers is not None:
        ax_main.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            c="red",
            s=80,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            zorder=20,
            label="Cluster Centers",
        )

    # Labels and title
    variance_pc1 = pca.explained_variance_ratio_[0] * 100
    variance_pc2 = pca.explained_variance_ratio_[1] * 100
    ax_main.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
    ax_main.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
    ax_main.set_title("PCA Projection with K-means Clustering", fontsize=16, pad=20)

    # Add histograms for PC1 distribution
    ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
    sns.histplot(x, kde=True, ax=ax_top, color="darkblue", alpha=0.6)
    ax_top.set_ylabel("Density", fontsize=12)
    ax_top.set_title("PC1 Distribution", fontsize=14)
    ax_top.tick_params(labelbottom=False)

    # Add histograms for PC2 distribution
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    sns.histplot(y=y, kde=True, ax=ax_right, color="darkblue", alpha=0.6, orientation="horizontal")
    ax_right.set_xlabel("Density", fontsize=12)
    ax_right.set_title("PC2 Distribution", fontsize=14)
    ax_right.tick_params(labelleft=False)

    # Create explained variance ratio plot
    ax_var = fig.add_subplot(gs[2, :])
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Plot individual and cumulative explained variance
    bars = ax_var.bar(
        components, pca.explained_variance_ratio_, color="steelblue", alpha=0.7, label="Individual"
    )

    ax_var2 = ax_var.twinx()
    line = ax_var2.plot(
        components,
        cumulative,
        "o-",
        color="firebrick",
        linewidth=2.5,
        markersize=8,
        label="Cumulative",
    )

    # Add explained variance labels
    ax_var.set_xlabel("Principal Component", fontsize=14)
    ax_var.set_ylabel("Explained Variance Ratio", fontsize=14)
    ax_var2.set_ylabel("Cumulative Explained Variance", fontsize=14)
    ax_var.set_title("Explained Variance by Principal Components", fontsize=16, pad=20)

    # Set x-axis to integers
    ax_var.set_xticks(components)
    ax_var2.set_ylim([0, 1.05])

    # Combine legends
    lines, labels = ax_var.get_legend_handles_labels()
    lines2, labels2 = ax_var2.get_legend_handles_labels()
    ax_var.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=12)

    # Add colorbar for cluster labels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Cluster Label", fontsize=14, labelpad=15)

    # Save the figure
    plt.tight_layout()
    pca_plot_path = os.path.join(plots_dir, "pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Create 3D PCA plot if we have at least 3 components
    if pca_coords.shape[1] >= 3:
        logger.info("Creating 3D PCA plot...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            pca_coords[:, 2],
            c=cluster_labels,
            cmap="viridis",
            s=30,
            alpha=0.7,
        )

        # Add cluster centers to 3D plot if available
        if cluster_centers is not None:
            ax.scatter(
                cluster_centers[:, 0],
                cluster_centers[:, 1],
                cluster_centers[:, 2],
                c="red",
                s=100,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
            )

        variance_pc3 = pca.explained_variance_ratio_[2] * 100
        ax.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=12)
        ax.set_zlabel(f"PC3 ({variance_pc3:.1f}% variance)", fontsize=12)
        ax.set_title("3D PCA Projection with Clusters", fontsize=16)

        plt.colorbar(scatter, ax=ax, label="Cluster Label")
        plt.tight_layout()

        pca_3d_path = os.path.join(plots_dir, "pca_3d.png")
        plt.savefig(pca_3d_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Create cluster size distribution plot
    logger.info("Creating cluster size distribution plot...")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    plt.figure(figsize=(14, 8))
    sns.histplot(counts, kde=True, color="steelblue")
    plt.axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
    plt.axvline(
        np.median(counts), color="green", linestyle="--", label=f"Median: {np.median(counts):.1f}"
    )

    plt.xlabel("Frames per Cluster", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of Cluster Sizes", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    cluster_dist_path = os.path.join(plots_dir, "cluster_distribution.png")
    plt.savefig(cluster_dist_path, dpi=300)
    plt.close()

    return {
        "pca_plot": pca_plot_path,
        "pca_3d_plot": pca_3d_path if pca_coords.shape[1] >= 3 else None,
        "cluster_dist": cluster_dist_path,
    }


def create_pca_pairwise_hue_by_feature_plots(
    pca_coords, features_data, feature_names, pca_pairwise, output_dir
):
    """
    Create PCA plots of pairwise coordinates, hueing by each individual feature.
    Save these PCAs in a subfolder and plot them individually for each feature selected.
    """
    plots_dir = os.path.join(output_dir, "plots")
    hue_plots_dir = os.path.join(plots_dir, "pca_hue_by_features")
    os.makedirs(hue_plots_dir, exist_ok=True)
    logger.info("Creating PCA pairwise plots, hueing by individual features...")

    # Set style for publication-quality plots
    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.edgecolor"] = "black"

    x, y = pca_coords[:, 0], pca_coords[:, 1]
    variance_pc1 = pca_pairwise.explained_variance_ratio_[0] * 100
    variance_pc2 = pca_pairwise.explained_variance_ratio_[1] * 100

    for i, feature_name in enumerate(feature_names):
        plt.figure(figsize=(12, 10))
        ax = plt.gca()

        # Handle NaN values in feature data for hueing
        current_feature_data = features_data[:, i]
        # Replace NaN with a value that stands out or is handled by cmap (e.g., mean or a specific color)
        # For simplicity, we'll just skip if all are NaN, or use nan_to_num for plotting
        if np.all(np.isnan(current_feature_data)):
            logger.warning(f"Skipping hue plot for feature '{feature_name}' as all values are NaN.")
            plt.close()
            continue

        # Calculate point density for contour plot
        sns.kdeplot(x=x, y=y, ax=ax, levels=20, cmap="Blues", fill=True, alpha=0.5, zorder=0)

        scatter = ax.scatter(
            x,
            y,
            c=current_feature_data,
            cmap="viridis",  # Or a more suitable colormap for continuous data
            s=20,
            alpha=0.7,
            zorder=10,
            edgecolor="none",
        )

        ax.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
        ax.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
        ax.set_title(f"PCA of Pairwise Coordinates (Hued by {feature_name})", fontsize=16, pad=20)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(feature_name, fontsize=14, labelpad=15)

        plt.tight_layout()
        plot_filename = (
            f"pca_pairwise_hue_by_{feature_name.replace(' ', '_').replace('/', '_')}.png"
        )
        plt.savefig(os.path.join(hue_plots_dir, plot_filename), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Created PCA pairwise plot hued by '{feature_name}'.")

    logger.info("Finished creating PCA pairwise plots hued by individual features.")


def create_cluster_ratio_plot(cluster_labels, output_dir):
    """Create a plot showing the distribution ratio of frames across clusters."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Creating cluster ratio plot...")

    # Count frames per cluster
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    total_frames = len(cluster_labels)

    # Sort clusters by their label for consistent presentation
    sort_idx = np.argsort(unique_clusters)
    unique_clusters = unique_clusters[sort_idx]
    counts = counts[sort_idx]

    # Calculate percentages
    percentages = (counts / total_frames) * 100

    # Create figure with two subplots - bar chart and pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Bar chart showing counts per cluster
    bars = ax1.bar(
        unique_clusters,
        counts,
        color=plt.cm.viridis(np.linspace(0, 1, len(unique_clusters))),
        alpha=0.8,
    )

    # Add count labels above bars
    for bar, count, percentage in zip(bars, counts, percentages):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (total_frames * 0.01),  # Small offset above bar
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax1.set_xlabel("Cluster", fontsize=14)
    ax1.set_ylabel("Number of Frames", fontsize=14)
    ax1.set_title("Distribution of Frames per Cluster", fontsize=16)
    ax1.set_xticks(unique_clusters)
    ax1.set_xticklabels([f"Cluster {c}" for c in unique_clusters])
    ax1.grid(axis="y", alpha=0.3)

    # Pie chart showing percentage distribution
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=[f"Cluster {c}" for c in unique_clusters],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        shadow=False,
    )

    # Style the pie chart text
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_color("white")

    ax2.set_title("Percentage of Frames in Each Cluster", fontsize=16)
    ax2.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

    plt.tight_layout()
    ratio_plot_path = os.path.join(plots_dir, "cluster_ratios.png")
    plt.savefig(ratio_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a table with the data
    fig, ax = plt.figure(figsize=(10, len(unique_clusters) / 2 + 2)), plt.subplot(111)
    ax.axis("off")
    table_data = (
        [
            ["Cluster", "Frames", "Percentage"],
        ]
        + [
            [f"Cluster {c}", f"{count}", f"{percentage:.2f}%"]
            for c, count, percentage in zip(unique_clusters, counts, percentages)
        ]
        + [["Total", f"{total_frames}", "100.00%"]]
    )

    table = ax.table(cellText=table_data, cellLoc="center", loc="center", bbox=[0.2, 0.2, 0.6, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#D6EAF8")
        elif row == len(table_data) - 1:  # Total row
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EBF5FB")

    plt.title("Cluster Distribution Summary", fontsize=16, pad=20)
    table_path = os.path.join(plots_dir, "cluster_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Cluster ratio plots created: {ratio_plot_path} and {table_path}")
    return {"ratio_plot": ratio_plot_path, "table": table_path}


def save_cluster_trajectories(
    universe, cluster_labels, pca_coords, cluster_centers, output_dir, save_pdbs=False
):
    """Save cluster trajectories to a single XTC file and optionally save representative PDB files

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The universe containing all frames
    cluster_labels : numpy.ndarray
        Array of cluster labels for each frame
    pca_coords : numpy.ndarray
        PCA coordinates for each frame
    cluster_centers : numpy.ndarray
        K-means computed cluster centers in PCA space
    output_dir : str
        Output directory path
    save_pdbs : bool, optional
        Whether to save individual PDB files for each cluster representative
    """
    logger.info("Saving cluster trajectories...")

    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Find the frame in each cluster that is closest to the true cluster center
    representative_frames = {}

    for cluster_idx in unique_clusters:
        # Get indices of frames in this cluster
        cluster_mask = cluster_labels == cluster_idx
        if not np.any(cluster_mask):
            continue

        cluster_frame_indices = np.where(cluster_mask)[0]

        # Get PCA coordinates for frames in this cluster
        cluster_pca_coords = pca_coords[cluster_mask]

        # Get the center for this cluster
        center = cluster_centers[cluster_idx]

        # Calculate distances from each frame to the cluster center
        distances = np.sqrt(np.sum((cluster_pca_coords - center) ** 2, axis=1))

        # Find the frame with minimum distance to center
        min_dist_idx = np.argmin(distances)

        # Get the original frame index
        representative_frame_idx = cluster_frame_indices[min_dist_idx]

        # Store the representative frame
        representative_frames[cluster_idx] = representative_frame_idx

    # Create a single trajectory file with cluster centers only
    all_clusters_file = os.path.join(clusters_dir, "all_clusters.xtc")
    with mda.Writer(all_clusters_file, universe.atoms.n_atoms) as writer:
        # Go through each cluster and save only the representative frame
        for cluster_idx in tqdm(unique_clusters, desc="Saving cluster centers"):
            if cluster_idx in representative_frames:
                frame_idx = representative_frames[cluster_idx]
                universe.trajectory[frame_idx]
                writer.write(universe.atoms)

    # Optionally save representative frames as PDB files
    if save_pdbs:
        for cluster_idx, frame_idx in tqdm(representative_frames.items(), desc="Saving PDB files"):
            universe.trajectory[frame_idx]
            with mda.Writer(
                os.path.join(clusters_dir, f"cluster_{cluster_idx}_rep.pdb")
            ) as pdb_writer:
                pdb_writer.write(universe.atoms)

    # Also save a CSV file mapping frame to cluster
    frame_to_cluster = np.column_stack((np.arange(len(cluster_labels)), cluster_labels))
    np.savetxt(
        os.path.join(clusters_dir, "frame_to_cluster.csv"),
        frame_to_cluster,
        delimiter=",",
        header="frame_index,cluster_label",
        fmt="%d",
        comments="",
    )

    logger.info(
        f"Saved {n_clusters} true cluster centers to a single trajectory file: {all_clusters_file}"
    )
    if save_pdbs:
        logger.info(f"Saved {len(representative_frames)} representative PDB files")
    logger.info("Saved frame-to-cluster mapping to frame_to_cluster.csv")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Start timer
    start_time = datetime.datetime.now()

    # Set up output directory
    if args.output_dir is None:
        topology_name = os.path.splitext(os.path.basename(args.topology_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(os.getcwd())),
            f"{topology_name}_analysis_clusters{args.n_clusters}_{timestamp}",
        )

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    global logger
    log_file = os.path.join(args.output_dir, "analyse_cluster_PUFs.log")
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("Starting trajectory analysis and clustering")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")



    # Load universe
    logger.info(f"Loading topology from {args.topology_path}")
    logger.info(f"Loading trajectories: {args.trajectory_paths}")
    universe = mda.Universe(args.topology_path, *args.trajectory_paths)
    logger.info(
        f"Loaded universe with {len(universe.atoms)} atoms and {len(universe.trajectory)} frames"
    )

    # Set chunk size to the whole trajectory length
    chunk_size = len(universe.trajectory)
    logger.info(f"Setting PCA chunk size to full trajectory length: {chunk_size} frames")

    # Calculate distances and perform PCA on pairwise coordinates
    pca_coords, pca_pairwise = calculate_distances_and_perform_pca(
        universe, args.atom_selection, args.num_components_pca_coords, chunk_size
    )

    # --- Feature Selection ---
    # Determine which features to calculate
    feature_flags = {
        "rmsd": args.rmsd,
        "rog": args.rog,
        "sasa": args.sasa,
        "native_contacts": args.native_contacts,
        "secondary_structure": args.secondary_structure,
    }

    # If no specific features are selected or --all_features is specified, calculate all features
    if not any(feature_flags.values()) or args.all_features:
        for key in feature_flags:
            feature_flags[key] = True
        logger.info("No specific features selected. Calculating all available features.")
    else:
        logger.info("Calculating selected features only:")
        for feature, enabled in feature_flags.items():
            if enabled:
                logger.info(f"  - {feature}")

    # --- Feature Calculation ---
    features_data_list = []
    feature_names = []
    ref_universe = mda.Universe(args.topology_path)  # Reference for RMSD and native contacts

    # Calculate each selected feature
    if feature_flags["rmsd"]:
        logger.info("Calculating structural features: RMSD")
        rmsd_values = calculate_rmsd(universe, ref_universe, selection="protein and name CA")
        features_data_list.append(rmsd_values)
        feature_names.append("RMSD")

    if feature_flags["rog"]:
        logger.info("Calculating structural features: Radius of Gyration")
        rog_values = calculate_radius_of_gyration(universe, selection="protein")
        features_data_list.append(rog_values)
        feature_names.append("Radius of Gyration")

    if feature_flags["sasa"]:
        logger.info("Calculating structural features: SASA")
        sasa_values = calculate_sasa(universe, selection="protein")
        features_data_list.append(sasa_values)
        feature_names.append("SASA")

    if feature_flags["native_contacts"]:
        logger.info("Calculating structural features: Native Contacts")
        native_contacts_values = calculate_native_contacts(
            universe, ref_universe, selection="protein and name CA"
        )
        features_data_list.append(native_contacts_values)
        feature_names.append("Native Contacts")

    if feature_flags["secondary_structure"]:
        logger.info("Calculating structural features: Secondary Structure")
        alpha_helix_content, beta_sheet_content = calculate_secondary_structure(
            universe, selection="protein"
        )
        features_data_list.append(alpha_helix_content)
        features_data_list.append(beta_sheet_content)
        feature_names.append("Alpha Helix Content")
        feature_names.append("Beta Sheet Content")



    # Check if we have any features to analyze
    if not features_data_list:
        logger.error("No features were calculated. Please select at least one feature.")
        import sys

        sys.exit(1)

    # Combine features into a single array
    features_data = np.column_stack(features_data_list)
    logger.info(f"Calculated {len(feature_names)} features: {', '.join(feature_names)}")

    # Normalize features before clustering
    features_data_cleaned = np.nan_to_num(features_data, nan=np.nanmean(features_data, axis=0))
    features_scaled = StandardScaler().fit_transform(features_data_cleaned)

    # Perform k-means clustering on the features directly
    logger.info(f"Performing k-means clustering on features with {args.n_clusters} clusters...")
    cluster_labels, cluster_centers, kmeans_features = perform_kmeans_clustering(
        features_scaled, args.n_clusters
    )

    # Create publication-quality plots for PCA visualization of structures colored by feature clusters
    plot_paths_pairwise = create_publication_plots(
        pca_coords, cluster_labels, None, pca_pairwise, args.output_dir
    )

    # Create PCA pairwise plots, hueing by individual features
    if len(features_data_list) >= 1:
        create_pca_pairwise_hue_by_feature_plots(
            pca_coords, features_data, feature_names, pca_pairwise, args.output_dir
        )
    else:
        logger.info("Skipping PCA pairwise plots hued by features - no features to hue by.")

    # Create cluster ratio visualization
    cluster_ratio_plots = create_cluster_ratio_plot(cluster_labels, args.output_dir)

    # Save cluster trajectories and mapping
    save_cluster_trajectories(
        universe,
        cluster_labels,
        features_scaled,  # We're using feature space for finding representatives
        cluster_centers,
        args.output_dir,
        args.save_pdbs,
    )

    # Save feature data
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "features_data.npy"), features_data)
    with open(os.path.join(data_dir, "feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    logger.info("Structural features calculated and saved.")

    # --- Feature Plotting ---
    logger.info("Creating feature plots...")
    # Only create plots if we have at least one feature
    if len(features_data_list) > 0:
        create_feature_distribution_plots(features_data, feature_names, args.output_dir)

        # Only create pairwise and correlation plots if we have more than one feature
        if len(features_data_list) > 1:
            create_pairwise_scatter_plots(features_data, feature_names, args.output_dir)
            create_correlation_matrix_plot(features_data, feature_names, args.output_dir)

        # PCA only makes sense with at least 2 features
        if len(features_data_list) >= 2:
            # Pass the scaled features and cluster labels to the PCA feature plot function
            create_pca_feature_plot(
                features_scaled, feature_names, args.output_dir, cluster_labels=cluster_labels
            )
        else:
            logger.info("Skipping PCA feature plot - need at least 2 features for PCA")
    else:
        logger.info("No features to plot.")

    logger.info("Feature plots created.")

    # End timer and report
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    logger.info("=" * 80)
    logger.info(f"Analysis and clustering complete. Results saved to {args.output_dir}")
    logger.info(f"Total execution time: {elapsed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
