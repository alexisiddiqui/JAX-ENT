"""

This script is used to trim the trajectory files for the TeaA system to only include structures that are less than 1 Å (Calpha RMSD) from either the open or closed state reference structures.

output_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/TeaA/quick_auto_validation_results"

open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc"

"""

import os

# import mpl
# Set up matplotlib parameters
import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align, rms
from sklearn.decomposition import PCA
from tqdm import tqdm

mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 12,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)


full_dataset_colours = {
    "ISO-BiModal": "indigo",
    "ISO-TriModal": "saddlebrown",
}

# Paths defined in the script
output_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories"
open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced_adequate.xtc"
# trajectory_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_initial_reimaged.xtc"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load reference structures
u_open = mda.Universe(open_path)
u_closed = mda.Universe(closed_path)

# Load trajectory
u_traj = mda.Universe(topology_path, trajectory_path)

# Select CA atoms for RMSD calculation
ca_atoms = "name CA"
open_ca = u_open.select_atoms(ca_atoms)
closed_ca = u_closed.select_atoms(ca_atoms)
traj_ca = u_traj.select_atoms(ca_atoms)

# Number of frames in trajectory
n_frames = len(u_traj.trajectory)
print(f"Trajectory has {n_frames} frames")

# First, align trajectory to open state and calculate RMSD
print("Aligning trajectory to open state...")
open_aligner = align.AlignTraj(u_traj, u_open, select=ca_atoms, in_memory=True).run()
print("Calculating RMSD to open state...")
open_rmsd = rms.RMSD(u_traj, u_open, select=ca_atoms)
open_rmsd.run()
rmsd_to_open = open_rmsd.rmsd[:, 2]  # column 2 contains RMSD values

# Align trajectory to closed state and calculate RMSD
print("Aligning trajectory to closed state...")
closed_aligner = align.AlignTraj(u_traj, u_closed, select=ca_atoms, in_memory=True).run()
print("Calculating RMSD to closed state...")
closed_rmsd = rms.RMSD(u_traj, u_closed, select=ca_atoms)
closed_rmsd.run()
rmsd_to_closed = closed_rmsd.rmsd[:, 2]  # column 2 contains RMSD values

# Identify frames within 1 Å of either reference structure
threshold = 1.0  # Angstrom
near_open = rmsd_to_open < threshold
near_closed = rmsd_to_closed < threshold
selected_frames = np.logical_or(near_open, near_closed)
selected_indices = np.where(selected_frames)[0]

print(f"Found {np.sum(near_open)} frames near open state")
print(f"Found {np.sum(near_closed)} frames near closed state")
print(f"Total unique frames selected: {len(selected_indices)}")

# Create a new universe with selected frames
filtered_trajectory_path = os.path.join(output_dir, "TeaA_filtered.xtc")
# with mda.Writer(filtered_trajectory_path, u_traj.atoms.n_atoms) as W:
#     for i in selected_indices:
#         u_traj.trajectory[i]
#         W.write(u_traj.atoms)

# print(f"Filtered trajectory saved to {filtered_trajectory_path}")

# Re-align trajectory to open state for PCA
print("Re-aligning trajectory to open state for PCA...")
align.AlignTraj(u_traj, u_open, select=ca_atoms, in_memory=True).run()

# Collect CA coordinates for PCA
print("Collecting CA coordinates for original trajectory...")
all_coordinates = []
for ts in tqdm(u_traj.trajectory, desc="Processing frames"):
    all_coordinates.append(traj_ca.positions.copy())
all_coordinates = np.array(all_coordinates)

# Load filtered trajectory and align it to the same reference
u_filtered = mda.Universe(topology_path, filtered_trajectory_path)
filtered_ca = u_filtered.select_atoms(ca_atoms)
print("Aligning filtered trajectory...")
align.AlignTraj(u_filtered, u_open, select=ca_atoms, in_memory=True).run()

# Collect CA coordinates for filtered trajectory
print("Collecting CA coordinates for filtered trajectory...")
filtered_coordinates = []
for ts in tqdm(u_filtered.trajectory, desc="Processing filtered frames"):
    filtered_coordinates.append(filtered_ca.positions.copy())
filtered_coordinates = np.array(filtered_coordinates)

# Get pairwise CA distances for original trajectory
n_atoms = traj_ca.n_atoms
print(f"Calculating pairwise distances for {n_atoms} CA atoms in original trajectory...")
pairwise_distances = []

for frame_coords in tqdm(all_coordinates, desc="Computing pairwise distances"):
    # Calculate pairwise distances for each frame
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate Euclidean distance between CA atoms i and j
            dist = np.linalg.norm(frame_coords[i] - frame_coords[j])
            distances.append(dist)
    pairwise_distances.append(distances)

# Convert to numpy array
pairwise_distances = np.array(pairwise_distances)

# Do the same for filtered trajectory
print("Calculating pairwise distances for filtered trajectory...")
filtered_pairwise_distances = []
for frame_coords in tqdm(filtered_coordinates, desc="Computing filtered pairwise distances"):
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(frame_coords[i] - frame_coords[j])
            distances.append(dist)
    filtered_pairwise_distances.append(distances)
filtered_pairwise_distances = np.array(filtered_pairwise_distances)

# Perform PCA on pairwise distances
print("Performing PCA on pairwise distances...")
pca = PCA(n_components=2)
pca.fit(pairwise_distances)

# Project original and filtered trajectories onto the first two principal components
all_projected = pca.transform(pairwise_distances)
filtered_projected = pca.transform(filtered_pairwise_distances)

# Also project reference structures
open_pairwise_distances = []
distances = []
for i in range(n_atoms):
    for j in range(i + 1, n_atoms):
        dist = np.linalg.norm(open_ca.positions[i] - open_ca.positions[j])
        distances.append(dist)
open_pairwise_distances = np.array([distances])
open_projected = pca.transform(open_pairwise_distances)

closed_pairwise_distances = []
distances = []
for i in range(n_atoms):
    for j in range(i + 1, n_atoms):
        dist = np.linalg.norm(closed_ca.positions[i] - closed_ca.positions[j])
        distances.append(dist)
closed_pairwise_distances = np.array([distances])
closed_projected = pca.transform(closed_pairwise_distances)

# Create PCA plot
plt.figure(figsize=(12, 10))

# Plot original trajectory points
plt.scatter(
    all_projected[:, 0],
    all_projected[:, 1],
    alpha=0.5,
    label="Original trajectory",
    color="gray",
    s=10,
)

# Plot filtered trajectory points
plt.scatter(
    filtered_projected[:, 0],
    filtered_projected[:, 1],
    alpha=0.7,
    label="Filtered trajectory",
    color="blue",
    s=20,
)

# Plot reference structures
plt.scatter(
    open_projected[0, 0],
    open_projected[0, 1],
    s=200,
    label="Open reference",
    color="green",
    marker="*",
    edgecolor="black",
)
plt.scatter(
    closed_projected[0, 0],
    closed_projected[0, 1],
    s=200,
    label="Closed reference",
    color="red",
    marker="*",
    edgecolor="black",
)

# Add labels and legend
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of TeaA Trajectory - Pairwise CA Distances")
plt.legend()
plt.grid(alpha=0.3)

# Save the plot
pca_plot_path = os.path.join(output_dir, "TeaA_PCA_plot.png")
plt.savefig(pca_plot_path, dpi=300)
plt.close()

print(f"PCA plot saved to {pca_plot_path}")

# Plot RMSD to both references over time
plt.figure(figsize=(12, 6))
plt.plot(rmsd_to_open, label="RMSD to Open State")
plt.plot(rmsd_to_closed, label="RMSD to Closed State")
plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold} Å)")
plt.xlabel("Frame")
plt.ylabel("RMSD (Å)")
plt.title("RMSD to Reference Structures")
plt.legend()
plt.grid(alpha=0.3)

# Highlight selected frames
for i in selected_indices:
    plt.axvline(x=i, color="g", alpha=0.1)

# Save the RMSD plot
rmsd_plot_path = os.path.join(output_dir, "TeaA_RMSD_plot.png")
plt.savefig(rmsd_plot_path, dpi=300)
plt.close()

# Create a new RMSD clustering plot (RMSD to open vs RMSD to closed)
plt.figure(figsize=(8, 6))

# Set up color scheme as requested
open_color = "blue"
closed_color = "aquamarine"

# Identify points in clusters (within threshold)
near_open = rmsd_to_open < threshold
near_closed = rmsd_to_closed < threshold
selected_frames = np.logical_or(near_open, near_closed)

# Color array (blue for closer to open, orange for closer to closed)
colors = np.where(rmsd_to_open < rmsd_to_closed, open_color, closed_color)

# Plot non-clustered points with lower saturation first
plt.scatter(
    rmsd_to_open[~selected_frames],
    rmsd_to_closed[~selected_frames],
    c=np.take(colors, np.where(~selected_frames)[0]),
    alpha=0.3,  # Lower alpha for non-clustered points
    s=20,  # Smaller size for non-clustered points
)

# Plot clustered points with full saturation on top
plt.scatter(
    rmsd_to_open[selected_frames],
    rmsd_to_closed[selected_frames],
    c=np.take(colors, np.where(selected_frames)[0]),
    alpha=0.8,  # Higher alpha for clustered points
    s=40,  # Larger size for clustered points
    edgecolor="black",
    linewidth=0.5,
)

# Add threshold lines
plt.axvline(x=threshold, color=open_color, linestyle="--", alpha=0.7)
plt.axhline(y=threshold, color=closed_color, linestyle="--", alpha=0.7)

# Add diagonal line (equidistant from both states)
max_val = max(np.max(rmsd_to_open), np.max(rmsd_to_closed)) * 1.1
plt.plot([0, max_val], [0, max_val], "k--", alpha=0.5)

# Create custom legend items
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=open_color,
        markersize=10,
        alpha=0.8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Open State Cluster (<1 Å)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=closed_color,
        markersize=10,
        alpha=0.8,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="Closed State Cluster (<1 Å)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=open_color,
        markersize=8,
        alpha=0.3,
        label="Non-clustered (Open-like)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=closed_color,
        markersize=8,
        alpha=0.3,
        label="Non-clustered (Closed-like)",
    ),
    # Line2D([0], [0], linestyle="--", color=open_color, label=f"Open State Threshold ({threshold} Å)")
    Line2D(
        [0],
        [0],
        linestyle="--",
        color=closed_color,
        label=f"Closed State Threshold ({threshold} Å)",
    ),
    Line2D([0], [0], linestyle="--", color="black", alpha=0.5, label="Equal Distance Line"),
]

# Labels and formatting
plt.xlabel("RMSD to Open State (Å)", fontsize=20, color="black")
plt.ylabel("RMSD to Closed State (Å)", fontsize=20, color="black")
plt.title("TeaA | Iso Validation Clustering", fontsize=24)
plt.grid(alpha=0.3)

# Add explanatory text
explanation = (
    "RMSD Clustering Explanation:\n\n"
    "• Each point represents a single structure (frame) from the TeaA simulation\n"
    "• The x-axis shows how similar each structure is to the open reference state\n"
    "• The y-axis shows how similar each structure is to the closed reference state\n"
    "• Lower RMSD values (closer to 0) indicate higher structural similarity\n"
    "• Blue points are closer to the open state, orange points closer to the closed state\n"
    "• Faded points are not included in the clustering (>1 Å from both states)\n"
    "• Bright points are within the " + str(threshold) + " Å threshold of either state\n"
    "• The dashed lines mark the " + str(threshold) + " Å similarity threshold"
)

# Add the explanatory text in a box
# plt.figtext(
#     0.5,
#     0.01,
#     explanation,
#     ha="center",
#     fontsize=12,
#     bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9, "pad": 5},
# )

# Add legend
plt.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.6, fontsize=10)

# Set equal aspect ratio for clarity
plt.axis("equal")

# Set reasonable limits with buffer
buffer = 0
plt.xlim(0, 4)
plt.ylim(0, 4)

# Improve tick labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add shaded regions to highlight thresholds
plt.axvspan(0, threshold, alpha=0.1, color=open_color)
plt.axhspan(0, threshold, alpha=0.1, color=closed_color)

# Make plot tight
# plt.tight_layout(rect=[0, 0.15, 1, 1])  # Make room for the bottom explanation text

# Save the clustering plot
rmsd_cluster_path = os.path.join(output_dir, "TeaA_RMSD_clustering_plot.png")
plt.savefig(rmsd_cluster_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"RMSD clustering plot saved to {rmsd_cluster_path}")

print(f"RMSD plot saved to {rmsd_plot_path}")


# Calculate RMSD values for the filtered trajectory
# Re-align filtered trajectory to both reference states
print("Calculating RMSD values for filtered trajectory...")
print("Aligning filtered trajectory to open state...")
align.AlignTraj(u_filtered, u_open, select=ca_atoms, in_memory=True).run()
filtered_open_rmsd = rms.RMSD(u_filtered, u_open, select=ca_atoms)
filtered_open_rmsd.run()
filtered_rmsd_to_open = filtered_open_rmsd.rmsd[:, 2]

print("Aligning filtered trajectory to closed state...")
align.AlignTraj(u_filtered, u_closed, select=ca_atoms, in_memory=True).run()
filtered_closed_rmsd = rms.RMSD(u_filtered, u_closed, select=ca_atoms)
filtered_closed_rmsd.run()
filtered_rmsd_to_closed = filtered_closed_rmsd.rmsd[:, 2]

# Create histogram plots with separate panels for open and closed distributions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=True)

# Define colors and labels
trimodal_color = full_dataset_colours["ISO-TriModal"]  # saddlebrown
bimodal_color = full_dataset_colours["ISO-BiModal"]  # indigo

# Set histogram parameters
bins = 50
alpha = 0.4
density = True  # This normalizes the histograms

# Panel 1: RMSD to Open State
ax1.hist(
    rmsd_to_open,
    bins=bins,
    alpha=alpha,
    color=trimodal_color,
    density=density,
    label="ISO-Trimodal (Parent)",
    edgecolor="black",
    linewidth=0.5,
)
ax1.hist(
    filtered_rmsd_to_open,
    bins=bins,
    alpha=alpha,
    color=bimodal_color,
    density=density,
    label="ISO-Bimodal (Clustered)",
    edgecolor="black",
    linewidth=0.5,
)

ax1.axvline(
    x=threshold,
    color="black",
    linestyle="--",
    alpha=0.8,
    linewidth=2,
    label=f"Threshold ({threshold} Å)",
)
ax1.set_xlabel("RMSD to Open Reference State (Å)", fontsize=14, color="black")
ax1.set_ylabel("Density", fontsize=14)
# ax1.set_title("Distribution of RMSD to Open Reference", fontsize=14)
ax1.legend(fontsize=8, loc="upper center")
ax1.grid(alpha=0.0)
ax1.tick_params(labelsize=12)

# Panel 2: RMSD to Closed State
ax2.hist(
    rmsd_to_closed,
    bins=bins,
    alpha=alpha,
    color=trimodal_color,
    density=density,
    label="ISO-Trimodal (Parent)",
    edgecolor=trimodal_color,
    linewidth=0.5,
)
ax2.hist(
    filtered_rmsd_to_closed,
    bins=bins,
    alpha=alpha,
    color=bimodal_color,
    density=density,
    label="ISO-Bimodal (Clustered)",
    edgecolor=bimodal_color,
    linewidth=0.5,
)

ax2.axvline(
    x=threshold,
    color="black",
    linestyle="--",
    alpha=0.8,
    linewidth=2,
    label=f"Threshold ({threshold} Å)",
)
ax2.set_xlabel("RMSD to Closed Reference State (Å)", fontsize=14, color="black")
ax2.set_ylabel("Density", fontsize=14)
# ax2.set_title("Distribution of RMSD to Closed Reference", fontsize=14)
ax2.legend(fontsize=8, loc="upper center")
ax2.grid(alpha=0.0)
ax2.tick_params(labelsize=12)

# Adjust layout and save
plt.tight_layout()
histogram_plot_path = os.path.join(output_dir, "TeaA_RMSD_histograms.png")
plt.savefig(histogram_plot_path, dpi=300)
plt.close()

print(f"RMSD histogram plots saved to {histogram_plot_path}")

# Print some statistics for comparison
print("\n=== Distribution Statistics ===")
print("Original trajectory (Trimodal):")
print(f"  Frames: {len(rmsd_to_open)}")
print(f"  RMSD to Open - Mean: {np.mean(rmsd_to_open):.2f} Å, Std: {np.std(rmsd_to_open):.2f} Å")
print(
    f"  RMSD to Closed - Mean: {np.mean(rmsd_to_closed):.2f} Å, Std: {np.std(rmsd_to_closed):.2f} Å"
)

print("\nFiltered trajectory (Bimodal):")
print(f"  Frames: {len(filtered_rmsd_to_open)}")
print(
    f"  RMSD to Open - Mean: {np.mean(filtered_rmsd_to_open):.2f} Å, Std: {np.std(filtered_rmsd_to_open):.2f} Å"
)
print(
    f"  RMSD to Closed - Mean: {np.mean(filtered_rmsd_to_closed):.2f} Å, Std: {np.std(filtered_rmsd_to_closed):.2f} Å"
)

print("\nFiltering efficiency:")
print(f"  Retained {len(filtered_rmsd_to_open) / len(rmsd_to_open) * 100:.1f}% of original frames")
