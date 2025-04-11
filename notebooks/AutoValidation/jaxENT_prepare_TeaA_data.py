"""

This script is used to trim the trajectory files for the TeaA system to only include structures that are less than 1 Å (Calpha RMSD) from either the open or closed state reference structures.

output_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/TeaA/quick_auto_validation_results"

open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc"

"""

import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align, rms
from sklearn.decomposition import PCA

# Paths defined in the script
output_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories"
open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc"
# trajectory_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_initial_reimaged.xtc"

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
open_aligner = align.AlignTraj(u_traj, u_open, select=ca_atoms, in_memory=True).run()
open_rmsd = rms.RMSD(u_traj, u_open, select=ca_atoms)
open_rmsd.run()
rmsd_to_open = open_rmsd.rmsd[:, 2]  # column 2 contains RMSD values

# Align trajectory to closed state and calculate RMSD
closed_aligner = align.AlignTraj(u_traj, u_closed, select=ca_atoms, in_memory=True).run()
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
with mda.Writer(filtered_trajectory_path, u_traj.atoms.n_atoms) as W:
    for i in selected_indices:
        u_traj.trajectory[i]
        W.write(u_traj.atoms)

print(f"Filtered trajectory saved to {filtered_trajectory_path}")

# Re-align trajectory to open state for PCA
align.AlignTraj(u_traj, u_open, select=ca_atoms, in_memory=True).run()

# Collect CA coordinates for PCA
all_coordinates = []
for ts in u_traj.trajectory:
    all_coordinates.append(traj_ca.positions.copy())
all_coordinates = np.array(all_coordinates)

# Load filtered trajectory and align it to the same reference
u_filtered = mda.Universe(topology_path, filtered_trajectory_path)
filtered_ca = u_filtered.select_atoms(ca_atoms)
align.AlignTraj(u_filtered, u_open, select=ca_atoms, in_memory=True).run()

# Collect CA coordinates for filtered trajectory
filtered_coordinates = []
for ts in u_filtered.trajectory:
    filtered_coordinates.append(filtered_ca.positions.copy())
filtered_coordinates = np.array(filtered_coordinates)

# Get pairwise CA distances for original trajectory
n_atoms = traj_ca.n_atoms
pairwise_distances = []

for frame_coords in all_coordinates:
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
filtered_pairwise_distances = []
for frame_coords in filtered_coordinates:
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(frame_coords[i] - frame_coords[j])
            distances.append(dist)
    filtered_pairwise_distances.append(distances)
filtered_pairwise_distances = np.array(filtered_pairwise_distances)

# Perform PCA on pairwise distances
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

print(f"RMSD plot saved to {rmsd_plot_path}")
