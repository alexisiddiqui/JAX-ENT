"""
This script is used to get the HDXer AutoValidation data from the zenodo repository: https://zenodo.org/records/3629554
This is from the paper:
Interpretation of HDX Data by Maximum-Entropy Reweighting of Simulated Structural Ensembles
https://doi.org/10.1016/j.bpj.2020.02.005

Usage:
    python get_HDXer_AutoValidation_data.py [--interval N]

Arguments:
    --interval N   Slicing interval for trajectories (default: 100)
"""

import os
import sys
import tarfile
import urllib.request

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms
from tqdm import tqdm


# Define a progress bar callback for urllib
class DownloadProgressBar:
    def __init__(self, total=None):
        self.pbar = None
        self.total = total
        self.downloaded = 0

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.total = total_size
            self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=file_name)
        downloaded = block_num * block_size
        if downloaded < self.total:
            self.pbar.update(downloaded - self.downloaded)
            self.downloaded = downloaded
        else:
            self.pbar.close()


def slice_trajectories(data_dir, interval=100):
    """
    Slice trajectories to keep only every `interval`th frame using MDAnalysis.

    Parameters:
    data_dir (str): Directory containing the trajectory and topology files
    interval (int): Interval for slicing frames
    """
    print(f"Slicing trajectories to keep every {interval}th frame...")

    # Create a directory for sliced trajectories
    sliced_dir = os.path.join(data_dir, "sliced_trajectories")
    os.makedirs(sliced_dir, exist_ok=True)

    # Define topology-trajectory pairs based on naming conventions
    traj_pairs = []

    # For LeuT_WT files
    wt_psf = os.path.join(data_dir, "LeuT_WT_protonly.psf")
    for i in range(1, 4):
        wt_dcd = os.path.join(data_dir, f"LeuT_WT_protonly_run{i}.dcd")
        if os.path.exists(wt_psf) and os.path.exists(wt_dcd):
            traj_pairs.append((wt_psf, wt_dcd, f"LeuT_WT_run{i}_sliced.dcd"))

    # For LeuT_Y268A files
    y268a_psf = os.path.join(data_dir, "LeuT_Y268A_protonly.psf")
    for i in range(1, 4):
        y268a_dcd = os.path.join(data_dir, f"LeuT_Y268A_protonly_run{i}.dcd")
        if os.path.exists(y268a_psf) and os.path.exists(y268a_dcd):
            traj_pairs.append((y268a_psf, y268a_dcd, f"LeuT_Y268A_run{i}_sliced.dcd"))

    # For TeaA files - assuming the topology/trajectory pairings based on naming
    closed_pdb = os.path.join(data_dir, "TeaA_ref_closed_state.pdb")
    closed_xtc = os.path.join(data_dir, "TeaA_closed_reimaged.xtc")
    if os.path.exists(closed_pdb) and os.path.exists(closed_xtc):
        traj_pairs.append((closed_pdb, closed_xtc, "TeaA_closed_sliced.xtc"))

    open_pdb = os.path.join(data_dir, "TeaA_ref_open_state.pdb")
    open_xtc = os.path.join(data_dir, "TeaA_open_reimaged.xtc")
    if os.path.exists(open_pdb) and os.path.exists(open_xtc):
        traj_pairs.append((open_pdb, open_xtc, "TeaA_open_sliced.xtc"))

    initial_xtc = os.path.join(data_dir, "TeaA_initial_reimaged.xtc")
    if os.path.exists(open_pdb) and os.path.exists(initial_xtc):
        traj_pairs.append((open_pdb, initial_xtc, "TeaA_initial_sliced.xtc"))

    # Process each pair to slice trajectories
    for topology, trajectory, output_name in traj_pairs:
        print(f"Processing {os.path.basename(trajectory)}...")

        try:
            # Load the universe with the topology and trajectory
            u = mda.Universe(topology, trajectory)

            # Create writer for the output trajectory
            output_path = os.path.join(sliced_dir, output_name)

            # Determine the file format based on extension
            if output_name.endswith(".dcd"):
                writer = mda.coordinates.DCD.DCDWriter(output_path, n_atoms=len(u.atoms))
            elif output_name.endswith(".xtc"):
                writer = mda.coordinates.XTC.XTCWriter(output_path, n_atoms=len(u.atoms))
            else:
                raise ValueError(f"Unsupported output format for {output_name}")

            # Select and write every `interval`th frame
            for i, ts in enumerate(u.trajectory):
                if i % interval == 0:
                    writer.write(u.atoms)

            writer.close()
            print(f"  Saved sliced trajectory to {output_path}")

        except Exception as e:
            print(f"  Error processing {os.path.basename(trajectory)}: {e}")

    print(f"Slicing complete. Sliced trajectories are available in {sliced_dir}")


def plot_trajectory_assignments(assignments, output_path, rmsd_to_closed=None, rmsd_to_open=None):
    """
    Plot state assignments along the trajectory.
    Optionally, also plot RMSD to closed and open reference for each frame.

    Parameters:
    assignments (list): List of state assignments for each frame
    output_path (str): Path to save the plot
    rmsd_to_closed (array, optional): RMSD values to closed reference
    rmsd_to_open (array, optional): RMSD values to open reference
    """
    # Determine if RMSD arrays are provided and valid
    plot_rmsd = rmsd_to_closed is not None and rmsd_to_open is not None

    if plot_rmsd:
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]}
        )
        ax_assign = axes[0]
        ax_rmsd_closed = axes[1]
        ax_rmsd_open = axes[2]
    else:
        fig, ax_assign = plt.subplots(1, 1, figsize=(12, 4))

    frame_numbers = np.arange(len(assignments))
    colors = {"closed": "blue", "open": "red", "both": "purple", "unassigned": "gray"}

    # Assignment plot
    for assignment in ["closed", "open", "both", "unassigned"]:
        mask = np.array(assignments) == assignment
        if np.any(mask):
            ax_assign.scatter(
                frame_numbers[mask],
                np.ones(np.sum(mask)),
                c=colors[assignment],
                label=assignment,
                alpha=0.6,
                s=20,
            )

    ax_assign.set_ylabel("State Assignment")
    ax_assign.set_title("State Assignment Along Trajectory")
    ax_assign.legend()
    ax_assign.set_ylim(0.5, 1.5)
    ax_assign.set_yticks([1])
    ax_assign.set_yticklabels(["Assignment"])
    ax_assign.grid(True, alpha=0.3)

    # RMSD plots if data provided
    if plot_rmsd:
        ax_rmsd_closed.plot(frame_numbers, rmsd_to_closed, color="blue", label="RMSD to Closed")
        ax_rmsd_closed.set_ylabel("RMSD to Closed (Å)")
        ax_rmsd_closed.set_title("RMSD to Closed Reference")
        ax_rmsd_closed.grid(True, alpha=0.3)
        ax_rmsd_closed.legend()

        ax_rmsd_open.plot(frame_numbers, rmsd_to_open, color="red", label="RMSD to Open")
        ax_rmsd_open.set_ylabel("RMSD to Open (Å)")
        ax_rmsd_open.set_title("RMSD to Open Reference")
        ax_rmsd_open.grid(True, alpha=0.3)
        ax_rmsd_open.legend()

        ax_rmsd_open.set_xlabel("Frame Number")
        plt.tight_layout()
        plt.savefig(output_path.replace(".png", "_assignments.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        ax_assign.set_xlabel("Frame Number")
        plt.tight_layout()
        plt.savefig(output_path.replace(".png", "_assignments.png"), dpi=300, bbox_inches="tight")
        plt.close()


def plot_rmsd_paired_distances(
    rmsd_to_closed, rmsd_to_open, assignments, rmsd_threshold, output_path
):
    """
    Plot paired RMSD distances with cutoff lines.

    Parameters:
    rmsd_to_closed (array): RMSD values to closed reference
    rmsd_to_open (array): RMSD values to open reference
    assignments (list): State assignments for each frame
    rmsd_threshold (float): RMSD cutoff threshold
    output_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    colors = {"closed": "blue", "open": "red", "both": "purple", "unassigned": "gray"}

    for assignment in ["closed", "open", "both", "unassigned"]:
        mask = np.array(assignments) == assignment
        if np.any(mask):
            ax.scatter(
                rmsd_to_closed[mask],
                rmsd_to_open[mask],
                c=colors[assignment],
                label=assignment,
                alpha=0.6,
                s=20,
            )

    # Add cutoff lines
    ax.axhline(
        y=rmsd_threshold,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Cutoff ({rmsd_threshold} Å)",
    )
    ax.axvline(x=rmsd_threshold, color="black", linestyle="--", alpha=0.7)

    ax.set_xlabel("RMSD to Closed Reference (Å)")
    ax.set_ylabel("RMSD to Open Reference (Å)")
    ax.set_title("RMSD Paired Distance Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Make axes equal for better visualization
    max_rmsd = max(np.max(rmsd_to_closed), np.max(rmsd_to_open))
    ax.set_xlim(0, max_rmsd * 1.1)
    ax.set_ylim(0, max_rmsd * 1.1)

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_rmsd_pairs.png"), dpi=300, bbox_inches="tight")
    plt.close()


def create_TeaA_filtered_trajectories(data_dir, interval=100):
    """
    Create TeaA_filtered_sliced.xtc by filtering the initial sliced trajectory based on RMSD
    to reference structures. Only frames with RMSD ≤ 1.0 Å to either open or closed
    reference are included. Also creates plots showing assignments and RMSD distributions.
    """
    print("Creating TeaA_filtered_sliced.xtc by RMSD filtering...")

    # Define file paths
    closed_ref_pdb = os.path.join(data_dir, "TeaA_ref_closed_state.pdb")
    open_ref_pdb = os.path.join(data_dir, "TeaA_ref_open_state.pdb")

    sliced_dir = os.path.join(data_dir, "sliced_trajectories")
    initial_sliced_xtc = os.path.join(sliced_dir, "TeaA_initial_sliced.xtc")

    filtered_sliced_xtc = os.path.join(sliced_dir, "TeaA_filtered_sliced.xtc")

    # Check that all required files exist
    required_files = [closed_ref_pdb, open_ref_pdb, initial_sliced_xtc]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"  Missing required files: {missing_files}")
        print("  Skipping TeaA_filtered creation.")
        return

    # Load reference structures
    print("  Loading reference structures...")
    closed_ref = mda.Universe(closed_ref_pdb)
    open_ref = mda.Universe(open_ref_pdb)

    # Load initial trajectory
    print("  Loading initial trajectory...")
    u_initial = mda.Universe(open_ref_pdb, initial_sliced_xtc)  # Use open_ref as topology

    # Select protein atoms for RMSD calculation (using CA atoms for efficiency)
    mobile_ca = u_initial.select_atoms("name CA")
    closed_ref_ca = closed_ref.select_atoms("name CA")
    open_ref_ca = open_ref.select_atoms("name CA")

    # Calculate RMSD to closed reference
    print("  Calculating RMSD to closed reference...")
    rmsd_closed = rms.RMSD(mobile_ca, closed_ref_ca, select="name CA")
    rmsd_closed.run()

    # Calculate RMSD to open reference
    print("  Calculating RMSD to open reference...")
    rmsd_open = rms.RMSD(mobile_ca, open_ref_ca, select="name CA")
    rmsd_open.run()

    # Extract RMSD values (skip the time column)
    rmsd_to_closed_array = rmsd_closed.rmsd[:, 2]  # Column 2 contains RMSD values
    rmsd_to_open_array = rmsd_open.rmsd[:, 2]  # Column 2 contains RMSD values

    # Assign states based on RMSD threshold
    rmsd_threshold = 1.0  # Angstroms
    assignments = []
    filtered_frames = []
    frame_indices = []

    print("  Assigning states and filtering frames...")
    for i, (rmsd_closed_val, rmsd_open_val) in enumerate(
        zip(rmsd_to_closed_array, rmsd_to_open_array)
    ):
        # Assign state based on RMSD
        if rmsd_closed_val <= rmsd_threshold and rmsd_open_val <= rmsd_threshold:
            assignment = "both"
        elif rmsd_closed_val <= rmsd_threshold:
            assignment = "closed"
        elif rmsd_open_val <= rmsd_threshold:
            assignment = "open"
        else:
            assignment = "unassigned"

        assignments.append(assignment)

        # Keep frame if it passes the filter (assigned to either state)
        if assignment != "unassigned":
            # Need to go to the specific frame to get coordinates
            u_initial.trajectory[i]
            filtered_frames.append(u_initial.atoms.positions.copy())
            frame_indices.append(i)

    # Create plots
    print("  Creating analysis plots...")
    plot_path = os.path.join(os.path.dirname(__file__), "_output", "TeaA_RMSD_analysis.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # Create assignment plot (now with RMSD arrays)
    plot_trajectory_assignments(assignments, plot_path, rmsd_to_closed_array, rmsd_to_open_array)
    print(f"  Saved assignment plot to {plot_path.replace('.png', '_assignments.png')}")

    # Create paired distance plot
    plot_rmsd_paired_distances(
        rmsd_to_closed_array, rmsd_to_open_array, assignments, rmsd_threshold, plot_path
    )
    print(f"  Saved RMSD pairs plot to {plot_path.replace('.png', '_rmsd_pairs.png')}")

    # Print summary statistics
    assignment_counts = {
        assignment: assignments.count(assignment) for assignment in set(assignments)
    }
    total_frames = len(assignments)

    print("  Assignment summary:")
    for assignment, count in assignment_counts.items():
        percentage = (count / total_frames) * 100
        print(f"    {assignment}: {count} frames ({percentage:.1f}%)")

    # Write filtered trajectory
    if filtered_frames:
        print(f"  Writing {len(filtered_frames)} filtered frames to {filtered_sliced_xtc}")

        with mda.coordinates.XTC.XTCWriter(
            filtered_sliced_xtc, n_atoms=len(u_initial.atoms)
        ) as writer:
            for positions in tqdm(filtered_frames, desc="Writing frames"):
                # Set positions and write frame
                u_initial.atoms.positions = positions
                writer.write(u_initial.atoms)

        print(f"  Successfully created filtered trajectory with {len(filtered_frames)} frames")

        # Save frame mapping
        frame_mapping_path = os.path.join(sliced_dir, "TeaA_filtered_frame_mapping.txt")
        with open(frame_mapping_path, "w") as f:
            f.write("# Mapping of filtered frame indices to original frame indices\n")
            f.write("# FilteredFrame\tOriginalFrame\tAssignment\tRMSD_Closed\tRMSD_Open\n")
            for filtered_idx, orig_idx in enumerate(frame_indices):
                f.write(
                    f"{filtered_idx}\t{orig_idx}\t{assignments[orig_idx]}\t"
                    f"{rmsd_to_closed_array[orig_idx]:.3f}\t{rmsd_to_open_array[orig_idx]:.3f}\n"
                )
        print(f"  Saved frame mapping to {frame_mapping_path}")

    else:
        print("  No frames passed the RMSD filter!")


# Main script
zendo_webpage = "https://zenodo.org/records/3629554"
file_name = "Reproducibility_pack_v2.tar.gz"
download_url = f"https://zenodo.org/records/3629554/files/{file_name}?download=1"
output_dir = os.path.join(os.path.dirname(__file__), "_Bradshaw")

# Define trajectory directory path
traj_dir = os.path.join(output_dir, "Reproducibility_pack_v2/data/trajectories")

# Check if trajectory directory already exists
if os.path.exists(traj_dir):
    print(f"Found existing trajectory directory: {traj_dir}")
    print("Skipping download and extraction.")
else:
    # Create output directory
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Download the file
    print(f"Downloading {file_name} from Zenodo...")
    try:
        temp_file_path = os.path.join(output_dir, file_name)
        # Download with progress bar
        urllib.request.urlretrieve(download_url, temp_file_path, reporthook=DownloadProgressBar())

        # Extract the tar.gz file
        print(f"Extracting {file_name}...")
        with tarfile.open(temp_file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

        # Remove the tar.gz file after extraction
        os.remove(temp_file_path)

        print(f"Download and extraction complete. Data is available in {output_dir}")

        # Verify trajectory directory exists after extraction
        if not os.path.exists(traj_dir):
            print(f"Warning: Expected trajectory directory {traj_dir} not found after extraction.")
            sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except tarfile.TarError as e:
        print(f"Error extracting archive: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download, extract, and slice HDXer AutoValidation data."
    )
    parser.add_argument(
        "--interval", type=int, default=100, help="Slicing interval for trajectories (default: 100)"
    )
    args = parser.parse_args()

    # Slice trajectories
    print(f"Processing trajectory files in: {traj_dir}")
    try:
        slice_trajectories(traj_dir, interval=args.interval)
        create_TeaA_filtered_trajectories(traj_dir, interval=args.interval)

    except urllib.error.URLError as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except tarfile.TarError as e:
        print(f"Error extracting archive: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during trajectory slicing: {e}")
        sys.exit(1)
