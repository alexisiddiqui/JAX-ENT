"""
This script is used to get the HDXer AutoValidation data from the zenodo repository: https://zenodo.org/records/3629554
This is from the paper:
Interpretation of HDX Data by Maximum-Entropy Reweighting of Simulated Structural Ensembles
https://doi.org/10.1016/j.bpj.2020.02.005
"""

import os
import sys
import tarfile
import urllib.request

import MDAnalysis as mda
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


def slice_trajectories(data_dir):
    """
    Slice trajectories to keep only every 100th frame using MDAnalysis.

    Parameters:
    data_dir (str): Directory containing the trajectory and topology files
    """
    print("Slicing trajectories to keep every 100th frame...")

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

            # Select and write every 100th frame
            for i, ts in enumerate(u.trajectory):
                if i % 100 == 0:
                    writer.write(u.atoms)

            writer.close()
            print(f"  Saved sliced trajectory to {output_path}")

        except Exception as e:
            print(f"  Error processing {os.path.basename(trajectory)}: {e}")

    print(f"Slicing complete. Sliced trajectories are available in {sliced_dir}")


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

# Slice trajectories
print(f"Processing trajectory files in: {traj_dir}")
try:
    slice_trajectories(traj_dir)

except urllib.error.URLError as e:
    print(f"Error downloading file: {e}")
    sys.exit(1)
except tarfile.TarError as e:
    print(f"Error extracting archive: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during trajectory slicing: {e}")
    sys.exit(1)
