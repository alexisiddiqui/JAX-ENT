import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from jaxent.tests.test_utils import get_inst_path


def test_efficient_k_cluster_cli():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "cluster_output"
        os.makedirs(output_dir)

        inst_dir = get_inst_path(Path(__file__).parent.parent.parent.parent)
        topology_path = inst_dir / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
        trajectory_path = inst_dir / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

        if not topology_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(
                f"Required files not found: {topology_path} or {trajectory_path}"
            )

        command = [
            "jaxent-kCluster",
            "--topology_path",
            str(topology_path),
            "--trajectory_paths",
            str(trajectory_path),
            "--number_of_clusters",
            "5",
            "--num_components",
            "3",
            "--output_dir",
            str(output_dir),
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"

        # Check for expected output directories and files
        assert (output_dir / "plots").exists()
        assert (output_dir / "data").exists()
        assert (output_dir / "clusters").exists()

        # Check for specific output files
        assert (output_dir / "plots" / "pca_clusters.png").exists()
        assert (output_dir / "plots" / "pca_3d.png").exists()
        assert (output_dir / "plots" / "cluster_distribution.png").exists()
        assert (output_dir / "clusters" / "all_clusters.xtc").exists()
        assert (output_dir / "clusters" / "frame_to_cluster.csv").exists()
        assert (output_dir / "cluster_trajectory.log").exists()  # Log file is always created

        assert (output_dir / "data" / "pca_coordinates.npy").exists()
        assert (output_dir / "data" / "cluster_labels.npy").exists()
        assert (output_dir / "data" / "cluster_centers.npy").exists()

        # Optional: Load and check content of some output files
        pca_coords = np.load(output_dir / "data" / "pca_coordinates.npy")
        cluster_labels = np.load(output_dir / "data" / "cluster_labels.npy")
        cluster_centers = np.load(output_dir / "data" / "cluster_centers.npy")

        assert pca_coords.shape[1] == 3  # Should have 3 components
        assert len(cluster_labels) > 0
        assert cluster_centers.shape[0] == 5  # 5 clusters
        assert cluster_centers.shape[1] == 3  # 3 components
