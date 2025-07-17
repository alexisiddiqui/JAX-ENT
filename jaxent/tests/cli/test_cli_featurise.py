import subprocess
import tempfile
from pathlib import Path
import numpy as np
import json


def test_featurise_cli_bv_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "featurisation_output"

        topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        trajectory_path = (
            "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
        )

        command = [
            "python",
            "/home/alexi/Documents/JAX-ENT/jaxent/cli/featurise.py",
            "--top_path",
            str(topology_path),
            "--trajectory_path",
            str(trajectory_path),
            "--output_dir",
            str(output_dir),
            "--name",
            "test_bv",
            "bv",
            "--temperature",
            "300.0",
            "--ph",
            "7.0",
            "--num_timepoints",
            "1",
            "--timepoints",
            "0.167",
            "--residue_ignore",
            "-2",
            "2",
            "--mda_selection_exclusion",
            "resname PRO or resid 1",
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
        
        features_path = output_dir / "features.npz"
        topology_path = output_dir / "topology.json"

        assert features_path.exists()
        assert topology_path.exists()

        # Load and check content of the output files
        features = np.load(features_path)
        with open(topology_path, 'r') as f:
            topology = json.load(f)

        # Summary statistics and assertions
        num_residues = 52  # 58 residues in BPTI - 5 prolines and resid 1
        num_frames = 500
        num_timepoints = 1

        assert "k_ints" in features
        assert "heavy_contacts" in features
        assert "acceptor_contacts" in features

        assert features["k_ints"].shape == (num_residues,)
        assert features["heavy_contacts"].shape == (num_residues, num_frames)
        assert features["acceptor_contacts"].shape == (num_residues, num_frames)
        
        assert topology['topology_count'] == num_residues

        # Check for non-negative values
        assert np.all(features["k_ints"] >= 0)
        assert np.all(features["heavy_contacts"] >= 0)
        assert np.all(features["acceptor_contacts"] >= 0)
