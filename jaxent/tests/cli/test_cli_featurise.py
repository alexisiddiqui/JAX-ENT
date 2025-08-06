import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from custom_types.features import AbstractFeatures

from jaxent.tests.test_utils import get_inst_path


def test_featurise_cli_bv_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "featurisation_output"

        inst_dir = get_inst_path(Path(__file__).parent.parent.parent.parent)
        topology_path = inst_dir / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
        trajectory_path = inst_dir / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

        if not topology_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(
                f"Required files not found: {topology_path} or {trajectory_path}"
            )

        command = [
            "jaxent-featurise",
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
            "resname PRO",
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
        features = AbstractFeatures.load(features_path)
        with open(topology_path, "r") as f:
            topology = json.load(f)

        # Summary statistics and assertions
        num_residues = 52  # 58 residues in BPTI - 5 prolines and resid 1
        num_frames = 500
        num_timepoints = 1

        assert features.k_ints is not None
        assert features.heavy_contacts is not None
        assert features.acceptor_contacts is not None
        num_residues = 52  # 58 residues in BPTI - 5 prolines and resid 1
        num_frames = 500
        assert features.k_ints.shape == (num_residues,)
        assert features.heavy_contacts.shape == (num_residues, num_frames)
        assert features.acceptor_contacts.shape == (num_residues, num_frames)

        assert topology["topology_count"] == num_residues

        # Check for non-negative values
        assert np.all(features.k_ints >= 0)
        assert np.all(features.heavy_contacts >= 0)
        assert np.all(features.acceptor_contacts >= 0)
