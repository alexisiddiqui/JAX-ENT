import subprocess
import tempfile
from pathlib import Path

import numpy as np

from jaxent.src.utils.jit_fn import jit_Guard


@jit_Guard.test_isolation()
def test_cli_forward_bv_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define paths
        tmp_path = Path(tmpdir)
        featurise_output_dir = tmp_path / "featurisation_output"
        forward_output_dir = tmp_path / "forward_output"

        # Use absolute paths from the project root
        topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        trajectory_path = (
            "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
        )
        featurise_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/featurise.py"
        forward_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/forward.py"

        # --- Run Featurise CLI ---
        featurise_command = [
            "python",
            featurise_script_path,
            "--top_path",
            str(topology_path),
            "--trajectory_path",
            str(trajectory_path),
            "--output_dir",
            str(featurise_output_dir),
            "--name",
            "test_bv_featurisation",
            "bv",
            "--temperature",
            "300.0",
            "--ph",
            "7.0",
            "--num_timepoints",
            "1",
            "--residue_ignore",
            "-2",
            "2",
            "--peptide_trim",
            "2",
            "--mda_selection_exclusion",
            "resname PRO or resid 1",
        ]

        print(f"Running featurise command: {' '.join(featurise_command)}")
        featurise_result = subprocess.run(
            featurise_command, capture_output=True, text=True, check=False
        )

        print("Featurise STDOUT:", featurise_result.stdout)
        print("Featurise STDERR:", featurise_result.stderr)

        assert featurise_result.returncode == 0, (
            f"Featurise CLI command failed: {featurise_result.stderr}"
        )
        features_npz_path = featurise_output_dir / "features.npz"
        topology_json_path = featurise_output_dir / "topology.json"
        assert features_npz_path.exists()
        assert topology_json_path.exists()

        # --- Run Forward CLI ---
        num_simulations = 1
        num_frames = 500  # Based on the trajectory used in featurisation
        np.random.seed(42)
        random_weights = np.random.rand(num_frames)
        random_weights /= np.sum(random_weights)  # Normalize to sum to 1

        forward_command = [
            "python",
            forward_script_path,
            "--features_path",
            str(features_npz_path),
            "--topology_path",
            str(topology_json_path),
            "--output_dir",
            str(forward_output_dir),
            "--output_prefix",
            "test_bv_forward",
            "--num_simulations",
            str(num_simulations),
            "bv",
            "--num_timepoints",
            "0",
            "--bv_bc",
            "0.3",
            "--bv_bh",
            "2.0",
            "--temperature",
            "300.0",
            "--frame_weights",
            *[str(w) for w in random_weights],
        ]

        print(f"Running forward command: {' '.join(forward_command[:20])}...")
        forward_result = subprocess.run(
            forward_command, capture_output=True, text=True, check=False
        )

        print("Forward STDOUT:", forward_result.stdout)
        print("Forward STDERR:", forward_result.stderr)

        assert forward_result.returncode == 0, (
            f"Forward CLI command failed: {forward_result.stderr}"
        )

        # --- Verify Output ---
        for i in range(num_simulations):
            prediction_path = forward_output_dir / f"test_bv_forward_{i}.npz"
            assert prediction_path.exists(), f"Output file for simulation {i} not found."

            predictions_data = np.load(prediction_path)
            assert "predictions" in predictions_data
            predictions = predictions_data["predictions"]

            print(f"\n--- Prediction Summary Statistics (Sim {i}) ---")
            print(f"Shape: {predictions.shape}")
            print(f"Mean: {np.mean(predictions):.4f}")
            print(f"Standard Deviation: {np.std(predictions):.4f}")
            print(f"Min: {np.min(predictions):.4f}")
            print(f"Max: {np.max(predictions):.4f}")

            # Basic assertions on the output
            num_residues = 52  # From featurisation step
            num_timepoints = 1
            assert predictions.shape == (num_residues,)
            assert np.all(predictions >= 0), "Predictions should be non-negative"
            assert np.all(predictions <= 100), (
                "Predictions should be protection factors for BV model with 1 or 0 timepoints"
            )


@jit_Guard.test_isolation()
def test_cli_forward_bv_model_uptake():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define paths
        tmp_path = Path(tmpdir)
        featurise_output_dir = tmp_path / "featurisation_output"
        forward_output_dir = tmp_path / "forward_output"

        # Use absolute paths from the project root
        topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        trajectory_path = (
            "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
        )
        featurise_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/featurise.py"
        forward_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/forward.py"

        # --- Run Featurise CLI ---
        featurise_command = [
            "python",
            featurise_script_path,
            "--top_path",
            str(topology_path),
            "--trajectory_path",
            str(trajectory_path),
            "--output_dir",
            str(featurise_output_dir),
            "--name",
            "test_bv_featurisation",
            "bv",
            "--temperature",
            "300.0",
            "--ph",
            "7.0",
            "--num_timepoints",
            "1",
            "--residue_ignore",
            "-2",
            "2",
            "--peptide_trim",
            "2",
            "--mda_selection_exclusion",
            "resname PRO or resid 1",
        ]

        print(f"Running featurise command: {' '.join(featurise_command)}")
        featurise_result = subprocess.run(
            featurise_command, capture_output=True, text=True, check=False
        )

        print("Featurise STDOUT:", featurise_result.stdout)
        print("Featurise STDERR:", featurise_result.stderr)

        assert featurise_result.returncode == 0, (
            f"Featurise CLI command failed: {featurise_result.stderr}"
        )
        features_npz_path = featurise_output_dir / "features.npz"
        topology_json_path = featurise_output_dir / "topology.json"
        assert features_npz_path.exists()
        assert topology_json_path.exists()

        # --- Run Forward CLI ---
        num_simulations = 1
        num_frames = 500  # Based on the trajectory used in featurisation
        np.random.seed(42)
        random_weights = np.random.rand(num_frames)
        random_weights /= np.sum(random_weights)  # Normalize to sum to 1

        forward_command = [
            "python",
            forward_script_path,
            "--features_path",
            str(features_npz_path),
            "--topology_path",
            str(topology_json_path),
            "--output_dir",
            str(forward_output_dir),
            "--output_prefix",
            "test_bv_forward",
            "--num_simulations",
            str(num_simulations),
            "bv",
            "--num_timepoints",
            "2",
            "--bv_bc",
            "0.3",
            "--bv_bh",
            "2.0",
            "--temperature",
            "300.0",
            "--timepoints",
            "10.0",
            "120.0",
            "--frame_weights",
            *[str(w) for w in random_weights],
        ]

        print(f"Running forward command: {' '.join(forward_command[:20])}...")
        forward_result = subprocess.run(
            forward_command, capture_output=True, text=True, check=False
        )

        print("Forward STDOUT:", forward_result.stdout)
        print("Forward STDERR:", forward_result.stderr)

        assert forward_result.returncode == 0, (
            f"Forward CLI command failed: {forward_result.stderr}"
        )

        # --- Verify Output ---
        for i in range(num_simulations):
            prediction_path = forward_output_dir / f"test_bv_forward_{i}.npz"
            assert prediction_path.exists(), f"Output file for simulation {i} not found."

            predictions_data = np.load(prediction_path)
            assert "predictions" in predictions_data
            predictions = predictions_data["predictions"]

            print(f"\n--- Prediction Summary Statistics (Sim {i}) ---")
            print(f"Shape: {predictions.shape}")
            print(f"Mean: {np.mean(predictions):.4f}")
            print(f"Standard Deviation: {np.std(predictions):.4f}")
            print(f"Min: {np.min(predictions):.4f}")
            print(f"Max: {np.max(predictions):.4f}")

            # Basic assertions on the output
            num_residues = 52  # From featurisation step
            num_timepoints = 2
            assert predictions.shape == (num_timepoints, num_residues)
            assert np.all(predictions >= 0), "Predictions should be non-negative"
            assert np.all(predictions <= 1), (
                "Predictions should be fractional uptake (between 0 and 1)"
            )
