import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jaxent.src.utils.jit_fn import jit_Guard


@pytest.fixture(scope="module")
def featurised_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        featurise_output_dir = tmp_path / "featurisation_output"
        featurise_output_dir.mkdir()

        topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        trajectory_path = (
            "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
        )
        featurise_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/featurise.py"

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

        yield features_npz_path, topology_json_path


@jit_Guard.test_isolation()
def test_cli_forward_bv_model(featurised_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define paths
        tmp_path = Path(tmpdir)
        forward_output_dir = tmp_path / "forward_output"
        forward_output_dir.mkdir()

        # Use absolute paths from the project root
        forward_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/forward.py"

        features_npz_path, topology_json_path = featurised_data

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
def test_cli_forward_bv_model_uptake(featurised_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define paths
        tmp_path = Path(tmpdir)
        forward_output_dir = tmp_path / "forward_output"
        forward_output_dir.mkdir()

        # Use absolute paths from the project root
        forward_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/forward.py"

        features_npz_path, topology_json_path = featurised_data

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


@jit_Guard.test_isolation()
def test_cli_forward_multiple_simulations(featurised_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        forward_output_dir = tmp_path / "forward_output_multi"
        forward_output_dir.mkdir()

        forward_script_path = "/home/alexi/Documents/JAX-ENT/jaxent/cli/forward.py"
        features_npz_path, topology_json_path = featurised_data

        num_simulations = 3
        num_frames = 500
        np.random.seed(42)
        random_weights = np.random.rand(num_frames)
        random_weights /= np.sum(random_weights)

        # Varying parameters for each simulation
        bv_bcs = [0.3, 0.4, 0.5]
        bv_bhs = [1.5, 2.0, 2.5]
        temperatures = [290.0, 300.0, 310.0]
        num_timepoints_list = [0, 1, 2]
        timepoints_list = [
            None,
            "10.0",
            "10.0 120.0",
        ]  # Note: timepoints are passed as strings to CLI

        for i in range(num_simulations):
            current_timepoints_args = []
            if timepoints_list[i] is not None:
                current_timepoints_args = ["--timepoints"] + timepoints_list[i].split()

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
                f"test_multi_sim_{i}",  # Unique prefix for each simulation
                "--num_simulations",
                "1",  # Run one simulation at a time to test individual parameter sets
                "bv",
                "--num_timepoints",
                str(num_timepoints_list[i]),
                "--bv_bc",
                str(bv_bcs[i]),
                "--bv_bh",
                str(bv_bhs[i]),
                "--temperature",
                str(temperatures[i]),
                *current_timepoints_args,
                "--frame_weights",
                *[str(w) for w in random_weights],
            ]

            print(f"Running forward command (Sim {i}): {' '.join(forward_command[:20])}...")
            forward_result = subprocess.run(
                forward_command, capture_output=True, text=True, check=False
            )

            print(f"Forward STDOUT (Sim {i}):", forward_result.stdout)
            print(f"Forward STDERR (Sim {i}):", forward_result.stderr)

            assert forward_result.returncode == 0, (
                f"Forward CLI command failed for simulation {i}: {forward_result.stderr}"
            )

            # Verify Output for current simulation
            prediction_path = (
                forward_output_dir / f"test_multi_sim_{i}_0.npz"
            )  # _0 because num_simulations is 1
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

            num_residues = 52
            expected_shape = (
                (num_residues,)
                if num_timepoints_list[i] == 0
                else (num_timepoints_list[i], num_residues)
            )
            assert predictions.shape == expected_shape, (
                f"Shape mismatch for sim {i}: Expected {expected_shape}, got {predictions.shape}"
            )
            assert np.all(predictions >= 0), f"Predictions should be non-negative for sim {i}"

            if num_timepoints_list[i] == 0:
                assert np.all(predictions <= 100), (
                    f"Predictions should be protection factors for sim {i}"
                )
            else:
                assert np.all(predictions <= 1), (
                    f"Predictions should be fractional uptake for sim {i}"
                )
