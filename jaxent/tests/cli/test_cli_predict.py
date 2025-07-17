import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np

import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
from jaxent.src.utils.jit_fn import jit_Guard

@jit_Guard.test_isolation()
def test_cli_predict_bv_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define paths
        featurise_output_dir = Path(tmpdir) / "featurisation_output"
        predict_output_dir = Path(tmpdir) / "prediction_output"
        
        topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
        trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

        # --- Run Featurise CLI ---
        featurise_command = [
            "python",
            "/home/alexi/Documents/JAX-ENT/jaxent/cli/featurise.py",
            "--top_path", str(topology_path),
            "--trajectory_path", str(trajectory_path),
            "--output_dir", str(featurise_output_dir),
            "--name", "test_bv_featurisation",
            "bv",
            "--temperature", "300.0",
            "--ph", "7.0",
            "--num_timepoints", "3",
            "--timepoints", "0.167", "1.0", "10.0",
            "--residue_ignore", "-2", "2",
            "--peptide_trim", "2",
            "--mda_selection_exclusion", "resname PRO or resid 1"
        ]

        print(f"Running featurise command: {' '.join(featurise_command)}")
        featurise_result = subprocess.run(featurise_command, capture_output=True, text=True, check=False)

        print("Featurise STDOUT:", featurise_result.stdout)
        print("Featurise STDERR:", featurise_result.stderr)

        assert featurise_result.returncode == 0, f"Featurise CLI command failed: {featurise_result.stderr}"
        features_npz_path = featurise_output_dir / "features.npz"
        topology_json_path = featurise_output_dir / "topology.json"
        assert features_npz_path.exists()
        assert topology_json_path.exists()

        # --- Run Predict CLI ---
        predict_command = [
            "python",
            "/home/alexi/Documents/JAX-ENT/jaxent/cli/predict.py",
            "--features_path", str(features_npz_path),
            "--topology_path", str(topology_json_path),
            "--output_dir", str(predict_output_dir),
            "--output_name", "test_bv_predictions",
            "bv",
            "--bv_bc", "0.35",
            "--bv_bh", "2.0",
            "--temperature", "300.0",
            "--num_timepoints", "3",
            "--timepoints", "0.167", "1.0", "10.0"
        ]

        print(f"Running predict command: {' '.join(predict_command)}")
        predict_result = subprocess.run(predict_command, capture_output=True, text=True, check=False)

        print("Predict STDOUT:", predict_result.stdout)
        print("Predict STDERR:", predict_result.stderr)

        assert predict_result.returncode == 0, f"Predict CLI command failed: {predict_result.stderr}"
        predictions_npz_path = predict_output_dir / "test_bv_predictions.npz"
        assert predictions_npz_path.exists()

        # --- Verify Output and Summary Statistics ---
        predictions_data = np.load(predictions_npz_path)
        predictions = predictions_data["predictions"]

        print("\n--- Prediction Summary Statistics ---")
        print(f"Shape: {predictions.shape}")
        print(f"Mean: {np.mean(predictions):.4f}")
        print(f"Standard Deviation: {np.std(predictions):.4f}")
        print(f"Min: {np.min(predictions):.4f}")
        print(f"Max: {np.max(predictions):.4f}")

        # Basic assertions on the output
        assert predictions.shape[0] > 0, "Predictions array should not be empty"
        assert np.all(predictions >= 0), "Predictions should be non-negative"
        assert np.all(predictions <= 1), "Predictions should be between 0 and 1 (fractional uptake)"
