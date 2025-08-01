"""
This script is used to calculate the uptake/protection factors for a given trajectory using the bv model.
It first featurises the trajectory, and then calculates the residue-level uptake using the bv model using the predict method.

The outputs and intermediates are saved in the specified output directory.

This script acts as a wrapper around the `featurise.py` and `predict.py` CLI scripts.

Args:
    --topology: The topology path of the trajectory.
    --trajectory: The trajectory path.
    --output: The output path where the results will be saved.
    --name: The prefix given to the files.
    --bv_bc: bv model bc constant.
    --bv_bh: bv model bh constant.
    --timepoints: The timepoints at which the uptake is calculated - if none provided then predicts protection factors.
    --temperature: Temperature for BV model (K).

"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Main function to run the featurisation and prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="""
        Calculate uptake/protection factors for a trajectory using the BV model.
        This script featurises a trajectory and then calculates residue-level
        uptake or protection factors using the predict method.
        Outputs and intermediates are saved in the specified output directory.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--topology", type=str, required=True, help="The topology path of the trajectory."
    )
    parser.add_argument("--trajectory", type=str, required=True, help="The trajectory path.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output path where the results will be saved.",
    )
    parser.add_argument(
        "--name", type=str, default="prediction", help="The prefix given to the files."
    )
    parser.add_argument("--bv_bc", type=float, default=0.35, help="BV model 'bc' constant.")
    parser.add_argument("--bv_bh", type=float, default=2.0, help="BV model 'bh' constant.")
    parser.add_argument(
        "--timepoints",
        type=float,
        nargs="+",
        default=None,
        help="Timepoints for uptake calculation. If not provided, predicts protection factors.",
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0, help="Temperature for BV model (K)."
    )

    args = parser.parse_args()

    # Assume the script is run from the project root
    project_root = Path.cwd()
    cli_dir = project_root / "jaxent" / "cli"
    featurise_script = cli_dir / "featurise.py"
    predict_script = cli_dir / "predict.py"

    if not featurise_script.exists() or not predict_script.exists():
        print(
            "Error: CLI scripts not found. Make sure you are running this script "
            f"from the project root directory.\nExpected scripts in: {cli_dir}"
        )
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    featurise_output_dir = output_dir / "featurisation_output"
    featurise_output_dir.mkdir(exist_ok=True)

    # --- Run Featurise CLI ---
    featurise_num_timepoints = "1" if args.timepoints else "0"
    featurise_command = [
        sys.executable,
        str(featurise_script),
        "--top_path",
        str(args.topology),
        "--trajectory_path",
        str(args.trajectory),
        "--output_dir",
        str(featurise_output_dir),
        "--name",
        f"{args.name}_featurisation",
        "bv",
        "--num_timepoints",
        featurise_num_timepoints,
        "--temperature",
        str(args.temperature),
    ]

    print(f"Running featurise command: {' '.join(featurise_command)}")
    featurise_result = subprocess.run(
        featurise_command, capture_output=True, text=True, check=False
    )

    print("Featurise STDOUT:", featurise_result.stdout)
    if featurise_result.stderr:
        print("Featurise STDERR:", featurise_result.stderr)

    if featurise_result.returncode != 0:
        print(f"Featurise CLI command failed with exit code {featurise_result.returncode}")
        sys.exit(1)

    features_npz_path = featurise_output_dir / "features.npz"
    topology_json_path = featurise_output_dir / "topology.json"

    if not features_npz_path.exists() or not topology_json_path.exists():
        print("Featurisation did not produce expected output files.")
        sys.exit(1)

    # --- Run Predict CLI ---
    predict_output_dir = output_dir / "prediction_output"
    predict_output_dir.mkdir(exist_ok=True)

    predict_num_timepoints = str(len(args.timepoints)) if args.timepoints else "0"
    predict_command = [
        sys.executable,
        str(predict_script),
        "--features_path",
        str(features_npz_path),
        "--topology_path",
        str(topology_json_path),
        "--output_dir",
        str(predict_output_dir),
        "--output_name",
        args.name,
        "bv",
        "--bv_bc",
        str(args.bv_bc),
        "--bv_bh",
        str(args.bv_bh),
        "--num_timepoints",
        predict_num_timepoints,
        "--temperature",
        str(args.temperature),
    ]

    if args.timepoints:
        predict_command.extend(["--timepoints"] + [str(t) for t in args.timepoints])

    print(f"\nRunning predict command: {' '.join(predict_command)}")
    predict_result = subprocess.run(predict_command, capture_output=True, text=True, check=False)

    print("Predict STDOUT:", predict_result.stdout)
    if predict_result.stderr:
        print("Predict STDERR:", predict_result.stderr)

    if predict_result.returncode != 0:
        print(f"Predict CLI command failed with exit code {predict_result.returncode}")
        sys.exit(1)

    predictions_npz_path = predict_output_dir / f"{args.name}.npz"
    if not predictions_npz_path.exists():
        print("Prediction did not produce expected output file.")
        sys.exit(1)

    print("\nPrediction complete.")
    print(f"Final predictions saved to: {predictions_npz_path}")


if __name__ == "__main__":
    main()
