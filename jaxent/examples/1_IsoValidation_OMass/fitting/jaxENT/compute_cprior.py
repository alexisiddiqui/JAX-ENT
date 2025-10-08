import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis.analysis.rms import RMSF

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import PTSerialiser
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.predict import run_predict


# --- Helper Functions ---
def plot_heatmap(
    matrix,
    title,
    filename,
    output_dir,
    cmap="viridis",
    annot=False,
    fmt=".2f",
    log_scale=False,
    eps=1e-12,
):
    plt.figure(figsize=(10, 8))
    if log_scale:
        matrix_to_plot = np.abs(np.array(matrix, dtype=float))
        matrix_to_plot[matrix_to_plot <= eps] = eps
        norm = LogNorm(vmin=matrix_to_plot.min(), vmax=matrix_to_plot.max())
        sns.heatmap(
            matrix_to_plot,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            norm=norm,
            cbar_kws={"label": "Value (log scale)"},
        )
    else:
        sns.heatmap(matrix, cmap=cmap, annot=annot, fmt=fmt, cbar_kws={"label": "Value"})
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# New helper: plot diagonal as bar chart (same style as compute_sigma)
def plot_diagonal_bar(matrix, title, filename, output_dir, log_scale=False, eps=1e-12):
    diag = np.diag(matrix).astype(float)
    indices = np.arange(len(diag))
    plt.figure(figsize=(10, 4.5))
    if log_scale:
        diag_plot = np.abs(diag)
        diag_plot[diag_plot <= eps] = eps
        plt.bar(indices, diag_plot)
        plt.yscale("log")
        plt.ylabel("Absolute diagonal value (log scale)")
    else:
        plt.bar(indices, diag)
        plt.ylabel("Diagonal value")
    plt.xlabel("Index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot C_prior covariance matrices from MD ensemble data and model predictions."
    )
    script_dir = os.path.dirname(__file__)
    features_dir = os.path.join(script_dir, "_featurise")
    traj_dir = os.path.join(
        script_dir, "../../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    )

    default_full_topology_file_bi = os.path.join(features_dir, "topology_iso_bi.json")
    default_full_features_file_bi = os.path.join(features_dir, "features_iso_bi.npz")
    default_full_topology_file_tri = os.path.join(features_dir, "topology_iso_tri.json")
    default_full_features_file_tri = os.path.join(features_dir, "features_iso_tri.npz")
    default_full_closed_pdb = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    default_full_bi_traj = os.path.join(traj_dir, "sliced_trajectories/TeaA_filtered_sliced.xtc")
    default_full_tri_traj = os.path.join(traj_dir, "sliced_trajectories/TeaA_initial_sliced.xtc")

    parser.add_argument(
        "--ensemble_name",
        nargs="+",
        default=["ISO_BI", "ISO_TRI"],
        help="Name(s) of the ensemble(s) (e.g., ISO_BI, ISO_TRI).",
    )
    parser.add_argument(
        "--topology_file",
        nargs="+",
        default=[default_full_topology_file_bi, default_full_topology_file_tri],
        help="Path(s) to the topology JSON file(s) for each ensemble.",
    )
    parser.add_argument(
        "--features_file",
        nargs="+",
        default=[default_full_features_file_bi, default_full_features_file_tri],
        help="Path(s) to the features NPZ file(s) for each ensemble.",
    )
    parser.add_argument(
        "--trajectory_file",
        nargs="+",
        default=[default_full_bi_traj, default_full_tri_traj],
        help="Path(s) to the trajectory XTC file(s) for each ensemble.",
    )
    parser.add_argument(
        "--closed_pdb",
        type=str,
        default=default_full_closed_pdb,
        help="Path to the closed PDB file (for RMSF calculation).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="_covariance_matrices",
        help="Output directory for results. Defaults to '_covariance_matrices'.",
    )

    args = parser.parse_args()

    # Validate that lists of arguments have the same length
    num_ensembles = len(args.ensemble_name)
    if not (
        len(args.topology_file) == num_ensembles
        and len(args.features_file) == num_ensembles
        and len(args.trajectory_file) == num_ensembles
    ):
        raise ValueError(
            "All lists of ensemble-specific arguments (ensemble_name, topology_file, features_file, trajectory_file) must have the same length."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Features and Topology for Ensembles ---
    print("\n--- Loading Features and Topology for Ensembles ---")
    ensembles = {}
    for i in range(num_ensembles):
        ensemble_name = args.ensemble_name[i]
        full_topology_file = args.topology_file[i]
        full_features_file = args.features_file[i]
        full_trajectory_file = args.trajectory_file[i]

        ensembles[ensemble_name] = {
            "features": BV_input_features.load(full_features_file),
            "topology": PTSerialiser.load_list_from_json(full_topology_file),
            "trajectory": full_trajectory_file,
        }

    # --- Initialize BV_model ---
    print("\n--- Initializing BV_model ---")
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params  # Default model parameters

    # --- Compute C_prior for each ensemble ---
    for ensemble_name, data in ensembles.items():
        print(f"\n--- Computing C_prior for {ensemble_name} ---")
        features = data["features"]
        topology = data["topology"]
        trajectory_path = data["trajectory"]

        # Ensure the trajectory file exists
        if not os.path.exists(trajectory_path):
            print(
                f"Warning: Trajectory file not found for {ensemble_name}: {trajectory_path}. Skipping C_prior calculations for this ensemble."
            )
            continue

        # --- Structural Covariance (from RMSF) ---
        print(f"  Computing Structural Covariance (RMSF) for {ensemble_name}...")
        universe = mda.Universe(
            args.closed_pdb, trajectory_path
        )  # Use a common topology for universe

        # Calculate RMSF for each residue
        # Assuming 'protein' selection for RMSF, adjust if specific atoms are needed
        protein_selection = universe.select_atoms("protein")

        # Check if protein_selection is empty
        if not protein_selection:
            print(
                f"  Warning: No protein atoms found in {ensemble_name} trajectory. Skipping RMSF calculation."
            )
            C_prior_structural = np.zeros((len(topology), len(topology)))
        else:
            # Align trajectory to the first frame to remove global motion for RMSF calculation
            # This is a common practice for meaningful RMSF values
            # Select "protein and name CA" for alignment
            aligner = AlignTraj(
                universe, universe, select="protein and name CA", in_memory=True
            ).run()

            # Calculate RMSF for all protein alpha carbons (CA)
            protein_ca = universe.select_atoms("protein and name CA")

            # Create a dictionary to store RMSF per residue ID, mapped to the MDAnalysis residue object
            # This ensures correct mapping even if residue IDs are not contiguous in the PDB
            rmsf_per_mda_resid = {}
            rmsf_analysis = RMSF(protein_ca).run()
            for atom, rmsf_val in zip(protein_ca, rmsf_analysis.rmsf):
                rmsf_per_mda_resid[atom.resid] = rmsf_val

            # Aggregate RMSF values for each peptide (Partial_Topology fragment)
            peptide_rmsf_values = []
            for peptide_fragment in topology:
                # Get residue IDs covered by this peptide fragment from Partial_Topology
                # Use peptide_fragment.residues for the actual list of residues
                fragment_resids = peptide_fragment.residues

                # Collect RMSF values for residues within this fragment
                rmsf_in_fragment = [rmsf_per_mda_resid.get(resid, 0.0) for resid in fragment_resids]

                # Aggregate (e.g., average) RMSF for the peptide
                if rmsf_in_fragment:
                    peptide_rmsf_values.append(np.mean(rmsf_in_fragment))
                else:
                    # If no RMSF data for any residue in fragment, assign a small default value
                    peptide_rmsf_values.append(1e-9)

            # Convert to numpy array and add a small epsilon to avoid zero variance
            peptide_rmsf_values = np.array(peptide_rmsf_values)
            # Ensure no zero values before squaring, as it represents variance
            peptide_rmsf_values[peptide_rmsf_values == 0] = 1e-9
            C_prior_structural = np.diag(peptide_rmsf_values**2)  # Variance is RMSF squared

            print(f"  C_prior_structural shape: {C_prior_structural.shape}")
            plot_heatmap(
                C_prior_structural,
                f"{ensemble_name} Structural Covariance (RMSF)",
                f"{ensemble_name}_C_prior_structural_heatmap.png",
                args.output_dir,
            )
            plot_heatmap(
                np.linalg.inv(C_prior_structural),
                f"Inverse {ensemble_name} Structural Covariance",
                f"{ensemble_name}_C_prior_structural_inv_heatmap.png",
                args.output_dir,
                cmap="magma",
                log_scale=True,
            )
            np.savez(
                os.path.join(args.output_dir, f"{ensemble_name}_C_prior_structural.npz"),
                C_prior_structural=C_prior_structural,
                C_prior_structural_inv=np.linalg.inv(C_prior_structural),
            )
            print(f"  Structural Covariance for {ensemble_name} computed and saved.")

        # Plot diagonal for structural covariance (always present: zero or real)
        plot_diagonal_bar(
            C_prior_structural,
            f"{ensemble_name} Structural Covariance diagonal",
            f"{ensemble_name}_C_prior_structural_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )

        # --- Ensemble Covariance (from model predictions) ---
        print(f"  Computing Ensemble Covariance (from model predictions) for {ensemble_name}...")
        # Use run_predict to get frame-wise model predictions
        # The BV_model predicts a single value per peptide per timepoint
        # We need to get predictions for each frame of the trajectory

        # run_predict expects a list of input_features and forward_models
        # Here we have one input_feature and one forward_model per ensemble

        # The output of run_predict is a Sequence[Output_Features]
        # For BV_model, Output_Features will contain 'y_pred' which is (num_peptides, num_timepoints, num_frames)

        # We need to extract the y_pred from the Output_Features

        # Create a dummy Simulation_Parameters for run_predict
        # The frame_weights are not used in predict method, but Simulation_Parameters requires it
        num_frames = features.features_shape[-1]
        dummy_sim_params = Simulation_Parameters(
            frame_weights=jnp.ones(num_frames) / num_frames,
            frame_mask=jnp.ones(num_frames),
            model_parameters=(model_parameters,),
            forward_model_weights=jnp.array([1.0]),
            normalise_loss_functions=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
        )

        predictions_output_features = run_predict(
            input_features=[features],
            forward_models=[bv_model],
            model_parameters=dummy_sim_params,
            validate=False,  # Skip validation as we are constructing dummy params
        )

        # Extract y_pred from the first (and only) Output_Features object
        # predictions_output_features[0].y_pred will have shape (num_timepoints, num_peptides, num_frames)
        y_pred_all_frames = predictions_output_features[0].y_pred()
        print(f"  Shape of y_pred_all_frames: {y_pred_all_frames.shape}")  # Debug print

        # --- New: compute and save residue-wise mean uptake curves (averaged across frames) ---
        # y_pred_all_frames: (num_timepoints, num_peptides, num_frames)
        # Compute mean across frames -> (num_timepoints, num_peptides)
        predicted_mean_uptake = np.array(np.mean(y_pred_all_frames, axis=2))
        # Transpose to (num_peptides, num_timepoints) for saving/plotting
        predicted_mean_uptake_per_peptide = predicted_mean_uptake.T

        # Save the predicted mean uptake array as .npy
        np.save(
            os.path.join(args.output_dir, f"{ensemble_name}_predicted_mean_uptake.npy"),
            predicted_mean_uptake_per_peptide,
        )

        # Plot all residue-wise mean uptake curves in a single figure
        timepoints = np.array(bv_config.timepoints)
        plt.figure(figsize=(8, 6))
        for i, uptake in enumerate(predicted_mean_uptake_per_peptide):
            plt.plot(timepoints, uptake, label=f"peptide_{i}")
        plt.xlabel("Time (s)")
        plt.xscale("log")
        plt.ylabel("Predicted uptake")
        plt.title(f"{ensemble_name} Predicted Mean Uptake Curves (residue-wise)")
        # If many peptides, legend may be crowded; keep small font and multiple columns
        plt.legend(frameon=False, fontsize="small", ncol=2)
        plt.savefig(
            os.path.join(args.output_dir, f"{ensemble_name}_predicted_mean_uptake_curves.png")
        )
        plt.close()
        # --- End new code ---

        # Average y_pred over timepoints for each peptide and frame
        # Shape becomes (num_peptides, num_frames)
        y_pred_avg_timepoints = np.mean(y_pred_all_frames, axis=0)

        # Calculate variance for each peptide across frames
        # Shape becomes (num_peptides,)
        peptide_prediction_variances = np.var(y_pred_avg_timepoints, axis=1) + 1e-9

        C_prior_ensemble = np.diag(peptide_prediction_variances)

        print(f"  C_prior_ensemble shape: {C_prior_ensemble.shape}")
        plot_heatmap(
            C_prior_ensemble,
            f"{ensemble_name} Ensemble Covariance (Predictions)",
            f"{ensemble_name}_C_prior_ensemble_heatmap.png",
            args.output_dir,
        )
        plot_heatmap(
            np.linalg.inv(C_prior_ensemble),
            f"Inverse {ensemble_name} Ensemble Covariance",
            f"{ensemble_name}_C_prior_ensemble_inv_heatmap.png",
            args.output_dir,
            cmap="magma",
            log_scale=True,
        )
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_C_prior_ensemble.npz"),
            C_prior_ensemble=C_prior_ensemble,
            C_prior_ensemble_inv=np.linalg.inv(C_prior_ensemble),
        )
        print(f"  Ensemble Covariance for {ensemble_name} computed and saved.")

        # Plot diagonal for ensemble covariance
        plot_diagonal_bar(
            C_prior_ensemble,
            f"{ensemble_name} Ensemble Covariance diagonal",
            f"{ensemble_name}_C_prior_ensemble_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )

        # --- Empirical Covariance (from model predictions) ---
        print(f"  Computing Empirical Covariance (from model predictions) for {ensemble_name}...")
        # This implies a dense covariance matrix between peptides based on their predicted uptake.
        # y_pred_avg_timepoints has shape (num_peptides, num_frames)
        # We need to compute the covariance matrix of this data, where each row is a peptide and columns are frames.

        # np.cov expects rows as variables and columns as observations.
        # So, y_pred_avg_timepoints is already in the correct format.
        y_pred_avg_frames = np.mean(y_pred_all_frames, axis=0)  # Shape:  num_peptides, num_frames)
        # Transpose to get (num_peptides, num_timepoints) for covariance calculation
        # y_pred_avg_frames_T = y_pred_avg_frames.T
        # y_pred_avg_frames_T_mean = np.mean(y_pred_avg_frames_T, axis=1)

        C_prior_empirical = np.cov(y_pred_avg_frames) + np.diag(
            np.full(y_pred_avg_frames.shape[0], 1e-9)
        )  # Add small diagonal for numerical stability
        # C_prior_empirical = np.cov(y_pred_avg_frames)

        print(f"  C_prior_empirical shape: {C_prior_empirical.shape}")
        plot_heatmap(
            C_prior_empirical,
            f"{ensemble_name} Empirical Covariance (Predictions)",
            f"{ensemble_name}_C_prior_empirical_heatmap.png",
            args.output_dir,
        )
        plot_heatmap(
            np.linalg.inv(C_prior_empirical),
            f"Inverse {ensemble_name} Empirical Covariance",
            f"{ensemble_name}_C_prior_empirical_inv_heatmap.png",
            args.output_dir,
            cmap="magma",
            log_scale=True,
        )
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_C_prior_empirical.npz"),
            C_prior_empirical=C_prior_empirical,
            C_prior_empirical_inv=np.linalg.inv(C_prior_empirical),
        )
        print(f"  Empirical Covariance for {ensemble_name} computed and saved.")

        # Plot diagonal for empirical covariance
        plot_diagonal_bar(
            C_prior_empirical,
            f"{ensemble_name} Empirical Covariance diagonal",
            f"{ensemble_name}_C_prior_empirical_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )

        # --- New: correlation between diagonals of the three covariance matrices ---
        # Collect diagonals (shape: 3 x num_peptides)
        diag_struct = np.diag(C_prior_structural).astype(float)
        diag_ens = np.diag(C_prior_ensemble).astype(float)
        diag_emp = np.diag(C_prior_empirical).astype(float)

        # Stack and handle any constant vectors (which would produce NaNs in corrcoef)
        diags = np.vstack([diag_struct, diag_ens, diag_emp])
        # Replace any NaN/inf with small value
        diags = np.nan_to_num(diags, nan=0.0, posinf=0.0, neginf=0.0)

        # If any row is constant, add tiny jitter to avoid zero-variance in corrcoef
        row_std = diags.std(axis=1)
        for i, s in enumerate(row_std):
            if s == 0:
                diags[i] = diags[i] + (np.random.RandomState(0).randn(diags.shape[1]) * 1e-12)

        corr_matrix = np.corrcoef(diags)  # shape (3,3)
        labels = [f"{ensemble_name}_struct", f"{ensemble_name}_ens", f"{ensemble_name}_emp"]

        # Plot correlation heatmap with annotations and labels
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            corr_matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            cmap="vlag",
            cbar_kws={"label": "Pearson r"},
            center=0,
        )
        plt.title(f"{ensemble_name} Diagonals Correlation")
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"{ensemble_name}_diagonals_correlation_heatmap.png")
        )
        plt.close()

        # Save correlation matrix
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_diagonals_correlation.npz"),
            corr_matrix=corr_matrix,
            labels=labels,
        )

    print("\n--- Covariance Matrix Generation Complete ---")


if __name__ == "__main__":
    main()
