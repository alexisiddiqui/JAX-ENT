import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
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
        default=os.path.join(script_dir, "_covariance_matrices"),
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

    # Store all diagonals for correlation analysis
    all_diagonals = {}

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

        # Calculate RMSF for each residue using only CA positions
        # Select CA atoms (only these will be used for alignment and RMSF)
        protein_ca = universe.select_atoms("name CA and protein")

        # Check if any CA atoms were found
        if len(protein_ca) == 0:
            print(
                f"  Warning: No CA atoms found in {ensemble_name} trajectory. Skipping RMSF calculation."
            )
            C_prior_structural = np.zeros((len(topology), len(topology)))
            peptide_rmsf_covariance = np.zeros((len(topology), len(topology)))
        else:
            # Align trajectory to the reference PDB using CA atoms to remove global motion
            ref = mda.Universe(args.closed_pdb)
            AlignTraj(universe, ref, select="name CA and protein", in_memory=True).run()

            rmsf_analysis = RMSF(protein_ca).run()
            rmsf_per_mda_resid = {
                atom.resid: val for atom, val in zip(protein_ca, rmsf_analysis.rmsf)
            }

            peptide_rmsf_values = []
            for peptide_fragment in topology:
                fragment_resids = peptide_fragment.residues
                rmsf_in_fragment = [rmsf_per_mda_resid.get(resid, 0.0) for resid in fragment_resids]
                if rmsf_in_fragment:
                    peptide_rmsf_values.append(np.mean(rmsf_in_fragment))
                else:
                    peptide_rmsf_values.append(1e-9)

            peptide_rmsf_values = np.array(peptide_rmsf_values)
            peptide_rmsf_values[peptide_rmsf_values == 0] = 1e-9
            C_prior_structural = np.diag(peptide_rmsf_values**2)

            # Compute peptide-level RMSF covariance matrix
            # Create a mapping from residue index to CA atom index
            resid_to_ca_idx = {atom.resid: i for i, atom in enumerate(protein_ca)}

            # Get RMSF values as array
            rmsf_values = rmsf_analysis.rmsf  # (n_ca_atoms,)

            # Compute peptide-level RMSF covariance by averaging RMSF products
            num_peptides = len(topology)
            peptide_rmsf_covariance = np.zeros((num_peptides, num_peptides))

            for i, peptide_i in enumerate(topology):
                ca_indices_i = [
                    resid_to_ca_idx[resid]
                    for resid in peptide_i.residues
                    if resid in resid_to_ca_idx
                ]
                for j, peptide_j in enumerate(topology):
                    ca_indices_j = [
                        resid_to_ca_idx[resid]
                        for resid in peptide_j.residues
                        if resid in resid_to_ca_idx
                    ]
                    if ca_indices_i and ca_indices_j:
                        # Compute outer product of RMSF values and average
                        rmsf_i = rmsf_values[ca_indices_i]
                        rmsf_j = rmsf_values[ca_indices_j]
                        outer_product = np.outer(rmsf_i, rmsf_j)
                        peptide_rmsf_covariance[i, j] = np.mean(outer_product)

            print(f"  C_prior_structural shape: {C_prior_structural.shape}")
            print(f"  peptide_rmsf_covariance shape: {peptide_rmsf_covariance.shape}")
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
            plot_diagonal_bar(
                C_prior_structural,
                f"{ensemble_name} Structural Covariance Diagonal",
                f"{ensemble_name}_C_prior_structural_diagonal_bar.png",
                args.output_dir,
                log_scale=False,
            )

            # Plot peptide RMSF covariance
            plot_heatmap(
                peptide_rmsf_covariance,
                f"{ensemble_name} Peptide RMSF Covariance",
                f"{ensemble_name}_peptide_rmsf_covariance_heatmap.png",
                args.output_dir,
            )

            all_diagonals[f"{ensemble_name}_structural"] = np.diag(C_prior_structural)
            np.savez(
                os.path.join(args.output_dir, f"{ensemble_name}_C_prior_structural.npz"),
                C_prior_structural=C_prior_structural,
                C_prior_structural_inv=np.linalg.inv(C_prior_structural),
                peptide_rmsf_covariance=peptide_rmsf_covariance,
            )
            print(f"  Structural Covariance for {ensemble_name} computed and saved.")

            # --- New: Pairwise residue covariance**2 from CA motions (aligned to PDB) ---
            print(
                f"  Computing residue pairwise covariance^2 for CA motions for {ensemble_name}..."
            )
            # Gather CA positions for all frames: shape (n_frames, n_residues, 3)
            ca_atoms = protein_ca  # already CA protein selection
            n_residues = len(ca_atoms)
            # iterate frames and collect positions
            positions = []
            for ts in universe.trajectory:
                positions.append(ca_atoms.positions.copy())  # (n_residues, 3)
            positions = np.array(positions)  # (n_frames, n_residues, 3)

            n_frames = positions.shape[0]

            if n_frames < 2 or n_residues == 0:
                print(
                    f"  Warning: Not enough frames or CA atoms for residue covariance for {ensemble_name}. Skipping."
                )
                residue_covariance = np.zeros((n_residues, n_residues))
            else:
                # mean position per residue
                mean_pos = np.mean(positions, axis=0)  # (n_residues, 3)
                # compute per-frame scalar deviations (distance from mean) per residue
                deviations = np.linalg.norm(
                    positions - mean_pos[None, :, :], axis=2
                )  # (n_frames, n_residues)
                # cov_ij = sum_t (deviation_i(t) * deviation_j(t)) / (n_frames - 1)
                # deviations has shape (n_frames, n_residues) so use matrix multiplication
                cov_matrix = deviations.T.dot(deviations) / (n_frames - 1)

                # square element-wise as requested
                residue_covariance = cov_matrix**2

            print(f"  residue_covariance shape: {residue_covariance.shape}")

            # Map residue covariance to peptide covariance
            # Create a mapping from residue index to CA atom index
            resid_to_ca_idx = {atom.resid: i for i, atom in enumerate(protein_ca)}

            # Compute peptide-level covariance by averaging over residues in each peptide
            num_peptides = len(topology)
            peptide_residue_covariance = np.zeros((num_peptides, num_peptides))

            for i, peptide_i in enumerate(topology):
                ca_indices_i = [
                    resid_to_ca_idx[resid]
                    for resid in peptide_i.residues
                    if resid in resid_to_ca_idx
                ]
                for j, peptide_j in enumerate(topology):
                    ca_indices_j = [
                        resid_to_ca_idx[resid]
                        for resid in peptide_j.residues
                        if resid in resid_to_ca_idx
                    ]
                    if ca_indices_i and ca_indices_j:
                        # Average the covariance over all residue pairs between the two peptides
                        submatrix = residue_covariance[np.ix_(ca_indices_i, ca_indices_j)]
                        peptide_residue_covariance[i, j] = np.mean(submatrix)

            print(f"  peptide_residue_covariance shape: {peptide_residue_covariance.shape}")

            plot_heatmap(
                residue_covariance,
                f"{ensemble_name} Residue Pairwise Covariance^2 (CA motions)",
                f"{ensemble_name}_residue_covariance_heatmap.png",
                args.output_dir,
                cmap="viridis",
            )
            # For inverse plotting, add tiny jitter to diagonal to avoid singularity
            jitter = np.eye(residue_covariance.shape[0]) * 1e-12
            try:
                residue_covariance_inv = np.linalg.inv(residue_covariance + jitter)
            except np.linalg.LinAlgError:
                residue_covariance_inv = np.linalg.pinv(residue_covariance + jitter)
            plot_heatmap(
                np.abs(residue_covariance_inv),
                f"Inverse {ensemble_name} Residue Covariance^2 (abs, log)",
                f"{ensemble_name}_residue_covariance_inv_heatmap.png",
                args.output_dir,
                cmap="magma",
                log_scale=True,
            )
            plot_diagonal_bar(
                residue_covariance,
                f"{ensemble_name} Residue Covariance^2 Diagonal",
                f"{ensemble_name}_residue_covariance_diagonal_bar.png",
                args.output_dir,
                log_scale=False,
            )
            all_diagonals[f"{ensemble_name}_residue"] = np.diag(residue_covariance)
            np.savez(
                os.path.join(args.output_dir, f"{ensemble_name}_residue_covariance.npz"),
                residue_covariance=residue_covariance,
                residue_covariance_inv=residue_covariance_inv,
                peptide_residue_covariance=peptide_residue_covariance,
            )
            print(f"  Residue covariance^2 for {ensemble_name} computed and saved.")

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
            validate=False,
        )

        y_pred_all_frames = predictions_output_features[0].y_pred()
        print(f"  Shape of y_pred_all_frames: {y_pred_all_frames.shape}")

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

        # Average across timepoints to get (num_peptides, num_frames)
        _y_pred_all_frame = np.array(np.mean(y_pred_all_frames, axis=0))
        # Calculate variance for each peptide across frames
        # Shape becomes (num_peptides,)

        C_prior_ensemble = np.cov(_y_pred_all_frame) + np.diag(
            np.full(_y_pred_all_frame.shape[0], 1e-6)
        )  # Add small diagonal for numerical stability

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
        plot_diagonal_bar(
            C_prior_ensemble,
            f"{ensemble_name} Ensemble Covariance Diagonal",
            f"{ensemble_name}_C_prior_ensemble_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )
        all_diagonals[f"{ensemble_name}_ensemble"] = np.diag(C_prior_ensemble)
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_C_prior_ensemble.npz"),
            C_prior_ensemble=C_prior_ensemble,
            C_prior_ensemble_inv=np.linalg.inv(C_prior_ensemble),
        )
        print(f"  Ensemble Covariance for {ensemble_name} computed and saved.")

        # --- Empirical Covariance (from residue-wise ensemble average of model predictions) ---
        print(f"  Computing Empirical Covariance (from model predictions) for {ensemble_name}...")
        # This implies a dense covariance matrix between peptides based on their predicted uptake.
        # y_pred_avg_timepoints has shape (num_peptides, num_frames)
        # We need to compute the covariance matrix of this data, where each row is a peptide and columns are frames.

        # np.cov expects rows as variables and columns as observations.
        # So, y_pred_avg_timepoints is already in the correct format.
        y_pred_avg_frames = np.mean(y_pred_all_frames, axis=0)  # Shape:  num_peptides, num_frames)

        # Compute pairwise 1/R^2 matrix from ensemble mean CA coordinates
        print(
            f"  Computing pairwise 1/R^2 matrix from ensemble mean coordinates for {ensemble_name}..."
        )

        # Get CA positions for all frames from the universe (already loaded and aligned)
        ca_atoms_for_distance = protein_ca  # Use the same CA selection from RMSF

        # Collect CA positions for all frames
        ca_positions_all_frames = []
        for ts in universe.trajectory:
            ca_positions_all_frames.append(ca_atoms_for_distance.positions.copy())
        ca_positions_all_frames = np.array(ca_positions_all_frames)  # (n_frames, n_ca_atoms, 3)

        # Compute ensemble mean positions
        mean_ca_positions = np.mean(ca_positions_all_frames, axis=0)  # (n_ca_atoms, 3)

        # Map CA atoms to peptides - compute mean position for each peptide
        peptide_mean_positions = []
        for peptide_fragment in topology:
            fragment_resids = peptide_fragment.residues
            # Find CA atoms corresponding to this peptide's residues
            fragment_ca_indices = [
                i for i, atom in enumerate(ca_atoms_for_distance) if atom.resid in fragment_resids
            ]
            if fragment_ca_indices:
                peptide_mean_pos = np.mean(mean_ca_positions[fragment_ca_indices], axis=0)
                peptide_mean_positions.append(peptide_mean_pos)
            else:
                # If no CA found, use a dummy position (shouldn't happen)
                peptide_mean_positions.append(np.array([0.0, 0.0, 0.0]))

        peptide_mean_positions = np.array(peptide_mean_positions)  # (num_peptides, 3)

        # Compute pairwise distances between peptides
        num_peptides = len(peptide_mean_positions)
        pairwise_distances = np.zeros((num_peptides, num_peptides))
        for i in range(num_peptides):
            for j in range(num_peptides):
                pairwise_distances[i, j] = np.linalg.norm(
                    peptide_mean_positions[i] - peptide_mean_positions[j]
                )

        # Create 1/R^2 matrix, avoiding division by zero on diagonal
        epsilon = 1e-6  # Small value to avoid division by zero
        pairwise_inv_r2 = np.zeros_like(pairwise_distances)
        mask = pairwise_distances > epsilon
        pairwise_inv_r2[mask] = 1.0 / (pairwise_distances[mask] ** 2)
        # Set diagonal to 1.0 (self-interaction)
        np.fill_diagonal(pairwise_inv_r2, 1.0)

        # Compute empirical covariance and multiply by 1/R^2 matrix
        C_prior_empirical_base = np.cov(y_pred_avg_frames)
        normalised_peptide_rmsf_values = peptide_rmsf_values / np.mean(peptide_rmsf_values)

        C_prior_empirical = C_prior_empirical_base + np.diag(
            np.full(y_pred_avg_frames.shape[0], 1e-6)
        )  # Add small diagonal for numerical stability

        print(f"  C_prior_empirical shape: {C_prior_empirical.shape}")

        # Save the pairwise 1/R^2 matrix as well
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_pairwise_inv_r2.npz"),
            pairwise_inv_r2=pairwise_inv_r2,
            pairwise_distances=pairwise_distances,
        )
        plot_heatmap(
            pairwise_inv_r2,
            f"{ensemble_name} Pairwise 1/R^2 Matrix",
            f"{ensemble_name}_pairwise_inv_r2_heatmap.png",
            args.output_dir,
            cmap="viridis",
            log_scale=True,
        )
        plot_diagonal_bar(
            pairwise_inv_r2,
            f"{ensemble_name} Pairwise 1/R^2 Diagonal",
            f"{ensemble_name}_pairwise_r2_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )

        plot_heatmap(
            C_prior_empirical,
            f"{ensemble_name} Empirical Covariance (Predctions x RMSF)",
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
        plot_diagonal_bar(
            C_prior_empirical,
            f"{ensemble_name} Empirical Covariance Diagonal",
            f"{ensemble_name}_C_prior_empirical_diagonal_bar.png",
            args.output_dir,
            log_scale=False,
        )

        all_diagonals[f"{ensemble_name}_empirical"] = np.diag(C_prior_empirical)
        np.savez(
            os.path.join(args.output_dir, f"{ensemble_name}_C_prior_empirical.npz"),
            C_prior_empirical=C_prior_empirical,
            C_prior_empirical_inv=np.linalg.inv(C_prior_empirical),
        )
        print(
            f"  Empirical Covariance for {ensemble_name} computed and saved to {args.output_dir}."
        )

    # --- Compute and Plot Correlation Matrix Between Diagonals ---
    if all_diagonals:
        print("\n--- Computing Correlation Matrix Between Diagonals ---")
        diagonal_names = list(all_diagonals.keys())

        # Find the maximum length among all diagonal arrays
        max_len = max(len(arr) for arr in all_diagonals.values())

        # Pad shorter diagonal arrays with NaN to match the maximum length
        padded_diagonals = []
        for name in diagonal_names:
            arr = all_diagonals[name]
            padded_arr = np.pad(arr.astype(float), (0, max_len - len(arr)), constant_values=np.nan)
            padded_diagonals.append(padded_arr)

        # Convert the padded diagonals into a NumPy array
        diagonal_values_padded = np.array(padded_diagonals)

        # Use pandas to compute correlation matrix, which handles NaNs with min_periods
        df_diagonals = pd.DataFrame(diagonal_values_padded.T, columns=diagonal_names)
        correlation_matrix = df_diagonals.corr(
            min_periods=1
        )  # min_periods=1 to allow correlation even with one common non-NaN value

        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=diagonal_names,
            yticklabels=diagonal_names,
            vmin=-1,
            vmax=1,
            center=0,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Correlation Matrix Between Covariance Diagonals")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "diagonal_correlation_matrix.png"))
        plt.close()

        # Save correlation matrix
        np.savez(
            os.path.join(args.output_dir, "diagonal_correlation_matrix.npz"),
            correlation_matrix=correlation_matrix.values,  # Save the underlying numpy array
            diagonal_names=diagonal_names,
        )
        print("  Diagonal correlation matrix computed and saved.")

    print("\n--- Covariance Matrix Generation Complete ---")


if __name__ == "__main__":
    main()
