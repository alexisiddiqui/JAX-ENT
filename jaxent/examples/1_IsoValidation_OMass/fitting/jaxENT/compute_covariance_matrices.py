"""
This script generates covariance matrices for use in the loss function:
.. math::
 \min_{w \in \Delta M} \left\{ \mathrm{KL}(w \| u) + \frac{1}{2}(m(w) - m_0)^\top C_{\text{prior}}^{-1} (m(w) - m_0) + \frac{1}{2}(y - m(w))^\top \Sigma^{-1} (y - m(w)) \right\}
Where:
* :math:w is the vector of model weights.
* :math:\Delta M is the domain of the weights.
* :math:\mathrm{KL}(\cdot \| \cdot) is the Kullback-Leibler divergence.
* :math:u is the prior weight vector.
* :math:m(w) is the model prediction for a given :math:w.
* :math:m_0 is a prior mean for the predictions.
* :math:C_{\text{prior}} is the prior covariance matrix.
* :math:y is the observed data.
* :math:\Sigma is the observation noise covariance matrix.
\Sigma is computed from the experimental data itself whereas C_prior is computed from the structural ensemble itself.
Approaches to computing C_prior:
- (Empirical) covariance from the residue-wise ensemble average of model predictiosns.
- (Diagonal) Structural covriance from the residue-wise RMSF of the structural ensemble.
- (Diagonal) Ensemble covariance from the residue-wise variance of the model predictions at each residue positions.
This script generates each of these covariance matrices and saves them to disk for later use.
To ensure compatibility with the topology, the script loads in the topology .json files from ../fitting/jaxENT/_featurise such that the covariance matrices can be aligned with the model predictions.
The model features are also taken from here in order to generate the predictions.
\Sigma:
To generate the empirical covariance matrix (\Sigma), the experimental uptake curves are loadedf from ../data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat
C_prior:
To generate the diagonal covariance matrices these are computed from both ensembles each.
Structural covariance is computed from the RMSF of the ensembles
Ensemble covariance is computed from the variance of the model predictions at each residue position.
To generate the model predictions, the features (contacts and kints) are loaded from the BV_input_features .npz object saved in ../fitting/jaxENT/_featurise
The predict method is then used to generate the model predictions for each structure in the ensemble.
The predictions are saved to disk and the covariance matrices are computed from these predictions.
For all covariance matrices, these and the matrix inverse are plotted as heatmaps and saved to disk for visual inspection.
"""

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

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import Partial_Topology, PTSerialiser, TopologyFactory
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.predict import run_predict

# --- Configuration and File Paths ---
script_dir = os.path.dirname(__file__)

# Base directories
features_dir = os.path.join(script_dir, "_featurise")
data_dir = os.path.join(script_dir, "../../data/_output")
traj_dir = os.path.join(
    script_dir, "../../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
)
output_dir = os.path.join(script_dir, "_covariance_matrices")

# Specific file names
dfrac_file = "mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat"
segs_file = "mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"

topology_file_bi = "topology_iso_bi.json"
features_file_bi = "features_iso_bi.npz"
topology_file_tri = "topology_iso_tri.json"
features_file_tri = "features_iso_tri.npz"

closed_pdb = "TeaA_ref_closed_state.pdb"  # For topology
bi_traj = "sliced_trajectories/TeaA_filtered_sliced.xtc"
tri_traj = "sliced_trajectories/TeaA_initial_sliced.xtc"

# Full paths
full_dfrac_path = os.path.join(data_dir, dfrac_file)
full_segs_path = os.path.join(data_dir, segs_file)

full_topology_file_bi = os.path.join(features_dir, topology_file_bi)
full_features_file_bi = os.path.join(features_dir, features_file_bi)
full_topology_file_tri = os.path.join(features_dir, topology_file_tri)
full_features_file_tri = os.path.join(features_dir, features_file_tri)

full_closed_pdb = os.path.join(traj_dir, closed_pdb)
full_bi_traj = os.path.join(traj_dir, bi_traj)
full_tri_traj = os.path.join(traj_dir, tri_traj)


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
        # Use absolute values for log-scale display (inverse covariance matrices should be positive),
        # clamp small/zero values to eps to avoid issues with LogNorm.
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


def main():
    os.makedirs(output_dir, exist_ok=True)

    print("--- Loading Experimental Data ---")
    segs: np.ndarray = pd.read_csv(
        full_segs_path,you 
        sep="\\s+",
        comment="#",
        header=None,
    ).to_numpy()

    dfrac: np.ndarray = pd.read_csv(
        full_dfrac_path,
        sep="\\s+",
        comment="#",
        header=None,
    ).to_numpy()

    # Create HDX topology objects
    HDX_topology: list[Partial_Topology] = [
        TopologyFactory.from_range(
            chain="A",
            start=seg[0],
            end=seg[1],
            fragment_index=idx,
            peptide=True,
            peptide_trim=1,
            fragment_name="TeaISO",
        )
        for idx, seg in enumerate(segs)
    ]

    # Create HDX data objects
    HDX_data: list[HDX_peptide] = [
        HDX_peptide(dfrac=dfrac[idx], top=HDX_topology[idx]) for idx in range(len(segs))
    ]
    print(f"Loaded {len(HDX_data)} HDX_peptide objects.")

    # --- Compute Sigma (Observation Noise Covariance Matrix) ---
    print("\n--- Computing Sigma (Observation Noise Covariance Matrix) ---")
    # Compute empirical covariance matrix from experimental uptake data
    # dfrac has shape (num_peptides, num_timepoints)
    dfrac_values = np.array([peptide.dfrac for peptide in HDX_data])

    # Compute empirical covariance matrix where each peptide is a variable
    # and each timepoint is an observation
    # np.cov expects rows as variables, columns as observations
    # dfrac_values is already in this format (num_peptides, num_timepoints)
    print(dfrac_values.shape)  # Debug print
    _dfrac_values = dfrac_values - np.mean(dfrac_values, axis=1, keepdims=True)
    Sigma = np.cov(_dfrac_values) + np.diag(np.full(_dfrac_values.shape[0], 1e-9))
    # Sigma = np.cov(_dfrac_values)

    Sigma_inv = np.linalg.inv(Sigma)

    print(f"Sigma shape: {Sigma.shape}")
    plot_heatmap(Sigma, "Sigma (Observation Noise Covariance)", "Sigma_heatmap.png", output_dir)
    plot_heatmap(
        Sigma_inv,
        "Inverse Sigma",
        "Sigma_inv_heatmap.png",
        output_dir,
        cmap="magma",
        log_scale=True,
    )
    np.savez(os.path.join(output_dir, "Sigma.npz"), Sigma=Sigma, Sigma_inv=Sigma_inv)
    print("Sigma computed and saved.")

    # --- Load Features and Topology for Ensembles ---
    print("\n--- Loading Features and Topology for Ensembles ---")
    ensembles = {
        "ISO_BI": {
            "features": BV_input_features.load(full_features_file_bi),
            "topology": PTSerialiser.load_list_from_json(full_topology_file_bi),
            "trajectory": full_bi_traj,
        },
        "ISO_TRI": {
            "features": BV_input_features.load(full_features_file_tri),
            "topology": PTSerialiser.load_list_from_json(full_topology_file_tri),
            "trajectory": full_tri_traj,
        },
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

        # --- Structural Covariance (from keep) ---
        print(f"  Computing Structural Covariance (RMSF) for {ensemble_name}...")
        universe = mda.Universe(
            full_closed_pdb, trajectory_path
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
            R = RMSF(protein_selection).run()
            # RMSF values are per atom, need to average per residue or take max/min
            # For simplicity, let's assume we can map RMSF to residues in topology
            # This part needs careful mapping from MDAnalysis residue indices to HDX_peptide topology indices
            # For now, let's create a dummy RMSF per residue based on the number of peptides

            # Placeholder: create a diagonal matrix with arbitrary values if RMSF cannot be directly mapped
            # A better approach would be to get RMSF per residue and then map to the peptides
            # For now, let's create a diagonal matrix with placeholder values.

            # Get RMSF per residue (assuming protein_selection is residues)
            # This is a simplified approach. A more robust solution would involve iterating through the residues
            # and calculating RMSF for each.

            # For now, let's assume we can get a single RMSF value per peptide/fragment
            # This is a simplification, as RMSF is typically per atom or per residue.
            # If the topology represents peptides, we need to aggregate RMSF for residues within each peptide.

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
                output_dir,
            )
            plot_heatmap(
                np.linalg.inv(C_prior_structural),
                f"Inverse {ensemble_name} Structural Covariance",
                f"{ensemble_name}_C_prior_structural_inv_heatmap.png",
                output_dir,
                cmap="magma",
                log_scale=True,
            )
            np.savez(
                os.path.join(output_dir, f"{ensemble_name}_C_prior_structural.npz"),
                C_prior_structural=C_prior_structural,
                C_prior_structural_inv=np.linalg.inv(C_prior_structural),
            )
            print(f"  Structural Covariance for {ensemble_name} computed and saved.")

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
            os.path.join(output_dir, f"{ensemble_name}_predicted_mean_uptake.npy"),
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
        plt.savefig(os.path.join(output_dir, f"{ensemble_name}_predicted_mean_uptake_curves.png"))
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
            output_dir,
        )
        plot_heatmap(
            np.linalg.inv(C_prior_ensemble),
            f"Inverse {ensemble_name} Ensemble Covariance",
            f"{ensemble_name}_C_prior_ensemble_inv_heatmap.png",
            output_dir,
            cmap="magma",
            log_scale=True,
        )
        np.savez(
            os.path.join(output_dir, f"{ensemble_name}_C_prior_ensemble.npz"),
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
            output_dir,
        )
        plot_heatmap(
            np.linalg.inv(C_prior_empirical),
            f"Inverse {ensemble_name} Empirical Covariance",
            f"{ensemble_name}_C_prior_empirical_inv_heatmap.png",
            output_dir,
            cmap="magma",
            log_scale=True,
        )
        np.savez(
            os.path.join(output_dir, f"{ensemble_name}_C_prior_empirical.npz"),
            C_prior_empirical=C_prior_empirical,
            C_prior_empirical_inv=np.linalg.inv(C_prior_empirical),
        )
        print(f"  Empirical Covariance for {ensemble_name} computed and saved.")

    print("\n--- Covariance Matrix Generation Complete ---")


if __name__ == "__main__":
    main()
