"""
This script computes and plots the Sigma (observation noise covariance) matrix
from experimental HDX data.

Sigma is computed from the experimental data itself.
The script takes the HDX data .dat file and the output directory as arguments.

Exanmple usage:
    python compute_covariance_matrices.py \
        --dfrac_file ./../data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat \
        --segs_file ./../data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt \
        --output_dir ./_covariance_matrices 

        
    python /home/alexi/Documents/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
        --dfrac_file /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_dfrac.dat \
        --segs_file  /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_segments.txt \
        --output_dir ./_MoPrP_covariance_matrices

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.interfaces.topology import Partial_Topology, TopologyFactory


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


# New helper: plot diagonal as bar chart
def plot_diagonal_bar(matrix, title, filename, output_dir, log_scale=False, eps=1e-12):
    diag = np.diag(matrix).astype(float)
    indices = np.arange(len(diag))
    plt.figure(figsize=(10, 4.5))
    if log_scale:
        # avoid zeros for log scale
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
        description="Compute and plot Sigma (observation noise covariance) matrix from HDX data."
    )
    parser.add_argument(
        "--dfrac_file",
        type=str,
        default="../../data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat",
        help="Path to the HDX dfrac data file.",
    )
    parser.add_argument(
        "--segs_file",
        type=str,
        default="../../data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt",
        help="Path to the HDX segments data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="_covariance_matrices",
        help="Directory to save the output covariance matrices and plots.",
    )
    args = parser.parse_args()

    # Convert all paths to absolute paths
    # - If already absolute, use as-is
    # - If relative, treat as relative to current working directory
    full_dfrac_path = os.path.abspath(args.dfrac_file)
    full_segs_path = os.path.abspath(args.segs_file)
    output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Loading Experimental Data from {full_dfrac_path} and {full_segs_path} ---")
    segs: np.ndarray = pd.read_csv(
        full_segs_path,
        sep=r"\s+",
        comment="#",
        header=None,
    ).to_numpy()

    dfrac: np.ndarray = pd.read_csv(
        full_dfrac_path,
        sep=r"\s+",
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
    dfrac_values = np.array([peptide.dfrac for peptide in HDX_data])

    _dfrac_values = dfrac_values
    Sigma = np.cov(_dfrac_values) + np.diag(np.full(_dfrac_values.shape[0], 1e-6))

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

    # Plot diagonals as bar charts
    plot_diagonal_bar(
        Sigma,
        "Sigma diagonal (variance per observation)",
        "Sigma_diagonal_bar.png",
        output_dir,
        log_scale=False,
    )
    plot_diagonal_bar(
        Sigma_inv,
        "Inverse Sigma diagonal (log scale)",
        "Sigma_inv_diagonal_bar.png",
        output_dir,
        log_scale=True,
    )

    np.savez(os.path.join(output_dir, "Sigma.npz"), Sigma=Sigma, Sigma_inv=Sigma_inv)
    print("Sigma computed and saved.")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
