"""
reads in jaxent/examples/4_aSyn/data/_aSyn/aSyn_k_obs.csv
uses the kints from jaxent/examples/4_aSyn/data/_cluster_aSyn/data/features.npz
and the topology from jaxent/examples/4_aSyn/data/_cluster_aSyn/data/topology.json

outputs jaxent/examples/4_aSyn/data/_aSyn/aSyn_PF.csv over all conditions
and a jaxENT BV protection features object for each condition, matching the topology.json provided
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
import MDAnalysis as mda
import matplotlib.pyplot as plt
import seaborn as sns

from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.interfaces.topology import PTSerialiser, mda_TopologyAdapter
from jaxent.src.models.HDX.BV.features import BV_output_features
from jaxent.src.models.func.uptake import calculate_HDXrate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate protection factors from k_obs and compute intrinsic rates to create BV protection feature objects."
    )
    # Default paths matching the docstring relative to the script location
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    parser.add_argument(
        "--kobs_csv",
        default=SCRIPT_DIR / "_aSyn" / "aSyn_k_obs.csv",
        help="Path to observed rate constants CSV",
    )
    parser.add_argument(
        "--pdb_path",
        default=SCRIPT_DIR / "_aSyn" / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb",
        help="Path to the system PDB to extract intrinsic rates",
    )
    parser.add_argument(
        "--ph_csv",
        default=SCRIPT_DIR / "_aSyn" / "aSyn_pH.csv",
        help="Path to pH conditions CSV",
    )
    parser.add_argument(
        "--topology_json",
        default=SCRIPT_DIR / "_cluster_aSyn" / "data" / "topology.json",
        help="Path to topology.json (Optional - if not provided, will use the PDB to create a topology)",
    )
    parser.add_argument(
        "--output_dir",
        default=SCRIPT_DIR / "_aSyn",
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--residue_selection",
        default=None,
        help="Residues to include, e.g. '1-100', '1,5,10', or '1-50,60-80,100'. Default: all residues in topology.",
    )
    return parser.parse_args()


def parse_residue_selection(selection_str: str) -> set:
    """Parse '1-50,60,80-90' into a set of integer residue IDs."""
    resids = set()
    for token in selection_str.split(","):
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", 1)
            resids.update(range(int(start), int(end) + 1))
        else:
            resids.add(int(token))
    return resids


def load_topologies(topo_path):
    """Load topologies from JSON or PDB."""
    topo_path = Path(topo_path)
    if topo_path.suffix == ".json":
        return PTSerialiser.load_list_from_json(topo_path)
    elif topo_path.suffix in [".pdb", ".pdbqt"]:
        logging.info(f"Loading topology from PDB: {topo_path}")
        u = mda.Universe(str(topo_path))
        # Create one topology per residue
        return mda_TopologyAdapter.from_mda_universe(u, mode="residue", include_selection="protein")
    else:
        raise ValueError(f"Unsupported topology format: {topo_path.suffix}")


def plot_hdx_metrics(plot_data_list, output_dir, topology_resids=None):
    """Plot k_obs, k_int, and log_PF for all conditions, highlighting missing residues."""
    if not plot_data_list:
        logging.warning("No data available for plotting.")
        return

    df_plot = pd.DataFrame(plot_data_list)
    
    if topology_resids is None:
        topology_resids = sorted(df_plot["Residue"].unique())
    
    res_min, res_max = min(topology_resids), max(topology_resids)
    full_range = np.arange(res_min, res_max + 1)
    
    # Identify gaps: residues in the full range that have no valid k_obs in ANY condition
    covered_resids = df_plot[df_plot["k_obs"] > 0]["Residue"].unique()
    missing_resids = [r for r in full_range if r not in covered_resids]
    
    # helper to find contiguous blocks for axvspan
    def get_blocks(resids):
        if not resids: return []
        resids = sorted(resids)
        blocks = []
        start = resids[0]
        for i in range(1, len(resids)):
            if resids[i] != resids[i-1] + 1:
                blocks.append((start, resids[i-1]))
                start = resids[i]
        blocks.append((start, resids[-1]))
        return blocks

    missing_blocks = get_blocks(missing_resids)

    # Apply scientific style
    sns.set_theme(style="ticks", context="paper")
    sns.set_palette("colorblind")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    metrics = [
        ("k_obs", r"$k_{obs}$ ($min^{-1}$)", "Observed Rates"),
        ("k_int", r"$k_{int}$ ($min^{-1}$)", "Intrinsic Rates"),
        ("log_PF", r"$\ln(PF)$", "Protection Factors")
    ]
    
    for i, (col, ylabel, title) in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(data=df_plot, x="Residue", y=col, hue="Condition", ax=ax, marker='o', markersize=3, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        
        # Set x-axis limit to full topology range
        ax.set_xlim(res_min - 1, res_max + 1)

        # set x ticks to integers
        ax.set_xticks(np.arange(res_min, res_max + 1, 10))
        
        # Add grey bars for missing residues
        for start, end in missing_blocks:
            ax.axvspan(start - 0.5, end + 0.5, color='grey', alpha=0.2, lw=0, label="MissingData" if i==0 and start==missing_blocks[0][0] else "")
        
        if i == 0:
            ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc='upper left')
        else:
            ax.legend().set_visible(False)
            
    # set k_ints to log scale
    axes[1].set_yscale("log")

            
    axes[2].set_xlabel("Residue ID")
    
    plt.tight_layout()
    
    png_path = output_dir / "aSyn_metrics_comparison.png"
    pdf_path = output_dir / "aSyn_metrics_comparison.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    logging.info(f"Saved plots to {png_path} and {pdf_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    kobs_csv = Path(args.kobs_csv).expanduser().resolve()
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    ph_csv = Path(args.ph_csv).expanduser().resolve()
    topology_json = Path(args.topology_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not kobs_csv.exists():
        raise FileNotFoundError(f"Observed rates CSV not found: {kobs_csv}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    if not ph_csv.exists():
        raise FileNotFoundError(f"pH CSV not found: {ph_csv}")
    if not topology_json.exists():
        raise FileNotFoundError(f"Topology JSON not found: {topology_json}")

    logging.info(f"Loading topology: {topology_json}")
    topologies = load_topologies(topology_json)
    n_topology_residues = len(topologies)
    # Map residue ID to the topology object
    resid_to_topo = {top.residues[0]: top for top in topologies}

    if args.residue_selection:
        selected_resids = parse_residue_selection(args.residue_selection)
        n_before = len(resid_to_topo)
        resid_to_topo = {k: v for k, v in resid_to_topo.items() if k in selected_resids}
        logging.info(f"Residue selection applied: {n_before} → {len(resid_to_topo)} residues retained.")

    logging.info(f"Loading PDB for intrinsic rates: {pdb_path}")
    universe = mda.Universe(str(pdb_path))
    protein_residues = universe.select_atoms("protein").residues

    logging.info(f"Loading pH conditions: {ph_csv}")
    df_ph = pd.read_csv(ph_csv, skiprows=1)

    logging.info(f"Loading observed rates: {kobs_csv}")
    # The file has a descriptive header on row 1, and the actual column headers on row 2.
    df_kobs = pd.read_csv(kobs_csv, skiprows=1)

    conditions = [col for col in df_kobs.columns if col != "Amino acid residues"]
    logging.info(f"Conditions found: {conditions}")

    # Prepare DataFrame for Protection Factors
    df_pf = df_kobs.copy()

    # Data collection for plotting
    plot_data_list = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        logging.info(f"Processing condition: '{condition}'")
        
        if condition not in df_ph.columns:
            logging.warning(f"Condition '{condition}' not found in pH CSV. Skipping...")
            continue
            
        pD = float(df_ph[condition].iloc[0])
        logging.info(f"Using pD = {pD} for condition '{condition}'")
        
        # Calculate intrinsic rates for this condition's pH/pD
        kint_dict = calculate_HDXrate(protein_residues, temperature=293.0, pD=pD)
        
        # Map intrinsic rates - kint_dict maps mda.Residue -> rate
        topo_k_ints = {}
        for res, rate in kint_dict.items():
            if res.resid in resid_to_topo:
                topo_k_ints[res.resid] = rate

        datapoints = []
        filtered_log_pf = []
        filtered_k_ints = []
        
        pfs_in_condition = []

        for i, row in df_kobs.iterrows():
            resid = int(row["Amino acid residues"])
            k_obs = row[condition] / 60   # Convert from s^-1 to min^-1
            
            pf = np.nan
            k_int = np.nan
            if resid in resid_to_topo:
                topo = resid_to_topo[resid]
                k_int = topo_k_ints.get(resid, np.nan)
                
                if k_obs > 0 and not np.isnan(k_int):
                    pf = float(k_int / k_obs)
                    
                    # Create HDX data object (using logPF as requested)
                    dp = HDX_protection_factor(top=topo, protection_factor=np.log(pf))
                    datapoints.append(dp)
                    
                    filtered_log_pf.append(np.log(pf))
                    filtered_k_ints.append(k_int)
            
            pfs_in_condition.append(pf)
            
            # Add to plotting data
            plot_data_list.append({
                "Residue": resid,
                "Condition": condition,
                "k_obs": k_obs,
                "k_int": k_int if not np.isnan(k_int) else None,
                "log_PF": np.log(pf) if pf > 0 and not np.isnan(pf) else None
            })

        # Update the PF DataFrame for this condition (full list with NaNs for the CSV)
        df_pf[condition] = pfs_in_condition

        condition_safe_name = condition.replace(" ", "_").replace("/", "_")
        
        if datapoints:
            # Save via HDX data object (CSV + JSON)
            # This automatically handles NaN removal because we only added valid datapoints
            output_base = output_dir / f"aSyn_PF_{condition_safe_name}"
            logging.info(f"Saving HDX data objects to {output_base}.csv/json")

            
            ExpD_Datapoint.save_list_to_files(datapoints, base_name=str(output_base))

            # Also save the filtered NPZ for JAX compatibility as requested
            features_obj = BV_output_features(
                log_Pf=jnp.array(filtered_log_pf),
                k_ints=jnp.array(filtered_k_ints)
            )
            npz_output_path = output_dir / f"aSyn_PF_{condition_safe_name}.npz"
            logging.info(f"Saving filtered BV protection features to {npz_output_path}")
            features_obj.save(str(npz_output_path))
        else:
            logging.warning(f"No valid protection factors found for condition '{condition}'")

    # Save the combined PF CSV
    pf_csv_path = output_dir / "aSyn_PF.csv"
    logging.info(f"Saving composite PF CSV to {pf_csv_path}")
    
    # We should reconstruct the 2-line header layout to match the input form, if desired.
    # We can write the descriptive header manually then the dataframe.
    with open(pf_csv_path, "w") as f:
        f.write("Amino acid residues,Protection Factors,,,\n")
    df_pf.to_csv(pf_csv_path, mode="a", index=False)

    # Generate the requested visualizations
    logging.info("Generating comparison plots...")
    plot_hdx_metrics(plot_data_list, output_dir, topology_resids=list(resid_to_topo.keys()))

    logging.info("PF extraction and feature generation complete.")


if __name__ == "__main__":
    main()