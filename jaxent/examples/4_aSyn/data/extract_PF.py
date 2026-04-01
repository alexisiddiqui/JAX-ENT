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
        help="Path to topology.json",
    )
    parser.add_argument(
        "--output_dir",
        default=SCRIPT_DIR / "_aSyn",
        help="Directory to save the outputs",
    )
    return parser.parse_args()


def load_topology_map(topo_path):
    """Load topology JSON and build residue ID to array index map."""
    with open(topo_path) as f:
        topo = json.load(f)

    resid_to_idx = {t["residues"][0]: t["fragment_index"] for t in topo["topologies"]}
    return resid_to_idx, len(topo["topologies"])


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
    resid_to_idx, n_topology_residues = load_topology_map(topology_json)

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

    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        logging.info(f"Processing condition: '{condition}'")
        
        if condition not in df_ph.columns:
            logging.warning(f"Condition '{condition}' not found in pH CSV. Skipping...")
            continue
            
        pD = float(df_ph[condition].iloc[0])
        logging.info(f"Using pD = {pD} for condition '{condition}'")
        
        # Calculate intrinsic rates for this condition's pH/pD
        kint_dict = calculate_HDXrate(protein_residues, temperature=293.0, pD=7.4)
        
        # Map intrinsic rates to topology fragments
        k_ints = np.full(n_topology_residues, np.nan, dtype=np.float32)
        for res, rate in kint_dict.items():
            if res.resid in resid_to_idx:
                k_ints[resid_to_idx[res.resid]] = rate

        pf_values = []
        log_pf_array = np.full(n_topology_residues, np.nan, dtype=np.float32)

        for i, row in df_kobs.iterrows():
            resid = int(row["Amino acid residues"])
            k_obs = row[condition] * 60  # Convert from ms^-1 to min^-1
            
            pf = np.nan
            if resid in resid_to_idx:
                idx = resid_to_idx[resid]
                k_int = k_ints[idx]
                
                # PF = k_int / k_obs. If k_obs == 0, PF is effectively infinity
                if k_obs > 0:
                    pf = float(k_int / k_obs)
                    log_pf_array[idx] = np.log(pf)
                elif k_obs == 0 and k_int > 0:
                    # Depending on how infinite PF should be handled downstream,
                    # we could set it to inf or nan. We'll leave it as NaN in log_pf_array
                    # since jnp can't safely fit to inf targets.
                    pf = np.inf
            
            pf_values.append(pf)

        # Update the PF DataFrame for this condition
        df_pf[condition] = pf_values

        # Create and save BV_output_features matching topology
        features_obj = BV_output_features(
            log_Pf=jnp.array(log_pf_array),
            k_ints=jnp.array(k_ints)
        )
        
        condition_safe_name = condition.replace(" ", "_").replace("/", "_")
        npz_output_path = output_dir / f"aSyn_PF_{condition_safe_name}.npz"
        
        logging.info(f"Saving BV protection features to {npz_output_path}")
        features_obj.save(str(npz_output_path))

    # Save the combined PF CSV
    pf_csv_path = output_dir / "aSyn_PF.csv"
    logging.info(f"Saving composite PF CSV to {pf_csv_path}")
    
    # We should reconstruct the 2-line header layout to match the input form, if desired.
    # We can write the descriptive header manually then the dataframe.
    with open(pf_csv_path, "w") as f:
        f.write("Amino acid residues,Protection Factors,,,\n")
    df_pf.to_csv(pf_csv_path, mode="a", index=False)

    logging.info("PF extraction and feature generation complete.")


if __name__ == "__main__":
    main()