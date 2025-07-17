# Featurisation Script Usage

This document provides an example of how to use the `featurise.py` script to generate features from a molecular dynamics trajectory.

## BV Model Featurisation Example

The following command runs featurisation using the Best-Vendruscolo (BV) model on a sample trajectory of Bovine Pancreatic Trypsin Inhibitor (BPTI).

### Command

```bash
python jaxent/cli/featurise.py \
    --top_path /path/to/your/topology.pdb \
    --trajectory_path /path/to/your/trajectory.xtc \
    --output_dir /path/to/your/featurisation_output \
    --name test_bv \
    bv \
    --temperature 300.0 \
    --ph 7.0 \
    --num_timepoints 1 \
    --timepoints 0.167 \
    --residue_ignore -2 2 \
    --mda_selection_exclusion "resname PRO or resid 1"
```

### Argument Explanations

-   `--top_path`: Path to the topology file (e.g., PDB).
-   `--trajectory_path`: Path to the trajectory file (e.g., XTC, DCD).
-   `--output_dir`: Directory where the output files will be saved.
-   `--name`: A name for this featurisation run.
-   `bv`: Specifies that we are using the BV model.
-   `--temperature`: Temperature in Kelvin.
-   `--ph`: The pH value.
-   `--num_timepoints`: The number of timepoints.
-   `--timepoints`: The timepoint values in minutes.
-   `--residue_ignore`: A range of residues to ignore relative to the donor.
-   `--mda_selection_exclusion`: An MDAnalysis selection string to exclude certain residues (e.g., prolines).

### Output Files

The script will generate two files in the specified output directory:

1.  `features.npz`: A NumPy archive containing the featurised data. For the BV model, this includes:
    -   `k_ints`: Intrinsic rates.
    -   `heavy_contacts`: Heavy atom contacts.
    -   `acceptor_contacts`: Acceptor atom contacts.
2.  `topology.json`: A JSON file describing the topology of the system, including information about the residues processed.

This example is based on the test case found in `jaxent/tests/cli/test_cli_featurise.py`.
