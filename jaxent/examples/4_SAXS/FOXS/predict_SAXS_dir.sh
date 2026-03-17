#!/bin/bash

# Takes in a directory of hydrogenated structures and predicts their SAXS profiles using FOXS via conda IMP

# This script iterates through all PDB files to predict the SAXS profile using FOXS using a hydration layer and including hydrogens. Standard parameters are used in all steps. Output uses the same name as the input PDB file but with a .dat extension.

# (imp_env) (jax-ent) alexi@Celestial:~/Documents$ foxs
# Usage: <pdb_file1> <pdb_file2> ... <profile_file1> <profile_file2> ...

# Any number of input PDBs and profiles is supported.
# Each PDB will be fitted against each profile.

# This program is part of IMP, the Integrative Modeling Platform,
# which is Copyright 2007-2024 IMP Inventors.

# Options:
#   --help                            Show command line arguments and exit.
#   --version                         Show version info and exit.
#   -s [ --profile_size ] arg (=500)  number of points in the profile
#   -q [ --max_q ] arg (=0.50)        max q value
#   --min_c1 arg (=0.99)              min c1 value
#   --max_c1 arg (=1.05)              max c1 value
#   --min_c2 arg (=-2.00)             min c2 value
#   --max_c2 arg (=4.00)              max c2 value
#   -h [ --hydrogens ]                explicitly consider hydrogens in PDB files
#                                     (default = false)
#   -r [ --residues ]                 fast coarse grained calculation using CA
#                                     atoms only (default = false)
#   -b [ --background_q ] arg (=0)    background adjustment, not used by default.
#                                     if enabled, recommended q value is 0.2
#   -o [ --offset ]                   use offset in fitting (default = false)
#   -p [ --write-partial-profile ]    write partial profile file (default =
#                                     false)
#   -m [ --multi-model-pdb ] arg (=1) 1 - read the first MODEL only (default), 2
#                                     - read each MODEL into a separate
#                                     structure, 3 - read all models into a
#                                     single structure
#   -u [ --units ] arg (=1)           1 - unknown --> determine automatically
#                                     (default) 2 - q values are in 1/A, 3 - q
#                                     values are in 1/nm
#   -v [ --volatility_ratio ]         calculate volatility ratio score (default =
#                                     false)
#   -l [ --score_log ]                use log(intensity) in fitting and scoring
#                                     (default = false)
#   -g [ --gnuplot_script ]           print gnuplot script for gnuplot viewing
#                                     (default = false)

# Usage: predict_SAXS_dir.sh <input_directory> <output_directory>




INPUT_DIR=$1
OUTPUT_DIR=$2

# Check that the input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: predict_SAXS_dir.sh <input_directory> [output_directory]"
    exit 1
fi

# Default output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${INPUT_DIR%/}_SAXS"
    echo "No output directory provided. Defaulting to: $OUTPUT_DIR"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check that the input directory exists
if [ ! -d "$INPUT_DIR" ]; then  
    echo "Input directory $INPUT_DIR not found!"
    exit 1
fi




# Iterate through all PDB files in the input directory using GNU parallel (8 threads)
export INPUT_DIR OUTPUT_DIR

process_pdb() {
    PDB_FILE="$1"
    BASENAME=$(basename "$PDB_FILE" .pdb)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.dat"

    # Skip if output already exists (idempotent / resume support)
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Skipping $BASENAME (already done)"
        return
    fi

    echo "Processing: $BASENAME"
    # Run foxs; it writes <input>.pdb.dat in the same dir as the input
    conda run -n imp_env foxs -h "$PDB_FILE" 2>&1 | grep -v "^$"

    # Move output to OUTPUT_DIR
    if [ -f "${PDB_FILE}.dat" ]; then
        mv "${PDB_FILE}.dat" "$OUTPUT_FILE"
    else
        echo "ERROR: FoXS failed to produce output for $BASENAME"
    fi
}

export -f process_pdb
find "$INPUT_DIR" -maxdepth 1 -name "*.pdb" | xargs -P 8 -I {} bash -c 'process_pdb "{}"'
echo "Done! SAXS profiles written to $OUTPUT_DIR"



