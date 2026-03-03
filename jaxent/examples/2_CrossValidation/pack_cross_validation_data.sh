#!/bin/bash
# pack_cross_validation_data.sh
# Packs the original source files for Example 2 into a tarball.

# Set working directory to the script's location
cd "$(dirname "$0")" || exit
BASE_DIR=$(pwd)

OUTPUT_TAR="cross_validation_data.tar.gz"

echo "Packaging original files for CrossValidation (Example 2)..."

# Define the files relative to the example root directory
FILES=(
    "data/MoPrP_max_plddt_4334.pdb"
    "data/_MoPrP/2L1H_crop.pdb"
    "data/_MoPrP/2L39_crop.pdb"
    "data/_cluster_MoPrP/clusters/all_clusters.xtc"
    "data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
    "data/_MoPrP/_output/MoPrP_dfrac.dat"
    "data/_MoPrP/_output/MoPrP_segments.txt"
    "data/_MoPrP/_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat"
    "data/_MoPrP/key_residues.json"
    "analysis/MoPrP_unfolding_spec.json"
    "analysis/MoPrP_rules_spec.json"
)

# Create the tarball
tar -zcvf "$OUTPUT_TAR" "${FILES[@]}"

echo "------------------------------------------------"
echo "Package created at: $BASE_DIR/$OUTPUT_TAR"
echo "Contents:"
tar -ztvf "$OUTPUT_TAR"
