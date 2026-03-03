#!/bin/bash
# pack_iso_validation_data.sh
# Packs the original source files for Example 1 into a tarball.

# Set working directory to the script's location
cd "$(dirname "$0")" || exit
BASE_DIR=$(pwd)

DATA_DIR="data"
OUTPUT_TAR="iso_validation_data.tar.gz"

echo "Packaging original files for IsoValidation (Example 1)..."

# Define the files relative to the data directory
FILES=(
    "_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
    "_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
    "_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_filtered_sliced.xtc"
    "_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc"
    "_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat"
)

# Navigate to the data directory to maintain correct relative paths in tar
cd "$DATA_DIR" || exit

# Create the tarball
tar -zcvf "../$OUTPUT_TAR" "${FILES[@]}"

echo "------------------------------------------------"
echo "Package created at: $BASE_DIR/$OUTPUT_TAR"
echo "Contents:"
tar -ztvf "../$OUTPUT_TAR"
