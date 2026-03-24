#!/bin/bash

# Collect minimised and unminimised structures into an organised folder

PULLDOWN_DIR="/home/alexi/Documents/NeuralPLexer/APO-CaM/_sub_cluster_output/pulldown"
OUTPUT_DIR="/home/alexi/Documents/NeuralPLexer/APO-CaM/_sub_cluster_output/collected_structures"

# Create output directories
mkdir -p "$OUTPUT_DIR/liganded_ions"
mkdir -p "$OUTPUT_DIR/apo_unminimised"

echo "Collecting structures..."
echo "=========================================="

# Counter variables
liganded_count=0
apo_count=0

# 1. Copy minimised liganded/ion structures
echo ""
echo "1. Copying minimised liganded/ion structures..."
echo "   (from clusters with ligands/ions)"
echo ""

for cluster_dir in "$PULLDOWN_DIR"/cluster_*; do
    cluster_name=$(basename "$cluster_dir")

    # Skip APO clusters (ca0_cdz0)
    if [[ "$cluster_name" == *"ca0_cdz0" ]]; then
        continue
    fi

    minimised_dir="$cluster_dir/combined_outputs/minimised"

    if [ -d "$minimised_dir" ]; then
        # Count PDB files
        pdb_count=$(ls -1 "$minimised_dir"/*.pdb 2>/dev/null | wc -l)

        if [ $pdb_count -gt 0 ]; then
            # Create subdirectory for this cluster
            mkdir -p "$OUTPUT_DIR/liganded_ions/$cluster_name"

            # Copy files
            cp "$minimised_dir"/*.pdb "$OUTPUT_DIR/liganded_ions/$cluster_name/" 2>/dev/null

            echo "   ✓ $cluster_name ($pdb_count files)"
            ((liganded_count += pdb_count))
        fi
    fi
done

# 2. Copy unminimised APO structures
echo ""
echo "2. Copying unminimised APO structures..."
echo "   (from ca0_cdz0 clusters - no ligands)"
echo ""

for cluster_dir in "$PULLDOWN_DIR"/cluster_*_ca0_cdz0; do
    cluster_name=$(basename "$cluster_dir")

    combined_dir="$cluster_dir/combined_outputs"

    if [ -d "$combined_dir" ]; then
        # Count unminimised PDB files (prot_rank*.pdb - protein-only APO structures)
        pdb_count=$(ls -1 "$combined_dir"/prot_rank*.pdb 2>/dev/null | wc -l)

        if [ $pdb_count -gt 0 ]; then
            # Create subdirectory for this cluster
            mkdir -p "$OUTPUT_DIR/apo_unminimised/$cluster_name"

            # Copy files
            cp "$combined_dir"/prot_rank*.pdb "$OUTPUT_DIR/apo_unminimised/$cluster_name/" 2>/dev/null

            echo "   ✓ $cluster_name ($pdb_count files - unminimised)"
            ((apo_count += pdb_count))
        fi
    fi
done

echo ""
echo "=========================================="
echo "Collection complete!"
echo ""
echo "Summary:"
echo "  Liganded/ions structures: $liganded_count PDB files"
echo "  APO unminimised structures: $apo_count PDB files"
echo ""
echo "Output location:"
echo "  $OUTPUT_DIR"
echo ""
echo "Structure:"
echo "  collected_structures/"
echo "  ├── liganded_ions/"
echo "  │   ├── cluster_0_0_ca0_cdz1/"
echo "  │   ├── cluster_0_0_ca0_cdz2/"
echo "  │   └── ..."
echo "  └── apo_unminimised/"
echo "      ├── cluster_0_0_ca0_cdz0/"
echo "      ├── cluster_0_1_ca0_cdz0/"
echo "      └── ..."
