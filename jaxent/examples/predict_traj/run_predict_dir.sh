#!/bin/bash
# Flexible script to run predict_traj.py for trajectories in a directory
# Automatically discovers XTC files and matches them with PDB topology files
# Usage: ./run_predict_flexible.sh [directory_name]


# Get directory name from argument or use default
TARGET_DIR="${1:-TFES-500}"
SCRIPT_DIR="$(dirname "$0")"
TARGET_DIRPATH="$SCRIPT_DIR/$TARGET_DIR"
PREDICT_SCRIPT="$SCRIPT_DIR/predict_traj.py"

echo "Running predict_traj for trajectories in $TARGET_DIRPATH"

# Check if directories and script exist
if [[ ! -d "$TARGET_DIRPATH" ]]; then
    echo "Error: Target directory not found at $TARGET_DIRPATH"
    exit 1
fi

if [[ ! -f "$PREDICT_SCRIPT" ]]; then
    echo "Error: predict_traj.py not found at $PREDICT_SCRIPT"
    exit 1
fi

# Define proteins to process
TARGET_PROTEINS=("BPTI" "BRD4" "HOIP" "LXR" "MBP" "P00974" "LXRa200")

# Function to extract protein identifier from filename
extract_protein_name() {
    local filename="$1"
    # Remove file extension and extract first part before underscore
    local basename=$(basename "$filename" .xtc)
    echo "${basename%%_*}"
}

# Function to check if protein should be processed
should_process_protein() {
    local protein_name="$1"
    for target in "${TARGET_PROTEINS[@]}"; do
        if [[ "$protein_name" == "$target" ]]; then
            return 0  # Should process
        fi
    done
    return 1  # Should not process
}

# Function to find matching topology file
find_topology_file() {
    local protein_name="$1"
    local search_dir="$2"
    
    # Look for PDB files that start with the protein name
    local topology_file=$(find "$search_dir" -maxdepth 1 -name "${protein_name}*.pdb" | head -1)
    
    # If not found, try case-insensitive search
    if [[ -z "$topology_file" ]]; then
        topology_file=$(find "$search_dir" -maxdepth 1 -iname "${protein_name}*.pdb" | head -1)
    fi
    
    echo "$topology_file"
}

# Find all trajectory files
echo "Scanning for trajectory files (*.xtc) in $TARGET_DIRPATH"
mapfile -t trajectory_files < <(find "$TARGET_DIRPATH" -maxdepth 1 -name "*.xtc" | sort)

if [[ ${#trajectory_files[@]} -eq 0 ]]; then
    echo "Error: No trajectory files (*.xtc) found in $TARGET_DIRPATH"
    exit 1
fi

echo "Found ${#trajectory_files[@]} trajectory file(s):"
for traj in "${trajectory_files[@]}"; do
    echo "  $(basename "$traj")"
done
echo

echo "Target proteins for processing: ${TARGET_PROTEINS[*]}"
echo

# Process each trajectory file
successful_predictions=0
failed_predictions=0
skipped_predictions=0

for trajectory_file in "${trajectory_files[@]}"; do
    trajectory_basename=$(basename "$trajectory_file")
    protein_name=$(extract_protein_name "$trajectory_basename")
    
    echo "=== Processing trajectory: $trajectory_basename ==="
    echo "  Extracted protein name: $protein_name"
    
    # Check if this protein should be processed
    if ! should_process_protein "$protein_name"; then
        echo "  → Skipping: '$protein_name' not in target protein list"
        ((skipped_predictions++))
        echo
        continue
    fi
    
    # Find matching topology file
    topology_file=$(find_topology_file "$protein_name" "$TARGET_DIRPATH")
    
    if [[ -z "$topology_file" ]]; then
        echo "  Warning: No topology file found for protein '$protein_name'"
        echo "  Skipping $trajectory_basename"
        ((failed_predictions++))
        echo
        continue
    fi
    
    topology_basename=$(basename "$topology_file")
    output_dir="$SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")/${protein_name}_${trajectory_basename}_prediction"
    
    echo "  Topology: $topology_basename"
    echo "  Output: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run predict_traj.py
    echo "  Running prediction..."
    
    if python "$PREDICT_SCRIPT" \
        --topology "$topology_file" \
        --trajectory "$trajectory_file" \
        --output "$output_dir" \
        --name "${protein_name}_${trajectory_basename}_prediction" \
        --bv_bc 0.35 \
        --bv_bh 2.0 \
        --temperature 300.0
        # --timepoints "0.167 1.0 10.0 120.0" \  # Example timepoints, adjust as needed
        ; then
        echo "  ✓ Successfully completed prediction for $protein_name"
        ((successful_predictions++))
    else
        prediction_exit_code=$?
        echo "  ✗ Failed to run prediction for $protein_name (exit code: $prediction_exit_code)"
        ((failed_predictions++))
    fi
    
    echo "  Continuing to next trajectory..."
    echo
done

# Summary
echo "Prediction Summary:"
echo "  Successful: $successful_predictions"
echo "  Failed: $failed_predictions"
echo "  Skipped (not in target list): $skipped_predictions"
echo "  Total trajectories found: ${#trajectory_files[@]}"
echo "  Total processed: $((successful_predictions + failed_predictions))"
echo
echo "Results saved in: $SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")/"