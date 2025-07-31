#!/bin/bash
# Comprehensive script to run clustering followed by predict_traj.py
# First clusters trajectories, then runs predictions on the clustered output
# Usage: ./run_cluster_and_predict.sh [directory_name] [number_of_clusters] [num_components]

# Get arguments or use defaults
TARGET_DIR="${1:-TFES-500}"
NUM_CLUSTERS="${2:-25}"
NUM_COMPONENTS="${3:-10}"

SCRIPT_DIR="$(dirname "$0")"
TARGET_DIRPATH="$SCRIPT_DIR/$TARGET_DIR"
PREDICT_SCRIPT="$SCRIPT_DIR/predict_traj.py"

echo "Running clustering and prediction pipeline for trajectories in $TARGET_DIRPATH"
echo "Clustering parameters: $NUM_CLUSTERS clusters, $NUM_COMPONENTS components"

# Check if directories and script exist
if [[ ! -d "$TARGET_DIRPATH" ]]; then
    echo "Error: Target directory not found at $TARGET_DIRPATH"
    exit 1
fi

if [[ ! -f "$PREDICT_SCRIPT" ]]; then
    echo "Error: predict_traj.py not found at $PREDICT_SCRIPT"
    exit 1
fi

# Check if jaxent-kCluster command is available
if ! command -v jaxent-kCluster &> /dev/null; then
    echo "Error: jaxent-kCluster command not found. Please ensure it's in your PATH."
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

# Function to run clustering
run_clustering() {
    local topology_file="$1"
    local trajectory_file="$2"
    local output_dir="$3"
    local protein_name="$4"
    
    echo "  Running clustering with jaxent-kCluster..."
    echo "    Topology: $(basename "$topology_file")"
    echo "    Trajectory: $(basename "$trajectory_file")"
    echo "    Output: $output_dir"
    echo "    Clusters: $NUM_CLUSTERS, Components: $NUM_COMPONENTS"
    
    # Create clustering output directory
    mkdir -p "$output_dir"
    
    # Run jaxent-kCluster command
    if jaxent-kCluster \
        --topology_path "$topology_file" \
        --trajectory_paths "$trajectory_file" \
        --number_of_clusters "$NUM_CLUSTERS" \
        --num_components "$NUM_COMPONENTS" \
        --output_dir "$output_dir" \
        --atom_selection "name CA" \
        --chunk_size 100; then
        echo "  ✓ Clustering completed successfully"
        return 0
    else
        clustering_exit_code=$?
        echo "  ✗ Clustering failed (exit code: $clustering_exit_code)"
        return 1
    fi
}

# Function to run prediction on clustered trajectory
run_prediction() {
    local topology_file="$1"
    local clustered_trajectory="$2"
    local output_dir="$3"
    local protein_name="$4"
    
    echo "  Running prediction on clustered trajectory..."
    echo "    Topology: $(basename "$topology_file")"
    echo "    Clustered trajectory: $(basename "$clustered_trajectory")"
    echo "    Output: $output_dir"
    
    # Create prediction output directory
    mkdir -p "$output_dir"
    
    # Run predict_traj.py on the clustered trajectory
    if python "$PREDICT_SCRIPT" \
        --topology "$topology_file" \
        --trajectory "$clustered_trajectory" \
        --output "$output_dir" \
        --name "${protein_name}_${trajectory_basename}_clustered_prediction" \
        --bv_bc 0.35 \
        --bv_bh 2.0 \
        --temperature 300.0 \
        # --timepoints "0.167 1.0 10.0 120.0" \  # Example timepoints, adjust as needed
        ; then
        echo "  ✓ Prediction completed successfully"
        return 0
    else
        prediction_exit_code=$?
        echo "  ✗ Prediction failed (exit code: $prediction_exit_code)"
        return 1
    fi
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

# Initialize counters
successful_clusterings=0
failed_clusterings=0
successful_predictions=0
failed_predictions=0
skipped_proteins=0

# Process each trajectory file
for trajectory_file in "${trajectory_files[@]}"; do
    trajectory_basename=$(basename "$trajectory_file")
    protein_name=$(extract_protein_name "$trajectory_basename")
    
    echo "=== Processing trajectory: $trajectory_basename ==="
    echo "  Extracted protein name: $protein_name"
    
    # Check if this protein should be processed
    if ! should_process_protein "$protein_name"; then
        echo "  → Skipping: '$protein_name' not in target protein list"
        ((skipped_proteins++))
        echo
        continue
    fi
    
    # Find matching topology file
    topology_file=$(find_topology_file "$protein_name" "$TARGET_DIRPATH")
    
    if [[ -z "$topology_file" ]]; then
        echo "  Warning: No topology file found for protein '$protein_name'"
        echo "  Skipping $trajectory_basename"
        ((failed_clusterings++))
        echo
        continue
    fi
    
    topology_basename=$(basename "$topology_file")

    # Define output directories
    clustering_output_dir="$SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")/${protein_name}_${trajectory_basename}_clustering"
    prediction_output_dir="$SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")/${protein_name}_${trajectory_basename}_clustered_prediction"
    
    echo "  Topology: $topology_basename"
    echo "  Clustering output: $clustering_output_dir"
    echo "  Prediction output: $prediction_output_dir"
    
    # Step 1: Run clustering
    echo "--- Step 1: Clustering ---"
    if run_clustering "$topology_file" "$trajectory_file" "$clustering_output_dir" "$protein_name"; then
        ((successful_clusterings++))
        
        # Check if clustered trajectory file was created
        clustered_trajectory="$clustering_output_dir/clusters/all_clusters.xtc"
        if [[ ! -f "$clustered_trajectory" ]]; then
            echo "  ✗ Clustered trajectory file not found at $clustered_trajectory"
            ((failed_predictions++))
            echo
            continue
        fi
        
        # Step 2: Run prediction on clustered trajectory
        echo "--- Step 2: Prediction on Clustered Data ---"
        if run_prediction "$topology_file" "$clustered_trajectory" "$prediction_output_dir" "$protein_name"; then
            ((successful_predictions++))
            echo "  ✓ Complete pipeline successful for $protein_name"
        else
            ((failed_predictions++))
            echo "  ✗ Prediction step failed for $protein_name"
        fi
    else
        ((failed_clusterings++))
        echo "  ✗ Clustering step failed for $protein_name, skipping prediction"
    fi
    
    echo "  Continuing to next trajectory..."
    echo
done

# Final Summary
echo "==============================================="
echo "PIPELINE SUMMARY"
echo "==============================================="
echo "Trajectory files found: ${#trajectory_files[@]}"
echo "Proteins skipped (not in target list): $skipped_proteins"
echo "Proteins processed: $((successful_clusterings + failed_clusterings))"
echo
echo "CLUSTERING RESULTS:"
echo "  Successful: $successful_clusterings"
echo "  Failed: $failed_clusterings"
echo
echo "PREDICTION RESULTS:"
echo "  Successful: $successful_predictions"
echo "  Failed: $failed_predictions"
echo
echo "COMPLETE PIPELINE SUCCESS: $successful_predictions proteins"
echo
echo "Results saved in: $SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")/"
echo
if [[ $successful_predictions -gt 0 ]]; then
    echo "Successful complete pipelines for:"
    # List successful proteins (this is a simplified approach)
    find "$SCRIPT_DIR/outputs_$(basename "$TARGET_DIR")" -name "*_clustered_prediction" -type d 2>/dev/null | while read -r dir; do
        protein=$(basename "$dir" _clustered_prediction)
        echo "  - $protein"
    done
fi

# Exit with appropriate code
if [[ $failed_clusterings -gt 0 ]] || [[ $failed_predictions -gt 0 ]]; then
    echo
    echo "Some operations failed. Check the output above for details."
    exit 1
else
    echo
    echo "All operations completed successfully!"
    exit 0
fi