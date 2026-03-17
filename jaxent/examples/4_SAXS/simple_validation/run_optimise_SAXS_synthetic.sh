#!/bin/bash

# Runner script to perform the SAXS reweighting over a range of hyper-parameters
# Pattern from 1_IsoValidation_OMass/fitting/jaxENT/run_maxent_parallel_SIGMA.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../..:${PYTHONPATH}"

# Defaults
PARALLEL_JOBS=10
N_STEPS=5000
LEARNING_RATE=1.0
TARGETS="APO,nosol"
LOSSES="MSE,Chi2"
SPLIT_TYPES="random,stratified,data-cluster"
SPLIT_INDICES="0,1,2"
MAXENT_VALUES="0.0,1.0,10.0,100.0,1000.0,10000.0,100000.0"
OUTPUT_BASE="_optimise_SAXS_synthetic"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --n-steps)
            N_STEPS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --losses)
            LOSSES="$2"
            shift 2
            ;;
        --split-types)
            SPLIT_TYPES="$2"
            shift 2
            ;;
        --split-indices)
            SPLIT_INDICES="$2"
            shift 2
            ;;
        --maxent-values)
            MAXENT_VALUES="$2"
            shift 2
            ;;
        --dir-name)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to wait for available job slot
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do
        sleep 1
    done
}

# Create output directory with timestamp
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "Starting SAXS fitting sweep..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
echo "N steps: ${N_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"

# Nested loops over all parameter combinations
IFS=',' read -ra TARGET_ARRAY <<< "$TARGETS"
IFS=',' read -ra LOSS_ARRAY <<< "$LOSSES"
IFS=',' read -ra SPLIT_TYPE_ARRAY <<< "$SPLIT_TYPES"
IFS=',' read -ra SPLIT_IDX_ARRAY <<< "$SPLIT_INDICES"
IFS=',' read -ra MAXENT_ARRAY <<< "$MAXENT_VALUES"

TOTAL_JOBS=0
for TARGET in "${TARGET_ARRAY[@]}"; do
    for LOSS in "${LOSS_ARRAY[@]}"; do
        for SPLIT_TYPE in "${SPLIT_TYPE_ARRAY[@]}"; do
            for SPLIT_IDX in "${SPLIT_IDX_ARRAY[@]}"; do
                for MAXENT in "${MAXENT_ARRAY[@]}"; do
                    ((TOTAL_JOBS++))
                done
            done
        done
    done
done

echo "Total jobs to run: ${TOTAL_JOBS}"

JOB_COUNT=0
for TARGET in "${TARGET_ARRAY[@]}"; do
    for LOSS in "${LOSS_ARRAY[@]}"; do
        for SPLIT_TYPE in "${SPLIT_TYPE_ARRAY[@]}"; do
            for SPLIT_IDX in "${SPLIT_IDX_ARRAY[@]}"; do
                for MAXENT in "${MAXENT_ARRAY[@]}"; do
                    ((JOB_COUNT++))
                    wait_for_slot

                    LOG_NAME="${TARGET}_${LOSS}_${SPLIT_TYPE}_split${SPLIT_IDX}_maxent${MAXENT}"
                    LOG_FILE="${OUTPUT_DIR}/logs/${LOG_NAME}.log"

                    echo "[${JOB_COUNT}/${TOTAL_JOBS}] Starting: ${LOG_NAME}"

                    python "${SCRIPT_DIR}/fit_SAXS_Synthetic_APO.py" \
                        --target-curve "$TARGET" \
                        --loss-function "$LOSS" \
                        --split-type "$SPLIT_TYPE" \
                        --split-index "$SPLIT_IDX" \
                        --maxent-strength "$MAXENT" \
                        --n-steps "$N_STEPS" \
                        --learning-rate "$LEARNING_RATE" \
                        --output-dir "${OUTPUT_DIR}" \
                        > "$LOG_FILE" 2>&1 &
                done
            done
        done
    done
done

echo "Waiting for all jobs to complete..."
wait

echo "All jobs completed!"
echo "Results saved to: ${OUTPUT_DIR}"

# Run analysis on results
echo ""
echo "Starting analysis of results..."
python "${SCRIPT_DIR}/analyse_simple_synthetic_SAXS.py" --results-dir "${OUTPUT_DIR}"

echo "Analysis complete! Plots saved to: ${OUTPUT_DIR}/analysis"
