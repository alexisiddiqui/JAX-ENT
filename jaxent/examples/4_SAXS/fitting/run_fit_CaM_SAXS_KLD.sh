#!/bin/bash
# SAXS-only MaxEnt reweighting sweep for CaM+CDZ and CaM-CDZ.
#
# Parallelises at the bash level (one Python process per combination).
# After all jobs complete, runs the analysis script automatically.
#
# Usage:
#   bash run_fit_CaM_SAXS_KLD.sh [options]
#
# Options:
#   --jobs          N     Max parallel jobs          (default: 10)
#   --n-steps       N     Optimisation steps          (default: 50000)
#   --learning-rate F     Learning rate               (default: 1.0)
#   --targets       A,B   Comma-separated targets     (default: CaM+CDZ,CaM-CDZ)
#   --split-types   A,B   Comma-separated split types (default: all three)
#   --split-indices 0,1,2 Comma-separated indices     (default: 0,1,2)
#   --maxent-values A,B   Comma-separated strengths   (default: 7-value sweep)
#   --dir-name      NAME  Output base directory name  (default: _optimise_CaM_SAXS_KLD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../..:${PYTHONPATH:-}"

# ---------- Defaults ----------
PARALLEL_JOBS=10
N_STEPS=5000
LEARNING_RATE=1.0
TARGETS="CaM+CDZ,CaM-CDZ"
SPLIT_TYPES="random,stratified,random-stratified"
SPLIT_INDICES="0,1,2"
MAXENT_VALUES="0.001,0.01,0.1,1.0,10.0,100.0,1000.0"
OUTPUT_BASE="_optimise_CaM_SAXS_KLD"

# ---------- Argument parsing ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)           PARALLEL_JOBS="$2"; shift 2 ;;
        --n-steps)        N_STEPS="$2";       shift 2 ;;
        --learning-rate)  LEARNING_RATE="$2"; shift 2 ;;
        --targets)        TARGETS="$2";       shift 2 ;;
        --split-types)    SPLIT_TYPES="$2";   shift 2 ;;
        --split-indices)  SPLIT_INDICES="$2"; shift 2 ;;
        --maxent-values)  MAXENT_VALUES="$2"; shift 2 ;;
        --dir-name)       OUTPUT_BASE="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- Setup ----------
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do
        sleep 1
    done
}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_BASE}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}/logs"

IFS=',' read -ra TARGET_ARRAY   <<< "$TARGETS"
IFS=',' read -ra SPLIT_TYPE_ARRAY <<< "$SPLIT_TYPES"
IFS=',' read -ra SPLIT_IDX_ARRAY  <<< "$SPLIT_INDICES"
IFS=',' read -ra MAXENT_ARRAY     <<< "$MAXENT_VALUES"

# Count total jobs
TOTAL_JOBS=0
for _ in "${TARGET_ARRAY[@]}"; do
    for _ in "${SPLIT_TYPE_ARRAY[@]}"; do
        for _ in "${SPLIT_IDX_ARRAY[@]}"; do
            for _ in "${MAXENT_ARRAY[@]}"; do
                ((TOTAL_JOBS++))
            done
        done
    done
done

echo "========================================"
echo "  SAXS KLD fitting — CaM pulldown"
echo "========================================"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Targets:      ${TARGETS}"
echo "  Split types:  ${SPLIT_TYPES}"
echo "  Split indices:${SPLIT_INDICES}"
echo "  MaxEnt values:${MAXENT_VALUES}"
echo "  N steps:      ${N_STEPS}"
echo "  Parallel jobs:${PARALLEL_JOBS}"
echo "  Total jobs:   ${TOTAL_JOBS}"
echo "========================================"

# ---------- Sweep ----------
JOB_COUNT=0
for TARGET in "${TARGET_ARRAY[@]}"; do
    for SPLIT_TYPE in "${SPLIT_TYPE_ARRAY[@]}"; do
        for SPLIT_IDX in "${SPLIT_IDX_ARRAY[@]}"; do
            for MAXENT in "${MAXENT_ARRAY[@]}"; do
                ((JOB_COUNT++))
                wait_for_slot

                LOG_NAME="SAXS_${TARGET}_${SPLIT_TYPE}_split${SPLIT_IDX}_maxent${MAXENT}"
                LOG_FILE="${OUTPUT_DIR}/logs/${LOG_NAME}.log"

                echo "[${JOB_COUNT}/${TOTAL_JOBS}] ${LOG_NAME}"

                python "${SCRIPT_DIR}/fit_CaM_SAXS_KLD.py" \
                    --target         "$TARGET" \
                    --split-type     "$SPLIT_TYPE" \
                    --split-index    "$SPLIT_IDX" \
                    --maxent-strength "$MAXENT" \
                    --n-steps        "$N_STEPS" \
                    --learning-rate  "$LEARNING_RATE" \
                    --output-dir     "${OUTPUT_DIR}" \
                    > "$LOG_FILE" 2>&1 &
            done
        done
    done
done

echo "Waiting for all jobs to complete..."
wait
echo "All jobs completed."

# ---------- Analysis ----------
echo ""
echo "Running analysis..."
python "${SCRIPT_DIR}/CaM_pulldown_analyse_results.py" \
    --results-dir "${OUTPUT_DIR}"

echo ""
echo "Done. Results and plots in: ${OUTPUT_DIR}"
