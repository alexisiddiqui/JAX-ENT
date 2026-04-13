#!/bin/bash
# SAXS+HDX combined MaxEnt reweighting sweep for CaM+CDZ and CaM-CDZ.
#
# Parallelises at the bash level (one Python process per combination).
# After all jobs complete, runs the analysis script automatically.
#
# Usage:
#   bash run_fit_CaM_HDX_SAXS_KLD.sh [options]
#
# Options:
#   --jobs          N     Max parallel jobs          (default: 10)
#   --n-steps       N     Optimisation steps          (default: 50000)
#   --learning-rate F     Learning rate               (default: 1.0)
#   --targets       A,B   Comma-separated targets     (default: CaM+CDZ,CaM-CDZ)
#   --split-indices 0,1   Comma-separated indices   (default: 0,1,2)
#   --saxs-weights  A,B   Comma-separated SAXS weights (default: 1.0)
#   --hdx-weights   A,B   Comma-separated HDX weights  (default: 1.0)
#   --maxent-values A,B   Comma-separated strengths   (default: 7-value sweep)
#   --dir-name      NAME  Output base directory name  (default: _optimise_CaM_HDX_SAXS_KLD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../..:${PYTHONPATH:-}"

# ---------- Defaults ----------
PARALLEL_JOBS=5
N_STEPS=500
LEARNING_RATE=1.0
TARGETS="CaM+CDZ,CaM-CDZ"
SPLIT_INDICES="0,1,2"
SAXS_WEIGHTS="1.0,10.0"
HDX_WEIGHTS="1.0,10.0"
MAXENT_VALUES="1.0,10.0,100.0,1000.0,10000.0"
OUTPUT_BASE="_optimise_CaM_HDX_SAXS_KLD"

# ---------- Argument parsing ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)               PARALLEL_JOBS="$2"; shift 2 ;;
        --n-steps)            N_STEPS="$2";       shift 2 ;;
        --learning-rate)      LEARNING_RATE="$2"; shift 2 ;;
        --targets)            TARGETS="$2";       shift 2 ;;
        --split-indices)      SPLIT_INDICES="$2"; shift 2 ;;
        --saxs-weights)       SAXS_WEIGHTS="$2";  shift 2 ;;
        --hdx-weights)        HDX_WEIGHTS="$2";   shift 2 ;;
        --maxent-values)      MAXENT_VALUES="$2"; shift 2 ;;
        --dir-name)           OUTPUT_BASE="$2";   shift 2 ;;
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
IFS=',' read -ra SPLIT_IDX_ARRAY  <<< "$SPLIT_INDICES"
IFS=',' read -ra SAXS_W_ARRAY   <<< "$SAXS_WEIGHTS"
IFS=',' read -ra HDX_W_ARRAY    <<< "$HDX_WEIGHTS"
IFS=',' read -ra MAXENT_ARRAY     <<< "$MAXENT_VALUES"

# Count total jobs
# Note: typically you might lock HDX and SAXS split index to be the same to avoid explosion,
# but if they differ, this nests them. Here we just loop over them independently.
TOTAL_JOBS=0
for _ in "${TARGET_ARRAY[@]}"; do
    for _ in "${SPLIT_IDX_ARRAY[@]}"; do
            for _ in "${SAXS_W_ARRAY[@]}"; do
                for _ in "${HDX_W_ARRAY[@]}"; do
                    for _ in "${MAXENT_ARRAY[@]}"; do
                        ((TOTAL_JOBS++))
                    done
                done
            done
        done
    done

echo "========================================"
echo "  HDX+SAXS KLD fitting — CaM pulldown"
echo "========================================"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Targets:      ${TARGETS}"
echo "  Split indices:${SPLIT_INDICES}"
echo "  SAXS weights: ${SAXS_WEIGHTS}"
echo "  HDX weights:  ${HDX_WEIGHTS}"
echo "  MaxEnt values:${MAXENT_VALUES}"
echo "  N steps:      ${N_STEPS}"
echo "  Parallel jobs:${PARALLEL_JOBS}"
echo "  Total jobs:   ${TOTAL_JOBS}"
echo "========================================"

# ---------- Sweep ----------
# ---------- Sweep ----------
JOB_COUNT=0
for TARGET in "${TARGET_ARRAY[@]}"; do
    for SPLIT_IDX in "${SPLIT_IDX_ARRAY[@]}"; do
        for SAXS_W in "${SAXS_W_ARRAY[@]}"; do
            for HDX_W in "${HDX_W_ARRAY[@]}"; do
                for MAXENT in "${MAXENT_ARRAY[@]}"; do
                    ((JOB_COUNT++))
                    wait_for_slot

                    LOG_NAME="HDX_SAXS_${TARGET}_split${SPLIT_IDX}_maxent${MAXENT}_w${SAXS_W}_v${HDX_W}"
                    LOG_FILE="${OUTPUT_DIR}/logs/${LOG_NAME}.log"

                    echo "[${JOB_COUNT}/${TOTAL_JOBS}] ${LOG_NAME}"

                    python "${SCRIPT_DIR}/fit_CaM_HDX_SAXS_KLD.py" \
                        --target         "$TARGET" \
                        --split-index    "$SPLIT_IDX" \
                        --saxs-weight    "$SAXS_W" \
                        --hdx-weight     "$HDX_W" \
                        --maxent-strength "$MAXENT" \
                        --n-steps        "$N_STEPS" \
                        --learning-rate  "$LEARNING_RATE" \
                        --output-dir     "${OUTPUT_DIR}" \
                        > "$LOG_FILE" 2>&1 &
                done
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
