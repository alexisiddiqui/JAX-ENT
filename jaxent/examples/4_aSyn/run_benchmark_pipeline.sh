#!/bin/bash
# run_benchmark_pipeline.sh
# Full benchmark pipeline for ensemble-size sweep:
#   1. Truncate trajectories (gmx trjconv)
#   2. Featurise all combos (jaxent-featurise)
#   3. Generate shape-space clustering for all combos
#   4. Fit + analyse all 6 (md_source x time) combos
#
# Usage:
#   bash run_benchmark_pipeline.sh [OPTIONS]
#
# Options passed through to run_benchmark_fitting.sh:
#   -j|--jobs N           Parallel fitting jobs (default: 4)
#   --n-steps N           Optimisation steps per run (default: 5000)
#   --maxent-values A,... Maxent sweep values
#   --bvreg-values A,...  BV reg sweep values
#   --bv-reg-losses L,... BV reg loss types (default: L1)
#   --split-types S,...   Split types (default: sequence_cluster)
#   --combos KEY,...      Subset of combos to run (default: all 6)
#   --skip-analysis       Skip comprehensive analysis after fitting
#
# Pipeline control:
#   --skip-truncate       Skip trajectory truncation step
#   --skip-featurise      Skip featurisation step
#   --skip-cluster        Skip clustering step
#   --skip-fit            Skip fitting step (dry-run of pipeline steps only)

set -euo pipefail
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR="$(pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
FITTING_DIR="${SCRIPT_DIR}/fitting"

SKIP_TRUNCATE=0
SKIP_FEATURISE=0
SKIP_CLUSTER=0
SKIP_FIT=0

# Collect fitting passthrough args
FITTING_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-truncate)  SKIP_TRUNCATE=1; shift;;
    --skip-featurise) SKIP_FEATURISE=1; shift;;
    --skip-cluster)   SKIP_CLUSTER=1; shift;;
    --skip-fit)       SKIP_FIT=1; shift;;
    # All other args pass through to run_benchmark_fitting.sh
    *) FITTING_ARGS+=("$1"); shift;;
  esac
done

TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
PIPELINE_LOG="${SCRIPT_DIR}/benchmark_pipeline_${TIMESTAMP}.log"

run_step() {
    local name="$1"; shift
    echo ""
    echo "========================================"
    echo "[STEP] ${name}"
    echo "========================================"
    if "$@"; then
        echo "[OK]   ${name}"
    else
        echo "[FAIL] ${name} — aborting pipeline"
        exit 1
    fi
}

echo "========================================"
echo "Benchmark pipeline: $(date)"
echo "Log: ${PIPELINE_LOG}"
echo "========================================"

{

if [[ "$SKIP_TRUNCATE" -eq 0 ]]; then
    run_step "Truncate trajectories" \
        bash "${DATA_DIR}/truncate_trajectories.sh"
else
    echo "[SKIP] Truncate trajectories"
fi

if [[ "$SKIP_FEATURISE" -eq 0 ]]; then
    run_step "Featurise benchmark combos" \
        bash "${DATA_DIR}/featurise_benchmark.sh"
else
    echo "[SKIP] Featurise benchmark combos"
fi

if [[ "$SKIP_CLUSTER" -eq 0 ]]; then
    run_step "Generate shape-space clustering" \
        python "${DATA_DIR}/cluster_benchmark_ensembles.py" \
            --base-dir "${SCRIPT_DIR}"
else
    echo "[SKIP] Generate shape-space clustering"
fi

if [[ "$SKIP_FIT" -eq 0 ]]; then
    run_step "Fitting and analysis" \
        bash "${FITTING_DIR}/run_benchmark_fitting.sh" "${FITTING_ARGS[@]}"
else
    echo "[SKIP] Fitting and analysis"
fi

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "========================================"

} 2>&1 | tee "${PIPELINE_LOG}"
