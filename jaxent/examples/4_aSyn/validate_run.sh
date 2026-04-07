#!/bin/bash
# Validation analysis for 4_aSyn (continue-on-error).

cd "$(dirname "$0")" || exit 1

ANA_DIR="analysis"
DIR_WD="$(pwd)/fitting"
RESULTS_DIR_DEFAULT="$(pwd)/fitting/_optimise_aSyn_BV_20260402_021807"
RESULTS_DIR="${1:-$RESULTS_DIR_DEFAULT}"

LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Using optimization output directory: $RESULTS_DIR"

RECOVERY_STATUS="not_run"
WEIGHTS_STATUS="not_run"
LOSS_STATUS="not_run"

run_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2

  echo "[RUN] $step_name"
  if "$@" > "$log_file" 2>&1; then
    echo "[OK]  $step_name"
    return 0
  else
    echo "[FAIL] $step_name (see $log_file)"
    return 1
  fi
}

if run_step "recovery_analysis" "$LOG_DIR/recovery_analysis.log" \
  python "$ANA_DIR/recovery_analysis_aSyn_conditions_2d_bv.py" \
    --results-dir "$RESULTS_DIR" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "$(pwd)/data/_cluster_aSyn/features" \
    --absolute-paths; then
  RECOVERY_STATUS="ok"
else
  RECOVERY_STATUS="fail"
fi

if run_step "weights_validation" "$LOG_DIR/weights_validation.log" \
  python "$ANA_DIR/weights_validation_aSyn_conditions_2d_bv.py" \
    --results-dir "$RESULTS_DIR" \
    --absolute-paths; then
  WEIGHTS_STATUS="ok"
else
  WEIGHTS_STATUS="fail"
fi

if run_step "analyse_loss" "$LOG_DIR/analyse_loss.log" \
  python "$ANA_DIR/analyse_loss_aSyn_conditions_2d_bv.py" \
    --results-dir "$RESULTS_DIR"; then
  LOSS_STATUS="ok"
else
  LOSS_STATUS="fail"
fi

echo
echo "=== Validation summary ==="
echo "recovery_analysis: $RECOVERY_STATUS"
echo "weights_validation: $WEIGHTS_STATUS"
echo "analyse_loss: $LOSS_STATUS"

echo "Logs: $LOG_DIR"
