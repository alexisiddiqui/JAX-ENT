#!/bin/bash
# Comprehensive analysis pipeline for 4_aSyn (continue-on-error).

cd "$(dirname "$0")" || exit 1

ANA_DIR="analysis"
DIR_WD="$(pwd)/fitting"
RESULTS_DIR_DEFAULT="$(pwd)/fitting/_optimise_aSyn_BV_20260402_021807"
RESULTS_DIR="${1:-$RESULTS_DIR_DEFAULT}"

LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "Using optimization output directory: $RESULTS_DIR"

PROCESS_STATUS="not_run"
SCORE_STATUS="not_run"
MLM_STATUS="not_run"
PLOT_STATUS="not_run"
EXTRACT_STATUS="not_run"
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
    --features-dir "$(pwd)/data/_aSyn/features" \
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

if run_step "process_optimisation_results" "$LOG_DIR/process_optimisation_results.log" \
  python "$ANA_DIR/process_optimisation_results_aSyn_conditions.py" \
    --results-dir "$RESULTS_DIR" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "$(pwd)/data/_aSyn/features" \
    --absolute-paths; then
  PROCESS_STATUS="ok"
else
  PROCESS_STATUS="fail"
fi

BASENAME="$(basename "$RESULTS_DIR")"
PROCESSED_DIR="${DIR_WD}/_processed_${BASENAME}"

if run_step "score_models" "$LOG_DIR/score_models.log" \
  python "$ANA_DIR/score_models_aSyn_conditions.py" \
    --processed-data-dir "$PROCESSED_DIR" \
    --results-dir "$RESULTS_DIR" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --absolute-paths; then
  SCORE_STATUS="ok"
else
  SCORE_STATUS="fail"
fi

SCORES_BASENAME="$(basename "$PROCESSED_DIR")"
SCORES_DIR="${PROCESSED_DIR}/_scores_${SCORES_BASENAME}"
SCORES_CSV="${SCORES_DIR}/model_scores.csv"

if run_step "analyse_scores_mixed_linear_model" "$LOG_DIR/analyse_scores_mixed_linear_model.log" \
  python "$ANA_DIR/analyse_scores_mixed_linear_model_aSyn_conditions.py" \
    --scores-csv-path "$SCORES_CSV" \
    --target-metric "recovery_percent" \
    --filter-mode "both" \
    --analyze-subsets \
    --absolute-paths; then
  MLM_STATUS="ok"
else
  MLM_STATUS="fail"
fi

ANALYSIS_DIR="${PROCESSED_DIR}/_analysis_$(basename "$SCORES_DIR")"
BEFORE_CSV="${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv"
AFTER_CSV="${ANALYSIS_DIR}_filtered/whole_dataset/model_selection_performance_summary.csv"

if run_step "plot_selected_models" "$LOG_DIR/plot_selected_models.log" \
  python "$ANA_DIR/plot_selected_models_aSyn_conditions.py" \
    --before-csv "$BEFORE_CSV" \
    --after-csv "$AFTER_CSV" \
    --output-dir "${ANALYSIS_DIR}/plots_selection"; then
  PLOT_STATUS="ok"
else
  PLOT_STATUS="fail"
fi

EXTRACT_DIR="${PROCESSED_DIR}/_extracted_$(basename "$PROCESSED_DIR")"

if run_step "extract_selected_models" "$LOG_DIR/extract_selected_models.log" \
  python "$ANA_DIR/extract_selected_models_aSyn_conditions.py" \
    --processed-data-dir "$PROCESSED_DIR" \
    --scores-csv "$SCORES_CSV" \
    --selection-csv "$BEFORE_CSV" \
    --output-dir "$EXTRACT_DIR" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --absolute-paths; then
  EXTRACT_STATUS="ok"
else
  EXTRACT_STATUS="fail"
fi

FEAT_DIST_STATUS="not_run"

if run_step "plot_feature_distributions" "$LOG_DIR/plot_feature_distributions.log" \
  python "$ANA_DIR/plot_feature_distributions_aSyn_conditions.py" \
    --extracted-dir "$EXTRACT_DIR" \
    --feature-npz "$(pwd)/data/_aSyn/features/aSyn_featurised.npz" \
    --topology-json "$(pwd)/data/_aSyn/features/topology.json" \
    --top-pdb "$(pwd)/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb" \
    --traj-xtc "$(pwd)/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc" \
    --absolute-paths; then
  FEAT_DIST_STATUS="ok"
else
  FEAT_DIST_STATUS="fail"
fi

echo
echo "=== Pipeline summary ==="
echo "recovery_analysis: $RECOVERY_STATUS"
echo "weights_validation: $WEIGHTS_STATUS"
echo "analyse_loss: $LOSS_STATUS"
echo "process_optimisation_results: $PROCESS_STATUS"
echo "score_models: $SCORE_STATUS"
echo "analyse_scores_mixed_linear_model: $MLM_STATUS"
echo "plot_selected_models: $PLOT_STATUS"
echo "extract_selected_models: $EXTRACT_STATUS"
echo "plot_feature_distributions: $FEAT_DIST_STATUS"

echo "Logs: $LOG_DIR"
