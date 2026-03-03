#!/bin/bash
# Comprehensive Analysis Pipeline for 1_IsoValidation_OMass

cd "$(dirname "$0")" || exit

ANA_DIR="analysis"
DIR_WD="$(pwd)/fitting/jaxENT"
# Example optimization output directory - CHANGE THIS to match your actual output
OPT_OUTPUT_DIR="${DIR_WD}/_optimise_test_SIGMA_500__20260226_011958"

mkdir -p "${OPT_OUTPUT_DIR}/logs"

echo "Using Optimization Output Directory: $OPT_OUTPUT_DIR"
echo "Starting comprehensive analysis pipeline..."

# 1. Process Optimization Results
echo "Step 1: Processing optimization results..."
python "${ANA_DIR}/process_optimisation_results.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  --datasplit-dir "${DIR_WD}/_datasplits" \
  --features-dir "${DIR_WD}/_featurise" \
  --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
  > "${OPT_OUTPUT_DIR}/logs/process_optimisation_results.log" 2>&1

BASENAME=$(basename "$OPT_OUTPUT_DIR")
PROCESSED_DIR="${DIR_WD}/_processed_${BASENAME}"

if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Error: Processed directory not found: $PROCESSED_DIR"
    exit 1
fi

# 2. Score Models
echo "Step 2: Scoring models..."
python "${ANA_DIR}/score_models_ISO_TRI_BI.py" \
  --processed-data-dir "$PROCESSED_DIR" \
  --datasplit-dir "${DIR_WD}/_datasplits" \
  --features-dir "${DIR_WD}/_featurise" \
  --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
  > "${OPT_OUTPUT_DIR}/logs/score_models.log" 2>&1

SCORES_BASENAME=$(basename "$PROCESSED_DIR")
SCORES_DIR="${PROCESSED_DIR}/_scores_${SCORES_BASENAME}"

if [ ! -d "$SCORES_DIR" ]; then
    echo "Error: Scores directory not found: $SCORES_DIR"
    exit 1
fi

# 3. Analyze Scores (Linear Modelling)
echo "Step 3: Analyzing scores with mixed linear model..."
python "${ANA_DIR}/analyse_scores_mixed_linear_model.py" \
  --scores-csv-path "${SCORES_DIR}/model_scores.csv" \
  --target-metric "recovery_percent" \
  --filter-mode "both" \
  --analyze-subsets \
  > "${OPT_OUTPUT_DIR}/logs/analyse_scores_mixed_linear_model.log" 2>&1

ANALYSIS_DIR="${PROCESSED_DIR}/_analysis__scores_${SCORES_BASENAME}"

# 4. Plot Selected Models
echo "Step 4: Plotting selected models..."
python "${ANA_DIR}/plot_selected_models_ISO_TRI_BI.py" \
  --before-csv "${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv" \
  --after-csv "${ANALYSIS_DIR}_filtered/whole_dataset/model_selection_performance_summary.csv" \
  --output-dir "${ANALYSIS_DIR}/plots_selection" \
  > "${OPT_OUTPUT_DIR}/logs/plot_selected_models.log" 2>&1

echo "All analysis tasks completed successfully."
echo "Results are located in and around: $PROCESSED_DIR"
