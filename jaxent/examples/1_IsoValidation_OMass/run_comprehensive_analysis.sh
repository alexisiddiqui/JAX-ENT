#!/bin/bash
# Comprehensive Analysis Pipeline for 1_IsoValidation_OMass

cd "$(dirname "$0")" || exit

ANA_DIR="analysis"
DIR_WD="$(pwd)/fitting/jaxENT"
# Example optimization output directory - CHANGE THIS to match your actual output
OPT_OUTPUT_DIR="${DIR_WD}/_optimise_FIGURE_SIGMA_5000__20260410_201754"

mkdir -p "${OPT_OUTPUT_DIR}/logs"

echo "Using Optimization Output Directory: $OPT_OUTPUT_DIR"
echo "Starting comprehensive analysis pipeline..."

# # 1. Process Optimization Results
# echo "Step 1: Processing optimization results..."
# python "${ANA_DIR}/process_optimisation_results.py" \
#   --results-dir "$OPT_OUTPUT_DIR" \
#   --datasplit-dir "${DIR_WD}/_datasplits" \
#   --features-dir "${DIR_WD}/_featurise" \
#   --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
#   > "${OPT_OUTPUT_DIR}/logs/process_optimisation_results.log" 2>&1

# BASENAME=$(basename "$OPT_OUTPUT_DIR")
# PROCESSED_DIR="${DIR_WD}/_processed_${BASENAME}"

# if [ ! -d "$PROCESSED_DIR" ]; then
#     echo "Error: Processed directory not found: $PROCESSED_DIR"
#     exit 1
# fi

# # 2. Score Models
# echo "Step 2: Scoring models..."
# python "${ANA_DIR}/score_models_ISO_TRI_BI.py" \
#   --processed-data-dir "$PROCESSED_DIR" \
#   --datasplit-dir "${DIR_WD}/_datasplits" \
#   --features-dir "${DIR_WD}/_featurise" \
#   --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
#   > "${OPT_OUTPUT_DIR}/logs/score_models.log" 2>&1

# SCORES_BASENAME=$(basename "$PROCESSED_DIR")
# SCORES_DIR="${PROCESSED_DIR}/_scores_${SCORES_BASENAME}"

# if [ ! -d "$SCORES_DIR" ]; then
#     echo "Error: Scores directory not found: $SCORES_DIR"
#     exit 1
# fi

# # 3. Analyze Scores (Linear Modelling)
# echo "Step 3: Analyzing scores with mixed linear model..."
# python "${ANA_DIR}/analyse_scores_mixed_linear_model.py" \
#   --scores-csv-path "${SCORES_DIR}/model_scores.csv" \
#   --target-metric "recovery_percent" \
#   --filter-mode "both" \
#   --analyze-subsets \
#   > "${OPT_OUTPUT_DIR}/logs/analyse_scores_mixed_linear_model.log" 2>&1

# ANALYSIS_DIR="${PROCESSED_DIR}/_analysis__scores_${SCORES_BASENAME}"


# echo "Plotting selected models (unfiltered)..."
# CLUSTER_POP_CSV="${ANA_OUTPUT_DIR}/conformational_recovery_maxent_data.csv"
# PLOT_EXTRA_ARGS=()
# if [ -f "$CLUSTER_POP_CSV" ]; then
#   PLOT_EXTRA_ARGS+=(--cluster-populations-csv "$CLUSTER_POP_CSV")
# fi
# # 4. Plot Selected Models
# echo "Step 4: Plotting selected models..."
# python "${ANA_DIR}/plot_selected_models_ISO_TRI_BI.py" \
#   --before-csv "${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv" \
#   --after-csv "${ANALYSIS_DIR}_filtered/whole_dataset/model_selection_performance_summary.csv" \
#   --output-dir "${ANALYSIS_DIR}/plots_selection" \
#   "${PLOT_EXTRA_ARGS[@]}" \
#   > "${OPT_OUTPUT_DIR}/logs/plot_selected_models.log" 2>&1


# echo "Extracting selected models..."


# python "${ANA_DIR}/extract_selected_models.py" \
#   --processed-data-dir "$PROCESSED_DIR" \
#   --scores-csv "${SCORES_DIR}/model_scores.csv" \
#   --selection-csv "${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv" \
#   > "${OPT_OUTPUT_DIR}/logs/extract_selected_models.log" 2>&1

# echo "All analysis tasks completed successfully."
# echo "Results are located in and around: $PROCESSED_DIR"

# Run analysis scripts sequentially
echo "Running recovery analysis..."
python "${ANA_DIR}/recovery_analysis_ISO_TRI_BI_precluster.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/recovery_analysis.log" 2>&1
echo "Running weights validation..."
python "${ANA_DIR}/weights_validation_ISO_TRI_BI_precluster.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/weights_validation.log" 2>&1
echo "Running CV validation..."
python "${ANA_DIR}/CV_validation_ISO_TRI_BI_precluster.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/CV_validation.log" 2>&1
python "${ANA_DIR}/analyse_loss_ISO_TRI_BI.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/Analyse_Loss.log" 2>&1

# New comprehensive analysis pipeline
echo "Processing optimization results..."
python "${ANA_DIR}/process_optimisation_results.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  --datasplit-dir "${DIR_WD}/_datasplits" \
  --features-dir "${DIR_WD}/_featurise" \
  --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
  > "${OPT_OUTPUT_DIR}/logs/process_optimisation_results.log" 2>&1

# Determine the processed data directory name
# process_optimisation_results.py creates _processed_<basename> as a SIBLING of OPT_OUTPUT_DIR
BASENAME=$(basename "$OPT_OUTPUT_DIR")
PROCESSED_DIR="${DIR_WD}/_processed_${BASENAME}"

echo "Scoring models..."
python "${ANA_DIR}/score_models_ISO_TRI_BI.py" \
  --processed-data-dir "$PROCESSED_DIR" \
  --datasplit-dir "${DIR_WD}/_datasplits" \
  --features-dir "${DIR_WD}/_featurise" \
  --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
  > "${OPT_OUTPUT_DIR}/logs/score_models.log" 2>&1

# Determine the scores directory name
# score_models_ISO_TRI_BI.py creates _scores_<basename> INSIDE PROCESSED_DIR
SCORES_BASENAME=$(basename "$PROCESSED_DIR")
SCORES_DIR="${PROCESSED_DIR}/_scores_${SCORES_BASENAME}"

echo "Analyzing scores with mixed linear model..."
python "${ANA_DIR}/analyse_scores_mixed_linear_model.py" \
  --scores-csv-path "${SCORES_DIR}/model_scores.csv" \
  --target-metric "recovery_percent" \
  --filter-mode "both" \
  --analyze-subsets \
  > "${OPT_OUTPUT_DIR}/logs/analyse_scores_mixed_linear_model.log" 2>&1

# Determine the analysis directory name
# analyse_scores_mixed_linear_model.py creates _analysis_<scores_parent_basename> as a SIBLING of SCORES_DIR
# For unfiltered: _analysis__scores_<SCORES_BASENAME>
# For filtered:   _analysis__scores_<SCORES_BASENAME>_filtered
ANALYSIS_DIR="${PROCESSED_DIR}/_analysis__scores_${SCORES_BASENAME}"

# Plot model selection results for both filtered and unfiltered
echo "Plotting selected models (unfiltered)..."
CLUSTER_POP_CSV="${ANA_OUTPUT_DIR}/conformational_recovery_maxent_data.csv"
PLOT_EXTRA_ARGS=()
if [ -f "$CLUSTER_POP_CSV" ]; then
  PLOT_EXTRA_ARGS+=(--cluster-populations-csv "$CLUSTER_POP_CSV")
fi
python "${ANA_DIR}/plot_selected_models_ISO_TRI_BI.py" \
  --before-csv "${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv" \
  --after-csv "${ANALYSIS_DIR}_filtered/whole_dataset/model_selection_performance_summary.csv" \
  --output-dir "${ANALYSIS_DIR}/plots_selection" \
  "${PLOT_EXTRA_ARGS[@]}" \
  > "${OPT_OUTPUT_DIR}/logs/plot_selected_models.log" 2>&1

echo "Extracting selected models..."
python "${ANA_DIR}/extract_selected_models.py" \
  --processed-data-dir "$PROCESSED_DIR" \
  --scores-csv "${SCORES_DIR}/model_scores.csv" \
  --selection-csv "${ANALYSIS_DIR}/whole_dataset/model_selection_performance_summary.csv" \
  > "${OPT_OUTPUT_DIR}/logs/extract_selected_models.log" 2>&1

echo "All analysis tasks completed."
echo "Results are saved in $OPT_OUTPUT_DIR"
echo "Script finished."

