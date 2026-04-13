#!/bin/bash
# Runs:
# optimise_ISO_TRI_BI_splits_maxENT.py
# ../analysis/recovery_analysis_ISO_TRI_BI_precluster.py
# ../analysis/weights_validation_ISO_TRI_BI_precluster.py
# ../analysis/CV_validation_ISO_TRI_BI_precluster.py


# set working directory to the script's location
cd "$(dirname "$0")" || exit
DIR_WD=$(pwd)
ANA_DIR="../../analysis"
echo "Working directory: $DIR_WD"

# --- Changed: add configurable defaults and extended argument parsing ---
# Defaults (can be overridden via CLI)
PARALLEL_JOBS=8
DEFAULT_MAXENT_VALUES_STR="1,10,100,1000"


MAXENT_VALUES_STR="$DEFAULT_MAXENT_VALUES_STR"
DIR_NAME="_optimise_test_SIGMA_500"
N_STEPS=500
INITIAL_STEPS=0
INITIAL_LR=1.0
LEARNING_RATE=1.0
EMA_ALPHA=0.5
FORWARD_MODEL_SCALING=1000.0

# --- Added defaults for ensembles, losses and split types ---
DEFAULT_ENSEMBLES_STR="ISO_TRI,ISO_BI"
ENSEMBLES_STR="$DEFAULT_ENSEMBLES_STR"
DEFAULT_LOSSES_STR="mcMSE,MSE,Sigma_MSE"

LOSSES_STR="$DEFAULT_LOSSES_STR"
DEFAULT_SPLIT_TYPES_STR="random,sequence,sequence_cluster,stratified,spatial"
DEFAULT_SPLIT_TYPES_STR="sequence_cluster"

SPLIT_TYPES_STR="$DEFAULT_SPLIT_TYPES_STR"
# --- end added block ---

# Parse args (supports --flag value and --flag=value)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)
      PARALLEL_JOBS="$2"; shift 2;;
    --jobs=*)
      PARALLEL_JOBS="${1#*=}"; shift;;
    --maxent-values)
      MAXENT_VALUES_STR="$2"; shift 2;;
    --maxent-values=*)
      MAXENT_VALUES_STR="${1#*=}"; shift;;
    --dir-name)
      DIR_NAME="$2"; shift 2;;
    --dir-name=*)
      DIR_NAME="${1#*=}"; shift;;
    --n-steps)
      N_STEPS="$2"; shift 2;;
    --n-steps=*)
      N_STEPS="${1#*=}"; shift;;
    --initial-steps)
      INITIAL_STEPS="$2"; shift 2;;
    --initial-steps=*)
      INITIAL_STEPS="${1#*=}"; shift;;
    --initial-learning-rate)
      INITIAL_LR="$2"; shift 2;;
    --initial-learning-rate=*)
      INITIAL_LR="${1#*=}"; shift;;
    --learning-rate)
      LEARNING_RATE="$2"; shift 2;;
    --learning-rate=*)
      LEARNING_RATE="${1#*=}"; shift;;
    --ema-alpha)
      EMA_ALPHA="$2"; shift 2;;
    --ema-alpha=*)
      EMA_ALPHA="${1#*=}"; shift;;
    --forward-model-scaling)
      FORWARD_MODEL_SCALING="$2"; shift 2;;
    --forward-model-scaling=*)
      FORWARD_MODEL_SCALING="${1#*=}"; shift;;
    --ensembles)
      ENSEMBLES_STR="$2"; shift 2;;
    --ensembles=*)
      ENSEMBLES_STR="${1#*=}"; shift;;
    --losses)
      LOSSES_STR="$2"; shift 2;;
    --losses=*)
      LOSSES_STR="${1#*=}"; shift;;
    --split-types)
      SPLIT_TYPES_STR="$2"; shift 2;;
    --split-types=*)
      SPLIT_TYPES_STR="${1#*=}"; shift;;
    -h|--help)
      echo "Usage: $0 [--ensembles a,b] [--losses x,y] [--split-types s,t] [--maxent-values a,b,c] [--dir-name name] [--n-steps N] [--initial-steps M] [--initial-learning-rate X] [--learning-rate Y] [--ema-alpha Z] [--forward-model-scaling S] [-j|--jobs N]"
      exit 0;;
    *)
      break;;
  esac
done

echo "Parallel jobs limit: $PARALLEL_JOBS"
echo "Maxent values (raw): $MAXENT_VALUES_STR"
echo "DIR_NAME: $DIR_NAME"
echo "n-steps: $N_STEPS, initial-steps: $INITIAL_STEPS, initial-learning-rate: $INITIAL_LR, learning-rate: $LEARNING_RATE, ema-alpha: $EMA_ALPHA, forward-model-scaling: $FORWARD_MODEL_SCALING"
echo "Ensembles (raw): $ENSEMBLES_STR"
echo "Losses (raw): $LOSSES_STR"
echo "Split types (raw): $SPLIT_TYPES_STR"

# Convert comma-separated strings into arrays
IFS=',' read -r -a MAXENT_VALUES <<< "$MAXENT_VALUES_STR"
IFS=',' read -r -a ENSEMBLES <<< "$ENSEMBLES_STR"
IFS=',' read -r -a LOSSES <<< "$LOSSES_STR"
IFS=',' read -r -a SPLIT_TYPES <<< "$SPLIT_TYPES_STR"
# --- end changed block ---

# --- Added: helpers to limit parallel background jobs ---
running_jobs_count() {
  jobs -rp | wc -l
}

wait_for_slot() {
  # Wait until the number of running background jobs is less than PARALLEL_JOBS
  while [ "$(running_jobs_count)" -ge "$PARALLEL_JOBS" ]; do
    sleep 1
  done
}

# Ensure we wait for all background jobs on exit
cleanup() {
  wait
}
trap cleanup EXIT
# --- end added block ---

rm -rf logs
mkdir -p logs
# --- Removed hard-coded arrays and using parsed arrays instead ---
# ENSEMBLES and SPLIT_TYPES and LOSSES now come from parsed inputs above
# ENSEMBLES=("ISO_TRI" "ISO_BI")
# ENSEMBLES=("ISO_TRI")
# 
# SPLIT_TYPES=("random" "sequence" "sequence_cluster" "stratified" "spatial")
# SPLIT_TYPES=("random")
# LOSSES=("mcMSE" "MSE")
# LOSSES=("MSE" )
# --- end replacement ---

# --- Removed hard-coded MAXENT_VALUES definitions; using parsed MAXENT_VALUES array ---
# MAXENT_VALUES will come from the parsed MAXENT_VALUES above
# MAXENT_VALUES=(1 10 100  1000 10000)
# MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000)
# MAXENT_VALUES=(1 10)



# MAXENT_VALUES=(100000 1000000 10000000 100000000 1000000000)
# MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000 1000000 1000000000)
time_data="_$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${DIR_NAME}_${time_data}"
OPT_OUTPUT_DIR="${DIR_WD}/${OUTPUT_DIR}"
# --- Fixed: remove stray brace in ANA_OUTPUT_DIR ---
ANA_OUTPUT_DIR="${ANA_DIR}/${OUTPUT_DIR}"
# --- end fix ---
echo "Output directory: $OPT_OUTPUT_DIR"
mkdir -p "${OPT_OUTPUT_DIR}/logs"
for ENSEMBLE in "${ENSEMBLES[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    echo "Running $ENSEMBLE-$LOSS in parallel for maxent log-scaled values"
    for SPLIT in "${SPLIT_TYPES[@]}"; do
      echo "  Split type: $SPLIT"
      for MAXENT in "${MAXENT_VALUES[@]}"; do
        echo "    Maxent: $MAXENT"
        # --- Changed: ensure no more than PARALLEL_JOBS are running concurrently ---
        wait_for_slot
        python optimise_ISO_TRI_BI_splits_Sigma.py \
          --ensemble "$ENSEMBLE" \
          --loss-function "$LOSS" \
          --maxent-range "$MAXENT,$MAXENT" \
          --split-types "$SPLIT" \
          --n-steps "$N_STEPS" \
          --initial-steps "$INITIAL_STEPS" \
          --initial-learning-rate "$INITIAL_LR" \
          --learning-rate "$LEARNING_RATE" \
          --ema-alpha "$EMA_ALPHA" \
          --forward-model-scaling "$FORWARD_MODEL_SCALING" \
          --output-dir "$OPT_OUTPUT_DIR" \
          > "${OPT_OUTPUT_DIR}/logs/${ENSEMBLE}_${LOSS}_maxent${MAXENT}_split${SPLIT}.log" 2>&1 &
      done
      wait  # Wait for all background jobs for this SPLIT to finish
      echo "Completed $ENSEMBLE-$LOSS with $SPLIT"
    done
  done
done
wait  # Wait for all background jobs to finish
echo "All optimisation tasks completed."
echo "Starting analysis scripts..."
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

