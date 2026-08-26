#!/bin/bash
set -euo pipefail

# Runs:
# optimise_ISO_TRI_BI_splits_maxENT.py
# ../analysis/recovery_analysis_ISO_TRI_BI_precluster.py
# ../analysis/weights_validation_ISO_TRI_BI_precluster.py
# ../analysis/CV_validation_ISO_TRI_BI_precluster.py


# set working directory to the script's location
cd "$(dirname "$0")" || exit
DIR_WD=$(pwd)
ANA_DIR="../../analysis"
PYTHON_RUNNER=(env UV_CACHE_DIR=/tmp/jaxent-uv-cache uv run --no-sync python)
echo "Working directory: $DIR_WD"

# --- Changed: add configurable defaults and extended argument parsing ---
# Defaults (can be overridden via CLI)
PARALLEL_JOBS=10
DEFAULT_MAXENT_VALUES_STR="1,10,100,1000,10000,100000,1000000"
DEFAULT_MAXENT_VALUES_STR="1,5,10,50,100,500,1000"


MAXENT_VALUES_STR="$DEFAULT_MAXENT_VALUES_STR"
DIR_NAME="_optimise_test_FIGURE_SIGMA_5000"
N_STEPS=5000
LEARNING_RATE=1.0
EMA_ALPHA=0.5
FORWARD_MODEL_SCALING=1000.0

# --- Added defaults for ensembles, losses and split types ---
DEFAULT_ENSEMBLES_STR="ISO_TRI,ISO_BI"
ENSEMBLES_STR="$DEFAULT_ENSEMBLES_STR"
DEFAULT_LOSSES_STR="MSE"

LOSSES_STR="$DEFAULT_LOSSES_STR"
DEFAULT_FRAME_AVERAGING_MODES_STR="log_pf,uptake"
FRAME_AVERAGING_MODES_STR="$DEFAULT_FRAME_AVERAGING_MODES_STR"
DEFAULT_SPLIT_TYPES_STR="random,sequence,sequence_cluster,stratified,spatial"
DEFAULT_SPLIT_TYPES_STR="sequence_cluster,spatial"

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
    --frame-averaging-modes)
      FRAME_AVERAGING_MODES_STR="$2"; shift 2;;
    --frame-averaging-modes=*)
      FRAME_AVERAGING_MODES_STR="${1#*=}"; shift;;
    --split-types)
      SPLIT_TYPES_STR="$2"; shift 2;;
    --split-types=*)
      SPLIT_TYPES_STR="${1#*=}"; shift;;
    -h|--help)
      echo "Usage: $0 [--ensembles a,b] [--losses x,y] [--frame-averaging-modes log_pf,rate,uptake] [--split-types s,t] [--maxent-values a,b,c] [--dir-name name] [--n-steps N] [--learning-rate Y] [--ema-alpha Z] [--forward-model-scaling S] [-j|--jobs N]"
      exit 0;;
    *)
      break;;
  esac
done

echo "Parallel jobs limit: $PARALLEL_JOBS"
echo "Maxent values (raw): $MAXENT_VALUES_STR"
echo "DIR_NAME: $DIR_NAME"
echo "n-steps: $N_STEPS, learning-rate: $LEARNING_RATE, ema-alpha: $EMA_ALPHA, forward-model-scaling: $FORWARD_MODEL_SCALING"
echo "Ensembles (raw): $ENSEMBLES_STR"
echo "Losses (raw): $LOSSES_STR"
echo "Frame averaging modes (raw): $FRAME_AVERAGING_MODES_STR"
echo "Split types (raw): $SPLIT_TYPES_STR"

# Convert comma-separated strings into arrays
IFS=',' read -r -a MAXENT_VALUES <<< "$MAXENT_VALUES_STR"
IFS=',' read -r -a ENSEMBLES <<< "$ENSEMBLES_STR"
IFS=',' read -r -a LOSSES <<< "$LOSSES_STR"
IFS=',' read -r -a FRAME_AVERAGING_MODES <<< "$FRAME_AVERAGING_MODES_STR"
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
for FRAME_AVERAGING_MODE in "${FRAME_AVERAGING_MODES[@]}"; do
  if [[ "$FRAME_AVERAGING_MODE" != "log_pf" && "$FRAME_AVERAGING_MODE" != "rate" && "$FRAME_AVERAGING_MODE" != "uptake" ]]; then
    echo "Invalid frame averaging mode: $FRAME_AVERAGING_MODE (expected log_pf, rate, or uptake)" >&2
    exit 2
  fi
done

RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RESULT_DIRS=()

run_campaign() {
  local frame_averaging_mode="$1"
  local output_dir="${DIR_NAME}_${frame_averaging_mode}_${RUN_TIMESTAMP}"
  local opt_output_dir="${DIR_WD}/${output_dir}"
  local ana_output_dir="${ANA_DIR}/${output_dir}"
  local basename
  local processed_dir
  local scores_basename
  local scores_dir
  local analysis_dir
  local cluster_pop_csv
  local plot_extra_args=()

  RESULT_DIRS+=("$opt_output_dir")
  echo "Starting frame averaging campaign: $frame_averaging_mode"
  echo "Output directory: $opt_output_dir"
  mkdir -p "${opt_output_dir}/logs"

  for ENSEMBLE in "${ENSEMBLES[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      echo "Running $ENSEMBLE-$LOSS with $frame_averaging_mode frame averaging"
      for SPLIT in "${SPLIT_TYPES[@]}"; do
        local batch_pids=()
        echo "  Split type: $SPLIT"
        for MAXENT in "${MAXENT_VALUES[@]}"; do
          echo "    Maxent: $MAXENT"
          wait_for_slot
          "${PYTHON_RUNNER[@]}" optimise_ISO_TRI_BI_splits_Sigma.py \
            --ensemble "$ENSEMBLE" \
            --loss-function "$LOSS" \
            --maxent-range "$MAXENT,$MAXENT" \
            --split-types "$SPLIT" \
            --n-steps "$N_STEPS" \
            --learning-rate "$LEARNING_RATE" \
            --ema-alpha "$EMA_ALPHA" \
            --forward-model-scaling "$FORWARD_MODEL_SCALING" \
            --frame-averaging-mode "$frame_averaging_mode" \
            --output-dir "$opt_output_dir" \
            > "${opt_output_dir}/logs/${ENSEMBLE}_${LOSS}_maxent${MAXENT}_split${SPLIT}.log" 2>&1 &
          batch_pids+=("$!")
        done
        for batch_pid in "${batch_pids[@]}"; do
          wait "$batch_pid"
        done
        echo "Completed $ENSEMBLE-$LOSS with $SPLIT using $frame_averaging_mode"
      done
    done
  done
  wait
  echo "All $frame_averaging_mode optimisation tasks completed."
  echo "Starting $frame_averaging_mode analysis scripts..."

  echo "Running recovery analysis..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/recovery_analysis_ISO_TRI_BI_precluster.py" \
    --results-dir "$opt_output_dir" \
    > "${opt_output_dir}/logs/recovery_analysis.log" 2>&1
  echo "Running weights validation..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/weights_validation_ISO_TRI_BI_precluster.py" \
    --results-dir "$opt_output_dir" \
    > "${opt_output_dir}/logs/weights_validation.log" 2>&1
  echo "Running CV validation..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/CV_validation_ISO_TRI_BI_precluster.py" \
    --results-dir "$opt_output_dir" \
    > "${opt_output_dir}/logs/CV_validation.log" 2>&1
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/analyse_loss_ISO_TRI_BI.py" \
    --results-dir "$opt_output_dir" \
    > "${opt_output_dir}/logs/Analyse_Loss.log" 2>&1

  echo "Processing optimization results..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/process_optimisation_results.py" \
    --results-dir "$opt_output_dir" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "${DIR_WD}/_featurise" \
    --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
    > "${opt_output_dir}/logs/process_optimisation_results.log" 2>&1

  basename=$(basename "$opt_output_dir")
  processed_dir="${DIR_WD}/_processed_${basename}"

  echo "Scoring models..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/score_models_ISO_TRI_BI.py" \
    --processed-data-dir "$processed_dir" \
    --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "${DIR_WD}/_featurise" \
    --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
    > "${opt_output_dir}/logs/score_models.log" 2>&1

  scores_basename=$(basename "$processed_dir")
  scores_dir="${processed_dir}/_scores_${scores_basename}"

  echo "Analyzing scores with mixed linear model..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/analyse_scores_mixed_linear_model.py" \
    --scores-csv-path "${scores_dir}/model_scores.csv" \
    --target-metric "recovery_percent" \
    --filter-mode "both" \
    --analyze-subsets \
    > "${opt_output_dir}/logs/analyse_scores_mixed_linear_model.log" 2>&1

  analysis_dir="${processed_dir}/_analysis__scores_${scores_basename}"

  echo "Plotting selected models (unfiltered)..."
  cluster_pop_csv="${ana_output_dir}/conformational_recovery_maxent_data.csv"
  if [ -f "$cluster_pop_csv" ]; then
    plot_extra_args+=(--cluster-populations-csv "$cluster_pop_csv")
  fi
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/plot_selected_models_ISO_TRI_BI.py" \
    --before-csv "${analysis_dir}/whole_dataset/model_selection_performance_summary.csv" \
    --after-csv "${analysis_dir}_filtered/whole_dataset/model_selection_performance_summary.csv" \
    --output-dir "${analysis_dir}/plots_selection" \
    "${plot_extra_args[@]}" \
    > "${opt_output_dir}/logs/plot_selected_models.log" 2>&1

  echo "Extracting selected models..."
  "${PYTHON_RUNNER[@]}" "${ANA_DIR}/extract_selected_models.py" \
    --processed-data-dir "$processed_dir" \
    --scores-csv "${scores_dir}/model_scores.csv" \
    --selection-csv "${analysis_dir}/whole_dataset/model_selection_performance_summary.csv" \
    > "${opt_output_dir}/logs/extract_selected_models.log" 2>&1

  echo "All $frame_averaging_mode analysis tasks completed."
  echo "Results are saved in $opt_output_dir"
}

for FRAME_AVERAGING_MODE in "${FRAME_AVERAGING_MODES[@]}"; do
  run_campaign "$FRAME_AVERAGING_MODE"
done

echo "All frame averaging campaigns completed."
echo "Result directories:"
for RESULT_DIR in "${RESULT_DIRS[@]}"; do
  echo "  $RESULT_DIR"
done
echo "Script finished."
