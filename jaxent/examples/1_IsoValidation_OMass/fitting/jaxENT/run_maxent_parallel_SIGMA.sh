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
echo "Working directory: $DIR_WD"

# --- Changed: add configurable defaults and extended argument parsing ---
# Defaults (can be overridden via CLI)
PARALLEL_JOBS=10
DEFAULT_MAXENT_VALUES_STR="1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000"


MAXENT_VALUES_STR="$DEFAULT_MAXENT_VALUES_STR"
DIR_NAME="_optimise_FIGURE_SIGMA_5000"
N_STEPS=5000
LEARNING_RATE=1.0
LR_ADJUSTMENT=on
FRAME_AVERAGE_IMPL=tensordot
STEP_CHUNK_SIZE=100
EMA_ALPHA=0.5
FORWARD_MODEL_SCALING=1000.0

# --- Added defaults for ensembles, losses and split types ---
DEFAULT_ENSEMBLES_STR="ISO_TRI,ISO_BI"
ENSEMBLES_STR="$DEFAULT_ENSEMBLES_STR"
DEFAULT_LOSSES_STR="MSE,Sigma_MSE"

LOSSES_STR="$DEFAULT_LOSSES_STR"
DEFAULT_FRAME_AVERAGING_MODES_STR="log_pf"
FRAME_AVERAGING_MODES_STR="$DEFAULT_FRAME_AVERAGING_MODES_STR"
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
    --learning-rate)
      LEARNING_RATE="$2"; shift 2;;
    --learning-rate=*)
      LEARNING_RATE="${1#*=}"; shift;;
    --lr-adjustment)
      LR_ADJUSTMENT="$2"; shift 2;;
    --lr-adjustment=*)
      LR_ADJUSTMENT="${1#*=}"; shift;;
    --frame-average-impl)
      FRAME_AVERAGE_IMPL="$2"; shift 2;;
    --frame-average-impl=*)
      FRAME_AVERAGE_IMPL="${1#*=}"; shift;;
    --step-chunk-size)
      STEP_CHUNK_SIZE="$2"; shift 2;;
    --step-chunk-size=*)
      STEP_CHUNK_SIZE="${1#*=}"; shift;;
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
      echo "Usage: $0 [--ensembles a,b] [--losses x,y] [--frame-averaging-modes log_pf,rate,uptake,frame_uptake] [--split-types s,t] [--maxent-values a,b,c] [--dir-name name] [--n-steps N] [--learning-rate Y] [--lr-adjustment on|off] [--frame-average-impl tensordot|legacy_sum] [--step-chunk-size N] [-j|--jobs N]"
      exit 0;;
    *)
      break;;
  esac
done

echo "Parallel jobs limit: $PARALLEL_JOBS"
echo "Maxent values (raw): $MAXENT_VALUES_STR"
echo "DIR_NAME: $DIR_NAME"
echo "n-steps: $N_STEPS, learning-rate: $LEARNING_RATE, lr-adjustment: $LR_ADJUSTMENT, frame-average-impl: $FRAME_AVERAGE_IMPL, step-chunk-size: $STEP_CHUNK_SIZE, ema-alpha: $EMA_ALPHA, forward-model-scaling: $FORWARD_MODEL_SCALING"
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
if [[ "$LR_ADJUSTMENT" != "on" && "$LR_ADJUSTMENT" != "off" ]]; then echo "Invalid --lr-adjustment" >&2; exit 2; fi
if [[ "$FRAME_AVERAGE_IMPL" != "tensordot" && "$FRAME_AVERAGE_IMPL" != "legacy_sum" ]]; then echo "Invalid --frame-average-impl" >&2; exit 2; fi
if ! [[ "$STEP_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then echo "--step-chunk-size must be >= 1" >&2; exit 2; fi
for FRAME_AVERAGING_MODE in "${FRAME_AVERAGING_MODES[@]}"; do
  case "$FRAME_AVERAGING_MODE" in
    log_pf|rate|uptake|frame_uptake) ;;
    *) echo "Invalid frame averaging mode: $FRAME_AVERAGING_MODE" >&2; exit 2;;
  esac
done

RUN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RESULT_DIRS=()

run_campaign() {
  local frame_averaging_mode="$1"
  local output_dir="${DIR_NAME}_${frame_averaging_mode}_${RUN_TIMESTAMP}"
  local opt_output_dir="${DIR_WD}/${output_dir}"
  local ana_output_dir="${ANA_DIR}/${output_dir}"
  local basename processed_dir scores_basename scores_dir analysis_dir cluster_pop_csv
  local plot_extra_args=()

  if [[ -e "$opt_output_dir" ]]; then
    echo "Refusing to reuse existing output directory: $opt_output_dir" >&2
    exit 2
  fi
  python -m jaxent.examples.common.manifest \
    --output-dir "$opt_output_dir" --example 1 \
    --ensembles "$ENSEMBLES_STR" --losses "$LOSSES_STR" \
    --split-types "$SPLIT_TYPES_STR" --maxent-values "$MAXENT_VALUES_STR" \
    --learning-rate "$LEARNING_RATE" --lr-adjustment "$LR_ADJUSTMENT" \
    --frame-average-impl "$FRAME_AVERAGE_IMPL" \
    --frame-averaging-mode "$frame_averaging_mode" \
    --step-chunk-size "$STEP_CHUNK_SIZE" --n-steps "$N_STEPS" \
    --jobs "$PARALLEL_JOBS"
  RESULT_DIRS+=("$opt_output_dir")
  mkdir -p "${opt_output_dir}/logs"
  echo "Starting frame averaging campaign: $frame_averaging_mode"
  echo "Output directory: $opt_output_dir"

  for ENSEMBLE in "${ENSEMBLES[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      for SPLIT in "${SPLIT_TYPES[@]}"; do
        local batch_pids=()
        echo "Running $ENSEMBLE-$LOSS, split $SPLIT, mode $frame_averaging_mode"
        for MAXENT in "${MAXENT_VALUES[@]}"; do
          wait_for_slot
          python optimise_ISO_TRI_BI_splits_Sigma.py \
            --ensemble "$ENSEMBLE" --loss-function "$LOSS" \
            --maxent-range "$MAXENT,$MAXENT" --split-types "$SPLIT" \
            --n-steps "$N_STEPS" --learning-rate "$LEARNING_RATE" \
            --lr-adjustment "$LR_ADJUSTMENT" \
            --frame-average-impl "$FRAME_AVERAGE_IMPL" \
            --frame-averaging-mode "$frame_averaging_mode" \
            --step-chunk-size "$STEP_CHUNK_SIZE" --ema-alpha "$EMA_ALPHA" \
            --forward-model-scaling "$FORWARD_MODEL_SCALING" \
            --output-dir "$opt_output_dir" \
            > "${opt_output_dir}/logs/${ENSEMBLE}_${LOSS}_maxent${MAXENT}_split${SPLIT}.log" 2>&1 &
          batch_pids+=("$!")
        done
        for batch_pid in "${batch_pids[@]}"; do
          wait "$batch_pid"
        done
      done
    done
  done

  echo "Running analysis for $frame_averaging_mode..."
  python "${ANA_DIR}/recovery_analysis_ISO_TRI_BI_precluster.py" --results-dir "$opt_output_dir" > "${opt_output_dir}/logs/recovery_analysis.log" 2>&1
  python "${ANA_DIR}/weights_validation_ISO_TRI_BI_precluster.py" --results-dir "$opt_output_dir" > "${opt_output_dir}/logs/weights_validation.log" 2>&1
  python "${ANA_DIR}/CV_validation_ISO_TRI_BI_precluster.py" --results-dir "$opt_output_dir" > "${opt_output_dir}/logs/CV_validation.log" 2>&1
  python "${ANA_DIR}/analyse_loss_ISO_TRI_BI.py" --results-dir "$opt_output_dir" > "${opt_output_dir}/logs/Analyse_Loss.log" 2>&1
  python "${ANA_DIR}/process_optimisation_results.py" \
    --results-dir "$opt_output_dir" --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "${DIR_WD}/_featurise" \
    --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
    --frame-averaging-mode "$frame_averaging_mode" \
    > "${opt_output_dir}/logs/process_optimisation_results.log" 2>&1

  basename=$(basename "$opt_output_dir")
  processed_dir="${DIR_WD}/_processed_${basename}"
  python "${ANA_DIR}/score_models_ISO_TRI_BI.py" \
    --processed-data-dir "$processed_dir" --datasplit-dir "${DIR_WD}/_datasplits" \
    --features-dir "${DIR_WD}/_featurise" \
    --clustering-dir "${DIR_WD}/../../data/_clustering_results" \
    > "${opt_output_dir}/logs/score_models.log" 2>&1

  scores_basename=$(basename "$processed_dir")
  scores_dir="${processed_dir}/_scores_${scores_basename}"
  python "${ANA_DIR}/analyse_scores_mixed_linear_model.py" \
    --scores-csv-path "${scores_dir}/model_scores.csv" \
    --target-metric recovery_percent --filter-mode both --analyze-subsets \
    > "${opt_output_dir}/logs/analyse_scores_mixed_linear_model.log" 2>&1

  analysis_dir="${processed_dir}/_analysis__scores_${scores_basename}"
  cluster_pop_csv="${ana_output_dir}/conformational_recovery_maxent_data.csv"
  if [[ -f "$cluster_pop_csv" ]]; then
    plot_extra_args+=(--cluster-populations-csv "$cluster_pop_csv")
  fi
  python "${ANA_DIR}/plot_selected_models_ISO_TRI_BI.py" \
    --before-csv "${analysis_dir}/whole_dataset/model_selection_performance_summary.csv" \
    --after-csv "${analysis_dir}_filtered/whole_dataset/model_selection_performance_summary.csv" \
    --output-dir "${analysis_dir}/plots_selection" "${plot_extra_args[@]}" \
    > "${opt_output_dir}/logs/plot_selected_models.log" 2>&1
  python "${ANA_DIR}/extract_selected_models.py" \
    --processed-data-dir "$processed_dir" --scores-csv "${scores_dir}/model_scores.csv" \
    --selection-csv "${analysis_dir}/whole_dataset/model_selection_performance_summary.csv" \
    > "${opt_output_dir}/logs/extract_selected_models.log" 2>&1
  echo "Completed frame averaging campaign: $frame_averaging_mode"
}

for FRAME_AVERAGING_MODE in "${FRAME_AVERAGING_MODES[@]}"; do
  run_campaign "$FRAME_AVERAGING_MODE"
done

echo "All frame averaging campaigns completed."
printf '  %s\n' "${RESULT_DIRS[@]}"
