#!/bin/bash
# Runs optimise_aSyn_conditions_BV.py in parallel across conditions,
# BV reg values, maxent values, and split types.

# Set working directory to the script's location
cd "$(dirname "$0")" || exit
DIR_WD=$(pwd)
echo "Working directory: $DIR_WD"

# --- Defaults (can be overridden via CLI) ---
PARALLEL_JOBS=4
# DEFAULT_MAXENT_VALUES_STR="1,5,10,50,100,500,1000"
DEFAULT_MAXENT_VALUES_STR="1,10,100,1000,10000,100000,1000000"
MAXENT_VALUES_STR="$DEFAULT_MAXENT_VALUES_STR"
BV_REG_VALUES_STR="0.5,0.75,1.0"
# BV_REG_VALUES_STR="0.5,1.0"
BV_REG_LOSSES_STR="L1"
RUN_ANALYSIS=1

DIR_NAME="_optimise_aSyn_BV_5000"
N_STEPS=5000
INITIAL_STEPS=0
INITIAL_LR=1.0
LEARNING_RATE=1.0
EMA_ALPHA=0.5
FORWARD_MODEL_SCALING=1000.0
MODEL_PARAMETERS_LR_SCALE=1.0

DEFAULT_CONDITIONS_STR="Tris_only,Extracellular,Intracellular,Lysosomal"

CONDITIONS_STR="$DEFAULT_CONDITIONS_STR"
LOSSES_STR="MSE"
DEFAULT_SPLIT_TYPES_STR="sequence_cluster"
SPLIT_TYPES_STR="$DEFAULT_SPLIT_TYPES_STR"
# --- end defaults ---

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
    --bvreg-values)
      BV_REG_VALUES_STR="$2"; shift 2;;
    --bvreg-values=*)
      BV_REG_VALUES_STR="${1#*=}"; shift;;
    --bv-reg-losses)
      BV_REG_LOSSES_STR="$2"; shift 2;;
    --bv-reg-losses=*)
      BV_REG_LOSSES_STR="${1#*=}"; shift;;
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
    --model-parameters-lr-scale)
      MODEL_PARAMETERS_LR_SCALE="$2"; shift 2;;
    --model-parameters-lr-scale=*)
      MODEL_PARAMETERS_LR_SCALE="${1#*=}"; shift;;
    --conditions)
      CONDITIONS_STR="$2"; shift 2;;
    --conditions=*)
      CONDITIONS_STR="${1#*=}"; shift;;
    --split-types)
      SPLIT_TYPES_STR="$2"; shift 2;;
    --split-types=*)
      SPLIT_TYPES_STR="${1#*=}"; shift;;
    --skip-analysis)
      RUN_ANALYSIS=0; shift;;
    -h|--help)
      echo "Usage: $0 [--conditions a,b] [--split-types s,t] [--maxent-values a,b,c] [--bvreg-values a,b] [--bv-reg-losses L1,L2] [--dir-name name] [--n-steps N] [--initial-steps M] [--initial-learning-rate X] [--learning-rate Y] [--ema-alpha Z] [--forward-model-scaling S] [--model-parameters-lr-scale P] [--skip-analysis] [-j|--jobs N]"
      exit 0;;
    *)
      break;;
  esac
done

echo "Parallel jobs limit:  $PARALLEL_JOBS"
echo "Maxent values:        $MAXENT_VALUES_STR"
echo "BV reg values:        $BV_REG_VALUES_STR"
echo "BV reg losses:        $BV_REG_LOSSES_STR"
echo "DIR_NAME:             $DIR_NAME"
echo "n-steps:              $N_STEPS"
echo "initial-steps:        $INITIAL_STEPS"
echo "initial-learning-rate:$INITIAL_LR"
echo "learning-rate:        $LEARNING_RATE"
echo "ema-alpha:            $EMA_ALPHA"
echo "forward-model-scaling:$FORWARD_MODEL_SCALING"
echo "model-params-lr-scale:$MODEL_PARAMETERS_LR_SCALE"
echo "Conditions:           $CONDITIONS_STR"
echo "Losses:               $LOSSES_STR"
echo "Split types:          $SPLIT_TYPES_STR"
echo "Run analysis:         $RUN_ANALYSIS"

# Convert comma-separated strings into arrays
IFS=',' read -r -a MAXENT_VALUES <<< "$MAXENT_VALUES_STR"
IFS=',' read -r -a BV_REG_VALUES <<< "$BV_REG_VALUES_STR"
IFS=',' read -r -a BV_REG_LOSSES <<< "$BV_REG_LOSSES_STR"
IFS=',' read -r -a CONDITIONS <<< "$CONDITIONS_STR"
IFS=',' read -r -a SPLIT_TYPES <<< "$SPLIT_TYPES_STR"

# Parallel job helpers
running_jobs_count() {
  jobs -rp | wc -l
}

wait_for_slot() {
  while [ "$(running_jobs_count)" -ge "$PARALLEL_JOBS" ]; do
    sleep 1
  done
}

cleanup() {
  wait
}
trap cleanup EXIT

rm -rf logs
mkdir -p logs

time_data="_$(date +'%Y%m%d_%H%M%S')"
OUTPUT_DIR="${DIR_NAME}${time_data}"
OPT_OUTPUT_DIR="${DIR_WD}/${OUTPUT_DIR}"
echo "Output directory: $OPT_OUTPUT_DIR"
mkdir -p "${OPT_OUTPUT_DIR}/logs"

for CONDITION in "${CONDITIONS[@]}"; do
  echo "Running condition: $CONDITION"
  for SPLIT in "${SPLIT_TYPES[@]}"; do
    echo "  Split type: $SPLIT"
    for MAXENT in "${MAXENT_VALUES[@]}"; do
      for BV_REG in "${BV_REG_VALUES[@]}"; do
        for BV_REG_LOSS in "${BV_REG_LOSSES[@]}"; do
          echo "    Maxent: $MAXENT, BV reg: $BV_REG ($BV_REG_LOSS)"
          wait_for_slot
          python optimise_aSyn_conditions_BV.py \
            --condition "$CONDITION" \
            --loss-function "MSE" \
            --maxent-range "$MAXENT,$MAXENT" \
            --bvreg-range "$BV_REG,$BV_REG" \
            --bv-reg-function "$BV_REG_LOSS" \
            --split-types "$SPLIT" \
            --n-steps "$N_STEPS" \
            --initial-steps "$INITIAL_STEPS" \
            --initial-learning-rate "$INITIAL_LR" \
            --learning-rate "$LEARNING_RATE" \
            --ema-alpha "$EMA_ALPHA" \
            --forward-model-scaling "$FORWARD_MODEL_SCALING" \
            --model-parameters-lr-scale "$MODEL_PARAMETERS_LR_SCALE" \
            --output-dir "$OPT_OUTPUT_DIR" \
            > "${OPT_OUTPUT_DIR}/logs/${CONDITION}_maxent${MAXENT}_bvreg${BV_REG}_${BV_REG_LOSS}_split${SPLIT}.log" 2>&1 &
        done
      done
    done
    wait  # Wait for all jobs for this SPLIT to finish before moving to next
    echo "  Completed $CONDITION with $SPLIT"
  done
done

wait  # Final wait for any remaining jobs
echo "All optimisation tasks completed."
echo "Results saved in: $OPT_OUTPUT_DIR"

if [ "$RUN_ANALYSIS" -eq 1 ]; then
  ANALYSIS_RUNNER="${DIR_WD}/../run_comprehensive_analysis.sh"
  ANALYSIS_LOG="${OPT_OUTPUT_DIR}/logs/comprehensive_analysis_launcher.log"
  if [ -f "$ANALYSIS_RUNNER" ]; then
    echo "Starting comprehensive analysis pipeline..."
    echo "Runner: $ANALYSIS_RUNNER"
    echo "Log:    $ANALYSIS_LOG"
    bash "$ANALYSIS_RUNNER" "$OPT_OUTPUT_DIR" | tee "$ANALYSIS_LOG"
    ANALYSIS_EXIT=${PIPESTATUS[0]}
    if [ "$ANALYSIS_EXIT" -eq 0 ]; then
      echo "Comprehensive analysis runner finished."
    else
      echo "Comprehensive analysis runner exited with code $ANALYSIS_EXIT"
    fi
  else
    echo "Skipping analysis: runner not found at $ANALYSIS_RUNNER"
  fi
fi
