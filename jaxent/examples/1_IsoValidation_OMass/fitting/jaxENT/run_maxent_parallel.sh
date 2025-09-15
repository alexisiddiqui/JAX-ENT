#!/bin/bash
# set working directory to the script's location
cd "$(dirname "$0")" || exit


rm -rf logs
mkdir -p logs
ENSEMBLES=("ISO_TRI" "ISO_BI")
# ENSEMBLES=("ISO_TRI")
# 

SPLIT_TYPES=("random" "sequence" "sequence_cluster" "stratified" "spatial")
SPLIT_TYPES=("random" "sequence" "sequence_cluster" "spatial")
LOSSES=("mcMSE" "MSE")
# LOSSES=("MSE" )

MAXENT_VALUES=(1000 1000000 1000000000 1000000000000 1000000000000000)
MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000)
MAXENT_VALUES=(1 10 100  1000 10000)
# MAXENT_VALUES=(1 2 5 7 9 10)



# MAXENT_VALUES=(100000 1000000 10000000 100000000 1000000000)
# MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000 1000000 1000000000)
time_data="_$(date +'%Y%m%d_%H%M%S')"

OUTPUT_DIR="_optimise_quick_test_splits_${time_data}"
echo "Output directory: $OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/logs"
for ENSEMBLE in "${ENSEMBLES[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    echo "Running $ENSEMBLE-$LOSS in parallel for maxent log-scaled values"
    for SPLIT in "${SPLIT_TYPES[@]}"; do
      echo "  Split type: $SPLIT"
      for MAXENT in "${MAXENT_VALUES[@]}"; do
        echo "    Maxent: $MAXENT"
        python optimise_ISO_TRI_BI_splits_maxENT.py \
          --ensemble "$ENSEMBLE" \
          --loss-function "$LOSS" \
          --maxent-range "$MAXENT,$MAXENT" \
          --split-types "$SPLIT" \
          --n-steps 500 \
          --initial-steps 2 \
          --initial-learning-rate 1.0 \
          --learning-rate 1.0 \
          --ema-alpha 0.5 \
          --forward-model-scaling 100.0 \
          --output-dir "$OUTPUT_DIR" \
          > "${OUTPUT_DIR}/logs/${ENSEMBLE}_${LOSS}_maxent${MAXENT}_split${SPLIT}.log" 2>&1 &
      done
      wait  # Wait for all background jobs for this SPLIT to finish
      echo "Completed $ENSEMBLE-$LOSS with $SPLIT"
    done
  done
done

