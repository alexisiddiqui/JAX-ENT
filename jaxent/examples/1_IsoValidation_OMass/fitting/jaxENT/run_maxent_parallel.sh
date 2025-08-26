#!/bin/bash
# set working directory to the script's location
cd "$(dirname "$0")" || exit


rm -rf logs
mkdir -p logs
ENSEMBLES=("ISO_TRI" "ISO_BI")


LOSSES=("mcMSE" "MSE")
LOSSES=("mcMSE" )

MAXENT_VALUES=(1000 1000000 1000000000 1000000000000 1000000000000000)
MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000)

# MAXENT_VALUES=(1 2 5 10 50 100 500 1000 10000)
MAXENT_VALUES=(100000 1000000 10000000 100000000 1000000000)

for ENSEMBLE in "${ENSEMBLES[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    echo "Running $ENSEMBLE-$LOSS in parallel for maxent log-scaled values"
    for MAXENT in "${MAXENT_VALUES[@]}"; do
      python optimise_ISO_TRI_BI_splits_maxENT.py \
        --ensemble "$ENSEMBLE" \
        --loss-function "$LOSS" \
        --maxent-range "$MAXENT,$MAXENT" \
        --split-types random \
       > "logs/${ENSEMBLE}_${LOSS}_maxent${MAXENT}.log" 2>&1 &
    done
    wait
    echo "Completed $ENSEMBLE-$LOSS"
  done
done
