#!/bin/bash
# Validation Analysis for 3_CrossValidationBV

cd "$(dirname "$0")" || exit

ANA_DIR="analysis"
DIR_WD="$(pwd)/fitting/jaxENT"
# Example optimization output directory - CHANGE THIS to match your actual output
OPT_OUTPUT_DIR="${DIR_WD}/_optimise_quick_test_test_SIGMA_500_lr1.0_BV_objectve_scale1.0__20260226_011952"

mkdir -p "${OPT_OUTPUT_DIR}/logs"
echo "Using Optimization Output Directory: $OPT_OUTPUT_DIR"
echo "Starting validation pipeline..."

# Step 1: Recovery analysis
echo "Running recovery analysis..."
python "${ANA_DIR}/recovery_analysis_ISO_TRI_BI_2D_BV.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/recovery_analysis.log" 2>&1

# Step 2: Weights validation
echo "Running weights validation..."
python "${ANA_DIR}/weights_validation_ISO_TRI_2D_BV.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/weights_validation.log" 2>&1

# Step 3: Loss analysis
echo "Running Loss Analysis..."
python "${ANA_DIR}/analyse_loss_ISO_TRI_BI_2D_BV.py" \
  --results-dir "$OPT_OUTPUT_DIR" \
  > "${OPT_OUTPUT_DIR}/logs/Analyse_Loss.log" 2>&1

echo "Validation complete. Logs in: ${OPT_OUTPUT_DIR}/logs/"
