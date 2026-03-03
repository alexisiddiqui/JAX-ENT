# JAX-ENT Example 3: MoPrP Cross-Validation with BV Regularization

This example implements a cross-validation workflow for modeling MoPrP conformational state recovery using both ensemble reweighting (MaxEnt) and physical regularization (Best-Vendruscolo model).

## 1. Data Setup

This example relies on structural features and data splits generated in [Example 2 (Cross-Validation)](../2_CrossValidation/). If you haven't run Example 2, you must copy the necessary directories:

```bash
# From within the 3_CrossValidationBV/ directory:
cp -R ../2_CrossValidation/fitting/jaxENT/_datasplits ./fitting/jaxENT/
cp -R ../2_CrossValidation/fitting/jaxENT/_featurise ./fitting/jaxENT/
cp -R ../2_CrossValidation/data/_MoPrP_covariance_matrices ./data/
cp ../2_CrossValidation/analysis/state_ratios.json ./analysis/
```

## 2. Model Fitting (Optimization)

Run the 2D hyperparameter sweep (MaxEnt scaling vs. BV regularization scale). This can take significant time and is typically run on a cluster.

### Parallel Execution (Cluster/Multi-core)
The provided bash scripts demonstrate how to run the sweep:
```bash
# Standard 2D sweep
bash run_maxent_parallel_BV_objective.sh

# Testing/Quick sweep
bash run_maxent_parallel_BV_objective_test.sh
```

### Manual Execution
To run a single optimization combination:
```bash
cd fitting/jaxENT/
python optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py \
    --ensemble AF2_MSAss \
    --loss-function mcMSE \
    --split-types sequence \
    --maxent-range 1,10 \
    --bvreg-range 0.0,1.0
```

## 3. Analysis Pipeline

Once optimization is complete, follow these steps to analyze the results:

### A. Preliminary Recovery Analysis
Analyze conformational state recovery across the 2D grid:
```bash
python analysis/recovery_analysis_ISO_TRI_BI_2D_BV.py \
    --results-dir fitting/jaxENT/_results_dir/ \
    --clustering-dir ../2_CrossValidation/analysis/_MoPrP_analysis_clusters/clusters/ \
    --state-ratios-json analysis/state_ratios.json
```

### B. Process Results
Extract residue-wise predictions and Divergence metrics from HDF5 files:
```bash
python analysis/process_optimisation_results.py \
    --results-dir fitting/jaxENT/_results_dir/ \
    --datasplit-dir fitting/jaxENT/_datasplits/ \
    --features-dir fitting/jaxENT/_featurise/
```

### C. Model Scoring
Compute MSE and Thermodynamic Work metrics for all converged models:
```bash
python analysis/score_models_ISO_TRI_BI.py \
    --processed-data-dir analysis/_processed_results/ \
    --datasplit-dir fitting/jaxENT/_datasplits/
```

### D. Statistical Modeling
Perform mixed-effects modeling to assessment metric robustness:
```bash
python analysis/analyse_scores_mixed_linear_model.py \
    --scores-csv-path analysis/_scores/model_scores.csv
```

### E. Visualization
Generate publication-quality plots comparing ensemble performances:
```bash
python analysis/plot_selected_models_ISO_TRI_BI.py \
    --before-csv analysis/_mixed_models/model_selection_performance_summary.csv \
    --after-csv analysis/_mixed_models_filtered/model_selection_performance_summary.csv
```

<!-- NMR/HDX data obtained from: https://doi.org/10.1074/jbc.M115.677575 -->