# JAX-ENT Example 3: MoPrP Cross-Validation with BV Regularization

This example implements a cross-validation workflow for modeling MoPrP conformational state recovery using both ensemble reweighting (MaxEnt) and physical regularization (Best-Vendruscolo model).

This workflow strictly follows the process outlined in `alpha-commands.sh`, leveraging structural features and data splits generated in [Example 2 (Cross-Validation)](../2_CrossValidation/).

## 1. Data Setup

This example relies on data from Example 2. You can set it up by copying the necessary directories:

```bash
# Copy datasplits, features, covariance matrices and state ratios from Example 2
cp -R jaxent/examples/2_CrossValidation/fitting/jaxENT/_datasplits jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_datasplits
cp -R jaxent/examples/2_CrossValidation/fitting/jaxENT/_featurise jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_featurise
mkdir -p jaxent/examples/3_CrossValidationBV/data/
cp -R jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices jaxent/examples/3_CrossValidationBV/data/_MoPrP_covariance_matrices
cp jaxent/examples/2_CrossValidation/analysis/state_ratios.json jaxent/examples/3_CrossValidationBV/analysis/state_ratios.json
```

## 2. Optimization & Automated Analysis

Run the 2D hyperparameter sweep (MaxEnt scaling vs. BV regularization scale). The provided test script iterates through a range of values and automatically triggers a comprehensive analysis pipeline upon completion.

```bash
# Run the parallel test script
bash jaxent/examples/3_CrossValidationBV/fitting/jaxENT/run_maxent_parallel_BV_objective_test.sh
```

## Post-Optimization Analysis Pipeline

The `run_maxent_parallel_BV_objective_test.sh` script executes the following analysis sequence (adapted for 2D regularisation grids) to validate the ensemble optimization:

1.  **2D Recovery Analysis**: Evaluates conformational state recovery across the 2D hyperparameter grid (`recovery_analysis_ISO_TRI_BI_2D_BV.py`).
2.  **2D Weights Validation**: Examines weight distributions across the 2D grid (`weights_validation_ISO_TRI_2D_BV.py`).
3.  **2D Loss Analysis**: Summarizes convergence and cross-validation performance across the grid and splits (`analyse_loss_ISO_TRI_BI_2D_BV.py`).
4.  **Result Processing**: Aggregates optimization outputs and aligns them with clustering data for downstream scoring (`process_optimisation_results.py`).
5.  **Model Scoring**: Calculates performance metrics (e.g., Pearson R, recovery percent) for every optimized ensemble (`score_models_ISO_TRI_BI.py`).
6.  **Mixed Linear Model (MLM) Analysis**: Statistically determines the impact of different factors (Ensemble, Loss, MaxEnt, BV-Reg) on model performance (`analyse_scores_mixed_linear_model.py`).
7.  **Model Selection Visualization**: Generates summary plots for the model selection process (`plot_selected_models_ISO_TRI_BI.py`).

<!-- NMR/HDX data obtained from: https://doi.org/10.1074/jbc.M115.677575 -->