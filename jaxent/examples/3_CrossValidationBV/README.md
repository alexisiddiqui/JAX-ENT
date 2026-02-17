# Example 3: Cross-Validation with BV Regularization (MoPrP)

## Overview

The **CrossValidation with BV Regularization** example extends the methodology of Example 2 (pure MaxEnt reweighting) by introducing **physical regularization** on the forward model parameters.

In Example 2, the Best-Vendruscolo (BV) model parameters ($$\beta_c, \beta_I$$) were fixed or treated as simple scalars. In this example, we explore a **joint optimization** landscape where we simultaneously:
1.  **Reweight the ensemble** (MaxEnt on frame weights) to match experimental data.
2.  **Regularize the physical model** (L1/L2 penalty on BV parameters) to ensure physical realism and prevent overfitting.

This results in a **2D hyperparameter sweep** over:
-   **MaxEnt Scaling** (controlling the deviation from the prior ensemble).
-   **BV Regularization Scaling** (controlling the constraint on physical parameters).

### Scientific Motivation

While MaxEnt reweighting is powerful, it can sometimes "force" a fit by aggressively reweighting frames, even if the underlying physical model (the mapping from structure to observable) is imperfect. By regularizing the physical model parameters, we aim to find a balance where:
-   The ensemble reweighting explains conformational heterogeneity.
-   The physical model remains robust and generalizable.

This example uses the same **Mouse Prion Protein (MoPrP)** system and dataset as Example 2.

---

## Prerequisites

This example **depends on data generated in Example 2**. You must have successfully run the data preparation steps in `2_CrossValidation/`.

### Data Setup

Copy the necessary feature and data split files from Example 2:

```bash
# From within jaxent/examples/3_CrossValidationBV/
cp -R ../2_CrossValidation/fitting/jaxENT/_datasplits ./fitting/jaxENT/
cp -R ../2_CrossValidation/fitting/jaxENT/_featurise ./fitting/jaxENT/
cp -R ../2_CrossValidation/data/_MoPrP_covariance_matrices ./data/
cp ../2_CrossValidation/analysis/state_ratios.json ./analysis/
```

> [!IMPORTANT]
> Ensure that `2_CrossValidation` has been run first to generate these files.

---

## Directory Structure

```
3_CrossValidationBV/
├── README.md                              # This file
├── INSTRUCTIONS.md                        # Quick-start instructions
│
├── data/                                  # Data directory (populated from Ex 2)
│   └── _MoPrP_covariance_matrices/        # Covariance matrices
│
├── fitting/jaxENT/                        # Optimisation scripts
│   ├── _featurise/                        # (Copied) BV features
│   ├── _datasplits/                       # (Copied) Train/val splits
│   ├── optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py  # 2D sweep optimiser
│   ├── optimise_fn.py                     # Optimisation helper functions
│   ├── run_maxent_parallel_BV_objective.sh     # Production parallel runner
│   └── run_maxent_parallel_BV_objective_test.sh # Test/quick runner
│
└── analysis/                              # Post-optimisation analysis
    ├── recovery_analysis_ISO_TRI_BI_2D_BV.py   # 2D recovery analysis
    ├── weights_validation_ISO_TRI_BI_2D_BV.py  # 2D weights analysis
    ├── analyse_loss_ISO_TRI_BI_2D_BV.py        # 2D loss analysis
    ├── process_optimisation_results.py         # Standard result processing
    ├── score_models_ISO_TRI_BI.py              # Model scoring
    ├── analyse_scores_mixed_linear_model.py    # Statistical model selection
    ├── plot_selected_models_ISO_TRI_BI.py      # Final plotting
    └── cross_experiment_mixed_effects_model.py # Cross-experiment comparison
```

---

## Step-by-Step Workflow

### Step 1: 2D Hyperparameter Sweep

The core of this example is the 2D sweep over MaxEnt scaling and BV regularization scaling.

**Script**: [run_maxent_parallel_BV_objective.sh](fitting/jaxENT/run_maxent_parallel_BV_objective.sh)

```bash
cd fitting/jaxENT/
bash run_maxent_parallel_BV_objective.sh
```

This script runs `optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py` in parallel across combinations of:
-   **Ensembles**: `AF2_filtered`, `AF2_MSAss`
-   **Loss Functions**: `mcMSE`, `MSE`, `Sigma_MSE`
-   **MaxEnt Scaling**: Range (e.g., 1 to 1000)
-   **BV Reg Scaling**: Range (e.g., 0.0 to 1.0)
-   **BV Reg Type**: `L1` or `L2`

**Output**: HDF5 files in `fitting/jaxENT/_optimise_<timestamp>/`.

### Step 2: Analysis Pipeline

The analysis pipeline visualizes the performance across this 2D landscape.

#### 2a. Recovery Analysis (2D)

**Script**: [recovery_analysis_ISO_TRI_BI_2D_BV.py](analysis/recovery_analysis_ISO_TRI_BI_2D_BV.py)

```bash
python analysis/recovery_analysis_ISO_TRI_BI_2D_BV.py \
    --results-dir fitting/jaxENT/_results_dir/ \
    --clustering-dir ../2_CrossValidation/analysis/
```

Produces 2D heatmaps showing conformational state recovery percentages for each (MaxEnt, BV Reg) combination.

#### 2b. Weights Validation (2D)

**Script**: [weights_validation_ISO_TRI_BI_2D_BV.py](analysis/weights_validation_ISO_TRI_BI_2D_BV.py)

Analyzes the effective sample size (ESS) and KL divergence of the optimized weights across the 2D grid.

#### 2d. Standard Model Validation

After exploring the 2D landscape, the pipeline proceeds with the standard model selection and scoring workflow identical to Example 2:

1.  **Process Results**: `process_optimisation_results.py` extracts predictions and metrics.
2.  **Score Models**: `score_models_ISO_TRI_BI.py` computes detailed scoring matrices (MSE, dMSE, Work).
3.  **Statistical Analysis**: `analyse_scores_mixed_linear_model.py` uses mixed effects models to determine the most robust modeling strategy.

---

## Key Differences from Example 2

| Feature | Example 2 (CrossValidation) | Example 3 (CrossValidationBV) |
| :--- | :--- | :--- |
| **Optimisation** | 1D Sweep (MaxEnt Scaling only) | **2D Sweep** (MaxEnt + BV Reg Scaling) |
| **Physical Model** | Fixed / Scalar parameters | **Regularized** (L1/L2 penalty) |
| **Analysis** | Standard recovery & loss curves | **2D Heatmaps** & Slice plots |
| **Hypothesis** | Ensemble reweighting effectively recovers populations | Joint reweighting + physical regularization yields more robust models |

---

## Configuration Reference

### Optimisation Parameters

| Parameter | Description |
| :--- | :--- |
| `--maxent-range` | Range of MaxEnt scaling values (e.g., "1,100") |
| `--bvreg-range` | Range of BV regularization scaling values (e.g., "0.0,1.0") |
| `--bv-reg-function` | Regularization type: `L1` (Lasso) or `L2` (Ridge) |
| `--n-steps` | Optimisation steps (default: 500-10000 depending on config) |
| `--learning-rate` | Adam learning rate (default: 0.1 - 1.0) |

### CLI Usage Example

```bash
python optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py \
    --ensemble AF2_MSAss \
    --loss-function mcMSE \
    --maxent-range 1,10 \
    --bvreg-range 0.1,0.5 \
    --bv-reg-function L2
```

---

## References

-   **Best & Vendruscolo (2006)**: Origin of the structural forward model.
-   **JAX-ENT Manuscript**: See `output_omc.pdf` for discussion on the impact of physical regularization on HDX ensemble fitting.
