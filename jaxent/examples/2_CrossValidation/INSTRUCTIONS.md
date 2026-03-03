# Cross-Validation Example (MoPrP)

This workflow strictly follows the process outlined in `alpha-commands.sh`, providing a robust sequence for data setup, feature analysis, and ensemble optimization with cross-validation.

## Workflow Steps

### 1. Data Setup
Automatically download and unpack the required data package including PDBs, trajectories, and experimental data.
```bash
python jaxent/examples/2_CrossValidation/unpack_cross_validation_data.py
```

### 2. Local Feature Analysis & Clustering
Analyze local features (PUFs) across multiple ensembles (AF2, NMR) and apply clustering rules to identify conformational states.

```bash
# 1. Analyse Local Features (Generates feature specs)
python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb

# 2. Apply Clustering Rules
python jaxent/examples/2_CrossValidation/analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --input_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --output_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test \
    --save_pdbs \
    --json_feature_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --json_rules_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_rules_spec.json

# 3. Auxiliary Analysis
python jaxent/examples/2_CrossValidation/analysis/calculate_state_ratios.py
python jaxent/examples/2_CrossValidation/analysis/compute_recovery%_PUF.py
```

### 3. Fitting Preparation
Prepare the ensembles for optimization by generating features, splitting data for cross-validation, and computing covariance matrices.

```bash
# 1. Featurise the target ensembles
python jaxent/examples/2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py

# 2. Split data for Cross-Validation
python jaxent/examples/2_CrossValidation/fitting/jaxENT/splitdata_CrossVal.py

# 3. Analyze data splits
python jaxent/examples/2_CrossValidation/analysis/analyse_split_CrossVal.py

# 4. Compute Sigma (Covariance Matrices)
python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
    --dfrac_file jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_dfrac.dat \
    --segs_file  jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_segments.txt \
    --output_dir jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices/
```

### 4. Optimization & Automated Analysis
Run the MaxEnt optimization in parallel across different hyperparameter ranges. This script automatically triggers a comprehensive analysis pipeline upon completion.

```bash
bash jaxent/examples/2_CrossValidation/fitting/jaxENT/run_maxent_parallel_test.sh
```

## Post-Optimization Analysis Pipeline

The `run_maxent_parallel_test.sh` script executes the following analysis sequence to validate the ensemble optimization:

1.  **Recovery Analysis**: Evaluates how well the optimization recovers specific conformational populations (`recovery_analysis_ISO_TRI_BI_precluster.py`).
2.  **Weights Validation**: Examines the distribution of frame weights across different regularization strengths (`weights_validation_ISO_TRI_BI_precluster.py`).
3.  **Loss Analysis**: Summarizes convergence behavior and cross-validation performance across splits (`analyse_loss_ISO_TRI_BI.py`).
4.  **Result Processing**: Aggregates optimization outputs and aligns them with clustering data for downstream scoring (`process_optimisation_results.py`).
5.  **Model Scoring**: Calculates performance metrics (e.g., Pearson R, recovery percent) for every optimized ensemble (`score_models_ISO_TRI_BI.py`).
6.  **Mixed Linear Model (MLM) Analysis**: Statistically determines the impact of different factors (Ensemble type, Loss function, Regularization) on model performance (`analyse_scores_mixed_linear_model.py`).
7.  **Model Selection Visualization**: Generates summary plots for the model selection process (`plot_selected_models_ISO_TRI_BI.py`).