# Cross-Validation Example (MoPrP)

This workflow strictly follows the process outlined in `commands.sh`.

## Workflow Steps

### 1. Data Setup & Extraction
Extract the MoPrP data, ValDX data, and calculate state ratios.
```bash
# 1. Extract MoPrP tarball
tar -xvf data/_MoPrP.tar -C data/

# 2. Extract and format ValDX data
python data/extract_data_ValDX.py

# 3. Calculate state ratios from literature
python analysis/calculate_state_ratios.py
```

### 2. PDB Preparation
Get NMR PDBs and renumber them to match the analysis range.
```bash
# 1. Download NMR PDBs
bash data/get_NMR_PDBs.sh

# 2. Renumber 2L39 (residues 119-231 starting at 1)
python data/renumber_pdb.py \
    --input_pdb data/_MoPrP/2L39.pdb \
    --output_pdb data/_MoPrP/2L39_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1

# 3. Renumber 2L1H (residues 119-231 starting at 1)
python data/renumber_pdb.py \
    --input_pdb data/_MoPrP/2L1H.pdb \
    --output_pdb data/_MoPrP/2L1H_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1
```

### 3. Feature Analysis & Clustering
Analyze local features for multiple ensembles and apply clustering.
```bash
# 1. Analyse Local Features (Generates data for clustering)
# Note: Ensure you have the required ensembles in 'data/_cluster_MoPrP' etc.
python analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "data/MoPrP_max_plddt_4334.pdb,data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" \
                "data/MoPrP_max_plddt_4334.pdb,data/_cluster_MoPrP/clusters/all_clusters.xtc" \
                "data/_MoPrP/2L1H_crop.pdb,data/_MoPrP/2L1H_crop.pdb" \
                "data/_MoPrP/2L39_crop.pdb,data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path data/_MoPrP/key_residues.json \
    --output_dir analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec analysis/MoPrP_unfolding_spec.json \
    --reference_pdb data/MoPrP_max_plddt_4334.pdb

# 2. Apply Clustering Rules
python analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "data/MoPrP_max_plddt_4334.pdb,data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" \
                "data/MoPrP_max_plddt_4334.pdb,data/_cluster_MoPrP/clusters/all_clusters.xtc" \
                "data/_MoPrP/2L1H_crop.pdb,data/_MoPrP/2L1H_crop.pdb" \
                "data/_MoPrP/2L39_crop.pdb,data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path data/_MoPrP/key_residues.json \
    --input_dir analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --output_dir analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test \
    --save_pdbs \
    --json_feature_spec analysis/MoPrP_unfolding_spec.json \
    --json_rules_spec analysis/MoPrP_rules_spec.json

# 3. Compute Recovery %
python analysis/compute_recovery%_PUF.py
```

### 4. Model Fitting Preparation
Featurize the specific ensembles for fitting and split the data.
```bash
# 1. Featurise AF2_MSAss and AF2_Filtered
python fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py

# 2. Split Data (Training/Validation)
python fitting/jaxENT/splitdata_CrossVal.py

# 3. Analyze Splits
python analysis/analyse_split_CrossVal.py

# 4. Compute Covariance Matrix (Sigma)
# Note: Uses script from Example 1
python ../1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
    --dfrac_file data/_MoPrP/_output/MoPrP_dfrac.dat \
    --segs_file  data/_MoPrP/_output/MoPrP_segments.txt \
    --output_dir data/_MoPrP_covariance_matrices/
```

### 5. Optimization
Run the MaxEnt optimization.
```bash
# Run the parallel test script
bash fitting/jaxENT/run_maxent_parallel_test.sh
```

### 6. Comparison
Compare results with HDXer (if applicable).
```bash
python analysis/plot_compare_jaxENT_HDXer.py
```