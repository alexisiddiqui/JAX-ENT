


# Setup data

# extract _MoPrP.tar

tar -xvf jaxent/examples/2_CrossValidation/data/_MoPrP.tar -C jaxent/examples/2_CrossValidation/data/

# extract _ValDX.tar
python jaxent/examples/2_CrossValidation/data/extract_data_ValDX.py

python jaxent/examples/2_CrossValidation/analysis/calculate_state_ratios.py

# get reference PDBs for NMR structures

bash jaxent/examples/2_CrossValidation/data/get_NMR_PDBs.sh

python jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39.pdb \
    --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1


python jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H.pdb \
    --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1

# to create clustering map

# Featurise Ensure you have: 
# jaxent/examples/2_CrossValidation/data/_cluster_MoPrP
# jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered


# requires 
# jaxent/examples/2_CrossValidation/analysis/MoPrP_rules_spec.json
# jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json

# featurise the ensemble for clustering
python jaxent/examples/2_CrossValidation/analysis/analyse_LocalFeatures_PUFs.py \
    --ensembles "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --output_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --save_pdbs \
    --json_feature_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --reference_pdb jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb 

# apply clustering rules
python jaxent/examples/2_CrossValidation/analysis/cluster_LocalFeatures_PUF.py \
    --ensembles "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb,jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb" "jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb,jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb" \
    --names "AF2-Filtered" "AF2-MSAss" "NMR-20C" "NMR-37C" \
    --json_data_path jaxent/examples/2_CrossValidation/data/_MoPrP/key_residues.json \
    --input_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2 \
    --output_dir jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test \
    --save_pdbs \
    --json_feature_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_unfolding_spec.json \
    --json_rules_spec jaxent/examples/2_CrossValidation/analysis/MoPrP_rules_spec.json


python jaxent/examples/2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py


python jaxent/examples/2_CrossValidation/fitting/jaxENT/splitdata_CrossVal.py

python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
    --dfrac_file jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_dfrac.dat \
    --segs_file  jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_segments.txt \
    --output_dir jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices/

bash jaxent/examples/2_CrossValidation/fitting/jaxENT/run_maxent_parallel_test.sh

