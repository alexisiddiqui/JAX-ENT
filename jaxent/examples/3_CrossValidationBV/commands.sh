



# # Setup data

# # extract _MoPrP.tar

# tar -xvf jaxent/examples/2_CrossValidation/data/_MoPrP.tar -C jaxent/examples/2_CrossValidation/data/

# # extract _ValDX.tar
# python jaxent/examples/2_CrossValidation/data/extract_data_ValDX.py


# # get reference PDBs for NMR structures

# bash jaxent/examples/2_CrossValidation/data/get_NMR_PDBs.sh

# python jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
#     --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39.pdb \
#     --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb \
#     --selection "resid 119-231" \
#     --resi_start 1


# python jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
#     --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H.pdb \
#     --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb \
#     --selection "resid 119-231" \
#     --resi_start 1






# # Featurise Ensure you have: 
# # jaxent/examples/2_CrossValidation/data/_cluster_MoPrP
# # jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered


# python jaxent/examples/2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py


# python jaxent/examples/2_CrossValidation/fitting/jaxENT/splitdata_CrossVal.py

# python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
#     --dfrac_file jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_dfrac.dat \
#     --segs_file  jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_segments.txt \
#     --output_dir jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices/




# Uses data from the previous steps

cp -R jaxent/examples/2_CrossValidation/fitting/jaxENT/_datasplits jaxent/examples/3_CrossValidationBV/fitting/jaxENT/

cp -R jaxent/examples/2_CrossValidation/fitting/jaxENT/_featurise jaxent/examples/3_CrossValidationBV/fitting/jaxENT/


mkdir -p jaxent/examples/3_CrossValidationBV/data/

cp -R jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices jaxent/examples/3_CrossValidationBV/data/_MoPrP_covariance_matrices

cp jaxent/examples/2_CrossValidation/analysis/state_ratios.json jaxent/examples/3_CrossValidationBV/analysis/state_ratios.json

bash jaxent/examples/3_CrossValidationBV/fitting/jaxENT/run_maxent_parallel_BV_objective_test.sh

