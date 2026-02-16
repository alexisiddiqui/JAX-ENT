


# Setup data

# extract _MoPrP.tar

tar -xvf jaxent/examples/2_CrossValidation/data/_MoPrP.tar -C jaxent/examples/2_CrossValidation/data/

# extract _ValDX.tar
python jaxent/examples/2_CrossValidation/data/extract_data_ValDX.py


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






# Featurise Ensure you have: 
# jaxent/examples/2_CrossValidation/data/_cluster_MoPrP
# jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered


python jaxent/examples/2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py


python jaxent/examples/2_CrossValidation/fitting/jaxENT/splitdata_CrossVal.py

python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py \
    --dfrac_file jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_dfrac.dat \
    --segs_file  jaxent/examples/2_CrossValidation/data/_MoPrP/_output/MoPrP_segments.txt \
    --output_dir jaxent/examples/2_CrossValidation/data/_MoPrP_covariance_matrices/

bash jaxent/examples/2_CrossValidation/fitting/jaxENT/run_maxent_parallel_test.sh

