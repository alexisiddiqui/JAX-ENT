
# download the quick testing data

# bash jaxent/examples/1_IsoValidation_OMass/data/download_data.sh

# this script fails for some reason - the files need to be moved to zenodo later

# make sliced_20 folder

cp jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_20 -r



python /homes/hussain/dragon315/dragon315_Documents/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/get_HDXer_AutoValidation_data.py


python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py
python jaxent/examples/1_IsoValidation_OMass/analysis/plot_intrinsic_rates_simple.py


python jaxent/examples/1_IsoValidation_OMass/data/extract_synthetic_data.py


python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/splitdata_ISO.py
python jaxent/examples/1_IsoValidation_OMass/analysis/analyse_split_ISO_TRI_BI.py

python jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py


python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \
    --clustering_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/../../data/_clustering_results \
    --features_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise \
    --ensemble_name ISO_BI \
    --output_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_covariance_matrices_sigma

python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \
    --clustering_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/../../data/_clustering_results \
    --features_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise \
    --ensemble_name ISO_TRI \
    --output_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_covariance_matrices_sigma



# this is for testing only - the full run TBC
bash jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/run_maxent_parallel_SIGMA_TEST.sh

# for comparison with valdxer (manuscript only)
python jaxent/examples/1_IsoValidation_OMass/analysis/plot_compare_jaxENT_HDXer.py