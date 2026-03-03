
# download the quick testing data

python jaxent/examples/1_IsoValidation_OMass/unpack_iso_validation_data.py


python jaxent/examples/1_IsoValidation_OMass/data/extract_synthetic_data.py

python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py
python jaxent/examples/1_IsoValidation_OMass/analysis/plot_intrinsic_rates_simple.py




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
