
This example provides methods to download the datasets directly from the Zenodo link from Bradshaw et al. The zip folder is ~13GB

Alternatively, trimmed ValDXer trajectories can be found in:
https://drive.google.com/drive/folders/1Y9294af-SLca80Xk4D2gmg460NXlt7wX?usp=sharing

Within the latest directory, this example requires the _Bradshaw directory - place this in the data/ dir.

total example dir ~1GB, required _Bradshaw folder within ~275MB

### All paths are relative to the scripts themselves - you are intended to run the scripts from the main repo dir: JAX-ENT/

# Instructions
```bash
cd /path/to/installdir/
rm -rf JAX-ENT/
git clone https://github.com/alexisiddiqui/JAX-ENT.git
cd JAX-ENT/
uv venv
source .venv/bin/activate
uv pip install -e .  # you dont need extra flags for the examples to work
```

[Optional]: Download the '_Bradshaw' folder from the latest 'ValDXer_500' directory in the google drive link above into the 'data' folder.

## Workflow (based on commands.sh)

1. **Get and Prepare Validation Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/data/get_HDXer_AutoValidation_data.py
   ```
   *Downloads/prepares the initial dataset.*

2. **Split Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/splitdata_ISO.py
   ```
   *Splits the data into training and validation sets.*

3. **Plot Intrinsic Rates (Simple)**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/analysis/plot_intrinsic_rates_simple.py
   ```
   *Plots simple intrinsic rates.*

4. **Analyse Splits**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/analysis/analyse_split_ISO_TRI_BI.py
   ```
   *Analyses the generated data splits (ISO, TRI, BI).*

5. **Extract Synthetic Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/data/extract_synthetic_data.py
   ```
   *Extracts synthetic HDX data.*

6. **Featurise Ensembles**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py
   ```
   *Featurises ISO, TRI, and BI ensembles.*

7. **Extract Open/Closed Clusters**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py
   ```
   *Extracts clusters representing Open and Closed states.*

8. **Compute Sigma (Covariance Matrices) for Synthetic Data**
   ```bash
   # For ISO_BI
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \
       --clustering_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/../../data/_clustering_results \
       --features_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise \
       --ensemble_name ISO_BI \
       --output_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_covariance_matrices_sigma

   # For ISO_TRI
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py \
       --clustering_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/../../data/_clustering_results \
       --features_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise \
       --ensemble_name ISO_TRI \
       --output_dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_covariance_matrices_sigma
   ```

9. **Run Optimisation (MaxEnt Parallel)**
   ```bash
   # This is for testing only - the full run TBC
   bash jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/run_maxent_parallel_SIGMA_TEST.sh
   ```

10. **Compare with HDXer (Manuscript Comparison)**
    ```bash
    python jaxent/examples/1_IsoValidation_OMass/analysis/plot_compare_jaxENT_HDXer.py
    ```



