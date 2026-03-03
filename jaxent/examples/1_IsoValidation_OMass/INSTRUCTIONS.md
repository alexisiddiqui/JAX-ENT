
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

[Optional]: Use the `unpack_iso_validation_data.py` script below to automatically download and prepare the required validation datasets.

## Workflow (based on alpha-commands.sh)

1. **Prepare Validation Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/unpack_iso_validation_data.py
   ```
   *Downloads and unpacks the quick testing data package (approx. 275MB when unpacked).*

2. **Extract Synthetic Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/data/extract_synthetic_data.py
   ```
   *Extracts synthetic HDX data for the validation study.*

3. **Featurise Ensembles**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py
   ```
   *Generates descriptors and features for the ISO, TRI, and BI ensembles.*

4. **Plot Intrinsic Rates (Simple)**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/analysis/plot_intrinsic_rates_simple.py
   ```
   *Visualizes the intrinsic exchange rates used in the forward model.*

5. **Split Data**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/splitdata_ISO.py
   ```
   *Creates training and validation data splits for cross-validation.*

6. **Analyse Splits**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/analysis/analyse_split_ISO_TRI_BI.py
   ```
   *Evaluates the coverage and characteristics of the generated data splits.*

7. **Extract Open/Closed Clusters**
   ```bash
   python jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py
   ```
   *Identifies and extracts conformational clusters representing distinct molecular states.*

8. **Compute Sigma (Covariance Matrices)**
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
   *Computes covariance matrices (Sigma) for the synthetic ensembles to account for ensemble uncertainty.*

9. **Run Optimisation and Comprehensive Analysis**
   ```bash
   # Note: This runs several configurations in parallel using background jobs.
   bash jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/run_maxent_parallel_SIGMA_TEST.sh
   ```
   *Runs a parallel MaxEnt optimization sweep followed by an automated analysis pipeline.*

   ### Analysis Pipeline Discussion
   The `run_maxent_parallel_SIGMA_TEST.sh` script doesn't just perform the fitting; it automatically executes a series of analysis modules:
   - **Recovery Analysis**: Determines how accurately the optimization recovers the known target populations (e.g., 60/40 mix of Open/Closed states).
   - **Weights Validation**: Inspects the final ensemble weights, ensuring they respect the MaxEnt regularization and reporting the resulting Shannon entropy.
   - **CV Validation**: Aggregates performance metrics (MSE, $R^2$) across all cross-validation splits to ensure model robustness.
   - **Model Scoring & Selection**: Processes the results through a **Mixed Linear Model** (MLM) framework to statistically rank different hyperparameter combinations (loss functions, split types, $\lambda$ values).
   - **Process & Plot**: Finally, it scores all models and generates summary plots (e.g., population recovery vs. regularization strength) to facilitate final model selection.

10. **Compare with HDXer (Manuscript Comparison)**
    ```bash
    python jaxent/examples/1_IsoValidation_OMass/analysis/plot_compare_jaxENT_HDXer.py
    ```
    *Generates final comparison figures against the original HDXer implementation as described in the accompanying manuscript.*
