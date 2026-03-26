# SAXS+HDX Fitting example  
In this example we will demonstrate combined fitting of SAXS and HDX-MS data using the jaxent framework.

We use the CaM ensemble generated from NeuralPlexer from the previous scripts.

This ensemble consists of 1125 structures - 5 x 225 conditions. These conditions are a pulldown of:
- 3 initial clusters (APOlike, HOLOlike, and Neither) - from these clusters 5 subclusters mediods were obtained.
- 5 different Ca2+ binding states (0, 1, 2, 3, or 4 Ca2+ bound)
- 3 different CDZ binding states (0, 1, or 2 CDZ bound)

This is to be fit against the data obtained for CaM with and without CDZ. (CaM+CDZ and CaM-CDZ).
- As we have HDX-MS and SAXS data for both conditions, we compare 3 scenarios:
    1. Fit to SAXS data only
    2. Fit to HDX-MS data only
    3. Fit to both SAXS and HDX-MS data simultaneously


Fitting setup:
- Reweighting only
- Loss function:
    - For SAXS: chi-squared between experimental and predicted curves
    - For HDX-MS: Sigma MSE loss function
    - For weights: KL divergence between the weights and the uniform distribution (To be updated with ESS loss function in the future)

Note: We need to obtain a correct scaling factor for the SAXS and HDX-MS loss functions so that they are on the same order of magnitude for combined fitting.
- This will be obtained from the initial fitting experiments via the ratio of the loss function values at the start of training from the SAXS-only and HDX-MS-only fitting experiments. 


Data splitting:
- 3 split-replicates for each target data (CaM+CDZ and CaM-CDZ)
    - HDX-MS: random split, sequence-cluster and spatial splits
    - SAXS: random split, stratified, and statified-random splits

Note: For simplicity we will only consider one split type (HDX: sequence-cluster and SAXS stratified-random) but the pipeline should be setup to handle multipe split types across the replicates and targets.




## Analysis 
Plots will be generated for the following combinations:
- fitting scenario (SAXS-only, HDX-MS-only, combined) (columns)
- target data (CaM+CDZ, CaM-CDZ) (rows)




For each split replicate/hyperparameter combination:
- Representative parameters (weights) are obtained by the best validation loss from the optimisation history over convergence thresholds.
    - To handle multiple target data - we rank the validation losses for each target specifically and then average the ranks across targets to obtain an overall ranking of the parameter sets. The best ranking is then selected as the representative parameter set for that split/replicate.


Frame details:
jaxent/examples/4_SAXS/ensemble_generation/neuralplexer/collected_structures/frame_ordering.csv
- this csv contains the mapping information for clustering and ligand information.
    - Headers: Primary_cluster (n, 0, 1), Secondary_cluster (0-5), n_Ca_final (0-4), n_CDZ_ligands_final (0-2)


Plots:
For each scenario and target data combination:
- ESS of the weights distribution 
- Bar chart of the Primary cluster distribution of the reweighted ensembles
- Heatmap of n_Ca_final vs n_CDZ_ligands_final for the reweighted ensembles


# Order of operations:
Most of the data preparation scripts have been run and the data is available.
A simple SAXS fitting script has been performed. There are multiple examples for HDX-MS fitting.
1. SAXS-only + KLD fitting script + bash runner
2. Analysis script
3. HDX-MS-only + KLD fitting script + bash runner
4. Calibrate loss scaling factors - plot loss curves vs maxent strength
5. Combined fitting script using KLD loss function + bash runner


# Notes
The analysis script/pipeline should be agnostic to the fitting script used - it should be able to handle the outputs from any of the fitting scripts as long as they are in the correct format. This is to ensure that the analysis can be easily reused for future fitting experiments with different setups.

For the fitting scripts - this means that the structure of the code and the outputs should be consistent and interoperable. Despite fewwer SAXS examples, avoid writing rigid code. Understandability of the pipeline is of most importance.

Save shared functions in jaxent/examples/4_SAXS/fitting/common.py


# Future
- Update the loss function for the combined fitting to be based on ESS rather than KL divergence to the uniform distribution. Fixed at 20 effective sampples. 