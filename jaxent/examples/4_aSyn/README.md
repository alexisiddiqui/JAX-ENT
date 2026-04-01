
aSyn HDX data over 4 conditions:

Tris only,Extracellular,Intracellular,Lysosomal
Baseline, high Ca, high K, low pH



k_obs data for each amino acid residue
- convert to protection factors by dividing by the baseline k_int

-> use the featurise method on the topology to obtain intrinsic rates
Pf = k_int/k_obs

Af2 MSA subsampled dataset
12700 structures
1000 clusters


Fit BV model (fitting frames and model parameters to account for experimental conditions)

Select models using valdiation loss for each replicate (x3) over maxent Hparam sweep.

For now the analysis is simply the recovery obtained between the predicted protection factors and the experimental protection factors.

When creating the scripts - essentially just treat each condition like an ensemble.
- the only difference to the other examples is that the bash runner and optimisation script will be written to accept a list of conditions instead of a list of ensembles. 
But the output names etc should follow the same structure.