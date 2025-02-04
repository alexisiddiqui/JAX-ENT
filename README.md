# JAX-ENT
Maximum Entropy Optimisation for Experimental Biophysics-Simulation Ensemble Integration using JAX.

Currently for HDX-MS data.

Supports:

To do:
HDX-MS peptide data
HDX-MS residue protection factor

Upcoming:
SAXS
NMR
CryoEM???


##############################################################################

Featuriser: Multiple ensembles requires handling of the common resiudes
--> created topology fragment to handle this.

Optimiser: Each simulation should have its own experimental data section 
- Should tie this together with the data splitting class


TODO:
run_optimise -> seperate SGD into a sperate function
-> extract optimiser step out
-> map using jax
BV_model.initialise -> fragment topology hadnler

frame_average_features - need to find a way to make this jaxable - maybe is fine for now?


calc_BV_contacts_universe -> fix typing to use numpy - same with the rest of the featuriser code

##############################################################################





This package is aimed at bioinformatics researchers across a range of experience levels we aim to have a package that is batteries included but can also be extende to suit needs.


This package is centered around 3 main functions:

Featurise

Optimise 

Analysis


CLI wrappers will be included under 'scripts'.


uses uv for package management

Please check that jax is correctly installed with CUDA/ROCm if desired.

To do:
Add CUDA/ROCm flag to uv installation

