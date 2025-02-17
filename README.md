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


TODO:

Need to create data splitting class 
the data splits should be added to the experimental dataset class

Training and Validation dataset also need to contain the mappings from input features to each of the training data

 for the simulation class - both the inputs and outputs should be accessible by keys - output features can stay as an array until its accessed via a property decorator

for now just keep the loss functions as indexed functions





run_optimise -> swap out SGD function with optax
-> extract optimiser step out
-> map using jax

frame_average_features - need to find a way to make this jaxable - maybe is fine for now?

calc_BV_contacts_universe -> fix typing to use numpy - same with the rest of the featuriser code

# create Enum to handle which parameters are being updated

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

