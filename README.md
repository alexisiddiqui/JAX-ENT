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

Align the shapes of the inputs and outputs Arrays to be consistent
- Add type hinting to the shape of the arrays (Chex?)

Printing of classes and Logging
CLI Interface
CI

***Docs



## need to fix implementation of optax optimiser to be better fit for jax 
- removal of prints
- seperation of losses -> fix computation to actually use constants
- also need to find a way to record loss and simulation parameters for further analysis 

# change peptide to refer to hdx_peptide


# Speed optimisations:
for the simulation class - both the inputs and outputs should be accessible by keys - output features can stay as an array until its accessed via a property decorator



for now just keep the loss functions as indexed functions



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

