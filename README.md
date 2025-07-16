# jaxENT
### Version: Pre-Alpha 0.1.0 
Disclaimer: This project is currently under construction, the code is mostly functional but is far from complete as a package. \
Initial release expected in the next few months. Want to stay up to date? Questions/Comments/Suggestions? Use our google form below to help drive the development of this tool.

### Feedback and Mailing list: 
https://forms.gle/UDzjdfViMC8i8A1X8



## Maximum Entropy Optimisation for Experimental Biophysics-Simulation Ensemble Integration using JAX.


### Overview 
This package is aimed at bioinformatics researchers across a range of experience levels we aim to have a package that is batteries included but can also be extended to suit needs.

Currently for HDX-MS data.


This package is centered around 3 main functions/scripst:

Featurise

Optimise 

Analysis


---


## Installation

This project uses `uv` for package management. If you don't have `uv` installed, you can install it via `pipx`:

```bash
pip install uv
```

Once `uv` is installed, navigate to the project root directory and install the dependencies and the package in editable mode:

```bash
uv venc
source .venv/bin/activate
uv pip install -e .[flag]
```
Please ensure that JAX is correctly installed with CUDA/ROCm if desired, following the official JAX installation instructions for your specific hardware.



Tests can be found in jaxent/tests. Some of these tests require the test files found in jaxent/tests/inst.zip - this needs to be extracted to jaxent/tests/inst/ - a search and replace is then performed to correct the suffix before 'JAX-ENT'.
Tests also require the dev branch (for pytest).

```bash
uv venc
source .venv/bin/activate
uv pip install -e .[dev]
```




## CLI Tools

The following command-line interface (CLI) tools are available after installation:

*   `jaxent-featurise`: For featurising molecular dynamics trajectories.
*   `jaxent-predict`: For running predictions with a single set of model parameters.
*   `jaxent-forward`: For running forward predictions across multiple sets of simulation parameters.




---



### Supports:
- HDX-MS for fitting



### Upcoming:
SAXS
NMR
CryoEM???

---
## Known issues:

- jax jit compile sometimes hangs
- additional computations during optimisation loop



---



## TODO:

### Urgent

- Align the shapes of the inputs and outputs Arrays to be consistent
- Add type hinting to the shape of the arrays (Chex?)
- fix partial topology, needd methods to create and combine appropriately
- jit boundary issues and simulation class, should the forward method be a function?
- Printing of classes and Logging
- CLI Interface
- Shiny web app 
- CI

### Important

- ***Docs

### Technical

#### Perormance optimisations:
for the simulation class - both the inputs and outputs should be accessible by keys - output features can stay as an array until its accessed via a property decorator

#### Usability:

- need to fix implementation of optax optimiser to be better fit for jax 
    - removal of prints
    - seperation of losses -> fix computation to actually use constants
    - also need to find a way to record loss and simulation parameters for further analysis 
- change peptide to refer to hdx_peptide



#### Other/Notes:


- for now just keep the loss functions as indexed functions


- frame_average_features - need to find a way to make this jaxable - maybe is fine for now?

- calc_BV_contacts_universe -> fix typing to use numpy - same with the rest of the featuriser code

- create Enum to handle which parameters are being updated

---







