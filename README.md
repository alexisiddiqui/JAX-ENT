# jaxENT
### Version: Alpha 0.5.1
This project is in active development, however the core API is now in place and thoroughly tested, and we are working on improving the usability and performance of the package.

Initial release expected in the next few months. Want to stay up to date? Questions/Comments/Suggestions? Use our google form below to help drive the development of this tool.

### Feedback and Mailing list: 
https://forms.gle/UDzjdfViMC8i8A1X8

## Maximum Entropy Optimisation for Experimental Biophysics-Simulation Ensemble Integration using JAX.

### Overview 
This package is aimed at bioinformatics researchers who wish to integrate biophysical data with structural simulations.

Built on JAX, this package provides a flexible and powerful framework to differentiably solve for any inverse problem.


Currently strictly validated for HDX-MS/NMR data but readily supports SAXS and XL-MS data.

This package is centered around a few main functions/scripts:

Featurise (CLI/API)

Predict (CLI/API)

Optimise (API - CLI coming soon)

Analysis (API - CLI planned)

PCA-kmeans clustering (CLI/API)

Pymol visualisation (CLI)

---

## Installation

This project uses `uv` for package management. If you don't have `uv` installed, you can install it via `pipx`:

```bash
pip install uv
```

Once `uv` is installed, navigate to the project root directory and install the dependencies and the package in editable mode:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[flag]
```
Please ensure that JAX is correctly installed with CUDA/ROCm if desired, following the official JAX installation instructions for your specific hardware.

**Tests:**

Tests can be found in jaxent/tests. 
Some of these tests require the test files found in jaxent/tests/inst.zip - this needs to be extracted to jaxent/tests/inst/ - a search and replace is then performed to correct the suffix before 'JAX-ENT'.
Tests also require the test or dev branch (for pytest).
A setup_test.sh script is provided but this should be handled by the tests now.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[test]
```

### PYMOL

To use the pymol visualisation features, you will need to have pymol installed. You can install pymol via conda:

```bash
conda create -n PYMOL pymol-open-source
cd JAX-ENT/
pip install -e .

```

While PyMOL is not required for the featurisation and optimisation steps, it is a useful tool for visualising the results of the featurisation and optimisation. We recommend installing it onto a local machine if you want to use the visualisation features of this package.


---


## Quick Start


### Examples:

All experiments presented in the paper can be found in the examples folder. These are designed to provide understanding of the exposed API to integrate into more complex workflows.

On a modern laptop these examples should take 20-60 minutes each on the CPU, GPUs not yet tested.

For users looking to just perform integration of biophysics-simulation ensembles we would recommend using the CLI methods.



We provide test files for BPTI and HOIP, these can be found in jaxent/test/inst.zip.
```bash
cd jaxent/tests

unzip inst.zip -d inst/
```


**Command:**

```bash
jaxent-featurise \
    --top_path jaxent/tests/inst/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb \
    --trajectory_path jaxent/tests/inst/inst/clean/BPTI/BPTI_sampled_500.xtc \
    --output_dir quick_start_output \
    --name test_bv \
    bv \


jaxent-kCluster \ 
    --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
    --trajectory_paths /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
    --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn \
    --number_of_clusters 1000 \
    --num_components 20 


jaxent-featurise \ 
    --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
    --trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
    --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/features \
    --name aSyn_featurised \
    bv \
    --switch \
    --peptide_trim 0 


jaxent-predict \
    --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data/topology.json \
    --features_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data/features.npz \
    --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data \
    --output_name aSyn_featurised \
    bv 

```


### PYMOL

To visualise the featurised data in pymol, we have a variety of scripts that can be used to generate pymol sessions and are controllable via a yaml config. 

```bash
pymol -r jaxent/src/analysis/pymol/Top_N_structures.py --config jaxent/src/analysis/pymol/test_configs/top_n_MoPrP_weights.yaml
```

This will require you to have data to visualise, you can run the examples and simply update the yaml config to point to the new output folder name.


---


### Supports:
- Flexible loss functions
- Combined fitting of multiple data types
- Flexible featurisation and prediction of experimental observables
- Analysis pipelines for understanding the results of the optimisation and the underlying data
- 

### Upcoming:
NMR??
CryoEM

---

## Known issues:

- Logging during optimisation loop needs reworking to be more informative and less verbose.
- CLI interface is currently very basic and needs work to be more user friendly and flexible. yaml configs for more complex workflows.


---

## TODO:

### Urgent

- Printing of classes and Logging
- CLI Interface
- CI

### Important

- ***Docs

### Technical

#### Performance optimisations:
- Integrate BioFeaturerisers into package and use these.
- Dynamic feature attributeS? Flatten methods consider these as static so are retraced if the features change - this is dangeroud

#### Usability:

- separation of losses -> fix computation to actually use constants
- 

#### Other/Notes:

- for now just keep the loss functions as indexed functions

---
