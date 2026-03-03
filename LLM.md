# JAX-ENT: JAX-based Ensemble Modeling and Optimization Toolkit

By default the Python environment is found in `.venv`.

JAX-ENT is a JAX-based library for ensemble modeling and optimization in structural biology/biophysics, with a strong focus on HDX-MS workflows. This document is a codebase-oriented guide for a mixed audience: contributors who need architectural detail and users who need practical workflow grounding.

## What this guide prioritizes

- **Current implementation reality** in `jaxent/src` (the main codebase).
- **Grounded workflows** from `jaxent/examples` (real, end-to-end pipelines).
- **How components connect** across featurization, prediction, splitting, and optimization.

---

## Repository orientation

### Main implementation surface: `jaxent/src`

- `analysis/` - plotting and analysis helpers (for example `analysis/plots/optimisation.py`).
- `config/` - configuration infrastructure.
- `custom_types/` - typed interfaces/data containers (`ForwardModel`, feature bases, datapoints, keys, settings).
- `data/` - data loading, sparse mapping, dataset splitting.
- `interfaces/` - builder, simulation/model interfaces, topology system.
- `models/` - forward models and model-specific feature/parameter logic.
- `opt/` - loss definitions, optimization state/history, optimizer orchestration.
- `utils/` - JAX helpers, JIT guards, HDF persistence, MDAnalysis utilities.
- Top-level execution helpers:
  - `featurise.py` - orchestration for feature generation.
  - `predict.py` - single-parameter prediction pathway.
  - `predict_forward.py` - batched/multi-parameter forward pathway.

### CLI entrypoints: `jaxent/cli`

- `featurise.py` - generate model input features from topology/trajectory.
- `predict.py` - run one prediction set from saved features/topology.
- `forward.py` - run many simulation parameter sets in one pass.
- `efficient_k_cluster.py` - memory-aware PCA + clustering for trajectory reduction.

### Workflow grounding: `jaxent/examples`

Active example suites include:

- `1_IsoValidation_OMass`
- `2_CrossValidation`
- `3_CrossValidationBV`
- `predict_traj`
- shared helpers in `examples/common`

These are extensive and script-heavy; use the context strategy described later when navigating them.

---

## Core architecture and data flow

JAX-ENT is modular, but the modules are connected by a key-driven contract and topology-aware mapping.

### 1) Experimental data layer

- Experimental points inherit from `ExpD_Datapoint` (`jaxent/src/custom_types/datapoint.py`).
- HDX-specific datapoints live in `jaxent/src/custom_types/HDX.py`:
  - `HDX_peptide` (`m_key("HDX_peptide")`)
  - `HDX_protection_factor` (`m_key("HDX_resPf")`)
- `ExpD_Dataloader` (`jaxent/src/data/loader.py`) stores raw datapoints and builds JAX-traceable train/val/test `Dataset` objects.

### 2) Forward model contract and key dispatch

- `ForwardModel` base (`jaxent/src/custom_types/base.py`) defines:
  - `initialise(ensemble)`
  - `featurise(ensemble)`
  - `forward` map keyed by `m_key`
- Model config objects (for example `BV_model_Config`, `NetHDXConfig`) in `jaxent/src/models/config.py` determine whether outputs are residue-level (`HDX_resPF`) or peptide-level (`HDX_peptide`) via `num_timepoints`/key selection.

### 3) Featurisation

- `run_featurise` (`jaxent/src/featurise.py`) orchestrates feature generation from an `Experiment_Builder`.
- `Experiment_Builder` (`jaxent/src/interfaces/builder.py`) links universes, forward models, optional data, and can construct a `Simulation`.
- For HDX BV, `BV_model.featurise` computes heavy/acceptor contacts from MD trajectories and returns `BV_input_features`.

### 4) Topology alignment and sparse mapping

- Topology subsystem (`jaxent/src/interfaces/topology/`) is central for robust fragment/residue alignment:
  - `Partial_Topology` (`core.py`)
  - `TopologyFactory` (`factory.py`)
  - `PairwiseTopologyComparisons` (`pairwise.py`)
  - `mda_TopologyAdapter` (`mda_adapter.py`)
  - `PTSerialiser` (`serialise.py`)
- `create_sparse_map` (`jaxent/src/data/splitting/sparse_map.py`) maps residue-level model outputs to experimental fragments.
- This map is the critical bridge for comparing residue-wise predictions to peptide-wise HDX observations.

### 5) Simulation and prediction

- `Simulation` (`jaxent/src/models/core.py`) is the execution core:
  - validates input/model dimensions
  - normalizes simulation parameters
  - manages JIT/non-JIT forward execution
- `run_predict` (`jaxent/src/predict.py`) executes one parameter set.
- `run_forward` (`jaxent/src/predict_forward.py`) iterates over many `Simulation_Parameters` sets.

### 6) Optimization

- `run_optimise` (`jaxent/src/opt/run.py`) controls the training loop.
- `OptaxOptimizer` (`jaxent/src/opt/optimiser.py`) wraps Optax update logic and masking behavior.
- `OptimizationState` / `OptimizationHistory` (`jaxent/src/opt/base.py`) track parameter/loss trajectories.
- Losses are composed from `jaxent/src/opt/losses.py` and `jaxent/src/opt/loss/`.

### 7) Persistence and analysis

- `jaxent/src/utils/hdf.py` serializes/deserializes optimization history, states, and parameters to HDF5.
- `jaxent/src/analysis/plots/optimisation.py` provides plotting utilities for loss and weight evolution.

---

## Forward model families

### Best-Vendruscolo (BV)

- Core implementation: `jaxent/src/models/HDX/BV/forwardmodel.py`
- Feature dataclasses: `jaxent/src/models/HDX/BV/features.py`
- Forward pass logic: `jaxent/src/models/HDX/forward.py`
- Contacts/intrinsic-rate utilities:
  - `jaxent/src/models/func/contacts.py`
  - `jaxent/src/models/func/uptake.py`

BV supports both residue protection factor outputs and peptide uptake outputs through key-based forward-pass selection.

### Linear BV variant

- Implemented via `linear_BV_model` / `linear_BV_ForwardPass`.
- Useful when treating BV coefficients as explicit timepoint-wise linear parameters.

### netHDX

- Forward model: `jaxent/src/models/HDX/netHDX/forwardmodel.py`
- Features: `jaxent/src/models/HDX/netHDX/features.py`
- Network construction + metrics: `jaxent/src/models/func/netHDX.py`

netHDX builds hydrogen-bond contact networks and derives graph metrics for HDX prediction.

---

## Data splitting and experimental alignment details

`DataSplitter` (`jaxent/src/data/splitting/split.py`) supports multiple strategies and topology-aware filtering:

- random
- stratified
- spatial
- cluster-based/sequence-based variants used in examples

The splitter operates after filtering to common residues and uses topology comparison methods to avoid invalid overlaps.

---

## Practical workflow map (user-facing)

For most workflows, the sequence is:

1. **Prepare topology + trajectories + experimental inputs**
2. **Featurise** (`jaxent/cli/featurise.py` or Python API)
3. **Split/map data** (examples use split scripts + sparse maps)
4. **Predict/forward test** (`jaxent/cli/predict.py`, `jaxent/cli/forward.py`)
5. **Optimize** (example fitting scripts and `run_optimise`)
6. **Analyze + persist outputs** (`analysis/` scripts, `utils/hdf.py`)

---

## Examples: real workflows and how to read them safely

The example directories are rich and large. Treat them as workflow references, not lightweight snippets.

### Active examples

- **Example 1 (`1_IsoValidation_OMass`)**
  - Synthetic/controlled validation workflow with known target populations.
  - End-to-end: data prep -> featurize -> split -> covariance -> optimization -> analysis.

- **Example 2 (`2_CrossValidation`)**
  - Real experimental MoPrP HDX workflow.
  - Emphasizes data extraction/prep, structural clustering, cross-validation, and model comparison.

- **Example 3 (`3_CrossValidationBV`)**
  - Extends Example 2 with 2D hyperparameter sweeps over MaxEnt and BV regularization.

- **`predict_traj`**
  - Directory-level and clustered trajectory prediction entrypoints.

### Context-management strategy for examples (recommended)

Because examples are extensive, inspect in this order:

1. `README.md` (workflow narrative and structure)
2. `INSTRUCTIONS.md` (minimal runnable sequence)
3. `commands.sh` + `config.yaml` (actual execution surface)
4. Only then drill into specific Python scripts needed for the task

This avoids loading unnecessary analysis scripts too early while preserving workflow grounding.

---

## Extension points (for contributors)

### Add a new experimental datatype

1. Create a new datapoint type under `custom_types/` with a unique `m_key`.
2. Ensure extraction methods and topology semantics are defined.
3. Update loaders/splitting/mapping flow if fragment behavior differs.

### Add a new forward model

1. Implement a new `ForwardModel` subclass in `models/`.
2. Define `initialise`, `featurise`, and key->forward-pass map.
3. Add model parameters/config in `models/config.py` and model-specific parameter files.
4. Wire into CLI/API paths as needed.

### Add a new loss

1. Implement loss callable under `opt/losses.py` or `opt/loss/`.
2. Ensure compatibility with `run_optimise` data target routing.
3. Expose/compose in fitting scripts used by examples.

---

## Common gotchas

- Many example scripts assume paths are run from repository root.
- Topology alignment is strict; mismatches can break sparse mapping or splitting.
- `Simulation_Parameters` dimensional consistency matters (`frame_weights`, masks, model counts).
- `num_timepoints` changes forward-key behavior (`HDX_resPF` vs `HDX_peptide`).
- Example workflows are not small; prefer staged reading to manage context.

---

## Quick navigation

- Start from architecture: `jaxent/src/models/core.py`, `jaxent/src/featurise.py`, `jaxent/src/opt/run.py`
- Start from data alignment: `jaxent/src/data/loader.py`, `jaxent/src/data/splitting/sparse_map.py`, `jaxent/src/interfaces/topology/`
- Start from practical workflows: `jaxent/examples/*/README.md`
- Start from command usage: `jaxent/cli/*.py` and `jaxent/cli/README.md`
