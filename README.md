# jaxENT
### Version: Alpha 0.1.1 
Disclaimer: This project is currently under construction, the code is mostly functional but is far from complete as a package. \
Initial release expected in the next few months. Want to stay up to date? Questions/Comments/Suggestions? Use our google form below to help drive the development of this tool.

### Feedback and Mailing list: 
https://forms.gle/UDzjdfViMC8i8A1X8

## Maximum Entropy Optimisation for Experimental Biophysics-Simulation Ensemble Integration using JAX.

### Overview 
This package is aimed at bioinformatics researchers across a range of experience levels we aim to have a package that is batteries included but can also be extended to suit needs.

Currently for HDX-MS data.

This package is centered around 3 main functions/scripts:

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
uv venv
source .venv/bin/activate
uv pip install -e .[flag]
```
Please ensure that JAX is correctly installed with CUDA/ROCm if desired, following the official JAX installation instructions for your specific hardware.

Tests can be found in jaxent/tests. 
Some of these tests require the test files found in jaxent/tests/inst.zip - this needs to be extracted to jaxent/tests/inst/ - a search and replace is then performed to correct the suffix before 'JAX-ENT'.
Tests also require the test or dev branch (for pytest).
A setup_test.sh script is provided but this should be handled by the tests now.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[test]
```

---

## Usage Examples


## Quick Start


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
```

****Output Files:**

The script will generate two files in the specified output directory:

1.  `features.npz`: A NumPy archive containing the featurised data. For the BV model, this includes `k_ints`, `heavy_contacts`, and `acceptor_contacts`.
2.  `topology.json`: A JSON file describing the topology of the system, including information about the valid residues processed.

BPTI has 58 residues and 5 Prolines, by defualt we exclude the first residue resulting in features of dimension (52, 500) for both heavy and acceptor contacts.
k_ints are ensemble-averaged so that there is one per residue resulting in an array of shape (52,).

---




### Python API Usage

#### 1. Feature Generation (`run_featurise`)

Generate features from molecular dynamics trajectories for use in maximum entropy optimization:

```python
from pathlib import Path
from MDAnalysis import Universe
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config

# Set up paths to your trajectory files
topology_path = "path/to/your/topology.pdb"
trajectory_path = "path/to/your/trajectory.xtc"

# Create MDAnalysis Universe
test_universe = Universe(str(topology_path), str(trajectory_path))
universes = [test_universe]

# Configure the BV (Best-Vendruscolo) model
bv_config = BV_model_Config()
models = [BV_model(bv_config)]

# Set up featuriser settings
featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

# Create experiment ensemble
ensemble = Experiment_Builder(universes, models)

# Generate features
features, feature_topology = run_featurise(ensemble, featuriser_settings)

print(f"Generated {len(features)} feature sets")
print(f"Feature topology: {feature_topology}")
```

#### 2. Forward Prediction (`run_forward`)

Run forward predictions across multiple simulation parameter sets:

```python
from jaxent.src.predict import run_forward
from jaxent.src.interfaces.simulation import Simulation_Parameters
import jax.numpy as jnp

# Assume you have features and models from run_featurise
# input_features = [features from run_featurise]
# forward_models = [your forward models]

# Create multiple simulation parameter sets for ensemble prediction
simulation_parameters = []
for i in range(10):  # Example: 10 different parameter sets
    params = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames, dtype=jnp.bool_),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1) * (0.8 + 0.4 * i / 10),  # Varying weights
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.bool_),
    )
    simulation_parameters.append(params)

# Run forward prediction across all parameter sets
all_outputs = run_forward(
    input_features=input_features,
    forward_models=forward_models,
    simulation_parameters=simulation_parameters,
    raise_jit_failure=False,
    validate=True
)

print(f"Generated {len(all_outputs)} sets of predictions")
for i, outputs in enumerate(all_outputs):
    print(f"Parameter set {i}: {len(outputs)} outputs")
```

#### 3. Single Prediction (`run_predict`)

Run a single prediction with specific model parameters:

```python
from jaxent.src.predict import run_predict
from jaxent.src.interfaces.simulation import Simulation_Parameters

# Option 1: Using Simulation_Parameters
simulation_params = Simulation_Parameters(
    frame_weights=jnp.ones(n_frames) / n_frames,
    frame_mask=jnp.ones(n_frames, dtype=jnp.bool_),
    model_parameters=[bv_config.forward_parameters],
    forward_model_weights=jnp.ones(1),
    forward_model_scaling=jnp.ones(1),
    normalise_loss_functions=jnp.ones(1, dtype=jnp.bool_),
)

output_features = run_predict(
    input_features=input_features,
    forward_models=forward_models,
    model_parameters=simulation_params,
    raise_jit_failure=False,
    validate=True
)

print(f"Prediction output shape: {output_features[0].y_pred().shape}")

# Option 2: Using sequence of Model_Parameters directly
model_parameters = [bv_config.forward_parameters]  # One for each model

output_features = run_predict(
    input_features=input_features,
    forward_models=forward_models,
    model_parameters=model_parameters,
    raise_jit_failure=False,
    validate=True
)
```

#### 4. Data Splitting for Cross-Validation

Split experimental data into training and validation sets with topology-aware methods:

```python
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitter import DataSplitter
from jaxent.src.data.HDX.protection_factor import HDX_protection_factor
from jaxent.src.interfaces.topology import Partial_Topology
import matplotlib.pyplot as plt

# Create experimental dataset
# First, get common residues from your simulation ensemble
common_residues, excluded_residues = Partial_Topology.find_common_residues(
    universes, 
    exclude_selection="resname PRO or resid 1"
)

print(f"Found {len(common_residues)} common residues")

# Create fake experimental data for demonstration
# In practice, you would load your actual HDX-MS data
exp_data = [
    HDX_protection_factor(protection_factor=10.0, top=top) 
    for top in common_residues
]

dataset = ExpD_Dataloader(data=exp_data)
print(f"Created dataset with {len(dataset.data)} datapoints")

# Create data splitter
splitter = DataSplitter(
    dataset=dataset,
    random_seed=42,
    ensemble=universes,  # Your MD simulation ensemble
    common_residues=set(common_residues),
    check_trim=False,
    centrality=True,  # Use centrality-based sampling
    train_size=0.7,   # 70% for training
)

# Perform random split
train_data, val_data = splitter.random_split(remove_overlap=True)

print(f"Training set: {len(train_data)} fragments")
print(f"Validation set: {len(val_data)} fragments")
print(f"Total coverage: {(len(train_data) + len(val_data))/len(dataset.data)*100:.1f}%")

# Access split topology information
print("Training topologies by chain:")
for chain, topo in splitter.last_split_train_topologies_by_chain.items():
    print(f"  Chain {chain}: {len(topo.residues)} residues")

print("Validation topologies by chain:")
for chain, topo in splitter.last_split_val_topologies_by_chain.items():
    print(f"  Chain {chain}: {len(topo.residues)} residues")
```

#### 5. Complete Workflow Example

Here's a complete example combining featurization, data splitting, optimization, and analysis:

```python
import os
import jax
import jax.numpy as jnp
from pathlib import Path
from MDAnalysis import Universe
import matplotlib.pyplot as plt

# Set JAX to use CPU (optional)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_platform_name", "cpu")

# Import jaxENT modules
from jaxent.src.custom_types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.featurise import run_featurise
from jaxent.src.predict import run_predict
from jaxent.src.opt.run import run_optimise
from jaxent.src.opt.losses import hdx_pf_l2_loss
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.data.loader import ExpD_Dataloader, Dataset
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.data.splitting.sparse_map import create_sparse_map

def complete_workflow_with_optimization():
    # 1. Load trajectory data
    topology_path = "path/to/topology.pdb"
    trajectory_path = "path/to/trajectory.xtc"
    universe = Universe(str(topology_path), str(trajectory_path))
    
    # 2. Set up models and featurization
    bv_config = BV_model_Config()
    model = BV_model(bv_config)
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    
    ensemble = Experiment_Builder([universe], [model])
    
    # 3. Generate features
    print("Generating features...")
    features, feature_topology = run_featurise(ensemble, featuriser_settings)
    
    # Get trajectory length for parameter setup
    trajectory_length = features[0].features_shape[1]
    
    # 4. Set up simulation parameters
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length, dtype=jnp.bool_),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1, dtype=jnp.bool_),
    )
    
    # 5. Create simulation object
    simulation = Simulation(
        forward_models=[model], 
        input_features=features, 
        params=params
    )
    simulation.initialise()
    
    # 6. Create experimental dataset
    print("Setting up experimental data...")
    # For demonstration, create synthetic HDX data
    # In practice, you would load your actual experimental data
    exp_data = [
        HDX_protection_factor(protection_factor=10.0 + i * 0.5, top=top)
        for i, top in enumerate(feature_topology[0])
    ]
    
    dataset = ExpD_Dataloader(data=exp_data)
    
    # 7. Split data for cross-validation
    print("Splitting data...")
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=[universe],
        common_residues=set(feature_topology[0]),
        train_size=0.7
    )
    
    train_data, val_data = splitter.random_split()
    print(f"Training: {len(train_data)}, Validation: {len(val_data)} fragments")
    
    # 8. Create sparse maps for efficient indexing
    print("Creating sparse maps...")
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)
    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)
    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)
    
    # 9. Set up dataset with train/val/test splits
    dataset.train = Dataset(
        data=train_data,
        y_true=jnp.array([data.extract_features() for data in train_data]),
        residue_feature_ouput_mapping=train_sparse_map,
    )
    
    dataset.val = Dataset(
        data=val_data,
        y_true=jnp.array([data.extract_features() for data in val_data]),
        residue_feature_ouput_mapping=val_sparse_map,
    )
    
    dataset.test = Dataset(
        data=exp_data,
        y_true=jnp.array([data.extract_features() for data in exp_data]),
        residue_feature_ouput_mapping=test_sparse_map,
    )
    
    # 10. Set up optimization configuration
    opt_settings = OptimiserSettings(name="hdx_optimization", n_steps=100)
    
    # 11. Run optimization
    print("Running optimization...")
    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset,),
        config=opt_settings,
        forward_models=[model],
        indexes=[0],  # Index of the model to optimize
        loss_functions=[hdx_pf_l2_loss],
    )
    
    optimized_simulation, optimization_history = opt_simulation
    
    # 12. Analyze results
    print("Optimization complete!")
    print(f"Final loss: {optimization_history['total_loss'][-1]:.4f}")
    
    # 13. Make predictions with optimized parameters
    print("Making predictions with optimized parameters...")
    final_predictions = run_predict(
        input_features=features,
        forward_models=[model],
        model_parameters=optimized_simulation.params,
        validate=True
    )
    
    # 14. Optional: Visualize results
    print("Creating visualizations...")
    
    # Plot optimization history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve
    ax1.plot(optimization_history['total_loss'])
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Optimization Progress')
    ax1.grid(True)
    
    # Frame weights evolution (if available)
    if 'frame_weights' in optimization_history:
        weights_history = jnp.array(optimization_history['frame_weights'])
        im = ax2.imshow(weights_history.T, aspect='auto', origin='lower')
        ax2.set_xlabel('Optimization Step')
        ax2.set_ylabel('Frame Index')
        ax2.set_title('Frame Weights Evolution')
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Workflow complete!")
    print(f"- Generated {len(features)} feature sets")
    print(f"- Optimized simulation with {len(train_data)} training fragments")
    print(f"- Final predictions shape: {final_predictions[0].y_pred().shape}")
    
    return optimized_simulation, optimization_history, final_predictions

if __name__ == "__main__":
    complete_workflow_with_optimization()
```

---

## CLI Tools

The following command-line interface (CLI) tools are available after installation:

*   `jaxent-featurise`: For featurising molecular dynamics trajectories.
*   `jaxent-predict`: For running predictions with a single set of model parameters.
*   `jaxent-forward`: For running forward predictions across multiple sets of simulation parameters.

### `jaxent-featurise` Usage Example

Here is an example of how to use the `jaxent-featurise` tool to generate features from a molecular dynamics trajectory using the Best-Vendruscolo (BV) model.

**Command:**

```bash
jaxent-featurise \
    --top_path /path/to/your/topology.pdb \
    --trajectory_path /path/to/your/trajectory.xtc \
    --output_dir /path/to/your/featurisation_output \
    --name test_bv \
    bv \
    --temperature 300.0 \
    --ph 7.0 \
    --num_timepoints 1 \
    --timepoints 0.167 \
    --residue_ignore -2 2 \
    --mda_selection_exclusion "resname PRO or resid 1"
```

**Argument Explanations:**

*   `--top_path`: Path to the topology file (e.g., PDB).
*   `--trajectory_path`: Path to the trajectory file (e.g., XTC, DCD).
*   `--output_dir`: Directory where the output files will be saved.
*   `--name`: A name for this featurisation run.
*   `bv`: Specifies that we are using the BV model.
*   `--temperature`: Temperature in Kelvin.
*   `--ph`: The pH value.
*   `--num_timepoints`: The number of timepoints.
*   `--timepoints`: The timepoint values in minutes.
*   `--residue_ignore`: A range of residues to ignore relative to the donor.
*   `--mda_selection_exclusion`: An MDAnalysis selection string to exclude certain residues (e.g., prolines).

**Output Files:**

The script will generate two files in the specified output directory:

1.  `features.npz`: A NumPy archive containing the featurised data. For the BV model, this includes `k_ints`, `heavy_contacts`, and `acceptor_contacts`.
2.  `topology.json`: A JSON file describing the topology of the system, including information about the residues processed.

---

### Supports:
- HDX-MS for fitting

### Upcoming:
SAXS
NMR
CryoEM???

---

## Known issues:

- Partial_Topology objects renumbers termini residues
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
