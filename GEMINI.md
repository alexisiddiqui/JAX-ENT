# JAX-ENT: JAX-based Ensemble Modeling and Optimization Toolkit

By default the python env is found in the .venv directory. 

JAX-ENT is a Python library built on JAX for performing ensemble-based modeling and optimization, particularly focused on applications in structural biology and biophysics, such as Hydrogen Deuterium Exchange Mass Spectrometry (HDX-MS). It provides a flexible framework for defining forward models, featurizing molecular dynamics (MD) ensembles, handling experimental data, and optimizing model parameters using JAX's automatic differentiation and JIT compilation capabilities.

## Core Concepts and Architecture

The library is structured around several key components that interact to form a complete modeling and optimization pipeline.

### Connecting the Components: An Example Workflow

The flexibility of JAX-ENT comes from how its components are linked together, primarily through a key-based system. Let's walk through an example of how an HDX peptide uptake experiment is modeled.

1.  **Experimental Data (`ExpD_Datapoint`)**: The process starts with the experimental data. For peptide-level HDX data, this is represented by the `HDX_peptide` dataclass (`jaxent/src/custom_types/HDX.py`). A crucial attribute of this class is its `key`, which is set to `m_key("HDX_peptide")`. This key identifies the type of experimental data.

2.  **Forward Model Selection**: The `ForwardModel` (e.g., `BV_model` in `jaxent/src/models/HDX/BV/forwardmodel.py`) is designed to handle different types of experimental data. It contains a `forward` dictionary that maps data type keys to specific `ForwardPass` implementations. For the `BV_model`, this dictionary looks like:
    ```python
    self.forward: dict[m_key, ForwardPass] = {
        m_key("HDX_resPF"): BV_ForwardPass(),
        m_key("HDX_peptide"): BV_uptake_ForwardPass(),
    }
    ```
    When the model is set up with `HDX_peptide` data, the library uses the matching `m_key("HDX_peptide")` to select `BV_uptake_ForwardPass` as the correct prediction function. This is the "Uptake forward model".

3.  **Featurization (`Input_Features`)**: The `BV_model.featurise` method is called to process the MD trajectory. This method calculates contact information and produces an instance of `BV_input_features` (`jaxent/src/models/HDX/BV/features.py`). This object holds the structural features (like `heavy_contacts` and `acceptor_contacts`) for each residue at each frame of the trajectory.

4.  **Prediction (`Output_Features`)**: The selected `BV_uptake_ForwardPass` (`jaxent/src/models/HDX/forward.py`) is a function that takes the `BV_input_features` and the trainable `BV_Model_Parameters` as input. It performs the core scientific calculation to predict the HDX uptake at the residue level, producing an `uptake_BV_output_features` object. This output object also contains the `m_key("HDX_peptide")` key, confirming its association with the peptide data type.

5.  **Loss Calculation**: During optimization, the `Simulation` object (`jaxent/src/models/core.py`) orchestrates the process.
    - A `JaxEnt_Loss` function (e.g., `hdx_uptake_mae_loss` from `jaxent/src/opt/losses.py`) is called.
    - This loss function receives the model's predictions (`uptake_BV_output_features`) and the experimental data (`ExpD_Dataloader` containing `HDX_peptide` objects).
    - A critical step here is bridging the gap between residue-level predictions and peptide-level experimental data. The `create_sparse_map` function (`jaxent/src/data/splitting/sparse_map.py`) generates a matrix that maps the per-residue predictions to the corresponding peptides they belong to.
    - The loss function uses this sparse map to compare the mapped predictions against the true experimental `dfrac` values from the `HDX_peptide` data, and the resulting loss is used to update the model parameters.

This key-based, modular design allows JAX-ENT to be easily extended with new experimental data types, forward models, and loss functions while ensuring they are correctly connected during the modeling and optimization process.


### 1. Molecular Dynamics (MD) Ensemble Handling

The library is designed to work with molecular dynamics trajectories.
-   **MDAnalysis Integration**: It utilizes `MDAnalysis` for loading and manipulating MD trajectories and structures.
-   **Experiment Builder**: `Experiment_Builder` (`jaxent/src/interfaces/builder.py`) acts as an orchestrator to load MD `Universe` objects and prepare them for featurization and simulation.
-   **Residue Consistency**: `find_common_residues` (`jaxent/src/models/func/common.py`) ensures compatibility across ensemble members by identifying common protein residues, which is crucial for building consistent models.

### 2. Featurization

Featurization is the process of converting raw MD data into a format suitable for the simulation models.
-   **Forward Models**: `ForwardModel`s (`jaxent/src/types/base.py`) are abstract base classes that define how raw MD ensemble data is transformed into `Input_Features` (`jaxent/src/types/features.py`).
-   **Featurisation Runner**: The `run_featurise` function (`jaxent/src/featurise.py`) orchestrates the featurization process based on the provided configuration.

#### Best-Vendruscolo (BV) Model
-   **Implementation**: `BV_model` (`jaxent/src/models/HDX/BV/forwardmodel.py`) and `linear_BV_model` implement the BV model for HDX.
-   **Features**: The model uses features like heavy atom and H-bond acceptor contacts (`BV_input_features` in `jaxent/src/models/HDX/BV/features.py`), which are calculated using `calc_BV_contacts_universe` (`jaxent/src/models/func/contacts.py`).
-   **Intrinsic Rates**: Intrinsic exchange rates are calculated using `calculate_intrinsic_rates` in `jaxent/src/models/func/uptake.py`.

#### Network-based HDX (netHDX) Model
-   **Implementation**: `netHDX_model` (`jaxent/src/models/HDX/netHDX/forwardmodel.py`) uses hydrogen bond networks.
-   **Features**: `build_hbond_network` (`jaxent/src/models/func/netHDX.py`) generates contact matrices and various graph-theoretic network metrics (e.g., degree, clustering coefficient, betweenness centrality) as `NetHDX_input_features` (`jaxent/src/models/HDX/netHDX/features.py`).

### 3. Experimental Data Handling

The library provides a robust system for handling experimental data.

#### Data Loading and Representation
-   **Data Points**: `ExpD_Datapoint` (`jaxent/src/custom_types/datapoint.py`) is the base class for experimental data points. Concrete implementations for HDX-MS data include `HDX_peptide` and `HDX_protection_factor` (`jaxent/src/custom_types/HDX.py`).
-   **Data Loader**: `ExpD_Dataloader` (`jaxent/src/data/loader.py`) loads and manages experimental datasets, and can create `Dataset` objects for training, validation, and testing.

#### Topology Interface
The `jaxent/src/interfaces/topology/` directory contains a comprehensive set of modules for handling molecular fragments and their relationships, which is crucial for aligning experimental data with structural features.
-   `Partial_Topology` (`jaxent/src/interfaces/topology/core.py`): Represents molecular fragments.
-   `TopologyFactory` (`jaxent/src/interfaces/topology/factory.py`): Provides methods for creating and manipulating `Partial_Topology` objects.
-   `PairwiseTopologyComparisons` (`jaxent/src/interfaces/topology/pairwise.py`): Offers methods for comparing `Partial_Topology` objects.
-   `mda_TopologyAdapter` (`jaxent/src/interfaces/topology/mda_adapter.py`): Facilitates conversion between MDAnalysis Universe objects and `Partial_Topology` objects.
-   `PTSerialiser` (`jaxent/src/interfaces/topology/serialise.py`): Handles serialization and deserialization of `Partial_Topology` objects.

#### Data Splitting
-   **Sparse Mapping**: `create_sparse_map` (`jaxent/src/data/splitting/sparse_map.py`) creates a sparse mapping matrix to connect residue-wise features to experimental fragments. This is essential for handling incomplete or peptide-level data.
-   **Data Splitter**: `DataSplitter` (`jaxent/src/data/splitting/split.py`) provides functionalities for splitting experimental data into training and validation sets using various strategies (random, stratified, spatial, cluster-based).

### 4. Simulation and Optimization Core

This is the heart of the library, where models are trained and optimized.

#### Core Simulation Objects
-   `Simulation` (`jaxent/src/models/core.py`): The central object that encapsulates `ForwardModel`s, `Input_Features`, and `Simulation_Parameters`. It manages the overall forward pass and its JIT compilation.
-   `Simulation_Parameters` (`jaxent/src/interfaces/simulation.py`): Holds all trainable parameters, including `frame_weights` (for ensemble reweighting), `model_parameters` (for individual forward models), and `forward_model_weights` (for combining multiple models).
-   `Model_Parameters` (`jaxent/src/interfaces/model.py`): A base class for defining model-specific trainable parameters as JAX PyTrees.

#### Optimization
-   `OptaxOptimizer` (`jaxent/src/opt/optimiser.py`): Integrates with `optax` to provide various optimization algorithms (Adam, SGD, etc.). It supports gradient masking to selectively optimize parameters.
-   `run_optimise` (`jaxent/src/opt/run.py`): Orchestrates the entire optimization loop, applying chosen loss functions and updating parameters.
-   `OptimizationHistory` (`jaxent/src/opt/base.py`): Tracks the training and validation losses, and the state of parameters throughout the optimization process.

#### Loss Functions
A rich set of `JaxEnt_Loss` functions are available in `jaxent/src/opt/losses.py` and the `jaxent/src/opt/loss/` directory, including:
-   Mean Squared Error (MSE) and Mean Absolute Error (MAE).
-   Maximum Entropy (MaxEnt) losses (KL divergence, L2 penalty) for frame weights.
-   Monotonicity loss for HDX uptake.
-   Frame weight consistency losses (cosine, correlation, KL divergence).

#### Persistence
-   `hdf.py` (`jaxent/src/utils/hdf.py`): Provides utilities for saving and loading optimization history and parameters to/from HDF5 files.

## Command-Line Interface (CLI)

JAX-ENT provides a set of command-line scripts for common tasks, located in `jaxent/cli/`.

-   **`featurise.py`**: This script generates features from a molecular dynamics trajectory. It supports different models like the Best-Vendruscolo (BV) model and the netHDX model. You can specify the model and its parameters, input trajectory and topology, and output directory.

-   **`predict.py`**: This script runs predictions using pre-featurised data and a given set of model parameters. It allows you to quickly see the output of a model with specific parameters without running a full optimization.

-   **`forward.py`**: This script runs a forward prediction using featurised data and one or more sets of simulation parameters. This is useful for exploring the effect of different parameter sets.

-   **`efficient_k_cluster.py`**: This script performs memory-efficient k-means clustering on a trajectory to reduce the size of an ensemble. It uses PCA for dimensionality reduction and generates diagnostic plots.

## Key Workflows

1.  **Data Preparation**:
    *   Load MD trajectories and experimental HDX data.
    *   Identify common residues across the ensemble.
    *   Split experimental data into training and validation sets.
    *   Create sparse mappings to align experimental fragments with residue-wise features.

2.  **Feature Generation**:
    *   Apply chosen `ForwardModel`s (e.g., BV, netHDX) to the MD ensemble to compute `Input_Features` (e.g., contact maps, network metrics). This can be done using the `featurise.py` CLI script.

3.  **Model Setup**:
    *   Initialize `Simulation_Parameters` with initial guesses for frame weights, model parameters, and loss function weights/scaling.
    *   Construct a `Simulation` object, which combines the featurised data, forward models, and simulation parameters.

4.  **Optimization**:
    *   Define an `OptaxOptimizer` with a desired learning rate, algorithm, and parameter masks.
    *   Select one or more `JaxEnt_Loss` functions.
    *   Run the `run_optimise` function, which iteratively updates the `Simulation_Parameters` to minimize the combined loss.

5.  **Analysis and Visualization**:
    *   The `analysis/plots/optimisation.py` module provides tools to visualize the training progress, including loss curves and the evolution of frame weights.

## Utility Modules

-   `jaxent/src/utils/hdf.py`: Utilities for saving and loading data to/from HDF5 files.
-   `jaxent/src/utils/jax_fn.py`: Core JAX-based functions like `frame_average_features`.
-   `jaxent/src/utils/jit_fn.py`: A `jit_Guard` context manager to handle JAX's JIT compilation, clearing caches to prevent errors in interactive environments.
-   `jaxent/src/utils/mda.py`: Utilities for working with MDAnalysis, such as `determine_optimal_backend`.