# JAX-ENT: JAX-based Ensemble Modeling and Optimization Toolkit

JAX-ENT is a Python library built on JAX for performing ensemble-based modeling and optimization, particularly focused on applications in structural biology and biophysics, such as Hydrogen Deuterium Exchange Mass Spectrometry (HDX-MS). It provides a flexible framework for defining forward models, featurizing molecular dynamics (MD) ensembles, handling experimental data, and optimizing model parameters using JAX's automatic differentiation and JIT compilation capabilities.

## Core Concepts and Architecture

The library is structured around several key components that interact to form a complete modeling and optimization pipeline:

1.  **Molecular Dynamics (MD) Ensemble Handling**:
    *   Utilizes `MDAnalysis` for loading and manipulating MD trajectories and structures.
    *   `Experiment_Builder` (`jaxent/src/interfaces/builder.py`) acts as an orchestrator to load MD `Universe` objects and prepare them for featurization and simulation.
    *   `find_common_residues` (`jaxent/src/models/func/common.py`) ensures compatibility across ensemble members by identifying common protein residues.

2.  **Featurization**:
    *   `ForwardModel`s (`jaxent/src/types/base.py`) define how raw MD ensemble data is transformed into `Input_Features` (`jaxent/src/types/features.py`).
    *   **Best-Vendruscolo (BV) Model**:
        *   `BV_model` (`jaxent/src/models/HDX/BV/forwardmodel.py`) and `linear_BV_model` implement the BV model for HDX.
        *   Features include heavy atom and H-bond acceptor contacts (`BV_input_features` in `jaxent/src/models/HDX/BV/features.py`), calculated using `calc_BV_contacts_universe` (`jaxent/src/models/func/contacts.py`).
        *   Intrinsic exchange rates (`calculate_intrinsic_rates` in `jaxent/src/models/func/uptake.py`) are also incorporated.
    *   **Network-based HDX (netHDX) Model**:
        *   `netHDX_model` (`jaxent/src/models/HDX/netHDX/forwardmodel.py`) uses hydrogen bond networks.
        *   `build_hbond_network` (`jaxent/src/models/func/netHDX.py`) generates contact matrices and various graph-theoretic network metrics (e.g., degree, clustering coefficient, betweenness centrality, path lengths) as `NetHDX_input_features` (`jaxent/src/models/HDX/netHDX/features.py`).

3.  **Experimental Data Handling**:
    *   `ExpD_Datapoint` (`jaxent/src/data/loader.py`) is the base for experimental data points.
    *   `HDX_peptide` and `HDX_protection_factor` (`jaxent/src/types/HDX.py`) are concrete implementations for HDX-MS data.
    *   `ExpD_Dataloader` (`jaxent/src/data/loader.py`) loads and manages experimental datasets.
    *   The `jaxent/src/interfaces/topology/` directory now contains a comprehensive set of modules for handling molecular fragments and their relationships:
    *   `Partial_Topology` (`jaxent/src/interfaces/topology/core.py`) represents molecular fragments and is used to align experimental data with structural features.
    *   `TopologyFactory` (`jaxent/src/interfaces/topology/factory.py`) provides methods for creating and manipulating `Partial_Topology` objects.
    *   `PairwiseTopologyComparisons` (`jaxent/src/interfaces/topology/pairwise.py`) offers methods for comparing `Partial_Topology` objects.
    *   `mda_TopologyAdapter` (`jaxent/src/interfaces/topology/mda_adapter.py`) facilitates conversion between MDAnalysis Universe objects and `Partial_Topology` objects.
    *   `PTSerialiser` (`jaxent/src/interfaces/topology/serialise.py`) handles serialization and deserialization of `Partial_Topology` objects.
    *   `utils.py` (`jaxent/src/interfaces/topology/utils.py`) provides utility functions for topology handling.
    *   `create_sparse_map` (`jaxent/src/data/splitting/sparse_map.py`) creates a sparse mapping matrix to connect residue-wise features to experimental fragments, crucial for handling incomplete or peptide-level data.
    *   `DataSplitter` (`jaxent/src/data/splitting/split.py`) provides functionalities for splitting experimental data into training and validation sets (random, stratified, spatial, cluster-based splits).

4.  **Simulation and Optimization Core**:
    *   `Simulation` (`jaxent/src/models/core.py`) is the central object that encapsulates the `ForwardModel`s, `Input_Features`, and `Simulation_Parameters` (`jaxent/src/interfaces/simulation.py`). It manages the overall forward pass and its JIT compilation.
    *   `Simulation_Parameters` holds all trainable parameters of the system, including `frame_weights` (for ensemble reweighting), `model_parameters` (for individual forward models), and `forward_model_weights` (for combining multiple models).
    *   `Model_Parameters` (`jaxent/src/interfaces/model.py`) is a base class for defining model-specific trainable parameters as JAX PyTrees.
    *   `OptaxOptimizer` (`jaxent/src/opt/optimiser.py`) integrates with `optax` to provide various optimization algorithms (Adam, SGD, Adagrad, AdamW). It supports gradient masking to selectively optimize parameters.
    *   A rich set of `JaxEnt_Loss` functions (`jaxent/src/opt/losses.py`) are available, including:
        *   Mean Squared Error (MSE) and Mean Absolute Error (MAE) for protection factors and uptake.
        *   Maximum Entropy (MaxEnt) losses (KL divergence, L2 penalty) for frame weights, encouraging distributions similar to a prior or sparsity.
        *   Monotonicity loss for HDX uptake, ensuring physical consistency.
        *   Frame weight consistency losses (cosine, correlation, KL divergence) to align ensemble reweighting with structural similarity.
    *   `run_optimise` (`jaxent/src/opt/run.py`) orchestrates the entire optimization loop, applying chosen loss functions and updating parameters.
    *   `OptimizationHistory` (`jaxent/src/opt/base.py`) tracks the training and validation losses, and the state of parameters throughout the optimization process.
    *   `hdf.py` (`jaxent/src/utils/hdf.py`) provides utilities for saving and loading optimization history and parameters to/from HDF5 files for persistence.

## Key Workflows

1.  **Data Preparation**:
    *   Load MD trajectories and experimental HDX data.
    *   Identify common residues across the ensemble.
    *   Split experimental data into training and validation sets.
    *   Create sparse mappings to align experimental fragments with residue-wise features.

2.  **Feature Generation**:
    *   Apply chosen `ForwardModel`s (e.g., BV, netHDX) to the MD ensemble to compute `Input_Features` (e.g., contact maps, network metrics).

3.  **Model Setup**:
    *   Initialize `Simulation_Parameters` with initial guesses for frame weights, model parameters, and loss function weights/scaling.
    *   Construct a `Simulation` object, which combines the featurized data, forward models, and simulation parameters.

4.  **Optimization**:
    *   Define an `OptaxOptimizer` with desired learning rate, algorithm, and parameter masks (to specify which parameters are trainable).
    *   Select one or more `JaxEnt_Loss` functions, potentially with different scaling factors, to guide the optimization.
    *   Run the `run_optimise` function, which iteratively updates the `Simulation_Parameters` to minimize the combined loss.
    *   The optimization process leverages JAX's JIT compilation for performance and automatic differentiation for gradient computation.

5.  **Analysis and Visualization**:
    *   The `analysis/plots/optimisation.py` module provides tools to visualize the training progress, including loss curves and the evolution of frame weights.
    *   Integration tests demonstrate plotting of network metrics and H-bond networks, indicating capabilities for post-optimization analysis.
