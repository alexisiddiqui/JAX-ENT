# Architecture Plan: Multi-Modal Experimental Data Support (SAXS, XL-MS)

## Context

JAX-ENT currently supports only HDX-MS data (peptide-level deuterium exchange, residue-level protection factors). To become a general ensemble reweighting toolkit, it needs to support SAXS and XL-MS (cross-linking mass spectrometry). These data types have different topology structures and output-to-observation mappings, but all use the same execution model: **average features across frames, then apply the forward model**.

Any nonlinear observable (e.g., NMR r^-6 averaging) is handled by choosing what to featurise per frame (e.g., featurise 1/r^6 directly), not by changing the forward pass ordering. No `ForwardPassMode` enum is needed.

### Data type summary

| Type | Topology | Features (per frame) | Forward model | Mapping |
|------|----------|---------------------|--------------|---------|
| HDX-MS (existing) | Peptide/residue fragments (2D seq) | Contacts per residue | BV/netHDX -> log(Pf) or uptake | Sparse (n_peptides, n_residues) |
| SAXS 1D (pre-computed) | Whole construct | I(q) curves | Identity/scaling | Identity |
| SAXS 2D (Debye) | Whole construct | Pairwise distances | Debye formula -> I(q) | Identity |
| XL-MS | Residue pairs | Pairwise distances | Identity/distance extraction | PairIndex (i,j) -> scalar |

---

## Step 1: DataMapping Protocol and Implementations

**New file:** `jaxent/src/data/splitting/mapping.py`

Generalises the current sparse matrix mapping into a protocol hierarchy:

```python
@runtime_checkable
class DataMapping(Protocol):
    """Protocol for mapping model outputs to observation space."""
    def apply(self, predictions: Array) -> Array: ...

@dataclass(frozen=True, slots=True)
class SparseFragmentMapping:
    """Wraps existing BCOO sparse map (HDX). Shape: (n_fragments, n_residues)."""
    sparse_map: sparse.BCOO
    def apply(self, predictions: Array) -> Array:
        return apply_sparse_mapping(self.sparse_map, predictions)

@dataclass(frozen=True, slots=True)
class IdentityMapping:
    """No-op mapping for whole-construct outputs (SAXS)."""
    def apply(self, predictions: Array) -> Array:
        return predictions

@dataclass(frozen=True, slots=True)
class PairIndexMapping:
    """Extracts specific residue pairs from a pairwise distance matrix (XL-MS).
    indices_i, indices_j: arrays of shape (n_observations,)
    Each observation maps to predictions[indices_i[k], indices_j[k]]."""
    indices_i: Array
    indices_j: Array
    def apply(self, predictions: Array) -> Array:
        return predictions[self.indices_i, self.indices_j]
```

Register `SparseFragmentMapping` and `PairIndexMapping` as JAX pytree nodes (their array fields are dynamic). `IdentityMapping` is stateless but register for consistency.

**Reuses:** `apply_sparse_mapping` from `jaxent/src/data/splitting/sparse_map.py`

---

## Step 2: New Experimental Datapoint Types

**New file:** `jaxent/src/custom_types/SAXS.py`

```python
@dataclass()
class SAXS_curve(ExpD_Datapoint):
    """SAXS I(q) scattering profile for a whole construct."""
    key: ClassVar[m_key] = m_key("SAXS_Iq")
    top: Partial_Topology  # whole-construct topology
    intensities: ndarray   # I(q) values
    q_values: ndarray      # scattering vector values
    errors: ndarray | None = None

    def extract_features(self) -> np.ndarray:
        return self.intensities

    @classmethod
    def _create_from_features(cls, topology, features): ...
```

**New file:** `jaxent/src/custom_types/XLMS.py`

```python
@dataclass()
class XLMS_distance_restraint(ExpD_Datapoint):
    """XL-MS cross-link distance restraint between two residues."""
    key: ClassVar[m_key] = m_key("XLMS_distance")
    top: Partial_Topology    # first residue topology (from base class)
    top_j: Partial_Topology  # second residue topology (pairwise)
    distance: float          # observed/maximum distance
    lower_bound: float | None = None
    upper_bound: float | None = None

    def extract_features(self) -> np.ndarray:
        return np.array([self.distance])

    @classmethod
    def _create_from_features(cls, topology, features): ...
```

**Follows pattern of:** `jaxent/src/custom_types/HDX.py` (HDX_peptide, HDX_protection_factor)

---

## Step 3: Dataset and Dataloader Generalisation

**File:** `jaxent/src/data/loader.py`

Update `Dataset` to accept generalised mapping while preserving backwards compatibility:

```python
@partial(
    jax.tree_util.register_dataclass,
    data_fields=["y_true", "data_mapping", "covariance_matrix"],
    meta_fields=["data"],
)
@dataclass(frozen=True, slots=True)
class Dataset:
    data: Sequence[ExpDDatapointLike]
    y_true: Float[Array, "n_fragments ..."]
    data_mapping: DataMapping  # generalised - replaces residue_feature_ouput_mapping
    covariance_matrix: Float[Array, "n_fragments n_fragments"] | None = None

    @property
    def residue_feature_ouput_mapping(self):
        """Backwards compat: returns BCOO if mapping is SparseFragmentMapping."""
        if isinstance(self.data_mapping, SparseFragmentMapping):
            return self.data_mapping.sparse_map
        raise AttributeError("This dataset does not use sparse fragment mapping")
```

Update `ExpD_Dataloader.create_datasets()`:
- Accept optional `mapping_factory: Callable | None = None` parameter
- Default behaviour: wraps `create_sparse_map()` result in `SparseFragmentMapping`
- Custom factory receives `(features, feature_topology, data_split)` and returns a `DataMapping`
- Pass `data_mapping=...` instead of `residue_feature_ouput_mapping=...` when constructing `Dataset`

**File:** `jaxent/src/custom_types/protocols.py`

Update `DatasetLike`:

```python
@runtime_checkable
class DatasetLike(Protocol):
    y_true: Array
    data_mapping: Any  # DataMapping protocol
```

---

## Step 4: Simulation outputs_by_key and model_key_index

**File:** `jaxent/src/models/core.py`

Add `outputs_by_key` property to `Simulation`:

```python
@property
def outputs_by_key(self) -> dict[m_key, Output_Features]:
    return {output.key: output for output in self.outputs}
```

No changes to `__init__`, `forward_pure`, or `forward` — the existing execution path handles all models since they all use `AVERAGE_THEN_FORWARD`.

**File:** `jaxent/src/interfaces/simulation.py`

Add `model_key_index` to `Simulation_Parameters` as a static field:

```python
@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Float[Array, " n_frames"]
    frame_mask: ...
    model_parameters: Sequence[Model_Parameters]
    forward_model_weights: Float[Array, " n_models"]
    normalise_loss_functions: ...
    forward_model_scaling: Float[Array, " n_models"]
    model_key_index: tuple[m_key, ...] | None = None  # NEW - static

    def get_model_params_by_key(self, key: m_key) -> Model_Parameters:
        if self.model_key_index is None:
            raise ValueError("model_key_index not set")
        idx = self.model_key_index.index(key)
        return self.model_parameters[idx]
```

Update `tree_flatten` to include `model_key_index` in `static` tuple.
Update `tree_unflatten` to restore it.
Update `param_labels`, `normalize_weights`, `normalize_masked_loss_scalingweights`, `propagate_model_parameters` to pass through `model_key_index`.

**File:** `jaxent/src/custom_types/protocols.py`

Update `SimulationLike`:

```python
@runtime_checkable
class SimulationLike(Protocol):
    params: Simulation_Parameters
    outputs: Sequence[Output_Features]
    outputs_by_key: dict[m_key, Output_Features]
```

---

## Step 5: Loss Routing Generalisation

**File:** `jaxent/src/opt/loss/base.py`

Generalise `create_functional_loss` to use `y_pred()` and `DataMapping.apply()`:

```python
def create_functional_loss(loss_fn, transform_chain=None, post_mean=True, flatten=True):
    def functional_loss(model, dataset, prediction_index):
        # Support both int and m_key indexing
        if isinstance(prediction_index, str):
            predictions = model.outputs_by_key[m_key(prediction_index)]
        else:
            predictions = model.outputs[prediction_index]

        def compute_loss(data_mapping, y_true):
            pred_values = predictions.y_pred()
            if flatten:
                pred_mapped = data_mapping.apply(pred_values.reshape(-1))
                true_values = y_true.reshape(-1)
                pred_transformed = apply_transforms(pred_mapped, transform_chain)
                true_transformed = apply_transforms(true_values, transform_chain)
                loss = loss_fn(pred_transformed, true_transformed)
                return apply_post_processing(loss, true_values.shape[0], post_mean)
            else:
                # Multi-timepoint (HDX uptake): iterate over first axis
                total_loss = 0.0
                for t in range(y_true.shape[0]):
                    pred_t = data_mapping.apply(pred_values[t])
                    pred_t = apply_transforms(pred_t, transform_chain)
                    true_t = apply_transforms(y_true[t], transform_chain)
                    total_loss += loss_fn(pred_t, true_t)
                return apply_post_processing(total_loss, y_true.shape[0], post_mean)

        train_loss = compute_loss(dataset.train.data_mapping, dataset.train.y_true)
        val_loss = compute_loss(dataset.val.data_mapping, dataset.val.y_true)
        return train_loss, val_loss
    return functional_loss
```

Key changes vs current (`jaxent/src/opt/loss/base.py:120-169`):
- Uses `predictions.y_pred()` instead of hardcoded `.log_Pf` / `.uptake`
- Uses `data_mapping.apply()` instead of direct `apply_sparse_mapping()`
- Supports `m_key` indexing via `outputs_by_key`

**Backwards compatibility:** Existing loss functions in `opt/losses.py` that access `.log_Pf` directly continue to work for HDX. Migrate them to `create_functional_loss` + `y_pred()` gradually.

---

## Step 6: New Forward Model Stubs (architecture only)

**New directory:** `jaxent/src/models/SAXS/`
- `__init__.py`
- `config.py`: `SAXS_Debye_Config(BaseConfig)` with `q_values` array, key = `m_key("SAXS_Iq")`
- `parameters.py`: `SAXS_Model_Parameters(Model_Parameters)` with `scaling_factor`, `background` (both dynamic)
- `features.py`:
  - `SAXS_pairwise_input_features(Input_Features)`: pairwise distances `(n_residues, n_residues, n_frames)` for Debye 2D mode
  - `SAXS_curve_input_features(Input_Features)`: pre-computed I(q) curves `(n_q, n_frames)` for 1D mode
  - `SAXS_output_features(Output_Features)`: intensity array `(n_q,)` with `y_pred()` returning intensity
- `forwardmodel.py`: `SAXS_Debye_model(ForwardModel)`
- `forward.py`:
  - `SAXS_Debye_ForwardPass` -- Debye formula on ensemble-averaged pairwise distances
  - `SAXS_identity_ForwardPass` -- scaling/background correction on pre-averaged I(q) curves

**New directory:** `jaxent/src/models/XLMS/`
- `__init__.py`
- `config.py`: `XLMS_Config(BaseConfig)` with key = `m_key("XLMS_distance")`
- `parameters.py`: `XLMS_Model_Parameters(Model_Parameters)` (minimal -- may just be scaling)
- `features.py`:
  - `XLMS_input_features(Input_Features)`: pairwise distances `(n_residues, n_residues, n_frames)`
  - `XLMS_output_features(Output_Features)`: distance matrix `(n_residues, n_residues)` with `y_pred()` returning the matrix for `PairIndexMapping` to extract pairs from
- `forwardmodel.py`: `XLMS_distance_model(ForwardModel)`
- `forward.py`: `XLMS_distance_ForwardPass` -- extracts ensemble-averaged pairwise distance matrix

**Follows pattern of:** `jaxent/src/models/HDX/BV/` (forwardmodel.py, features.py, parameters.py, forward.py)

---

## Implementation Order

| Phase | Description | Risk | Files |
|-------|-------------|------|-------|
| 1 | `DataMapping` protocol + implementations | Low | new `mapping.py` |
| 2 | SAXS/XLMS datapoint types | Low | new `SAXS.py`, `XLMS.py` |
| 3 | `Dataset` generalisation + `DatasetLike` update | Medium | `loader.py`, `protocols.py` |
| 4 | `outputs_by_key` + `model_key_index` + `SimulationLike` | Medium | `core.py`, `simulation.py`, `protocols.py` |
| 5 | Loss routing generalisation (`create_functional_loss`) | Medium | `opt/loss/base.py` |
| 6 | Model stubs (SAXS, XLMS) | Low | new `models/SAXS/`, `models/XLMS/` |

No changes to `Simulation.forward_pure` -- all models use the existing average-then-forward path.

---

## Verification Plan

1. **Existing HDX tests pass unchanged** after each phase (run `pytest` after every phase)
2. **Unit test `DataMapping`**: verify `SparseFragmentMapping.apply()` matches existing `apply_sparse_mapping()`, `IdentityMapping` is identity, `PairIndexMapping` extracts correct pairs from a known distance matrix
3. **Unit test `Dataset` backwards compat**: verify `dataset.residue_feature_ouput_mapping` property works when `data_mapping` is `SparseFragmentMapping`
4. **Unit test `outputs_by_key`**: create `Simulation` with known outputs, verify dict access by `m_key`
5. **Unit test `model_key_index`**: verify `get_model_params_by_key()` returns correct parameters
6. **Unit test generalised `create_functional_loss`**: verify loss computation produces same results as legacy when using `SparseFragmentMapping` + HDX output features
7. **Integration test**: construct `Simulation` with mixed HDX + mock SAXS models, shared frame weights, verify both forward passes execute and outputs accessible by key
8. **JIT compilation test**: verify the simulation JIT-compiles with multiple model types
