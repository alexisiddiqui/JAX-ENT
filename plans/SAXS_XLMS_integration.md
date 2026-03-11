# Architecture Plan: Multi-Modal Experimental Data Support (SAXS, XL-MS)

## Context

JAX-ENT currently supports only HDX-MS data (peptide-level deuterium exchange, residue-level protection factors). To become a general ensemble reweighting toolkit, it needs to support SAXS and XL-MS (cross-linking mass spectrometry). These data types have different topology structures and output-to-observation mappings, but all use the same execution model: **average features across frames, then apply the forward model**.

Any nonlinear observable (e.g., NMR r^-6 averaging) is handled by choosing what to featurise per frame (e.g., featurise 1/r^6 directly), not by changing the forward pass ordering.

### Current state of the codebase

- **HDX-only today.** There is no existing SAXS support in `jaxent/src/custom_types/` or `jaxent/src/models/`. There is no existing XL-MS support either.
- `Dataset` (`jaxent/src/data/loader.py`) is registered as a JAX pytree dataclass with `residue_feature_ouput_mapping: sparse.BCOO` as a data field. Many tests instantiate `Dataset(...)` directly with this kwarg.
- `ExpD_Dataloader.create_datasets()` hardcodes `create_sparse_map()` calls for HDX residue-to-peptide mapping.
- `create_functional_loss` in `jaxent/src/opt/loss/base.py` hardcodes `.log_Pf` and `.uptake` attribute access. Additionally, `jaxent/src/opt/losses.py` contains many HDX-specific loss functions that access these attributes directly.
- The system is *partially* generic (ForwardModel protocol, m_key dispatch, Input/Output_Features base classes) but the data layer and loss layer are HDX-specific.

### Data type summary

| Type | Topology | Features (per frame) | Forward model | Mapping |
|------|----------|---------------------|--------------|---------|
| HDX-MS (existing) | Peptide fragments (2D seq) | Contacts per residue `(n_res, n_frames)` | BV/netHDX -> log(Pf) or uptake | Sparse (n_peptides, n_residues) |
| SAXS (Debye 6-term) | Whole construct | 6 basis profiles `(6, n_q, n_frames)` | Cross-term formula with c1, c2, c, b | Identity |
| XL-MS | Residue pairs | Pairwise distances `(n_res, n_res, n_frames)` | Identity/distance extraction | PairIndex (i,j) -> scalar |

---

## Step 1: DataMapping Protocol and Implementations

**New file:** `jaxent/src/data/splitting/mapping.py`

Generalises the current sparse matrix mapping into a protocol hierarchy:

```python
@runtime_checkable
class DataMapping(Protocol):
    """Protocol for mapping model outputs to observation space."""
    def apply(self, predictions: Array) -> Array: ...

@partial(jax.tree_util.register_dataclass, data_fields=["sparse_map"], meta_fields=[])
@dataclass(frozen=True, slots=True)
class SparseFragmentMapping:
    """Wraps existing BCOO sparse map (HDX). Shape: (n_fragments, n_residues)."""
    sparse_map: sparse.BCOO
    def apply(self, predictions: Array) -> Array:
        return apply_sparse_mapping(self.sparse_map, predictions)

@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
@dataclass(frozen=True, slots=True)
class IdentityMapping:
    """No-op mapping for whole-construct outputs (SAXS)."""
    def apply(self, predictions: Array) -> Array:
        return predictions

@partial(jax.tree_util.register_dataclass, data_fields=["indices_i", "indices_j"], meta_fields=[])
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

Note: Using `jax.tree_util.register_dataclass` avoids writing boilerplate `tree_flatten`/`tree_unflatten` functions while successfully registering these mapping protocol instances as JAX pytree nodes.

**Also add `create_pair_index_mapping()` factory** in the same file (analogous to `create_sparse_map()` for HDX):

```python
def create_pair_index_mapping(
    feature_topology: Sequence[Partial_Topology],
    xlms_datapoints: Sequence[XLMS_distance_restraint],
) -> PairIndexMapping:
    """Build a PairIndexMapping from XL-MS datapoints and feature topology.

    Maps each datapoint's (top, top_j) pair to indices into the feature
    topology's residue ordering.
    """
    # For each datapoint, find the index of top and top_j in feature_topology
    # Returns PairIndexMapping(indices_i, indices_j)
    ...
```

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

**Update:** `jaxent/src/custom_types/__init__.py` to export the new types.

**Follows pattern of:** `jaxent/src/custom_types/HDX.py` (HDX_peptide, HDX_protection_factor)

---

## Step 3: Dataset and Dataloader Generalisation

> Do not try to hack backwards compatibility into the data structure itself.
> Keep the `Dataset` class simple and pure: `data_mapping` is the only field.
> Perform a clean, localized "find-and-replace" refactor across the test suite, changing `Dataset(residue_feature_ouput_mapping=...)` to `Dataset(data_mapping=SparseFragmentMapping(sparse_map=...))` to keep the core architecture explicitly clean.

**File:** `jaxent/src/data/loader.py`

The current `Dataset` registers `residue_feature_ouput_mapping` as a JAX dataclass data field, and tests construct `Dataset(...)` directly with this kwarg. The approach is to add `data_mapping` as the new generalised field and simply remove the legacy path.

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
    data_mapping: DataMapping  # generalised mapping
    covariance_matrix: Float[Array, "n_fragments n_fragments"] | None = None
```

Update `ExpD_Dataloader.create_datasets()`:
- Accept optional `mapping_factory: Callable | None = None` parameter
- Default behaviour: wraps `create_sparse_map()` result in `SparseFragmentMapping` (HDX path unchanged)
- Custom factory receives `(features, feature_topology, data_split)` and returns a `DataMapping`
- Pass `data_mapping=...` when constructing `Dataset`

**File:** `jaxent/src/custom_types/protocols.py`

Update `DatasetLike` to use generalised mapping:

```python
@runtime_checkable
class DatasetLike(Protocol):
    y_true: Array
    data_mapping: Any  # DataMapping protocol
```

---

## Step 4: Simulation outputs_by_key

**File:** `jaxent/src/models/core.py`

Add `outputs_by_key` property to `Simulation` (low-risk convenience layer):

```python
@property
def outputs_by_key(self) -> dict[m_key, Output_Features]:
    return {output.key: output for output in self.outputs}
```

No changes to `__init__`, `forward_pure`, or `forward` -- the existing execution path handles all models.

**File:** `jaxent/src/custom_types/protocols.py`

Update `SimulationLike`:

```python
@runtime_checkable
class SimulationLike(Protocol):
    params: Simulation_Parameters
    outputs: Sequence[Output_Features]
    outputs_by_key: dict[m_key, Output_Features]
```

### Deferred: model_key_index on Simulation_Parameters

Persisting `model_key_index` inside `Simulation_Parameters` is deferred from the first implementation pass. `Simulation_Parameters` is a JAX pytree and changing its static/dynamic layout has wider implications for optax multi_transform labelling, gradient masking, and all static methods (`normalize_weights`, `param_labels`, etc.). Int-based indexing via `indexes` in `run_optimise` is sufficient for the initial mixed-model workflows. If real mixed-model usage reveals int-indexing is insufficient, `model_key_index` can be added as a follow-up.

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

### Legacy loss audit

`jaxent/src/opt/losses.py` contains many HDX-specific loss functions (`hdx_pf_l2_loss`, `hdx_pf_l1_loss`, `hdx_pf_kld_loss`, etc.) that directly access `.log_Pf` and call `apply_sparse_mapping`. These remain HDX-specific and continue to work unchanged. After `create_functional_loss` is generalised:
- Audit `opt/losses.py` to identify which losses are truly HDX-specific vs. generic
- Generic losses (L2, L1, KLD on mapped predictions) should be cloned to use `create_functional_loss` + `y_pred()` in `jaxent/src/opt/loss/`
- 'jaxent/src/opt/losses.py' should stay as is as these are currently used in the examples - the examples will be moved over to a functional loss style after verificatoin that everything is working as expected.

---

## Step 6: New Forward Models

### SAXS: Six Cross-Term Decomposition

The Debye SAXS forward model decomposes into offline O(N^2) precomputation + O(1) online optimization:

**Precomputation (offline, per frame k):** Compute 6 basis intensity profiles from atomic form factors and coordinates:
- `I_vv,k(q)` = vacuum-vacuum
- `I_ve,k(q)` = vacuum-excluded
- `I_vh,k(q)` = vacuum-hydration
- `I_ee,k(q)` = excluded-excluded
- `I_eh,k(q)` = excluded-hydration
- `I_hh,k(q)` = hydration-hydration

Output tensor per trajectory: `(n_frames, 6, n_q)`

**Online forward model:** Ensemble-average the 6 basis profiles using frame weights w_k, then combine with optimizable parameters c1, c2, c, b:

```
<I_xx(q)> = sum_k w_k * I_xx,k(q)     # frame averaging (existing machinery)

I_ens(q) = <I_vv> - 2*c1*<I_ve> + 2*c2*<I_vh> + c1^2*<I_ee> - 2*c1*c2*<I_eh> + c2^2*<I_hh>

I_calc(q) = c * I_ens(q) + b           # scale + background
```

This is scalar algebra on averaged features -- O(1) per q-point during optimization.

**New directory:** `jaxent/src/models/SAXS/`
- `__init__.py`
- `config.py`: `SAXS_Config(BaseConfig)` with `q_values` array, key = `m_key("SAXS_Iq")`
- `parameters.py`: `SAXS_Model_Parameters(Model_Parameters)`
  - Dynamic (optimizable): `c1` (excluded volume), `c2` (hydration), `c` (scale), `b` (background)
  - Static: `key`
- `features.py`:
  - `SAXS_basis_input_features(Input_Features)`: 6 basis profiles, shape `(6, n_q, n_frames)`. `__features__ = {"basis_profiles"}`. After frame averaging produces `(6, n_q)`.
  - `SAXS_output_features(Output_Features)`: intensity array `(n_q,)` with `y_pred()` returning `intensity`. `key = m_key("SAXS_Iq")`
- `forwardmodel.py`: `SAXS_model(ForwardModel)` -- dispatches to appropriate forward pass
- `forward.py`:
  - `SAXS_ForwardPass` -- takes averaged `(6, n_q)` basis profiles + `SAXS_Model_Parameters(c1, c2, c, b)`, applies the cross-term formula, returns `SAXS_output_features(intensity)`

### XL-MS

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
| 1 | `DataMapping` protocol + implementations + `create_pair_index_mapping` | Low | new `mapping.py` |
| 2 | SAXS/XLMS datapoint types + `__init__.py` exports | Low | new `SAXS.py`, `XLMS.py`, `custom_types/__init__.py` |
| 3 | `Dataset` generalisation (additive) + `DatasetLike` update | Medium | `loader.py`, `protocols.py` |
| 4 | `outputs_by_key` + `SimulationLike` update | Low | `core.py`, `protocols.py` |
| 5 | Loss routing generalisation (`create_functional_loss`) + audit `losses.py` | Medium | `opt/loss/base.py`, `opt/losses.py` (audit only) |
| 6 | Model stubs (SAXS, XLMS) | Low | new `models/SAXS/`, `models/XLMS/` |

No changes to `Simulation.forward_pure` -- all models use the existing average-then-forward path.
`model_key_index` on `Simulation_Parameters` is deferred unless int-based indexing proves insufficient.

---

## Files touched (complete list)

**New files:**
- `jaxent/src/data/splitting/mapping.py` -- DataMapping protocol, SparseFragmentMapping, IdentityMapping, PairIndexMapping, create_pair_index_mapping
- `jaxent/src/custom_types/SAXS.py` -- SAXS_curve datapoint
- `jaxent/src/custom_types/XLMS.py` -- XLMS_distance_restraint datapoint
- `jaxent/src/models/SAXS/__init__.py`, `config.py`, `parameters.py`, `features.py`, `forwardmodel.py`, `forward.py`
- `jaxent/src/models/XLMS/__init__.py`, `config.py`, `parameters.py`, `features.py`, `forwardmodel.py`, `forward.py`

**Modified files:**
- `jaxent/src/custom_types/__init__.py` -- export new types
- `jaxent/src/custom_types/protocols.py` -- update DatasetLike, SimulationLike
- `jaxent/src/data/loader.py` -- Dataset generalisation (additive), ExpD_Dataloader.create_datasets mapping_factory
- `jaxent/src/models/core.py` -- add outputs_by_key property
- `jaxent/src/opt/loss/base.py` -- generalise create_functional_loss

**Test files to add/update:**
- `jaxent/tests/unit/data/test_mapping.py` -- DataMapping implementations
- `jaxent/tests/unit/opt/` -- loss regression tests
- `jaxent/tests/integration/optimise/` -- mixed-model integration tests
- Existing tests under `jaxent/tests/losses/` -- verify HDX losses unchanged

---

## Verification Plan

1. **Existing HDX tests pass unchanged** after each phase (`pytest` after every phase - make sure you use the .venv and try to run targeted tests as it takes 4-5 minutes to run all key tests)
2. **Unit test `DataMapping`**: verify `SparseFragmentMapping.apply()` matches existing `apply_sparse_mapping()`, `IdentityMapping` is identity, `PairIndexMapping` extracts correct pairs from a known distance matrix
3. **Unit test `create_pair_index_mapping`**: verify factory builds correct index arrays from XL-MS datapoints + feature topology
4. **Update existing tests**: perform find-and-replace to migrate existing `Dataset(residue_feature_ouput_mapping=...)` constructions to use `data_mapping=SparseFragmentMapping(sparse_map=...)`.
5. **Regression tests for direct `Dataset(...)` construction** used throughout the existing test suite
6. **Unit test `outputs_by_key`**: create `Simulation` with known outputs, verify dict access by `m_key`
7. **Unit test generalised `create_functional_loss`**: verify loss computation produces same results as legacy when using `SparseFragmentMapping` + HDX output features
8. **Unit test Data Splitting**: Verify that the datasplitter can split SAXS and XL-MS data with random and stratified splitting strategies using simple synthetic data. Will test the other splits later.
9. **Regression tests for existing HDX loss functions** in `opt/losses.py` after generic loss refactor.
10. **Integration test**: construct `Simulation` with mixed HDX + mock SAXS models, shared frame weights, verify both forward passes execute and outputs accessible by key
11. **JIT compilation test**: verify the simulation JIT-compiles with multiple model types
