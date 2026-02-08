# jaxtyping & chex Integration Research — jax-ent Codebase

## Executive Summary

This document identifies every opportunity to use **jaxtyping** (shape-annotated array types) and **chex** (JAX-specific testing/assertion utilities) across the jax-ent codebase. The codebase already uses `beartype_this_package()` for runtime checking and imports `chex` in `opt/base.py`, so both libraries integrate naturally.

**Key findings:**
- 1 critical bug (P0) that `chex` would have prevented
- 6 core data structures with bare `Array` types that should have shape annotations
- 20+ functions where shape annotations would document and enforce implicit contracts
- 15+ locations where `assert` should be replaced with `chex` assertions (JIT-safe, not stripped in `-O` mode)
- 4 dataclasses that could migrate to `chex.dataclass` for automatic pytree registration

---

## Critical Bug — P0

### `opt/run.py` line 9638: NaN comparison always False

```python
# BUG: NaN != NaN in IEEE 754, so this is ALWAYS False
if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):
```

**Fix with chex:**
```python
# Option A: Direct replacement
if (current_loss < tolerance) or jnp.isnan(current_loss) or jnp.isinf(current_loss):

# Option B: chex assertion earlier in the loop (catch the root cause)
chex.assert_tree_all_finite(opt_state.losses)  # Fails at the source, not downstream
```

**Impact:** Optimization silently continues through NaN loss values, producing garbage results. This is the highest-priority fix.

---

## Part 1: jaxtyping Shape Annotations

### How it integrates with the existing setup

The codebase already uses `beartype_this_package()` in `__init__.py`. jaxtyping integrates directly:

```python
# In __init__.py — add one import
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

# Then on individual functions:
@jaxtyped(typechecker=typechecker)
def my_function(x: Float[Array, "n d"]) -> Float[Array, "n"]:
    ...
```

For gradual adoption, annotations alone (without the decorator) still serve as documentation and are checked by mypy/pyright with the jaxtyping plugin.

---

### 1.1 `Simulation_Parameters` — `interfaces/simulation.py` (line 4128)

**Current:**
```python
@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Array
    frame_mask: Array                          # "array of type int"
    model_parameters: Sequence[Model_Parameters]
    forward_model_weights: Array
    normalise_loss_functions: Array             # "array of type int"
    forward_model_scaling: Array
```

**Proposed:**
```python
from jaxtyping import Float, Int, Array

@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Float[Array, " n_frames"]
    frame_mask: Float[Array, " n_frames"]               # sigmoid-smoothed, not int despite comment
    model_parameters: Sequence[Model_Parameters]
    forward_model_weights: Float[Array, " n_models"]
    normalise_loss_functions: Int[Array, " n_models"]    # actual integer mask
    forward_model_scaling: Float[Array, " n_models"]
```

**Why this matters:** `frame_weights` and `frame_mask` must be the same length (they're zipped in `normalize_weights`). `forward_model_weights`, `normalise_loss_functions`, and `forward_model_scaling` must all be the same length as `model_parameters`. These shape names make those contracts explicit.

The comment says `frame_mask` is "array of type int" but `normalize_weights()` applies sigmoid to it (`Float`). The annotation would surface this inconsistency.

---

### 1.2 `LossComponents` — `opt/base.py` (line 8736)

**Current:**
```python
class LossComponents(NamedTuple):
    train_losses: Array
    val_losses: Array
    scaled_train_losses: Array
    scaled_val_losses: Array
    total_train_loss: Array
    total_val_loss: Array
```

**Proposed:**
```python
class LossComponents(NamedTuple):
    train_losses: Float[Array, " n_models"]
    val_losses: Float[Array, " n_models"]
    scaled_train_losses: Float[Array, " n_models"]
    scaled_val_losses: Float[Array, " n_models"]
    total_train_loss: Float[Array, ""]         # scalar
    total_val_loss: Float[Array, ""]           # scalar
```

**Why:** `compute_loss` (line 9509) constructs these by zipping over `loss_functions` and applying `forward_model_weights * forward_model_scaling`. The shape `n_models` ties them back to `Simulation_Parameters`.

---

### 1.3 `BV_input_features` — `models/HDX/BV/features.py` (line 7756)

**Current:**
```python
class BV_input_features(Input_Features):
    heavy_contacts: Sequence[Sequence[float]] | Array | ndarray  # (frames, residues)
    acceptor_contacts: Sequence[Sequence[float]] | Array | ndarray  # (frames, residues)
    k_ints: Optional[list] | Optional[Array] | Optional[ndarray] = None  # (residues,)
```

**Problems:**
1. The comment says `(frames, residues)` but `features_shape` returns `(n_residues, n_frames)` — the actual layout is `(residues, frames)`.
2. The 6-way union type (`Sequence[Sequence[float]] | Array | ndarray`) is sprawling and unhelpful.
3. After `cast_to_jax()`, these are always `Array`, so post-initialization they should be typed as such.

**Proposed:**
```python
class BV_input_features(Input_Features):
    heavy_contacts: Float[Array, "n_residues n_frames"]
    acceptor_contacts: Float[Array, "n_residues n_frames"]
    k_ints: Float[Array, " n_residues"] | None = None
```

**Why:** Eliminates the documented dimension confusion. The featuriser (`BV_model.featurise`, line 8056) concatenates along axis=1 (frames) and the result has shape `(n_residues, total_frames)`. Making this explicit prevents silent transpose bugs.

**Migration note:** The union type exists because features arrive as lists from MDAnalysis and get converted to JAX arrays. The conversion happens in `tree_flatten()` / `cast_to_jax()`. After conversion, only the `Array` type is relevant. Consider a factory method or `__post_init__` conversion instead of supporting lists in the type.

---

### 1.4 `BV_output_features` — `models/HDX/BV/features.py` (line 7788)

**Current:**
```python
class BV_output_features(Output_Features):
    log_Pf: list | Sequence[float] | Array | ndarray  # (1, residues)
    k_ints: Optional[list] | Optional[Array] | Optional[ndarray] = None
```

**Proposed:**
```python
class BV_output_features(Output_Features):
    log_Pf: Float[Array, " n_residues"]
    k_ints: Float[Array, " n_residues"] | None = None
```

---

### 1.5 `uptake_BV_output_features` — `models/HDX/BV/features.py` (line 7806)

**Current:**
```python
class uptake_BV_output_features(Output_Features):
    uptake: (
        list[list[list[float]]]
        | Sequence[Sequence[Sequence[float]]]
        | list[list[float]]
        | Sequence[Sequence[float]]
        | Array
        | ndarray
    )  # (batch, peptides, timepoints) or (peptides, timepoints)
```

**Proposed:**
```python
class uptake_BV_output_features(Output_Features):
    uptake: Float[Array, "n_timepoints n_residues"]
```

**Why:** The 6-type union with ambiguous dimensionality (`(batch, peptides, timepoints)` OR `(peptides, timepoints)`) is a source of downstream bugs. The `BV_uptake_ForwardPass.__call__` (line 7700) returns `vmap(compute_uptake_for_timepoint)(time_points)` which produces shape `(n_timepoints, n_residues)`. Making this explicit prevents shape mismatches in the loss functions that index `predictions.uptake[timepoint_idx]`.

---

### 1.6 `BV_Model_Parameters` — `models/HDX/BV/parameters.py` (line 8308)

**Current:**
```python
class BV_Model_Parameters(Model_Parameters):
    bv_bc: Array = field(default_factory=lambda: jnp.array([0.35]))
    bv_bh: Array = field(default_factory=lambda: jnp.array([2.0]))
    temperature: float = 300.0
    timepoints: Array = field(default_factory=lambda: jnp.array([0.167, 1.0, 10.0]))
```

**Proposed:**
```python
class BV_Model_Parameters(Model_Parameters):
    bv_bc: Float[Array, " n_timepoints"]
    bv_bh: Float[Array, " n_timepoints"]
    temperature: float = 300.0
    timepoints: Float[Array, " n_timepoints"]
```

**Why:** `bv_bc`, `bv_bh`, and `timepoints` must all be the same length (they're zipped in the forward pass). `BV_model_Config.__post_init__` (line 5596) asserts this at construction time, but a shape annotation catches it at the type level.

---

### 1.7 `linear_BV_Model_Parameters` — `models/HDX/BV/parameters.py` (line 8360)

Same pattern as above:
```python
class linear_BV_Model_Parameters(Model_Parameters):
    bv_bc: Float[Array, " n_timepoints"]
    bv_bh: Float[Array, " n_timepoints"]
    temperature: float = 300.0
    timepoints: Float[Array, " n_timepoints"]
```

---

### 1.8 `NetHDX_Model_Parameters` — `models/HDX/netHDX/parameters.py` (line 8656)

**Proposed:**
```python
class NetHDX_Model_Parameters(Model_Parameters):
    shell_energy_scaling: float = 0.84
    temperature: float = 300.0
    timepoints: Float[Array, " n_timepoints"]
```

---

### 1.9 `Dataset` — `data/loader.py` (line 2091)

**Current:**
```python
@dataclass(frozen=True, slots=True)
class Dataset:
    data: Sequence[ExpDDatapointLike]
    y_true: Array
    residue_feature_ouput_mapping: sparse.BCOO
    covariance_matrix: Array | None = None
```

**Proposed:**
```python
@dataclass(frozen=True, slots=True)
class Dataset:
    data: Sequence[ExpDDatapointLike]
    y_true: Float[Array, "n_fragments ..."]       # (n_fragments,) or (n_timepoints, n_fragments)
    residue_feature_ouput_mapping: sparse.BCOO     # shape (n_fragments, n_residues)
    covariance_matrix: Float[Array, "n_fragments n_fragments"] | None = None
```

**Why:** `y_true` shape varies between protection factors `(n_fragments,)` and uptake data `(n_timepoints, n_fragments)`. This is a common source of indexing bugs in the loss functions. Even a partial annotation helps.

---

### 1.10 Key Functions

#### `frame_average_features` — `utils/jax_fn.py` (line 11793)

**Current:** Comments say `(frames, residues)` for features and `(frames)` for weights. The actual code does `weights = frame_weights.reshape(1, -1)` and `jnp.sum(feature_array * weights, axis=-1)`, which means features are `(residues, frames)`.

**Proposed signature:**
```python
@jaxtyped(typechecker=typechecker)
def frame_average_features(
    frame_wise_features: T_In,                    # each feature: Float[Array, "n_residues n_frames"]
    frame_weights: Float[Array, " n_frames"],
) -> T_In:                                        # each feature: Float[Array, " n_residues"]
```

**Why:** The comment and the code disagree on axis ordering. The annotation would catch this and force a decision.

---

#### `apply_sparse_mapping` — `data/splitting/sparse_map.py` (line 2528)

**Proposed:**
```python
@jaxtyped(typechecker=typechecker)
def apply_sparse_mapping(
    sparse_map: sparse.BCOO,                      # (n_fragments, n_residues)
    features: Float[Array, " n_residues"],
) -> Float[Array, " n_fragments"]:
```

---

#### `create_covariance_mat` — `data/splitting/sparse_map.py` (line 2337)

**Proposed:**
```python
@jaxtyped(typechecker=typechecker)
def create_covariance_mat(
    covariance_matrix: Float[Array, "n n"] | None,
    indices: Int[Array, " k"],
) -> Float[Array, "k k"] | None:
```

---

#### `compute_loss` — `opt/base.py` (line 9509)

The internal vectorized operations `train_losses * weights * scaling` require all three to be `Float[Array, "n_models"]`. The shape annotation on `LossComponents` would catch mismatches.

---

#### `pairwise_cosine_similarity` — `opt/loss/base.py` (line 9793)

**Current:**
```python
def pairwise_cosine_similarity(array: Array) -> Array:
```

**Problem:** If passed a 1-D array, it silently reshapes to `(n, 1)` and produces a trivially all-ones similarity matrix. This is likely a bug.

**Proposed:**
```python
@jaxtyped(typechecker=typechecker)
def pairwise_cosine_similarity(
    array: Float[Array, "n d"],
) -> Float[Array, "n n"]:
```

---

#### `normalize_weights` — `opt/loss/base.py` (line 9771)

**Proposed:**
```python
@jaxtyped(typechecker=typechecker)
def normalize_weights(
    weights: Float[Array, " n"],
    normalise: bool,
    eps: float,
    scale_eps: bool,
) -> Float[Array, " n"]:
```

---

#### Loss utility functions (all in `opt/loss/base.py`)

| Function | Current | Proposed |
|---|---|---|
| `apply_transforms` | `(jnp.ndarray, list) -> jnp.ndarray` | `(Float[Array, "..."], list) -> Float[Array, "..."]` |
| `apply_post_processing` | `(jnp.ndarray, int, bool) -> jnp.ndarray` | `(Float[Array, ""], int, bool) -> Float[Array, ""]` |
| `extract_upper_triangle` | `(Array) -> Array` | `(Float[Array, "n n"]) -> Float[Array, " k"]` |
| `normalize_upper_triangle` | `(Array, bool) -> Array` | `(Float[Array, " k"], bool) -> Float[Array, " k"]` |

---

#### Consistency loss inner functions (`opt/loss/consistency.py`)

All loss lambdas like `_exponential_l2_loss`, `_normalized_exponential_loss`, `_convex_kl_consistency_loss` should be:
```python
def _exponential_l2_loss(
    weight_sim: Float[Array, " k"],
    struct_sim: Float[Array, " k"],
) -> Float[Array, ""]:  # scalar
```

---

#### `NetHDX_input_features` — `models/HDX/netHDX/features.py` (line 8476)

**Current:**
```python
class NetHDX_input_features:
    contact_matrices: list[np.ndarray]  # Shape: (n_frames, n_residues, n_residues)
    residue_ids: Sequence[int]          # Shape: (n_residues,)
```

**Proposed:**
```python
class NetHDX_input_features:
    contact_matrices: Float[Array, "n_frames n_residues n_residues"]
    residue_ids: Int[Array, " n_residues"]
```

---

## Part 2: chex Assertions

### Why replace `assert` with chex

Python's `assert` has three problems in JAX code:
1. **Stripped in `-O` mode** — `python -O` removes all assert statements
2. **Not JIT-compatible** — Python `assert` on traced values raises `ConcretizationTypeError`
3. **No structured error messages** — vanilla assert gives minimal debug info

`chex` assertions are JIT-safe (they use `jax.debug.callback`), never stripped, and provide structured error messages with shapes/dtypes.

---

### 2.1 `Simulation.initialise()` — `models/core.py` (line 5749)

**Current:**
```python
lengths = [feature.features_shape[-1] for feature in self.input_features]
assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
assert len(self.forward_models) == len(self.params.model_parameters), ...
```

**Proposed:**
```python
chex.assert_equal(len(set(lengths)), 1)
chex.assert_equal(len(self.forward_models), len(self.params.model_parameters))
chex.assert_equal_shape([self.params.frame_weights, self.params.frame_mask])
chex.assert_rank(self.params.frame_weights, 1)
```

---

### 2.2 `normalize_weights` — `interfaces/simulation.py` (line 4210)

**Proposed additions:**
```python
@staticmethod
def normalize_weights(params: "Simulation_Parameters") -> "Simulation_Parameters":
    chex.assert_equal_shape([params.frame_weights, params.frame_mask])
    chex.assert_rank(params.frame_weights, 1)
    frame_weights = jax.nn.softmax(params.frame_weights)
    chex.assert_trees_all_close(jnp.sum(frame_weights), 1.0, atol=1e-5)
    # ...
```

---

### 2.3 `normalize_masked_loss_scalingweights` — `interfaces/simulation.py` (line 4158)

**Proposed additions:**
```python
weights = params.forward_model_weights
mask = params.normalise_loss_functions
chex.assert_equal_shape([weights, mask])
```

---

### 2.4 `compute_loss` — `opt/base.py` (line 9509)

**Proposed additions:**
```python
train_losses, val_losses = map(jnp.array, zip(*losses))
weights = simulation.params.forward_model_weights
scaling = jnp.array(simulation.params.forward_model_scaling)

chex.assert_equal_shape([train_losses, val_losses, weights, scaling])
chex.assert_rank(train_losses, 1)
chex.assert_tree_all_finite(train_losses)
```

---

### 2.5 `create_datasets` — `data/loader.py` (line 2166)

**Current:**
```python
assert len(set(train_indices)) == len(train_data), "Training topology fragments not unique..."
assert len(set(val_indices)) == len(val_data), "Validation topology fragments not unique..."
```

**Proposed:**
```python
chex.assert_equal(len(set(train_indices)), len(train_data))
chex.assert_equal(len(set(val_indices)), len(val_data))
chex.assert_equal(train_y_true.shape[-1], val_y_true.shape[-1])
```

---

### 2.6 `create_sparse_map` — `data/splitting/sparse_map.py` (line 2360)

**Current:**
```python
assert input_features.features_shape[0] == len(feature_topology), ...
```

**Proposed:**
```python
chex.assert_equal(input_features.features_shape[0], len(feature_topology))
```

---

### 2.7 `create_covariance_mat` — `data/splitting/sparse_map.py` (line 2337)

**Current (JIT-unsafe):**
```python
assert jnp.all(indices < covariance_matrix.shape[0]), "Indices are out of bounds..."
```

**Proposed:**
```python
chex.assert_rank(covariance_matrix, 2)
chex.assert_equal(covariance_matrix.shape[0], covariance_matrix.shape[1])  # Square matrix
```

---

### 2.8 `ExpD_Dataloader.__init__` — `data/loader.py` (line 2117)

**Current:**
```python
assert len(set([data.key for data in self.data])) == 1, "Keys are not the same..."
assert len({id(data.top) for data in self.data}) == len(self.data), "Topology fragments not unique..."
```

**Proposed:**
```python
chex.assert_equal(len(set(data.key for data in self.data)), 1)
chex.assert_equal(len({id(data.top) for data in self.data}), len(self.data))
```

---

### 2.9 `BV_model_Config.__post_init__` — `models/config.py` (line 5590)

**Current:**
```python
assert self.num_timepoints == len(self.bv_bc) and self.num_timepoints == len(self.bv_bh), ...
```

**Proposed:**
```python
chex.assert_equal(self.num_timepoints, len(self.bv_bc))
chex.assert_equal(self.num_timepoints, len(self.bv_bh))
```

---

### 2.10 `_optimise` loop — `opt/run.py` (line 9600)

**Replace buggy NaN check:**
```python
# Before (BUG):
if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):

# After:
if not jnp.isfinite(current_loss) or (current_loss < tolerance):
    print(f"Stopping at step {step}: loss={current_loss}")
    break
```

**Add debug assertion in loop:**
```python
chex.assert_tree_all_finite(opt_state.params)  # Catch NaN params early
```

---

### 2.11 No-op `UserWarning()` calls

The codebase has several `UserWarning("...")` as standalone expressions that do nothing:

- Line 2123: `UserWarning("y_true is a numpy array...")`
- Line 2125: `UserWarning("y_true is a jax array...")`
- Line 2140: `UserWarning("Topology fragments are missing indices...")`

These should be `warnings.warn(...)` or chex assertions.

---

## Part 3: `chex.dataclass` Migration

`chex.dataclass` automatically registers classes as JAX pytrees, eliminating manual `register_pytree_node` boilerplate.

### Candidates

| Class | File | Current Registration | Lines Saved |
|---|---|---|---|
| `Simulation_Parameters` | `interfaces/simulation.py` | Manual `register_pytree_node` | ~35 lines |
| `Dataset` | `data/loader.py` | `register_dataclass` decorator | Minor cleanup |
| `BV_Model_Parameters` | `models/HDX/BV/parameters.py` | Manual `register_pytree_node` | ~25 lines |
| `linear_BV_Model_Parameters` | `models/HDX/BV/parameters.py` | Manual `register_pytree_node` | ~25 lines |
| `OptimizationHistory` | `opt/base.py` | `register_dataclass` decorator | Consistency |

**Example migration:**
```python
# Before: ~50 lines
@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Array
    # ...
    def tree_flatten(self): ...      # 12 lines
    @classmethod
    def tree_unflatten(cls, ...): ... # 15 lines
register_pytree_node(Simulation_Parameters, ...)

# After: automatic pytree
@chex.dataclass(frozen=True)
class Simulation_Parameters:
    frame_weights: Float[Array, " n_frames"]
    # ...
```

**Caveat:** `Simulation_Parameters` has custom `__add__`, `__sub__`, `__mul__`, `__truediv__` methods. These are preserved by `chex.dataclass`. However, `slots=True` is not supported — use `frozen=True` only.

---

## Priority Matrix

### P0 — Critical, Low Effort
| Item | File | Line(s) |
|---|---|---|
| Fix NaN comparison bug | `opt/run.py` | 9638 |
| jaxtyping on `Simulation_Parameters` | `interfaces/simulation.py` | 4128–4136 |
| jaxtyping on `LossComponents` | `opt/base.py` | 8736–8744 |
| `chex.assert_tree_all_finite` in `compute_loss` | `opt/base.py` | 9509–9546 |

### P1 — High Impact, Moderate Effort
| Item | File |
|---|---|
| BV features shape annotations | `models/HDX/BV/features.py` |
| `frame_average_features` annotations | `utils/jax_fn.py` |
| Model parameters (BV, linear_BV, NetHDX) | `models/HDX/*/parameters.py` |
| Loss utilities annotations | `opt/loss/base.py` |
| Replace asserts in `Simulation.initialise()` | `models/core.py` |
| Replace asserts in `normalize_weights` | `interfaces/simulation.py` |

### P2 — Medium Impact
| Item | File |
|---|---|
| `Dataset` shape annotations | `data/loader.py` |
| `create_datasets` assertions | `data/loader.py` |
| `create_sparse_map` / `create_covariance_mat` | `data/splitting/sparse_map.py` |
| `apply_sparse_mapping` annotations | `data/splitting/sparse_map.py` |
| `Simulation.forward_pure` annotations | `models/core.py` |
| NetHDX features | `models/HDX/netHDX/features.py` |

### P3 — Cleanup
| Item | File |
|---|---|
| `chex.dataclass` migration | Multiple |
| Consistency loss inner functions | `opt/loss/consistency.py` |
| Fix no-op `UserWarning()` calls | `data/loader.py` |
| Testing with `chex.variants` | Tests |

---

## Implementation Notes

### Gradual Adoption Path

1. **Phase 1 (annotations only):** Add `jaxtyping` annotations without `@jaxtyped` decorator. These serve as documentation and are checked by type-checkers but don't add runtime overhead.

2. **Phase 2 (runtime checking):** Add `@jaxtyped(typechecker=beartype)` to critical functions (`compute_loss`, `forward_pure`, `frame_average_features`). Since `beartype_this_package()` is already active, jaxtyping can also integrate at the package level.

3. **Phase 3 (chex assertions):** Replace `assert` statements. Start with the optimization loop (highest bug risk), then data loading.

4. **Phase 4 (chex.dataclass):** Migrate dataclasses one at a time, testing pytree flatten/unflatten equivalence.

### Performance

- **jaxtyping runtime checks:** ~microseconds per call. Negligible vs JIT-compiled computation.
- **chex assertions inside JIT:** Use `jax.debug.callback` — zero overhead when disabled via `chex.disable_asserts()`.
- **chex.dataclass:** Generates identical pytree code to manual registration.

### Compatibility

- jaxtyping requires Python ≥ 3.9, JAX ≥ 0.4.x — already satisfied.
- `chex` is already imported in `opt/base.py` (line 8684) — no new dependency.
- beartype integration works via `@jaxtyped(typechecker=beartype)` or the existing `beartype_this_package()`.