# JAX-ENT Codebase Review

Comprehensive code review of the JAX-ENT codebase — a JAX-based Maximum Entropy framework for reweighting MD simulation trajectories against HDX-MS experimental data. The review covers correctness, numerical stability, JAX purity, security, test coverage, and code quality across ~74 Python files and ~15k lines of code.

---

## Critical Issues (Bugs / Correctness)

<!-- ### 1. NaN/Inf comparison bug — `src/opt/run.py:203`
```python
if (current_loss < tolerance) or (current_loss == jnp.nan) or (current_loss == jnp.inf):
```
`x == jnp.nan` is **always False** per IEEE 754. This means NaN losses will never trigger the convergence break, causing the optimization loop to run to `n_steps` with garbage values.

**Fix:** `jnp.isnan(current_loss) or jnp.isinf(current_loss)`

### [x] 7. Boolean context for JAX arrays — `src/opt/run.py:203`
`current_loss < tolerance` produces a JAX array used in Python `if`. Works in eager mode but would fail under `jax.jit`. If this path is ever JIT-compiled, it will break. -->




---

## High Priority Issues

### 4. Forward pass computed inside loss function (performance) — `src/opt/losses.py:27,63,101,137`
Each loss function recomputes `model.outputs` instead of receiving pre-computed predictions. In the optimization loop with N loss functions, this means N redundant forward passes per step. -> Fixed to use more flexible model interface - need to update this so that features are selected in an efficient and extendable way.


### 6. Python for-loop over frame dimension — `src/models/core.py:187-206`
```python
for frame_idx in range(n_frames):
    frame_data[feat_name] = feat_array[..., frame_idx : frame_idx + 1]
```
This creates N separate trace calls instead of using `jax.vmap` or `jax.lax.scan`, losing JIT performance.


---

## Medium Priority Issues


### 10. Inconsistent error handling
- `src/interfaces/builder.py` uses bare `assert` (stripped in `-O` mode, no error messages)
- `src/data/splitting/sparse_map.py` uses `chex.assert_equal()` (good)
- `src/models/core.py` uses `chex.assert_*()` (good)
- Should standardize on chex or explicit `raise ValueError()`

### 11. Print statements instead of logging
15+ `print()` calls across production code (data loaders, models, optimization). Should use Python `logging` module for controllable verbosity.

### 12. Large blocks of commented-out code
- `src/models/func/contacts.py:13-110` — entire old function (~100 lines)
- `src/models/core.py:52-54` — commented post_init
- `src/models/HDX/forward.py` — extensive commented sections

---

## Test Coverage Gaps

### Well-covered (no action needed)
- Topology system (11 tests) — comprehensive with known-answer tests
- Data splitting (13+ tests) — excellent coverage including edge cases
- Sparse mapping (6 tests) — gold-standard known-answer validation

### Critical gaps
#### Potential Module tests
| Component | Tests | Issue |
|-----------|-------|-------|
| Optimization convergence | 5 integration | Tests run but don't assert loss decreases or converges |

#### Potential Unit tests
| Component | Tests | Issue |
|-----------|-------|-------|
| Forward models (BV, netHDX) | 2 (1 broken) | No numerical correctness validation, no gradient checks |
| Loss function correctness | 4 | Tests timing/creation but not mathematical correctness |
| Gradient flow | 0 | No finite-difference gradient verification anywhere |

### Broken/disabled tests
- `tests/integration/HDX/test_forwardmodel.py` — marked `# TODO: NOTWORKING 06/02/25`
- `tests/slow/optimise/broken/` — 2 broken test files
- Multiple `OLD*.py` test files that are stale

---

## Low Priority / Housekeeping / Performance

- `m_key = NewType("m_key", str)` provides no runtime enforcement
- `HDX_peptide.dfrac` accepts overly permissive types without 0-1 range validation
- `Partial_Topology` doesn't validate `peptide_trim > length` or negative residues
- CLI files lack return type hints
- `efficient_k_cluster.py` doesn't validate `chunk_size > 0`
- Unused imports: `TypedDict` in `opt/run.py`, `sys` in `utils/hdf.py`
- `opt/loss/legacy.py` usage unclear — may be dead code
- `pyproject.toml` sets `mypy ignore_missing_imports = true` — suppresses external type errors
### 5. Print statements in JIT-traced code — `src/models/core.py:87-88, 115, 123`
`print()` calls inside `initialise()` execute during JAX tracing, not at runtime. These should be removed or gated behind a debug flag outside traced paths. Since initialise is only called outside of the jit path, this is not a problem.

### 2. HDF5 deserialization allows arbitrary class loading — `src/utils/hdf.py:114-117`
```python
class_info = group.attrs["class_info"]
module_name, class_name = class_info.rsplit(".", 1)
module = importlib.import_module(module_name)
cls = getattr(module, class_name)
```
An untrusted HDF5 file can inject arbitrary `class_info` strings, leading to arbitrary code execution via `importlib.import_module`. This is a **deserialization vulnerability**.

**Fix:** Validate `class_info` against a whitelist of allowed classes before import.

### 3. Sparse matrix converted to dense — `src/data/splitting/sparse_map.py:235`
```python
return jnp.squeeze(sparse_map.todense() @ features)
```
Defeats the purpose of BCOO sparse representation. For large systems this causes unnecessary memory allocation.

**Fix:** Use `sparse_map @ features` directly (JAX BCOO supports matmul).

---

## Architectural Strengths

- Clean module separation with no circular imports
- Good use of `TYPE_CHECKING` blocks for forward references
- Sophisticated runtime config system with 5 preset modes and env var support
- Robust topology system with factory pattern, serialization, and pairwise comparisons
- JIT Guard (`utils/jit_fn.py`) is well-designed with context manager, decorator variants, and proper cleanup
- Examples are thoroughly documented with README, INSTRUCTIONS, and shell scripts
- Key-based dispatch (`m_key → ForwardPass`) is elegant and extensible
- Beartype integration via `beartype_this_package()` for runtime type checking

---

## Recommended Priority Order

1. **[x] Fix NaN comparison bug** (`opt/run.py:203`) — correctness, trivial fix
2. **Add HDF5 class whitelist** (`utils/hdf.py`) — security
3. **Use sparse matmul** (`sparse_map.py:235`) — correctness/performance
<!-- 4. **Add gradient verification tests** — testing gap, prevents regression -->
5. **Fix forward pass in loss loop** (`opt/losses.py`) — performance
6. **Remove prints from JIT path** (`models/core.py`) — JAX correctness
7. **Add epsilon to weight division** (`opt/loss/base.py`) — numerical stability
8. **Standardize error handling** — maintainability
9. **Replace print with logging** — observability
10. **Clean dead code** — maintainability
