# batch_optimise — VMapped Hyperparameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the JAX-ENT optimisation stack fully JAX-traceable and add `batch_optimise` that runs hyperparameter sweeps with `jax.vmap` within fixed-size batches iterated by `jax.lax.map`.

**Architecture:** Replace mutable Python state (`ConvergenceTracker`, `MutableLearningRate`, `OptimizationHistory.append`) with a pure functional `OptimisationCarry` NamedTuple. Add `_pure_step` (no side effects) consumed by `_optimise_pure` (`jax.lax.while_loop`). `batch_optimise` vmaps `_optimise_pure` over `HParamBatch` in `batch_size` chunks via `jax.lax.map`. The existing `_optimise`/`run_optimise` Python-loop path is refactored to use the same carry types but is otherwise left intact.

**Tech Stack:** JAX 0.4.35, Optax ≥ 0.2.4, Chex, jaxtyping, beartype, pytest

---

## Key Design Decisions

### Functional `Simulation.forward`

`Simulation.forward` becomes pure functional, returning `(new_sim, outputs)`:

```python
@staticmethod
def forward(sim, params: Simulation_Parameters) -> tuple["Simulation", tuple[Output_Features]]:
    """Pure forward — returns (new_sim, outputs)."""
    params = Simulation_Parameters.normalize_weights(params)
    outputs = tuple(sim._jit_forward_pure(params, sim._input_features, sim.forwardpass))
    _, aux_data = sim.tree_flatten()
    new_sim = Simulation.tree_unflatten(aux_data, (params, outputs))
    return new_sim, outputs
```

### `Simulation` Pytree Update

Move `outputs` from `aux_data` (static) to `dynamic_values` so JAX can trace updates:

```python
def tree_flatten(self):
    dynamic_values = (self.params, self.outputs)  # outputs now dynamic
    aux_data = (self.input_features, self.forward_models, self.forwardpass,
                self.length, self._input_features, self._jit_forward_pure)
    return dynamic_values, aux_data
```

### Gradient Masks

Pre-compute both initial and final gradient masks during `OptaxOptimizer.initialise`. Use `jax.lax.select` in `_pure_step` to switch at `initial_steps` — no recomputation inside traced context.

### Logging

Replace all `print()` statements with `logging.Logger`. Add a `silent: bool` toggle to `log.py` functions for batch mode where `jax.debug.callback` handles per-run logging.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `jaxent/src/opt/track.py` | Replace | `ConvergenceCarry` NamedTuple + pure convergence functions |
| `jaxent/src/models/core.py` | Modify | Functional `Simulation.forward` returning `(sim, outputs)`; move `outputs` to dynamic pytree leaf |
| `jaxent/src/opt/gradients.py` | Modify | Remove print statements; pre-compute masks |
| `jaxent/src/opt/base.py` | Modify | Add `OptimisationCarry`, `HParamBatch`, `BatchOptimisationResult` |
| `jaxent/src/opt/optimiser.py` | Modify | Store initial/final masks; update `compute_loss`/`_step` to propagate `sim`; add `_pure_step` |
| `jaxent/src/opt/run.py` | Modify | Add `_optimise_pure`; refactor `_optimise`/`run_optimise` |
| `jaxent/src/opt/log.py` | Modify | Accept injected `logging.Logger`; add `silent` toggle |
| `jaxent/src/opt/batch.py` | Create | `batch_optimise` entry point |
| `jaxent/examples/` | Patch | Update call sites: `sim, outputs = Simulation.forward(sim, params)` |

---

### Task 1: `ConvergenceCarry` and pure convergence functions
- [x] Step 1: Write tests
- [x] Step 2: Implement `ConvergenceCarry` in `track.py`

### Task 2: Functional `Simulation`
- [ ] **Step 1: Update `Simulation` pytree registration**
  Modify `jaxent/src/models/core.py`:
  1. Move `outputs` from `aux_data` to `dynamic_values` in `tree_flatten`/`tree_unflatten`.
  2. Update `tree_unflatten` to accept `(params, outputs)` as dynamic values.
- [ ] **Step 2: Make `Simulation.forward` pure functional**
  Change signature to return `tuple[Simulation, tuple[Output_Features]]`:
  ```python
  @staticmethod
  def forward(sim, params) -> tuple["Simulation", tuple[Output_Features]]:
      params = Simulation_Parameters.normalize_weights(params)
      outputs = tuple(sim._jit_forward_pure(params, sim._input_features, sim.forwardpass))
      _, aux_data = sim.tree_flatten()
      new_sim = Simulation.tree_unflatten(aux_data, (params, outputs))
      return new_sim, outputs
  ```
- [ ] **Step 3: Update `Simulation.initialise`**
  Use the functional forward and unpack: `self, self.outputs = Simulation.forward(self, self.params)`
- [ ] **Step 4: Replace print statements with logging**
  Use `logging.getLogger("jaxent.models")` for init messages.
- [ ] **Step 5: Verify model tests**
  ```bash
  pytest jaxent/tests/unit/models/ -v
  ```

### Task 3: Side-effect-free Gradients
- [ ] **Step 1: Remove prints from `create_gradient_masks`**
  Move diagnostic output to the caller (`OptaxOptimizer.initialise`).
- [ ] **Step 2: Pre-compute both initial and final gradient masks**
  In `OptaxOptimizer.initialise`, compute and store:
  - `self._initial_gradient_mask` (frame weights only)
  - `self._final_gradient_mask` (full partition masks)
  These are used via `jax.lax.select` in `_pure_step`.

### Task 4: Add Shared Carry Types
- [ ] **Step 1: Define `OptimisationCarry`**
  In `jaxent/src/opt/base.py`:
  ```python
  class OptimisationCarry(NamedTuple):
      opt_state: OptimizationState
      sim: Simulation                    # outputs is now a dynamic leaf
      convergence: ConvergenceCarry
      lr: Array                          # current frame LR
      model_lr: Array                    # current model LR
      gradient_mask_idx: Array           # int32: 0=initial, 1=final
      history_params: Simulation_Parameters
      history_losses: LossComponents
      write_idx: Array
  ```
- [ ] **Step 2: Define `HParamBatch` and `BatchOptimisationResult`**
  ```python
  class HParamBatch(NamedTuple):
      forward_model_weights: Float[Array, "n_hparams n_models"]
      forward_model_scaling: Float[Array, "n_hparams n_models"]
      learning_rate: Float[Array, "n_hparams"] | None

  class BatchOptimisationResult(NamedTuple):
      histories: OptimizationHistory
      best_states: OptimizationState
      convergence_steps: Array
      hparam_batch: HParamBatch
  ```

### Task 5: Refactor `OptaxOptimizer`
- [ ] **Step 1: Remove `MutableLearningRate`**
  LR is now carried as `OptimisationCarry.lr` / `model_lr`.
- [ ] **Step 2: Store pre-computed gradient masks**
  Add `_initial_gradient_mask` and `_final_gradient_mask` attributes.
- [ ] **Step 3: Switch to unit-LR optimizer chains**
  Scale gradients by `carry.lr` in `_pure_step` rather than modifying optax LR.

### Task 6: Functional Stack Updates
- [ ] **Step 1: Update `compute_loss` signature**
  Return `(LossComponents, Simulation)` to propagate updated sim.
- [ ] **Step 2: Update `_step` signature**
  Receive `sim`, return `(OptimizationState, Simulation)`.
- [ ] **Step 3: Implement `_pure_step`**
  Stateless step function that:
  1. Calls `compute_loss` → gets `(losses, new_sim)`
  2. Uses `jax.lax.select` to pick gradient mask based on `gradient_mask_idx`
  3. Applies LR scaling to gradients
  4. Uses `jax.lax.cond` for LR reduction on sign flip
  5. Updates history via `dynamic_update_slice`
  6. Returns new `OptimisationCarry`

### Task 7: Batch Optimization Logic
- [ ] **Step 1: Implement `_optimise_pure`**
  `jax.lax.while_loop` path in `run.py`:
  ```python
  def cond_fn(carry):
      return ~carry.convergence.converged & (carry.opt_state.step < n_steps)

  def body_fn(carry):
      return jax.lax.cond(carry.convergence.converged,
                          lambda c: c, lambda c: _pure_step(c, ...), carry)

  final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
  ```
- [ ] **Step 2: Implement `batch_optimise` in `batch.py`**
  Entry point that pads, reshapes, vmaps, and collects results.

### Task 8: Refactor Python Loop Path
- [ ] **Step 1: Update `_optimise` to use `OptimisationCarry`**
  Maintain compatibility with existing `run_optimise` callers.
- [ ] **Step 2: Update `run_optimise` to return `(result, Simulation)`**
  Propagate the updated simulation with final outputs.

### Task 9: Logging and Polish
- [ ] **Step 1: Add `silent` toggle to `log.py` functions**
  When `silent=True`, functions are no-ops (used in batch mode).
- [ ] **Step 2: Replace remaining prints with logging**
  Use `logging.getLogger("jaxent.opt")` throughout optimiser code.
- [ ] **Step 3: Add JAX debug callbacks for batch logging**
  In `_optimise_pure`, use `jax.debug.callback` to route per-run logs.

### Task 10: Call Site Migration
- [ ] **Step 1: Update all `Simulation.forward` call sites**
  Search: `Simulation\.forward\(.*?\)`
  Replace pattern: `sim, outputs = Simulation.forward(sim, params)`
  Affected locations:
  - `jaxent/src/opt/optimiser.py` (inside `compute_loss`)
  - `jaxent/examples/` scripts
  - Test files
- [ ] **Step 2: Verify example integration**
  ```bash
  pytest jaxent/tests/unit/opt/ -v
  python jaxent/examples/2_CrossValidation/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT.py --n-steps 5
  ```

---

## Verification Plan

### Automated Tests
1. **Unit Tests**: `pytest jaxent/tests/unit/opt/` (covers carry, pure step, and while_loop).
2. **Integration Tests**: Sweep validation using `batch_optimise` on a synthetic dataset.
3. **Regression Tests**: Run existing `2_CrossValidation` scripts using both `run_optimise` (old path) and `batch_optimise` (new path) and compare results.

### Manual Verification
1. **Log Inspection**: Verify that `batch_optimise` generates individual log files in the specified directory.
2. **Memory Profile**: Confirm that memory usage stays constant with batch count when iterating via `jax.lax.map`.
