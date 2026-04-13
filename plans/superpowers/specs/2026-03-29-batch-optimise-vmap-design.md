# Design: `batch_optimise` — vmapped hyperparameter sweep

**Date:** 2026-03-29
**Branch:** refactor-optimisation
**Status:** Approved

---

## Overview

Enable efficient hyperparameter sweeps by making the JAX-ENT optimisation stack fully
JAX-traceable and introducing a new `batch_optimise` entry point that runs multiple
hyperparameter configurations in parallel via `jax.vmap` within fixed-size batches
iterated by `jax.lax.map`.

The existing `run_optimise` / `_optimise` path is **left funcitonally untouched**. All new
machinery is additive alongside the current Python-loop path, with shared carry types
enforced across both paths.

---

## Motivation

The current hparam sweep (examples 2 and 3) runs each (maxent_scaling × bv_reg_scaling)
combination sequentially in a Python for-loop. The optimisation loop itself is also a
Python for-loop with mutable Python objects (`ConvergenceTracker`, `MutableLearningRate`,
`OptimizationHistory`) and Python-level conditionals on traced values, making it
incompatible with `jax.vmap`, `jax.lax.while_loop`, and full JIT compilation of a
complete optimisation run.

---

## Architecture

```
batch_optimise(hparam_batch, batch_size, ...)
    │
    ├─ pad n_hparams to multiple of batch_size
    ├─ reshape → [n_batches, batch_size, ...]
    ├─ setup per-run loggers
    │
    └─ jax.lax.map(
           jax.vmap(_optimise_pure),   # parallel within batch
           hparam_batches              # sequential across batches
       )
       → [n_batches, batch_size, OptimisationResult]
           │
           └─ unpad → BatchOptimisationResult[n_hparams]
```

Memory at any point scales with `batch_size × n_steps × state_size`, not
`n_hparams × n_steps × state_size`.

---

## New and changed files

| File | Change |
|---|---|
| `jaxent/src/opt/batch.py` | New — `batch_optimise`, `HParamBatch`, `BatchOptimisationResult` |
| `jaxent/src/opt/base.py` | Add `OptimisationCarry`, `ConvergenceCarry` NamedTuples |
| `jaxent/src/opt/run.py` | Add `_optimise_pure`; refactor `_optimise` to use shared carry types |
| `jaxent/src/opt/optimiser.py` | Store pre-computed initial/final gradient masks; add `_pure_step`; update `compute_loss` and `_step` to return `(result, Simulation)` |
| `jaxent/src/opt/gradients.py` | Remove all print statements (moved to `OptaxOptimizer.initialise`) |
| `jaxent/src/opt/log.py` | Add `silent: bool` toggle; accept injected `logging.Logger` |
| `jaxent/src/models/core.py` | Move `outputs` to dynamic pytree leaf; make `Simulation.forward` return `(sim, outputs)` |
| `jaxent/examples/` | Update call sites: `sim, outputs = Simulation.forward(sim, params)` |
| `jaxent/examples/common/optimization.py` | Remove scalar captures from loss closures; write scalings into `forward_model_scaling` or `weights` |

---

## Shared carry types

Both `_optimise` (Python loop) and `_optimise_pure` (while_loop) use these types,
ensuring a single source of truth for optimisation state.

### `ConvergenceCarry`

Replaces `ConvergenceTracker`. All fields are JAX arrays.

```python
class ConvergenceCarry(NamedTuple):
    ema_loss_delta: Array                  # scalar float
    ema_params: Simulation_Parameters      # EMA-averaged params
    steps_since_threshold_start: Array     # int32 scalar
    current_threshold_idx: Array           # int32 scalar
    converged: Array                       # bool scalar
```

Convergence thresholds (`convergence × learning_rate`, sorted descending) are
**static** — computed once at initialisation and passed as a fixed-shape array
argument to the pure functions rather than stored in the carry.

### `OptimisationCarry`

Full per-step carry. Everything that changes during optimisation.

```python
class OptimisationCarry(NamedTuple):
    opt_state: OptimizationState           # params, optax state, step, losses, grads
    sim: Simulation                        # outputs is now a dynamic pytree leaf
    convergence: ConvergenceCarry
    lr: Array                              # float32 scalar — current frame LR
    model_lr: Array                        # float32 scalar — current model LR
    gradient_mask_idx: Array               # int32 scalar: 0=initial, 1=final
    history_params: Simulation_Parameters  # [n_steps, ...] pre-allocated
    history_losses: LossComponents         # [n_steps, n_models] pre-allocated
    write_idx: Array                       # int32 scalar
```

`lr`, `model_lr`, and `gradient_mask_idx` replace `MutableLearningRate` and the
in-step mutation of `optimizer.gradient_mask`. The actual masks are pre-computed
and stored in `OptaxOptimizer` as `_initial_gradient_mask` and `_final_gradient_mask`;
`gradient_mask_idx` selects between them via `jax.lax.select`.

---

## Inner loop — `_pure_step`

Replaces the mutable logic in `OptaxOptimizer._step`. Used by both the Python loop
(via direct call) and the while_loop (via `body_fn`).

### LR oscillation reduction

```python
new_lr, new_model_lr = jax.lax.cond(
    (grad_dot_product < 0) & (carry.opt_state.step > 1),
    lambda lr, mlr: (lr / plateau_denominator, mlr / plateau_denominator),
    lambda lr, mlr: (lr, mlr),
    carry.lr, carry.model_lr,
)
```

### Gradient mask selection at `initial_steps`

Pre-computed masks are stored in the optimizer during initialisation. At runtime,
`jax.lax.select` picks between them — no recomputation inside traced context:

```python
# In OptaxOptimizer.initialise:
self._initial_gradient_mask = create_gradient_masks(frame_only_partition, params, None)
self._final_gradient_mask = create_gradient_masks(full_partition_masks, params, None)

# In _pure_step:
new_mask_idx = jax.lax.cond(
    carry.opt_state.step == initial_steps,
    lambda: jnp.array(1, dtype=jnp.int32),  # switch to final
    lambda: carry.gradient_mask_idx,
)
gradient_mask = jax.tree.map(
    lambda init, final: jax.lax.select(new_mask_idx == 0, init, final),
    optimizer._initial_gradient_mask,
    optimizer._final_gradient_mask,
)
```

### History write

```python
new_history_losses = jax.tree.map(
    lambda buf, val: jax.lax.dynamic_update_slice(
        buf, val[None], (carry.write_idx,) + (0,) * val.ndim
    ),
    carry.history_losses,
    new_losses,
)
new_write_idx = carry.write_idx + 1
```

---

## Inner loop — `while_loop` structure

```python
def cond_fn(carry: OptimisationCarry) -> bool:
    # n_steps is a static closure over config.n_steps — not part of the carry
    return ~carry.convergence.converged & (carry.opt_state.step < n_steps)

def body_fn(carry: OptimisationCarry) -> OptimisationCarry:
    return jax.lax.cond(
        carry.convergence.converged,
        lambda c: c,                   # converged — freeze state, no-op
        lambda c: _pure_step(c, ...),  # still running — real step
        carry,
    )

final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
```

`n_steps` is captured as a static value in the closure over `config.n_steps`. Both
the EMA convergence condition (via `check_and_advance_threshold`) and the hard step
limit terminate the loop by making `cond_fn` return `False` — the convergence path
sets `carry.convergence.converged = True`, the step limit is checked directly in
`cond_fn`.

### Converged-gating under `vmap`

Under `jax.vmap(_optimise_pure)`, JAX transforms the while_loop so the outer loop
continues until **all** batch elements are converged. The `jax.lax.cond` in `body_fn`
ensures converged elements do zero real work — they return their carry unchanged —
so running to `max(convergence_steps)` across the batch is cheap.

---

## Convergence pure functions

```python
def update_convergence(
    carry: ConvergenceCarry,
    previous_loss: Array,
    current_loss: Array,
    current_params: Simulation_Parameters,
    ema_alpha: float,                  # static
) -> tuple[ConvergenceCarry, Array]:  # (new carry, raw_loss_delta)

def check_and_advance_threshold(
    carry: ConvergenceCarry,
    current_loss: Array,
    step: Array,
    thresholds: Array,                 # static [n_thresholds]
    min_steps: int,                    # static
    initial_steps: int,                # static
) -> ConvergenceCarry:
```

All branching uses `jax.lax.cond`. No Python-level conditionals on traced values.

---

## `HParamBatch` and `BatchOptimisationResult`

```python
class HParamBatch(NamedTuple):
    forward_model_weights: Float[Array, "n_hparams n_models"]
    forward_model_scaling: Float[Array, "n_hparams n_models"]
    learning_rate: Float[Array, " n_hparams"] | None  # None → use config for all runs
```

`forward_model_scaling` is `[n_hparams, n_models]` regardless of sweep geometry.
The caller flattens the sweep grid before passing it in:

- 1D sweep (5 MaxEnt values): `[5, n_models]` — one column varies
- 2D sweep (5×5 grid): caller does `meshgrid` → flatten → `[25, n_models]`
- Single run: `[1, n_models]`

Reshaping back to grid shape for plotting is handled downstream by the caller
(e.g. existing `sweeps.py` utilities).

```python
class BatchOptimisationResult(NamedTuple):
    histories: OptimizationHistory      # [n_hparams, n_steps, ...]
    best_states: OptimizationState      # [n_hparams, ...]
    convergence_steps: Array            # [n_hparams] int32 — step at convergence
    hparam_batch: HParamBatch           # echoed for result matching
```

`convergence_steps` is `write_idx` from the final carry. Callers use it to mask
`histories[:convergence_steps[i]]` for each run.

`OptimizationHistory.from_carry(carry)` reconstructs a standard history object
from the pre-allocated buffers, sliced to `write_idx`.

---

## `batch_optimise` signature

```python
def batch_optimise(
    simulation: InitialisedSimulation,
    hparam_batch: HParamBatch,           # [n_hparams, ...]
    batch_size: int,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    config: OptimiserSettings,
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    run_names: list[str],                # length n_hparams
    log_dir: str,
    log_every_n: int = 50,
    logger: logging.Logger | None = None,
) -> BatchOptimisationResult:
```

### Execution pattern

```python
# 1. Validate run_names length matches n_hparams
# 2. Pad to multiple of batch_size
n_pad = (-n_hparams) % batch_size
hparam_padded = _pad_hparam_batch(hparam_batch, n_pad)

# 3. Reshape into [n_batches, batch_size, ...]
hparam_batches = jax.tree.map(
    lambda x: x.reshape(n_batches, batch_size, *x.shape[1:]),
    hparam_padded,
)

# 4. Setup per-run loggers (padded slots get None)
loggers = _setup_run_loggers(run_names, log_dir, n_pad)

# 5. Build vmapped function
vmapped_fn = jax.vmap(
    partial(_optimise_pure, simulation, data_to_fit=data_to_fit, ...),
    in_axes=0,
)

# 6. Run
results = jax.lax.map(vmapped_fn, hparam_batches)
# shape: [n_batches, batch_size, ...]

# 7. Unpad and return
return _unpad_and_collect(results, n_hparams)
```

---

## Logging

### Boundary

| Component | Logger type |
|---|---|
| `Simulation.initialise` | Ambient `logging.getLogger("jaxent.models")` |
| `create_gradient_masks` | No logging (pure function) — info moved to caller |
| `OptaxOptimizer.initialise` | Injected `logger` — logs mask info, JIT success/failure |
| `_optimise` | Injected `logger` — refactored `log.py` functions accept logger + `silent` toggle |
| `_optimise_pure` | Injected `logger` — captured in closure, used via `jax.debug.callback` |
| `run_optimise` | Public entry — `logger=None` falls back to ambient |
| `batch_optimise` | Public entry — `logger=None` falls back to ambient |

### `log.py` silent toggle

Functions in `log.py` accept `silent: bool = False`. When `silent=True`, they
are no-ops. This is used in batch mode where `jax.debug.callback` handles
per-run logging separately:

```python
def log_step(step: int, losses: LossComponents, lr: float, 
             logger: logging.Logger, silent: bool = False) -> None:
    if silent:
        return
    logger.info(f"Step {step}: loss={losses.total:.6f}, lr={lr:.2e}")
```

### Per-run log files

`batch_optimise` creates one `logging.FileHandler` per run before entering
`jax.lax.map`:

```python
loggers = [
    _make_file_logger(run_names[i], log_dir) if i < n_hparams else None
    for i in range(n_hparams + n_pad)
]
```

Inside `_optimise_pure`, the `jax.debug.callback` receives batched values
`(batch_indices, step, losses, lr)` — the Python callback iterates over
`batch_size` and routes each element to `loggers[batch_indices[i]]`, skipping
`None` slots (padded runs).

`log_every_n` gates whether the callback fires at all for a given step, keeping
the batched sweep quiet by default (recommended: 50).

All handlers are flushed and closed in `batch_optimise` after `jax.lax.map` returns.

---

## Group D — loss closure fix

Loss functions become **scaling-free structural callables**:

```python
# Before
maxent_loss = lambda sim, data, idx: _maxent_core(sim, data, idx) * maxent_scaling

# After
maxent_loss = lambda sim, data, idx: _maxent_core(sim, data, idx)
```

All per-run scaling lives in `simulation.params.forward_model_scaling`, which
`compute_loss` already multiplies by. `loss_functions` remains `static_argnames`
in `compute_loss` — same callable structure across all runs.

The column ordering of `forward_model_scaling` and `forward_model_weights` (and therefore `HParamBatch`) follows
the `loss_functions` list ordering passed to `batch_optimise`:
- column 0 → primary loss
- column 1 → MaxEnt loss
- column 2 → BV regularisation loss
- etc.

## Functional `Simulation`

### Pytree update

Move `outputs` from `aux_data` (static) to `dynamic_values` so JAX can trace updates
inside JIT/vmap contexts:

```python
def tree_flatten(self):
    """Flatten for JAX pytree registration."""
    dynamic_values = (self.params, self.outputs)  # outputs now dynamic
    aux_data = (
        self.input_features,
        self.forward_models,
        self.forwardpass,
        self.length,
        self._input_features,
        self._jit_forward_pure,
    )
    return dynamic_values, aux_data

@classmethod
def tree_unflatten(cls, aux_data, dynamic_values):
    (params, outputs) = dynamic_values
    (input_features, forward_models, forwardpass, 
     length, _input_features, _jit_forward_pure) = aux_data
    
    instance = cls(input_features, forward_models, params)
    instance.forwardpass = forwardpass
    instance.length = length
    instance.outputs = outputs
    instance._input_features = _input_features
    instance._jit_forward_pure = _jit_forward_pure
    return instance
```

### Pure functional `forward`

`Simulation.forward` returns `(new_sim, outputs)` instead of mutating in place:

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

This enables `Simulation` to be included in `OptimisationCarry` as a JAX pytree
where only `(params, outputs)` are traced/transformed, while all static
configuration remains unchanged.

---

## Functional Propagation

To support pure `Simulation.forward` without mutation:
1. **`compute_loss`** returns `(LossComponents, Simulation)` — propagates updated sim with new outputs.
2. **`_step`** and **`_pure_step`** receive `sim`, return `(OptimizationState, Simulation)`.
3. **`OptimisationCarry`** includes `sim: Simulation` — outputs is now a dynamic leaf, so JAX traces only the changing parts.
4. **`Simulation.initialise`** unpacks the functional forward: `self, self.outputs = Simulation.forward(self, self.params)`

---

## What is NOT changed

- `run_optimise` / `_optimise` Python loop path — refactored to use `OptimisationCarry` but API preserved
- `Simulation` pytree registration — updated (outputs moves to dynamic), but still a registered pytree
- `OptimizationHistory` in the Python loop path — unchanged
- Example scripts — update call sites to `sim, outputs = Simulation.forward(sim, params)`;
  callers adapt `LossConfig` construction to write scalings into `HParamBatch.forward_model_scaling`

---

## Barriers resolved

| Group | Barrier | Resolution |
|---|---|---|
| A | Python conditionals in `_step` on traced values | `jax.lax.cond` in `_pure_step` |
| A | `MutableLearningRate` mutation inside step | LR carried as `OptimisationCarry.lr` |
| A | `gradient_mask` mutation inside step | Pre-compute both masks; use `gradient_mask_idx` + `jax.lax.select` |
| A | `create_gradient_masks` print statements | Moved to `OptaxOptimizer.initialise` |
| B | Python `for` loop | `jax.lax.while_loop` in `_optimise_pure` |
| B | `.item()` calls on traced values | Removed; `jax.lax.cond` used throughout |
| B | `ConvergenceTracker` Python state | `ConvergenceCarry` + pure functions |
| B | `time.time()` in loop body | Moved outside the traced path |
| B | `try/except` around loop body | Moved outside the traced path |
| C | `OptimizationHistory.states.append` | Pre-allocated buffer + `dynamic_update_slice` |
| D | Loss closures capturing Python floats | Scaling-free callables; scalings in `forward_model_scaling` |
| E | `Simulation.forward` in-place mutation | Functional signature returning `(sim, outputs)` |
| E | `outputs` in `aux_data` (static) | Moved to `dynamic_values` in pytree registration |
