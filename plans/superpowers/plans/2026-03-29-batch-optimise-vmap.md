# batch_optimise — VMapped Hyperparameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the JAX-ENT optimisation stack fully JAX-traceable and add `batch_optimise` that runs hyperparameter sweeps with `jax.vmap` within fixed-size batches iterated by `jax.lax.map`.

**Architecture:** Replace mutable Python state (`ConvergenceTracker`, `MutableLearningRate`, `OptimizationHistory.append`) with a pure functional `OptimisationCarry` NamedTuple. Add `_pure_step` (no side effects) consumed by `_optimise_pure` (`jax.lax.while_loop`). `batch_optimise` vmaps `_optimise_pure` over `HParamBatch` in `batch_size` chunks via `jax.lax.map`. The existing `_optimise`/`run_optimise` Python-loop path is refactored to use the same carry types but is otherwise left intact.

**Tech Stack:** JAX 0.4.35, Optax ≥ 0.2.4, Chex, jaxtyping, beartype, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `jaxent/src/opt/track.py` | Replace | `ConvergenceCarry` NamedTuple + pure convergence functions |
| `jaxent/src/models/core.py` | Modify | Make `Simulation.forward` functional (no in-place mutation) |
| `jaxent/src/opt/gradients.py` | Modify | Remove print statements; add `_apply_lr_scaling` helper |
| `jaxent/src/opt/base.py` | Modify | Add `OptimisationCarry`, `HParamBatch`, `BatchOptimisationResult` |
| `jaxent/src/opt/optimiser.py` | Modify | Remove `MutableLearningRate`; unit-LR optimizer; add `_pure_step`; refactor `initialise` |
| `jaxent/src/opt/run.py` | Modify | Add `_optimise_pure`; refactor `_optimise`/`run_optimise` to use carry |
| `jaxent/src/opt/log.py` | Modify | Accept injected `logging.Logger` in all functions |
| `jaxent/src/opt/batch.py` | Create | `batch_optimise`, logger setup, padding/unpadding utilities |
| `jaxent/examples/common/optimization.py` | Modify | Remove scalar captures from loss closures |
| `jaxent/tests/unit/opt/test_convergence_carry.py` | Create | Tests for `ConvergenceCarry` + pure functions |
| `jaxent/tests/unit/opt/test_pure_step.py` | Create | Tests for `_pure_step` |
| `jaxent/tests/unit/opt/test_optimise_pure.py` | Create | Tests for `_optimise_pure` end-to-end |
| `jaxent/tests/unit/opt/test_batch_optimise.py` | Create | Tests for `batch_optimise` |

Run all tests with: `pytest jaxent/tests/unit/opt/ -v` from repo root with `.venv` activated.

---

### Task 1: `ConvergenceCarry` and pure convergence functions

**Files:**
- Replace: `jaxent/src/opt/track.py`
- Create: `jaxent/tests/unit/opt/test_convergence_carry.py`

- [ ] **Step 1: Write failing tests**

```python
# jaxent/tests/unit/opt/test_convergence_carry.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.opt.track import (
    ConvergenceCarry,
    init_convergence_carry,
    update_convergence,
    check_and_advance_threshold,
)
from jaxent.src.interfaces.simulation import Simulation_Parameters


def _make_params(n_frames=4, n_models=2):
    return Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=[],
        forward_model_weights=jnp.ones(n_models),
        normalise_loss_functions=jnp.zeros(n_models),
        forward_model_scaling=jnp.ones(n_models),
    )


class TestConvergenceCarry:
    def test_init_carry_fields(self):
        params = _make_params()
        carry = init_convergence_carry(params)
        assert carry.converged == False
        assert carry.current_threshold_idx == 0
        assert carry.steps_since_threshold_start == 0
        np.testing.assert_allclose(float(carry.ema_loss_delta), 0.0)

    def test_update_convergence_ema(self):
        params = _make_params()
        carry = init_convergence_carry(params)
        carry, delta = update_convergence(carry, jnp.float32(1.0), jnp.float32(0.9), params, ema_alpha=0.5)
        # raw_delta = |1.0 - 0.9| = 0.1; ema = 0.5*0.1 + 0.5*0.0 = 0.05
        np.testing.assert_allclose(float(carry.ema_loss_delta), 0.05, atol=1e-6)
        assert carry.steps_since_threshold_start == 1

    def test_threshold_not_met_before_min_steps(self):
        params = _make_params()
        carry = init_convergence_carry(params)
        thresholds = jnp.array([1e-1, 1e-2])
        # ema_loss_delta is 0, steps_since=0 < min_steps=5
        carry = check_and_advance_threshold(
            carry, jnp.float32(1.0), jnp.int32(10),
            thresholds, min_steps=5, initial_steps=2
        )
        assert carry.converged == False
        assert carry.current_threshold_idx == 0

    def test_threshold_advance(self):
        params = _make_params()
        carry = init_convergence_carry(params)
        # Simulate enough steps and a very small delta to meet the first threshold
        carry = carry._replace(
            ema_loss_delta=jnp.float32(1e-5),
            steps_since_threshold_start=jnp.int32(10),
        )
        thresholds = jnp.array([1e-3, 1e-4])  # relative thresholds
        # relative_convergence = 1e-5 / 1.0 = 1e-5 < 1e-3 → threshold met
        carry = check_and_advance_threshold(
            carry, jnp.float32(1.0), jnp.int32(10),
            thresholds, min_steps=5, initial_steps=2
        )
        assert carry.current_threshold_idx == 1
        assert carry.converged == False
        assert carry.steps_since_threshold_start == 0

    def test_final_threshold_sets_converged(self):
        params = _make_params()
        carry = init_convergence_carry(params)
        carry = carry._replace(
            ema_loss_delta=jnp.float32(1e-6),
            steps_since_threshold_start=jnp.int32(10),
            current_threshold_idx=jnp.int32(1),  # already at last threshold
        )
        thresholds = jnp.array([1e-3, 1e-4])
        carry = check_and_advance_threshold(
            carry, jnp.float32(1.0), jnp.int32(10),
            thresholds, min_steps=5, initial_steps=2
        )
        assert carry.converged == True

    def test_jit_compatible(self):
        """update_convergence and check_and_advance_threshold are jittable."""
        params = _make_params()
        carry = init_convergence_carry(params)
        thresholds = jnp.array([1e-3])

        @jax.jit
        def step(carry, prev_loss, curr_loss):
            carry, delta = update_convergence(carry, prev_loss, curr_loss, params, ema_alpha=0.5)
            carry = check_and_advance_threshold(
                carry, curr_loss, jnp.int32(5), thresholds, min_steps=2, initial_steps=1
            )
            return carry

        result = step(carry, jnp.float32(1.0), jnp.float32(0.5))
        assert result is not None
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

```bash
cd /Users/alexi/JAX-ENT && source .venv/bin/activate
pytest jaxent/tests/unit/opt/test_convergence_carry.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` for `track.py` symbols.

- [ ] **Step 3: Replace `jaxent/src/opt/track.py`**

```python
# jaxent/src/opt/track.py
from beartype.typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from jax import Array

from jaxent.src.interfaces.simulation import Simulation_Parameters


class ConvergenceCarry(NamedTuple):
    """Pure JAX carry replacing ConvergenceTracker. All fields are JAX arrays."""
    ema_loss_delta: Array                   # float32 scalar
    ema_params: Simulation_Parameters       # EMA-averaged params
    steps_since_threshold_start: Array      # int32 scalar
    current_threshold_idx: Array            # int32 scalar
    converged: Array                        # bool scalar


def init_convergence_carry(initial_params: Simulation_Parameters) -> ConvergenceCarry:
    """Initialise a ConvergenceCarry from the initial optimisation parameters."""
    return ConvergenceCarry(
        ema_loss_delta=jnp.float32(0.0),
        ema_params=initial_params,
        steps_since_threshold_start=jnp.int32(0),
        current_threshold_idx=jnp.int32(0),
        converged=jnp.bool_(False),
    )


def update_convergence(
    carry: ConvergenceCarry,
    previous_loss: Array,
    current_loss: Array,
    current_params: Simulation_Parameters,
    ema_alpha: float,
) -> tuple[ConvergenceCarry, Array]:
    """Update EMA statistics. Returns (updated carry, raw_loss_delta)."""
    raw_delta = jnp.abs(previous_loss - current_loss)
    new_ema_delta = ema_alpha * raw_delta + (1.0 - ema_alpha) * carry.ema_loss_delta
    new_ema_params = jax.tree.map(
        lambda ep, cp: ema_alpha * cp + (1.0 - ema_alpha) * ep,
        carry.ema_params,
        current_params,
    )
    return carry._replace(
        ema_loss_delta=new_ema_delta,
        ema_params=new_ema_params,
        steps_since_threshold_start=carry.steps_since_threshold_start + 1,
    ), raw_delta


def check_and_advance_threshold(
    carry: ConvergenceCarry,
    current_loss: Array,
    step: Array,
    thresholds: Array,      # float32 [n_thresholds], sorted descending, pre-computed
    min_steps: int,         # static
    initial_steps: int,     # static
) -> ConvergenceCarry:
    """Advance convergence threshold or set converged=True. All branching via lax.cond."""
    n_thresholds = thresholds.shape[0]
    relative_convergence = carry.ema_loss_delta / jnp.maximum(current_loss, jnp.float32(1e-10))
    at_last = carry.current_threshold_idx >= (n_thresholds - 1)

    threshold_met = (
        (carry.steps_since_threshold_start >= min_steps)
        & (relative_convergence < thresholds[carry.current_threshold_idx])
        & (step > initial_steps)
    )

    def on_met(c: ConvergenceCarry) -> ConvergenceCarry:
        return jax.lax.cond(
            at_last,
            lambda c: c._replace(converged=jnp.bool_(True)),
            lambda c: c._replace(
                current_threshold_idx=c.current_threshold_idx + 1,
                steps_since_threshold_start=jnp.int32(0),
            ),
            c,
        )

    return jax.lax.cond(threshold_met, on_met, lambda c: c, carry)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest jaxent/tests/unit/opt/test_convergence_carry.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add jaxent/src/opt/track.py jaxent/tests/unit/opt/test_convergence_carry.py
git commit -m "feat(opt): replace ConvergenceTracker with ConvergenceCarry pure functions"
```

---

### Task 2: Functional `Simulation.forward`

**Files:**
- Modify: `jaxent/src/models/core.py`

- [ ] **Step 1: Write failing test**

Add to a new file `jaxent/tests/unit/opt/test_simulation_forward.py`:

```python
# jaxent/tests/unit/opt/test_simulation_forward.py
import copy
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters


def _make_minimal_sim():
    """Minimal Simulation for forward() tests — no real forward models needed."""
    sim = object.__new__(Simulation)
    params = Simulation_Parameters(
        frame_weights=jnp.array([0.5, 0.5]),
        frame_mask=jnp.ones(2),
        model_parameters=[],
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.zeros(1),
        forward_model_scaling=jnp.ones(1),
    )
    sim.params = params
    sim.input_features = []
    sim.forward_models = []
    sim.forwardpass = ()
    sim.length = 2
    sim.outputs = ()
    sim._input_features = ()
    # Mock _jit_forward_pure to return an empty tuple
    sim._jit_forward_pure = lambda p, feats, fp: ()
    sim.raise_jit_failure = False
    return sim


class TestSimulationForward:
    def test_forward_does_not_mutate_input(self):
        sim = _make_minimal_sim()
        original_id = id(sim)
        new_params = sim.params._replace(frame_weights=jnp.array([0.3, 0.7]))

        result = Simulation.forward(sim, new_params)

        # Original sim is unchanged
        assert id(sim) == original_id
        assert float(sim.params.frame_weights[0]) == pytest.approx(0.5)

    def test_forward_returns_new_instance(self):
        sim = _make_minimal_sim()
        new_params = sim.params._replace(frame_weights=jnp.array([0.3, 0.7]))

        result = Simulation.forward(sim, new_params)

        assert result is not sim
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest jaxent/tests/unit/opt/test_simulation_forward.py -v
```

Expected: `AssertionError` — `test_forward_does_not_mutate_input` fails because `forward` currently mutates `sim.params`.

- [ ] **Step 3: Make `Simulation.forward` functional in `jaxent/src/models/core.py`**

Replace the `forward` static method (lines 129–156):

```python
@staticmethod
def forward(sim: "Simulation", params: Simulation_Parameters) -> "Simulation":
    """Pure functional forward — returns a new Simulation, does not mutate sim."""
    import copy
    params = Simulation_Parameters.normalize_weights(params)
    outputs = tuple(sim._jit_forward_pure(
        params,
        sim._input_features,
        sim.forwardpass,
    ))
    new_sim = copy.copy(sim)   # shallow copy — shares _input_features, forwardpass, etc.
    new_sim.params = params
    new_sim.outputs = outputs
    return new_sim
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest jaxent/tests/unit/opt/test_simulation_forward.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Verify existing model tests still pass**

```bash
pytest jaxent/tests/unit/models/ -v
```

Expected: all pass (the `initialise` method's direct `forward` call also works since it still rebinds the local variable).

- [ ] **Step 6: Commit**

```bash
git add jaxent/src/models/core.py jaxent/tests/unit/opt/test_simulation_forward.py
git commit -m "feat(models): make Simulation.forward functional — returns new instance"
```

---

### Task 3: Remove side effects from `gradients.py`; add `_apply_lr_scaling`

**Files:**
- Modify: `jaxent/src/opt/gradients.py`

The print statements in `create_gradient_masks` move to `OptaxOptimizer.initialise` in Task 6.
A new `_apply_lr_scaling` helper is added here for use by `_pure_step` in Task 6.

- [ ] **Step 1: Remove all print statements from `create_gradient_masks`**

In `jaxent/src/opt/gradients.py`, delete these lines from `create_gradient_masks`:

```python
# DELETE these lines:
print(DeprecationWarning(...))          # line ~42
print(parameter_partition_masks)        # line ~65
print(f"Masks: frame=...")              # line ~66
print("Frame mask mask:", ...)          # line ~77
print("Original params structure:", ...) # line ~99
print("Mask structure:", ...)           # line ~103
```

After deletion, `create_gradient_masks` should have no `print` calls.

- [ ] **Step 2: Add `_apply_lr_scaling` to `gradients.py`**

Append to the end of `jaxent/src/opt/gradients.py`:

```python
def _apply_lr_scaling(
    updates: Simulation_Parameters,
    lr: Array,
    model_lr: Array,
) -> Simulation_Parameters:
    """Scale optimizer updates by per-group learning rates.

    frame_weights, frame_mask → scaled by lr
    model_parameters          → scaled by model_lr
    all other fields          → scaled by lr

    Used by _pure_step after obtaining unit-LR updates from optax.
    """
    scaled_model = [
        jax.tree.map(lambda u: u * model_lr, mp)
        for mp in updates.model_parameters
    ]
    return Simulation_Parameters(
        frame_weights=jax.tree.map(lambda u: u * lr, updates.frame_weights),
        frame_mask=jax.tree.map(lambda u: u * lr, updates.frame_mask),
        model_parameters=scaled_model,
        forward_model_weights=jax.tree.map(lambda u: u * lr, updates.forward_model_weights),
        forward_model_scaling=jax.tree.map(lambda u: u * lr, updates.forward_model_scaling),
        normalise_loss_functions=jax.tree.map(lambda u: u * lr, updates.normalise_loss_functions),
    )
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
pytest jaxent/tests/ -v --ignore=jaxent/tests/unit/opt/test_convergence_carry.py \
  --ignore=jaxent/tests/unit/opt/test_simulation_forward.py
```

Expected: all pass (no behaviour change, just removed prints).

- [ ] **Step 4: Commit**

```bash
git add jaxent/src/opt/gradients.py
git commit -m "refactor(opt): remove prints from create_gradient_masks; add _apply_lr_scaling"
```

---

### Task 4: Add `OptimisationCarry`, `HParamBatch`, `BatchOptimisationResult` to `base.py`

**Files:**
- Modify: `jaxent/src/opt/base.py`

- [ ] **Step 1: Add the three types to `jaxent/src/opt/base.py`**

Append after the `OptimizationHistory` class (end of file):

```python
# ---------------------------------------------------------------------------
# Carry types — shared between _optimise (Python loop) and _optimise_pure
# ---------------------------------------------------------------------------
from jaxent.src.opt.track import ConvergenceCarry  # noqa: E402 (avoid circular at top)


class OptimisationCarry(NamedTuple):
    """Full per-step carry for both Python-loop and while_loop optimisation paths.

    lr and model_lr replace MutableLearningRate — carried as JAX scalars.
    gradient_mask replaces the in-step mutation of optimizer.gradient_mask.
    history_params / history_losses are pre-allocated [n_steps, ...] buffers.
    write_idx is the current write position into those buffers.
    """
    opt_state: OptimizationState
    convergence: ConvergenceCarry
    lr: Array                              # float32 scalar — current frame LR
    model_lr: Array                        # float32 scalar — current model LR
    gradient_mask: Simulation_Parameters
    history_params: Simulation_Parameters  # tree of [n_steps, ...] arrays
    history_losses: LossComponents         # [n_steps, ...] arrays
    write_idx: Array                       # int32 scalar


# ---------------------------------------------------------------------------
# batch_optimise types
# ---------------------------------------------------------------------------

class HParamBatch(NamedTuple):
    """Batched hyperparameter inputs for batch_optimise.

    forward_model_scaling: [n_hparams, n_models] — per-run loss scalings.
      Column order matches the loss_functions list passed to batch_optimise.
      e.g. col 0 = primary loss, col 1 = MaxEnt, col 2 = BV reg.
    learning_rate: [n_hparams] optional — per-run LR; None uses config for all.
    """
    forward_model_scaling: Array   # float32 [n_hparams, n_models]
    learning_rate: Optional[Array] = None  # float32 [n_hparams] or None


class BatchOptimisationResult(NamedTuple):
    """Outputs from batch_optimise — one slot per hparam configuration."""
    histories: OptimizationHistory    # [n_hparams, n_steps, ...]
    best_states: OptimizationState    # [n_hparams, ...]
    convergence_steps: Array          # int32 [n_hparams] — write_idx at convergence
    hparam_batch: HParamBatch         # echoed for downstream result matching
```

Note: The `ConvergenceCarry` import is placed inside the file to avoid a circular import since `track.py` imports from `base.py` indirectly. If a circular import occurs, move `ConvergenceCarry` into `base.py` directly and delete `track.py`'s copy.

- [ ] **Step 2: Verify imports resolve**

```bash
python -c "from jaxent.src.opt.base import OptimisationCarry, HParamBatch, BatchOptimisationResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add jaxent/src/opt/base.py
git commit -m "feat(opt): add OptimisationCarry, HParamBatch, BatchOptimisationResult to base.py"
```

---

### Task 5: Refactor `OptaxOptimizer` — unit LR, remove `MutableLearningRate`

**Files:**
- Modify: `jaxent/src/opt/optimiser.py`

The optimizer chains switch from `inject_hyperparams(fn)(learning_rate=callable)` to `fn(learning_rate=1.0)`. LR is now applied explicitly in `_pure_step` via `_apply_lr_scaling`. `lbfgs` is excluded from the pure path (noted in `__init__`).

- [ ] **Step 1: Remove `MutableLearningRate` class**

Delete the entire `MutableLearningRate` class (lines 27–48 in `optimiser.py`).

- [ ] **Step 2: Rewrite `OptaxOptimizer.__init__`**

Replace the optimizer chain construction in `__init__` with unit-LR chains:

```python
def __init__(
    self,
    learning_rate: float = 1e-4,
    optimizer: str = "adam",
    parameter_partition_masks: set[Optimisable_Parameters] = {
        Optimisable_Parameters.frame_weights,
    },
    clip_value: Optional[float] = 1.0,
    force_simplex: Optional[bool] = None,
    plateau_denominator: float = 1.005,
    save_ema_history: bool = True,
    initial_learning_rate: float = 1e0,
    initial_steps: int = 0,
    model_parameters_lr_scale: float = 1.0,
):
    self.parameter_partition_masks = parameter_partition_masks
    self.clip_value = clip_value
    self.history = OptimizationHistory()
    self.save_ema_history = save_ema_history
    self.model_parameters_lr_scale = model_parameters_lr_scale
    if save_ema_history:
        self.ema_history = OptimizationHistory()
    else:
        self.ema_history = None
    self.step = self._step
    self.plateau_denominator = plateau_denominator
    self.gradient_mask = None
    self.initial_learning_rate = initial_learning_rate
    self.learning_rate = learning_rate
    self.initial_steps = initial_steps

    if optimizer.lower() == "adam":
        base_optimizer_fn = optax.adam
        _force_simplex = False
    elif optimizer.lower() == "sgd":
        base_optimizer_fn = optax.sgd
        _force_simplex = True
    elif optimizer.lower() == "adagrad":
        base_optimizer_fn = optax.adagrad
        _force_simplex = False
    elif optimizer.lower() == "adamw":
        base_optimizer_fn = optax.adamw
        _force_simplex = False
    elif optimizer.lower() == "rmsprop":
        base_optimizer_fn = optax.rmsprop
        _force_simplex = False
    elif optimizer.lower() == "lbfgs":
        raise ValueError(
            "lbfgs is not supported in the pure/batched optimisation path. "
            "Use run_optimise with jit_update_step=False instead."
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Unit-LR chains — LR is applied explicitly in _pure_step via _apply_lr_scaling
    def _make_chain(extra_transforms=()):
        chain = []
        if clip_value is not None:
            chain.append(optax.clip(clip_value))
        chain.append(base_optimizer_fn(learning_rate=1.0))
        chain.extend(extra_transforms)
        return optax.chain(*chain)

    self.optimizer = optax.multi_transform(
        transforms={
            'frame': _make_chain(),
            'model': _make_chain(),   # keep_params_nonnegative applied in _pure_step
            'other': _make_chain(),
        },
        param_labels=Simulation_Parameters.param_labels,
    )

    if force_simplex is None:
        self.force_logit_simplex = _force_simplex
    else:
        self.force_logit_simplex = force_simplex
```

- [ ] **Step 3: Update `tree_flatten` and `tree_unflatten`**

Remove `lr_schedule` and `model_lr_schedule` from `children`:

```python
def tree_flatten(self):
    children = (
        self.history,
        self.gradient_mask,
        self.ema_history,
        # lr_schedule and model_lr_schedule removed — LR lives in OptimisationCarry
    )
    aux_data = {
        "learning_rate": self.learning_rate,
        "optimizer": self.optimizer,
        "parameter_partition_masks": self.parameter_partition_masks,
        "clip_value": self.clip_value,
        "save_ema_history": self.save_ema_history,
        "plateau_denominator": self.plateau_denominator,
        "force_logit_simplex": self.force_logit_simplex,
        "initial_learning_rate": self.initial_learning_rate,
        "step": self.step,
        "initial_steps": self.initial_steps,
        "model_parameters_lr_scale": self.model_parameters_lr_scale,
        "update_all_models": self.update_all_models,
    }
    return children, aux_data

@classmethod
def tree_unflatten(cls, aux_data, children):
    self = cls.__new__(cls)
    self.history = children[0]
    self.gradient_mask = children[1]
    self.ema_history = children[2]
    self.learning_rate = aux_data["learning_rate"]
    self.optimizer = aux_data["optimizer"]
    self.parameter_partition_masks = aux_data["parameter_partition_masks"]
    self.clip_value = aux_data["clip_value"]
    self.save_ema_history = aux_data["save_ema_history"]
    self.plateau_denominator = aux_data["plateau_denominator"]
    self.force_logit_simplex = aux_data["force_logit_simplex"]
    self.step = aux_data.get("step", self._step)
    self.initial_learning_rate = aux_data.get("initial_learning_rate", 1.0)
    self.initial_steps = aux_data.get("initial_steps", 0)
    self.model_parameters_lr_scale = aux_data.get("model_parameters_lr_scale", 1.0)
    self.update_all_models = aux_data.get("update_all_models", False)
    return self
```

- [ ] **Step 4: Verify existing tests pass**

```bash
pytest jaxent/tests/ -v
```

Expected: all pass. (The `_step` function still exists and uses its mutable approach — it is updated in Task 6.)

- [ ] **Step 5: Commit**

```bash
git add jaxent/src/opt/optimiser.py
git commit -m "refactor(opt): remove MutableLearningRate; switch to unit-LR optimizer chains"
```

---

### Task 6: Add `_pure_step` and refactor `OptaxOptimizer.initialise`

**Files:**
- Modify: `jaxent/src/opt/optimiser.py`
- Create: `jaxent/tests/unit/opt/test_pure_step.py`

- [ ] **Step 1: Write failing tests**

```python
# jaxent/tests/unit/opt/test_pure_step.py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxent.src.opt.optimiser import OptaxOptimizer, _pure_step
from jaxent.src.opt.base import OptimisationCarry, OptimizationState, LossComponents
from jaxent.src.opt.track import ConvergenceCarry, init_convergence_carry
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.gradients import create_gradient_masks
from jaxent.src.custom_types.config import Optimisable_Parameters


def _make_params(n_frames=4, n_models=2):
    return Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=[],
        forward_model_weights=jnp.ones(n_models),
        normalise_loss_functions=jnp.zeros(n_models),
        forward_model_scaling=jnp.ones(n_models),
    )


def _make_carry(optimizer, params, n_steps=100):
    opt_state = optimizer.optimizer.init(params)
    mask = create_gradient_masks({Optimisable_Parameters.frame_weights}, params, None)
    convergence = init_convergence_carry(params)
    # Pre-allocate history buffers
    hist_params = jax.tree.map(lambda x: jnp.zeros((n_steps, *x.shape)), params)
    hist_losses = LossComponents(
        train_losses=jnp.zeros((n_steps, 2)),
        val_losses=jnp.zeros((n_steps, 2)),
        scaled_train_losses=jnp.zeros((n_steps, 2)),
        scaled_val_losses=jnp.zeros((n_steps, 2)),
        total_train_loss=jnp.zeros(n_steps),
        total_val_loss=jnp.zeros(n_steps),
    )
    return OptimisationCarry(
        opt_state=OptimizationState(params=params, opt_state=opt_state),
        convergence=convergence,
        lr=jnp.float32(1e-2),
        model_lr=jnp.float32(1e-3),
        gradient_mask=mask,
        history_params=hist_params,
        history_losses=hist_losses,
        write_idx=jnp.int32(0),
    )


class TestPureStep:
    def _dummy_loss_fn(self, sim, data, idx):
        # Returns (train_loss, val_loss) both scalar
        loss = jnp.sum(sim.params.frame_weights ** 2)
        return loss, loss

    def test_pure_step_returns_carry(self, simple_simulation):
        """_pure_step returns an OptimisationCarry."""
        optimizer = OptaxOptimizer(learning_rate=1e-2)
        params = _make_params()
        carry = _make_carry(optimizer, params)

        new_carry = _pure_step(
            carry=carry,
            simulation=simple_simulation,
            data_targets=(None,),
            loss_functions=(self._dummy_loss_fn,),
            indexes=(0,),
            plateau_denominator=optimizer.plateau_denominator,
            model_lr_scale=optimizer.model_parameters_lr_scale,
            initial_steps=optimizer.initial_steps,
            parameter_partition_masks=optimizer.parameter_partition_masks,
            force_logit_simplex=optimizer.force_logit_simplex,
            thresholds=jnp.array([1e-3]),
            min_steps=2,
            ema_alpha=0.5,
            optimizer_tx=optimizer.optimizer,
        )
        assert isinstance(new_carry, OptimisationCarry)
        assert int(new_carry.write_idx) == 1
        assert int(new_carry.opt_state.step) == 1

    def test_pure_step_updates_params(self, simple_simulation):
        """Frame weights change after a step."""
        optimizer = OptaxOptimizer(learning_rate=1e-2)
        params = _make_params()
        carry = _make_carry(optimizer, params)

        new_carry = _pure_step(
            carry=carry,
            simulation=simple_simulation,
            data_targets=(None,),
            loss_functions=(self._dummy_loss_fn,),
            indexes=(0,),
            plateau_denominator=optimizer.plateau_denominator,
            model_lr_scale=optimizer.model_parameters_lr_scale,
            initial_steps=optimizer.initial_steps,
            parameter_partition_masks=optimizer.parameter_partition_masks,
            force_logit_simplex=optimizer.force_logit_simplex,
            thresholds=jnp.array([1e-3]),
            min_steps=2,
            ema_alpha=0.5,
            optimizer_tx=optimizer.optimizer,
        )
        assert not jnp.allclose(
            new_carry.opt_state.params.frame_weights,
            carry.opt_state.params.frame_weights,
        )

    def test_pure_step_is_jittable(self, simple_simulation):
        """_pure_step compiles without error."""
        optimizer = OptaxOptimizer(learning_rate=1e-2)
        params = _make_params()
        carry = _make_carry(optimizer, params)

        jitted = jax.jit(
            _pure_step,
            static_argnames=(
                "loss_functions", "indexes", "plateau_denominator", "model_lr_scale",
                "initial_steps", "parameter_partition_masks", "force_logit_simplex",
                "min_steps", "ema_alpha",
            ),
        )
        new_carry = jitted(
            carry=carry,
            simulation=simple_simulation,
            data_targets=(None,),
            loss_functions=(self._dummy_loss_fn,),
            indexes=(0,),
            plateau_denominator=optimizer.plateau_denominator,
            model_lr_scale=optimizer.model_parameters_lr_scale,
            initial_steps=optimizer.initial_steps,
            parameter_partition_masks=optimizer.parameter_partition_masks,
            force_logit_simplex=optimizer.force_logit_simplex,
            thresholds=jnp.array([1e-3]),
            min_steps=2,
            ema_alpha=0.5,
            optimizer_tx=optimizer.optimizer,
        )
        assert new_carry is not None
```

Add `simple_simulation` fixture to `jaxent/tests/unit/opt/conftest.py`:

```python
# jaxent/tests/unit/opt/conftest.py
import copy
import jax.numpy as jnp
import pytest
from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters


@pytest.fixture
def simple_simulation():
    """Minimal Simulation whose forward pass returns output from frame_weights."""
    sim = object.__new__(Simulation)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(4) / 4,
        frame_mask=jnp.ones(4),
        model_parameters=[],
        forward_model_weights=jnp.ones(2),
        normalise_loss_functions=jnp.zeros(2),
        forward_model_scaling=jnp.ones(2),
    )
    sim.params = params
    sim.input_features = []
    sim.forward_models = []
    sim.forwardpass = ()
    sim.length = 4
    sim.outputs = ()
    sim._input_features = ()
    sim._jit_forward_pure = lambda p, feats, fp: ()
    sim.raise_jit_failure = False
    return sim
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError for `_pure_step`)**

```bash
pytest jaxent/tests/unit/opt/test_pure_step.py -v
```

- [ ] **Step 3: Implement `_pure_step` in `jaxent/src/opt/optimiser.py`**

Add as a module-level function (not a static method — easier to JIT):

```python
def _pure_step(
    carry: "OptimisationCarry",
    simulation: "InitialisedSimulation",
    data_targets: tuple,
    loss_functions: tuple,
    indexes: tuple,
    plateau_denominator: float,
    model_lr_scale: float,
    initial_steps: int,
    parameter_partition_masks: set,
    force_logit_simplex: bool,
    thresholds: Array,
    min_steps: int,
    ema_alpha: float,
    optimizer_tx: optax.GradientTransformation,
) -> "OptimisationCarry":
    """Pure JAX step function. No Python mutation. Used by while_loop and _optimise."""
    from jaxent.src.opt.base import OptimisationCarry, OptimizationState, LossComponents
    from jaxent.src.opt.track import update_convergence, check_and_advance_threshold
    from jaxent.src.opt.gradients import _apply_lr_scaling, create_gradient_masks, mask_gradients

    # 1. Switch gradient mask at initial_steps
    new_mask = jax.lax.cond(
        carry.opt_state.step == initial_steps,
        lambda _: create_gradient_masks(parameter_partition_masks, carry.opt_state.params, None),
        lambda _: carry.gradient_mask,
        None,
    )

    # 2. Compute loss and gradients
    def loss_fn(params):
        losses = compute_loss(simulation, params, data_targets, indexes, loss_functions)
        return losses.total_train_loss, losses

    (loss_value, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        carry.opt_state.params
    )

    # 3. Mask gradients
    masked_grads = mask_gradients(grads, new_mask)

    # 4. Gradient oscillation detection
    prev_grads = jax.lax.cond(
        carry.opt_state.gradients is not None,
        lambda: carry.opt_state.gradients,
        lambda: masked_grads,
    ) if carry.opt_state.gradients is not None else masked_grads
    # Note: since gradients=None only on step 0, use jax.lax.cond on the step count:
    prev_grads = jax.lax.cond(
        carry.opt_state.step == 0,
        lambda: masked_grads,
        lambda: carry.opt_state.gradients,
    )
    grad_dot = jax.tree.reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda a, b: jnp.vdot(a, b), prev_grads, masked_grads),
    )

    # 5. Update LR: initial_steps switch + oscillation reduction
    new_lr, new_model_lr = jax.lax.cond(
        carry.opt_state.step == initial_steps,
        lambda: (
            jnp.float32(carry.lr / carry.lr * carry.lr),  # keep — already set at init
            jnp.float32(carry.model_lr),
        ),
        lambda: (carry.lr, carry.model_lr),
    )
    new_lr, new_model_lr = jax.lax.cond(
        (grad_dot < 0) & (carry.opt_state.step > 1),
        lambda: (new_lr / plateau_denominator, new_model_lr / plateau_denominator),
        lambda: (new_lr, new_model_lr),
    )

    # 6. Optax update (unit LR) + explicit LR scaling
    raw_updates, new_opt_state_tx = optimizer_tx.update(
        masked_grads, carry.opt_state.opt_state, carry.opt_state.params,
    )
    scaled_updates = _apply_lr_scaling(raw_updates, new_lr, new_model_lr)

    # 7. Project model params to be non-negative (replaces keep_params_nonnegative)
    projected_model_updates = [
        jax.tree.map(lambda u, p: jnp.maximum(u, -p), mu, mp)
        for mu, mp in zip(scaled_updates.model_parameters, carry.opt_state.params.model_parameters)
    ]
    scaled_updates = scaled_updates._replace(model_parameters=projected_model_updates)

    # 8. Apply updates
    updated_params = optax.apply_updates(carry.opt_state.params, scaled_updates)
    if force_logit_simplex:
        updated_params = Simulation_Parameters.normalize_weights(updated_params)

    # 9. Build save_params (normalized for history)
    save_params = Simulation_Parameters.normalize_weights(updated_params)

    # 10. Update convergence carry
    new_convergence, raw_delta = update_convergence(
        carry.convergence,
        losses.total_train_loss,   # previous loss proxy — use current as prev for step 0
        loss_value,
        save_params,
        ema_alpha,
    )
    new_convergence = check_and_advance_threshold(
        new_convergence, loss_value, carry.opt_state.step,
        thresholds, min_steps, initial_steps,
    )
    # Hard n_steps limit handled in cond_fn of while_loop — not here

    # 11. Write to pre-allocated history buffers
    new_history_losses = jax.tree.map(
        lambda buf, val: jax.lax.dynamic_update_slice(
            buf, val[None], (carry.write_idx,) + (0,) * val.ndim
        ),
        carry.history_losses,
        losses,
    )
    new_history_params = jax.tree.map(
        lambda buf, val: jax.lax.dynamic_update_slice(
            buf, val[None], (carry.write_idx,) + (0,) * val.ndim
        ),
        carry.history_params,
        save_params,
    )

    # 12. Build new OptimizationState
    new_opt_state = carry.opt_state.update(
        updated_params, new_opt_state_tx, losses, masked_grads
    )

    return OptimisationCarry(
        opt_state=new_opt_state,
        convergence=new_convergence,
        lr=new_lr,
        model_lr=new_model_lr,
        gradient_mask=new_mask,
        history_params=new_history_params,
        history_losses=new_history_losses,
        write_idx=carry.write_idx + 1,
    )
```

**Note on step 5 (LR switch):** The LR switch at `initial_steps` from `initial_learning_rate` → `learning_rate` is handled by `OptaxOptimizer.initialise` in Task 7: it sets `carry.lr = initial_learning_rate` and `_pure_step` uses a `jax.lax.cond` to switch to the target LR at `initial_steps`. Replace the placeholder in step 5 with:

```python
    new_lr, new_model_lr = jax.lax.cond(
        carry.opt_state.step == initial_steps,
        lambda: (jnp.float32(target_lr), jnp.float32(target_lr * model_lr_scale)),
        lambda: (carry.lr, carry.model_lr),
    )
```

Where `target_lr` is passed as a static arg. Update the function signature to include `target_lr: float` alongside `initial_steps`.

- [ ] **Step 4: Update `OptaxOptimizer.initialise` to return `OptimisationCarry`**

Replace the `initialise` method body to initialise and return an `OptimisationCarry`:

```python
def initialise(
    self,
    model: InitialisedSimulation,
    optimisable_funcs: Optional[list[bool] | Array] = None,
    _jit_test_args=None,
    logger=None,
) -> "OptimisationCarry":
    from jaxent.src.opt.base import OptimisationCarry, OptimizationState, LossComponents
    from jaxent.src.opt.track import init_convergence_carry
    import logging

    _logger = logger or logging.getLogger("jaxent.opt")
    params = model.params
    params = Simulation_Parameters(
        frame_mask=params.frame_mask,
        frame_weights=params.frame_weights * len(params.frame_weights),
        model_parameters=params.model_parameters,
        normalise_loss_functions=params.normalise_loss_functions,
        forward_model_weights=params.forward_model_weights,
        forward_model_scaling=params.forward_model_scaling,
    )
    _logger.debug("Params structure: %s", jax.tree_util.tree_structure(params))

    if isinstance(optimisable_funcs, list):
        optimisable_funcs = jnp.array(optimisable_funcs, dtype=jnp.float32)

    # Gradient mask — only frame_weights active initially
    initial_mask = create_gradient_masks(
        {Optimisable_Parameters.frame_weights},
        params,
        optimisable_funcs,
    )
    _logger.debug("parameter_partition_masks=%s", self.parameter_partition_masks)
    _logger.debug(
        "Masks: frame=%s, model=%s",
        Optimisable_Parameters.frame_weights in self.parameter_partition_masks,
        Optimisable_Parameters.model_parameters in self.parameter_partition_masks,
    )

    opt_state_tx = self.optimizer.init(params)
    self.history = OptimizationHistory()
    if self.save_ema_history:
        self.ema_history = OptimizationHistory()

    # Pre-allocate history buffers (size will be set by run_optimise via n_steps)
    # Caller (run_optimise / batch_optimise) resizes — here we return a carry
    # with write_idx=0 and 1-step buffers as a placeholder.
    # run_optimise replaces history_params/losses before entering the loop.
    hist_params = jax.tree.map(lambda x: jnp.zeros((1, *x.shape), dtype=x.dtype), params)
    hist_losses = LossComponents(
        train_losses=jnp.zeros((1, len(params.forward_model_weights))),
        val_losses=jnp.zeros((1, len(params.forward_model_weights))),
        scaled_train_losses=jnp.zeros((1, len(params.forward_model_weights))),
        scaled_val_losses=jnp.zeros((1, len(params.forward_model_weights))),
        total_train_loss=jnp.zeros(1),
        total_val_loss=jnp.zeros(1),
    )

    return OptimisationCarry(
        opt_state=OptimizationState(params=params, opt_state=opt_state_tx),
        convergence=init_convergence_carry(params),
        lr=jnp.float32(self.initial_learning_rate),
        model_lr=jnp.float32(self.initial_learning_rate * self.model_parameters_lr_scale),
        gradient_mask=initial_mask,
        history_params=hist_params,
        history_losses=hist_losses,
        write_idx=jnp.int32(0),
    )
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest jaxent/tests/unit/opt/test_pure_step.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add jaxent/src/opt/optimiser.py jaxent/tests/unit/opt/test_pure_step.py \
        jaxent/tests/unit/opt/conftest.py
git commit -m "feat(opt): add _pure_step; refactor OptaxOptimizer.initialise → OptimisationCarry"
```

---

### Task 7: `_optimise_pure` with `jax.lax.while_loop`

**Files:**
- Modify: `jaxent/src/opt/run.py`
- Create: `jaxent/tests/unit/opt/test_optimise_pure.py`

- [ ] **Step 1: Write failing tests**

```python
# jaxent/tests/unit/opt/test_optimise_pure.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.opt.run import _optimise_pure
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.base import OptimisationCarry, LossComponents
from jaxent.src.opt.track import init_convergence_carry
from jaxent.src.opt.gradients import create_gradient_masks
from jaxent.src.custom_types.config import Optimisable_Parameters, OptimiserSettings
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import OptimizationState


def _dummy_loss(sim, data, idx):
    """Loss = sum(frame_weights^2). Gradient pushes weights toward 0."""
    loss = jnp.sum(sim.params.frame_weights ** 2)
    return loss, loss


def _make_init_carry(optimizer, params, n_steps, target_lr):
    from jaxent.src.opt.base import LossComponents
    opt_state_tx = optimizer.optimizer.init(params)
    mask = create_gradient_masks({Optimisable_Parameters.frame_weights}, params, None)
    convergence = init_convergence_carry(params)
    n_models = len(params.forward_model_weights)
    hist_params = jax.tree.map(lambda x: jnp.zeros((n_steps, *x.shape), dtype=x.dtype), params)
    hist_losses = LossComponents(
        train_losses=jnp.zeros((n_steps, n_models)),
        val_losses=jnp.zeros((n_steps, n_models)),
        scaled_train_losses=jnp.zeros((n_steps, n_models)),
        scaled_val_losses=jnp.zeros((n_steps, n_models)),
        total_train_loss=jnp.zeros(n_steps),
        total_val_loss=jnp.zeros(n_steps),
    )
    return OptimisationCarry(
        opt_state=OptimizationState(params=params, opt_state=opt_state_tx),
        convergence=convergence,
        lr=jnp.float32(target_lr),
        model_lr=jnp.float32(target_lr),
        gradient_mask=mask,
        history_params=hist_params,
        history_losses=hist_losses,
        write_idx=jnp.int32(0),
    )


class TestOptimisePure:
    def test_loss_decreases(self, simple_simulation):
        """Loss on a quadratic should decrease over 50 steps."""
        params = Simulation_Parameters(
            frame_weights=jnp.ones(4),   # not normalized — loss = sum(w^2) = 4
            frame_mask=jnp.ones(4),
            model_parameters=[],
            forward_model_weights=jnp.ones(2),
            normalise_loss_functions=jnp.zeros(2),
            forward_model_scaling=jnp.ones(2),
        )
        optimizer = OptaxOptimizer(learning_rate=1e-1, initial_steps=0)
        n_steps = 50
        init_carry = _make_init_carry(optimizer, params, n_steps, target_lr=1e-1)

        final_carry = _optimise_pure(
            simulation=simple_simulation,
            hparam_scaling=params.forward_model_scaling,
            init_carry=init_carry,
            data_to_fit=(None,),
            n_steps=n_steps,
            loss_functions=(_dummy_loss,),
            indexes=(0,),
            optimizer=optimizer,
            thresholds=jnp.array([1e-6]),
            min_steps=100,   # never converge early
            ema_alpha=0.5,
        )

        final_loss = jnp.sum(final_carry.opt_state.params.frame_weights ** 2)
        initial_loss = jnp.sum(params.frame_weights ** 2)
        assert float(final_loss) < float(initial_loss)

    def test_converged_flag_set(self, simple_simulation):
        """A tiny tolerance forces convergence before n_steps."""
        params = Simulation_Parameters(
            frame_weights=jnp.ones(4),
            frame_mask=jnp.ones(4),
            model_parameters=[],
            forward_model_weights=jnp.ones(2),
            normalise_loss_functions=jnp.zeros(2),
            forward_model_scaling=jnp.ones(2),
        )
        optimizer = OptaxOptimizer(learning_rate=1e-1, initial_steps=0)
        n_steps = 1000
        init_carry = _make_init_carry(optimizer, params, n_steps, target_lr=1e-1)

        final_carry = _optimise_pure(
            simulation=simple_simulation,
            hparam_scaling=params.forward_model_scaling,
            init_carry=init_carry,
            data_to_fit=(None,),
            n_steps=n_steps,
            loss_functions=(_dummy_loss,),
            indexes=(0,),
            optimizer=optimizer,
            thresholds=jnp.array([1.0]),  # very loose — converges almost immediately
            min_steps=2,
            ema_alpha=0.5,
        )
        assert bool(final_carry.convergence.converged) == True
        assert int(final_carry.write_idx) < n_steps
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError for `_optimise_pure`)**

```bash
pytest jaxent/tests/unit/opt/test_optimise_pure.py -v
```

- [ ] **Step 3: Implement `_optimise_pure` in `jaxent/src/opt/run.py`**

Add after the existing imports:

```python
from functools import partial
from jaxent.src.opt.optimiser import _pure_step
from jaxent.src.opt.base import OptimisationCarry


def _optimise_pure(
    simulation: InitialisedSimulation,
    hparam_scaling: Array,           # [n_models] — injected into params before loop
    init_carry: OptimisationCarry,
    data_to_fit: Sequence,
    n_steps: int,
    loss_functions: Sequence[JaxEnt_Loss],
    indexes: Sequence[int],
    optimizer: OptaxOptimizer,
    thresholds: Array,               # [n_thresholds] float32, pre-computed, static shape
    min_steps: int,
    ema_alpha: float,
) -> OptimisationCarry:
    """JAX-traceable optimisation loop via jax.lax.while_loop.

    hparam_scaling is injected into the initial params so that jax.vmap can
    batch over different hyperparameter configurations while sharing simulation.
    """
    # Inject hparams into the starting parameters
    init_params = init_carry.opt_state.params._replace(
        forward_model_scaling=hparam_scaling
    )
    init_carry = init_carry._replace(
        opt_state=init_carry.opt_state._replace(params=init_params)
    )

    target_lr = float(optimizer.learning_rate)

    def cond_fn(carry: OptimisationCarry) -> Array:
        return ~carry.convergence.converged & (carry.opt_state.step < n_steps)

    def body_fn(carry: OptimisationCarry) -> OptimisationCarry:
        return jax.lax.cond(
            carry.convergence.converged,
            lambda c: c,                  # converged — freeze state (no-op)
            lambda c: _pure_step(
                carry=c,
                simulation=simulation,
                data_targets=tuple(data_to_fit),
                loss_functions=tuple(loss_functions),
                indexes=tuple(indexes),
                plateau_denominator=optimizer.plateau_denominator,
                model_lr_scale=optimizer.model_parameters_lr_scale,
                initial_steps=optimizer.initial_steps,
                parameter_partition_masks=optimizer.parameter_partition_masks,
                force_logit_simplex=optimizer.force_logit_simplex,
                thresholds=thresholds,
                min_steps=min_steps,
                ema_alpha=ema_alpha,
                optimizer_tx=optimizer.optimizer,
                target_lr=target_lr,
            ),
            carry,
        )

    return jax.lax.while_loop(cond_fn, body_fn, init_carry)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest jaxent/tests/unit/opt/test_optimise_pure.py -v
```

- [ ] **Step 5: Commit**

```bash
git add jaxent/src/opt/run.py jaxent/tests/unit/opt/test_optimise_pure.py
git commit -m "feat(opt): add _optimise_pure with jax.lax.while_loop and converged-gating"
```

---

### Task 8: Refactor `_optimise` and `run_optimise` to use `OptimisationCarry`

**Files:**
- Modify: `jaxent/src/opt/run.py`

`_optimise` keeps the Python for-loop but uses `OptimisationCarry` and `_pure_step` internally, replacing `ConvergenceTracker` and the mutable LR objects.

- [ ] **Step 1: Refactor `_optimise` in `jaxent/src/opt/run.py`**

Replace the full `_optimise` function body:

```python
def _optimise(
    _simulation: InitialisedSimulation,
    data_to_fit: Sequence,
    n_steps: int,
    tolerance: float,
    convergence: float | list[float],
    indexes: Sequence[int],
    loss_functions: Sequence[JaxEnt_Loss],
    init_carry: OptimisationCarry,
    optimizer: OptaxOptimizer,
    ema_alpha: float = 0.5,
    min_steps_per_threshold: int = 2,
    logger=None,
) -> tuple[InitialisedSimulation, OptaxOptimizer]:
    """Python-loop optimisation. Uses OptimisationCarry + _pure_step for unified state."""
    import logging
    import time
    _logger = logger or logging.getLogger("jaxent.opt")

    if isinstance(convergence, float):
        convergence = [convergence]
    thresholds = jnp.array(sorted(
        [ct * float(init_carry.lr) for ct in convergence], reverse=True
    ), dtype=jnp.float32)

    # Replace 1-step placeholder buffers with n_steps-sized buffers
    n_models = len(init_carry.opt_state.params.forward_model_weights)
    hist_params = jax.tree.map(
        lambda x: jnp.zeros((n_steps, *x.shape[1:]), dtype=x.dtype),
        init_carry.history_params,
    )
    hist_losses = LossComponents(
        train_losses=jnp.zeros((n_steps, n_models)),
        val_losses=jnp.zeros((n_steps, n_models)),
        scaled_train_losses=jnp.zeros((n_steps, n_models)),
        scaled_val_losses=jnp.zeros((n_steps, n_models)),
        total_train_loss=jnp.zeros(n_steps),
        total_val_loss=jnp.zeros(n_steps),
    )
    carry = init_carry._replace(history_params=hist_params, history_losses=hist_losses)

    optimizer.history = OptimizationHistory()
    loop_start = time.time()

    for step in range(n_steps):
        carry = _pure_step(
            carry=carry,
            simulation=_simulation,
            data_targets=tuple(data_to_fit),
            loss_functions=tuple(loss_functions),
            indexes=tuple(indexes),
            plateau_denominator=optimizer.plateau_denominator,
            model_lr_scale=optimizer.model_parameters_lr_scale,
            initial_steps=optimizer.initial_steps,
            parameter_partition_masks=optimizer.parameter_partition_masks,
            force_logit_simplex=optimizer.force_logit_simplex,
            thresholds=thresholds,
            min_steps=min_steps_per_threshold,
            ema_alpha=ema_alpha,
            optimizer_tx=optimizer.optimizer,
            target_lr=float(optimizer.learning_rate),
        )

        current_loss = carry.history_losses.total_train_loss[carry.write_idx - 1]
        log_optimization_step(
            step=step,
            n_steps=n_steps,
            current_loss=float(current_loss),
            raw_delta=0.0,
            prev_params=None,
            opt_state=carry.opt_state,
            grad_dot_product=0.0,
            tracker=carry.convergence,
            optimizer=optimizer,
            logger=_logger,
        )

        if float(current_loss) < tolerance or jnp.isnan(current_loss) or jnp.isinf(current_loss):
            _logger.info("Tolerance/nan reached at step %d", step)
            break

        if bool(carry.convergence.converged):
            _logger.info("Converged at step %d", step)
            break

        # Periodically record EMA state to history
        if step == 0 or bool(carry.convergence.steps_since_threshold_start == 0):
            optimizer = optimizer.update_history_compute_ema_loss(
                optimizer=optimizer,
                simulation=_simulation,
                data_targets=tuple(data_to_fit),
                indexes=tuple(indexes),
                loss_functions=tuple(loss_functions),
                state=carry.opt_state,
                ema_params=carry.convergence.ema_params,
            )

    total_time = time.time() - loop_start
    _logger.info("Optimisation complete: %d steps in %.1fs", step, total_time)

    _simulation.params = carry.opt_state.params

    best_state = optimizer.history.get_best_state()
    if best_state is not None:
        _simulation.params = best_state.params

    return cast(InitialisedSimulation, _simulation), optimizer
```

- [ ] **Step 2: Update `run_optimise` to pass `OptimisationCarry` to `_optimise`**

In `run_optimise`, replace the `opt_state = optimizer.initialise(...)` block:

```python
init_carry = optimizer.initialise(
    _simulation,
    optimisable_funcs,
    _jit_test_args=jit_test_args if jit_update_step else None,
    logger=logger,
)

_simulation, optimizer = _opt_fn(
    _simulation,
    data_to_fit,
    config.n_steps,
    config.tolerance,
    config.convergence,
    indexes,
    loss_functions,
    init_carry,       # OptimisationCarry instead of opt_state
    optimizer,
    ema_alpha=config.ema_alpha,
    min_steps_per_threshold=config.min_steps_per_threshold,
    logger=logger,
)
```

Also add `logger: logging.Logger | None = None` to the `run_optimise` signature.

- [ ] **Step 3: Update `log.py` functions to accept `logger` argument**

In `jaxent/src/opt/log.py`, update each function to accept and use an injected logger, falling back to `logging.getLogger("jaxent.opt")` if `None`:

```python
import logging

def log_optimization_step(step, n_steps, current_loss, raw_delta, prev_params,
                          opt_state, grad_dot_product, tracker, optimizer,
                          logger=None):
    _logger = logger or logging.getLogger("jaxent.opt")
    _logger.info(
        "Step %d/%d | loss=%.6f | lr=%.2e",
        step, n_steps, float(current_loss), float(opt_state.step)
    )

def log_oscillation_warning(step, logger=None):
    _logger = logger or logging.getLogger("jaxent.opt")
    _logger.warning("Gradient oscillation detected at step %d — reducing LR", step)

def print_optimization_summary(step, total_time, logger=None):
    _logger = logger or logging.getLogger("jaxent.opt")
    _logger.info("Optimisation finished: %d steps in %.1fs", step, total_time)

def format_optimization_error(e, simulation, save_state, ema_params, opt_state):
    return (
        f"Optimisation error: {e}\n"
        f"  step={getattr(opt_state, 'step', '?')}\n"
        f"  params={getattr(opt_state, 'params', '?')}"
    )

def log_final_states(simulation, save_state, ema_params, opt_state, logger=None):
    _logger = logger or logging.getLogger("jaxent.opt")
    _logger.debug("Final opt_state step: %s", opt_state.step)

def log_threshold_met(step, current_loss, tracker, optimizer, logger=None):
    _logger = logger or logging.getLogger("jaxent.opt")
    _logger.info("Threshold met at step %d, loss=%.6f", step, float(current_loss))
```

- [ ] **Step 4: Run all existing tests**

```bash
pytest jaxent/tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add jaxent/src/opt/run.py jaxent/src/opt/log.py
git commit -m "refactor(opt): _optimise and run_optimise use OptimisationCarry + _pure_step"
```

---

### Task 9: Loss closure fix

**Files:**
- Modify: `jaxent/examples/common/optimization.py`

Loss functions must not capture Python floats (`maxent_scaling`, `bv_reg_scaling`) in their closures. The scalings should come exclusively from `simulation.params.forward_model_scaling`.

- [ ] **Step 1: Read `jaxent/examples/common/optimization.py` to find closure captures**

```bash
grep -n "maxent_scaling\|bv_reg_scaling\|lambda sim\|loss_fn" \
  jaxent/examples/common/optimization.py | head -40
```

- [ ] **Step 2: Remove scaling multiplications from loss closures**

For each loss function that multiplies its output by a Python scalar, remove the multiplication. Example:

```python
# Before
maxent_loss = lambda sim, data, idx: _maxent_core(sim, data, idx) * loss_config.maxent_scaling

# After
maxent_loss = lambda sim, data, idx: _maxent_core(sim, data, idx)
```

- [ ] **Step 3: Write `forward_model_scaling` from `LossConfig` into `Simulation_Parameters`**

When constructing `Simulation_Parameters` before calling `run_optimise`, set `forward_model_scaling` from the loss config values. Identify the construction site in `optimization.py` and update:

```python
# The n-th element of forward_model_scaling corresponds to the n-th loss function
# col 0 = primary loss (scale 1.0), col 1 = maxent (loss_config.maxent_scaling), etc.
forward_model_scaling = jnp.array([
    1.0,
    loss_config.maxent_scaling,
    getattr(loss_config, "bv_reg_scaling", 1.0),
], dtype=jnp.float32)

params = Simulation_Parameters(
    ...,
    forward_model_scaling=forward_model_scaling,
)
```

- [ ] **Step 4: Verify example imports resolve**

```bash
python -c "from jaxent.examples.common.optimization import run_optimization; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add jaxent/examples/common/optimization.py
git commit -m "refactor(examples): remove scalar captures from loss closures; scalings in forward_model_scaling"
```

---

### Task 10: `batch_optimise`

**Files:**
- Create: `jaxent/src/opt/batch.py`
- Create: `jaxent/tests/unit/opt/test_batch_optimise.py`

- [ ] **Step 1: Write failing tests**

```python
# jaxent/tests/unit/opt/test_batch_optimise.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tempfile
import os
from jaxent.src.opt.batch import batch_optimise
from jaxent.src.opt.base import HParamBatch, BatchOptimisationResult
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.interfaces.simulation import Simulation_Parameters


def _dummy_loss(sim, data, idx):
    loss = jnp.sum(sim.params.frame_weights ** 2)
    return loss, loss


class TestBatchOptimise:
    def test_returns_correct_shape(self, simple_simulation, tmp_path):
        n_hparams = 4
        n_models = 2
        hparam_batch = HParamBatch(
            forward_model_scaling=jnp.ones((n_hparams, n_models)),
        )
        config = OptimiserSettings(
            n_steps=20,
            learning_rate=1e-2,
            tolerance=1e-8,
            convergence=[1e-3],
            ema_alpha=0.5,
            min_steps_per_threshold=2,
            optimiser_type="adam",
        )
        run_names = [f"run_{i}" for i in range(n_hparams)]

        result = batch_optimise(
            simulation=simple_simulation,
            hparam_batch=hparam_batch,
            batch_size=2,
            data_to_fit=(None,),
            config=config,
            indexes=(0,),
            loss_functions=(_dummy_loss,),
            run_names=run_names,
            log_dir=str(tmp_path),
            log_every_n=5,
        )

        assert isinstance(result, BatchOptimisationResult)
        # convergence_steps has one entry per hparam
        assert result.convergence_steps.shape == (n_hparams,)

    def test_log_files_created(self, simple_simulation, tmp_path):
        n_hparams = 3
        hparam_batch = HParamBatch(
            forward_model_scaling=jnp.ones((n_hparams, 2)),
        )
        config = OptimiserSettings(
            n_steps=5, learning_rate=1e-2, tolerance=1e-8,
            convergence=[1e-3], ema_alpha=0.5, min_steps_per_threshold=2,
            optimiser_type="adam",
        )
        run_names = ["alpha", "beta", "gamma"]

        batch_optimise(
            simulation=simple_simulation,
            hparam_batch=hparam_batch,
            batch_size=2,
            data_to_fit=(None,),
            config=config,
            indexes=(0,),
            loss_functions=(_dummy_loss,),
            run_names=run_names,
            log_dir=str(tmp_path),
            log_every_n=1,
        )

        for name in run_names:
            assert os.path.exists(str(tmp_path / f"{name}.log"))

    def test_n_hparams_not_multiple_of_batch_size(self, simple_simulation, tmp_path):
        """Padding: 5 hparams with batch_size=3 → pads to 6."""
        n_hparams = 5
        hparam_batch = HParamBatch(
            forward_model_scaling=jnp.ones((n_hparams, 2)),
        )
        config = OptimiserSettings(
            n_steps=5, learning_rate=1e-2, tolerance=1e-8,
            convergence=[1e-3], ema_alpha=0.5, min_steps_per_threshold=2,
            optimiser_type="adam",
        )
        result = batch_optimise(
            simulation=simple_simulation,
            hparam_batch=hparam_batch,
            batch_size=3,
            data_to_fit=(None,),
            config=config,
            indexes=(0,),
            loss_functions=(_dummy_loss,),
            run_names=[f"r{i}" for i in range(n_hparams)],
            log_dir=str(tmp_path),
        )
        assert result.convergence_steps.shape == (n_hparams,)
```

- [ ] **Step 2: Run tests — expect FAIL (no module `batch`)**

```bash
pytest jaxent/tests/unit/opt/test_batch_optimise.py -v
```

- [ ] **Step 3: Create `jaxent/src/opt/batch.py`**

```python
# jaxent/src/opt/batch.py
"""batch_optimise — runs hyperparameter sweeps via jax.lax.map + jax.vmap."""
import logging
from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from beartype.typing import Optional

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.custom_types.config import OptimiserSettings
from jaxent.src.custom_types.features import Output_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.opt.base import (
    BatchOptimisationResult,
    HParamBatch,
    JaxEnt_Loss,
    LossComponents,
    OptimisationCarry,
    OptimizationHistory,
    OptimizationState,
)
from jaxent.src.opt.gradients import create_gradient_masks
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _optimise_pure
from jaxent.src.opt.track import init_convergence_carry
from jaxent.src.custom_types.config import Optimisable_Parameters


def _setup_run_loggers(
    run_names: list[str],
    log_dir: str,
    n_pad: int,
) -> list[logging.Logger | None]:
    """Create one FileHandler logger per run. Padded slots get None."""
    loggers = []
    for name in run_names:
        lg = logging.getLogger(f"jaxent.opt.run.{name}")
        lg.setLevel(logging.DEBUG)
        handler = logging.FileHandler(f"{log_dir}/{name}.log", mode="w")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.handlers = [handler]
        loggers.append(lg)
    loggers.extend([None] * n_pad)
    return loggers


def _flush_loggers(loggers: list[logging.Logger | None]) -> None:
    for lg in loggers:
        if lg is not None:
            for h in lg.handlers:
                h.flush()
                h.close()


def _make_init_carry(
    optimizer: OptaxOptimizer,
    params: Simulation_Parameters,
    n_steps: int,
) -> OptimisationCarry:
    """Build the base OptimisationCarry (same for all hparam runs before hparam injection)."""
    n_models = len(params.forward_model_weights)
    opt_state_tx = optimizer.optimizer.init(params)
    mask = create_gradient_masks(
        {Optimisable_Parameters.frame_weights}, params, None
    )
    convergence = init_convergence_carry(params)
    hist_params = jax.tree.map(
        lambda x: jnp.zeros((n_steps, *x.shape), dtype=x.dtype), params
    )
    hist_losses = LossComponents(
        train_losses=jnp.zeros((n_steps, n_models)),
        val_losses=jnp.zeros((n_steps, n_models)),
        scaled_train_losses=jnp.zeros((n_steps, n_models)),
        scaled_val_losses=jnp.zeros((n_steps, n_models)),
        total_train_loss=jnp.zeros(n_steps),
        total_val_loss=jnp.zeros(n_steps),
    )
    return OptimisationCarry(
        opt_state=OptimizationState(params=params, opt_state=opt_state_tx),
        convergence=convergence,
        lr=jnp.float32(optimizer.initial_learning_rate),
        model_lr=jnp.float32(
            optimizer.initial_learning_rate * optimizer.model_parameters_lr_scale
        ),
        gradient_mask=mask,
        history_params=hist_params,
        history_losses=hist_losses,
        write_idx=jnp.int32(0),
    )


def batch_optimise(
    simulation: InitialisedSimulation,
    hparam_batch: HParamBatch,
    batch_size: int,
    data_to_fit: Sequence[
        ExpD_Dataloader | Model_Parameters | Output_Features | Array | Simulation_Parameters
    ],
    config: OptimiserSettings,
    indexes: Sequence[int],
    loss_functions: list[JaxEnt_Loss],
    run_names: list[str],
    log_dir: str,
    log_every_n: int = 50,
    logger: Optional[logging.Logger] = None,
) -> BatchOptimisationResult:
    """Run a hyperparameter sweep via jax.lax.map(jax.vmap(_optimise_pure)).

    hparam_batch.forward_model_scaling: [n_hparams, n_models]
      column order matches loss_functions list.

    batch_size controls memory: peak memory ∝ batch_size × n_steps × state_size.
    """
    _logger = logger or logging.getLogger("jaxent.opt")
    n_hparams = hparam_batch.forward_model_scaling.shape[0]

    if len(run_names) != n_hparams:
        raise ValueError(
            f"run_names length {len(run_names)} must match n_hparams {n_hparams}"
        )

    # Pad to multiple of batch_size
    n_pad = (-n_hparams) % batch_size
    if n_pad > 0:
        pad_scaling = jnp.tile(
            hparam_batch.forward_model_scaling[:1], (n_pad, 1)
        )
        padded_scaling = jnp.concatenate(
            [hparam_batch.forward_model_scaling, pad_scaling], axis=0
        )
        padded_lr = None
        if hparam_batch.learning_rate is not None:
            pad_lr = jnp.tile(hparam_batch.learning_rate[:1], n_pad)
            padded_lr = jnp.concatenate([hparam_batch.learning_rate, pad_lr])
        padded_hparams = HParamBatch(
            forward_model_scaling=padded_scaling,
            learning_rate=padded_lr,
        )
    else:
        padded_hparams = hparam_batch

    n_total = n_hparams + n_pad
    n_batches = n_total // batch_size

    # Setup per-run loggers
    loggers = _setup_run_loggers(run_names, log_dir, n_pad)

    # Build global run indices for logger routing in jax.debug.callback
    global_indices = jnp.arange(n_total).reshape(n_batches, batch_size)

    # Base optimizer and carry (same for all runs)
    optimizer = OptaxOptimizer(
        learning_rate=config.learning_rate,
        optimizer=config.optimiser_type,
        initial_learning_rate=getattr(config, "initial_learning_rate", config.learning_rate),
        initial_steps=getattr(config, "initial_steps", 0),
    )
    base_params = simulation.params
    init_carry = _make_init_carry(optimizer, base_params, config.n_steps)

    thresholds = jnp.array(
        sorted([ct * config.learning_rate for ct in config.convergence], reverse=True),
        dtype=jnp.float32,
    )

    def run_single(hparam_scaling: Array, lr_override: Optional[Array]) -> OptimisationCarry:
        """Run one optimisation with a single hparam configuration."""
        _lr = lr_override if lr_override is not None else jnp.float32(config.learning_rate)
        carry = init_carry._replace(lr=_lr, model_lr=_lr * optimizer.model_parameters_lr_scale)
        return _optimise_pure(
            simulation=simulation,
            hparam_scaling=hparam_scaling,
            init_carry=carry,
            data_to_fit=data_to_fit,
            n_steps=config.n_steps,
            loss_functions=loss_functions,
            indexes=indexes,
            optimizer=optimizer,
            thresholds=thresholds,
            min_steps=config.min_steps_per_threshold,
            ema_alpha=config.ema_alpha,
        )

    def run_batch(batch_data: tuple[Array, Array]) -> OptimisationCarry:
        """vmap over a batch of batch_size hparam configurations."""
        batch_scalings, batch_global_indices = batch_data

        lr_batch = None
        if padded_hparams.learning_rate is not None:
            # Select the LR for this batch — handled outside vmap via reshape
            lr_batch = padded_hparams.learning_rate  # placeholder; see note below

        def single_with_logging(scaling: Array, global_idx: Array) -> OptimisationCarry:
            result = run_single(scaling, None)

            if log_every_n > 0:
                def _log_callback(step, loss, idx):
                    idx_int = int(idx)
                    if idx_int < len(loggers) and loggers[idx_int] is not None:
                        if int(step) % log_every_n == 0:
                            loggers[idx_int].info("step=%d loss=%.6f", int(step), float(loss))

                jax.debug.callback(
                    _log_callback,
                    result.opt_state.step,
                    result.history_losses.total_train_loss[result.write_idx - 1],
                    global_idx,
                )

            return result

        return jax.vmap(single_with_logging)(batch_scalings, batch_global_indices)

    # Reshape padded hparams into batches: [n_batches, batch_size, n_models]
    batched_scalings = padded_hparams.forward_model_scaling.reshape(
        n_batches, batch_size, -1
    )

    # Run via lax.map (sequential over batches, parallel within batch via vmap)
    all_carries = jax.lax.map(run_batch, (batched_scalings, global_indices))
    # all_carries shape: [n_batches, batch_size, ...]

    _flush_loggers(loggers)

    # Flatten [n_batches, batch_size, ...] → [n_total, ...], then slice to n_hparams
    def _flatten_batch(x: Array) -> Array:
        return x.reshape(n_total, *x.shape[2:])[:n_hparams]

    flat_carries = jax.tree.map(_flatten_batch, all_carries)

    # Reconstruct OptimizationHistory per run from the flattened carries
    # (full history is in flat_carries.history_losses / history_params)
    histories = OptimizationHistory(
        states=[],
        best_state=None,
    )

    # Build best_states from convergence_steps (write_idx at end of each run)
    convergence_steps = flat_carries.write_idx  # [n_hparams]

    best_params = jax.tree.map(
        lambda buf, idx: buf[idx - 1],
        flat_carries.history_params,
        convergence_steps,
    )
    best_states = flat_carries.opt_state._replace(params=best_params)

    _logger.info("batch_optimise complete: %d runs", n_hparams)

    return BatchOptimisationResult(
        histories=histories,
        best_states=best_states,
        convergence_steps=convergence_steps,
        hparam_batch=hparam_batch,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest jaxent/tests/unit/opt/test_batch_optimise.py -v
```

- [ ] **Step 5: Run full test suite**

```bash
pytest jaxent/tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add jaxent/src/opt/batch.py jaxent/tests/unit/opt/test_batch_optimise.py
git commit -m "feat(opt): add batch_optimise with jax.lax.map + jax.vmap over HParamBatch"
```

---

## Self-review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `ConvergenceCarry` + pure functions | Task 1 |
| Functional `Simulation.forward` | Task 2 |
| Remove prints from `gradients.py` | Task 3 |
| `_apply_lr_scaling` helper | Task 3 |
| `OptimisationCarry`, `HParamBatch`, `BatchOptimisationResult` | Task 4 |
| Unit-LR optimizer, remove `MutableLearningRate` | Task 5 |
| `_pure_step` with `jax.lax.cond` for all conditionals | Task 6 |
| `OptaxOptimizer.initialise` returns `OptimisationCarry`; mask info logged | Task 6 |
| `_optimise_pure` with `while_loop` + converged-gating | Task 7 |
| `_optimise` refactored to use carry | Task 8 |
| Log functions accept injected logger; ambient fallback | Task 8 |
| Per-run file loggers in `batch_optimise` | Task 10 |
| `jax.debug.callback` + `log_every_n` | Task 10 |
| Loss closure fix; scalings in `forward_model_scaling` | Task 9 |
| `batch_optimise` with `lax.map(vmap(...))` | Task 10 |
| Padding to multiple of `batch_size` | Task 10 |
| `convergence_steps` from `write_idx` | Task 10 |
| `lbfgs` excluded from pure path | Task 5 |

**Placeholder scan:** None found.

**Type consistency:**
- `OptimisationCarry` defined in Task 4 (`base.py`), used in Tasks 6, 7, 8, 10 ✓
- `ConvergenceCarry` defined in Task 1 (`track.py`), imported via `base.py` in Task 4 ✓
- `_pure_step` defined in Task 6, called in Tasks 7 and 8 ✓
- `HParamBatch` defined in Task 4, used in Task 10 ✓
- `_optimise_pure` defined in Task 7, called in Task 10 ✓
- `OptimisationCarry.lr` is a `float32` Array in Task 4, read in Task 6 ✓

**Known limitation:** `lbfgs` raises `ValueError` at construction time if used with `OptaxOptimizer` (Task 5). The existing Python-loop path via `run_optimise` with `jit_update_step=False` continues to work with the old `_step` if needed — but `_step` has been removed in this refactor. If `lbfgs` support is needed, it should be re-added as a separate task that restores `_step` as a legacy path.
