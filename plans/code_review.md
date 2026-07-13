# JAX-ENT Code Review

This code review highlights critical bugs, JAX-specific performance improvements, code quality issues, and structural concerns across the `/jaxent/src/` codebase.

## 1. Critical Bugs & Logic Flaws

### 1.1. `predict.py` Broken Initialization with Sequence Parameters
In `predict.py`, `run_predict` takes a `model_parameters` argument that can be a `Sequence[Model_Parameters]` or a `Simulation_Parameters` object. However, when initializing the `Simulation` object:
```python
simulation = Simulation(
    input_features=input_features_list,
    forward_models=forward_models_list,
    params=model_parameters if isinstance(model_parameters, Simulation_Parameters) else None,
    # ...
)
```
If `model_parameters` is a sequence, `params` is passed as `None`.
In `models/core.py`, `Simulation.initialise()` expects `self.params` to not be `None` and raises a `ValueError` unconditionally:
```python
if self.params is None:
    raise ValueError("No simulation parameters were provided. Exiting.")
```
**Fix:** Provide a valid `Simulation_Parameters` (or a mock/default state) to the `Simulation` initialization, or rework `initialise()` to support sequence initialization.

### 1.2. Context Manager Leak in `predict_forward.py`
In `run_forward`, the `jit_Guard` context manager wraps `simulation`:
```python
with jit_Guard(simulation, cleanup_on_exit=True) as sim:
    # ...
    for params in tqdm(simulation_parameters, desc="Forward prediction"):
        sim, outputs = sim.forward(sim, params, mutate=False)
```
Using `mutate=False` means `sim.forward` returns a **new instance** of `Simulation` (`new_sim`). 
Rebinding `sim = new_sim` within the loop does **not** update the `jit_Guard`'s tracked object (`self.simulation_obj`). The original object is preserved by the context manager, and the loop-generated `sim` objects might not benefit from `jit_Guard` cleanup on exit.
**Fix:** If you are treating the simulation state immutably (`mutate=False`), you need to ensure any caches associated with intermediate instances are managed correctly, or use `mutate=True` within the guarded block. Reassigning the context manager variable (`sim`) is a Python anti-pattern.

### 1.3. Exception Swallowing in `utils/jit_fn.py`
In `jit_Guard.clear_caches_after`:
```python
if original_exception:
    raise cleanup_error from original_exception
```
This syntax causes the original exception to be nested/suppressed as the secondary context of the new `cleanup_error`. When execution blows up, the developer sees `cleanup_error` as the primary cause instead of the actual domain logic error.
**Fix:** Standard practice is to log the cleanup error but raise the `original_exception` as the primary failure, or carefully chain them using `raise original_exception from cleanup_error`.

---

## 2. JAX Usage & Performance

### 2.1. Unnecessary Array Copying
In `jaxent/src/models/HDX/forward.py` (`linear_BV_ForwardPass`):
```python
heavy_contacts = jnp.array(input_features.heavy_contacts)
```
`jnp.array` unconditionally copies data. The other classes correctly use `jnp.asarray`, which avoids the copy if the data is already a compatible JAX array.
**Fix:** Replace `jnp.array` with `jnp.asarray`.

### 2.2. Overriding PyTree Nodes Uncleanly
In `models/core.py`, `Simulation.tree_unflatten`:
```python
instance = cls(input_features, forward_models, params)
instance.forwardpass = forwardpass
instance.length = length
# ...
```
Because the `__init__` method performs setup (like re-deriving `self.forwardpass`), dynamically overriding properties straight after `__init__` causes redundant initializations and could lead to state mismatches. 
**Fix:** Refactor `__init__` to explicitly accept these internal states (e.g., as private kwargs) if they are known, or skip executing setup logic if data is already populated.

---

## 3. Code Quality & Typing Issues

### 3.1. Dynamic Attributes Lacking Hints
In `models/core.py` (`Simulation`), attributes such as `self.length` and `self._input_features` are dynamically assigned in `.initialise()` but are never hinted or initialized (`= None`) at the class level. This breaks type checkers (like mypy/pyright) and IDE autocomplete.

### 3.2. Dead/Unused Code in Core Simulation
`Simulation.__init__` has several lingering commented-out blocks:
```python
# self.model_name_index: list[tuple[m_key, int, m_id]] = model_name_index
# self.outputs: Sequence[Array]
```
These should be removed for a clean codebase.

### 3.3. Generator vs List Comprehensions
In `models/core.py` and `predict.py`, you have expressions like:
```python
tuple([feature.cast_to_jax() for feature in self.input_features])
```
Passing a list comprehension to `tuple()` creates an intermediate list in memory.
**Fix:** Use generator expressions directly: `tuple(feature.cast_to_jax() for feature in ...)`

---

## 4. Error Handling & Standardization

### 4.1. Icecream (`ic`) in Production Modules
`predict.py`, `predict_forward.py`, and `featurise.py` use `ic()` statements paired with `ic.disable()`. If an end user enables `ic` globally, they will be flooded with library debug output.
**Fix:** Remove `icecream` from production library files. Use `logging` instead, which is already correctly configured and utilized in `models/core.py` and `opt/run.py`.

### 4.2. Invalid Exception Throwing
In `featurise.py`:
```python
raise UserWarning("Name is required")
```
`UserWarning` is a warning category designed for the `warnings` module (`warnings.warn()`), not an `Exception` subclass meant for direct raising to halt execution.
**Fix:** Raise `ValueError` instead.

### 4.3. Dummy Exception Objects
In `opt/run.py`:
```python
if forward_models:
    Warning("forward_models arg not yet implemented in run_optimise")
```
This merely instantiates a `Warning` object and immediately discards it; it does not actually emit a warning.
**Fix:** Replace with `import warnings; warnings.warn("...", UserWarning)`.

### 4.4. `print` inside Exception Handlers
Modules like `predict.py` catch errors and print messages:
```python
except Exception as e:
    print(f"Failed to run prediction: {e}")
    raise e
```
**Fix:** This breaks unified log handling. Use `LOGGER.error(f"Failed to run prediction: {e}")` before re-raising.
