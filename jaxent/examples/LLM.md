# JAX-ENT Examples Guide (Bridge Document)

This file bridges:

- repository-level guidance in `LLM.md` (root), and
- deep shared-example implementation details in `jaxent/examples/common/README.md`.

If root `LLM.md` tells you *what JAX-ENT is*, this file tells you *how the examples layer is organized and how to navigate it efficiently*.

---

## Documentation hierarchy (read in this order)

1. **`/LLM.md` (root)**
   - Core architecture (`jaxent/src`), model/data/optimization flow, and repository-wide conventions.
2. **`jaxent/examples/LLM.md` (this file)**
   - Examples-layer orientation, routing, and workflow map.
3. **`jaxent/examples/common/README.md`**
   - Shared utilities API, module-by-module reference, config patterns, and plotting/analysis internals.

Use this hierarchy to manage context: broad -> examples-specific -> deep shared details.

---

## What lives in `jaxent/examples/`

### Active workflow suites

- `1_IsoValidation_OMass/`
  - Synthetic/controlled validation workflow with known target populations.
- `2_CrossValidation/`
  - Real MoPrP HDX cross-validation workflow.
- `3_CrossValidationBV/`
  - Extension of Example 2 with BV regularization sweeps.
- `predict_traj/`
  - Directory-level trajectory prediction and clustered-prediction helpers.

### Shared infrastructure

- `common/`
  - Reusable configs, loading, optimization orchestration, analysis, and plotting used across active examples.

### Supporting/archival areas

- `combined_fixed_effects/`
  - Cross-experiment aggregation and fixed-effects analysis helpers.
- `deprecated/`
  - Historical workflows kept for reference; do not treat as default execution surface.

---

## How examples connect to core code

Typical call chain:

`example script` -> `jaxent.examples.common.*` -> `jaxent.src.*`

- Example scripts define experiment-specific paths/choices.
- `common` centralizes shared patterns to avoid duplication.
- `jaxent/src` provides the core modeling, featurization, and optimization primitives.

This layering is intentional: examples should prefer `common` imports over repeated direct wiring to `jaxent.src`.

---

## Where to look first (task routing)

### A) Understand an example end-to-end

Start in the example directory:

1. `README.md` (full narrative, scientific context, outputs)
2. `INSTRUCTIONS.md` (condensed run order)
3. `commands.sh` + `config.yaml` (actual execution surface)
4. only then inspect individual Python scripts

### B) Update shared analysis/plotting/loading logic

Go directly to `jaxent/examples/common/README.md`, then the target module in `common/`.

### C) Reconcile example behavior with core implementation

Use this order:

1. this file (examples routing)
2. `common` module docs/code
3. root `LLM.md` + referenced `jaxent/src` modules

### D) Debug path and execution issues

Check:

- whether commands are being run from repository root,
- whether example configs use expected relative paths,
- whether outputs are being written under expected `_optimise*` / analysis folders.

---

## Context-management strategy for large examples

Examples are extensive; avoid loading everything at once.

1. **Load structure first** (README + INSTRUCTIONS + commands/config).
2. **Identify the single failing/target stage** (data prep, featurize, split, optimize, analysis).
3. **Open only scripts for that stage**.
4. **Use `common` docs/code for shared behavior** rather than re-deriving from each example script.

This gives high signal while controlling context size.

---

## Conventions and gotchas

- Many example scripts assume execution from repository root.
- Active examples may share naming from earlier workflows; verify directories and file names before assumptions.
- `common` is the primary reuse layer for analysis/plotting/optimization wrappers.
- `deprecated` contains historical implementations and may not match current defaults.
- Prefer path-accurate references to scripts/configs over inferred behavior.

---

## Practical quick-start map

- Need scientific/experiment context? -> example `README.md`
- Need runnable sequence? -> example `INSTRUCTIONS.md` + `commands.sh`
- Need shared API/module details? -> `jaxent/examples/common/README.md`
- Need core architecture internals? -> root `LLM.md` and linked `jaxent/src` files

---

## Maintenance intent for this file

Keep this document as a **navigation bridge**, not a duplicate of:

- root architecture docs (`/LLM.md`), or
- deep `common` module reference (`jaxent/examples/common/README.md`).

When example structure changes, update this file's routing map and read-order guidance first.
