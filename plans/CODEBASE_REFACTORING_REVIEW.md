# JAX-ENT codebase refactoring review

_Review date: 6 September 2026_

## Overall verdict

JAX-ENT should not be rewritten wholesale. The numerical and scientific kernels are generally the strongest part of the repository. The main risk is the surrounding orchestration: models, features, datasets, losses, and parameters are connected through parallel lists, string keys, mutable metadata, and partially migrated interfaces.

Shared code exists and is improving, but it is not yet the default path. Examples frequently bypass it through `jaxent.src` imports, private optimizer functions, and locally duplicated workflow logic.

A useful target architecture would be:

```text
ExperimentSpec
    -> ModelBinding[]
    -> SimulationPlan.evaluate()
    -> Prediction[]
    -> LossTerm[]
    -> Optimizer
    -> versioned RunArtifact
```

One `ModelBinding` should own a model, parameters, input features, observation mapping, dataset, and output identity. This would remove most alignment and routing hazards.

## Main codebase

### 1. Rewrite the execution and orchestration boundary

This is the highest-value structural change.

`Simulation` in `jaxent/src/models/core.py` accepts separate sequences for features, models, forward passes, and model parameters. It validates models against parameters, but not all sequence lengths, and ultimately combines them with `zip()` in `forward_pure`. Missing entries can therefore be silently discarded. Loss calculation repeats this pattern across loss functions, targets, and indexes in `jaxent/src/opt/optimiser.py::compute_loss`.

`Simulation` also mixes several responsibilities:

- Mutable session state and pure evaluation.
- JIT compilation and fallback.
- Output routing.
- PyTree serialization.
- Frame-averaging policy.
- Model, feature, and parameter alignment.

Replace its outer API with:

- An immutable `ModelBinding`.
- A pure `SimulationPlan.evaluate(parameters)` operation.
- A separate mutable runner or session if caching compiled functions is needed.
- An explicit `PredictionId`, rather than requiring every output modality key to be unique.

Keep the underlying forward calculations; rewrite how they are assembled.

### 2. Fix proven correctness failures before refactoring

Several cases are directly broken or dangerous:

- `linear_BV_model_Config.forward_parameters` passes `num_timepoints` to a parameter class that has no such field. Constructing these parameters raises `TypeError`. Its `field()` defaults and `__post_init__` are also ineffective because the config is not a dataclass.
- `netHDX_model.initialise` calls `Partial_Topology.find_common_residues`, which does not exist.
- `HDX_protection_factor` uses `HDX_resPf`, while configs and model features use `HDX_resPF`. Since keys are only `NewType(str)` aliases, this receives no runtime protection.
- Test covariance slicing in `jaxent/src/data/loader.py` uses `arange(len(test_data))`, rather than the test datapoints' fragment indexes. A reordered or non-prefix test set gets the wrong covariance submatrix.
- `Partial_Topology` is mutable but hashable, and its hash depends on fields changed by `_set_peptide`. Mutating one after inserting it into a set or dictionary breaks hash invariants.
- Covariance trace normalization can divide by zero.

These defects should receive regression tests before larger API work begins.

### 3. Consolidate the interface and type system

There are currently two overlapping contract systems:

- `ForwardPass`, `Featuriser`, and the `ForwardModel` ABC in `jaxent/src/custom_types/base.py`.
- `SimulationLike`, `InputFeaturesLike`, `ModelParametersLike`, and dataset protocols in `jaxent/src/custom_types/protocols.py`.

They do not completely describe actual behavior. For example, `Simulation.forward_pure()` calls `average_frames`, but `ForwardPass` does not declare it. `ForwardModel.__post_init__()` is not automatically invoked because it is not a dataclass. `ModelParametersLike` is empty.

SAXS and XL-MS demonstrate another mismatch: the base model requires `featurise`, while these models intentionally raise `NotImplementedError` because their features are externally generated.

Define a small set of capability-focused contracts:

- `Predictor`
- Optional `FeatureProvider`
- `FrameAverager`
- `ObservationMapping`
- `ParameterTree`

Use structural protocols at integration boundaries and frozen registered dataclasses for concrete JAX PyTrees. Avoid parallel ABC and protocol hierarchies describing the same object.

### 4. Refactor data splitting around indexes, not mutable datapoints

`DataSplitter` in `jaxent/src/data/splitting/split.py` is 1,559 lines and combines:

- Whole-system and fragment splitting.
- Multiple selection strategies.
- Topology merging and overlap removal.
- Centrality analysis.
- Validation.
- Retry state.
- Global random-number seeding.
- Data mutation.

Most strategies repeat the same select, merge, remove overlaps, filter, validate, and retry pipeline. It also calls `random.seed`, changing process-global randomness.

Refactor it so each strategy only returns index partitions. A single coordinator should apply common validation, retry, topology, and dataset construction. Use a local `random.Random` or NumPy generator.

The generalized `DataMapping` protocol in `jaxent/src/data/splitting/mapping.py`, including `SparseFragmentMapping`, `QSubsetMapping`, and `PairIndexMapping`, is a good abstraction to retain. However, the HDX sparse implementation currently builds a dense matrix before converting it to BCOO and later calls `todense()` for application. That removes the scalability benefit of sparse storage.

### 5. Complete the loss-system migration

The legacy `jaxent/src/opt/losses.py` remains a 1,983-line primary dependency. The newer `opt/loss/` modules are a promising replacement, but the legacy adapter imports the monolith back into the new registry.

Across tests and examples, 24 files import the monolith versus 11 importing the modular loss package. The migration has therefore added a second system without retiring the first.

Introduce an explicit `LossTerm` containing:

- Callable or registered loss name.
- Target dataset or parameter object.
- Prediction identifier or selector.
- Weight and normalization.
- Train and validation split policy.

Then make `compute_loss` consume `tuple[LossTerm, ...]`. This eliminates its three parallel tuples and allows `losses.py` to become a temporary compatibility facade.

### 6. Rewrite persistence and remove import-time runtime configuration

`jaxent/src/utils/hdf.py` changes JAX to CPU and modifies `sys.path` merely by being imported. Persistence code must not select the process backend.

More importantly, model serialization writes dynamic parameter slots but does not write the static scientific values; loading reconstructs them from current class defaults. Values such as temperature or timepoints can therefore change silently across a save/load round trip.

This portion warrants a rewrite around:

- A versioned artifact schema.
- Explicit registered codecs rather than arbitrary dynamic imports.
- All dynamic and static scientific parameters.
- Package and configuration version plus provenance.
- Schema migrations and round-trip tests.

The versioned and atomic manifest work in `jaxent/examples/common/manifest.py` is a good foundation to promote into the library.

### 7. Finish or remove `Experiment_Builder`

`Experiment_Builder.create_model` currently uses `assert` for user-input validation, may access `self.params` before initialization, and filters invalid models and data without applying the same mask to features and parameters. Its default parameter implementation immediately raises `NotImplementedError` before dead code.

It should either become the authoritative factory for `ExperimentSpec` and bindings or be removed. In its current form it gives the appearance of a safe shared construction path without actually providing one.

### 8. Decompose large modules without rewriting stable algorithms

Good decomposition candidates are:

- `jaxent/src/interfaces/topology/mda_adapter.py`, 1,558 lines: separate chain-ID policy, selection parsing, topology conversion, and geometry.
- `jaxent/src/opt/losses.py`, 1,983 lines.
- `jaxent/src/data/splitting/split.py`, 1,559 lines.
- `jaxent/examples/common/loading.py`, 1,052 lines: separate artifact catalog, experimental-data loading, feature loading, topology loading, and workflow preparation.

Retain BV forward mathematics, mapping types, topology factory and pairwise operations, and the newer pure-JAX chunk runner in `jaxent/src/opt/chunk.py`.

## Examples and research workflows

The examples need a separate policy from the library. Excluding deprecated, data, and output directories, they contain roughly 71,000 Python lines, compared with roughly 39,000 across `src`, CLI, and `examples/common`. They are effectively a second application codebase.

### What is being shared successfully

The `examples/common` migration is useful:

- Common config is referenced by about 40 example files.
- Common paths by 19.
- Common loading by 15.
- Manifests by 8.

The declarative configuration in `jaxent/examples/common/config.py` and manifest validation are worth keeping.

### Where examples still bypass shared interfaces

- Only seven files use common optimization and three use common losses.
- `jaxent/examples/common/optimization.py` imports implementation modules and private `_optimise` functions, then reconstructs loaders, simulation, optimizer, provenance, and result writing itself.
- It defines another BV per-frame forward pass, duplicating behavior now available through core prediction paths.
- `jaxent/examples/common/losses.py` adds another registry on top of the legacy monolithic loss module.
- Optimization configuration duplicates core optimizer settings under different names and defaults.

This is partly an examples problem, but primarily signals that the library lacks one convenient public workflow API.

### Separate tutorials, reproducible campaigns, and exploratory research

Use four categories:

- `examples/`: small, tested, runnable tutorials using only public APIs.
- `workflows/`: Experiments 1-5 and paper-reproduction pipelines with YAML configs and manifests.
- `research/`: exploratory and checkpoint investigations.
- External artifacts or small `fixtures/`: generated results and test inputs.

ATLAS is a research pipeline rather than an example: the current tree contains 45 analysis scripts and over 10,000 lines, with checkpoint-numbered stages and repeated per-system execution and checkpoint-validation patterns. Extract an ATLAS-local pipeline library for system iteration, fold handling, atomic checkpointing, aggregation, and reporting. Each scientific checkpoint should then be responsible only for its model-specific calculation.

There is also literal duplication: the Experiment 2 and Experiment 3 `plot_selected_models_ISO_TRI_BI.py` files are byte-identical. More broadly, 88 tracked example files are generated CSV, PDF, or NPZ artifacts or `.original` archive files. Historical versions belong in Git history; generated results belong in artifact storage unless they are deliberately small fixtures.

Finally, wheel packaging includes the whole `jaxent` package but does not exclude `jaxent/examples`. The tracked examples currently add about 13 MB to the source tree and may be shipped unintentionally.

## Public API and quality gates

Examples consistently import `jaxent.src...` because the top-level package exports almost nothing beyond runtime configuration. Establish supported namespaces such as `jaxent.models`, `jaxent.data`, `jaxent.optim`, and `jaxent.artifacts`, then prohibit examples from importing `jaxent.src` or names beginning with `_`.

The repository's current validation baseline is not clean:

- Pytest discovers 1,636 tests but stops during collection on missing `mdtraj` and a broken `profiling.eager_native_memory` import.
- Running unit tests while excluding the latter produced 80 passes before failing because an example integration test depends on a missing manifest artifact.
- Ruff reports 88 findings. Many are `F722` conflicts with jaxtyping shape annotations, but there are also real undefined and unused imports.
- Ruff explicitly excludes all examples, so the larger codebase receives no lint enforcement.
- Root-level `tests/`, including current ATLAS tests, are outside the configured `testpaths` in `pyproject.toml`.
- Versions disagree: README `0.5.2`, package metadata `0.5.0`, and runtime `0.4.1`.

## Recommended order

1. Fix the proven correctness defects, collection failures, version mismatch, and persistence side effects.
2. Introduce `ModelBinding`, `PredictionId`, and `LossTerm` alongside the existing API.
3. Move `Simulation` and optimizer orchestration onto those types; add strict length and compatibility validation before JIT.
4. Complete the modular loss migration and make `Experiment_Builder` authoritative or remove it.
5. Refactor splitting into strategies plus a shared coordinator.
6. Publish a stable public API and migrate Examples 1-3 through it.
7. Reorganize ATLAS and other campaign code into workflows and research, then remove archived and generated material from the wheel.

## Rewrite versus refactor summary

### Rewrite the outer seams

- Simulation and experiment assembly.
- Loss wiring.
- Artifact serialization.
- The splitting coordinator.

### Refactor and consolidate

- Model configuration and parameter types.
- Interface and protocol definitions.
- Topology adapter responsibilities.
- Loss implementations and registries.
- Example loading and optimization helpers.

### Keep and stabilize

- Scientific forward calculations.
- BV contact and uptake mathematics.
- Generalized data mappings.
- Topology factory and pairwise operations.
- Pure-JAX optimizer chunk execution.
- Declarative example configs and versioned manifests.
