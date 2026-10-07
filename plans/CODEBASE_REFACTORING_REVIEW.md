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

## Numerical impact assessment

_Assessment date: 17 September 2026. This assessment includes the current working tree. The pending BV rate-distribution, interval-hazard, PCA, and HDF changes are uncommitted at the time of review._

### Impact scale

- **High:** can directly change the objective, gradients, fitted frame weights, fitted model parameters, selected checkpoint, or reported validation result.
- **Medium:** should be numerically neutral, but ordering, defaults, JAX PyTree structure, random-number consumption, or cached artifacts can change results.
- **Low:** organizational change with no expected numerical effect, provided imports and defaults remain identical.

| Review section | Intended impact | Migration risk |
|---|---:|---:|
| 1. Execution and orchestration | None | High |
| 2. Correctness failures | Corrective and often intentional | High |
| 3. Interfaces and types | None | Medium-high |
| 4. Data splitting and mappings | Usually none for a parity refactor | High |
| 5. Loss-system migration | None for like-for-like losses | High |
| 6. Persistence | None for uninterrupted fits; exact preservation for resumed fits | High |
| 7. `Experiment_Builder` | None | High if adopted; low while bypassed |
| 8. Module decomposition | None | Medium |
| Shared example configuration | None | Medium-high |
| Example optimization/loss consolidation | None | High |
| Workflow reorganization | None | Medium |
| Public API aliases | None | Low-medium |

### 1. Execution and orchestration boundary

**Expected numerical effect:** none if the replacement is a strict compatibility layer. **Risk:** high because the present parallel-list design encodes scientific choices implicitly through position and forward-pass metadata.

Likely jagged edges:

- A binding reorder can connect a loss to the wrong prediction, dataset, feature tensor, or model parameter tree while retaining compatible shapes.
- `zip()` currently stops at the shortest sequence without reporting which input was short. In the common path a later loss-weight shape operation will often turn this into an error; in callers whose weights also match the shorter length, a trailing model or loss could instead be omitted. Replacing this with up-front validation is a corrective behavior change, not numerical parity.
- Frame averaging is part of the scientific model, not an execution detail. For per-frame log protection factors `z`, the following are different:
  - log-PF averaging: `1 - exp(-t k exp(-E[z]))`
  - rate averaging: `1 - exp(-t E[k exp(-z)])`
  - frame-uptake averaging: `E[1 - exp(-t k exp(-z))]`
- As a simple example with `z = [0, 4]`, equal weights, `k = 1`, and `t = 1`, these give approximately `0.127`, `0.399`, and `0.325`. A refactor that changes `frame_averaging_mode` therefore changes the fitted ensemble substantially.
- `tensordot` and `legacy_sum` are algebraically equivalent but use different reduction orders. The existing test permits an absolute difference of about `3.8e-6`. That is normally small, but can change gradient-sign oscillation detection, threshold-crossing steps, or near-tied model selection.
- Forward-pass objects are static JIT arguments. Changing equality, hashing, object identity, or closure construction can change recompilation and expose stale static configuration.
- Best-state selection currently uses the first unscaled validation component, not total validation loss. Introducing named `LossTerm` objects must preserve this policy explicitly or deliberately version a new policy.

Required parity checks:

1. Compare every bound prediction by identity and shape before optimization.
2. Compare individual train/validation loss components and total loss.
3. Compare gradients with respect to frame logits and every model-parameter leaf.
4. Compare one optimizer step and a short fixed-step trajectory.
5. Run separate golden tests for every frame-averaging mode.

### 2. Proven correctness failures

**Expected numerical effect:** fixes in this section are allowed to change results because several current behaviors are wrong. Each fix needs a result-migration note rather than being presented as a neutral refactor.

- **Linear BV:** the old configuration could not construct its parameter object. The current pending implementation goes much further: it replaces a direct contact linear combination with cumulative interval hazards and bounded uptake. That changes the model, parameter interpretation, gradients, and fitted values. It should have a new model name/artifact version rather than silently reusing `linear_BV` semantics.
- **NetHDX:** fixing the missing topology method enables a previously broken route. There is no trustworthy old-fit parity target, so validation must use synthetic or scientific golden cases.
- **`HDX_resPf` versus `HDX_resPF`:** correcting the key can reroute a dataset to a different output or make a formerly skipped loss active. This is discontinuous, high impact.
- **Test covariance indexes:** correcting test covariance selection should not change training, but it changes test likelihoods and any model selection or publication result that improperly used test scores. The same index contract should be checked for train and validation paths.
- **Mutable topology hashes:** a fix is usually neutral, but existing set/dictionary corruption can have changed residue inclusion and ordering. Rebuild mappings rather than reusing cached mappings after this change.
- **Zero-trace covariance:** returning an explicit error instead of NaNs changes failure behavior; adding a numerical fallback changes the objective. Prefer rejecting the dataset unless a scientifically justified regularization is configured.
- **Rate/time units:** the pending linear and rate-distribution passes explicitly convert between seconds and minutes, while the standard `BV_uptake_ForwardPass` still multiplies stored rates and timepoints directly. `BV_model_Config` defaults to `s^-1`, while common example timepoints are generally minutes. A missing conversion changes exposure by a factor of 60. Units must become part of every model parameter/artifact and be handled consistently across standard, linear, and distributional BV models.

### 3. Interface and type consolidation

**Expected numerical effect:** none. **Risk:** medium-high around PyTrees and parameter constraints.

Likely jagged edges:

- Changing whether a field is a dynamic PyTree leaf or static metadata changes what receives gradients, optimizer state, batching, and JIT cache keys.
- Activating a previously dormant `__post_init__` can normalize values, replace keys, or validate/reject configurations that old runs accepted.
- Changing list/tuple structure or leaf order invalidates saved Optax state even if the mathematical parameters are unchanged.
- Moving from physical parameters to raw transformed parameters changes optimization geometry. Softplus, log, or simplex parameterizations can start at the same physical value but produce different gradients and Adam moments.
- Parameter constraints must be leaf-specific. The current optimizer applies `optax.keep_params_nonnegative()` to the entire model-parameter partition. The pending softplus parameterization initializes physical `bv_bc=0.35` as raw `-0.8697`; even with a zero gradient, the optimizer constraint moves it to raw zero, changing physical `bv_bc` to `softplus(0)=0.6931` on the first update. Negative interval offsets are also impossible under this partition-wide constraint.
- Pending soft-mixture support points have the same conflict: a support point near zero is represented by a raw gap near `-13.8`, which the current model constraint clamps to zero.

Before adopting raw/transformed parameters:

- Remove partition-wide non-negativity and put constraints in the parameter transform itself.
- Snapshot `jax.tree_util.tree_structure`, leaf names, shapes, and dtypes.
- Test physical-value equality, gradient chain rules, mask assignment, one optimizer step, and HDF round trips.

### 4. Data splitting and mappings

**Expected numerical effect:** none only if the exact split membership and mapping matrix are preserved. In practice this is a high-impact area because the split defines the fitted objective.

Likely jagged edges:

- Switching from the module-global RNG to a local generator is desirable, but the same integer seed may produce different partitions. Preserve old partitions through explicit index manifests rather than promising seed parity.
- Set iteration, topology hashing, sorting tie-breakers, retry order, centrality sampling, and overlap removal can all change which observations enter train and validation sets.
- `fragment_index` is currently mutated. Separating data from indexes can reveal implicit reliance on that mutation in covariance and mapping code.
- The legacy sparse map divides each represented overlap by the full experimental fragment length. If residues are absent from the feature topology, rows sum to less than one. Renormalizing rows changes predicted peptide uptake and fitted weights; it must be an explicit mapping policy, not a sparse-performance refactor.
- Constructing BCOO directly can sum duplicate coordinates, whereas the current dense `.at[...].set(...)` path overwrites them. Direct sparse construction must coalesce with legacy-equivalent semantics.
- Sparse and dense matrix multiplication can differ slightly through accumulation order and dtype. Test both predictions and gradients.
- `jnp.squeeze` can remove the observation axis for a single peptide and trigger different broadcasting downstream.
- Covariance matrices must be subset using observation identities, not positional prefixes. Persist the ordered observation IDs alongside every covariance.

The split artifact should contain ordered train/validation/test IDs, the mapping policy, mapping checksum, covariance IDs, seed, strategy version, and topology checksum.

### 5. Loss-system migration

**Expected numerical effect:** none for a like-for-like migration. **Current risk:** very high; the modular loss system is not numerically equivalent yet.

Concrete current differences:

- `hdx_pf_l2_builder` supplies a loss function that already computes a mean, after which `apply_post_processing` divides by the observation count again. For predictions `[1, 3]` and targets `[0, 0]`, the legacy PF L2 is `5.0`; the modular builder returns `2.5`.
- Real HDX uptake targets from `ExpD_Dataloader.create_datasets` have shape `(fragments, timepoints, 1)`, while predictions have shape `(timepoints, residues)`. The generic non-flattened loss iterates `y_true.shape[0]` as though it were timepoints. The enabled modular tests do not currently exercise the production shape; most uptake builders in `test_loss_builder.py` are commented out.
- The legacy-alias loop registers `LossRegistry.get(...)`, which is a built loss function, in a registry that expects a zero-argument factory. Looking up `hdx_pf_l2_loss` through that alias currently raises `TypeError`.
- Mean versus sum, the factor `0.5` in quadratic losses, timepoint averaging, fragment averaging, covariance trace normalization, and epsilon placement all rescale gradients.
- Mean-centering before mapping, after mapping, per timepoint, or over the complete flattened target are different objectives.
- `forward_model_weights` are normalized according to `normalise_loss_functions`. With raw slots `[1, lambda]`, both masked, the effective weights become `[1/(1+lambda), lambda/(1+lambda)]`. Adding another normalized loss rescales every existing term. That changes gradient scale, clipping, Adam epsilon effects, weight decay, and convergence timing even when relative ratios appear unchanged.
- An all-zero normalization mask currently divides by zero; multiplying the resulting NaN by a zero mask does not recover finite weights.
- Common example optimization appends one parameter target when any regularizers are present, rather than one target per regularizer. With two regularizers, the loss/index lists have four entries but the target list has three. `zip()` evaluates only three; the standard helper should then fail when the three returned losses are multiplied by four weights. Thus the usual outcome is a delayed shape error, not a silently successful fit, although lower-level callers can still omit a trailing loss if their weight arrays are also sized to the shorter list.
- Best-state selection depends on loss slot zero. Named loss terms must preserve or explicitly replace that rule.

Do not switch production names to modular implementations until differential tests compare legacy and new values and gradients on production-shaped PF, uptake, covariance, MaxEnt, and model-parameter fixtures. Require exact agreement where operations are identical and documented tolerances where reduction order differs.

### 6. Persistence and runtime configuration

**Expected numerical effect:** no effect during an uninterrupted fit, but high impact for resumed fitting, re-analysis, and cross-machine reproduction.

Likely jagged edges:

- Importing `utils/hdf.py` still forces the JAX platform to CPU. CPU/GPU reduction and transcendental implementations can differ slightly; more importantly, an import unexpectedly changes the entire process execution backend.
- The pending static-metadata JSON addition preserves new time grids, units, and backend names, but it has no schema version or migration policy. Legacy files still reconstruct missing fields from current defaults.
- Renaming or moving parameter classes breaks dynamic `module.class` loading.
- Changing from physical to raw parameters can load an old physical value into a raw field unless the artifact records the parameterization version.
- Saved Optax state is tied to the exact PyTree structure. Restoring parameters while discarding or misaligning moments produces a different trajectory.
- Tuple/list and dtype changes should be normalized deliberately. JSON round trips tuples as lists.
- Field equality is insufficient: a persistence test must compare predictions and loss before/after round trip, then compare the next optimizer step after resume.

Artifacts should include a schema version, model equation/version, parameterization version, units, averaging mode, mapping/split identity, dtype/runtime policy, code commit, and explicit migrations.

### 7. `Experiment_Builder`

**Expected numerical effect:** none. **Risk:** high if it becomes authoritative without first fixing alignment.

- Invalid-model filtering must remove the corresponding model, features, parameters, datasets, loss terms, and output IDs atomically. Filtering only models/data changes positional binding.
- Feature-dependent parameter initialization must occur once and must update the parameter tree actually passed to `Simulation`. The pending rate-distribution model mutates `self.params` during featurization, which can diverge from separately supplied simulation parameters.
- The dead default-parameter code builds frame weights using `len(self.features)`, which is the number of feature/model objects rather than the number of trajectory frames. It must not be revived unchanged.
- Replacing `assert` with exceptions should expose invalid runs rather than alter valid numerical behavior.

Build a tuple of complete bindings first, validate it, and only then derive the parallel JAX structures during the compatibility period.

### 8. Decomposing large modules

**Expected numerical effect:** none for a pure move. **Risk:** medium because this repository uses import-time registration and runtime state.

Likely jagged edges:

- Loss registries depend on import side effects; moving imports can change which implementation owns a name.
- The joint-Gaussian Cholesky factor is module-global state. Moving or importing the module twice under different paths can create independent globals.
- HDF stores module/class paths, so moving a class is an artifact schema change.
- Runtime configuration and environment variables execute during imports. Reordering imports can change backend or precision setup.
- Splitting a JITted callable into closures can change static arguments and compilation behavior.

Do not combine module moves with mathematical changes. In particular, the pending interval-hazard and rate-distribution BV work is new model development, not decomposition. The pending IncrementalPCA correction also changes all projected coordinates after the first batch and can change clusters, splits, and downstream fits. It is a valid correctness fix, but all cached PCA coordinates and cluster assignments must be invalidated.

## Numerical impact in examples and workflows

### Shared configuration, paths, loading, and manifests

Paths and plotting configuration are low impact. Optimization configuration and scientific loading are medium-to-high impact.

- Core `OptimiserSettings` and example `OptimizationConfig` have different defaults for steps, tolerance, learning rate, and optimizer (`adam` versus `adamw`). Consolidation must use fully resolved existing run configuration, not whichever default survives.
- AdamW applies decay to frame logits and model parameters; Adam does not. Decay of logits biases weights toward the uniform simplex, so this is scientifically relevant.
- Common optimization ignores `OptimizationConfig.clip_value`, passing `clip_value=None`, and hard-codes optimizer tolerance to `1e-10`. Correcting either changes trajectories.
- Loading helpers can alter file ordering, transposition, units, timepoint order, or selection of duplicate result files. Return typed records with explicit axes and identities rather than nested dictionaries inferred from filenames.
- Manifests should record resolved values after all CLI/config overrides, not only user-supplied values.

### Consolidating example optimization and losses

**Risk:** high. The duplicate code has already drifted, so choosing either version as authoritative can change results.

- The example per-frame uptake implementation and the core averaging modes must agree on whether timepoints are seconds or minutes and whether uptake is averaged before or after the exponential.
- `execution_mode`, chunk size, EMA state, LR oscillation handling, clipping, optimizer type, model-parameter LR scale, and initial weights must all be held fixed during parity tests.
- Compiled and Python paths should be compared step-by-step. Small reduction differences are acceptable only if loss components, endpoint observables, and selected checkpoints remain within declared tolerances.
- Provenance should include the forward-model equation/version and loss implementation version, not merely the function name.

### Reorganizing tutorials, workflows, research, and artifacts

Moving files is low impact; cache and checkpoint reuse is not.

- Relative paths and current-working-directory assumptions can select different data after a move.
- Stable seeds must retain the same token construction and sampling order if exact replay is required.
- Many ATLAS checkpoint validators only check that files contain the expected system and held-out replicas. They do not validate the source-data hash, configuration hash, or code/model version. A refactored script can therefore accept numerically stale checkpoints.
- Pipeline extraction should introduce content-addressed stage identities or manifests containing upstream hashes. Any change to PCA, contacts, topology, units, frame filtering, model equations, or loss normalization must invalidate downstream artifacts.
- Historical generated files should remain immutable reference artifacts where needed for parity tests; do not silently overwrite them with refactored output.

### Public API and quality gates

Public aliases should be numerically neutral, but imports must not introduce new defaults or runtime side effects. API parity tests should invoke both the old and new import paths on the same objects and compare outputs.

The quality gate needs scientific differential tests in addition to unit tests:

- Existing unit tests mostly verify the current implementation against formulas repeated in the test. They do not prove parity with prior production results.
- The 55 focused BV/gradient/PyTree tests and seven HDF/optimizer-ablation tests currently pass, but they do not catch the partition-wide non-negativity conflict or establish old/new fit parity.
- Add small committed golden fixtures for PF fitting, uptake fitting, covariance fitting, mixed modalities, save/resume, and each averaging mode.
- Record per-step loss components, selected weights, physical model parameters, gradients, and final predictions. Comparing only total loss can hide compensating errors.

## Immediate numerical blockers before architectural migration

1. Resolve the `s^-1` versus minute-timepoint contract in the standard BV uptake path.
2. Remove partition-wide `keep_params_nonnegative` before optimizing raw softplus parameters or interval offsets.
3. Give the interval-hazard and rate-distribution models new, versioned identities; do not present them as neutral `linear_BV` refactors.
4. Repair modular loss axes, normalization, and registry-factory behavior, then add legacy differential tests.
5. Replace parallel loss target/index lists before supporting multiple regularizers.
6. Version HDF artifacts and prove prediction plus next-step equivalence across save/resume.
7. Freeze split/mapping manifests and invalidate all affected caches after topology, PCA, mapping, unit, or model-equation changes.

## Acceptance standard for a numerically neutral refactor

For a fixed seed, device, dtype, dataset, and resolved configuration:

1. Initial predictions match for every output.
2. Every individual loss component matches before weighting.
3. Weighted total loss matches.
4. Gradients match leaf-by-leaf.
5. One optimizer update, including Optax state, matches.
6. A short fixed-step trajectory matches within a declared tolerance.
7. Final physical observables and selected best/convergence states match.
8. Save/load preserves predictions, losses, and the next optimizer update.

If any item fails because the old behavior was incorrect, classify the change as a versioned numerical correction, document the expected result change, and regenerate downstream artifacts deliberately.
