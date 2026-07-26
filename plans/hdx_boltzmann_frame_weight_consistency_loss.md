# Boltzmann Frame-Weight Consistency Loss (MaxEnt Replacement) — Handoff

## Status

Planning stage only. Nothing in this document has been implemented yet. This is the
derivation and implementation plan for a new regularization loss that replaces
`maxent_convex_kl` with a physically-motivated alternative.

## Context

The optimizer fits per-frame ensemble weights (`Simulation_Parameters.frame_weights`,
stored pre-softmax as logits, normalized via `jax.nn.softmax` in
`Simulation_Parameters.normalize_weights`) to match experimental HDX data. Today,
regularization comes from `maxent_convex_kl` — a KL-divergence-to-uniform-prior term
with no physical content beyond "don't stray far from a flat prior."

The goal is to replace this with a regularizer derived from the BV forward model's own
physics: frames with similar protection-factor character should get similar weights,
with the relationship between "similar" and "weight" set by an actual Arrhenius/Boltzmann
argument rather than an arbitrary kernel. This reuses the existing loss-registry
architecture (same `JaxEnt_Loss` protocol, same `forward_model_weights` ×
`forward_model_scaling` strength-sweep mechanism MaxEnt already uses) — no optimizer or
gradient-masking changes are needed, because a quadratic penalty's gradient is itself the
"projection of the weight update by frame difference" this feature was originally
described as; autodiff produces it for free from a loss term.

This plan covers **only** the mechanism (loss kernel + wiring). Validation methodology
(e.g. a shuffled-graph negative control, mirroring `build_rate_geometries`'
`shuffled_geometry` pattern already used for the analogous residue-axis regularizer in
`jaxent/src/analysis/hdx_target_variance.py`) is explicitly deferred — scope it later.

## Physical derivation

### Rigorous

`BV_ForwardPass.__call__` (`jaxent/src/models/HDX/forward.py:18-37`) computes, per
residue r, per frame i:

```
log_Pf_i(r) = bc * heavy_contacts_i(r) + bh * acceptor_contacts_i(r)
```

Under standard EX2 HDX kinetics (`PF = k_int/k_obs ≈ 1/K_op`), this is exactly the local
opening free energy in RT units: `log_Pf(r) = ΔG_op(r)/RT`. Large `log_Pf(r)` means it is
costly to locally unfold that residue — i.e. it is well-protected/stable there.

Aggregate per-frame proxy free energy via a **sum** over residues (not mean — this is
what makes the reduction below exact and cheap):

```
G_i = Σ_r log_Pf_i(r) = bc · total_heavy_i + bh · total_acceptor_i
```

where `total_heavy_i = Σ_r heavy_contacts_i(r)`, `total_acceptor_i = Σ_r acceptor_contacts_i(r)`.
These are fixed reductions of `BV_input_features.heavy_contacts`/`.acceptor_contacts`
(`jaxent/src/models/HDX/BV/features.py:23-24`, already stored un-averaged as
`(n_residues, n_frames)`), independent of `bc`/`bh`. `G_i` itself is bc/bh-dependent
(i.e. "dynamic": it moves as the BV coefficients are optimized), but its ingredients need
no duplicate forward pass — computing `G` each call is one cheap `O(n_frames)`
contraction, not a second full model evaluation.

Because `Simulation_Parameters.frame_weights` are logits pre-softmax,
`logit_i - logit_j = ln(w_i) - ln(w_j)` exactly (the softmax normalizing constant
cancels). **Confirmed by tracing `compute_loss`** (`jaxent/src/opt/optimiser.py:608-644`)
→ `Simulation.forward` (`jaxent/src/models/core.py:132-163`): every loss, including this
one, receives `model.params.frame_weights` **after** `normalize_weights` has already
applied softmax — no loss function ever sees the raw pre-softmax logits directly. This is
not a blocker: since the penalty below only ever uses *differences* of a log-weight-like
quantity, `logit_like = jnp.log(model.params.frame_weights + eps)` is an exact stand-in
for the true logit difference (the same cancellation applies). No protocol change needed.

### The modeling step (assumption, not derivation)

A conformer that is more compact/protected (large `G_i`, i.e. more heavy-atom contacts
and H-bonds everywhere) is the kind of conformer expected to sit in a deeper, more stable
region of a folded protein's free-energy landscape — that is the physical reason
compact/native-like states dominate an equilibrium MD ensemble at all. So larger `G_i`
should correspond to a **lower** conformational free energy `E_i` for that frame:
`E_i ≈ -G_i` (up to a scale/offset absorbed into the regularization-strength
hyperparameter). Boltzmann:

```
w_i ∝ exp(-E_i) ≈ exp(+G_i)
ln(w_i) - ln(w_j) ≈ G_i - G_j
logit_i - logit_j ≈ +(G_i - G_j)          <-- target relationship
```

Equivalently: `logit_i - G_i` should be approximately constant across frames.

The `ΔG_op → log_Pf` step is rigorous EX2 thermodynamics; the `G_i → E_i` step is a
**modeling assumption** (compact/H-bonded ⇒ native-like ⇒ more populated, for a
folded-protein MD ensemble), not a derived fact. It is exactly the kind of claim a future
shuffled-control / sign-flip sanity check should verify, since getting the sign wrong
actively pushes weight the wrong way rather than merely being a weaker prior.

### Penalty and its exact cheap form

```
L = [ Σ_{i<j} A_ij · ((logit_i - logit_j) - (G_i - G_j))² ] / Σ_{i<j} A_ij
```

**v1 uses uniform edge weights `A_ij = 1`.** Profile-similarity-weighted `A_ij`
(correlation of full per-residue `log_Pf` profiles between frames, gating how strongly
the Boltzmann-consistency pull applies between structurally dissimilar frames) is real
future work but needs the full `(n_residues, n_frames)` log_Pf matrix and is genuinely
`O(n_frames²)` — explicitly out of scope for v1.

For uniform edges, let `x_i = logit_like_i - G_i`. Using the identity
`Σ_{i<j}(x_i-x_j)² = n · Σ_i(x_i - x̄)²` (verified exactly, not approximately) and
`Σ_{i<j} 1 = n(n-1)/2`:

```
L = (2n / (n-1)) · Var(x)                  where x = logit_like - G
```

This is the exact value of the `O(n²)` pairwise sum, not merely "same optimum" —
implement this closed form directly; use the literal `O(n²)` pairwise sum only as a test
oracle. At the realistic scale in this codebase's examples (`n_frames` ~ 500-1200), this
is the difference between a trivial `O(n)` reduction and a ~250k-1.4M-entry pairwise
matrix every optimizer step.

### Accepted tradeoff (do not solve now, just document)

`G_i` depends on the currently-optimized `bc`/`bh`, so gradients from this loss reach
`bc`/`bh` as well as `frame_weights` whenever both are being jointly optimized (active in
some `3_CrossValidationBV`/deprecated sweep scripts; not in the single-run
`frame_weights`-only examples). The optimizer could in principle satisfy the penalty by
moving `bc`/`bh` (flattening the effective PF landscape) rather than by adjusting
weights. This is a known consequence of choosing a dynamic (recomputed every step) graph
over a static one — accepted, not fixed here.

## Implementation plan

### New module: `jaxent/src/opt/loss/boltzmann_consistency.py`

Not `weights.py` (KL-to-prior losses, no BV coupling) and not `consistency.py` — that
module's `create_consistency_loss`/`pairwise_cosine_similarity`
(`jaxent/src/opt/loss/base.py:61-78`) is confirmed degenerate when applied to the 1-D
`frame_weights` vector: cosine similarity of scalars collapses to `sign(w_i)·sign(w_j)`,
a constant `+1` for always-positive post-softmax weights. Do not build on it.

```python
def compute_total_bv_contacts(input_features: BV_input_features) -> tuple[Array, Array]:
    """Reduce (n_residues, n_frames) contacts to (n_frames,) totals."""

def boltzmann_penalty_pairwise_reference(logit_like: Array, G: Array) -> Array:
    """O(n_frames^2) direct pairwise sum, uniform A_ij=1 — test oracle only."""

def create_boltzmann_frame_consistency_loss(eps: float = 1e-12) -> JaxEnt_Loss:
    def boltzmann_frame_consistency_loss(
        model: InitialisedSimulation,
        dataset: Simulation_Parameters,   # unused; kept for JaxEnt_Loss signature parity
        prediction_index: int,            # bv_idx: position in model_parameters/_input_features
    ) -> tuple[Array, Array]:
        bv_params = model.params.model_parameters[prediction_index]
        bc, bh = bv_params.bv_bc, bv_params.bv_bh

        feats = model._input_features[prediction_index]
        total_heavy = jnp.sum(jnp.asarray(feats.heavy_contacts), axis=0)
        total_acceptor = jnp.sum(jnp.asarray(feats.acceptor_contacts), axis=0)
        G = bc * total_heavy + bh * total_acceptor

        logit_like = jnp.log(model.params.frame_weights + eps)

        n = logit_like.shape[0]
        loss = (2.0 * n / (n - 1)) * jnp.var(logit_like - G)   # note sign: minus G
        return loss, loss

    return boltzmann_frame_consistency_loss

@register_loss("boltzmann_frame_consistency")
def boltzmann_frame_consistency_builder():
    return create_boltzmann_frame_consistency_loss()
```

`register_loss`/`LossRegistry` imported from `jaxent.src.opt.loss.base` (confirmed
decorator pattern, matching `jaxent/src/opt/loss/weights.py:94,105`).

**`bv_idx` resolution:** use `prediction_index` directly — it's already threaded through
`run_optimise`'s `indexes=[...]` and zipped 1:1 with `loss_functions` in `compute_loss`.
No new key-based dispatch needed; unlike `maxent`'s `prediction_index: None`, this loss
requires a real int identifying the BV model's slot.

**Registration:** callers add `import jaxent.src.opt.loss.boltzmann_consistency  # noqa: F401`
(side-effect import, matching the existing `import jaxent.src.opt.loss.weights` pattern —
`jaxent/src/opt/loss/__init__.py` has no central re-export list).

### Example wiring: `jaxent/examples/5_SAXS/fitting/fit_CaM_HDX_KLD.py`

(Not `fit_CaM_SAXS_KLD.py` — that script has no BV model in `forward_models`, so it
cannot host this loss.) `fit_CaM_HDX_KLD.py` already uses `bv_model` at index 0:

- Add `import jaxent.src.opt.loss.boltzmann_consistency  # noqa: F401`
- `--maxent-strength` → `--boltzmann-strength` (`type=float, required=True`)
- `LossRegistry.get("maxent_convex_kl")` → `LossRegistry.get("boltzmann_frame_consistency")`
- `forward_model_weights=jnp.array([1.0, args.maxent_strength])` →
  `jnp.array([1.0, args.boltzmann_strength])` (both `init_params` and `prior_params`)
- `loss_functions=[hdx_loss_fn, maxent_loss]` → `[hdx_loss_fn, boltzmann_loss]`
- `indexes=[0, 0]` unchanged — the second `0` now does double duty as `bv_idx`
- `data_to_fit=[hdx_dataloader, prior_params]` unchanged — `prior_params` becomes an
  unused-but-harmless `dataset` placeholder
- rename `maxent_strength` → `boltzmann_strength` in run-name/config-dict fields

### Tests: `jaxent/tests/unit/opt/test_boltzmann_frame_consistency.py`

Follow the direct-construction pattern in `jaxent/tests/unit/opt/test_generalised_loss.py:100-119`
(build `Simulation(input_features=[...], forward_models=[], params=...)`, set
`sim._input_features` manually — no need for `sim.initialise()` since this loss touches
neither `.outputs` nor the JIT forward pass).

- `test_On2_reference_equals_On_reduction` — parametrize `n_frames ∈ {5, 10, 50, 137}`,
  random `logit_like`/`G`; assert `boltzmann_penalty_pairwise_reference(...)` equals the
  closed form `(2n/(n-1))·Var(logit_like - G)` to `atol=1e-5`. Load-bearing given the
  "exact, not approximate" claim above.
- `test_gradient_flows_to_frame_weights_and_bc_bh` — wrap `Simulation.forward` + the loss
  in one function of `Simulation_Parameters` (mirroring the real `compute_loss` path so
  it exercises the softmax boundary), `jax.grad` w.r.t. the full pytree, assert nonzero
  finite gradient on both `params.frame_weights` and
  `params.model_parameters[0].bv_bc`/`.bv_bh`. Documents the accepted bc/bh-drift
  tradeoff — not something to "fix".
- `test_zero_penalty_when_frames_identical` — two frames with identical
  `heavy_contacts`/`acceptor_contacts` columns (`G_1 == G_2`) and equal frame-weight
  logits (`log_w_1 == log_w_2`) ⇒ loss exactly `0.0`.
- `test_registry_contains_boltzmann_frame_consistency` — matches
  `jaxent/tests/unit/opt/test_loss_registry.py` style.

## Non-goals (this plan)

- Profile-similarity-weighted edge weights (`A_ij` beyond uniform).
- Validation methodology (shuffled-graph negative control, synthetic-truth comparison) —
  decide scope later.
- Any change to `jaxent/src/opt/gradients.py` or `optimiser.py`'s `_step`/`_step_with_rates`
  — this is purely a new loss term in the existing registry, exactly like MaxEnt.

## Verification (once implemented)

- `pytest jaxent/tests/unit/opt/test_boltzmann_frame_consistency.py -v`
- `pytest jaxent/tests/unit/opt/test_loss_registry.py -v` (registration still intact)
- Run `fit_CaM_HDX_KLD.py --boltzmann-strength <value>` for a short `n_steps` smoke run,
  confirm it completes and the loss value is finite and changes across steps.
- Existing MaxEnt-based examples/tests must be unaffected (loss module addition, no
  changes to `weights.py`, `create_parameter_loss`, or any existing registered loss).

## Critical files

- `jaxent/src/opt/loss/boltzmann_consistency.py` (new)
- `jaxent/src/opt/loss/base.py` (reference: `JaxEnt_Loss`, `LossRegistry`, `register_loss`)
- `jaxent/src/models/core.py` (reference: `Simulation.forward`/`normalize_weights` boundary)
- `jaxent/examples/5_SAXS/fitting/fit_CaM_HDX_KLD.py`
- `jaxent/tests/unit/opt/test_boltzmann_frame_consistency.py` (new)
