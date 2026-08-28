This experiment uses an MD dataset to investigate protection factors against the Boltzmann
distribution. JAX-ENT computes log protection factors, which are related to the experimental opening
free energy by `DeltaG_open = RT ln(PF)` under the usual EX2 interpretation. The conformational PMF
is `F_conf = -RT ln(p)`, so the physical comparison is `F_conf` against `-DeltaG_open`.

> **Current status (2026-08-28).** The occupancy-based Stage 1 and exploratory Stage 2 below are
> retained as a historical, falsified track. Stage 2 already fitted `(bc,bh)` separately for every
> system with leave-one-replica-out evaluation; it did not pool proteins into one fit. Its parameter
> collapse was caused by scale non-identifiability in the Absolute-L1 PMF objective. The current
> redesign measures fixed-coefficient PF geometry against pairwise C-alpha RMSD and coordinate-W1
> across all 111 systems, without basin calls or occupancy bins. Its strict six-way conformal audit
> is now complete: global common-support coverage is approximately nominal, but common-support
> W1-q5 coverage is only 60.0% for the selected per-residue ridge model (95% CI 54.5–65.3%). The
> evidence therefore supports a representation limit in fixed BV features for large structural
> departures, while retaining useful residue-aware signal. See `CHECKPOINT8_STRICT_CONFORMAL.md`.

# Aim
The investigation aims to understand whether the laplacian concept in plans/ makes sense from a physical basis. The question we wish to answer is how differences in population are related to differences in measured protection factors. The options for the distance space is:
L1, Absolute-L1, and L2. 

We can leverage jaxent to fine tune parameters of the forward model to better fit the boltzmann distribution as well as test other structure-HDX forward models later down the line.


# Experiments:

1. First experiment will simply use the defualt parameters and test Signed-L1, Absolute-L1 and L2 and plot the population distribution from the highest density point. Ideally we should compare against an actual energy function which we can do as we have MD trajectories but a simpler approach like FoldX or even simple RMSD might be sufficient.

2. If #1 shows that the concept is reasonable we can then fit BV parameters to a boltzmann distribution using the best distnace metric and plot the distribution of the BV parameters over all systems.

3. Repeat with:
- BV+B0 model: N_c*\beta_c + N_h*\beta_h + beta_0
- M8 model: \beta_s*SASA^(-\gamma_s) + \beta_p*dist_polar^(-\gamma_p)
- M8+B0 model: \beta_s*SASA^(-\gamma_s) + \beta_p*dist_polar^(-\gamma_p) + beta_0

Notes:
*all N_contacts use a switch function.
**dist_polar is the distance to the nearest polar atom.
***BV+B0 can be implemented very easily but M8 requires SASA features which have not been integrated yet.
****SASA is computed at the amide-H probe position.
*****M8 is from Mohammadiarani, Shaw, Neubig & Vashisth, J. Phys. Chem. B 2018 (PMC6430106).
It is M8 in that paper's numbering, not M7 — M7 there is a *fractional population* model
`PF = tau_C/tau_O`, a different class of model entirely (see Implementation plan §4).

4. Test again with MISATO (ligand MD dataset)


# Implementation plan
The plan is staged so that each stage produces a falsifiable readout before the next is built. Stages
1–2 need **no new forward-model code** — everything runs off the per-frame contact features that
`jaxent/cli/featurise.py` already writes. Stage 3 is where new model code (`b0`, SASA) is required.

## Current geometry measurement

Run `./jaxent/examples/ATLAS_BV/commands.sh geometry-stage1 --limit 1` as a smoke test, then remove
`--limit` for all systems. Each fold trains an isotonic mapping from PF distance to pairwise C-alpha
RMSD on two replicas and measures normalized MAE, skill, rank correlation, conditional 90% coverage,
and RMSD-conditioned error curves on the third. The fixed default BV model is evaluated with
normalized L1 and L2 distances under a full raw/centred/residue-standardised factorial. RMSD bands
measure scale dependence only; they are not conformational-state definitions.

JSD/KL is deferred until log-PF is converted into an opening-probability model. A later likelihood
stage should use `p_open = sigmoid(-logPF)` and compare Bernoulli marginals before attempting a
correlated joint model. The empirical 60--65% unfitted and 85--90% fitted conformational-recovery
ranges remain descriptive context and are not gates for this ATLAS measurement.

**Observed geometry measurement (2026-08-27):** all 111 systems and all 333 held-out-replica folds
completed with finite results. Median held-out skill relative to a training-median RMSD predictor
was `0.2105` for raw L1 and `0.2381` for raw L2. Frame centring produced `0.2339`/`0.2403`, whereas
residue standardisation reduced performance. Rg and native-contact controls reached only `0.0518`
and `0.0273`. Across the six held-out RMSD bands, median raw L1 distance rose from `6.18` to `10.03`
and raw L2 from `8.11` to `14.71` as median C-alpha RMSD rose from `1.11` to `2.88 Å`. Nominal 90%
conditional interval coverage was about 84%, indicating remaining distributional miscalibration.
No arm has been promoted and Stage 2 remains unauthorized.

## 0. The physical statement being tested

An equilibrium MD trajectory is *already* a Boltzmann sample: every frame carries equal weight, and
the free energy of a region of configuration space is read off its **occupancy**, not from any
per-frame energy. So the test does not need an external energy function at all in its primary form.

Write the per-frame, per-residue BV log protection factor (`BV_ForwardPass`,
`jaxent/src/models/HDX/forward.py:31`):

```
z_f(r) = bc·h_f(r) + bh·o_f(r)            ΔG_op,f(r) = RT · z_f(r)
```

and the per-frame scalar aggregate used as the Boltzmann coordinate
(the `G_i` of `plans/hdx_boltzmann_frame_weight_consistency_loss.md`, reused as an instrument in
`plans/hdx_rate_space_pivot_reweighting.md` §7b):

```
G_f = Σ_r z_f(r) = bc·H_f + bh·O_f        H_f = Σ_r h_f(r),  O_f = Σ_r o_f(r)
```

If protection tracks conformational stability, then `E_f ≈ −RT·G_f + const`, and Boltzmann gives

```
ln p_f  =  α · G_f + const                                        (★)
```

**The claim under test is that `α` is positive, stable, and large enough to matter — not that
`α = 1`.** `α` is dimensionless (`RT` cancels), which makes it a clean effect size, but a unit slope
is not the hypothesis: `G` is a *sensitivity* claim about an empirical coordinate, not an assertion
that `G` is the microscopic potential energy. `α ≈ 0`, or a sign that flips across systems, is what
falsifies the coordinate. A stable `α ≠ 1` means the relation is real and the BV coefficients are
miscalibrated — which is exactly what Stage 2 fits.

For sign consistency, statistical fits use `ln(P/P_ref)` against `G-G_ref`, so the expected slope
is positive. The physical difference `ΔF=-RT·ln(P/P_ref)` is reported in kcal/mol and necessarily
has the opposite sign relative to `G-G_ref`.

Equivalently and more robustly, `F(G) = −RT ln p(G)` is the **potential of mean force along the BV
coordinate**. The density-of-states term is part of that PMF by definition, not a contaminant. See
§2.3 for the framing this implies, and §2.4–§2.5 for the basin-level, held-out form in which the
question is actually tested.

### Where the three distance spaces enter

Relative to a reference `*` (the most populated structural basin, §2.3b):

| space | coordinate | type |
|---|---|---|
| **Signed-L1** | `s_f = Σ_r (z_f(r) − z_*(r)) = G_f − G_*` | signed linear functional — this *is* `ΔG` |
| **Absolute-L1** | `d¹_f = Σ_r \|z_f(r) − z_*(r)\|` | metric — non-negative, non-linear |
| **L2** | `d²_f = ‖z_f − z_*‖₂` | metric — non-negative, non-linear |

Signed-L1 is the primary coordinate because it is exactly the aggregate protection change, so its
PMF maps directly onto `ΔG`. Abs-L1 and L2 answer a **secondary and genuinely different** question:
whether population is better organised by the *direction* of BV protection change or merely by its
*magnitude*. Both are distances, so their PMFs necessarily include a volume term — that is a
property of the coordinate, not a defect to correct, and it is why they are not interchangeable with
Signed-L1 as free-energy coordinates.

Two known weaknesses to record now rather than discover later:

1. `G_f` sums a **local, per-residue opening** free energy over residues. Local openings are not
   independent and their sum is not a global stability; `Σ_r ΔG_op` therefore over-counts and is at
   best proportional to `ΔG_global`. This is a further reason not to demand `α = 1`: a stable
   `α ≠ 1` is the expected signature of exactly this, and is a *useful* result.
2. Only residues that are resolved and non-terminal contribute; `mda_selection_exclusion="resname
   PRO"` and the chain-wise N-terminal policy already drop residues. `G_f` must be summed over an
   **identical residue set across all frames of a system** (guaranteed within one trajectory) and
   the residue count reported, since `G` is extensive and `α` is not comparable across proteins
   without it (see §2.7 normalisation).

## 1. Data acquisition (`data/`)

Target dataset: **ATLAS** (Vander Meersche et al., *NAR* 2024, D384), `https://www.dsimb.inserm.fr/ATLAS`.
All facts below were verified against the live API and one archive on 2026-08-26.

### 1.1 What ATLAS actually ships

Per system (`{pdbid}_{chain}`, e.g. `3omd_B`), three archive tiers exist:

| archive | size (3omd_B) | trajectory | contents |
|---|---:|---|---|
| **`_analysis.zip`** | **35 MB** | **1,001 frames @ 100 ps** | **what this study uses** |
| `_protein.zip` | 331 MB | 10,001 frames @ 10 ps | full-rate trajectory + `.tpr` |
| `_total.zip` | 2.96 GB | 10,001 frames @ 10 ps | full system incl. explicit solvent |

**Use `_analysis.zip`.** ATLAS already publishes a 10×-strided trajectory, and it is not a reduced
*representation* — verified directly on `3omd_B_R1.xtc` with MDAnalysis:

```
atoms 2343   (1150 hydrogens, 1193 heavy)     residues 145
amide N 145      amide H 134                  frames 1001,  t = 0 … 100,000 ps @ 100 ps
```

**All-atom including hydrogens**, so the amide N and amide H that
`calc_BV_contacts_universe` needs are both present. The 10× saving is entirely in frame rate, which
costs this experiment nothing: the readout is an equilibrium *density*, not short-time dynamics, and
a coarser stride only moves the samples closer to independence (§3.2c).

`_analysis.zip` contents:

```
{id}.pdb                       structure after minimisation+equilibration (MD start)
{id}_R{1,2,3}.xtc      11 MB   trajectory, solvent-free, 1,000 frames / 100 ns
{id}_R{1,2,3}.tpr     1.4 MB   topology + run parameters
{id}_corresp.tsv               author (PDB/CIF) residue number  ->  renumbered-from-1
{id}_RMSD.tsv  {id}_gyrate.tsv                per-replica, per-ns
{id}_RMSF.tsv  {id}_Neq.tsv  {id}_Bfactor.tsv  {id}_pLDDT.tsv   per-residue
{id}_contacts.tsv              co-crystallised chain/ligand/ion/nucleotide within 6 Å of Cα
README.txt
```

So the archive that carries the sliced trajectory also carries the RMSD / Rg / RMSF series needed
for §2.7's structural proxies and §1.5's convergence gate — no second download.

`{id}_corresp.tsv` matters more than it looks: ATLAS renumbers residues from 1, and jaxent's
topology alignment is strict (`jaxent/src/interfaces/topology/`). Carry this mapping into the
manifest so `Partial_Topology` residue IDs can be traced back to author numbering.

Drop to `_protein.zip` only if the frame count turns out to bind — 3,000 frames/system across three
replicas is ample for the binned occupancy fit, but it does tighten §2.4's "≥ 20 frames per bin"
rule, so report bin occupancies. `_protein.zip` remains the fallback and its URL pattern is
identical with `protein` in place of `analysis`.

Simulation parameters (`curl https://www.dsimb.inserm.fr/ATLAS/api/MD_parameters -o params.zip`,
`3_production.mdp`), verified:

```
integrator = md, dt = 0.002 (2 fs), nsteps = 5e7 (100 ns)
nstxout_compressed = 5000 (10 ps, before the analysis-tier stride)
tcoupl = Nose-Hoover,  ref_t = 300 K
pcoupl = Parrinello-Rahman, ref_p = 1.0 bar (NPT), CHARMM36m
```

**`ref_t = 300 K` uniformly across ATLAS**, so `BV_model_Config.temperature`'s default is correct
here and `RT = 0.5922 kcal/mol`. The ensemble is NPT rather than NVT; the `pV` term is negligible at
this scale, but state it in the writeup rather than leaving the ensemble unnamed.

Both trajectory tiers are solvent-free, so `mda_contact_environment` is unambiguous for ATLAS.

Licence: **CC-BY-NC 4.0**. Non-commercial use with attribution; cite the NAR paper.

### 1.2 The master index (drives system selection)

Do not scrape the site. One request returns the full catalogue:

```
curl -s "https://www.dsimb.inserm.fr/ATLAS/api/parsable?dataset=ATLAS" -o parsable.zip
unzip parsable.zip     # -> ATLAS_parsable_latest/2023_03_09_ATLAS_info.tsv  (+ _pdb.txt)
```

`2023_03_09_ATLAS_info.tsv` holds **1938 entries × 50 columns** (note: more than the 1390 quoted in
the 2024 paper — the database has grown, so take counts from the TSV, not the paper). Columns
relevant to selection:

`PDB` (the `{pdbid}_{chain}` accession used in every download URL), `length`, `PDB_resolution`,
`contact_chain`, `contact_ligand`, `contact_ion`, `contact_nucleotide`, `no_contact`, `alpha%`,
`beta%`, `coil%`, `avg_RMSF`, `avg_gyration`, `CATH_class`/`architecture`/`topology`,
`ECOD_*`, `SCOP_*`, `non_redundant_protein`, `non_redundant_domain`, `sequence`.

`avg_RMSF` is ATLAS's own flexibility annotation — use it directly to stratify, rather than
inventing a flexibility measure.

### 1.3 Selection, with the real pool sizes

Filter counts computed from the TSV:

| filter | systems |
|---|---:|
| all | 1938 |
| `60 ≤ length ≤ 250` | 1231 |
| + `non_redundant_protein == True` | 670 |
| + no ligand and no nucleotide contact (ions allowed) | **111** |
| + strict `no_contact == True` | 31 |

**Use the 111-system relaxed pool**, not the strict 31. Strict `no_contact`
costs ~4× the pool for little gain: the ATLAS MD is of the isolated chain in water regardless of the
crystal's contacts, so the contact flags bear on whether the *starting conformation* is
apo-representative, not on what is simulated. Excluding ligand and nucleotide contacts is the part
that matters; ion contacts are not worth the pool.

The relaxed pool is well spread — CATH `Alpha Beta` 32, `Mainly Alpha` 22, `Mainly Beta` 13,
unclassified 29 — and spans `avg_RMSF` 0.55 / 0.79 / 1.13 / 1.84 / 8.68 Å (min/Q1/median/Q3/max).

**Take all 111 — do not subsample.** At the analysis tier the whole pool costs ~6 GB and ~2 h of
featurisation (§1.4), so a draw would buy nothing and cost cross-system resolution. Stage 2's
deliverable is the *distribution* of `(bc, bh)` across systems, and 111 supports a real distribution
where 20 supports an anecdote. It also means no sampling design to defend and no seed to report.

`data/select_systems.py` therefore applies the filters above and emits all 111 rows to
`data/systems.csv`:
`system_id, length, cath_class, avg_RMSF, avg_gyration, resolution, pdb_path, replica_paths, n_frames`.
Record the CATH class × `avg_RMSF`-tercile cell for each system anyway — not to sample on, but so
that §2.6 can report whether `α` varies systematically with fold class or flexibility, and so that
any post-hoc exclusions (§1.5) can be checked for stratum bias.

Scaling beyond 111 is a filter change, not a redesign: dropping the non-redundancy requirement gives
~200, and the whole database is ~1,938 systems / 104 GB / ~35 h (§1.4). Keep `select_systems.py`
parameterised by the filter predicate so that stays a one-line change.

### 1.4 Fetching

URL pattern (verified, no auth, supports HTTP range requests):

```
https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{id}/{id}_analysis.zip
# equivalently: https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{id}
# fallback tier:                        .../{id}/{id}_protein.zip
```

`data/fetch_atlas.sh` reads `systems.csv` and pulls `_analysis.zip` per system. Requirements:
- **≤ 2 concurrent connections** and resume enabled (`curl -C -` or `aria2c -x2`); this is a small
  academic server. ATLAS's own bulk script uses `aria2c`. Do not parallelise aggressively.
- record `content-length` per file and re-fetch on mismatch;
- unpack to `data/raw/{id}/`, keep the zip until unpack verifies, then drop it.

#### Measured budget

Archive size is essentially linear in chain length (59 systems sampled by HEAD across 38–751
residues):

```
size_MB = 1.72 + 0.2265 x n_residues      R^2 = 0.997,  residual sigma 1.8 MB
```

Dataset mean length is 229 residues (median 175, max 2128), so the mean system is **54 MB**, not the
35 MB of the 145-residue `3omd_B` — do not scale from that one system.

The pre-run estimate used a historical hard-contact benchmark on `3omd_B_R1.xtc`: 6.9 s for 145
residues × 1,001 frames for one contact type. The required Bradshaw protocol was then measured on
the median target `1u6t_A_R1` (121 residues × 1,001 frames, both contact types): **6.12 s wall,
23.27 s user CPU, 312 MiB peak RSS**. The output contains 113 eligible amides × 1,001 frames and
passed finite/non-negative validation.

| scope | download | frames | featurisation |
|---|---:|---:|---:|
| **111-pool (this study)** | **~6 GB** | **~0.33 M** | **~0.60 h** |
| 500 systems | 27 GB | 1.5 M | ~2.7 h |
| all 1,938 | 104 GB | 5.8 M | ~10.5 h |

(Full-rate `_protein.zip` for comparison: ~990 GB for the database; `_total.zip` ~8.8 TB.)

Featurisation parallelises trivially across systems, so ~0.60 h is a single-process projection
(~0.72 h with a 20% margin). Two caveats
on these numbers: the size fit is calibrated on 38–751 residues, so the few systems above that are a
modest extrapolation; and the timing assumes cost linear in residue count, whereas neighbour search
may grow slightly superlinearly, so treat the hours as a floor.

**The untested quantity is sustained download throughput** at ≤2 connections from a small academic
server. At 6 GB that is unlikely to bind, but time one full system before launching the batch — at
the 104 GB scale it would become the dominant cost, ahead of CPU.

### 1.5 Convergence gate

**Mandatory, before any analysis.** The Boltzmann assumption is the whole experiment, so it must be
tested rather than assumed — and 100 ns is short enough that it will genuinely fail for some
systems. Per system:

- discard the first 10 ns of each replica as equilibration (leaves 900 of 1,000 frames at the
  analysis tier, i.e. 2,700 frames per system across three replicas);
- split each remaining replica into halves and compare the `G` histograms (two-sample KS,
  Jensen–Shannon divergence);
- compare `G` distributions **between the three replicas** — this is the strong test, since the
  replicas are independent;
- cross-check against the RMSD/Rg traces in `_analysis.zip` (§1.4).

A system whose replicas disagree is not sampling a Boltzmann distribution on this timescale; exclude
it from the primary analysis and report it as excluded. Write the verdict to
`data/convergence_report.csv`. Expect a substantial fraction to fail — that is a result about the
dataset, not a bug, and it is worth reporting against `avg_RMSF`, since the flexible systems that
fail hardest are also the ones where the `G` coordinate would have the most dynamic range.

Because all 111 systems are taken (§1.3) rather than sampled, exclusions simply reduce `N` — there
is no draw to re-balance. Do check the **survivors for stratum bias**, though: if failures
concentrate in the high-`avg_RMSF` tercile or one CATH class, the surviving set is no longer
representative and §2.6's cross-system claims must be stated over the survivors, not over ATLAS.

## 2. Stage 1 — fixed Bradshaw-switch parameters, three distance spaces

Directory: `analysis/stage1/`. Fixed `bc = 0.35`, `bh = 2.0`; no fitting. All contacts use
Bradshaw's `rational_6_12` switch from
`examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/code/calc_hdx/Functions.py`:

```
S(r; d0, k) = 1 / (1 + ((r - d0) / k)^6)
```

The removable singularities in the reference quotient form are evaluated by this algebraically
equivalent expression. Use `d0 = 6.5 Å, k = 10 Å` for heavy contacts and `d0 = 2.4 Å, k = 10 Å`
for acceptor contacts. The switch is summed over **every eligible atom pair**, including pairs with
`r > d0`; `d0` is the midpoint, not a truncation radius. This samples the complete trajectory and
does not impose a single structural basin. Structural basins are identified later and are used for
the held-out comparisons in §2.3b–§2.5.

### 2.1 Featurise

Before the 111-system batch, run only the median-system benchmark:

```
./jaxent/examples/ATLAS_BV/commands.sh benchmark
```

This benchmarks `1u6t_A_R1` (121 residues, 1,001 frames), writes timing and validation metadata to
`outputs/benchmark/1u6t_A/R1/benchmark_report.yaml`, projects the full cost from the measured
throughput, and then stops. The prior estimate printed before execution is 11.5 s for this replica
and 1.13 serial hours for all 111 systems. Do not launch the full batch until this report has been
reviewed.

After accepting the benchmark gate, the complete resumable batch is:

```
./jaxent/examples/ATLAS_BV/commands.sh featurise --workers 2
```

The production run completed on 2026-08-26 in **25.1 minutes** using two concurrent workers with
ten contact threads each. All **333/333 replicas** passed independent shape, finiteness, and
non-negativity validation: 333,333 total frames, 82,096,014 contact-feature values, and 336 MiB of
Stage-1 output. Eligible amides range from 55–237 per system-replica (median 116). Per-replica wall
time was 6.85 s median / 30.55 s maximum; maximum observed RSS was 351 MiB. The resumable audit
record is `outputs/stage1/batch_report.yaml`.

Reuse the existing pathway; do not write a new featuriser.

```
python jaxent/cli/featurise.py --top <pdb> --traj <xtc> --output_dir outputs/<system>/<replica> ...
```

This writes `features.npz` (`BV_input_features`: `heavy_contacts`, `acceptor_contacts`, both
`(n_residues, n_frames)`) and `topology.json`. `jaxent/examples/predict_traj/run_predict_dir.sh`
is the closest existing driver — copy it to `commands.sh` and adapt the directory walk. Everything
downstream reads `features.npz`; no trajectory re-reads.

`_protein.zip` is already solvent-stripped (§1.1), so `mda_contact_environment` is unambiguous for
ATLAS — still write the value into the manifest, because MISATO (§5) reopens exactly this choice.

### 2.2 Per-frame quantities

`analysis/stage1/compute_frame_coordinates.py` → `frame_coords.parquet` per system/replica:
`frame, H_f, O_f, G_f, s_f, d1_f, d2_f, rmsd_to_ref, rg, n_contacts_native`.

`z_f` never needs materialising through `Simulation`; `G_f = bc·H_f + bh·O_f` is a two-number
contraction of `features.npz`, which is what makes Stage 2 a closed-form fit.

### 2.3 What is actually being tested (framing)

The question this stage answers is **empirical, not microscopic**:

> Does the BV coordinate `G` vary in a way that is sensitive to conformational free-energy
> differences?

It is *not* "is `G` the microscopic potential energy", and it does not require the Boltzmann slope
`α` to equal 1. Two consequences for the design:

1. **`F(G) = −RT ln p(G)` is the potential of mean force along the BV coordinate.** The
   density-of-states contribution is *part of* that PMF, not an artifact contaminating it. So
   curvature in `F(G)` is not evidence of a broken coordinate, and there is nothing to "correct out".
   What survives from the linear-functional/metric distinction of §0 is interpretive: for Signed-L1,
   `G` is a linear functional of `z`, so the PMF along it maps directly onto the protection change;
   for Abs-L1 and L2 the coordinate is a *distance*, so its PMF necessarily folds in a volume term
   and the profile answers a different question — whether population is organised by the **direction**
   of BV protection change or merely by its **magnitude**.
2. **Basins, not 1-D bins, are the primary unit.** The headline result is basin-level and
   held-out — see §2.4 — with the 1-D PMF profile demoted to a descriptive secondary readout (§2.6).

### 2.3b Structural basins (must be defined independently of `G`)

**Hard requirement: basins are defined in structural space only.** Defining them in log-PF space
makes the basin↔`G` comparison circular and destroys the result. So the earlier `ref_z` mode is
demoted to a descriptive diagnostic; `ref_struct` becomes the operative definition.

Per system:
- pool frames from **all three replicas**, so basin labels are shared and per-replica populations are
  comparable (this is what makes §2.5's held-out test possible);
- cluster on Cα RMSD (or PCA of Cα coordinates), using the repo's existing clustering path
  (`jaxent/cli/efficient_k_cluster.py`, and `extract_OpenClosed_clusters.py`-style RMSD assignment as
  used in `plans/hdx_rate_space_pivot_reweighting.md` §7b);
- fix the basin count per system by a stated criterion (silhouette or an explicit RMSD threshold),
  and **record it** — it is a free parameter and must not be tuned against the outcome;
- take the most populated basin as `ref`;
- carry `-1`/unassigned frames as an explicit class, never a silent drop.

**Counting-noise floor.** With ~900 post-equilibration frames per replica, a basin holding 50 frames
has ~14% relative SE on its population and ~0.08 kcal/mol on `ΔF`. Require **≥ 50 frames per basin
per replica** for a basin to enter the primary analysis, which in practice caps usable basins at
roughly 3–8 per system.

**This is the experiment's main practical risk, and it should be measured first.** Many ATLAS systems
at 100 ns will occupy a single basin — a folded protein that never leaves its native well gives
`ΔF ≈ 0` across the board and has no dynamic range to detect anything. The systems with real basin
structure are the flexible ones, which are also the ones most likely to fail the §1.5 convergence
gate. **Before building any of the fitting machinery, run the basin census over the 111-pool and
report the distribution of (usable basins per system, `ΔF` range per system).** If most systems yield
one basin, the basin-level design cannot be the headline and the plan needs rethinking — better to
learn that from a ~2 h featurisation run than after Stage 2 is built.

**Observed census (2026-08-26): this gate failed.** With pooled Cα-RMSD DBSCAN (`eps=2 Å`,
`min_samples=50`) and the requirement of at least 50 frames in every replica, 89 systems have one
usable basin and 22 have none; **zero systems have three usable basins**. Some raw DBSCAN labels
split further, but those small states fail the per-replica counting floor. The required minimum of
20 informative systems is therefore impossible. `outputs/analysis/stage1_decision.yaml` records
the census decision. This motivated the within-basin redesign below rather than forcing artificial
K-means subdivisions.

### 2.3c Adopted redesign — continuous fluctuations within the shared basin

The adopted `within_basin_pc1_v1` protocol retains the 89 systems whose dominant DBSCAN basin has
at least 50 frames in every replica. It does not subdivide that basin. For each held-out-replica
fold, PCA is fitted on **Cα intramolecular pairwise-distance vectors from the other two replicas**;
the held-out replica is transformed without refitting. PC1 is therefore structural, invariant to
rigid motion, and independent of `G`.

Training PC1 is divided into fixed-width Freedman–Diaconis bins (15–30 bins). A bin enters the
primary comparison only with at least 20 training and 20 held-out frames; each fold needs at least
five common bins and may place no more than 5% of held-out frames outside the training range. The
response is the structural density

```
ln p(PC1_k) = ln[n_k / (N · ΔPC1)]       F(PC1_k) = -RT ln p(PC1_k)
```

and the predictor is the bin mean of `G` (Signed-L1). Absolute-L1 and L2 are sensitivity arms;
`H`, `O`, `Rg`, RMSD, and native contacts are controls. Fits use two replicas and predict the third.
Contiguous-block permutations use the measured `G` autocorrelation time. This tests whether BV
orders continuous population fluctuations *inside* a folded basin; it does not claim to measure
inter-basin opening free energies.

Of 111 systems, 89 have a shared dominant basin and 65 support all three held-out PC1 folds (median
nine common bins). Replica KS/JS disagreement is retained as a sampling-uncertainty diagnostic, not
as an exclusion gate: the experimental free-energy relation does not cease to exist because finite
replicas sample different portions of a basin. The primary analysis estimates a PMF separately per
replica and uses leave-one-replica-out transfer plus block resampling to quantify reproducibility.

**Primary coordinate correction (2026-08-27).** Absolute-L1, not Signed-L1, is the intended primary
hypothesis. All three spaces are now subjected to their own autocorrelation-aware block-permutation
null and reported side by side. Stage 2 authorization is determined only by the predeclared
Absolute-L1 population-null, compactness-control, and expected-negative-slope criteria.

The 65-system result is:

| coordinate | median held-out rho | population null 95% | one-sided p | expected sign | beats best compactness control |
|---|---:|---:|---:|---:|---:|
| Signed-L1 | 0.057 | 0.061 | 0.0649 | 66.2% | 15.4% |
| **Absolute-L1** | **0.279** | 0.068 | **0.000999** | **96.9%** | **52.3%** |
| L2 | 0.273 | 0.066 | **0.000999** | 90.8% | 50.8% |

Absolute-L1 therefore has a reproducible association with occupancy and the expected direction. It
does not pass the declared Stage 2 gate solely because 52.3% is below the 60% compactness-control
threshold. This is a failure to establish incremental performance over compactness, not a null
Absolute-L1 result and not a rejection of the experimental PF/free-energy relation.

### 2.4 Primary readout — basin-level log occupancy vs `ΔG`

**Superseded by §2.3c for the current analysis.** The material below records the original
multi-basin design and is retained for provenance.

For each basin `k` against the reference basin:

```
ΔF_k  =  −RT ln ( P_k / P_ref )                     P_k = n_k / N   (frames are equiprobable)
ΔG_k  =  ⟨G⟩_k − ⟨G⟩_ref                            (Signed-L1; and the Abs-L1 / L2 analogues)
```

Fit `ln(P_k/P_ref)` as the response and derive `ΔF_k` afterward for physical-unit reporting. Report,
per system:

- **monotonicity** — Spearman ρ between `ln(P_k/P_ref)` and `ΔG_k` over basins;
- **effect size** — the fitted slope `α` and the spread of `ΔF` actually spanned, in kcal/mol.
  Report both; a perfect correlation over a 0.2 kcal/mol range is not a result;
- **basin separation** — whether `G` distributions of distinct basins are separated relative to their
  within-basin width (e.g. AUC / Cohen's d for each basin pair), which is the direct measure of
  "sensitive enough";
- CIs from the §3.2c contiguous block bootstrap, not from frame-level resampling.

`α` is reported as an **effect size, not tested against 1**. Stage 2 exists to absorb a consistent
scale error into `bc`/`bh`.

### 2.5 Held-out-replica prediction (the headline result)

Basins are defined on pooled frames (§2.3b), so populations can be computed per replica. Then:

> Fit the `ln(P/P_ref) ↔ ΔG` relation on two replicas; predict the third replica's **basin
> ordering** and **`ΔF` values**.

Report Spearman ρ for the ordering and MAE in kcal/mol for the values, aggregated over the 3 folds
and over systems. **This is the number the experiment should be judged on** — it is the only readout
that is both quantitative and out-of-sample, and it directly answers "reliably distinguishes high-
and low-free-energy regions".

#### Controls (replacing the withdrawn residue-shuffle null)

**The previously proposed residue-shuffle null was wrong and is withdrawn.** Signed-L1, Abs-L1 and L2
are all symmetric functions of the residue components of `Δz`, so permuting residues within a frame
leaves all three coordinates numerically unchanged — the control has no effect to measure. Replaced
by two controls that do bite:

1. **Frame-label permutation.** Permute whole-frame `G_f` values against structural-bin labels and
   recompute the statistic, giving a null distribution for ρ and the slope. Because frames are temporally
   correlated, permute **contiguous blocks** (block length ≥ the autocorrelation time of `G`, §3.2c)
   or permute at the bin level; naive frame-wise permutation will overstate significance. The primary
   null test is the median held-out ρ across informative systems against its population permutation
   distribution. Requiring a majority of systems to *each* exceed its own 95th percentile would
   incorrectly conflate cross-system consistency with 65 simultaneous significance tests; individual
   exceedances are therefore diagnostic only.
2. **Baseline coordinates — the more demanding control.** Repeat the entire §2.4/§2.5 pipeline with
   `G` replaced by: `RMSD` to reference, `Rg`, raw total heavy-contact count `H_f` alone, and native
   contact count. The interesting question is not "does `G` beat noise" but **"does `G` beat
   compactness"** — `H_f` alone is essentially a compactness proxy, and if `G = bc·H + bh·O` does no
   better than `H` alone, BV's specific structure is adding nothing. Report held-out basin-ordering
   ρ for every coordinate side by side.

### 2.6 The 1-D PMF profile (secondary, descriptive)

Retained as a figure, not a test. Frames are equiprobable, so along any coordinate `c`:

```
F̂(c_k) = −RT ln ( n_k / (N · Δc) )
```

with ~2,700 post-equilibration frames per system, so budget ~20–30 bins; drop bins with `n_k < 20`
rather than merging, report the retained range, and show bin-width sensitivity (×0.5, ×2) since
`−ln p` binning artifacts are the classic way to manufacture a slope. Plot the profile for all three
distance spaces from the highest-density structural point, which is what the aim asks for.

Read these as PMFs (§2.3): curvature is expected and interpretable, especially for the two metric
coordinates, and is not grounds for rejecting a space.

### 2.7 Cross-system comparison

`G` is extensive in residue count, so report both raw and per-residue (`G/n_res`) versions of every
slope. Aggregate over the 111 systems into distributions of held-out ρ, MAE, and `α` per coordinate.

With `N ≈ 111` (fewer after §1.5 exclusions) the cross-system distribution is resolvable enough to
report shape, not just a median: show the full distribution per coordinate, and test against
`avg_RMSF` tercile and CATH class. **Stability across proteins is itself a primary criterion** — a
coordinate whose relationship holds across folds is a much stronger result than one whose median
merely differs from zero.

### 2.8 Independent energy reference (secondary)

The occupancy readout needs no energy function, which is why it is primary. Secondary cross-checks,
in increasing order of cost, on a subsample of frames:
- structural proxies (RMSD, `Rg`, native contacts) — free, and already required as §2.5 baselines;
- **FoldX** `Stability` on ~200 frames per system;
- implicit-solvent (GBSA) rescoring of the same frames.

Do **not** use the raw GROMACS potential energy of individual frames as the reference. In explicit
solvent its per-frame fluctuation is dominated by water and swamps any conformational signal — that
comparison would look like a null result for reasons unrelated to the hypothesis.

### 2.9 Stage 1 decision rule (write this down before running)

For `within_basin_pc1_v1`, an informative system must retain three valid held-out folds and span at
least 0.5 kcal/mol of held-out PC1 PMF. Replica convergence is reported but is not a hard exclusion.
Proceed to Stage 2 only with at least 20 informative systems, a one-sided population block-permutation
test at `p <= 0.05`, ≥60% beating the best compactness control, and ≥70% with a positive
`ln p(PC1)` versus `G` slope. The original rule below applies only to the superseded multi-basin design.

Proceed to Stage 2 **iff**, across systems passing the §1.5 convergence gate and the §2.3b basin
census:

- **held-out basin ordering** (§2.5) is positive with a clear majority of systems above the
  frame-label-permutation null;
- `G` **beats the compactness baselines** (`H_f` alone, `Rg`, RMSD) on held-out ordering in a
  majority of systems — not merely beating noise;
- the sign of the relationship is **consistent across systems**;
- the `ΔF` range spanned is large enough for the result to mean anything (state a floor, e.g.
  ≥ 0.5 kcal/mol, and report how many systems clear it).

`α` is **not** required to be 1, and curvature in the PMF is **not** disqualifying. What kills the
concept is: ordering no better than the permutation null, no improvement over `H_f` alone, a sign
that flips across systems, or a basin census with no dynamic range.

## 3. Stage 2 — fit BV parameters to the Boltzmann distribution

Directory: `analysis/absolute_l1_stage2.py`. Runs only if the corrected Absolute-L1 §2.9 gate passes.
Because the current result misses only the predeclared compactness-majority criterion, an explicit
`--exploratory-override` may continue without rewriting the Stage 1 decision. Every Stage 2 output
records that provenance and remains labelled exploratory.

### 3.1 Absolute-L1 thermodynamic estimator

The fitted frame coordinate is

```
d1_f(bc,bh) = Σ_r |bc·(h_f,r-h_ref,r) + bh·(o_f,r-o_ref,r)|
```

The density zero is profiled out by weighted centering, giving

```
ln p_k - weighted_mean(ln p) = -(mean_k(d1) - weighted_mean(mean(d1))).
```

There is no fitted intercept. A residue-level `beta_0` cancels exactly from `z_f,r-z_ref,r`, so it is
not identifiable in an Absolute-L1 experiment and must not be confused with the arbitrary density
normalisation. Positive `bc,bh` select the physical sign branch. The implementation searches their
direction on `[0,pi/2]` and profiles the non-negative scale analytically.

For `within_basin_pc1_v1`, reuse the held-out structural PC1 construction from Stage 1 and never
re-bin using a protection coordinate. Fit two replicas, predict the third, then fit all three for the
reported per-system coefficients and obtain moving-block-bootstrap intervals with fixed structural
bin edges.

Evaluate fitted and default `(0.35, 2.0)` coefficients on the identical held-out bins. The primary
Stage 2 statistic is the per-system three-fold mean `Delta rho = rho_fitted-rho_default`; calibration
is predictively successful only when its one-sided 95% paired population-bootstrap lower bound is
above zero. PMF MAE and performance relative to the strongest compactness control are reported as
secondary diagnostics.

### 3.1b The target must be occupancies, not frame weights

An equilibrium MD trajectory has a **uniform** empirical frame distribution — every frame carries
weight `1/N`. So fitting `softmax(bc·H_f + bh·O_f)` against the observed per-frame weights is
degenerate: the uniform target is matched only by `bc, bh → 0`. The signal is not in the per-frame
weights but in **how many frames land at a given `G`**, so the loss target must be the state
occupancies `n_k/N` of §3.1's bins/clusters.

This makes the binning step load-bearing rather than cosmetic. The only binning-free alternative is
to regress a per-frame kNN/KDE log-density estimate on `G_f`, which requires density estimation in
high dimensions and is strictly harder; bin/cluster instead.

### 3.2 JAX-ENT parity and deterministic fit

The experiment calls `BV_ForwardPass` to form per-frame, per-residue `z`, then performs the
residue-wise absolute difference and structural-bin averaging. An independent NumPy implementation
is the reference test. The production estimator is deterministic: a coarse direction scan followed
by bounded scalar refinement, with the coefficient norm profiled analytically. It therefore has no
learning-rate or optimiser-convergence ambiguity and does not optimise frame weights.

### 3.2b Constraints and regularisation

No regularisation is used. The simultaneous sign of `(bc,bh)` is unidentifiable after an absolute
value, so both coefficients are constrained non-negative to select the conventional physical
branch. Boundary solutions are reported rather than hidden. Collinearity is diagnosed through the
bootstrap spread and `bh/bc`; no ridge penalty is allowed to manufacture a narrow coefficient
distribution.

Regularisation becomes arguable only for **M8** in Stage 3, where `β·x^(−γ)` carries a genuine
scale/exponent degeneracy — and even there the first moves are bounding `γ` and standardising
features, not adding a penalty.

### 3.2c Replicates

Two different jobs; conflating them will understate the error bars.

- **Leave-one-replica-out (3 folds)** — ATLAS ships 3 independent replicas per system. Independent
  sampling of the same ensemble, so this is the honest generalisation test. **Primary.**
- **Within-replica block bootstrap (5 blocks)** — for the per-system CI on `(bc, bh)`. Frames within
  a replica are temporally correlated, so *random* frame splits leak and produce falsely tight CIs.
  Use **contiguous block** resampling with block length ≥ the autocorrelation time of Absolute-L1; estimate
  that time per system rather than assuming it.

Neither adds a sweep dimension: total Stage-2 fits ≈ systems × 3 folds × arms, with the bootstrap a
deterministic coefficient fit.

### 3.3 Readout

Distribution of fitted `(bc, bh)` over all systems: joint scatter with per-system block-bootstrap
ellipses, marginal histograms, and the literature `(0.35, 2.0)` marked.

Report the condition number of the `2×2` normal-equation matrix and `corr(H, O)` per system
alongside every fit. Where the system is ill-conditioned, quote the **ratio** `bh/bc` — well
determined along the principal axis — with its CI, and say explicitly that the individual
coefficients are not separately identified for that system, rather than presenting two unstable
numbers as if they were measurements. Leave-one-replica-out (§3.2c) gives the within-system
stability check.

### 3.4 Observed exploratory Stage 2 result (2026-08-27)

The user-authorized exploratory run completed all 65 systems and all 65,000 requested pooled
moving-block bootstrap draws. Fitted and default coefficients were evaluated on identical held-out
replica folds.

| readout | result |
|---|---:|
| median held-out rho, default `(0.35, 2.0)` | 0.2920 |
| median held-out rho, fitted | 0.2429 |
| median paired `rho_fitted-rho_default` | -0.0162 |
| one-sided 95% lower bound for median paired delta | -0.0342 |
| systems with positive paired delta rho | 35.4% |
| median PMF MAE, default | 23.04 kcal/mol |
| median PMF MAE, fitted | 0.246 kcal/mol |
| pooled boundary solutions | 69.2% |
| fitted model beats strongest compactness control | 55.4% |

The predeclared Stage 2 predictive criterion therefore fails: fitting does not improve held-out
ordering over the default BV direction. The large MAE reduction is not evidence of a better
ordering model; it is achieved by shrinking the coefficients by roughly three orders of magnitude,
with 23/65 pooled fits effectively dropping `bc` and 22/65 effectively dropping `bh`. The
residue-summed Absolute-L1 coordinate has far too large a thermodynamic scale under the literature
coefficients, while the data do not identify a stable replacement `bc:bh` direction. Report the
boundary frequency rather than interpreting the extremely broad ratio distribution as calibrated
BV chemistry.

This exploratory fit used the deterministic NumPy/SciPy reference estimator. `BV_ForwardPass` has
JAX parity tests, but the requested pure/vmapped JAX-ENT optimisation harness remains future work;
it must reproduce this reference before it replaces it for Stage 3.

## 4. Stage 3 — model variants

Only after Stage 2 has a stable `(bc, bh)` distribution. Each variant is a separate arm evaluated on
the same systems, same bins, same metric, and compared by held-out (leave-one-replica-out)
log-likelihood, not by in-sample fit.

| arm | form | code needed |
|---|---|---|
| **BV** | `bc·H + bh·O` | none — Stage 2 |
| **BV+B0** | `bc·H + bh·O + b0` | add `bv_b0` to `BV_Model_Parameters` / `BV_model_Config` / `BV_ForwardPass`; small and mechanical. Note the §3.1 identifiability caveat: `b0` is only meaningful in a joint cross-system fit. |
| **M8** | `ln PF = β_s·SASA^(−γ_s) + β_p·D^(−γ_p)` | **new features required** — see below |
| **M8+B0** | M8 `+ b0` | as above |

### 4.1 M8 provenance and numbering

M8 is from **Mohammadiarani, Shaw, Neubig & Vashisth, "Interpreting Hydrogen-Deuterium Exchange
Events in Proteins Using Atomistic Simulations: Case Studies on Regulators of G-protein Signaling
Proteins", *J. Phys. Chem. B* 2018** (PMC6430106). Use that paper's numbering:

- **M1–M6** — prior empirical models (M2/M3 are the BV form `ln PF = βc·Nc + βh·Nh`, i.e. the model
  this study already fits in Stage 2).
- **M7** — a **fractional-population** model, `PF = τ_C/τ_O`, the ratio of mean residence times in
  closed vs open states. **This is not an instantaneous structure→PF map at all** and does not fit
  the `ForwardPass` slot: it needs open/closed state assignment plus residence times along the
  trajectory. Do not implement it as a BV-style arm.
- **M8** — the authors' new **empirical** model, `ln PF = β_s·SASA^(−γ_s) + β_p·D^(−γ_p)`. This is
  the SASA/polar-distance arm, and the one this stage means.
- **M9** — the authors' new fractional-population model: same `τ_C/τ_O` form as M7, but with
  open/closed defined by SASA and polar-distance criteria.

**Note the exponent signs: both are inverse powers.** That is what makes the model physical — larger
SASA means more solvent exposure and *less* protection, larger `D` means no polar partner nearby and
*less* protection, so both terms must decrease in their descriptor. The aim section as originally
written had `SASA^γ`; it should be `SASA^(−γ)`. Fit `γ_s, γ_p > 0` and read the sign off the
exponent, not off `β`.

**Retrieve the fitted `β_s, γ_s, β_p, γ_p` from that paper's SI (Tables S2/S3) before implementing**
— they are not in the main text, and they set the initialisation and the sanity range. The main text
does give the M9 SASA thresholds for exchange competence (8.02 Å² CHARMM, 9.15 Å² AMBER), which is a
useful scale check on the SASA featuriser: ATLAS is CHARMM36m, so the 8.02 Å² figure is the relevant
one.

**M9 is a genuinely attractive future arm for this dataset** and worth recording even though it is
out of scope here: ATLAS gives 100 ns × 3 replicas, which is exactly what a residence-time model
needs, and the SASA featuriser built for M8 supplies its state-assignment criterion for free. But it
predicts a PF from *kinetics*, not from an ensemble average, so it does not slot into the
Stage-2 occupancy fit and would need its own comparison design. Out of scope — note it, don't build
it.

### 4.2 Contact-construction sensitivity arm

The primary analysis already uses `contact_mode="bradshaw_switch"`. As a sensitivity analysis, run
the hard-count mode and the historical truncated JAX-ENT mode (`contact_mode="hard"` and
`contact_mode="legacy_switch"`) from the same trajectories. Keep these arms separate: the legacy
`--switch` flag maps only to `legacy_switch` for backward compatibility and is not the Bradshaw
protocol.

### 4.3 M8 featurisation is the real work in this stage

Nothing SASA-related exists in `jaxent/src` (`grep -ri sasa jaxent/src` returns nothing). It needs:

- per-frame, per-residue **SASA at the amide-H probe position** — a new function alongside
  `jaxent/src/models/func/contacts.py`, e.g. Shrake–Rupley on the amide H with the protein as
  environment (`freesasa` or `MDAnalysis`/`mdtraj` backends). **Benchmark before committing**: BV
  contacts measured 0.0475 ms per residue-frame per type (§1.4), and SASA is typically far more
  expensive per frame; if the ratio is bad it, not the trajectory tier, sets the system budget for
  this stage. Falling back to a subset of the 111 for M8 alone is acceptable — say so explicitly
  rather than silently comparing arms on different system sets.
- per-frame, per-residue **distance to the nearest polar atom** `D` from the amide H, reusing the
  neighbour-search machinery and residue-ignore convention already in `contacts.py` so the exclusion
  window matches BV exactly. Record which elements count as polar (N/O, and whether S is included);
  the paper's definition should be taken from its methods rather than assumed.
- a new `Input_Features` dataclass carrying both (mirroring `BV_input_features`), a new
  `ForwardModel`/`ForwardPass` pair, and config/parameter objects, per the CLAUDE.md extension
  points.

The exponents `γ` make M8 non-linear, so this arm genuinely needs the optax path from §3.2 — which
is why §3.2's closed-form gate must pass first. Bound `γ ∈ (0, 4]` and initialise from the SI values
(or 1 if unavailable). Watch the `β·x^(−γ)` scale/exponent degeneracy flagged in §3.2b: standardise
the descriptors before fitting, and report the `(β, γ)` correlation per system.

## 5. Stage 4 — MISATO

Repeat Stages 1–3 on MISATO (protein–ligand MD). Deliberately deferred: MISATO adds ligand atoms to
the contact environment, and whether the ligand counts toward `heavy_contacts` is a modelling
decision (`mda_contact_environment`) that must be run **both ways** and reported. Do not start this
until the apo (ATLAS) answer is settled, or the ligand-inclusion choice becomes a free parameter
fitted against an unvalidated hypothesis.

## 6. Deliverables and layout

```
jaxent/examples/ATLAS_BV/
├── README.md
├── INSTRUCTIONS.md              # minimal runnable sequence (repo convention)
├── commands.sh                  # fetch → featurise → stage1 → stage2
├── config.yaml                  # systems, bc/bh defaults, RT, bins, seeds, bandwidths
├── data/  fetch_atlas.sh  select_systems.py  systems.csv  convergence_report.csv
├── outputs/<system>/<replica>/  features.npz  topology.json  frame_coords.parquet
├── analysis/stage1/             compute_frame_coordinates.py  fit_free_energy.py  plots.py
├── fitting/                     closed_form_bv_fit.py  optimise_bv_boltzmann.py
└── analysis/stage3/             arm comparison
```

Follow the repo convention of a `validate_run.sh` that re-derives the headline numbers from the
persisted outputs, as examples 1–3 do.

## 7. Guardrails

- **Pre-register §2.5 and §2.9 before running.** The failure mode of this experiment is reading a
  relationship that compactness alone would have produced as evidence that BV carries free-energy
  information.
- **The baseline comparison is a hard gate, not a diagnostic.** `G` must beat `H_f` alone, `Rg` and
  RMSD on held-out basin ordering (§2.5). `plans/hdx_effective_rate_variance_physics.md` records
  exactly this class of failure: a geometry claim that looked good on self-prediction and died
  against its control.
- **A control must actually change the statistic.** The originally proposed residue-shuffle null did
  not: Signed-L1, Abs-L1 and L2 are symmetric in the residue components, so permuting residues
  within a frame leaves all three numerically identical. Withdrawn and replaced (§2.5). Check any
  future control against this test before relying on it.
- **Define basins in structural space only** (§2.3b). Defining them in log-PF space makes the
  basin↔`G` comparison circular.
- **Run the basin census before building anything** (§2.3b). If most systems occupy one basin at
  100 ns, the basin-level design has no dynamic range and the plan needs rethinking.
- **Never fit and evaluate on the same coordinate.** Stage 2 fits `(bc, bh)` against occupancy; it
  cannot then be evaluated by occupancy agreement on the same frames. Use leave-one-replica-out.
- **`RT` at the simulation temperature.** Verified as `ref_t = 300 K` across ATLAS (§1.1), so
  `BV_model_Config.temperature`'s default is correct here — but assert it against the `.mdp` in code
  rather than relying on the default, since MISATO (§5) will not match.
- **Freeze the residue set per system** and record `n_res` with every `α`; `G` is extensive.
- Report systems excluded by the convergence gate alongside the results, with counts.
