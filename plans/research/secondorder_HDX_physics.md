# Physics-Grounded Ingredients for an HDX-Driven Loss-Function / Observable-Design Programme

## TL;DR
- **The diagonal-vs-covariance asymmetry your team observes is a fundamental identifiability property, not an implementation bug**: peptide centroid uptake is a *linear (first-moment) functional* of per-residue mean deuteration, so it identifies marginal per-residue rate/protection statistics (the diagonal D) but is provably *blind* to pairwise cross-residue correlations (the geometry R). Correlations enter the physics only at the *second moment of the isotopic envelope* — an observable currently discarded by using centroids.
- **The single strongest, most under-exploited physical ingredient in the literature is the isotopic envelope shape (width/skew/bimodality)**, which Poisson-binomial/EX1–EX2 theory shows is exactly where within-molecule correlated exchange (positive covariance) manifests as super-binomial broadening or bimodality [5,8]. Englander's HDsite [9] already demonstrated the envelope carries information the centroid loses; no existing method uses it to infer a residue-residue *covariance matrix*.
- **Correlated exchange is largely an ensemble/population-dependent property, not a purely structural one**: MD evidence [3,4] shows amide open-states are short-lived (~100 ps) and *nearly uncorrelated between residues even when adjacent residues unfold cooperatively*. This is the physical reason static structural/graph geometries lose to a spectrum-preserving, locality-destroying shuffle — the "true" R is dominated by cooperative-unit (foldon/EX1) physics [10,11] that is weakly encoded in a single static structure and only appears in higher-moment observables.

## Key Findings

1. **Coordinate of heterogeneity.** The field overwhelmingly treats **log protection factor z = ln PF (equivalently ΔG_op = RT ln PF)** as the natural, additive, physically-interpretable coordinate — the Best-Vendruscolo model is *linear in ln PF* [1], and PyHDX explicitly reports residue-level ΔG [2]. But the *observable* (deuterium uptake) is linear in the **rate k = k_int·PF⁻¹**, not in z. Because uptake = 1−exp(−k·t) is nonlinear in k and sigmoidal in log-space, ensemble-averaging over a heterogeneous PF distribution and then computing uptake is *not* equal to computing uptake at the mean ln PF — a genuine Jensen's-inequality gap. That gap is the physical content the diagonal variance D captures.

2. **Physical meaning of D (per-residue rate variance).** Per-residue PF heterogeneity across a conformational ensemble corresponds to the frequency/amplitude of local "breathing"/sub-global unfolding openings. Persson & Halle [3] show the open-state residence time is ~100 ps and roughly constant across amides, so the large variation in measured HX rate must be attributed to opening *frequency*. So D maps onto opening-*frequency* (opening-probability) heterogeneity, which is exactly what an ensemble reweighting scheme can move. D correlates loosely with RMSF/B-factor patterns but is not reducible to them (H-bonding dominates) [3,4].

3. **Identifiability theory (the core result).** For a peptide of N residues with per-residue deuteration probabilities p_i(t)=1−exp(−k_i t), the isotopic envelope is a **Poisson-binomial** distribution with **mean = Σp_i** (the centroid) and **variance = Σp_i(1−p_i) + 2Σ_{i<j}Cov(X_i,X_j)** [5]. The centroid contains *no covariance terms whatsoever*; covariance appears *only* in the envelope's second moment. This is the rigorous reason D is recoverable and R is not from centroid data — a moment-identifiability fact directly analogous to results in mixture-of-linear-regressions [6] and factor analysis [7], where first/second moments of linear (additive) observations identify only mean structure, while pairwise cross-moments require product/higher-order observations.

4. **The envelope is the missing observable.** EX1/EX2 theory [8] establishes: uncorrelated exchange (EX2) → binomial envelope; correlated/cooperative exchange (EX1) → super-binomial broadening or bimodality. Englander's HDsite [9] explicitly uses envelope shape to break the peptide→residue degeneracy that centroids cannot, since centroids provide only a single-parameter population average and lose the more detailed distribution information in the multiparameter envelope shape. No current method inverts the envelope for a residue-residue covariance matrix — a clear field gap.

5. **Correlated exchange is population-dependent.** The best MD evidence [3,4] is that amide open-state transitions are *nearly uncorrelated between residues*, even for residues in the same cooperative unfolding unit — open states of different amides show little or no temporal correlation even when adjacent residues unfold cooperatively [3]. Cooperativity manifests thermodynamically (shared ΔG_op / same foldon) rather than as temporally correlated single-amide openings. The "true" R being chased is therefore closer to a *foldon/cooperative-unit block structure* [10,11] than to a smooth structural covariance — and it is a property of the ensemble/energy landscape, not of any single static structure. This explains why static structural graph geometries lose to a spectrum-preserving shuffle: they encode the wrong kind of correlation, and the centroid can't adjudicate anyway.

6. **Existing methods and the gap.** ExPfact [12], PyHDX [2], HDXer [13], Saltzberg's BayesianHDX [14], ReX [15], and HDsite [9] all deconvolve peptide→residue; most report point estimates or marginal uncertainty. **None model or report a residue-residue rate-covariance matrix.** ReX explicitly models sequence-dimension correlation via a change-point prior but treats it as a smoothing device, not as recoverable physical covariance [15].

7. **EX1/EX2/EXX regime.** Cooperative units (foldons) that would produce nontrivial rate covariance are *precisely* the EX1/EXX signal — and that signal lives in envelope bimodality, not the centroid. Native-state HX [10,11] detects cooperative units through pH-dependence and EX1 plateaus. In pure EX2 (the assumed regime) the covariance is *thermodynamically real but kinetically invisible in the mean*: it re-emerges only in the envelope second moment.

## Details

### Thread 1 — Coordinate of variance / identifiability (k vs ln PF vs ln k)

The literature is consistent that the **physically natural, additive coordinate for protection heterogeneity is z = ln PF = ΔG_op/RT**. The Best-Vendruscolo phenomenological model is explicitly *linear in ln PF*: ln PF_i = β_C⟨N_C⟩_i + β_H⟨N_H⟩_i, with the two terms interpreted as free energies of residue burial and hydrogen bonding respectively [1]. PyHDX is built entirely around extracting residue-level **ΔG = −RT ln(k_close/k_open)** as a "universal quantity" for cross-protein comparison [2]. So the standard field coordinate for the *latent* variable is log-space.

However, the **observable** — fractional deuterium uptake — is linear in the *rate* k, not in z. In EX2, a single residue's deuteration is d_i(t) = 1 − exp(−k_obs,i·t) with k_obs,i = k_int,i·PF_i⁻¹ = k_int,i·exp(−z_i). The peptide fractional uptake is the mean over residues, D_pep(t) = (1/N)Σ_i [1 − exp(−k_int,i e^{−z_i} t)]. Because 1−exp(−k t) is concave in k and sigmoidal in z, **the ensemble average of the uptake is not equal to the uptake evaluated at the ensemble-mean z**: E_f[1−e^{−k t}] ≠ 1−e^{−E_f[k] t} ≠ 1−e^{−k(E_f[z]) t}. This is a genuine Jensen's-inequality-type bias, with two distinct nesting levels (average over the conformational ensemble f, and average over residues in a peptide).

Field practice is heterogeneous and matters here:
- **HDXer / Best-Vendruscolo ensemble reweighting** [13] computes PF as an **ensemble average of the (N_C, N_H) contacts** — i.e., it averages the *arguments* of ln PF, producing a single mean ln PF per residue, then a single-exponential uptake. This is the "average log-PF then exponentiate" branch and will systematically mis-estimate uptake when the PF distribution is broad, because it collapses the distribution before the nonlinear step.
- **ExPfact / PyHDX / HDsite** [2,9,12] fit uptake curves as **sums of single-exponentials with per-residue k** — i.e., they operate in rate space and, in the multi-conformer limit, correspond to "average the rate directly."
- The distinction is exactly the D-carrying quantity: the difference between these two averaging orders is, to leading order, controlled by Var_f(k_i) (or Var_f(z_i)) — a second-order Taylor term. **This is the physical reason multi-exponential curvature in centroid curves encodes D**: deviation from single-exponential behaviour (curvature of the uptake curve on a log-time axis) encodes the marginal variance of the rate.

**Confidence:** High on the coordinate convention (ln PF/ΔG) and on the linearity of BV in ln PF. High on the qualitative Jensen gap. Medium on which specific averaging each tool uses in edge cases — often buried in implementation and worth a direct code/paper check for the exact tools being benchmarked against.

**Relevance to D-vs-R:** Directly explains D recoverability: D = Var_f(k_i) is the leading correction term distinguishing the two averaging orders, and it shows up as multi-exponential curvature in the peptide uptake — a first-moment-of-each-residue quantity that survives peptide summation.

### Thread 2 — Physical meaning of per-residue rate variance (D)

Per-residue PF variance across a conformational ensemble is physically the **heterogeneity in local opening free energy / opening frequency**. Persson & Halle's analysis of an ultralong BPTI trajectory [3] established a crucial mechanistic fact: the exchange-competent open (O) state is a locally distorted conformation with two waters coordinated to the N–H, its mean residence time is ~100 ps and roughly uniform across amides, so the large variation in measured HX rate must be attributed to opening *frequency*. The entire dynamic range of protection factors — and hence D — lives in *how often* each amide opens, exactly the quantity an ensemble-population reweighting can adjust. D is therefore a well-posed reweighting target.

Connection to RMSF/B-factors is real but loose: HX protection correlates qualitatively with flexibility descriptors, but the MD literature [3,4] shows **hydrogen bonding is the dominant determinant** and that other structural descriptors correlate poorly with single-amide rates. So D is not simply RMSF variance; it is opening-probability variance, dominated by H-bond making/breaking.

Detectability of heterogeneity via non-single-exponentiality: a peptide's uptake is a **sum of single-exponentials with different k_i**; when the k_i span a range, the summed curve deviates from a single exponential (it is "stretched"). This is the same mathematics as stretched-exponential (Kohlrausch-Williams-Watts) relaxation, where a stretching exponent β<1 quantifies a distribution of rates. ReX [15] explicitly parameterizes per-residue uptake as a Weibull/stretched-exponential–vs–logistic mixture and finds a strong data-driven preference for stretched (Weibull) kinetics in most peptides — direct evidence that rate heterogeneity is detectable in real HDX-MS uptake curves.

**Confidence:** High that D = opening-frequency heterogeneity and that H-bonding dominates. High on stretched-exponential detectability. Medium on quantitative detectability floors (see Thread 3).

**Relevance:** Confirms D is physically meaningful and reweightable, and identifies the *shape* (curvature/stretching) of the centroid uptake curve as its experimental fingerprint.

### Thread 3 — Identifiability theory for peptide-summed, time-sampled kinetic data

This is the mathematical heart of the asymmetry. Model each residue's deuteration as a Bernoulli variable X_i(t) with P(X_i=1)=p_i(t)=1−exp(−k_i t). The number of deuterons on a peptide is S=ΣX_i, whose distribution is the **Poisson-binomial** [5]:

- **Mean (centroid):** E[S] = Σ_i p_i
- **Variance (envelope width):** Var(S) = Σ_i p_i(1−p_i) + 2 Σ_{i<j} Cov(X_i, X_j)

Two facts follow immediately and decisively:

(a) **The centroid depends only on the marginals {p_i}.** No pairwise covariance term appears in E[S]. Therefore *any* residue-residue correlation structure R is completely invisible to centroid data — not merely hard to estimate, but formally non-identifiable. Given a set of overlapping peptides sampled at multiple times, the design matrix (which residues belong to which peptide) lets you invert the linear system for the marginal p_i(t) trajectories, hence for the marginal rate distribution moments (mean and, via multi-exponential curvature, variance D). But because the map from residues to centroid is *linear and additive*, it carries zero cross-moment information.

(b) **Covariance enters only at the second moment of the envelope.** Positive correlation between sites (cooperative/EX1 exchange) makes Var(S) exceed the independent Poisson-binomial value Σp_i(1−p_i) — the "correlated binomial" result [16] that positive correlation makes the distribution "more spread out." So R is identifiable *in principle* from the envelope width, but not at all from the centroid.

This mirrors general statistical-identifiability results:
- **Mixture-of-linear-regressions** [6]: first and second moments of purely linear observations recover only the first-moment parameter; identifying further latent structure requires higher-order (product/tensor) moments.
- **Method-of-moments for mixtures/HMMs** [17]: second-order moments are generally insufficient to identify latent structure; third-order (product) moments and multiple joint "views" are required.
- **Factor analysis / covariance-structure modelling** [7]: off-diagonal cross-dependence (ΛΛᵀ) is recoverable *only* from the observed covariance matrix of the data, never from the mean vector. Collapsing observations to a scalar linear sum (the centroid) destroys the cross-terms.

There is also a **Laplace-inversion / ill-posedness** layer even for the diagonal: recovering a full continuous rate distribution from a sum of exponentials is the classic ill-posed inverse-Laplace problem (NMR relaxometry, dielectric spectroscopy, rheology) [18]. Achievable resolution is severely limited and noise-sensitive. So while the *low-order moments* of the rate distribution (mean, variance D) are robustly identifiable, the *full shape* of the per-residue rate distribution is not.

**Confidence:** Very high — the decisive, rigorously-grounded finding. The Poisson-binomial mean/variance decomposition is textbook; the HDX-specific instantiation is in [8,9]; the statistical analogues are well-established.

**Relevance:** This *is* the answer to "why diagonal, not off-diagonal." Centroids are first-moment/linear → identify marginals (D). Covariance (R) requires the envelope second moment. Any structural-geometry candidate that "wins" on centroid-based self-prediction is exploiting marginal information (or spurious smoothing), which is why a spectrum-preserving shuffle — which preserves the marginal eigenvalue spectrum but destroys locality — can beat it: the centroid data simply cannot distinguish them.

### Thread 4 — Isotope envelope width/shape as a separate observable

The HDX-MS literature has a mature toolkit for the envelope as an EX1/correlation observable, but has used it almost exclusively for *detection and binary classification*, not for covariance inference:

- **EX1 detection by peak-width analysis** [8]: bimodal/broadened envelopes are the hallmark of correlated cooperative exchange; binomial envelopes indicate uncorrelated EX2. Peak width is explicitly a second-moment quantity.
- **Deconvolution software:** HX-Express (and v2/v3, binomial and multimodal fitting with statistical tests) [19], HDExaminer, Deuteros, MEMHDX, HDXBoxeR. These fit one, two, or three binomial/Gaussian subpopulations to detect bimodality; HX-Express v3 adds rigorous statistical tests for multimodality.
- **HDsite** [9] is the key precedent: it fits the *full envelope* of overlapping peptides to extract residue-resolved D-occupancies, explicitly because centroids provide only a single-parameter population average and lose the more detailed distribution information contained in the multiparameter envelope shape. It resolves "switchable" residues (variable D occupancy) that centroids cannot.

Critically: **no published method uses the envelope to infer a residue-residue rate-covariance matrix R.** The envelope has been used to (i) detect EX1, (ii) count subpopulations, and (iii) improve marginal residue resolution — but the second-moment information about *which residues co-exchange* has not been turned into a covariance estimator. This is the single clearest opportunity: the envelope variance/covariance is the *only* observable even in-principle sensitive to R.

**Confidence:** High that the envelope carries second-moment information and that EX1 detection is mature. High that covariance inference from the envelope is an unfilled gap.

**Relevance:** This is where R becomes identifiable at all. Recovering R requires adding an envelope-derived second-moment observable to the loss; the centroid alone is provably insufficient.

### Thread 5 — Population-dependent vs population-free structural correlates of R

The deprioritization of ANM/GNM structural covariance is *physically well-justified*, and the reason generalizes to most static-structure geometries. The key evidence:

- **Persson & Halle** [3]: in BPTI, open-state transitions of different amides show little or no temporal correlation, even if adjacent residues unfold cooperatively. Opening is local, transient (~100 ps), and asynchronous.
- **RGS-protein MD study** [4]: open-state correlations are short-range and largely uncorrelated between residue pairs.

Together these imply the physically-relevant correlation is **thermodynamic block structure** (residues sharing the same cooperative unfolding unit / foldon, hence the same ΔG_op), *not* the smooth spatial covariance that ANM/GNM or dynamic cross-correlation maps produce. Foldon/native-state-HX physics establishes this concretely: cytochrome c is composed of five cooperative folding units (foldons) that unfold all-or-none, arranged on an energy ladder of partially unfolded forms differing by one foldon [10,11]. So R should look like near-block-diagonal structure organized by cooperative units — and those units are defined by the *energy landscape / ensemble*, not by a single structure's contact topology.

Structure-derived (population-free) candidates that exist in the literature but with weak justification for HX covariance specifically:
- Contact-map graph distances / hydrogen-bond network community structure and percolation.
- Coevolution / direct-coupling analysis (DCA) — captures functional/evolutionary coupling, not necessarily co-fluctuation of protection.
- Allosteric community detection on MD correlations — but these are *derived from an MD population*, so they are population-dependent, not population-free.

The theoretical argument: **correlated protection-factor fluctuation is an ensemble/population property**, because it depends on the joint distribution of opening events, set by the free-energy landscape (which foldons co-open), not by instantaneous geometry. A purely structural prior will therefore be systematically wrong for multi-domain and allosteric systems. There is no strong literature argument that HX covariance is a population-independent structural invariant; the weight of evidence is the opposite.

**Confidence:** High that correlated single-amide opening is weak/short-range in MD. High that cooperative-unit (foldon) block structure is the physically-relevant R. Medium-high that this makes static structural priors unreliable for R.

**Relevance:** Explains *why* structural-geometry R candidates lose to a locality-destroying shuffle — the true R is foldon/thermodynamic block structure that static geometry encodes poorly, and the centroid can't adjudicate anyway. Any R prior should be population-aware (foldon/cooperative-unit-based) rather than static-graph-based.

### Thread 6 — Existing methods that infer per-residue rate distributions

| Method | Peptide→residue deconvolution | Reports per-residue variance/uncertainty? | Models residue-residue covariance? |
|---|---|---|---|
| **ExPfact** [12] | Multi-exponential fit of overlapping peptides; finds multiple degenerate solutions | Yes — reports the spread/degeneracy of compatible PF sets as an uncertainty proxy | No |
| **PyHDX** [2] | Fused-LASSO along sequence; extracts ΔG | Bootstrap/covariance of fit; marginal | No (sequence smoothing only) |
| **HDXer** [13] | Ensemble reweighting; BV forward model | Uncertainty via subsampling replicates | No |
| **BayesianHDX** [14] | Bayesian residue-resolved | Yes — full posterior credible intervals per residue | No |
| **HDsite** [9] | Full envelope fitting of overlapping peptides | Resolves "switchable" residues; envelope-based | Implicitly uses envelope but does not output covariance |
| **ReX** [15] | Bayesian non-parametric change-point model | Yes — per-residue credible intervals + resolution metrics | Models *sequence-dimension* correlation as a prior, not as recovered physical covariance |
| **MEMHDX / Deuteros / HDXBoxeR** | Peptide-level statistical testing | Peptide-level significance | No |

**The field-wide gap is unambiguous:** every method reports (at best) *marginal* per-residue uncertainty; none infers or reports a residue-residue rate-covariance matrix. ReX is the closest in spirit (it exploits sequence-neighbour correlation) but treats correlation as a smoothing prior, not a recoverable observable. This confirms the covariance-recovery goal is genuinely novel — and that the reason it is unfilled is the Thread-3 identifiability barrier, which no centroid-based method can overcome.

**Confidence:** High on the table's qualitative content; medium on some fine details of each tool's uncertainty model (worth verifying per tool if cited precisely).

### Thread 7 — EX1/EX2/EXX regime and cooperative units

- **EX2** (k_cl ≫ k_int; native folded proteins, the typical assumed regime): each amide exchanges via many independent transient openings; k_obs = k_int·P⁻¹; envelope is binomial; cooperativity is *thermodynamically encoded* (shared ΔG_op) but *kinetically averaged away in the mean*. Covariance survives only in the envelope second moment.
- **EX1** (k_cl ≪ k_int; high pH, low stability, cooperative unfolding): a whole cooperative unit opens and all its amides label together before reclosing → bimodal envelope; k_obs = k_op. This is where residue-residue *correlation is maximal and directly visible*.
- **EXX / mixed:** intermediate; two envelopes persist over the time course.

Englander's native-state HX programme [10,11] established that proteins are built of **foldons — cooperative units that unfold all-or-none** — detected via EX1 signatures (pH-dependence, EX1 plateaus, bimodal MS).

Implication: in pure EX2, the cooperative-unit covariance sought is real but produces *no* first-moment signature — it re-emerges only in the envelope. If any peptides show even partial EX1/EXX (bimodality or super-binomial width), those are the *only* places R is directly measurable and should be weighted heavily. Forcing an R geometry from EX2 centroids is fighting an identifiability wall.

**Confidence:** High — textbook Englander HX, directly corroborated by the envelope literature.

## Recommendations (staged)

**Stage 0 — Reframe the objective (immediate).** Accept the Thread-3 result as a hard constraint: **R is not identifiable from centroid uptake, full stop.** Stop evaluating structural R candidates against centroid-based held-out self-prediction — that metric is provably insensitive to R and will reward spectrum-matching/smoothing artefacts (hence the shuffle winning). Any honest R recovery must use a second-moment observable.

**Stage 1 — Add the isotopic envelope as an observable (highest priority).** Extend the forward model from centroid to the **Poisson-binomial envelope** (mean + variance, at minimum) per peptide per timepoint. The envelope variance term Σp_i(1−p_i) + 2Σ_{i<j}Cov(X_i,X_j) is the *only* observable that couples to R. Concretely: add a loss term matching predicted vs measured envelope width (2nd moment), and where data permit, the full envelope shape (bimodality). **Benchmark:** if a physical R geometry now beats the shuffled control on *envelope-width* held-out prediction (not centroid), that is a real signal. **Threshold that would change the recommendation:** if envelope-width prediction still cannot separate physical R from the shuffle even with clean synthetic envelope data, then R is unidentifiable even in principle for that peptide map, and effort should shift to experimental design (Stage 3).

**Stage 2 — Make the R prior population-aware, not static-structural (medium priority).** Replace static graph/ANM-style covariance priors with a **foldon/cooperative-unit block prior** derived from the *candidate ensemble itself* (e.g., cluster residues by co-variation of opening probability across the trajectory, or by shared ΔG_op). This aligns the prior with the physically-correct block structure (Thread 5) and preserves multi-domain transferability better than a fixed structural graph. Use DCA/coevolution or community detection only as weak, secondary tie-breakers, not primary priors.

**Stage 3 — Exploit EX1/EXX peptides and experimental design (medium priority).** Identify any peptides showing bimodality or super-binomial width; these are the direct R windows. Where possible, recommend/prioritize acquisition conditions (higher pH, pulse-labelling) that push borderline residues toward EX1 to expose correlation. Millisecond-HDX and improved envelope fitting increase the second-moment information.

**Stage 4 — Handle the Jensen/coordinate bias explicitly (low-effort, do early).** When averaging the BV model over the ensemble, average in **rate space** (or carry both the mean and variance of z through the nonlinearity) rather than collapsing to mean ln PF before exponentiating. This removes a known systematic bias and makes D estimation cleaner. Essentially free; do it regardless.

## Candidate physics ingredients (bullet-level; not fully-designed loss terms)

- **Isotopic envelope second moment (peptide width).** The one observable that couples to R (Poisson-binomial variance = independent term + 2·Σcovariance). *Building block for the primary covariance-recovery loss term.*
- **Envelope bimodality / EX1 indicator.** Direct fingerprint of a correlated cooperative unit; a strong per-peptide flag for where R is measurable.
- **Multi-exponential / stretched (KWW/Weibull) curvature of the centroid uptake curve.** Encodes the marginal rate variance D; the fingerprint already exploited. *Building block for a diagonal-variance regularizer.*
- **Opening-frequency (not residence-time) heterogeneity** [3]: model D as variance in opening probability, dominated by H-bond making/breaking — a physical prior on what D *is*.
- **Foldon / cooperative-unit block structure of R** [10,11]: the physically-correct shape for an R prior — near-block-diagonal by shared ΔG_op, derived from the ensemble, not from static geometry.
- **ΔG_op / ln PF as the latent coordinate** with rate-space averaging in the forward model: removes Jensen bias and gives an additive, interpretable latent.
- **pH-dependence / EX1–EX2 transition** as an auxiliary observable separating k_op (correlation-bearing) from PF (marginal) — a design-level ingredient if multi-pH data exist.
- **Inverse-Laplace ill-posedness bound** [18]: a principled cap on how much rate-distribution *shape* (beyond mean and variance) can be claimed — useful as a regularization-strength/identifiability diagnostic rather than a loss term.
- **Weak secondary structural correlates** (H-bond network community structure; DCA): usable only as low-weight tie-breakers, explicitly *not* primary R priors, given the population-dependence argument.

## Caveats

- **The central identifiability claim is regime- and observable-specific.** It holds cleanly in pure EX2 with centroid-only data. In EX1/EXX, or with full envelope data, R becomes partially identifiable — so the "impossibility" is conditional on the observable, not absolute.
- **"Little/no temporal correlation" of amide openings** [3,4] is drawn from a small number of MD studies (notably BPTI and RGS proteins) and specific force fields; it is strong mechanistic evidence but not universal proof. Multi-domain and allosteric proteins may show more correlation — worth testing directly against your own trajectories.
- **Which averaging each downstream tool uses** (rate-space vs log-space) is often an undocumented implementation detail; verify for the exact tools/benchmarks being compared against before attributing bias.
- **The envelope observable is experimentally demanding:** back-exchange, overlapping peptides, and limited resolving power degrade envelope shape. The second-moment signal is noisier than the centroid, so covariance recovery will always be lower-SNR than diagonal recovery even when identifiable.
- **Foldon/cooperative-unit block structure** is best-established for a handful of proteins (cytochrome c, RNase H, others in the Englander canon); its universality across all target systems is an assumption, not a certainty.
- Some sources are preprints (bioRxiv) or reviews; the load-bearing quantitative claims (Poisson-binomial moments, envelope-based EX1 detection, MD opening-frequency mechanism, Best-Vendruscolo linearity in ln PF) are from peer-reviewed primary literature.

## Bibliography

[1] Best, R.B. & Vendruscolo, M. (2006). Structural interpretation of hydrogen exchange protection factors in proteins: characterization of the native state fluctuations of CI2. *Structure*, 14(1), 97–106.

[2] Smit, J.H., et al. (PyHDX). Fast analysis of hydrogen deuterium exchange mass spectrometry data with PyHDX. Related preprint/publication describing fused-LASSO ΔG extraction from HDX-MS uptake data.

[3] Persson, F. & Halle, B. (2015). How amide hydrogens exchange in native proteins. *PNAS*, 112(33), 10383–10388.

[4] MD-based interpretation of HDX events using atomistic simulations, case studies on Regulators of G-protein Signaling (RGS) proteins (PMC article on open-state correlation and force-field dependence).

[5] Poisson-binomial distribution — standard probability result for sums of independent (or correlated) Bernoulli variables; applied to isotope envelope mean/variance decomposition in HDX-MS.

[6] Chaganty, N. & Liang, P. (2013). Spectral experts for estimating mixtures of linear regressions. *ICML*.

[7] Jöreskog, K.G. (and related factor-analysis / covariance-structure literature, e.g. Browne, M.W.) — identifiability of factor loadings (ΛΛᵀ) from observed covariance matrices vs mean vectors.

[8] Weis, D.D., Engen, J.R., & Kass, I.J. (2006). Semi-automated data processing of hydrogen exchange mass spectra using HX-Express. *J Am Soc Mass Spectrom*, 17(11), 1498–1509 (bimodal/EX1 vs binomial/EX2 envelope theory).

[9] Kan, Z.-Y., Walters, B.T., Mayne, L., & Englander, S.W. (2013). Protein hydrogen exchange at residue resolution by full-envelope HDX-MS (HDsite). *PNAS*, 110(41), 16438–16443.

[10] Hu, W., Kan, Z.-Y., Mayne, L., & Englander, S.W. (2016). Cytochrome c folds through foldon-dependent native-like intermediates. *PNAS*, 113(14), 3809–3814.

[11] Englander, S.W., Mayne, L., et al. — native-state hydrogen exchange and foldon theory (broader review/series, including Bai, Y., Sosnick, T.R., Mayne, L., & Englander, S.W. (1995). Protein folding intermediates: native-state hydrogen exchange. *Science*, 269, 192–197; Maity, H., et al. (2005). Protein folding: the stepwise assembly of foldon units. *PNAS*, 102(13), 4741–4746).

[12] Skinner, J.J. & Paci, E. (ExPfact) — extraction of protection factors from sparse HDX-MS data via multi-exponential deconvolution of overlapping peptides.

[13] Bradshaw, R.T., Marinelli, F., Faraldo-Gómez, J.D., & Forrest, L.R. (2020). Interpretation of HDX-MS data for protein conformational states via ensemble reweighting (HDXer). *Biophys J* / related methods papers.

[14] Saltzberg, D.J., et al. (BayesianHDX) — Bayesian residue-resolved protection factor inference with posterior credible intervals.

[15] Crook, O.M., et al. (ReX). Inferring residue-level hydrogen-deuterium exchange with ReX. *Communications Chemistry* (2025), https://www.nature.com/articles/s42004-025-01719-4

[16] Altham, P.M.E. (1978). Two generalizations of the binomial distribution. *J R Stat Soc Ser C (Applied Statistics)*, 27(2), 162–167 (correlated-binomial variance inflation result).

[17] Anandkumar, A., Hsu, D., & Kakade, S.M. (2012). A method of moments for mixture models and hidden Markov models. *COLT*.

[18] Parker, R.L. & Song, Y.-Q., and related inverse-Laplace-transform ill-posedness literature (NMR relaxometry, dielectric spectroscopy, rheology stretched-exponential fitting).

[19] Guttman, M., Tuttle, K., et al. HX-Express v3: rigorous statistical analysis of multimodal HDX-MS spectra. *J Am Soc Mass Spectrom* (2025).

*Note: several entries above (2, 4, 6, 7, 12, 13, 14, 18) are cited by method/finding rather than exact page-level bibliographic detail, since they were consulted via search snippets rather than full-text retrieval — verify exact citation details against the primary source before use in a manuscript.*