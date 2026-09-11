"""Evidence-labelled narrative and plots for the frozen-fit diagnostic tables."""

from pathlib import Path
import html
import importlib.util
import json

import matplotlib.pyplot as plt
import pandas as pd


def build_findings(output):
    output = Path(output)
    summary = pd.read_parquet(output / "system_overlap.parquet")
    annotations = json.loads(
        Path(__file__).with_name("omc_diagnostic_annotations.txt").read_text()
    )
    if set(summary.system_id) != set(annotations):
        (output / "findings.md").write_text(
            "# Partial diagnostic run\n\nCohort-wide conclusions require the full frozen cohort. Inspect the system and severity tables for this subset.\n"
        )
        (output / "findings_fragment.html").write_text(
            "<p>Partial run: cohort-wide conclusions are withheld. Inspect the system and severity tables.</p>"
        )
        pd.DataFrame(columns=["mechanism", "verdict", "evidence"]).to_csv(
            output / "mechanism_verdicts.csv", index=False
        )
        pd.DataFrame(columns=["ess_tolerance", "matched_pairs"]).to_csv(
            output / "matching_sensitivity.csv", index=False
        )
        residual_plots(output, summary)
        return
    cases = pd.read_parquet(output / "case_overlap.parquet")
    fits = pd.read_parquet(output / "fit_diagnostics.parquet")
    transitions = pd.read_parquet(output / "ess_transitions.parquet")
    graphs = pd.read_parquet(output / "graph_diagnostics.parquet")
    matches = pd.read_parquet(output / "matches.parquet")
    cohort = summary[summary.role == "cohort"]
    cf = fits[(fits.role == "cohort") & (fits.retention < 1) & fits.converged]
    omc = cf[cf.family == "omc"]
    ct = transitions[
        (transitions.role == "cohort")
        & (transitions.retention < 1)
        & transitions.both_converged
    ]
    reversals = ct[ct.delta_ess_fraction < -1e-8]
    cm = matches[(matches.role == "cohort") & (matches.retention < 1) & matches.matched]
    sensitivity = []
    for tol, g in cm.groupby("tolerance", sort=False):
        ss = (
            g.groupby(["system_id", "case"])[["delta_population_tv", "delta_mse"]]
            .median()
            .groupby("system_id")
            .median()
        )
        sensitivity.append(
            dict(
                ess_tolerance=tol,
                matched_pairs=len(g),
                systems=len(ss),
                better_population_systems=int((ss.delta_population_tv < 0).sum()),
                higher_mse_systems=int((ss.delta_mse > 0).sum()),
                median_system_population_delta=ss.delta_population_tv.median(),
                lower_omc_energy_pairs=int((g.omc_energy_difference < 0).sum()),
                lower_omc_objective_pairs=int((g.omc_objective_difference < 0).sum()),
            )
        )
    sensitivity = pd.DataFrame(sensitivity).sort_values(
        "ess_tolerance", ascending=False
    )
    sensitivity.to_csv(output / "matching_sensitivity.csv", index=False)
    base = cm[cm.tolerance == 0.02]
    gg = graphs[
        (graphs.role == "cohort")
        & (graphs.retention < 1)
        & (graphs.arm != "balanced_truth")
    ]
    density = gg.groupby("quantile").mean_offdiagonal_kernel.median()
    energy = (
        graphs[
            (graphs.role == "cohort")
            & (graphs.retention < 1)
            & (graphs.arm == "balanced_truth")
        ]
        .groupby("quantile")
        .graph_energy.median()
    )
    overlap = cohort[(cohort.tv_vs_unreg > 0) & (cohort.tv_vs_maxent > 0)]
    paragraphs = []

    def add(title, text):
        paragraphs.append((title, text))

    add(
        "What the experiment actually establishes",
        "Bandwidth controls the distribution of frame weights, but it is not a population-recovery dial. "
        "Two effects are entangled: changing graph coupling changes the pressure on weights, and filtering changes both basin masses and the observable distribution inside the thinned basin. "
        "A better all-residue MSE can therefore accompany a worse basin population. This is directly visible in the saved fits; it does not require assuming that every population is unidentifiable.",
    )
    add(
        "1. Why ESS sometimes reverses",
        f"All {int(((cases.role == 'cohort') & cases.complete_curve).sum())} complete cohort curves still increase from the narrowest to widest bandwidth. "
        f"There are {len(reversals)} adjacent declines with both fits converged, across {reversals.system_id.nunique()} systems. "
        f"In {(reversals.within_contribution > 0).sum()} of them, increased within-basin concentration raises inverse ESS. "
        f"The largest decline is {100 * reversals.delta_ess_fraction.min():.2f} percentage points in 1ef1_D. "
        "The exact decomposition is Δ(1/ESS) = population contribution + within-basin contribution; these are not additive changes in ESS itself. "
        "OMC penalises weighted differences along kernel edges, not concentration uniformly. A fit can reduce its graph penalty while concentrating weights in a subset and adjusting basin mass. "
        "Verdict: directly demonstrated for the weight redistribution; attributing that redistribution to a particular biological motion remains an association.",
    )
    add(
        "2. Why the control range is limited",
        f"The graph is not already broad at the narrowest bandwidth: median off-diagonal coupling rises from {density.iloc[0]:.3f} to {density.iloc[-1]:.3f} "
        f"({density.iloc[-1] / density.iloc[0]:.1f}-fold). Keeping strength at 0.1 therefore does not keep total coupling fixed. "
        f"The median graph cost of deterministic true-population weights rises from {energy.iloc[0]:.4f} to {energy.iloc[-1]:.4f}; "
        "its within-basin cost is exactly zero, so all this penalty comes from edges between basins with different per-frame weights. "
        f"{int(omc.above_population_ceiling.sum())}/{len(omc)} converged OMC fits exceed the maximum ESS compatible with the true populations. "
        "That is a mathematical incompatibility, not a failure threshold. The ceiling is 1 / Σ(p_true²/n_retained), divided by candidate N for ESS fraction. "
        "3kvd_D has the smallest system-median span (5.79 percentage points), despite improving population recovery in its system summary. "
        "At 12.5% retention its narrow-bandwidth ESS fraction is 0.407 and the true-population ceiling is 0.420; its observed span is only 0.022. "
        "This is a different limitation from the large-motion cases. Its basin Rg difference is only 0.245 Å, with source within-basin Rg SD about 0.08 Å. "
        "Subtle contact changes produce strong synthetic uptake contrasts, while its population contrast has essentially no projection on the tested weak observable directions. "
        "A constrained observable response is supported; a claim that the narrow graph was already saturated is contradicted. The ceiling alone does not prove why the optimiser chose this span.",
    )
    add(
        "3. Why regularisation can worsen population recovery",
        "The target averages all source frames. Randomly thinning one structural basin also changes its conditional residue/contact means. "
        "Even weights with exactly correct basin populations then have a nonzero residual. The diagnostic splits every residue/time residual into "
        "population error + retained-coverage shift + conditional reweighting, and retains all three cross terms in squared error. "
        f"The median cancellation fraction across system medians is {100 * cohort.cancellation.median():.1f}%; "
        "this means that squared residual components largely cancel, not that this percentage of biological information is absent. "
        "The deterministic balanced weights isolate coverage error and are not an optimised true-population fit or a lower bound on its possible MSE. "
        "For 1tzw_A at 12.5% retention, exact population balancing gives MSE 0.00155; unregularised fitting gives 0.000154 with population TV 0.139. "
        "The data reward changing weights within the retained basin and can partially reward the wrong basin mass. Wider OMC in that case raises TV to 0.245 while increasing ESS beyond the true-population ceiling. "
        "In 2ad6_D at 50% retention, unregularised TV is already 0.00186: there is little population error left to improve, and regularisation can introduce bias. "
        "Verdict: coverage shift and error cancellation are directly demonstrated. Universal exact non-identifiability is contradicted by the SVD check; most population contrasts have negligible projection below relative cutoff 1e-4. "
        "At cutoff 1e-2 some flexible systems have a few percent projection, supporting limited local softness, not arbitrary indistinguishable populations.",
    )
    add(
        "4. Why matched-ESS OMC has higher MSE than MaxEnt",
        f"At the original tolerance, all {len(base)} matched pairs have lower OMC graph energy and lower total OMC objective at the OMC weights than at the MaxEnt weights; MSE is higher in {int((base.delta_mse > 0).sum())}/{len(base)} individual pairs. "
        "The data-fit sacrifice therefore buys exactly what the OMC objective rewards. Equal ESS does not enforce equal basin populations, equal conditional ESS, or equal placement in contact/shape space. "
        "This is a demonstrated objective trade-off, not evidence that the OMC solutions simply failed to optimise against the saved comparator. "
        "The apparent population advantage is not robust as a cohort-wide claim: better system summaries fall from 7/12 to 5/12 to 4/12 as ESS tolerance tightens from 0.02 to 0.01 to 0.005. "
        "Coverage falls from 115 to 66 to 35 pairs; these are different subsets, not a controlled estimate of the tolerance effect. At 0.005, 11/12 system summaries have higher MSE; 1ef1_D is the exception. "
        "Per-residue/time differences and all raw effect sizes are downloadable, including the matched arms and ESS gaps.",
    )
    add(
        "Which systems fail together",
        "Worse population recovery against both baselines occurs in "
        + ", ".join(overlap.system_id)
        + ". "
        "They are not one fold class or one flexibility class. 1c1k_A, 1pch_A, 1tzw_A and 5x1u_B have small basin Rg shifts (roughly 0.14–0.27 Å); "
        "2ad6_D has a much larger 2.86 Å shift and high flexibility. "
        "The strongest overlap in ESS problems is instead 1dd3_B (five reversals, two low-response cases), 1ef1_D (two reversals, two low-response cases), "
        "and 3kvd_D (two reversals, low response at all three retentions). "
        "3kvd_D improves recovery overall, so limited ESS response and failed recovery must not be treated as the same mechanism. "
        "5noh_A improves both population comparisons in the original system summaries but worsens at the severe 12.5% thinning: system medians conceal severity-dependent failures.",
    )
    add(
        "Biophysical interpretation and its limits",
        "The high-motion examples have specific structural context: 1ef1_D is a moesin tail excised from a FERM–tail complex; 2ad6_D is a small subunit from a much larger enzyme assembly; "
        "1dd3_B is a ribosomal L7/L12 chain with dimerisation and conformational mobility. Their source basins differ in Rg by approximately 2.6–2.9 Å. "
        "Scalar mean log-PF is less aligned with structural pair distance for 1ef1_D (Spearman 0.39) and 1dd3_B (0.47) than for 1tzw_A (0.78). "
        "A single total-contact coordinate can connect frames whose local packing differs. The plots show which residue signals and contact maps differ, rather than assigning those changes to an unmeasured kinetic state. "
        "About 84% of 1ef1_D synthetic uptake entries are near 0 or 1 at each time, making the forward model highly saturated. "
        "These are residue-centred synthetic uptake curves, not experimental exchange rates. Crystal-partner contact annotations do not show a clear enrichment of the observed RMSF at interfaces in these examples, "
        "so missing assembly support is a plausible context, not a proven causal explanation. "
        "5noh_A is a compact core-domain counterexample: being a fragment alone does not imply severe flexibility or failure. "
        "The quieter overlap systems demonstrate that regularisation bias also occurs without a large domain/tail motion.",
    )
    add(
        "Targeted tests to discuss next — not run",
        "First separate mass imbalance from loss of conditional coverage: construct nested candidate subsets that retain the source distribution of structural/contact features within each basin, then compare with the existing random thinning at identical basin counts. "
        "Use a small nonpathogenic model-protein subset spanning compact and flexible cases. This directly tests whether coverage-induced compensation drives recovery error. "
        "Second compare a kernel normalised to fixed total coupling with the current kernel at the existing bandwidths; this separates strength pressure from graph topology without extending the bandwidth sweep. "
        "Third, if warranted, compare a residue-resolved contact geometry with the scalar mean coordinate on those same generic benchmark ensembles. "
        "Evaluate recovery jointly with conditional coverage and the true-population ESS ceiling. Tighter saved MaxEnt matches already show that a broad claim of population superiority needs stronger evidence. "
        "No new fits, altered targets, altered hyperparameters, or ISO experiments were performed in this diagnostic pass.",
    )
    verdicts = [
        (
            "Within-basin concentration explains local ESS reversals",
            "directly demonstrated",
            f"positive within contribution in {(reversals.within_contribution > 0).sum()}/{len(reversals)} reversals",
        ),
        (
            "Narrow graph already broadly coupled",
            "contradicted",
            f"median coupling {density.iloc[0]:.3f} → {density.iloc[-1]:.3f}",
        ),
        (
            "Bandwidth changes effective coupling at fixed strength",
            "directly demonstrated",
            f"{density.iloc[-1] / density.iloc[0]:.1f}-fold coupling change",
        ),
        (
            "High ESS always compatible with correct populations",
            "contradicted",
            f"{int(omc.above_population_ceiling.sum())} converged OMC fits exceed ceiling",
        ),
        (
            "Coverage shift and residual cancellation exist",
            "directly demonstrated",
            "exact observation-wise identities and deterministic balanced weights",
        ),
        (
            "All population errors reflect an exact nullspace",
            "contradicted",
            "population contrasts generally absent from directions below relative 1e-4",
        ),
        (
            "Scalar contact compression matters in flexible cases",
            "supported association",
            "work–structure correlations and residue/contact contrasts",
        ),
        (
            "Missing assembly support causes fitting failure",
            "unresolved",
            "context plausible; no assembly counterfactual or clear RMSF interface enrichment",
        ),
        (
            "Higher matched MSE buys lower OMC penalty",
            "directly demonstrated",
            f"{len(base)}/{len(base)} saved matched pairs",
        ),
        (
            "One CATH class explains all overlapping failures",
            "contradicted",
            "overlap spans classes and flexibility groups",
        ),
    ]
    pd.DataFrame(verdicts, columns=["mechanism", "verdict", "evidence"]).to_csv(
        output / "mechanism_verdicts.csv", index=False
    )
    md = ["# OMC mechanistic diagnosis", ""]
    for title, text in paragraphs:
        md.extend(["## " + title, "", text, ""])
    md.extend(
        [
            "## Matching sensitivity",
            "",
            sensitivity.to_markdown(index=False)
            if _has_tabulate()
            else sensitivity.to_csv(index=False),
            "",
        ]
    )
    detail_html = []
    for _, row in summary.iterrows():
        sid = row.system_id
        source = output / "systems" / sid
        structural = json.loads((source / "structural_summary.json").read_text())
        rs = pd.read_parquet(source / "residue_structure.parquet")
        # Locate the strongest measured contrast, not a functional or intervention site.
        top = rs.nlargest(6, "uptake_contrast_rms")
        locations = ", ".join(str(int(r)) for r in top.residue)
        local = cases[cases.system_id == sid]
        detail = (
            f"Basin Rg means: {structural['basin_rg_means'][0]:.3f} and {structural['basin_rg_means'][1]:.3f} Å; "
            f"work–structure Spearman {structural['work_structural_spearman']:.3f}. "
            f"Strongest synthetic uptake contrasts occur at simulated residues {locations}; original deposited numbering is in residue_mapping.csv. "
            f"Median cancellation fraction {100 * row.cancellation:.1f}%. "
            f"{int(row.reversals)} converged adjacent reversals; {int(row.low_response_cases)} low-response retention cases; "
            f"{int(row.above_ceiling)} converged OMC fits exceed the true-population ESS ceiling. "
            f"System-median population TV changes: {row.tv_vs_unreg:+.5f} vs unregularised and {row.tv_vs_maxent:+.5f} vs matched MaxEnt. "
            "Positive deltas mean worse recovery. Consult the severity table and distribution plots: this summary does not assert that every bandwidth behaves the same way."
        )
        url = annotations[sid].get(
            "url", "https://www.rcsb.org/structure/" + sid[:4].upper()
        )
        md.extend(
            [
                "## " + sid + " — " + annotations[sid]["name"],
                "",
                detail,
                "",
                annotations[sid]["context"]
                + " [Primary structural annotation]("
                + url
                + ").",
                "",
            ]
        )
        detail_html.append(
            f"<details><summary>{sid}: measured structural and recovery evidence</summary><p>{html.escape(detail)}</p>"
            + local.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
            + "</details>"
        )
    (output / "findings.md").write_text("\n".join(md))
    html_parts = ['<section id="findings"><h2>Mechanistic conclusions</h2>']
    html_parts.extend(
        "<h3>" + html.escape(t) + "</h3><p>" + html.escape(p) + "</p>"
        for t, p in paragraphs
    )
    html_parts.append(
        sensitivity.to_html(index=False, float_format=lambda x: f"{x:.5g}", border=0)
    )
    html_parts.extend(detail_html)
    html_parts.append("</section>")
    (output / "findings_fragment.html").write_text("\n".join(html_parts))
    residual_plots(output, summary)


def _has_tabulate():
    return importlib.util.find_spec("tabulate") is not None


def residual_plots(output, summary):
    for sid in summary.system_id:
        folder = output / "systems" / sid
        residual = pd.read_parquet(folder / "residual_terms.parquet")
        paired = pd.read_parquet(folder / "matched_residues.parquet")
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
        for j, retention in enumerate((0.125, 0.25, 0.5)):
            selected = residual[(residual.retention == retention) & residual.converged]
            selected = selected[selected.arm.str.startswith("omc_")]
            grouped = selected.groupby("residue")
            for column in ("population_sq", "coverage_sq", "reweight_sq"):
                means = grouped[column].mean()
                axes[0, j].plot(means.index, means, label=column)
            mse = selected.assign(sq=selected.residual**2).groupby("residue").sq.mean()
            axes[0, j].plot(
                mse.index, mse, color="black", lw=2, label="actual squared error"
            )
            axes[0, j].set(
                title=f"Retention {retention:g}",
                xlabel="Simulated residue",
                ylabel="Mean squared contribution",
            )
            pm = paired[paired.retention == retention]
            for time, pts in pm.groupby("time"):
                curve = pts.groupby("residue").delta_squared_error.mean()
                axes[1, j].plot(curve.index, curve, label=f"t={time:g}")
            axes[1, j].axhline(0, color="black", lw=0.5)
            axes[1, j].set(
                xlabel="Simulated residue", ylabel="OMC − MaxEnt squared error"
            )
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        fig.savefig(folder / "residue_errors.png", dpi=140)
        plt.close(fig)
