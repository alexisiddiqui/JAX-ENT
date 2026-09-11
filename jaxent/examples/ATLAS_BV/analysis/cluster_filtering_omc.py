"""Single-system OMC calibration using fixed targets and cluster-filtered candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import system_data
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    pseudo_uptake,
    stable_seed,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 import (
    bandwidth_from_quantile,
)
from jaxent.examples.ATLAS_BV.analysis.original_omc_iso_validation_checkpoint35 import (
    arm_specs,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/cluster_filtering_omc"
STRENGTHS = (0.0, *tuple(10.0**power for power in range(-7, 3)))
VERSION = 2


def partitions(distance, seed):
    """Freeze source-ensemble labels, keeping one eligible k per complexity band."""
    records, labels_by_k = [], {}
    for k in range(2, 9):
        if k >= len(distance):
            continue
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(
            distance
        )
        counts = np.bincount(labels, minlength=k)
        score = (
            float(silhouette_score(distance, labels, metric="precomputed"))
            if len(np.unique(labels)) > 1
            else -1.0
        )
        records.append(
            dict(
                k=k,
                silhouette=score,
                minimum_size=int(counts.min()),
                counts=counts.tolist(),
                eligible=bool(score > 0 and counts.min() >= 20),
                selected=False,
            )
        )
        labels_by_k[k] = labels
    for low, high in ((2, 3), (4, 5), (6, 8)):
        eligible = [r for r in records if low <= r["k"] <= high and r["eligible"]]
        if eligible:
            min(eligible, key=lambda r: (-r["silhouette"], r["k"]))["selected"] = True
    return pd.DataFrame(records), {
        r["k"]: labels_by_k[r["k"]] for r in records if r["selected"]
    }


def filtered_indices(labels, thinned, fraction, seed):
    """Nested seeded subsets, retaining at least two members of every cluster."""
    keep = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        order = np.random.default_rng(stable_seed(seed, int(cluster))).permutation(
            members
        )
        count = (
            min(len(members), max(2, int(np.floor(fraction * len(members)))))
            if cluster in thinned
            else len(members)
        )
        keep.extend(order[:count])
    return np.sort(np.asarray(keep, dtype=int))


def candidate_cases(labels, seed):
    yield "unfiltered", (), 1.0, np.arange(len(labels))
    clusters = np.unique(labels)
    largest = tuple(sorted(clusters, key=lambda c: (-np.sum(labels == c), int(c)))[:2])
    groups = [(int(c),) for c in clusters]
    # For k=2, the joint-filtering case is a sample-size control, since both
    # clusters shrink at approximately the same rate (up to integer rounding).
    groups.append(largest)
    for group in groups:
        for fraction in (0.5, 0.25, 0.125):
            key = "clusters_" + "_".join(map(str, group)) + f"_retain_{fraction:g}"
            yield key, group, fraction, filtered_indices(labels, group, fraction, seed)


def kernel_batch(work_distance, keep, seed):
    """Pad candidate kernels to source size so all candidates share one compiled fit."""
    specs = arm_specs()
    distance = work_distance[np.ix_(keep, keep)]
    permutation = np.random.default_rng(seed).permutation(len(keep))
    kernels, kinds, sigmas = [], [], []
    for spec in specs:
        sigma = (
            bandwidth_from_quantile(distance, spec.sigma_quantile)
            if spec.sigma_quantile is not None
            else np.nan
        )
        chosen = (
            distance[np.ix_(permutation, permutation)] if spec.rewired else distance
        )
        small = (
            np.exp(-np.minimum((chosen / sigma) ** 2 / 2, 80))
            if np.isfinite(sigma)
            else np.zeros_like(distance)
        )
        if spec.family == "prior_rbf_locked":
            np.fill_diagonal(small, 0)
        kernel = np.zeros_like(work_distance)
        kernel[np.ix_(keep, keep)] = small
        kernels.append(kernel)
        kinds.append(
            2
            if spec.family == "maxent"
            else 3
            if spec.family == "prior_rbf_locked"
            else int(spec.normalise)
        )
        sigmas.append(sigma)
    return np.asarray(kernels), np.asarray(kinds), np.asarray(sigmas)


def trajectory_objective(logits, strength, kernel, kind, values, target, mask):
    """Exact dense OMC contraction; masked padding has no fitted probability mass."""
    n = jnp.sum(mask)
    weights = jax.nn.softmax(jnp.where(mask, logits, -jnp.inf))
    data = jnp.mean((values @ weights - target) ** 2) / (jnp.var(target) + 1e-8)
    # Centre differences before contraction to preserve exact zero at uniform w.
    centered = weights - jnp.where(mask, 1.0 / n, 0.0)
    weighted_centered = weights * centered
    raw = n**2 * (
        jnp.dot(weights * centered**2, kernel @ weights)
        - jnp.dot(weighted_centered, kernel @ weighted_centered)
    )
    raw = jnp.maximum(raw, 0.0)
    denominator = jnp.dot(weights, kernel @ weights)
    norm = raw / jnp.where(denominator > 0, denominator, 1.0)
    log_weights = jnp.where(
        mask, jax.nn.log_softmax(jnp.where(mask, logits, -jnp.inf)), 0
    )
    entropy = jnp.sum(jnp.where(mask, (-jnp.log(n) - log_weights) / n, 0))
    degree = kernel.sum(axis=1)
    laplacian = (
        2
        * (jnp.dot(degree, logits**2) - jnp.dot(logits, kernel @ logits))
        / jnp.where(kernel.sum() > 0, kernel.sum(), 1.0)
    )
    reg = jnp.where(
        kind == 0,
        raw,
        jnp.where(kind == 1, norm, jnp.where(kind == 2, entropy, laplacian)),
    )
    return data + strength * reg


def fit_grid(
    values,
    target,
    mask,
    kernels,
    kinds,
    strengths,
    initial_steps=1000,
    maximum_steps=3000,
):
    return _compiled_fit(
        jnp.asarray(values),
        jnp.asarray(target),
        jnp.asarray(mask),
        jnp.asarray(kernels),
        jnp.asarray(kinds),
        jnp.asarray(strengths),
        initial_steps=initial_steps,
        maximum_steps=maximum_steps,
    )


@partial(jax.jit, static_argnames=("initial_steps", "maximum_steps"))
def _compiled_fit(
    values, target, mask, kernels, kinds, strengths, *, initial_steps, maximum_steps
):
    def losses(current):
        def per_arm(logits, kernel, kind):
            return jax.vmap(
                trajectory_objective, in_axes=(0, 0, None, None, None, None, None)
            )(logits, strengths, kernel, kind, values, target, mask)

        return jax.vmap(per_arm)(current, kernels, kinds)

    optimizer = optax.adam(0.05)
    logits = jnp.zeros((len(kinds), len(strengths), len(mask)))
    state = optimizer.init(logits)

    def advance(count, current, opt_state, active):
        def step(_, carry):
            x, s = carry
            grad = jax.grad(lambda z: losses(z).sum())(x)
            updates, next_state = optimizer.update(grad, s, x)
            next_x = optax.apply_updates(x, updates)
            return jnp.where(active[..., None], next_x, x), next_state

        return jax.lax.fori_loop(0, count, step, (current, opt_state))

    window = min(250, initial_steps // 2)
    logits, state = advance(
        initial_steps - window, logits, state, jnp.ones(logits.shape[:2], bool)
    )
    before = losses(logits)
    logits, state = advance(window, logits, state, jnp.ones(logits.shape[:2], bool))
    after = losses(logits)
    relative = jnp.abs(after - before) / jnp.maximum(jnp.abs(after), 1e-12)
    active = relative > 0.01
    if maximum_steps > initial_steps:

        def extend(carry):
            x, s = carry
            x, s = advance(maximum_steps - initial_steps - window, x, s, active)
            earlier = losses(x)
            x, s = advance(window, x, s, active)
            return x, s, earlier

        logits, state, extended_before = jax.lax.cond(
            jnp.any(active), extend, lambda carry: (*carry, before), (logits, state)
        )
        before = jnp.where(active, extended_before, before)
    after = losses(logits)
    relative = jnp.abs(after - before) / jnp.maximum(jnp.abs(after), 1e-12)
    weights = jax.nn.softmax(jnp.where(mask, logits, -jnp.inf))
    gradient = jax.grad(lambda x: losses(x).sum())(logits)
    return dict(
        weights=weights,
        objective=after,
        relative_change=relative,
        grad_norm=jnp.linalg.norm(gradient, axis=-1),
        steps=jnp.where(active, maximum_steps, initial_steps),
    )


def score_fits(
    fit, flat, target, structural, labels, keep, sigmas, strengths, metadata
):
    records, populations = [], []
    target_mass = np.bincount(labels) / len(labels)
    candidate_mass = np.bincount(labels[keep], minlength=len(target_mass)) / len(keep)
    for a, spec in enumerate(arm_specs()):
        for s, strength in enumerate(strengths):
            weights = np.asarray(fit["weights"])[a, s].astype(float)
            assert (
                np.isfinite(weights).all()
                and (weights >= 0).all()
                and np.isclose(weights.sum(), 1, atol=1e-6)
            )
            mass = np.bincount(labels, weights=weights, minlength=len(target_mass))
            support = weights[weights >= 1 / (10 * len(keep))]
            row = dict(
                **metadata,
                arm=spec.key,
                family=spec.family,
                sigma_quantile=spec.sigma_quantile,
                sigma=float(sigmas[a]),
                strength=float(strength),
                n_candidate=len(keep),
                mse=float(np.mean((flat @ weights - target) ** 2)),
                population_tv=float(np.abs(mass - target_mass).sum() / 2),
                candidate_population_tv=float(
                    np.abs(candidate_mass - target_mass).sum() / 2
                ),
                ess=float(1 / np.sum(weights**2)),
                ess_fraction=float(1 / np.sum(weights**2) / len(keep)),
                support_size=len(support),
                plateau_flag=bool(support.std() / support.mean() < 0.05),
                structural_dispersion=float(weights @ structural @ weights),
                objective=float(fit["objective"][a, s]),
                relative_objective_change=float(fit["relative_change"][a, s]),
                final_grad_norm=float(fit["grad_norm"][a, s]),
                steps=int(fit["steps"][a, s]),
                converged=bool(fit["relative_change"][a, s] <= 0.01),
                simplex_valid=True,
            )
            records.append(row)
            for c, p in enumerate(target_mass):
                inside = weights[labels == c]
                count = int(np.sum(labels[keep] == c))
                ess = float(inside.sum() ** 2 / max(np.sum(inside**2), 1e-30))
                populations.append(
                    dict(
                        **metadata,
                        arm=spec.key,
                        strength=float(strength),
                        cluster=c,
                        target_population=float(p),
                        candidate_population=float(candidate_mass[c]),
                        recovered_population=float(mass[c]),
                        n_candidate_cluster=count,
                        required_enrichment=float(p / candidate_mass[c]),
                        conditional_ess=ess,
                        conditional_ess_fraction=ess / count,
                    )
                )
    return pd.DataFrame(records), pd.DataFrame(populations)


def report(results, populations, audit, destination):
    keys = ["k", "case"]
    results = results.copy()
    results["case_kind"] = [
        "unfiltered_control"
        if row.case == "unfiltered"
        else "sample_size_control"
        if len(row.thinned_clusters.split(",")) == row.k
        else "population_bias"
        for row in results.itertuples()
    ]
    conditional = (
        populations.groupby(keys + ["arm", "strength"])
        .conditional_ess_fraction.min()
        .rename("minimum_conditional_ess_fraction")
    )
    results = results.merge(
        conditional, on=keys + ["arm", "strength"], validate="one_to_one"
    )
    selected = results.loc[results.groupby(keys + ["family"]).mse.idxmin()]
    atomic_parquet(selected, destination / "mse_selected.parquet")
    tables = populations.merge(
        selected[keys + ["arm", "strength"]], on=keys + ["arm", "strength"]
    )
    atomic_parquet(tables, destination / "selected_populations.parquet")
    tables.to_csv(destination / "selected_populations.csv", index=False)
    recovery_summary = selected[selected.family == "omc"][
        [
            *keys,
            "case_kind",
            "retention",
            "candidate_population_tv",
            "population_tv",
            "mse",
            "strength",
            "sigma_quantile",
            "ess_fraction",
            "converged",
        ]
    ]
    recovery_summary.to_csv(destination / "recovery_summary.csv", index=False)
    # A compact overview uses the most population-distorted candidate in each
    # partition, chosen from initial counts alone, before considering fit quality.
    worst = (
        recovery_summary[recovery_summary.case != "unfiltered"]
        .sort_values(
            ["k", "candidate_population_tv", "case"], ascending=[True, False, True]
        )
        .drop_duplicates("k")
    )
    fig, axes = plt.subplots(1, len(worst), figsize=(7 * len(worst), 4), squeeze=False)
    for ax, row in zip(axes.ravel(), worst.itertuples()):
        case_table = tables[(tables.k == row.k) & (tables.case == row.case)]
        reference = case_table.drop_duplicates("cluster").sort_values("cluster")
        x = np.arange(len(reference))
        ax.bar(x - 0.27, reference.target_population, width=0.18, label="Target")
        ax.bar(
            x - 0.09,
            reference.candidate_population,
            width=0.18,
            label="Filtered candidate",
        )
        for offset, family, label in (
            (0.09, "omc", "OMC: minimum MSE"),
            (0.27, "maxent", "MaxEnt: minimum MSE"),
        ):
            key = selected[
                (selected.k == row.k)
                & (selected.case == row.case)
                & (selected.family == family)
            ].iloc[0]
            fitted = case_table[
                (case_table.arm == key.arm) & (case_table.strength == key.strength)
            ].sort_values("cluster")
            ax.bar(x + offset, fitted.recovered_population, width=0.18, label=label)
        ax.set_xticks(x, reference.cluster)
        ax.set(
            xlabel="Cluster",
            ylabel="Population",
            ylim=(0, 1),
            title=f"k={row.k}: {row.case}",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "population_recovery.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(worst), 4, figsize=(18, 4 * len(worst)), squeeze=False)
    for row_axes, case_row in zip(axes, worst.itertuples()):
        example = results[
            (results.k == case_row.k)
            & (results.case == case_row.case)
            & (results.family == "omc")
        ]
        for strength in (1e-4, 1e-3, 1e-2, 1e-1):
            curve = example[example.strength == strength].sort_values("sigma_quantile")
            for ax, metric in zip(
                row_axes,
                (
                    "ess_fraction",
                    "minimum_conditional_ess_fraction",
                    "population_tv",
                    "mse",
                ),
            ):
                ax.plot(
                    curve.sigma_quantile,
                    curve[metric],
                    "o-",
                    label=f"strength={strength:g}",
                )
                ax.set(xlabel="Bandwidth quantile", ylabel=metric)
        row_axes[0].set_title(f"k={case_row.k}: {case_row.case}", fontsize=10)
        row_axes[0].legend(fontsize=8)
        row_axes[-1].set_yscale("log")
    fig.suptitle(
        "Bandwidth control at fixed positive strengths; examples chosen by initial population distortion"
    )
    fig.tight_layout()
    fig.savefig(destination / "bandwidth_control.png", dpi=150)
    plt.close(fig)
    operating = []
    for case, block in results.groupby(keys):
        best = float(block.mse.min())
        for tolerance in (0.01, 0.05, 0.10):
            acceptable = block[
                (block.mse <= best + max(best * tolerance, 1e-8)) & (block.strength > 0)
            ]
            for family, group in acceptable.groupby("family"):
                operating.append(
                    dict(
                        k=int(case[0]),
                        case=case[1],
                        tolerance=tolerance,
                        family=family,
                        fits=len(group),
                        min_strength=float(group.strength.min()),
                        max_strength=float(group.strength.max()),
                        min_population_tv=float(group.population_tv.min()),
                        max_population_tv=float(group.population_tv.max()),
                        min_ess=float(group.ess.min()),
                        max_ess=float(group.ess.max()),
                        min_conditional_ess_fraction=float(
                            group.minimum_conditional_ess_fraction.min()
                        ),
                        max_conditional_ess_fraction=float(
                            group.minimum_conditional_ess_fraction.max()
                        ),
                        min_structural_dispersion=float(
                            group.structural_dispersion.min()
                        ),
                        max_structural_dispersion=float(
                            group.structural_dispersion.max()
                        ),
                        unconverged_fraction=float((~group.converged).mean()),
                    )
                )
    atomic_parquet(pd.DataFrame(operating), destination / "operating_ranges.parquet")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(audit.k, audit.silhouette, "o-")
    chosen = audit[audit.selected]
    ax.scatter(
        chosen.k,
        chosen.silhouette,
        s=150,
        facecolors="none",
        edgecolors="red",
        label="retained",
    )
    ax.set(xlabel="Cluster count", ylabel="Structural silhouette")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "silhouette.png", dpi=150)
    plt.close(fig)
    for k, block in results[results.family == "omc"].groupby("k"):
        positive = block[block.strength > 0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for case, group in positive.groupby("case"):
            for ax, x, y in zip(
                axes,
                ("mse", "population_tv", "strength"),
                ("population_tv", "ess_fraction", "mse"),
            ):
                ax.scatter(group[x], group[y], s=8, alpha=0.3, label=case)
                ax.set(xlabel=x, ylabel=y)
        axes[0].set_xscale("log")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        fig.suptitle(
            f"k={k}: all positive-strength OMC fits; colours identify candidate filters"
        )
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=4, fontsize=6)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.savefig(destination / f"tradeoffs_k{k}.png", dpi=150)
        plt.close(fig)
        for case, group in positive.groupby("case"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for ax, measure in zip(
                axes.ravel(),
                (
                    "mse",
                    "population_tv",
                    "ess_fraction",
                    "minimum_conditional_ess_fraction",
                    "structural_dispersion",
                    "support_size",
                ),
            ):
                table = group.pivot(
                    index="strength", columns="sigma_quantile", values=measure
                ).sort_index()
                display = (
                    np.log10(np.maximum(table.values, 1e-15))
                    if measure == "mse"
                    else table.values
                )
                im = ax.imshow(display, aspect="auto", origin="lower")
                ax.set_xticks(
                    range(len(table.columns)),
                    [f"{v:g}" for v in table.columns],
                    rotation=45,
                )
                ax.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
                ax.set(
                    xlabel="Bandwidth quantile",
                    ylabel="Strength",
                    title="log10 MSE" if measure == "mse" else measure,
                )
                fig.colorbar(im, ax=ax)
            fig.suptitle(f"k={k}, {case}; positive strengths only")
            fig.tight_layout()
            fig.savefig(destination / f"heatmap_k{k}_{case}.png", dpi=120)
            plt.close(fig)
    # Side-by-side target / biased candidate / recovered populations for every case.
    for k, block in tables[tables.arm.str.startswith("omc[")].groupby("k"):
        cases = list(block.case.unique())
        fig, axes = plt.subplots(
            int(np.ceil(len(cases) / 3)),
            3,
            figsize=(14, 3 * int(np.ceil(len(cases) / 3))),
            squeeze=False,
        )
        for ax, case in zip(axes.ravel(), cases):
            group = block[block.case == case].sort_values("cluster")
            x = np.arange(len(group))
            for offset, field, label in (
                (-0.25, "target_population", "Target"),
                (0, "candidate_population", "Candidate"),
                (0.25, "recovered_population", "OMC, min MSE"),
            ):
                ax.bar(x + offset, group[field], width=0.25, label=label)
            ax.set(title=case, xlabel="Cluster", ylabel="Population", ylim=(0, 1))
            ax.set_xticks(x, group.cluster)
        for ax in axes.ravel()[len(cases) :]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"k={k}: fixed target versus filtering and fitted populations")
        fig.tight_layout()
        fig.savefig(destination / f"populations_k{k}.png", dpi=130)
        plt.close(fig)
    summary = dict(
        system=str(results.system_id.iloc[0]),
        selected_k=chosen.k.tolist(),
        candidates=int(results[keys].drop_duplicates().shape[0]),
        expected_candidates=int(sum(1 + 3 * (k + 1) for k in chosen.k)),
        complete_grid=bool(
            len(results)
            == sum(1 + 3 * (k + 1) for k in chosen.k)
            * len(arm_specs())
            * len(STRENGTHS)
            and (
                results.groupby(keys).size() == len(arm_specs()) * len(STRENGTHS)
            ).all()
            and not results.duplicated(keys + ["arm", "strength"]).any()
        ),
        fits=len(results),
        unconverged_fraction=float((~results.converged).mean()),
        positive_omc_selected=int(
            ((selected.family == "omc") & (selected.strength > 0)).sum()
        ),
        positive_omc_selected_biased_candidates=int(
            (
                (selected.family == "omc")
                & (selected.strength > 0)
                & (selected.case_kind == "population_bias")
            ).sum()
        ),
        scope="Single-system calibration; no cross-system or ISO verdict",
        operating_ranges=operating,
    )
    atomic_yaml(destination / "report.yaml", summary)
    # Post-calibration development recommendations, distinct from the executed grid.
    proposed = dict(
        status="proposed after single-system calibration; not yet expanded",
        system="1yoz_B",
        sigma_quantiles=[0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        primary_and_rewired_strengths=[
            0.0,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            0.1,
        ],
        normalised_strengths=[
            0.0,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
        ],
        reason="Resolve the observed onset of distribution changes and the MSE tradeoff; retain zero as reference and every bandwidth",
        limitation="Many low-error trajectories remain optimizer-sensitive. These are exploration ranges, not demonstrated optimum settings.",
    )
    atomic_yaml(destination / "proposed_hyperparameter_ranges.yaml", proposed)
    from html import escape

    images = ["silhouette.png", "population_recovery.png", "bandwidth_control.png"] + [
        f"{prefix}_k{k}.png"
        for k in chosen.k
        for prefix in ("tradeoffs", "populations")
    ]
    sections = [
        "<!doctype html><meta charset='utf-8'><title>Cluster-filtering OMC calibration</title>",
        "<style>body{font:16px sans-serif;max-width:1400px;margin:30px auto}img{max-width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>",
        "<h1>Single-system cluster-filtering OMC calibration</h1>",
        "<p>Fixed source-ensemble targets; all-residue MSE; full positive-strength sweeps. Population TV is half the sum of absolute cluster-mass errors. No cross-system inference.</p>",
        audit.to_html(index=False),
        f"<p>{len(results)} fits; {(~results.converged).mean():.1%} have unresolved final-window objective change. Interpret these fits as optimizer-sensitive.</p>",
    ]
    sections += [f"<img src='{name}' alt='{escape(name)}'>" for name in images]
    sections += ["<h2>Positive-strength heatmaps by candidate</h2>"]
    for k, case in results[keys].drop_duplicates().itertuples(index=False, name=None):
        name = f"heatmap_k{k}_{case}.png"
        sections.append(
            f"<details><summary>k={k}: {escape(case)}</summary><img src='{name}'></details>"
        )
    sections += [
        "<h2>Proposed development ranges</h2><p>Primary/re-wired strengths: 1e-5 to 1e-1; normalised strengths: 1e-6 to 1e-2, with half-decade refinement and zero reference. Retain every bandwidth. See proposed_hyperparameter_ranges.yaml. These ranges are proposed after this single-system calibration; optimizer-sensitive regions require care.</p>",
        "<h2>All-residue MSE selected fits</h2>",
        selected[
            [
                *keys,
                "family",
                "strength",
                "sigma_quantile",
                "mse",
                "population_tv",
                "ess_fraction",
                "converged",
            ]
        ].to_html(index=False),
        "<p><a href='selected_populations.csv'>Download target, candidate and recovered population table</a></p>",
    ]
    (destination / "index.html").write_text("\n".join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="1yoz_B")
    parser.add_argument("--frame-cap", type=int, default=512)
    parser.add_argument("--initial-steps", type=int, default=1000)
    parser.add_argument("--maximum-steps", type=int, default=3000)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    window = min(250, args.initial_steps // 2)
    if args.initial_steps < 2 or args.maximum_steps < args.initial_steps:
        parser.error(
            "maximum steps must be at least initial steps, and initial steps at least 2"
        )
    if (
        args.maximum_steps != args.initial_steps
        and args.maximum_steps < args.initial_steps + window
    ):
        parser.error("extension must include a complete convergence window")
    config = load_config()
    destination = OUTPUT / args.system
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        version=VERSION,
        system=args.system,
        frame_cap=args.frame_cap,
        initial_steps=args.initial_steps,
        maximum_steps=args.maximum_steps,
        strengths=list(STRENGTHS),
        seed=config["analysis"]["seed"],
        protocol=config["protocol"],
    )
    encoded = json.dumps(manifest, sort_keys=True)
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    identity = destination / "identity.txt"
    if identity.exists() and identity.read_text() != fingerprint:
        raise ValueError(
            "Run settings differ from existing shards; use a fresh output directory"
        )
    identity.write_text(fingerprint)
    atomic_yaml(destination / "protocol.yaml", manifest)
    if not args.report_only:
        row = next(r for r in load_systems() if r["system_id"] == args.system)
        cache = destination / "source.npz"
        if cache.exists():
            with np.load(cache) as source:
                structural, flat, work, indices = [
                    source[key] for key in ("structural", "flat", "work", "indices")
                ]
        else:
            data = system_data(row, config)
            global_indices, distances = data["matrices"][1]
            take = np.linspace(
                0,
                len(global_indices) - 1,
                min(args.frame_cap, len(global_indices)),
                dtype=int,
            )
            indices = np.asarray(global_indices)[take]
            structural = np.asarray(distances)[np.ix_(take, take)]
            z = data["z"][:, indices]
            flat = pseudo_uptake(z).reshape(-1, len(indices))
            work = np.abs(z.mean(axis=0)[:, None] - z.mean(axis=0)[None, :])
            np.savez_compressed(
                cache, structural=structural, flat=flat, work=work, indices=indices
            )
        target = flat.mean(axis=1)
        audit, selected = partitions(structural, config["analysis"]["seed"])
        atomic_parquet(audit, destination / "clustering.parquet")
        atomic_yaml(
            destination / "clustering.yaml", dict(partitions=audit.to_dict("records"))
        )
        print(audit.to_string(index=False), flush=True)
        if not selected:
            atomic_yaml(
                destination / "report.yaml", dict(status="no eligible partitions")
            )
            return
        parts = destination / "parts"
        parts.mkdir(exist_ok=True)
        for k, labels in selected.items():
            for case, thinned, fraction, keep in candidate_cases(
                labels, config["analysis"]["seed"]
            ):
                name = f"k{k}_{case}"
                path = parts / f"{name}.fits.parquet"
                pop_path = parts / f"{name}.populations.parquet"
                if path.exists() and pop_path.exists():
                    continue
                mask = np.zeros(len(labels), bool)
                mask[keep] = True
                kernels, kinds, sigmas = kernel_batch(
                    work, keep, stable_seed(args.system, k, case)
                )
                fit = jax.tree.map(
                    np.asarray,
                    fit_grid(
                        flat,
                        target,
                        mask,
                        kernels,
                        kinds,
                        STRENGTHS,
                        args.initial_steps,
                        args.maximum_steps,
                    ),
                )
                metadata = dict(
                    system_id=args.system,
                    k=k,
                    case=case,
                    retention=fraction,
                    thinned_clusters=",".join(map(str, thinned)),
                )
                fits, populations = score_fits(
                    fit,
                    flat,
                    target,
                    structural,
                    labels,
                    keep,
                    sigmas,
                    STRENGTHS,
                    metadata,
                )
                np.savez_compressed(
                    parts / f"{name}.weights.npz",
                    weights=fit["weights"],
                    source_indices=indices,
                    candidate_indices=indices[keep],
                    labels=labels,
                    target=target,
                )
                atomic_parquet(fits, path)
                atomic_parquet(populations, pop_path)
                print(
                    f"[{name}] n={len(keep)} fits={len(fits)} unconverged={(~fits.converged).mean():.1%}",
                    flush=True,
                )
    results = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((destination / "parts").glob("*.fits.parquet"))
        ],
        ignore_index=True,
    )
    populations = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((destination / "parts").glob("*.populations.parquet"))
        ],
        ignore_index=True,
    )
    atomic_parquet(results, destination / "results.parquet")
    atomic_parquet(populations, destination / "populations.parquet")
    report(
        results,
        populations,
        pd.read_parquet(destination / "clustering.parquet"),
        destination,
    )


if __name__ == "__main__":
    main()
