"""Paired graph comparisons at fixed mean off-diagonal coupling."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import omc_graph_control as g
from . import omc_decoy_control as e
from .omc_decoy_report import table, figure

LABELS = {
    "scalar_logpf": "Scalar log-PF",
    "profile_logpf": "Full log-PF profile",
    "structure": "Cα structural geometry",
    "ref2015_total": "PyRosetta total score",
    "maxent": "MaxEnt",
    "unregularised": "Unregularised",
}
COLORS = dict(zip(LABELS, plt.get_cmap("tab10").colors))


def score(case, data, fit, index, graph, spec, source, reused):
    w = fit["weights"][index]
    labels = data["native_labels"]
    count = len(labels)
    masses = np.bincount(
        labels, weights=w[:count], minlength=len(data["target_masses"])
    )
    target_masses = data["target_masses"]
    retained = target_masses > 0
    retained_mass = masses[retained].sum()
    conditional_tv = (
        0.5 * abs(masses[retained] / retained_mass - target_masses[retained]).sum()
        if retained_mass > 0
        else 1.0
    )
    predicted = e.uptake(data["log_pf"], data["rates"], w, data["times"])
    mse = float(np.mean((predicted - data["target"]) ** 2))
    if spec["family"] == "omc":
        penalty = e.regularisation(w, data["kernels"][index])
    elif spec["family"] == "maxent":
        penalty = np.mean(-np.log(len(w)) - np.log(w))
    else:
        penalty = 0.0
    recomputed_objective = mse / float(data["scale"]) + spec["strength"] * penalty
    np.testing.assert_allclose(
        recomputed_objective, fit["objective"][index], rtol=1e-7, atol=1e-10
    )
    active = np.flatnonzero(w[:count] >= 1e-4)
    coverage = (
        float(
            data["target_weights"]
            @ source["structural_distance"][:, active].min(axis=1)
        )
        if len(active)
        else np.nan
    )
    result = dict(
        case=case["name"],
        kind=case["kind"],
        graph=graph,
        arm=index,
        **spec,
        reused=reused,
        converged=bool(fit["converged"][index]),
        mse=mse,
        population_tv=float(
            0.5 * (abs(masses - target_masses).sum() + w[count:].sum())
        ),
        retained_population_tv=float(conditional_tv),
        decoy_mass=float(w[data["decoy"]].sum()),
        retained_state_coverage=float((masses[retained] >= 1e-4).mean()),
        active_native_coverage_distance=coverage,
        **e.ess_stats(w, count),
        objective=float(fit["objective"][index]),
        grad_norm=float(fit["grad_norm"][index]),
        relative_change=float(fit["relative_change"][index]),
        steps=int(fit["steps"][index]),
        initialisation_objective_gap=float(fit["initialisation_objective_gap"][index]),
        initialisation_weight_tv=float(fit["initialisation_weight_tv"][index]),
    )
    return result, masses, predicted


def paired_results(fits):
    rows = []
    scalar = fits[fits.graph == "scalar_logpf"].set_index(["case", "quantile"])
    for _, row in fits[fits.graph.isin(g.GRAPHS)].iterrows():
        old = scalar.loc[(row["case"], row["quantile"])]
        pair = dict(
            case=row["case"],
            kind=row["kind"],
            graph=row["graph"],
            quantile=row["quantile"],
            valid_pair=bool(row["converged"] and old["converged"]),
        )
        for metric in [
            "population_tv",
            "retained_population_tv",
            "decoy_mass",
            "mse",
            "native_ess_fraction",
            "active_native_coverage_distance",
        ]:
            pair["delta_" + metric] = (
                row[metric] - old[metric] if pair["valid_pair"] else np.nan
            )
        rows.append(pair)
    return pd.DataFrame(
        rows,
        columns=[
            "case",
            "kind",
            "graph",
            "quantile",
            "valid_pair",
            *[
                "delta_" + m
                for m in [
                    "population_tv",
                    "retained_population_tv",
                    "decoy_mass",
                    "mse",
                    "native_ess_fraction",
                    "active_native_coverage_distance",
                ]
            ],
        ],
    )


def report(output):
    output = Path(output)
    manifest = g.load_manifest(output)
    source = e.load_npz(output / "source.npz")
    native_labels = source["native_labels"]
    native_count = len(native_labels)
    struct_distance = g.structural_distance(source["xyz"][source["candidate_indices"]])
    n_states = len(np.unique(native_labels))
    fit_rows, populations, graph_rows, residual_rows = [], [], [], []
    completed = 0
    failures = []
    identity = e.digest(output / "manifest.json")
    for case in manifest["cases"]:
        folder = output / "cases" / case["name"]
        data = e.load_npz(folder / "input.npz")
        old = e.load_npz(folder / "reused_fit.npz")
        groups = [("scalar_logpf", data["kernels"], data["sigmas"], None, old, True)]
        for graph in g.GRAPHS:
            if not g.applicable(graph, case["kind"]):
                continue
            values = e.load_npz(folder / graph / "graph.npz")
            fit = None
            marker = folder / graph / "complete.json"
            if marker.exists():
                complete = json.loads(marker.read_text())
                if complete["identity"] != identity or complete["sha256"] != e.digest(
                    folder / graph / "fit.npz"
                ):
                    raise ValueError(f"Fit identity mismatch: {case['name']}/{graph}")
                fit = e.load_npz(folder / graph / "fit.npz")
                g.validate_fit(fit, data["log_pf"].shape[1])
                completed += 1
            elif (folder / graph / "failure.json").exists():
                failure = json.loads((folder / graph / "failure.json").read_text())
                if failure["identity"] != identity:
                    raise ValueError("Numerical-failure record identity mismatch")
                failures.append(failure)
            groups.append(
                (graph, values["kernels"], values["sigmas"], values, fit, False)
            )
        for graph, kernels, sigmas, values, fit, reused in groups:
            off = ~np.eye(len(data["projection"]), dtype=bool)
            native_off = ~np.eye(native_count, dtype=bool)
            all_labels = np.r_[
                native_labels, np.full(len(data["projection"]) - native_count, -1)
            ]
            cross_states = (all_labels[:, None] != all_labels[None, :]) & off
            cross_decoy = data["decoy"][:, None] != data["decoy"][None, :]
            for qi, kernel in enumerate(kernels):
                native = kernel[:native_count, :native_count]
                strong = (native >= np.quantile(native[native_off], 0.9)) & native_off
                graph_rows.append(
                    dict(
                        case=case["name"],
                        kind=case["kind"],
                        graph=graph,
                        quantile=float(e.QUANTILES[qi]),
                        sigma=float(sigmas[qi]),
                        raw_coupling=float(values["raw_coupling"][qi])
                        if values
                        else float(kernel[off].mean()),
                        scaling_factor=float(values["scaling_factor"][qi])
                        if values
                        else 1.0,
                        final_coupling=float(kernel[off].mean()),
                        target_coupling=float(data["kernels"][qi][off].mean()),
                        coupling_error=float(
                            abs(kernel[off].mean() - data["kernels"][qi][off].mean()),
                        ),
                        cross_state_edge_fraction=float(
                            kernel[cross_states].sum() / kernel[off].sum()
                        ),
                        cross_decoy_edge_fraction=float(
                            kernel[cross_decoy].sum() / kernel[off].sum()
                        ),
                        projected_weight_penalty=e.regularisation(
                            data["projection"], kernel
                        ),
                        mean_structural_distance_strong_edges=float(
                            struct_distance[strong].mean()
                        ),
                        weighted_native_structural_distance=float(
                            (native[native_off] * struct_distance[native_off]).sum()
                            / native[native_off].sum()
                        ),
                    )
                )
            if fit is None:
                continue
            specs = e.arm_specs() if reused else e.arm_specs()[:6]
            fit_data = {**data, "kernels": kernels}
            for index, spec in enumerate(specs):
                label = graph if spec["family"] == "omc" else spec["family"]
                row, masses, predicted = score(
                    case, fit_data, fit, index, label, spec, source, reused
                )
                fit_rows.append(row)
                for state, mass in enumerate(masses):
                    populations.append(
                        dict(
                            case=case["name"],
                            graph=label,
                            arm=index,
                            state=state,
                            target=float(data["target_masses"][state]),
                            fitted=float(mass),
                        )
                    )
                for ti, time in enumerate(data["times"]):
                    for ri, residue in enumerate(source["residues"]):
                        residual_rows.append(
                            dict(
                                case=case["name"],
                                graph=label,
                                arm=index,
                                time_seconds=float(time),
                                residue=int(residue),
                                target=float(data["target"][ti, ri]),
                                predicted=float(predicted[ti, ri]),
                                residual=float(
                                    predicted[ti, ri] - data["target"][ti, ri]
                                ),
                            )
                        )
    fits = table(output, "fits", fit_rows)
    pops = table(output, "populations", populations)
    graphs = table(output, "graph_diagnostics", graph_rows)
    table(output, "residuals", residual_rows)
    paired = table(output, "paired_vs_scalar", paired_results(fits))
    selected = (
        fits[fits.converged]
        .sort_values(["mse", "arm"])
        .groupby(["case", "graph"], sort=False)
        .head(1)
    )
    table(output, "mse_selected", selected)
    pair_summary = (
        paired.loc[paired.valid_pair.astype(bool)]
        .groupby(["kind", "graph"])
        .agg(
            valid_pairs=("case", "size"),
            median_delta_population_tv=("delta_population_tv", "median"),
            median_delta_mse=("delta_mse", "median"),
            median_delta_native_ess_fraction=("delta_native_ess_fraction", "median"),
        )
    )
    table(output, "paired_summary", pair_summary.reset_index())
    selected_summary = selected.groupby(["kind", "graph"]).agg(
        cases=("case", "size"),
        population_tv=("population_tv", "median"),
        decoy_mass=("decoy_mass", "median"),
        mse=("mse", "median"),
        native_ess_fraction=("native_ess_fraction", "median"),
    )
    table(output, "selected_summary", selected_summary.reset_index())
    # Matched-ESS checks remain descriptive and never interpolate or extrapolate.
    matches = []
    for case, frame in fits[fits.converged].groupby("case"):
        ref = frame[frame.graph == "scalar_logpf"]
        for _, row in frame[frame.graph.isin(g.GRAPHS)].iterrows():
            if ref.empty:
                continue
            delta = abs(ref.native_ess_fraction - row.native_ess_fraction)
            near = ref.loc[delta.idxmin()]
            matches.append(
                dict(
                    case=case,
                    graph=row.graph,
                    quantile=row["quantile"],
                    scalar_quantile=near["quantile"],
                    ess_gap=float(delta.min()),
                    within_two_points=bool(delta.min() <= 0.02),
                    delta_population_tv=float(row.population_tv - near.population_tv),
                    delta_mse=float(row.mse - near.mse),
                )
            )
    table(
        output,
        "nearest_ess",
        matches if matches else pd.DataFrame(columns=["case", "graph", "ess_gap"]),
    )
    exclusions = json.loads((output / "not_applicable.json").read_text())
    table(
        output,
        "numerical_failures",
        failures
        if failures
        else pd.DataFrame(columns=["case", "graph", "status", "error"]),
    )
    new = fits[~fits.reused]
    parts = [
        "<h1>OMC graph comparison: protection, structure and energy</h1>",
        f'<p class="status">{completed} of {len(manifest["jobs"])} new graph/case jobs complete; '
        f"{len(new)} of {manifest['new_arms']} new arms available, {int(new.converged.sum())} converged. "
        f"{manifest['reused_arms']} existing arms reused.</p>",
        "<p>Same 1tzw_A candidates, empirical targets, 0.5 Å contact features and HDX predictions. "
        "OMC strength is 0.1. New kernels match the scalar control’s mean off-diagonal coupling separately "
        "for every case and bandwidth. Total coupling still changes across bandwidths.</p>",
        "<p>Profile distance retains every residue’s log-PF. Structural distance uses Cα pair-distance vectors. "
        "Energy distance is the absolute difference in cached ref2015 total scores. Energetic similarity "
        "does not preferentially weight low-energy frames or turn Rosetta scores into free energies.</p>",
        "<h2>MSE-selected recovery</h2>",
        "<p>Values below select the lowest all-residue MSE among converged arms separately for each graph and case. "
        "No population truth enters selection. Counts can differ because unresolved fits are excluded. "
        "External rows are medians across five perturbations of one system.</p>",
        selected_summary.to_html(float_format=lambda x: f"{x:.5g}"),
        "<h2>Paired effects at the same bandwidth and coupling</h2>",
        "<p>Deltas are new graph minus scalar control. Negative population-error and MSE deltas favour the new graph. "
        "Only pairs with both arms converged contribute. Bandwidths and decoy seeds are not independent biological replicates.</p>",
        pair_summary.to_html(float_format=lambda x: f"{x:.5g}"),
    ]
    if failures:
        parts.append(pd.DataFrame(failures).to_html(index=False))
    if len(paired):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for graph, rows in paired.loc[paired.valid_pair.astype(bool)].groupby("graph"):
            for ax, metric in zip(
                axes, ["delta_population_tv", "delta_mse", "delta_native_ess_fraction"]
            ):
                for kind, sub in rows.groupby("kind"):
                    summary = sub.groupby("quantile")[metric].median()
                    ax.plot(
                        summary.index,
                        summary.values,
                        marker="o",
                        color=COLORS[graph],
                        linestyle={
                            "baseline": "-",
                            "internal": "--",
                            "random": ":",
                            "donor": "-.",
                        }[kind],
                        label=f"{LABELS[graph]} / {kind}",
                    )
                ax.axhline(0, color="black", lw=0.7)
                ax.set(
                    xscale="log",
                    xlabel="Bandwidth quantile",
                    ylabel=metric.replace("delta_", "Change in ").replace("_", " "),
                )
        axes[0].legend(fontsize=5)
        parts.append(figure(output, "paired_effects", fig))
    parts.append("<h2>Case-level population recovery and ESS control</h2>")
    for case in manifest["cases"]:
        subset = fits[fits.case == case["name"]]
        chosen = selected[selected.case == case["name"]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for graph, rows in subset[subset.family == "omc"].groupby("graph"):
            rows = rows.sort_values("quantile")
            for ax, x, y in [
                (axes[0], "quantile", "population_tv"),
                (axes[1], "native_ess_fraction", "mse"),
            ]:
                ax.plot(rows[x], rows[y], color=COLORS[graph], label=LABELS[graph])
                good, bad = rows[rows.converged], rows[~rows.converged]
                ax.scatter(good[x], good[y], s=15, color=COLORS[graph])
                ax.scatter(
                    bad[x], bad[y], s=25, facecolors="none", edgecolors=COLORS[graph]
                )
        target = (
            pops[pops.case == case["name"]]
            .groupby("state")
            .target.first()
            .reindex(range(n_states))
        )
        axes[2].plot(target.index, target.values, "k--o", label="Target")
        for _, row in chosen.iterrows():
            population = pops[
                (pops.case == case["name"])
                & (pops.graph == row.graph)
                & (pops.arm == row.arm)
            ].sort_values("state")
            axes[2].plot(
                population.state,
                population.fitted,
                marker="o",
                color=COLORS[row.graph],
                label=LABELS[row.graph],
            )
        axes[0].set(
            xscale="log",
            xlabel="Bandwidth quantile",
            ylabel="Population TV including external mass",
        )
        axes[1].set(xlabel="Conditional native ESS fraction", ylabel="All-residue MSE")
        axes[2].set(
            xlabel="Target state",
            ylabel="MSE-selected population",
            xticks=range(n_states),
        )
        axes[0].legend(fontsize=6)
        axes[2].legend(fontsize=6)
        fig.suptitle(case["name"])
        parts.append(figure(output, "case_" + case["name"], fig))
    parts += [
        "<h2>Graph geometry and energy diagnostics</h2>",
        "<p>Strong native edges are the top 10% by kernel weight (including threshold ties). "
        "Cross-state and decoy connections describe the graph, not fitting targets. The structural graph "
        "uses the geometry that also defines target clusters, so its performance is not an independent "
        "validation of a physical population prior.</p>",
    ]
    baseline = graphs[graphs.case == "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for graph, rows in baseline.groupby("graph"):
        for ax, metric in zip(
            axes,
            [
                "final_coupling",
                "mean_structural_distance_strong_edges",
                "cross_state_edge_fraction",
            ],
        ):
            ax.plot(
                rows["quantile"],
                rows[metric],
                marker="o",
                label=LABELS[graph],
                color=COLORS[graph],
            )
            ax.set(
                xscale="log",
                xlabel="Bandwidth quantile",
                ylabel=metric.replace("_", " "),
            )
    axes[0].legend(fontsize=6)
    parts.append(figure(output, "graph_geometry", fig))
    energies = e.load_npz(output / "energy.npz")["total"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot([energies[native_labels == i] for i in range(n_states)])
    axes[0].set(
        xticks=np.arange(n_states) + 1,
        xticklabels=range(n_states),
        xlabel="Target state",
        ylabel="ref2015 total (REU)",
        title="Cached unrelaxed candidate energies",
    )
    off = np.triu_indices(native_count, 1)
    axes[1].scatter(
        abs(energies[:, None] - energies[None, :])[off],
        struct_distance[off],
        s=4,
        alpha=0.25,
    )
    axes[1].set(
        xlabel="Absolute energy difference (REU)",
        ylabel="Structural distance (Å)",
        title="Similar energies can describe different structures",
    )
    parts.append(figure(output, "energy_geometry", fig))
    sensitivity = pd.read_csv(output / "population_sensitivity.csv")
    representation = pd.read_csv(output / "representation_diagnostics.csv")
    parts += [
        "<h2>Shared representation and observable information</h2>",
        "<p>These diagnostics are unchanged across graph arms. Projection weights approximate reference masses "
        "on supported representatives; their fit errors are representation diagnostics, not exact ground-truth frame weights.</p>",
        representation[representation.kind.isin(["baseline", "internal"])].to_html(
            index=False, float_format=lambda x: f"{x:.4g}"
        ),
        sensitivity[
            sensitivity.case.isin(
                ["baseline", "internal_0", "internal_1", "internal_2"]
            )
        ].to_html(index=False, float_format=lambda x: f"{x:.4g}"),
        "<h2>Unresolved and inapplicable comparisons</h2>",
        fits[~fits.converged][
            [
                "case",
                "graph",
                "quantile",
                "relative_change",
                "initialisation_objective_gap",
            ]
        ].to_html(index=False, float_format=lambda x: f"{x:.4g}"),
        "<p>Structural and energy graphs are not applicable to the ten external-profile cases. "
        "No coordinates or scores are invented for these entries.</p>",
        pd.DataFrame(exclusions).to_html(index=False),
        "<h2>Downloads</h2>",
    ]
    links = sorted(
        p.name
        for p in output.iterdir()
        if p.suffix in (".csv", ".parquet", ".json", ".npz")
    )
    parts.append(
        "<ul>"
        + "".join(
            f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>'
            for name in links
        )
        + "</ul>"
    )
    parts.append(
        "<p>All graph arrays, raw/scaled coupling, both initialisations and fitted weights are retained under cases/.</p>"
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><title>OMC graph comparison</title><style>'
        "body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#182330}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:20px 0}"
        "td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}"
        ".status{background:#edf1f7;padding:16px;font-weight:600}figure{margin:25px 0}"
        "</style></head><body>" + "\n".join(parts) + "</body></html>"
    )
    (output / "index.html").write_text(document)
    print(f"Report: {output / 'index.html'}", flush=True)
