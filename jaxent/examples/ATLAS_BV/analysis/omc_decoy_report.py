"""Physical preflight, graph compatibility and decoy recovery report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from . import omc_decoy_control as e


def table(output, name, rows, columns=None):
    data = (
        rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    )
    data.to_csv(output / f"{name}.csv", index=False)
    data.to_parquet(output / f"{name}.parquet", index=False)
    return data


def figure(output, name, fig):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(folder / f"{name}.{suffix}", dpi=150)
    plt.close(fig)
    return f'<figure><img src="figures/{name}.png" alt="{html.escape(name.replace("_", " "))}"><figcaption><a href="figures/{name}.svg">Download SVG</a></figcaption></figure>'


def report(output):
    output = Path(output)
    manifest = e.load_manifest(output)
    source = e.load_npz(output / "source.npz")
    audit = json.loads((output / "feature_audit.json").read_text())
    preflight = json.loads((output / "preflight.json").read_text())
    labels = source["target_labels"]
    native_labels = source["native_labels"]
    n_states = int(labels.max()) + 1
    parts = ["<h1>OMC clustered-candidate decoy experiment: 1tzw_A</h1>"]
    if not preflight["passed"]:
        parts += [
            '<p class="status">Physical preflight blocked fitting. No population-recovery comparison was run.</p>',
            "<ul>"
            + "".join(
                f"<li>{html.escape(reason)}</li>" for reason in preflight["reasons"]
            )
            + "</ul>",
            "<p>This is a limitation of the configured features-to-uptake experiment. It is not evidence that OMC cannot recover populations.</p>",
        ]
    else:
        parts += [
            '<p class="status">Physical preflight passed. Completed and unresolved fits are distinguished below.</p>'
        ]
    parts += [
        "<p>100 fixed cluster representatives; independent silhouette-selected target states; empirical target populations. "
        "Contact switch: full contribution inside the radius and a rational tail with 0.5 Å scale. BV coefficients preserved. Targets and fits use mean log-PF followed by uptake, "
        "recipient intrinsic rates in seconds⁻¹, pD 7, and 300 K. No noise or peptide aggregation.</p>"
    ]
    metrics = table(
        output,
        "physical_summary",
        [
            dict(
                contact_cache_matches=audit["recipient"]["contact_passed"],
                donor_contact_cache_matches=audit["donor"]["contact_passed"],
                intrinsic_rates_match=audit["recipient"]["rates_passed"],
                donor_intrinsic_rates_match=audit["donor"]["rates_passed"],
                target_states=n_states,
                candidate_count=100,
                prepared_cases=len(manifest["cases"]),
                uptake_max=preflight["uptake_max"],
                uptake_min=preflight["uptake_min"],
                informative_observations=preflight[
                    "informative_reference_observations"
                ],
                physical_gate_passed=preflight["passed"],
            )
        ],
    )
    parts += [
        "<h2>Physical forward-model checks</h2>",
        metrics.T.to_html(header=False, float_format=lambda x: f"{x:.4g}"),
    ]
    contact = table(
        output,
        "contact_cache_checks",
        [r for a in audit.values() for r in a["contacts"]],
    )
    parts.append(contact.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
    observations = []
    for i, name in enumerate(preflight["target_names"]):
        values = source["target_predictions"][i]
        for ti, time in enumerate(e.TIMES):
            for r, residue in enumerate(source["residues"]):
                observations.append(
                    dict(
                        target=name,
                        time_seconds=time,
                        residue=int(residue),
                        uptake=values[ti, r],
                        delta_from_reference=values[ti, r]
                        - source["target_predictions"][0, ti, r],
                    )
                )
    table(output, "target_uptake", observations)
    signal = table(
        output,
        "target_signal",
        [
            dict(
                target=name,
                max_abs_uptake_change=value,
                distinguishable_above_floor=value > e.SIGNAL_FLOOR,
            )
            for name, value in zip(
                preflight["target_names"], preflight["target_max_abs_change"]
            )
        ],
    )
    parts.append(signal.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, name in enumerate(preflight["target_names"]):
        prediction = source["target_predictions"][i]
        curve = (
            prediction.mean(axis=1) if preflight["passed"] else prediction.max(axis=1)
        )
        axes[0].plot(e.TIMES, np.maximum(curve, 1e-300), marker="o", label=name)
    if preflight["passed"]:
        axes[0].set(ylim=(0, 1), ylabel="Mean residue uptake")
    else:
        axes[0].axhline(
            e.SIGNAL_FLOOR, color="black", ls="--", label="absolute signal floor"
        )
        axes[0].set(yscale="log", ylabel="Largest residue uptake")
    axes[0].set(
        xscale="log", xlabel="Exposure (seconds)", title="Absolute physical uptake"
    )
    axes[0].legend(fontsize=7)
    im = axes[1].imshow(
        np.log10(np.maximum(source["target_predictions"][0], 1e-300)),
        aspect="auto",
        origin="lower",
    )
    axes[1].set(
        xlabel="Eligible residue index",
        ylabel="Exposure (seconds)",
        title="Full-reference log₁₀ uptake",
        yticks=range(5),
        yticklabels=e.TIMES.astype(int),
    )
    fig.colorbar(im, ax=axes[1])
    parts.append(figure(output, "physical_uptake", fig))
    contact_config = preflight["contact_method"]
    mean_logpf = source["log_pf"].mean(axis=1)
    log_time_one_percent = (
        np.log(-np.log(0.99)) + mean_logpf - np.log(source["rates"])
    ) / np.log(10)
    table(
        output,
        "residue_scale",
        [
            dict(
                residue=int(r),
                mean_log_pf=mean_logpf[i],
                mean_heavy_contribution=float(
                    contact_config["bv_bc"] * source["heavy"][i].mean()
                ),
                mean_acceptor_contribution=float(
                    contact_config["bv_bh"] * source["acceptor"][i].mean()
                ),
                intrinsic_rate_s=float(source["rates"][i]),
                log10_seconds_to_one_percent=float(log_time_one_percent[i]),
            )
            for i, r in enumerate(source["residues"])
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(
        [
            contact_config["bv_bc"] * source["heavy"].ravel(),
            contact_config["bv_bh"] * source["acceptor"].ravel(),
        ],
        bins=40,
        label=["Heavy-contact contribution", "Acceptor-contact contribution"],
    )
    axes[0].set(
        xlabel="Contribution to log-PF",
        ylabel="Residue–frame count",
        title="Absolute protection scale",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(source["residues"], log_time_one_percent)
    axes[1].axhline(
        np.log10(e.TIMES.max()),
        color="black",
        ls="--",
        label="Longest planned exposure",
    )
    axes[1].set(
        xlabel="Simulated residue",
        ylabel="log₁₀ seconds",
        title="Time to 1% uptake at reference mean log-PF",
    )
    axes[1].legend(fontsize=8)
    parts.append(figure(output, "protection_scale", fig))
    parts.append(
        "<p>The contact components and time-to-uptake calculation show the absolute protection scale. "
        "Contact-feature agreement tests implementation consistency; it does not validate a physical "
        "parameterisation. The former residue-centred pseudo-uptake construction removed this absolute scale. "
        "Only contact construction changed: this experiment uses the agreed 0.5 Å smooth cutoff. No centring, coefficient adjustment or exposure-time adjustment is applied.</p>"
    )

    parts.append("<h2>Candidate compression and independent target states</h2>")
    clustering = pd.read_csv(output / "target_clustering.csv")
    parts.append(clustering.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    vectors = np.stack([pdist(frame) for frame in source["xyz"]])
    embedding = PCA(
        n_components=2, svd_solver="randomized", random_state=e.SEEDS[0]
    ).fit_transform(vectors)
    mixing = pd.read_csv(output / "candidate_target_mixing.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(*embedding.T, c=labels, cmap="tab10", s=9, alpha=0.5)
    axes[0].scatter(
        *embedding[source["candidate_indices"]].T,
        facecolors="none",
        edgecolors="black",
        s=35,
        label="100 representatives",
    )
    axes[0].set(
        xlabel="Structural PC1",
        ylabel="Structural PC2",
        title="Reference states and candidate representatives",
    )
    axes[0].legend(fontsize=8)
    im = axes[1].imshow(
        mixing.values / mixing.values.sum(axis=1, keepdims=True),
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[1].set(
        xlabel="Independent target state",
        ylabel="Candidate cluster",
        title="Target-state composition of candidate clusters",
    )
    fig.colorbar(im, ax=axes[1], label="Conditional frame fraction")
    parts.append(figure(output, "candidate_target_clustering", fig))
    table(
        output,
        "representatives",
        [
            dict(
                candidate=i,
                source_frame=int(f),
                target_state=int(native_labels[i]),
                parent_cluster_count=int((source["candidate_labels"] == i).sum()),
                parent_cluster_purity=float(
                    mixing.values[i].max() / mixing.values[i].sum()
                ),
            )
            for i, f in enumerate(source["candidate_frames"])
        ],
    )

    (
        diagnostics,
        graph_rows,
        population_rows,
        fit_rows,
        residual_rows,
        sensitivity_rows,
    ) = [], [], [], [], [], []
    for case in manifest["cases"]:
        data = e.load_npz(output / "cases" / case["name"] / "input.npz")
        projection = data["projection"]
        target_masses = data["target_masses"]
        projected_masses = np.bincount(
            native_labels, weights=projection[:100], minlength=n_states
        )
        pred = e.uptake(data["log_pf"], data["rates"], projection)
        uniform = np.ones(len(projection)) / len(projection)
        occupancies = np.bincount(
            source["candidate_labels"], weights=data["target_weights"], minlength=100
        )
        occupancy_weights = np.pad(occupancies, (0, len(projection) - 100))
        occupancy_prediction = e.uptake(
            data["log_pf"], data["rates"], occupancy_weights
        )
        diagnostics.append(
            dict(
                case=case["name"],
                kind=case["kind"],
                informative=case["informative"],
                candidate_count=len(projection),
                removed_state=case["removed"],
                projection_mse=float(np.mean((pred - data["target"]) ** 2)),
                occupancy_projection_mse=float(
                    np.mean((occupancy_prediction - data["target"]) ** 2)
                ),
                projection_population_tv=float(
                    0.5 * abs(projected_masses - target_masses).sum()
                ),
                mean_nearest_structural_distance=float(
                    data["target_weights"] @ source["structural_distance"].min(axis=1)
                ),
                initial_decoy_mass=float(uniform[data["decoy"]].sum()),
                projection_decoy_mass=float(projection[data["decoy"]].sum()),
            )
        )
        states = np.flatnonzero(target_masses > 0)
        means = np.stack(
            [source["log_pf"][:, labels == state].mean(axis=1) for state in states],
            axis=1,
        )
        m = means @ target_masses[states]
        exposure = e.TIMES[:, None] * data["rates"][None] * np.exp(-m)
        # Derivative for probability transfers against the final retained state.
        jacobian = (-exposure * np.exp(-exposure))[:, :, None] * (
            means[:, :-1] - means[:, -1:]
        )[None]
        singular = np.linalg.svd(
            jacobian.reshape(-1, len(states) - 1), compute_uv=False
        )
        for i, value in enumerate(singular):
            sensitivity_rows.append(
                dict(
                    case=case["name"],
                    direction=i,
                    singular_value=float(value),
                    above_absolute_floor=bool(value > e.SIGNAL_FLOOR),
                )
            )
        for state in range(n_states):
            population_rows.append(
                dict(
                    case=case["name"],
                    method="target",
                    arm=-1,
                    state=state,
                    mass=float(target_masses[state]),
                )
            )
            population_rows.append(
                dict(
                    case=case["name"],
                    method="reference_projection",
                    arm=-1,
                    state=state,
                    mass=float(projected_masses[state]),
                )
            )
        off = ~np.eye(len(projection), dtype=bool)
        decoy = data["decoy"]
        cross = decoy[:, None] != decoy[None, :]
        for i, kernel in enumerate(data["kernels"]):
            graph_rows.append(
                dict(
                    case=case["name"],
                    kind=case["kind"],
                    quantile=float(e.QUANTILES[i]),
                    sigma=float(data["sigmas"][i]),
                    mean_offdiagonal_coupling=float(kernel[off].mean()),
                    cross_decoy_mean_coupling=float(kernel[cross].mean())
                    if cross.any()
                    else 0.0,
                    cross_decoy_edge_fraction=float(
                        kernel[cross].sum() / kernel[off].sum()
                    ),
                    projected_weight_penalty=e.regularisation(projection, kernel),
                    projected_weight_loss=0.1 * e.regularisation(projection, kernel),
                )
            )
        folder = output / "cases" / case["name"]
        if not (folder / "complete.json").exists():
            continue
        complete = json.loads((folder / "complete.json").read_text())
        if complete["identity"] != e.digest(output / "manifest.json") or complete[
            "sha256"
        ] != e.digest(folder / "fit.npz"):
            raise ValueError(f"Fit provenance mismatch: {case['name']}")
        if not manifest["fit_gate_passed"] or not case["informative"]:
            raise ValueError("Unexpected fit for a case blocked by physical preflight")
        fit = e.load_npz(folder / "fit.npz")
        for arm, spec in enumerate(e.arm_specs()):
            w = fit["weights"][arm]
            predicted = e.uptake(data["log_pf"], data["rates"], w)
            masses = np.bincount(native_labels, weights=w[:100], minlength=n_states)
            external_mass = w[100:].sum()
            retained = target_masses > 0
            retained_mass = masses[retained].sum()
            conditional_tv = (
                0.5
                * abs(masses[retained] / retained_mass - target_masses[retained]).sum()
                if retained_mass > 0
                else 1.0
            )
            active = np.flatnonzero(w[:100] >= 1e-4)
            coverage = (
                float(
                    data["target_weights"]
                    @ source["structural_distance"][:, active].min(axis=1)
                )
                if len(active)
                else None
            )
            fit_rows.append(
                dict(
                    case=case["name"],
                    kind=case["kind"],
                    arm=arm,
                    **spec,
                    converged=bool(fit["converged"][arm]),
                    mse=float(np.mean((predicted - data["target"]) ** 2)),
                    population_tv=float(
                        0.5 * (abs(masses - target_masses).sum() + external_mass)
                    ),
                    retained_population_tv=float(conditional_tv),
                    decoy_mass=float(w[decoy].sum()),
                    retained_state_coverage=float((masses[retained] >= 1e-4).mean()),
                    active_native_coverage_distance=coverage,
                    **e.ess_stats(w, 100),
                    objective=float(fit["objective"][arm]),
                    gradient_norm=float(fit["grad_norm"][arm]),
                    relative_change=float(fit["relative_change"][arm]),
                    steps=int(fit["steps"][arm]),
                    initialisation_objective_gap=float(
                        fit["initialisation_objective_gap"][arm]
                    ),
                    initialisation_weight_tv=float(
                        fit["initialisation_weight_tv"][arm]
                    ),
                )
            )
            for state in range(n_states):
                population_rows.append(
                    dict(
                        case=case["name"],
                        method=spec["family"],
                        arm=arm,
                        state=state,
                        mass=float(masses[state]),
                    )
                )
            for ti, time in enumerate(e.TIMES):
                for ri, residue in enumerate(source["residues"]):
                    residual_rows.append(
                        dict(
                            case=case["name"],
                            arm=arm,
                            residue=int(residue),
                            time_seconds=float(time),
                            target=float(data["target"][ti, ri]),
                            predicted=float(predicted[ti, ri]),
                            residual=float(predicted[ti, ri] - data["target"][ti, ri]),
                        )
                    )
    diag = table(output, "representation_diagnostics", diagnostics)
    graphs = table(output, "graph_diagnostics", graph_rows)
    populations = table(output, "populations", population_rows)
    fits = table(
        output,
        "fits",
        fit_rows,
        None
        if fit_rows
        else ["case", "family", "converged", "mse", "population_tv", "decoy_mass"],
    )
    table(
        output,
        "residuals",
        residual_rows,
        None
        if residual_rows
        else [
            "case",
            "arm",
            "residue",
            "time_seconds",
            "target",
            "predicted",
            "residual",
        ],
    )
    sensitivity = table(output, "population_sensitivity", sensitivity_rows)
    parts += [
        diag.to_html(index=False, float_format=lambda x: f"{x:.3g}"),
        "<p>Occupancy weights preserve source-to-candidate cluster assignments. The supported-reference projection "
        "assigns retained source mass to its nearest target-supported representative. Both are diagnostic quadratures, "
        "not exact ground-truth frame weights. Their residual errors measure compression effects.</p>",
        "<h2>Graph compatibility and observable population information</h2>",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Repeated external sets are shown individually, not treated as independent systems.
    for case, group in graphs.groupby("case", sort=False):
        axes[0].plot(group["quantile"], group["mean_offdiagonal_coupling"], alpha=0.6)
        axes[1].plot(group["quantile"], group["projected_weight_penalty"], alpha=0.6)
    axes[0].set(
        xscale="log",
        xlabel="Bandwidth quantile",
        ylabel="Mean off-diagonal coupling",
        title="Coupling across prepared challenges",
    )
    axes[1].set(
        xscale="log",
        xlabel="Bandwidth quantile",
        ylabel="OMC penalty of reference projection",
        title="Compatibility of projected weights",
    )
    for case, group in sensitivity.groupby("case", sort=False):
        if case.startswith(("internal", "baseline")):
            axes[2].plot(
                group["direction"],
                np.maximum(group["singular_value"], 1e-300),
                marker="o",
                label=case,
            )
    axes[2].axhline(e.SIGNAL_FLOOR, color="black", ls="--")
    axes[2].set(
        yscale="log",
        xlabel="Population-transfer direction",
        ylabel="Jacobian singular value",
        title="Absolute local HDX sensitivity",
    )
    axes[2].legend(fontsize=7)
    parts.append(figure(output, "graph_and_information", fig))
    parts.append(
        "<p>The graph retains the previous scalar distance: absolute difference in mean residue log-PF. "
        "Similar values in this graph do not establish structural equivalence. External profiles have no recipient "
        "coordinates, so structural coverage is evaluated on native representatives only. A large penalty for the "
        "projected target weights identifies tension with the smoothing prior; it is not an unconditional failure "
        "of decoy rejection. Absolute HDX sensitivities identify cases where the observables cannot resolve population transfers.</p>"
    )
    parts.append("<h2>Target populations before fitting</h2>")
    target_cases = [
        c["name"] for c in manifest["cases"] if c["kind"] in {"baseline", "internal"}
    ]
    target_table = populations[
        (populations.method == "target") & populations.case.isin(target_cases)
    ]
    target_matrix = target_table.pivot(
        index="case", columns="state", values="mass"
    ).reindex(target_cases)
    parts.append(target_matrix.to_html(float_format=lambda x: f"{x:.4f}"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    target_matrix.plot.bar(stacked=True, ax=axes[0], colormap="tab10", rot=25)
    axes[0].set(
        xlabel="Target challenge",
        ylabel="Population",
        title="Empirical targets after state removal",
    )
    truth = target_matrix.loc["baseline"].to_numpy()
    prior = np.bincount(native_labels, minlength=n_states) / len(native_labels)
    projection_mass = populations[
        (populations.case == "baseline")
        & (populations.method == "reference_projection")
    ].mass.to_numpy()
    for offset, (name, values) in enumerate(
        [
            ("Empirical target", truth),
            ("Uniform candidate", prior),
            ("Reference projection", projection_mass),
        ]
    ):
        axes[1].bar(
            np.arange(n_states) + (offset - 1) * 0.25, values, width=0.25, label=name
        )
    axes[1].set(
        xlabel="Target state",
        ylabel="Population",
        title="Compression changes the uniform prior",
        xticks=range(n_states),
    )
    axes[1].legend(fontsize=8)
    parts.append(figure(output, "target_population_challenges", fig))
    parts.append("<h2>Population recovery and distribution control</h2>")
    if fits.empty:
        parts.append(
            "<p>No fit results are available. The tables and figures above describe the prepared experiment and its "
            "physical gate, not measured OMC or MaxEnt performance. No bandwidth or regularisation verdict is assigned.</p>"
        )
    else:
        parts.append(
            f"<p>Completed cases: {fits.case.nunique()} of {len(manifest['cases'])}. "
            f"{len(fits)} fitted arms; {int(fits.converged.sum())} passed convergence and initialisation checks. "
            "Only converged arms enter selected comparisons. Full curves include unresolved arms with open markers.</p>"
        )
        selected = []
        for case, group in fits[fits.converged].groupby("case"):
            for family, arms in group.groupby("family"):
                selected.append(arms.sort_values(["mse", "arm"]).iloc[0].to_dict())
        selection = table(output, "mse_selected", selected)
        parts.append(selection.to_html(index=False, float_format=lambda x: f"{x:.4g}"))
        matches = []
        for case, group in fits[fits.converged].groupby("case"):
            maxent = group[group.family == "maxent"]
            for _, omc in group[group.family == "omc"].iterrows():
                if maxent.empty:
                    continue
                delta = abs(maxent.native_ess_fraction - omc.native_ess_fraction)
                other = maxent.loc[delta.idxmin()]
                matches.append(
                    dict(
                        case=case,
                        omc_arm=int(omc.arm),
                        maxent_arm=int(other.arm),
                        native_ess_fraction_gap=float(delta.min()),
                        within_2_percentage_points=bool(delta.min() <= 0.02),
                        population_tv_difference=float(
                            omc.population_tv - other.population_tv
                        ),
                        mse_difference=float(omc.mse - other.mse),
                    )
                )
        table(output, "nearest_ess_comparisons", matches)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for case_index, (case, group) in enumerate(
            fits[fits.family == "omc"].groupby("case")
        ):
            color = plt.get_cmap("tab20")(case_index % 20)
            group = group.sort_values("quantile")
            for ax, x, y in [
                (axes[0], "quantile", "native_ess_fraction"),
                (axes[1], "native_ess_fraction", "population_tv"),
                (axes[2], "mse", "decoy_mass"),
            ]:
                ax.plot(group[x], group[y], alpha=0.75, color=color, label=case)
                good = group[group.converged]
                bad = group[~group.converged]
                ax.scatter(good[x], good[y], s=12, color=color)
                ax.scatter(bad[x], bad[y], s=25, facecolors="none", edgecolors="black")
        axes[0].set(
            xscale="log",
            xlabel="Bandwidth quantile",
            ylabel="Conditional native ESS fraction",
        )
        axes[0].legend(fontsize=5, ncol=2)
        axes[1].set(
            xlabel="Conditional native ESS fraction",
            ylabel="Population TV including external mass",
        )
        axes[2].set(xlabel="All-residue MSE", ylabel="Decoy mass")
        parts.append(figure(output, "recovery_tradeoffs", fig))
        # Selected per-state plots retain the actual target masses, including zero states.
        for case, group in selection.groupby("case"):
            fig, ax = plt.subplots(figsize=(8, 3))
            wanted = populations[
                (populations.case == case) & (populations.method == "target")
            ]
            ax.bar(np.arange(n_states) - 0.3, wanted.mass, width=0.2, label="Target")
            for offset, (_, row) in enumerate(group.iterrows()):
                subset = populations[
                    (populations.case == case) & (populations.arm == row.arm)
                ]
                ax.bar(
                    np.arange(n_states) - 0.1 + 0.2 * offset,
                    subset.mass,
                    width=0.2,
                    label=row.family,
                )
            ax.set(
                xlabel="Target state",
                ylabel="Population",
                title=case,
                xticks=range(n_states),
            )
            ax.legend(fontsize=8)
            parts.append(figure(output, f"populations_{case}", fig))
    parts.append(
        "<h2>Files and reproducibility</h2><p>Random and donor challenges each have five reproducible sets of 25 "
        "profiles appended to the same 100 native representatives. Donor profiles are randomly cropped or tiled; "
        "recipient intrinsic rates are retained. Sets are repeated perturbations of one system, not independent "
        "biological replicates. No ISO run is included.</p>"
    )
    links = sorted(
        p.name
        for p in output.iterdir()
        if p.suffix in {".csv", ".parquet", ".json", ".npz"}
    )
    parts.append(
        "<ul>"
        + "".join(
            f'<li><a href="{html.escape(p)}">{html.escape(p)}</a></li>' for p in links
        )
        + "</ul>"
    )
    parts.append(
        "<p>Per-case inputs, decoy provenance and any fitted weights are stored under <code>cases/</code>.</p>"
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><title>OMC decoy experiment</title><style>'
        "body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#182330}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:13px;margin:20px 0}"
        "td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}"
        ".status{background:#edf1f7;padding:16px;font-weight:600}figure{margin:25px 0}"
        "</style></head><body>" + "\n".join(parts) + "</body></html>"
    )
    (output / "index.html").write_text(document)
    print(f"Report: {output / 'index.html'}", flush=True)
