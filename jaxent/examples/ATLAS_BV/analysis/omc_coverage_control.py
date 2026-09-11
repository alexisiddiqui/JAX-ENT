"""Paired random/structurally stratified filtering with frozen OMC bandwidths."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance

from . import omc_bandwidth_control as old
from . import omc_bandwidth_diagnostics as diag
from .common import HERE

OUTPUT = old.OUTPUT.parent / "omc_coverage_control"
SYSTEMS = ("1tzw_A", "1dd3_B")
SEEDS = tuple(range(20260909, 20260914))
RETENTIONS = (0.5, 0.25, 0.125)


def balanced_strata(distance, count):
    """Deterministic farthest-pair bisection into count near-equal strata."""
    distance = np.asarray(distance)
    if not 1 <= count <= len(distance):
        raise ValueError("invalid stratum count")

    def split(ids, m):
        if m == 1:
            return [ids]
        q, r = divmod(len(ids), m)
        ml = m // 2
        nl = ml * q + min(r, ml)
        sub = distance[np.ix_(ids, ids)]
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        projection = distance[ids, ids[a]] ** 2 - distance[ids, ids[b]] ** 2
        order = ids[np.lexsort((ids, projection))]
        return split(order[:nl], ml) + split(order[nl:], m - ml)

    return split(np.arange(len(distance)), count)


def paired_subsets(labels, distance, retention, seed):
    largest = int(np.argmax(np.bincount(labels)))
    members = np.flatnonzero(labels == largest)
    others = np.flatnonzero(labels != largest)
    count = int(np.floor(len(members) * retention))
    strata = balanced_strata(distance[np.ix_(members, members)], count)
    priorities = np.random.default_rng(seed).random(len(members))
    selections = {
        "random": np.argsort(priorities, kind="stable")[:count],
        "stratified": np.array([s[np.argmin(priorities[s])] for s in strata]),
    }
    return {
        method: np.sort(np.concatenate([others, members[take]]))
        for method, take in selections.items()
    }


def frozen_kernels(work, keep, sigmas):
    if (
        len(sigmas) != 6
        or not np.all(np.isfinite(sigmas))
        or np.any(np.asarray(sigmas) <= 0)
    ):
        raise ValueError("six positive frozen bandwidths required")
    kernels = np.zeros((13, len(work), len(work)), dtype=work.dtype)
    distance = work[np.ix_(keep, keep)]
    for i, sigma in enumerate(sigmas):
        kernels[i][np.ix_(keep, keep)] = np.exp(
            -np.minimum(0.5 * (distance / sigma) ** 2, 80)
        )
    return kernels, np.r_[sigmas, np.full(7, np.nan)]


def scientific_identity():
    return {
        str(p.relative_to(HERE)): old.digest(p)
        for p in (
            Path(__file__),
            Path(old.__file__),
            Path(diag.__file__),
            HERE / "analysis/cluster_filtering_omc.py",
            HERE / "config.yaml",
        )
    }


def load_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["code"] != scientific_identity():
        raise ValueError("Implementation changed; use a fresh output directory")
    for name, expected in manifest["input_hashes"].items():
        if old.digest(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for row in manifest["systems"]:
        if (
            old.digest(output / "systems" / row["system_id"] / "source.npz")
            != row["source_sha256"]
        ):
            raise ValueError("Copied source changed")
    return manifest


def prepare(output):
    if (output / "manifest.json").exists():
        return load_manifest(output)
    output.mkdir(parents=True, exist_ok=True)
    previous = old.load_manifest(old.OUTPUT)
    rows = []
    audits = []
    controls = []
    input_hashes = {
        str(old.OUTPUT / "manifest.json"): old.digest(old.OUTPUT / "manifest.json")
    }
    for sid in SYSTEMS:
        row = next(r for r in previous["systems"] if r["system_id"] == sid).copy()
        source_path = old.OUTPUT / "systems" / sid / "source.npz"
        folder = output / "systems" / sid
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, folder / "source.npz")
        with np.load(source_path) as data:
            source = {k: data[k] for k in data}
        structure = diag.load_structure(sid, source, folder)
        # Pair distances are invariant to alignment; use all Cα pairs, no uptake.
        vectors = np.stack([pdist(frame) for frame in structure["xyz"]])
        distance = squareform(pdist(vectors)) / np.sqrt(vectors.shape[1])
        labels = source["labels"]
        largest = int(np.argmax(np.bincount(labels)))
        members = np.flatnonzero(labels == largest)
        truth = np.bincount(labels) / len(labels)
        x = source["flat"].astype(float)
        cases = []
        for retention in RETENTIONS:
            legacy = (
                old.OUTPUT / "systems" / sid / f"retain_{retention:g}" / "fits.parquet"
            )
            input_hashes[str(legacy)] = old.digest(legacy)
            legacy_fits = pd.read_parquet(legacy)
            sigmas = (
                legacy_fits[legacy_fits.family == "omc"]
                .sort_values("quantile")
                .sigma.tolist()
            )
            for seed in SEEDS:
                subsets = paired_subsets(labels, distance, retention, seed)
                for method, keep in subsets.items():
                    name = f"{method}_retain_{retention:g}_seed_{seed}"
                    cases.append(
                        dict(
                            case=name,
                            method=method,
                            retention=retention,
                            seed=seed,
                            thinned_cluster=largest,
                            keep=keep.tolist(),
                            sigmas=sigmas,
                        )
                    )
                    retained = keep[labels[keep] == largest]
                    dd = distance[np.ix_(members, members)]
                    energy_distance = (
                        2 * distance[np.ix_(retained, members)].mean()
                        - distance[np.ix_(retained, retained)].mean()
                        - dd.mean()
                    )
                    balanced, ceiling = diag.balanced_weights(labels[keep], truth)
                    coverage = x[:, retained].mean(axis=1) - x[:, members].mean(axis=1)
                    audits.append(
                        dict(
                            system_id=sid,
                            case=name,
                            method=method,
                            retention=retention,
                            seed=seed,
                            n_candidate=len(keep),
                            n_thinned_retained=len(retained),
                            structural_energy_distance=energy_distance,
                            rg_wasserstein=wasserstein_distance(
                                structure["rg"][retained], structure["rg"][members]
                            ),
                            contact_map_rms=float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            structure["contacts"][retained].mean(axis=0)
                                            - structure["contacts"][members].mean(
                                                axis=0
                                            )
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            conditional_uptake_mse=float(np.mean(coverage**2)),
                            balanced_mse=float(
                                np.mean((x[:, keep] @ balanced - x.mean(axis=1)) ** 2)
                            ),
                            correct_population_ess_ceiling=ceiling / len(keep),
                        )
                    )
        row["cases"] = cases
        rows.append(row)
        controls.append(
            dict(
                system_id=sid,
                mse=float(np.mean((x.mean(axis=1) - source["target"]) ** 2)),
                population_tv=0.0,
                ess_fraction=1.0,
                graph_energy=0.0,
                kind="analytic uniform source",
            )
        )
        input_hashes[str(source_path)] = old.digest(source_path)
        for path, expected in row["inputs"].items():
            input_hashes[str(HERE / path)] = expected
        for path in (
            HERE / f"outputs/stage1/{sid}/R1/topology.json",
            HERE / f"data/raw/{sid}/{sid}_corresp.tsv",
            HERE / f"data/raw/{sid}/{sid}_contacts.tsv",
        ):
            input_hashes[str(path)] = old.digest(path)
    audit = diag.table(output / "coverage_audit", audits)
    pairs = audit.pivot(
        index=["system_id", "retention", "seed"],
        columns="method",
        values="structural_energy_distance",
    )
    gate = (
        (pairs["stratified"] / pairs["random"])
        .groupby(["system_id", "retention"])
        .median()
        .rename("median_structural_error_ratio")
        .reset_index()
    )
    gate["passed"] = gate.median_structural_error_ratio < 1
    diag.table(output / "coverage_gate", gate)
    diag.table(output / "analytic_controls", controls)
    manifest = dict(
        code=scientific_identity(),
        systems=rows,
        input_hashes=input_hashes,
        expected_fits=780,
        coverage_passed=bool(gate.passed.all()),
        protocol=dict(
            seeds=list(SEEDS),
            retentions=list(RETENTIONS),
            quantiles=list(old.QUANTILES),
            strength=0.1,
            checkpoints=list(old.CHECKPOINTS),
            bandwidths="numerical legacy sigmas frozen by system and retention",
            selection="structural only; same frame priorities for both methods",
        ),
    )
    old.atomic_json(output / "manifest.json", manifest)
    print(gate.to_string(index=False), flush=True)
    return manifest


def run_case(output, row, case, identity):
    base = output / "systems" / row["system_id"]
    folder = base / case["case"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with np.load(base / "source.npz") as data:
            source = {k: data[k] for k in data}
        cached = old.load_complete(folder, source, case, identity)
        if cached is not None:
            return f"cached {row['system_id']} {case['case']}"
        keep = np.asarray(case["keep"])
        kernels, sigmas = frozen_kernels(source["work"], keep, case["sigmas"])
        specs = old.arms()
        mask = np.isin(np.arange(len(source["indices"])), keep)
        fit = jax.tree.map(
            np.asarray,
            old.fit_candidates(
                source["flat"],
                source["target"],
                mask,
                kernels,
                np.array([a["kind"] for a in specs]),
                np.array([a["strength"] for a in specs]),
            ),
        )
        metadata = {
            key: row[key]
            for key in ("system_id", "role", "rmsf_tercile", "cath_class", "k")
        }
        metadata.update(
            {key: value for key, value in case.items() if key not in ("keep", "sigmas")}
        )
        results, populations = old.score(source, keep, fit, sigmas, metadata)
        arrays = {k: source[k] for k in ("target", "indices", "labels")}
        arrays.update(weights=fit["weights"], candidate_indices=source["indices"][keep])
        old.verify(source, case, results, populations, arrays)
        old.atomic_npz(folder / "weights.npz", **arrays)
        old.atomic_parquet(results, folder / "fits.parquet")
        old.atomic_parquet(populations, folder / "populations.parquet")
        old.atomic_json(
            folder / "complete.json",
            dict(
                identity=identity,
                files={
                    name: old.digest(folder / name)
                    for name in ("fits.parquet", "populations.parquet", "weights.npz")
                },
            ),
        )
        return f"{row['system_id']} {case['case']}: {int(results.converged.sum())}/13 converged"


def run(output, manifest, workers):
    if not manifest["coverage_passed"]:
        raise RuntimeError(
            "Coverage manipulation failed: inspect coverage_gate.csv; no fits run"
        )
    identity = old.digest(output / "manifest.json")
    context = multiprocessing.get_context("spawn")
    slots = context.Queue()
    available = sorted(os.sched_getaffinity(0))
    count = min(workers, len(available), 60)
    for group in np.array_split(available, count):
        slots.put([int(c) for c in group[:2]])
    try:
        with ProcessPoolExecutor(
            max_workers=count,
            mp_context=context,
            initializer=old.worker_init,
            initargs=(slots,),
        ) as pool:
            jobs = [
                pool.submit(run_case, output, row, case, identity)
                for row in manifest["systems"]
                for case in row["cases"]
            ]
            for i, future in enumerate(as_completed(jobs), 1):
                print(f"[{i}/60] {future.result()}", flush=True)
    finally:
        slots.close()
        slots.join_thread()


def report(output, manifest):
    results = []
    populations = []
    decompositions = []
    residue_terms = []
    transitions = []
    identity = old.digest(output / "manifest.json")
    for row in manifest["systems"]:
        base = output / "systems" / row["system_id"]
        with np.load(base / "source.npz") as data:
            source = {k: data[k] for k in data}
        for case in row["cases"]:
            folder = base / case["case"]
            complete = old.load_complete(folder, source, case, identity)
            if complete is None:
                raise RuntimeError(f"Incomplete or invalid candidate: {folder}")
            f, p = complete
            results.append(f)
            populations.append(p)
            keep = np.asarray(case["keep"])
            labels = source["labels"]
            truth = np.bincount(labels) / len(labels)
            _, ceiling = diag.balanced_weights(labels[keep], truth)
            with np.load(folder / "weights.npz") as data:
                w = data["weights"][:, keep].astype(float)
            w /= w.sum(axis=1, keepdims=True)
            for i, fit in f.iterrows():
                residual, terms = diag.residual_decomposition(
                    source["flat"].astype(float), labels, keep, w[i]
                )
                item = dict(
                    system_id=row["system_id"],
                    case=case["case"],
                    method=case["method"],
                    retention=case["retention"],
                    seed=case["seed"],
                    arm=fit.arm,
                    family=fit.family,
                    converged=fit.converged,
                    correct_population_ess_ceiling=ceiling / len(keep),
                    above_ceiling=1 / (w[i] @ w[i]) > ceiling + 1e-7,
                    **{key: value.mean() for key, value in terms.items()},
                )
                item["cancellation_fraction"] = 1 - np.mean(residual**2) / max(
                    sum(
                        item[t] for t in ("population_sq", "coverage_sq", "reweight_sq")
                    ),
                    1e-30,
                )
                decompositions.append(item)
                for t, time in enumerate(diag.TIMES):
                    for r in range(len(residual) // 3):
                        residue_terms.append(
                            dict(
                                system_id=row["system_id"],
                                case=case["case"],
                                arm=fit.arm,
                                time=time,
                                feature_row=r,
                                residual=residual.reshape(3, -1)[t, r],
                                **{
                                    key: value.reshape(3, -1)[t, r]
                                    for key, value in terms.items()
                                },
                            )
                        )
            for i in range(5):
                transitions.append(
                    dict(
                        system_id=row["system_id"],
                        case=case["case"],
                        method=case["method"],
                        seed=case["seed"],
                        retention=case["retention"],
                        q_from=old.QUANTILES[i],
                        q_to=old.QUANTILES[i + 1],
                        both_converged=bool(
                            f.iloc[i].converged and f.iloc[i + 1].converged
                        ),
                        delta_ess_fraction=f.iloc[i + 1].ess_fraction
                        - f.iloc[i].ess_fraction,
                        **diag.inverse_ess_change(w[i], w[i + 1], labels[keep]),
                    )
                )
    f = diag.table(output / "results", pd.concat(results, ignore_index=True))
    diag.table(output / "populations", pd.concat(populations, ignore_index=True))
    diag.table(output / "residual_decomposition", decompositions)
    diag.table(output / "residue_terms", residue_terms)
    diag.table(output / "ess_transitions", transitions)
    valid = f[f.converged]
    baseline = valid[valid.family == "unregularised"][
        ["system_id", "case", "population_tv"]
    ].rename(columns={"population_tv": "baseline_tv"})
    effects = valid[valid.family == "omc"].merge(
        baseline, on=["system_id", "case"], validate="many_to_one"
    )
    effects["relative_tv"] = effects.population_tv - effects.baseline_tv
    keys = ["system_id", "retention", "seed", "quantile"]
    pairs = (
        effects.pivot(index=keys, columns="method", values="relative_tv")
        .dropna()
        .reset_index()
    )
    pairs["difference_in_relative_tv"] = pairs.stratified - pairs.random
    diag.table(output / "paired_effects", pairs)
    repetition = []
    for key, g in pairs.groupby(["system_id", "retention", "seed"]):
        repetition.append(
            dict(
                zip(["system_id", "retention", "seed"], key),
                n_bandwidths=len(g),
                complete_curve=len(g) == 6,
                median_paired_effect=g.difference_in_relative_tv.median()
                if len(g) == 6
                else np.nan,
            )
        )
    repeats = diag.table(output / "repetition_summary", repetition)
    summary = (
        repeats.groupby(["system_id", "retention"])
        .agg(
            complete_repetitions=("median_paired_effect", "count"),
            median_effect=("median_paired_effect", "median"),
            minimum_effect=("median_paired_effect", "min"),
            maximum_effect=("median_paired_effect", "max"),
        )
        .reset_index()
    )
    diag.table(output / "summary", summary)
    matched = []
    for (sid, case), g in f.groupby(["system_id", "case"]):
        me = g[(g.family == "maxent") & g.converged]
        for fit in g[g.family == "omc"].itertuples():
            for tolerance in (0.02, 0.01, 0.005):
                item = dict(
                    system_id=sid,
                    case=case,
                    method=fit.method,
                    seed=fit.seed,
                    retention=fit.retention,
                    arm=fit.arm,
                    tolerance=tolerance,
                    matched=False,
                    ess_gap=np.nan,
                    population_tv_difference=np.nan,
                    mse_difference=np.nan,
                    maxent_arm=None,
                )
                if fit.converged and len(me):
                    candidate = (
                        me.assign(gap=abs(me.ess_fraction - fit.ess_fraction))
                        .sort_values(["gap", "strength"])
                        .iloc[0]
                    )
                    item.update(
                        ess_gap=candidate.gap,
                        maxent_arm=candidate.arm,
                        matched=bool(candidate.gap <= tolerance),
                    )
                    if item["matched"]:
                        item.update(
                            population_tv_difference=fit.population_tv
                            - candidate.population_tv,
                            mse_difference=fit.mse - candidate.mse,
                        )
                matched.append(item)
    diag.table(output / "ess_matches", matched)
    make_report(output, f, pairs, summary, repeats)
    # Recheck every frozen input after fitting and reporting.
    load_manifest(output)
    old.atomic_json(
        output / "audit.json",
        dict(
            expected_fits=780,
            actual_fits=len(f),
            converged=int(f.converged.sum()),
            candidates=60,
            source_inputs_unchanged=True,
            complete_paired_repetitions=int(repeats.complete_curve.sum()),
            manifest_sha256=identity,
        ),
    )
    print(summary.to_string(index=False), flush=True)
    print(output / "index.html", flush=True)


def make_report(output, f, pairs, summary, repeats):
    coverage = pd.read_csv(output / "coverage_audit.csv")
    gate = pd.read_csv(output / "coverage_gate.csv")
    figures = []
    for sid in SYSTEMS:
        fig, axes = plt.subplots(3, 4, figsize=(17, 11), layout="constrained")
        for i, retention in enumerate(RETENTIONS):
            subset = f[
                (f.system_id == sid) & (f.retention == retention) & (f.family == "omc")
            ]
            for method, color in (("random", "tab:orange"), ("stratified", "tab:blue")):
                for j, metric in enumerate(("population_tv", "mse", "ess_fraction")):
                    for seed, g in subset[subset.method == method].groupby("seed"):
                        g = g.sort_values("quantile")
                        axes[i, j].plot(
                            g["quantile"],
                            g[metric],
                            color=color,
                            alpha=0.35,
                            label=method if seed == SEEDS[0] else None,
                        )
                        bad = g[~g.converged]
                        axes[i, j].scatter(
                            bad["quantile"], bad[metric], marker="x", color="red"
                        )
                    axes[i, j].set(
                        xscale="log",
                        xlabel="Frozen legacy bandwidth label",
                        ylabel=metric,
                        title=f"retain {retention:g}",
                    )
            for seed, g in pairs[
                (pairs.system_id == sid) & (pairs.retention == retention)
            ].groupby("seed"):
                axes[i, 3].plot(
                    g["quantile"],
                    g.difference_in_relative_tv,
                    "o-",
                    alpha=0.6,
                    label=str(seed),
                )
            axes[i, 3].axhline(0, color="black", lw=0.7)
            axes[i, 3].set(
                xscale="log",
                xlabel="Frozen legacy bandwidth label",
                ylabel="Change in OMC − unregularised TV\nstratified − random",
            )
        axes[0, 0].legend()
        fig.savefig(output / f"{sid}_fits.png", dpi=140)
        plt.close(fig)
        figures.append(f"{sid}_fits.png")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, sid in enumerate(SYSTEMS):
        for j, metric in enumerate(
            ("structural_energy_distance", "conditional_uptake_mse", "balanced_mse")
        ):
            for method, color in (("random", "tab:orange"), ("stratified", "tab:blue")):
                for seed, g in coverage[
                    (coverage.system_id == sid) & (coverage.method == method)
                ].groupby("seed"):
                    g = g.sort_values("retention")
                    axes[i, j].plot(
                        g.retention,
                        g[metric],
                        "o-",
                        color=color,
                        alpha=0.4,
                        label=method if seed == SEEDS[0] else None,
                    )
            axes[i, j].set(
                title=sid, xlabel="Retained fraction of largest basin", ylabel=metric
            )
    axes[0, 0].legend()
    fig.savefig(output / "coverage.png", dpi=140)
    plt.close(fig)
    statements = []
    for sid in SYSTEMS:
        for retention in RETENTIONS:
            c = coverage[
                (coverage.system_id == sid) & (coverage.retention == retention)
            ]
            uptake = c.pivot(
                index="seed", columns="method", values="conditional_uptake_mse"
            )
            ratio = (uptake.stratified / uptake.random).median()
            s = summary[
                (summary.system_id == sid) & (summary.retention == retention)
            ].iloc[0]
            if s.complete_repetitions < 3:
                verdict = "Too few complete paired repetitions for a stable descriptive conclusion."
            elif ratio >= 1:
                verdict = "Structural coverage improved, but the relevant uptake-mean coverage did not."
            elif s.median_effect < 0:
                verdict = "Improved coverage accompanies better relative OMC recovery; supports coverage distortion as a contributor."
            else:
                verdict = "Coverage improved without better relative OMC recovery; regularisation bias or graph limitations remain."
            statements.append(
                f"{sid}, retain {retention:g}: conditional uptake-error ratio {ratio:.3f}; median paired TV effect {s.median_effect:+.5f} across {int(s.complete_repetitions)}/5 complete repetitions. {verdict}"
            )
    text = "\n\n".join(statements)
    (output / "findings.md").write_text(
        "# Population imbalance versus within-basin coverage\n\n"
        + text
        + "\n\nNegative effects favour stratification. These are descriptive sampling repetitions in two fixed source ensembles, not independent biological replicates. All six bandwidths are reported; none is selected by population truth.\n"
    )
    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>OMC coverage control</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style></head><body><h1>Population imbalance versus within-basin coverage</h1>',
        "<p>Two fixed source ensembles; identical basin counts and numerical bandwidths across filtering methods. Five paired sampling repetitions. Negative paired TV effects favour stratification. Red crosses mark unconverged fits. Structural stratification uses no uptake or fitting outcomes.</p>",
        "<h2>Findings</h2>",
    ]
    parts.extend("<p>" + html.escape(s) + "</p>" for s in statements)
    parts.extend(
        [
            summary.to_html(index=False, float_format=lambda x: f"{x:.5g}"),
            "<h2>Coverage manipulation</h2>",
            gate.to_html(index=False),
            '<img src="coverage.png">',
        ]
    )
    parts.extend(f'<h2>{sid}</h2><img src="{sid}_fits.png">' for sid in SYSTEMS)
    parts.append("<h2>Downloadable results</h2>")
    for path in sorted(output.glob("*.csv")):
        parts.append(f'<p><a href="{path.name}">{path.stem}</a></p>')
    parts.append(
        '<p><a href="findings.md">Findings</a> · <a href="manifest.json">Frozen design</a> · <a href="audit.json">Verification</a></p><p>All-residue MSE means all eligible feature residues at three synthetic times. It is not experimental exchange. Coverage preservation is approximate. Repetitions and bandwidths are not independent biological replicates. The analytic full-source control has exact populations and uniform ESS; convergence failures remain excluded, not replaced. No ISO runs.</p></body></html>'
    )
    (output / "index.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "fit", "report", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("workers must be 1–10")
    if (
        args.output.resolve() == old.OUTPUT.resolve()
        or old.OUTPUT.resolve() in args.output.resolve().parents
    ):
        parser.error("use a separate output directory")
    manifest = (
        prepare(args.output)
        if args.phase in ("prepare", "all")
        else load_manifest(args.output)
    )
    if args.phase in ("fit", "all"):
        run(args.output, manifest, args.workers)
    if args.phase in ("report", "all"):
        report(args.output, manifest)


if __name__ == "__main__":
    main()
