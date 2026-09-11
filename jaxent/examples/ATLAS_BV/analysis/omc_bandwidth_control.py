"""Fixed-strength, multi-system OMC distribution-control experiment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import inspect
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import spearmanr

from jaxent.examples.ATLAS_BV.analysis.cluster_filtering_omc import (
    filtered_indices,
    partitions,
    trajectory_objective,
)
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
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/omc_bandwidth_control"
REFERENCE = "1yoz_B"
QUANTILES = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
MAXENT_STRENGTHS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RETENTIONS = (1.0, 0.5, 0.25, 0.125)
CHECKPOINTS = (1000, 3000, 10000)
MATCH_TOLERANCE = 0.02


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def arms():
    return (
        [
            dict(arm=f"omc_q{q:g}", family="omc", quantile=q, strength=0.1, kind=0)
            for q in QUANTILES
        ]
        + [
            dict(
                arm=f"maxent_{s:g}", family="maxent", quantile=None, strength=s, kind=2
            )
            for s in MAXENT_STRENGTHS
        ]
        + [
            dict(
                arm="unregularised",
                family="unregularised",
                quantile=None,
                strength=0.0,
                kind=2,
            )
        ]
    )


def stratified_order(rows, seed):
    """Round-robin CATH classes within each flexibility tercile, before screening."""
    output = {}
    for tercile in ("low", "middle", "high"):
        groups = {}
        for row in rows:
            if row["system_id"] != REFERENCE and row["rmsf_tercile"] == tercile:
                groups.setdefault(row["cath_class"], []).append(row)
        names = sorted(groups, key=lambda c: (stable_seed(seed, tercile, c), c))
        for name in names:
            groups[name].sort(
                key=lambda r: (stable_seed(seed, r["system_id"]), r["system_id"])
            )
        output[tercile] = [
            groups[name][index]
            for index in range(max(map(len, groups.values()), default=0))
            for name in names
            if index < len(groups[name])
        ]
    return output


def best_partition(distance, seed):
    audit, candidates = partitions(distance, seed)
    audit["selected"] = False
    eligible = audit[audit.eligible].sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    if eligible.empty:
        return audit, None
    k = int(eligible.iloc[0].k)
    audit.loc[audit.k == k, "selected"] = True
    # The global winner is necessarily the winner of its original complexity band.
    return audit, candidates[k]


def cases(labels, seed):
    largest = int(np.argmax(np.bincount(labels)))
    for fraction in RETENTIONS:
        key = "unfiltered" if fraction == 1 else f"retain_{fraction:g}"
        keep = filtered_indices(labels, (largest,), fraction, seed)
        yield dict(
            case=key, retention=fraction, thinned_cluster=largest, keep=keep.tolist()
        )


def input_paths(row):
    return [
        HERE / row["pdb_path"],
        *(HERE / p for p in row["replica_paths"].split(";")),
        *(
            HERE / f"outputs/stage1/{row['system_id']}/R{r}/features.npz"
            for r in (1, 2, 3)
        ),
        HERE
        / f"outputs/analysis/pairwise_geometry/checkpoint8_strict_conformal/parts/{row['system_id']}.pairs.parquet",
    ]


def protocol(config):
    numerical_code = "\n".join(
        inspect.getsource(f)
        for f in (
            arms,
            stratified_order,
            best_partition,
            cases,
            prepare_source,
            kernels_for,
            fit_candidates,
            trajectory_objective,
            partitions,
            filtered_indices,
            pseudo_uptake,
            bandwidth_from_quantile,
            system_data,
        )
    )
    return dict(
        version=1,
        seed=20260826,
        frame_cap=512,
        config=config,
        arms=arms(),
        retentions=list(RETENTIONS),
        checkpoints=list(CHECKPOINTS),
        match_tolerance=MATCH_TOLERANCE,
        code_sha256=hashlib.sha256(numerical_code.encode()).hexdigest(),
        systems_sha256=digest(HERE / "data/systems.csv"),
    )


def prepare_source(row, config, destination, settings):
    system = row["system_id"]
    folder = destination / "systems" / system
    folder.mkdir(parents=True, exist_ok=True)
    inputs = {str(p.relative_to(HERE)): digest(p) for p in input_paths(row)}
    identity = dict(inputs=inputs, protocol=settings)
    cache = folder / "source.npz"
    meta = folder / "source.json"
    if meta.exists() and cache.exists():
        existing = json.loads(meta.read_text())
        if existing["identity"] == identity and existing["sha256"] == digest(cache):
            with np.load(cache) as data:
                return {k: data[k] for k in data.files}, inputs
    data = system_data(row, config)
    indices, distance = data["matrices"][1]
    take = np.linspace(0, len(indices) - 1, min(512, len(indices)), dtype=int)
    indices = np.asarray(indices)[take]
    z = data["z"][:, indices]
    flat = pseudo_uptake(z).reshape(-1, len(indices))
    structural = np.asarray(distance)[np.ix_(take, take)]
    work = np.abs(z.mean(axis=0)[:, None] - z.mean(axis=0)[None, :])
    if not all(np.isfinite(x).all() for x in (flat, structural, work)):
        raise ValueError("non-finite source arrays")
    audit, labels = best_partition(structural, settings["seed"])
    atomic_parquet(audit, folder / "clustering.parquet")
    if labels is None:
        raise ValueError("no eligible structural partition")
    source = dict(
        indices=indices,
        flat=flat,
        target=flat.mean(axis=1),
        structural=structural,
        work=work,
        labels=labels,
    )
    for case in cases(labels, stable_seed(settings["seed"], system)):
        distance = work[np.ix_(case["keep"], case["keep"])]
        bandwidth_from_quantile(distance, QUANTILES[0])
    atomic_npz(cache, **source)
    atomic_json(meta, dict(identity=identity, sha256=digest(cache)))
    return source, inputs


def screen(destination):
    config = load_config()
    settings = protocol(config)
    if (destination / "manifest.json").exists():
        return load_manifest(destination)
    rows = load_systems()
    selected, audit = [], []

    def attempt(row, role):
        system = row["system_id"]
        print(
            f"[screen] {system} ({row['rmsf_tercile']}, {row['cath_class']})",
            flush=True,
        )
        try:
            source, inputs = prepare_source(row, config, destination, settings)
        except (OSError, ValueError, KeyError) as error:
            audit.append(
                dict(system_id=system, role=role, status="excluded", reason=str(error))
            )
            pd.DataFrame(audit).to_csv(destination / "screening.csv", index=False)
            return False
        folder = destination / "systems" / system
        selected.append(
            dict(
                system_id=system,
                role=role,
                rmsf_tercile=row["rmsf_tercile"],
                cath_class=row["cath_class"],
                k=int(source["labels"].max() + 1),
                n_source=len(source["indices"]),
                inputs=inputs,
                source_sha256=digest(folder / "source.npz"),
                clustering_sha256=digest(folder / "clustering.parquet"),
                cases=list(
                    cases(source["labels"], stable_seed(settings["seed"], system))
                ),
            )
        )
        audit.append(dict(system_id=system, role=role, status="selected", reason=""))
        pd.DataFrame(audit).to_csv(destination / "screening.csv", index=False)
        return True

    destination.mkdir(parents=True, exist_ok=True)
    for tercile, ordered in stratified_order(rows, settings["seed"]).items():
        count = 0
        for row in ordered:
            count += int(attempt(row, "cohort"))
            if count == 4:
                break
        if count != 4:
            raise RuntimeError(
                f"Only {count} eligible systems in {tercile}; no fits started"
            )
    if not attempt(next(r for r in rows if r["system_id"] == REFERENCE), "calibration"):
        raise RuntimeError("Calibration source ineligible; inspect screening.csv")
    manifest = dict(
        protocol=settings, systems=selected, expected_fits=len(selected) * 4 * 13
    )
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def load_manifest(destination):
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest["protocol"] != protocol(load_config()):
        raise ValueError("Protocol changed; use a fresh output directory")
    for row in manifest["systems"]:
        folder = destination / "systems" / row["system_id"]
        for name, expected in (
            ("source.npz", row["source_sha256"]),
            ("clustering.parquet", row["clustering_sha256"]),
        ):
            if digest(folder / name) != expected:
                raise ValueError(f"Frozen artifact changed: {folder / name}")
        for name, expected in row["inputs"].items():
            if digest(HERE / name) != expected:
                raise ValueError(f"Source input changed: {name}")
    return manifest


def kernels_for(work, keep):
    distance = work[np.ix_(keep, keep)]
    kernels, sigmas = [], []
    for arm in arms():
        kernel = np.zeros_like(work)
        sigma = np.nan
        if arm["quantile"] is not None:
            sigma = bandwidth_from_quantile(distance, arm["quantile"])
            kernel[np.ix_(keep, keep)] = np.exp(
                -np.minimum((distance / sigma) ** 2 / 2, 80)
            )
        kernels.append(kernel)
        sigmas.append(sigma)
    return np.asarray(kernels), np.asarray(sigmas)


@partial(jax.jit, static_argnames=("checkpoints", "window"))
def fit_candidates(
    values, target, mask, kernels, kinds, strengths, checkpoints=CHECKPOINTS, window=250
):
    """Thirteen independent trajectories with uninterrupted Adam continuation."""
    if (
        not checkpoints
        or window < 1
        or any(b - a < window for a, b in zip((0, *checkpoints), checkpoints))
    ):
        raise ValueError("Each checkpoint interval must contain the convergence window")

    def losses(x):
        return jax.vmap(trajectory_objective, in_axes=(0, 0, 0, 0, None, None, None))(
            x, strengths, kernels, kinds, values, target, mask
        )

    optimizer = optax.adam(0.05)
    logits = jnp.zeros((len(kinds), len(mask)))
    state = optimizer.init(logits)
    active = jnp.ones(len(kinds), bool)
    before = losses(logits)
    steps = jnp.zeros(len(kinds), int)

    def advance(count, carry, enabled):
        def step(_, xs):
            x, s = xs
            gradient = jax.grad(lambda z: losses(z).sum())(x)
            updates, next_state = optimizer.update(gradient, s, x)
            return jnp.where(
                enabled[:, None], optax.apply_updates(x, updates), x
            ), next_state

        return jax.lax.fori_loop(0, count, step, carry)

    previous = 0
    for checkpoint in checkpoints:

        def extend(carry):
            x, s = advance(checkpoint - previous - window, carry, active)
            earlier = losses(x)
            x, s = advance(window, (x, s), active)
            return x, s, earlier

        logits, state, earlier = jax.lax.cond(
            jnp.any(active), extend, lambda carry: (*carry, before), (logits, state)
        )
        before = jnp.where(active, earlier, before)
        steps = jnp.where(active, checkpoint, steps)
        after = losses(logits)
        relative = jnp.abs(after - before) / jnp.maximum(jnp.abs(after), 1e-12)
        active = active & ((relative > 0.01) | ~jnp.isfinite(relative))
        previous = checkpoint
    return dict(
        weights=jax.nn.softmax(jnp.where(mask, logits, -jnp.inf)),
        objective=after,
        relative_change=relative,
        absolute_change=jnp.abs(after - before),
        steps=steps,
        grad_norm=jnp.linalg.norm(jax.grad(lambda z: losses(z).sum())(logits), axis=-1),
    )


def score(source, keep, fit, sigmas, metadata):
    labels = source["labels"]
    k = int(labels.max() + 1)
    target_mass = np.bincount(labels, minlength=k) / len(labels)
    candidate_mass = np.bincount(labels[keep], minlength=k) / len(keep)
    records, populations = [], []
    for index, arm in enumerate(arms()):
        w = fit["weights"][index].astype(float)
        mass = np.bincount(labels, weights=w, minlength=k)
        conditional = []
        for cluster in range(k):
            inside = w[labels == cluster]
            count = int(np.sum(labels[keep] == cluster))
            ess = float(inside.sum() ** 2 / max(np.dot(inside, inside), 1e-30))
            conditional.append(ess / count)
            populations.append(
                dict(
                    **metadata,
                    arm=arm["arm"],
                    family=arm["family"],
                    cluster=cluster,
                    target_population=float(target_mass[cluster]),
                    candidate_population=float(candidate_mass[cluster]),
                    recovered_population=float(mass[cluster]),
                    conditional_ess=ess,
                    conditional_ess_fraction=ess / count,
                    n_candidate_cluster=count,
                )
            )
        records.append(
            dict(
                **metadata,
                **arm,
                sigma=float(sigmas[index]),
                n_candidate=len(keep),
                mse=float(np.mean((source["flat"] @ w - source["target"]) ** 2)),
                population_tv=float(np.abs(mass - target_mass).sum() / 2),
                candidate_population_tv=float(
                    np.abs(candidate_mass - target_mass).sum() / 2
                ),
                ess=float(1 / np.dot(w, w)),
                ess_fraction=float(1 / np.dot(w, w) / len(keep)),
                minimum_conditional_ess_fraction=min(conditional),
                structural_dispersion=float(w @ source["structural"] @ w),
                objective=float(fit["objective"][index]),
                relative_objective_change=float(fit["relative_change"][index]),
                absolute_objective_change=float(fit["absolute_change"][index]),
                final_grad_norm=float(fit["grad_norm"][index]),
                steps=int(fit["steps"][index]),
                converged=bool(fit["relative_change"][index] <= 0.01),
            )
        )
    return pd.DataFrame(records), pd.DataFrame(populations)


def verify(source, case, results, populations, saved):
    """Reconstruct scientific metrics from saved weights, independent of scoring."""
    weights = np.asarray(saved["weights"], dtype=float)
    keep = np.asarray(case["keep"])
    assert weights.shape == (13, len(source["indices"]))
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
    assert (weights[:, np.setdiff1d(np.arange(weights.shape[1]), keep)] == 0).all()
    for name in ("target", "indices", "labels"):
        np.testing.assert_array_equal(saved[name], source[name])
    np.testing.assert_array_equal(saved["candidate_indices"], source["indices"][keep])
    np.testing.assert_allclose(
        source["target"], source["flat"].mean(axis=1), atol=0, rtol=0
    )
    assert results.arm.tolist() == [a["arm"] for a in arms()]
    numeric = results.select_dtypes(include=[np.number]).drop(
        columns=["quantile", "sigma"]
    )
    assert np.isfinite(numeric.to_numpy()).all()
    np.testing.assert_allclose(
        results.mse,
        np.mean((weights @ source["flat"].T - source["target"]) ** 2, axis=1),
        atol=1e-14,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        results.ess, 1 / np.square(weights).sum(axis=1), rtol=1e-7
    )
    labels = source["labels"]
    k = int(labels.max() + 1)
    mass = np.stack([weights[:, labels == c].sum(axis=1) for c in range(k)], axis=1)
    target = np.bincount(labels, minlength=k) / len(labels)
    np.testing.assert_allclose(
        results.population_tv, np.abs(mass - target).sum(axis=1) / 2, atol=1e-10
    )
    assert len(populations) == 13 * k
    for i, arm in enumerate(arms()):
        p = populations[populations.arm == arm["arm"]].sort_values("cluster")
        np.testing.assert_allclose(p.recovered_population, mass[i], atol=1e-10)
        np.testing.assert_allclose(p.target_population, target, atol=1e-10)


def load_complete(folder, source, case, identity):
    receipt = folder / "complete.json"
    if not receipt.exists():
        return None
    try:
        payload = json.loads(receipt.read_text())
        if payload["identity"] != identity:
            return None
        for name in ("fits.parquet", "populations.parquet", "weights.npz"):
            if digest(folder / name) != payload["files"][name]:
                return None
        results = pd.read_parquet(folder / "fits.parquet")
        populations = pd.read_parquet(folder / "populations.parquet")
        with np.load(folder / "weights.npz") as saved:
            verify(source, case, results, populations, saved)
        return results, populations
    except (OSError, ValueError, KeyError, AssertionError):
        return None


def worker_init(cpu_slots):
    """Bound each spawned worker's native thread pools to its allocated CPUs."""
    assigned = cpu_slots.get()
    # Restrict existing import-time threads as well as future JAX threads.
    for thread in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(thread.name), assigned)
        except ProcessLookupError:
            pass


def run_case(destination, row, case, identity):
    specs = arms()
    base = destination / "systems" / row["system_id"]
    folder = base / case["case"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with np.load(base / "source.npz") as saved:
            source = {key: saved[key] for key in saved.files}
        if load_complete(folder, source, case, identity) is not None:
            print(f"[cached] {row['system_id']} {case['case']}", flush=True)
            return "cached"
        keep = np.asarray(case["keep"])
        mask = np.isin(np.arange(len(source["indices"])), keep)
        kernels, sigmas = kernels_for(source["work"], keep)
        print(
            f"[fit pid={os.getpid()}] {row['system_id']} {case['case']} n={len(keep)}",
            flush=True,
        )
        fit = jax.tree.map(
            np.asarray,
            fit_candidates(
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
        metadata.update({key: value for key, value in case.items() if key != "keep"})
        results, populations = score(source, keep, fit, sigmas, metadata)
        arrays = {key: source[key] for key in ("target", "indices", "labels")}
        arrays.update(weights=fit["weights"], candidate_indices=source["indices"][keep])
        verify(source, case, results, populations, arrays)
        atomic_npz(folder / "weights.npz", **arrays)
        atomic_parquet(results, folder / "fits.parquet")
        atomic_parquet(populations, folder / "populations.parquet")
        atomic_json(
            folder / "complete.json",
            dict(
                identity=identity,
                files={
                    name: digest(folder / name)
                    for name in ("fits.parquet", "populations.parquet", "weights.npz")
                },
            ),
        )
        print(
            f"[done] {row['system_id']} {case['case']}: {int(results.converged.sum())}/13 converged; steps={results.steps.value_counts().to_dict()}",
            flush=True,
        )
        return "computed"


def run(destination, manifest, workers=1):
    identity = digest(destination / "manifest.json")
    jobs = []
    for row in manifest["systems"]:
        base = destination / "systems" / row["system_id"]
        with np.load(base / "source.npz") as saved:
            source = {key: saved[key] for key in saved.files}
        for case in row["cases"]:
            folder = base / case["case"]
            if load_complete(folder, source, case, identity) is not None:
                continue
            jobs.append((destination, row, case, identity))
    if not jobs:
        print("All candidates already complete and validated", flush=True)
        return
    available = sorted(os.sched_getaffinity(0))
    count = min(workers, len(jobs), len(available))
    print(f"[run] {len(jobs)} candidates remaining; {count} workers", flush=True)
    if count == 1:
        for job in jobs:
            run_case(*job)
        return
    context = multiprocessing.get_context("spawn")
    slots = context.Queue()
    for group in np.array_split(available, count):
        slots.put([int(cpu) for cpu in group[:4]])
    try:
        with ProcessPoolExecutor(
            max_workers=count,
            mp_context=context,
            initializer=worker_init,
            initargs=(slots,),
        ) as pool:
            futures = [pool.submit(run_case, *job) for job in jobs]
            for future in as_completed(futures):
                future.result()
    finally:
        slots.close()
        slots.join_thread()


def ess_matches(results):
    records = []
    for (_, _), table in results.groupby(["system_id", "case"]):
        maxent = table[(table.family == "maxent") & table.converged]
        for omc in table[table.family == "omc"].itertuples():
            record = dict(
                system_id=omc.system_id,
                role=omc.role,
                case=omc.case,
                retention=omc.retention,
                omc_arm=omc.arm,
                quantile=omc.quantile,
                omc_ess_fraction=omc.ess_fraction,
                omc_population_tv=omc.population_tv,
                omc_mse=omc.mse,
                matched=False,
                nearest_maxent_arm=None,
                ess_mismatch=np.nan,
                population_tv_difference=np.nan,
                mse_difference=np.nan,
            )
            if not omc.converged:
                record["status"] = "omc_unconverged"
            elif maxent.empty:
                record["status"] = "no_converged_maxent"
            else:
                choices = maxent.assign(
                    mismatch=abs(maxent.ess_fraction - omc.ess_fraction)
                )
                tied = np.isclose(
                    choices.mismatch, choices.mismatch.min(), atol=1e-12, rtol=0
                )
                nearest = choices[tied].sort_values("strength").iloc[0]
                matched = bool(nearest.mismatch <= MATCH_TOLERANCE + 1e-12)
                record.update(
                    nearest_maxent_arm=nearest.arm,
                    ess_mismatch=float(nearest.mismatch),
                    maxent_strength=float(nearest.strength),
                    maxent_ess_fraction=float(nearest.ess_fraction),
                    maxent_population_tv=float(nearest.population_tv),
                    maxent_mse=float(nearest.mse),
                    matched=matched,
                    status="matched" if matched else "outside_tolerance",
                )
                if matched:
                    record.update(
                        population_tv_difference=float(
                            omc.population_tv - nearest.population_tv
                        ),
                        mse_difference=float(omc.mse - nearest.mse),
                    )
            records.append(record)
    return pd.DataFrame(records)


def case_summaries(results):
    rows = []
    for (system, case), table in results.groupby(["system_id", "case"]):
        omc = table[(table.family == "omc") & table.converged].sort_values("quantile")
        baseline = table[table.family == "unregularised"].iloc[0]
        complete = len(omc) == 6
        rows.append(
            dict(
                system_id=system,
                role=baseline.role,
                case=case,
                retention=float(baseline.retention),
                converged_bandwidths=len(omc),
                complete_bandwidth_curve=complete,
                ess_span=float(omc.ess_fraction.max() - omc.ess_fraction.min())
                if complete
                else np.nan,
                bandwidth_ess_spearman=float(
                    spearmanr(omc["quantile"], omc.ess_fraction).statistic
                )
                if complete and omc.ess_fraction.nunique() > 1
                else np.nan,
                median_population_tv_change=float(
                    omc.population_tv.median() - baseline.population_tv
                )
                if complete and baseline.converged
                else np.nan,
                median_mse_change=float(omc.mse.median() - baseline.mse)
                if complete and baseline.converged
                else np.nan,
            )
        )
    return pd.DataFrame(rows)


def plot_system(table, populations, folder):
    metrics = [
        ("ess_fraction", "ESS / candidate N"),
        ("population_tv", "Population TV error"),
        ("mse", "All-residue MSE"),
        ("minimum_conditional_ess_fraction", "Minimum within-cluster ESS fraction"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), squeeze=False)
    for row_index, fraction in enumerate(RETENTIONS[1:]):
        case = table[table.retention == fraction]
        omc = case[case.family == "omc"].sort_values("quantile")
        baseline = case[case.family == "unregularised"].iloc[0]
        for ax, (metric, label) in zip(axes[row_index], metrics):
            ax.plot(omc["quantile"], omc[metric], "o-", label="OMC strength 0.1")
            bad = omc[~omc.converged]
            ax.scatter(
                bad["quantile"],
                bad[metric],
                marker="x",
                s=100,
                color="red",
                label="Unconverged",
            )
            ax.axhline(
                baseline[metric],
                color="grey",
                ls="--",
                label="Unregularised"
                + (" (unconverged)" if not baseline.converged else ""),
            )
            ax.set(
                xscale="log",
                xlabel="Bandwidth quantile",
                ylabel=label,
                title=f"Largest basin retained: {fraction:g}",
            )
            if metric == "mse":
                ax.set_yscale("symlog", linthresh=1e-10)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{table.system_id.iloc[0]} · k={table.k.iloc[0]} · {table.role.iloc[0]}"
    )
    fig.tight_layout()
    fig.savefig(folder / "bandwidth.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
    for row_index, fraction in enumerate(RETENTIONS[1:]):
        case = table[table.retention == fraction]
        for ax, metric in zip(axes[row_index, :2], ("population_tv", "mse")):
            for family, marker in (("omc", "o"), ("maxent", "s")):
                points = case[case.family == family].sort_values(
                    "quantile" if family == "omc" else "strength"
                )
                ax.plot(
                    points.ess_fraction, points[metric], marker=marker, label=family
                )
                for point in points.itertuples():
                    setting = point.quantile if family == "omc" else point.strength
                    ax.annotate(
                        f"{setting:g}",
                        (point.ess_fraction, getattr(point, metric)),
                        fontsize=6,
                    )
                bad = points[~points.converged]
                ax.scatter(
                    bad.ess_fraction, bad[metric], marker="x", s=100, color="red"
                )
            baseline = case[case.family == "unregularised"]
            ax.scatter(
                baseline.ess_fraction,
                baseline[metric],
                marker="*",
                color="black",
                label="Unregularised",
            )
            unresolved_baseline = baseline[~baseline.converged]
            ax.scatter(
                unresolved_baseline.ess_fraction,
                unresolved_baseline[metric],
                marker="x",
                color="red",
                s=100,
            )
            ax.set(
                xlabel="ESS / candidate N",
                ylabel="Population TV error"
                if metric == "population_tv"
                else "All-residue MSE",
                title=f"Retention {fraction:g}",
            )
            ax.grid(alpha=0.2)
        p = populations[
            (populations.retention == fraction) & (populations.family == "omc")
        ]
        ax = axes[row_index, 2]
        for cluster, group in p.groupby("cluster"):
            group = group.merge(
                case[["arm", "quantile", "converged"]], on="arm"
            ).sort_values("quantile")
            (line,) = ax.plot(
                group["quantile"],
                group.recovered_population,
                "o-",
                label=f"Cluster {cluster}",
            )
            ax.axhline(
                group.target_population.iloc[0],
                color=line.get_color(),
                ls="--",
                alpha=0.6,
            )
            ax.axhline(
                group.candidate_population.iloc[0],
                color=line.get_color(),
                ls=":",
                alpha=0.6,
            )
            bad = group[~group.converged]
            ax.scatter(
                bad["quantile"], bad.recovered_population, marker="x", color="red"
            )
        ax.set(
            xscale="log",
            xlabel="Bandwidth quantile",
            ylabel="Cluster population",
            title="Dashed target; dotted candidate",
        )
        ax.legend(fontsize=7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{table.system_id.iloc[0]} · All six settings; red crosses are unconverged"
    )
    fig.tight_layout()
    fig.savefig(folder / "recovery.png", dpi=150)
    plt.close(fig)


def report(destination, manifest):
    identity = digest(destination / "manifest.json")
    fits, populations, missing = [], [], []
    for row in manifest["systems"]:
        base = destination / "systems" / row["system_id"]
        with np.load(base / "source.npz") as saved:
            source = {key: saved[key] for key in saved.files}
        for case in row["cases"]:
            loaded = load_complete(base / case["case"], source, case, identity)
            if loaded is None:
                missing.append(f"{row['system_id']}/{case['case']}")
            else:
                fits.append(loaded[0])
                populations.append(loaded[1])
    if not fits:
        raise ValueError("No complete, validated fits available")
    results, population_table = (
        pd.concat(fits, ignore_index=True),
        pd.concat(populations, ignore_index=True),
    )
    matches = ess_matches(results)
    summaries = case_summaries(results)
    metrics = [
        "ess_span",
        "bandwidth_ess_spearman",
        "median_population_tv_change",
        "median_mse_change",
    ]
    systems = (
        summaries[summaries.retention < 1]
        .groupby(["system_id", "role"])[metrics]
        .median()
        .reset_index()
    )
    match_systems = (
        matches[(matches.retention < 1) & matches.matched]
        .groupby(["system_id", "role", "case"])[
            ["population_tv_difference", "mse_difference"]
        ]
        .median()
        .groupby(["system_id", "role"])
        .median()
        .reset_index()
    )
    systems = systems.merge(match_systems, on=["system_id", "role"], how="left")
    coverage = (
        summaries[summaries.retention < 1]
        .groupby(["system_id", "role"])
        .agg(
            completed_cases=("case", "size"),
            complete_bandwidth_curves=("complete_bandwidth_curve", "sum"),
            baseline_comparison_cases=("median_population_tv_change", "count"),
        )
        .reset_index()
    )
    match_coverage = (
        matches[(matches.retention < 1) & matches.matched]
        .groupby(["system_id", "role"])
        .agg(
            matched_points=("omc_arm", "size"),
            matched_cases=("case", "nunique"),
        )
        .reset_index()
    )
    systems = systems.merge(coverage, on=["system_id", "role"], how="left").merge(
        match_coverage, on=["system_id", "role"], how="left"
    )
    systems[["matched_points", "matched_cases"]] = (
        systems[["matched_points", "matched_cases"]].fillna(0).astype(int)
    )
    systems = systems.merge(
        results[["system_id", "k", "rmsf_tercile", "cath_class"]].drop_duplicates(),
        on="system_id",
        validate="one_to_one",
    )
    for name, table in (
        ("results", results),
        ("populations", population_table),
        ("ess_matches", matches),
        ("case_summary", summaries),
        ("system_summary", systems),
    ):
        atomic_parquet(table, destination / f"{name}.parquet")
        table.to_csv(destination / f"{name}.csv", index=False)
    biased = results[(results.role == "cohort") & (results.retention < 1)]
    eligible_matches = matches[(matches.role == "cohort") & (matches.retention < 1)]
    cohort_systems = systems[systems.role == "cohort"]
    summary = dict(
        complete_grid=not missing,
        expected_fits=manifest["expected_fits"],
        fits=len(results),
        missing=missing,
        cohort_systems=int(results[results.role == "cohort"].system_id.nunique()),
        selected_cluster_counts=sorted(
            results[results.role == "cohort"].k.unique().tolist()
        ),
        unconverged_fits=int((~results.converged).sum()),
        unconverged_biased_cohort_fits=int((~biased.converged).sum()),
        matched_biased_cohort_points=int(eligible_matches.matched.sum()),
        total_biased_cohort_omc_points=len(eligible_matches),
        complete_biased_cohort_curves=int(
            summaries[
                (summaries.role == "cohort") & (summaries.retention < 1)
            ].complete_bandwidth_curve.sum()
        ),
        system_medians={
            key: (
                float(cohort_systems[key].median())
                if cohort_systems[key].notna().any()
                else None
            )
            for key in [*metrics, "population_tv_difference", "mse_difference"]
        },
        systems_contributing={
            key: int(cohort_systems[key].notna().sum())
            for key in [*metrics, "population_tv_difference", "mse_difference"]
        },
    )
    atomic_yaml(destination / "report.yaml", summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for _, table in systems.groupby("role"):
        axes[0].scatter(
            table.ess_span, table.median_population_tv_change, label=table.role.iloc[0]
        )
        axes[1].scatter(
            table.population_tv_difference,
            table.mse_difference,
            label=table.role.iloc[0],
        )
        for row in table.itertuples():
            if np.isfinite(row.ess_span) and np.isfinite(
                row.median_population_tv_change
            ):
                axes[0].annotate(
                    row.system_id,
                    (row.ess_span, row.median_population_tv_change),
                    fontsize=7,
                )
            if np.isfinite(row.population_tv_difference) and np.isfinite(
                row.mse_difference
            ):
                axes[1].annotate(
                    row.system_id,
                    (row.population_tv_difference, row.mse_difference),
                    fontsize=7,
                )
    axes[0].set(
        xlabel="Median bandwidth ESS-fraction range",
        ylabel="Population TV change vs unregularised",
    )
    axes[1].set(
        xlabel="Population TV: OMC − ESS-matched MaxEnt",
        ylabel="MSE: OMC − ESS-matched MaxEnt",
    )
    for ax in axes:
        ax.axhline(0, color="grey", ls="--")
        ax.axvline(0, color="grey", ls="--")
        ax.margins(x=0.15, y=0.12)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination / "cohort_summary.png", dpi=160)
    plt.close(fig)
    page = [
        "<!doctype html><meta charset='utf-8'><title>OMC bandwidth control</title>",
        "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>",
        "<h1>Fixed-strength OMC bandwidth control</h1>",
        "<p>Original OMC strength 0.1; six bandwidths versus six fixed MaxEnt strengths and an unregularised baseline. Targets are frozen full-source averages. All-residue MSE throughout.</p>",
        f"<p><b>{len(results)}/{manifest['expected_fits']} fits validated; {summary['unconverged_fits']} unconverged.</b> Matched biased cohort points: {summary['matched_biased_cohort_points']}/{summary['total_biased_cohort_omc_points']}. ESS matching tolerance: 0.02 of candidate N; no interpolation or additional tuning.</p>",
        "<p>Headline comparisons exclude unresolved fits. ESS-range summaries require all six bandwidths to converge; baseline comparisons also require a converged baseline. Medians aggregate retention levels within systems before systems. The calibration reference is separate. Matches can reuse a MaxEnt setting; unmatched points remain visible in the full curves.</p>",
        "<p>Greater ESS is distribution control, not proof of population recovery. Read population error and MSE together. Synthetic targets come from the same source ensemble before filtering.</p>",
        '<img src="cohort_summary.png" alt="System-level ESS control and matched recovery comparisons">',
        '<p><a href="results.csv">All fit metrics</a> · <a href="populations.csv">All cluster populations</a> · <a href="ess_matches.csv">ESS matches and exclusions</a> · <a href="case_summary.csv">Case summaries</a> · <a href="system_summary.csv">System summaries</a> · <a href="screening.csv">Screening audit</a> · <a href="manifest.json">Frozen protocol and cohort</a></p>',
        '<div style="overflow-x:auto">'
        + systems[
            [
                "system_id",
                "role",
                "k",
                "ess_span",
                "bandwidth_ess_spearman",
                "median_population_tv_change",
                "matched_points",
                "population_tv_difference",
                "mse_difference",
            ]
        ]
        .rename(
            columns={
                "system_id": "System",
                "role": "Role",
                "ess_span": "ESS span",
                "bandwidth_ess_spearman": "Bandwidth–ESS rho",
                "median_population_tv_change": "TV change vs baseline",
                "matched_points": "Matched points / 18",
                "population_tv_difference": "TV vs matched MaxEnt",
                "mse_difference": "MSE vs matched MaxEnt",
            }
        )
        .to_html(index=False, float_format=lambda x: f"{x:.5g}")
        + "</div>",
        "<h2>Completion and denominators</h2><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>",
    ]
    for row in manifest["systems"]:
        system = row["system_id"]
        table = results[results.system_id == system]
        if len(table) != 52:
            page.append(
                f"<h2>{system}</h2><p>Incomplete; missing candidates listed above.</p>"
            )
            continue
        base = destination / "systems" / system
        plot_system(table, population_table[population_table.system_id == system], base)
        audit = pd.read_parquet(base / "clustering.parquet")
        page.extend(
            [
                f"<h2>{system} · {row['role']} · k={row['k']}</h2>",
                f"<p>{html.escape(row['cath_class'])}; RMSF {row['rmsf_tercile']}. Unfiltered control is included in downloadable tables.</p>",
                f'<img src="systems/{system}/bandwidth.png" alt="Bandwidth control for {system}">',
                f'<img src="systems/{system}/recovery.png" alt="Population recovery and comparator curves for {system}">',
                "<details><summary>Silhouette selection</summary>"
                + audit.to_html(index=False)
                + "</details>",
            ]
        )
    (destination / "index.html").write_text("\n".join(page))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "run", "report"), default="run")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.phase == "screen":
        screen(args.output)
    elif args.phase == "run":
        manifest = screen(args.output)
        run(args.output, manifest, workers=args.workers)
        report(args.output, manifest)
    else:
        report(args.output, load_manifest(args.output))


if __name__ == "__main__":
    main()
