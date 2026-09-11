"""Coupling-matched profile, structural and ref2015 OMC graphs on frozen decoy cases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.distance import pdist, squareform

from . import omc_decoy_control as e

OUTPUT = e.OUTPUT.with_name("omc_graph_control")
GRAPHS = ("profile_logpf", "structure", "ref2015_total")
ENERGY = (
    e.HERE
    / "outputs/analysis/pairwise_geometry/checkpoint24_pyrosetta_energy/energies/1tzw_A"
)


def profile_distance(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite residue-by-frame profiles required")
    return squareform(pdist(values.T)) / np.sqrt(values.shape[0])


def structural_distance(xyz):
    vectors = np.stack([pdist(frame) for frame in np.asarray(xyz)])
    return squareform(pdist(vectors)) / np.sqrt(vectors.shape[1])


def energy_join(archive, requested_frames):
    frames = np.asarray(archive["frame"])
    values = np.asarray(archive["ref2015__total"], float)
    requested_frames = np.asarray(requested_frames)
    if (
        frames.ndim != 1
        or values.shape != frames.shape
        or len(np.unique(frames)) != len(frames)
    ):
        raise ValueError("Energy frames must be unique and aligned with scores")
    if not np.isfinite(values).all() or not np.array_equal(frames, frames.astype(int)):
        raise ValueError("Nonfinite scores or noninteger frame identifiers")
    if len(np.unique(requested_frames)) != len(requested_frames):
        raise ValueError("Requested frames must be unique")
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    if not set(requested_frames).issubset(lookup):
        raise ValueError("Missing candidate energy frames")
    return values[[lookup[int(frame)] for frame in requested_frames]]


def bandwidths(distance, native_count=100):
    native = np.asarray(distance)[:native_count, :native_count]
    pairs = native[np.triu_indices(len(native), 1)]
    pairs = pairs[pairs > 0]
    if not len(pairs) or not np.isfinite(pairs).all():
        raise ValueError("Graph has no finite positive native distances")
    return np.maximum(np.quantile(pairs, e.QUANTILES), np.finfo(float).eps)


def matched_kernels(distance, sigmas, reference):
    d = np.asarray(distance, float)
    sigmas = np.asarray(sigmas, float)
    reference = np.asarray(reference, float)
    if (
        d.ndim != 2
        or d.shape[0] != d.shape[1]
        or len(d) < 2
        or not np.isfinite(d).all()
        or (d < 0).any()
        or not np.allclose(d, d.T)
        or not np.allclose(np.diag(d), 0)
        or sigmas.shape != (6,)
        or not np.isfinite(sigmas).all()
        or (sigmas <= 0).any()
        or reference.shape != (6, *d.shape)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Invalid graph distances, bandwidths or reference kernels")
    raw = np.exp(-np.minimum(0.5 * (d[None] / sigmas[:, None, None]) ** 2, 80))
    off = ~np.eye(len(d), dtype=bool)
    original = raw[:, off].mean(axis=1)
    target = reference[:, off].mean(axis=1)
    if (original <= 0).any() or (target <= 0).any():
        raise ValueError("Degenerate coupling")
    factors = target / original
    kernels = raw.copy()
    kernels[:, off] *= factors[:, None]
    if not np.isfinite(kernels).all():
        raise ValueError("Nonfinite scaled kernels")
    np.testing.assert_allclose(
        kernels[:, off].mean(axis=1), target, rtol=1e-12, atol=1e-15
    )
    return dict(
        kernels=kernels,
        raw_kernels=raw,
        raw_coupling=original,
        target_coupling=target,
        scaling_factor=factors,
        sigmas=sigmas,
        distance=d,
    )


def scientific_identity():
    return {str(Path(__file__)): e.digest(__file__), **e.code_identity()}


def load_manifest(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["code"] != scientific_identity():
        raise ValueError(
            "Scientific implementation changed; use a fresh output directory"
        )
    for name, value in manifest["inputs"].items():
        if e.digest(name) != value:
            raise ValueError(f"Frozen input changed: {name}")
    for name, value in manifest["artifacts"].items():
        if e.digest(output / name) != value:
            raise ValueError(f"Prepared artifact changed: {name}")
    return manifest


def applicable(graph, kind):
    return graph == "profile_logpf" or kind in ("baseline", "internal")


def prepare(output, previous=e.OUTPUT):
    output, previous = Path(output), Path(previous)
    if (output / "manifest.json").exists():
        return load_manifest(output)
    old = e.load_manifest(previous)
    if not old["fit_gate_passed"]:
        raise ValueError("Source experiment failed physical preflight")
    source = e.load_npz(previous / "source.npz")
    scores = e.load_npz(ENERGY / "1tzw_A_R1.energies.npz")
    score_manifest = json.loads((ENERGY / "manifest.json").read_text())
    score_entry = next(r for r in score_manifest["replicas"] if r["replica"] == 1)
    if score_entry["sha256"] != e.digest(ENERGY / "1tzw_A_R1.energies.npz"):
        raise ValueError("Energy archive disagrees with scoring provenance")
    energy = energy_join(scores, source["candidate_frames"])
    core = source["log_pf"][:, source["candidate_indices"]]
    distances = dict(
        profile_logpf=profile_distance(core),
        structure=structural_distance(source["xyz"][source["candidate_indices"]]),
        ref2015_total=abs(energy[:, None] - energy[None, :]),
    )
    sigmas = {name: bandwidths(distance) for name, distance in distances.items()}
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "source.npz",
        "preflight.json",
        "population_sensitivity.csv",
        "representation_diagnostics.csv",
    ):
        shutil.copyfile(previous / name, output / name)
    e.save_npz(output / "energy.npz", frame=source["candidate_frames"], total=energy)
    inputs = {
        str(previous / "manifest.json"): e.digest(previous / "manifest.json"),
        **{
            str(ENERGY / name): e.digest(ENERGY / name)
            for name in ("manifest.json", "1tzw_A_R1.energies.npz")
        },
    }
    # Snapshot all reused data and fits, without relying on mutable report tables.
    jobs, exclusions, cases = [], [], []
    for case in old["cases"]:
        name = case["name"]
        folder = output / "cases" / name
        folder.mkdir(parents=True, exist_ok=True)
        complete = json.loads((previous / "cases" / name / "complete.json").read_text())
        if complete["identity"] != e.digest(previous / "manifest.json") or complete[
            "sha256"
        ] != e.digest(previous / "cases" / name / "fit.npz"):
            raise ValueError(f"Invalid source fit identity: {name}")
        for filename in ("input.npz", "fit.npz", "complete.json"):
            path = previous / "cases" / name / filename
            inputs[str(path)] = e.digest(path)
            shutil.copyfile(
                path,
                folder
                / ("reused_" + filename if filename != "input.npz" else filename),
            )
        data = e.load_npz(folder / "input.npz")
        if not case["informative"]:
            raise ValueError(f"Unexpected uninformative frozen case: {name}")
        for graph in GRAPHS:
            if not applicable(graph, case["kind"]):
                exclusions.append(
                    dict(
                        case=name,
                        graph=graph,
                        status="not_applicable",
                        reason="Synthetic profiles lack recipient coordinates and energies",
                    )
                )
                continue
            distance = (
                profile_distance(data["log_pf"])
                if graph == "profile_logpf"
                else distances[graph]
            )
            values = matched_kernels(distance, sigmas[graph], data["kernels"])
            e.save_npz(folder / graph / "graph.npz", **values)
            jobs.append(dict(case=name, graph=graph))
        cases.append(case)
    for name in (
        "source.npz",
        "preflight.json",
        "population_sensitivity.csv",
        "representation_diagnostics.csv",
    ):
        inputs[str(previous / name)] = e.digest(previous / name)
    e.write_json(output / "not_applicable.json", exclusions)
    artifacts = {
        str(p.relative_to(output)): e.digest(p)
        for p in output.rglob("*")
        if p.is_file()
    }
    manifest = dict(
        version=1,
        previous=str(previous),
        system="1tzw_A",
        cases=cases,
        jobs=jobs,
        graphs=list(GRAPHS),
        new_arms=len(jobs) * 6,
        reused_arms=len(cases) * 13,
        initialisations=2,
        coupling="match_scalar_per_case_and_quantile_offdiagonal",
        strength=0.1,
        quantiles=e.QUANTILES.tolist(),
        checkpoints=list(e.CHECKPOINTS),
        energy_score="ref2015__total",
        score_coordinates="unrelaxed, cached trajectory coordinates",
        code=scientific_identity(),
        inputs=inputs,
        artifacts=artifacts,
    )
    e.write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(jobs)} graph/case jobs; {len(jobs) * 6} new arms", flush=True)
    return manifest


def fit_graph(data, checkpoints=e.CHECKPOINTS, window=250):
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)
    count = data["log_pf"].shape[1]
    kernels = data["kernels"]
    kinds = np.zeros(6, int)
    strengths = np.full(6, 0.1)
    if not checkpoints or any(
        b - a < window for a, b in zip((0, *checkpoints), checkpoints)
    ):
        raise ValueError("Checkpoint intervals must contain the convergence window")

    def losses(logits):
        return jax.vmap(
            lambda x, k, s, kind: e.jax_objective(
                x,
                jnp.asarray(data["log_pf"]),
                jnp.asarray(data["rates"]),
                jnp.asarray(data["target"]),
                jnp.asarray(data["times"]),
                data["scale"],
                k,
                s,
                kind,
            )
        )(logits, jnp.asarray(kernels), jnp.asarray(strengths), jnp.asarray(kinds))

    optimiser = optax.adam(0.05)
    results = []
    for initial in [
        np.zeros((6, count)),
        np.random.default_rng(e.SEEDS[0]).normal(0, 0.01, (6, count)),
    ]:
        x = jnp.asarray(initial)
        state = optimiser.init(x)
        active = np.ones(6, bool)
        steps = np.zeros(6, int)

        @jax.jit
        def advance(x, state, enabled, count_steps):
            def step(_, carry):
                x, state = carry
                updates, state = optimiser.update(
                    jax.grad(lambda z: losses(z).sum())(x), state, x
                )
                return jnp.where(
                    enabled[:, None], optax.apply_updates(x, updates), x
                ), state

            return jax.lax.fori_loop(0, count_steps, step, (x, state))

        previous = 0
        relative = np.full(6, np.inf)
        for checkpoint in checkpoints:
            x, state = advance(
                x, state, jnp.asarray(active), checkpoint - previous - window
            )
            before = np.asarray(losses(x))
            x, state = advance(x, state, jnp.asarray(active), window)
            after = np.asarray(losses(x))
            relative = np.where(
                active, abs(after - before) / np.maximum(abs(after), 1e-12), relative
            )
            steps[active] = checkpoint
            active &= (relative > 0.01) | ~np.isfinite(relative)
            previous = checkpoint
        results.append(
            dict(
                weights=np.asarray(jax.nn.softmax(x)),
                objective=np.asarray(losses(x)),
                relative_change=relative,
                steps=steps,
                grad_norm=np.asarray(
                    jnp.linalg.norm(jax.grad(lambda z: losses(z).sum())(x), axis=1)
                ),
            )
        )
    objectives = np.stack([r["objective"] for r in results])
    best = np.argmin(objectives, axis=0)
    selected = {
        key: np.stack([r[key] for r in results])[best, np.arange(6)]
        for key in results[0]
    }
    selected["initialisation_objectives"] = objectives
    selected["initialisation_weights"] = np.stack([r["weights"] for r in results])
    selected["initialisation_relative_change"] = np.stack(
        [r["relative_change"] for r in results]
    )
    selected["initialisation_objective_gap"] = abs(
        objectives[0] - objectives[1]
    ) / np.maximum(abs(selected["objective"]), 1e-12)
    selected["initialisation_weight_tv"] = 0.5 * abs(
        results[0]["weights"] - results[1]["weights"]
    ).sum(axis=1)
    selected["converged"] = (
        (selected["relative_change"] <= 0.01)
        & (selected["initialisation_objective_gap"] <= 0.01)
        & (selected["initialisation_relative_change"] <= 0.01).all(axis=0)
        & np.isfinite(selected["weights"]).all(axis=1)
        & np.isfinite(selected["objective"])
        & np.isfinite(selected["grad_norm"])
    )
    return selected


def validate_fit(fit, count):
    if (
        fit["weights"].shape != (6, count)
        or not np.isfinite(fit["weights"]).all()
        or (fit["weights"] < 0).any()
    ):
        raise ValueError("Invalid fitted weights")
    np.testing.assert_allclose(fit["weights"].sum(axis=1), 1, atol=1e-12)
    if not np.isfinite(fit["objective"]).all():
        raise ValueError("Nonfinite fitting objective")


def verify_baseline(output):
    identity = e.digest(output / "manifest.json")
    marker = output / "adapter_verification.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if (
            old.get("passed")
            and old["identity"] == identity
            and old["sha256"] == e.digest(output / "adapter_scalar_fit.npz")
        ):
            return old
    data = e.load_npz(output / "cases/baseline/input.npz")
    previous = e.load_npz(output / "cases/baseline/reused_fit.npz")
    fit = fit_graph(data)
    validate_fit(fit, data["log_pf"].shape[1])
    # Changing the XLA batch size changes rounding near the uniform solution.
    # Bound the resulting probability-mass discrepancy, as well as the objective.
    weight_tv = 0.5 * abs(fit["weights"] - previous["weights"][:6]).sum(axis=1)
    np.testing.assert_allclose(
        fit["weights"], previous["weights"][:6], rtol=0, atol=1e-5
    )
    if np.max(weight_tv) > 1e-5:
        raise ValueError("Scalar baseline probability-mass discrepancy exceeds 1e-5")
    np.testing.assert_array_equal(fit["steps"], previous["steps"][:6])
    np.testing.assert_allclose(
        fit["objective"], previous["objective"][:6], rtol=1e-7, atol=1e-11
    )
    np.testing.assert_array_equal(fit["converged"], previous["converged"][:6])
    e.save_npz(output / "adapter_scalar_fit.npz", **fit)
    result = dict(
        passed=True,
        identity=identity,
        sha256=e.digest(output / "adapter_scalar_fit.npz"),
        max_weight_error=float(abs(fit["weights"] - previous["weights"][:6]).max()),
        max_weight_tv=float(weight_tv.max()),
        weight_tv_tolerance=1e-5,
        objective_relative_tolerance=1e-7,
        max_objective_error=float(
            abs(fit["objective"] - previous["objective"][:6]).max()
        ),
    )
    e.write_json(marker, result)
    print("Scalar baseline adapter parity passed", flush=True)
    return result


def run_job(args):
    root, job, identity = args
    root = Path(root)
    folder = root / "cases" / job["case"] / job["graph"]
    marker = folder / "complete.json"
    with (folder / ".fit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            old = json.loads(marker.read_text())
            if old["identity"] == identity and old["sha256"] == e.digest(
                folder / "fit.npz"
            ):
                return {**job, "status": "resumed"}
        data = e.load_npz(root / "cases" / job["case"] / "input.npz")
        data["kernels"] = e.load_npz(folder / "graph.npz")["kernels"]
        try:
            fit = fit_graph(data)
            validate_fit(fit, data["log_pf"].shape[1])
        except (FloatingPointError, ValueError, AssertionError) as error:
            e.write_json(
                folder / "failure.json",
                dict(
                    identity=identity,
                    **job,
                    status="numerical_failure",
                    error=str(error),
                ),
            )
            return {**job, "status": "numerical_failure"}
        e.save_npz(folder / "fit.npz", **fit)
        e.write_json(
            marker, dict(identity=identity, sha256=e.digest(folder / "fit.npz"))
        )
    return {**job, "status": "complete"}


def run(output, workers=10, smoke=False):
    manifest = load_manifest(output)
    os.environ.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_PLATFORMS="cpu",
    )
    verify_baseline(output)
    identity = e.digest(output / "manifest.json")
    # Finish all baseline smoke arms before any remaining scientific jobs.
    baseline = [j for j in manifest["jobs"] if j["case"] == "baseline"]
    outcomes = []
    for job in baseline:
        result = run_job((str(output), job, identity))
        outcomes.append(result)
        print(result, flush=True)
    if any(r["status"] == "numerical_failure" for r in outcomes):
        e.write_json(
            output / "run_status.json", dict(status="blocked_smoke", outcomes=outcomes)
        )
        return
    rest = [] if smoke else [j for j in manifest["jobs"] if j["case"] != "baseline"]
    if rest and workers > 1:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(rest)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=e.worker_init,
        ) as pool:
            futures = [
                pool.submit(run_job, (str(output), job, identity)) for job in rest
            ]
            for future in as_completed(futures):
                result = future.result()
                outcomes.append(result)
                print(result, flush=True)
    else:
        for job in rest:
            result = run_job((str(output), job, identity))
            outcomes.append(result)
            print(result, flush=True)
    e.write_json(
        output / "run_status.json",
        dict(
            status="partial"
            if smoke
            else (
                "complete_with_failures"
                if any(r["status"] == "numerical_failure" for r in outcomes)
                else "complete"
            ),
            numerical_failures=sum(
                r["status"] == "numerical_failure" for r in outcomes
            ),
            outcomes=outcomes,
            completed_jobs=sum(r["status"] != "numerical_failure" for r in outcomes),
            new_arms=(len(baseline) + len(rest)) * 6,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["prepare", "run", "report", "all"], default="all"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Verify adapter and fit all three new baseline graphs only",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.phase in ("prepare", "all"):
        prepare(args.output)
    if args.phase in ("run", "all"):
        run(args.output, args.workers, args.smoke)
    from .omc_graph_report import report

    report(args.output)


if __name__ == "__main__":
    main()
