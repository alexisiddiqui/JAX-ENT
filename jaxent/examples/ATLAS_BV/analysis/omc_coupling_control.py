"""Fixed-coupling and fixed-edge-pattern controls on frozen candidate ensembles."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path

import jax
import numpy as np

from . import omc_bandwidth_control as old
from . import omc_coverage_control as previous
from .common import HERE

OUTPUT = previous.OUTPUT.parent / "omc_coupling_control"
REFERENCE = 2  # existing q=0.08, fixed before inspecting fit outcomes
VARIANTS = ("fixed_coupling", "fixed_pattern")


def graph_controls(work, keep, sigmas):
    """Scale retained kernels only; diagonal edges never enter coupling estimates."""
    keep = np.asarray(keep)
    if (
        len(keep) < 2
        or len(sigmas) != 6
        or np.any(np.asarray(sigmas) <= 0)
        or not np.isfinite(sigmas).all()
    ):
        raise ValueError("at least two frames and six positive finite sigmas required")
    d = work[np.ix_(keep, keep)]
    original = np.stack([np.exp(-np.minimum(0.5 * (d / s) ** 2, 80)) for s in sigmas])
    off = ~np.eye(len(keep), dtype=bool)
    coupling = original[:, off].mean(axis=1)
    if not np.isfinite(coupling).all() or np.any(coupling <= 0):
        raise ValueError("invalid graph coupling")
    scaled = original * (coupling[REFERENCE] / coupling)[:, None, None]
    pattern = (
        original[REFERENCE][None, :, :]
        * (coupling / coupling[REFERENCE])[:, None, None]
    )
    np.testing.assert_allclose(
        scaled[:, off].mean(axis=1), coupling[REFERENCE], rtol=1e-12
    )
    np.testing.assert_allclose(pattern[:, off].mean(axis=1), coupling, rtol=1e-12)
    np.testing.assert_array_equal(scaled[REFERENCE], original[REFERENCE])
    np.testing.assert_array_equal(pattern[REFERENCE], original[REFERENCE])
    return dict(
        existing=original, fixed_coupling=scaled, fixed_pattern=pattern
    ), coupling


def fit_specs():
    return [
        dict(variant=v, q_index=i, quantile=q)
        for v in VARIANTS
        for i, q in enumerate(old.QUANTILES)
        if i != REFERENCE
    ]


def fitting_identity():
    # Report code is intentionally excluded: report-only changes do not refit.
    return {
        str(p.relative_to(HERE)): old.digest(p)
        for p in (
            Path(__file__),
            Path(old.__file__),
            HERE / "analysis/cluster_filtering_omc.py",
        )
    }


def prepare(output):
    if (output / "manifest.json").exists():
        return load_manifest(output)
    parent = previous.load_manifest(previous.OUTPUT)
    inputs = {
        str(previous.OUTPUT / "manifest.json"): old.digest(
            previous.OUTPUT / "manifest.json"
        )
    }
    jobs = []
    for row in parent["systems"]:
        sid = row["system_id"]
        source = previous.OUTPUT / "systems" / sid / "source.npz"
        inputs[str(source)] = old.digest(source)
        for case in row["cases"]:
            if case["method"] != "stratified" or case["retention"] not in (0.25, 0.125):
                continue
            folder = source.parent / case["case"]
            with np.load(source) as data:
                frozen = {k: data[k] for k in data}
            if (
                old.load_complete(
                    folder, frozen, case, old.digest(previous.OUTPUT / "manifest.json")
                )
                is None
            ):
                raise ValueError(f"Invalid parent candidate: {folder}")
            controls, coupling = graph_controls(
                frozen["work"], case["keep"], case["sigmas"]
            )
            assert len(controls) == 3
            for name in (
                "weights.npz",
                "fits.parquet",
                "populations.parquet",
                "complete.json",
            ):
                inputs[str(folder / name)] = old.digest(folder / name)
            jobs.append(
                dict(
                    system_id=sid,
                    source=str(source),
                    parent_folder=str(folder),
                    case=case,
                    original_coupling=coupling.tolist(),
                )
            )
    if len(jobs) != 20:
        raise ValueError("Expected twenty frozen stratified candidates")
    manifest = dict(
        fitting_code=fitting_identity(),
        input_hashes=inputs,
        jobs=jobs,
        expected_new_fits=200,
        protocol=dict(
            reference_quantile=0.08,
            strength=0.1,
            checkpoints=list(old.CHECKPOINTS),
            quantiles=list(old.QUANTILES),
            variants=list(VARIANTS),
            reference_index=REFERENCE,
        ),
    )
    old.atomic_json(output / "manifest.json", manifest)
    return manifest


def load_manifest(output):
    m = json.loads((output / "manifest.json").read_text())
    if m["fitting_code"] != fitting_identity():
        raise ValueError("Fitting implementation changed: use a fresh output directory")
    for path, value in m["input_hashes"].items():
        if old.digest(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
    return m


def validate_fit(fit, source, keep):
    weights = fit["weights"]
    if weights.shape != (10, len(source["indices"])):
        raise ValueError("incorrect new-fit shape")
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
    assert not weights[:, np.setdiff1d(np.arange(weights.shape[1]), keep)].any()
    for name in (
        "objective",
        "relative_change",
        "absolute_change",
        "steps",
        "grad_norm",
    ):
        assert fit[name].shape == (10,) and np.isfinite(fit[name]).all()


def load_complete(folder, identity, source, keep):
    try:
        receipt = json.loads((folder / "complete.json").read_text())
        if receipt["identity"] != identity or receipt["sha256"] != old.digest(
            folder / "fit.npz"
        ):
            return None
        with np.load(folder / "fit.npz") as data:
            fit = {k: data[k] for k in data}
        validate_fit(fit, source, keep)
        return fit
    except (OSError, ValueError, KeyError, AssertionError):
        return None


def run_case(output, job, identity):
    case = job["case"]
    folder = output / "systems" / job["system_id"] / case["case"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with np.load(job["source"]) as data:
            source = {k: data[k] for k in data}
        keep = np.asarray(case["keep"])
        if load_complete(folder, identity, source, keep) is not None:
            return "cached " + case["case"]
        controls, _ = graph_controls(source["work"], keep, case["sigmas"])
        kernels = np.zeros((10, len(source["indices"]), len(source["indices"])))
        for i, spec in enumerate(fit_specs()):
            kernels[i][np.ix_(keep, keep)] = controls[spec["variant"]][spec["q_index"]]
        fit = jax.tree.map(
            np.asarray,
            old.fit_candidates(
                source["flat"],
                source["target"],
                np.isin(np.arange(len(source["indices"])), keep),
                kernels,
                np.zeros(10, dtype=int),
                np.full(10, 0.1),
            ),
        )
        validate_fit(fit, source, keep)
        old.atomic_npz(folder / "fit.npz", **fit)
        old.atomic_json(
            folder / "complete.json",
            dict(identity=identity, sha256=old.digest(folder / "fit.npz")),
        )
        return f"{job['system_id']} {case['case']}: {int((fit['relative_change'] <= 0.01).sum())}/10 converged"


def run(output, m, workers):
    identity = old.digest(output / "manifest.json")
    ctx = multiprocessing.get_context("spawn")
    slots = ctx.Queue()
    cpus = sorted(os.sched_getaffinity(0))
    count = min(workers, len(cpus), 20)
    for group in np.array_split(cpus, count):
        slots.put([int(c) for c in group[:2]])
    try:
        with ProcessPoolExecutor(
            max_workers=count,
            mp_context=ctx,
            initializer=old.worker_init,
            initargs=(slots,),
        ) as pool:
            futures = [pool.submit(run_case, output, j, identity) for j in m["jobs"]]
            for i, f in enumerate(as_completed(futures), 1):
                print(f"[{i}/20] {f.result()}", flush=True)
    finally:
        slots.close()
        slots.join_thread()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=("prepare", "fit", "report", "all"), default="all"
    )
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    if not 1 <= a.workers <= 10:
        p.error("workers must be 1–10")
    for protected in (old.OUTPUT, previous.OUTPUT):
        if (
            a.output.resolve() == protected.resolve()
            or protected.resolve() in a.output.resolve().parents
        ):
            p.error("use a separate output directory")
    m = prepare(a.output) if a.phase in ("prepare", "all") else load_manifest(a.output)
    if a.phase in ("fit", "all"):
        run(a.output, m, a.workers)
    if a.phase in ("report", "all"):
        from .omc_coupling_report import report

        report(a.output, m)


if __name__ == "__main__":
    main()
