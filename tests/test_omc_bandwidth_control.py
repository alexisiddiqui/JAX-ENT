"""Focused design, uninterrupted optimiser parity, matching and resume checks."""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis import omc_bandwidth_control as experiment


def source_fixture():
    x = np.linspace(0, 1, 40)
    flat = np.stack([x, x**2, np.sin(x)])
    distance = abs(x[:, None] - x[None, :])
    return dict(
        flat=flat,
        target=flat.mean(axis=1),
        structural=distance,
        work=distance,
        indices=np.arange(100, 140),
        labels=np.repeat([0, 1], [24, 16]),
    )


def test_cohort_order_is_deterministic_balanced_and_excludes_reference():
    rows = [
        dict(system_id=f"{tercile}_{c}_{i}", rmsf_tercile=tercile, cath_class=c)
        for tercile in ("low", "middle", "high")
        for c in ("alpha", "beta", "mixed")
        for i in range(4)
    ]
    rows.append(
        dict(system_id=experiment.REFERENCE, rmsf_tercile="low", cath_class="alpha")
    )
    ordered = experiment.stratified_order(rows, 123)
    assert ordered == experiment.stratified_order(rows[::-1], 123)
    for group in ordered.values():
        assert len({row["cath_class"] for row in group[:3]}) == 3
        assert len(group) == 12
        assert experiment.REFERENCE not in {row["system_id"] for row in group}


def test_partition_uses_global_eligible_silhouette_winner_and_smaller_k_tie(
    monkeypatch,
):
    audit = pd.DataFrame(
        [
            dict(k=2, silhouette=0.1, eligible=True),
            dict(k=4, silhouette=0.5, eligible=True),
            dict(k=6, silhouette=0.5, eligible=True),
            dict(k=8, silhouette=0.8, eligible=False),
        ]
    )
    candidates = {k: np.array([k]) for k in (2, 4, 6)}
    monkeypatch.setattr(experiment, "partitions", lambda *_: (audit.copy(), candidates))
    selected, labels = experiment.best_partition(np.zeros((10, 10)), 123)
    assert selected[selected.selected].k.tolist() == [4]
    np.testing.assert_array_equal(labels, [4])
    audit["eligible"] = False
    assert experiment.best_partition(np.zeros((10, 10)), 123)[1] is None


def test_exact_design_and_nested_largest_cluster_filtering():
    specs = experiment.arms()
    assert len(specs) == 13 and len({a["arm"] for a in specs}) == 13
    assert [a["quantile"] for a in specs[:6]] == list(experiment.QUANTILES)
    assert {a["strength"] for a in specs[:6]} == {0.1}
    assert [a["strength"] for a in specs[6:12]] == list(experiment.MAXENT_STRENGTHS)
    assert specs[-1]["strength"] == 0
    source = source_fixture()
    cases = list(experiment.cases(source["labels"], 123))
    assert len(cases) == 4
    target = source["target"].copy()
    for earlier, later in zip(cases, cases[1:]):
        assert set(later["keep"]) < set(earlier["keep"])
        assert set(range(24, 40)).issubset(later["keep"])
        assert later["thinned_cluster"] == 0
        assert np.bincount(source["labels"][later["keep"]]).min() >= 2
    np.testing.assert_array_equal(source["target"], target)
    assert not np.allclose(source["flat"][:, cases[-1]["keep"]].mean(axis=1), target)


def test_continuation_matches_independent_adam_and_preserves_stopped_fit():
    values = jnp.array([[0.1, 0.5, 0.9, 0.3], [0.4, 0.1, 0.3, 0.8]])
    target = jnp.array([0.75, 0.22])
    mask = jnp.array([True, True, True, False])
    kernels = jnp.stack(
        [jnp.zeros((4, 4)), jnp.ones((4, 4)) * mask[:, None] * mask[None, :]]
    )
    kinds, strengths = jnp.array([2, 0]), jnp.array([0.001, 0.1])
    result = experiment.fit_candidates(
        values,
        target,
        mask,
        kernels,
        kinds,
        strengths,
        checkpoints=(8, 16, 24),
        window=4,
    )
    assert max(np.asarray(result["steps"])) > 8
    for i in range(2):
        optimizer = optax.adam(0.05)
        logits = jnp.zeros(4)
        state = optimizer.init(logits)

        def loss(x):
            return experiment.trajectory_objective(
                x, strengths[i], kernels[i], kinds[i], values, target, mask
            )

        for _ in range(int(result["steps"][i])):
            update, state = optimizer.update(jax.grad(loss)(logits), state, logits)
            logits = optax.apply_updates(logits, update)
        np.testing.assert_allclose(
            result["weights"][i],
            jax.nn.softmax(jnp.where(mask, logits, -jnp.inf)),
            atol=1e-7,
        )
    assert (np.asarray(result["weights"])[:, ~np.asarray(mask)] == 0).all()
    uniform = experiment.fit_candidates(
        jnp.zeros_like(values),
        jnp.zeros(2),
        jnp.ones(4, bool),
        kernels,
        kinds,
        strengths,
        checkpoints=(8, 16, 24),
        window=4,
    )
    np.testing.assert_array_equal(uniform["steps"], [8, 8])
    np.testing.assert_allclose(uniform["weights"], 1 / 4, atol=1e-8)


def test_ess_matching_boundary_ties_and_unconverged_exclusions():
    base = dict(
        system_id="test",
        role="cohort",
        case="retain",
        retention=0.5,
        converged=True,
        mse=0.01,
        population_tv=0.2,
    )
    table = pd.DataFrame(
        [
            dict(
                base,
                family="omc",
                arm="a",
                quantile=0.02,
                strength=0.1,
                ess_fraction=0.50,
            ),
            dict(
                base,
                family="omc",
                arm="b",
                quantile=0.04,
                strength=0.1,
                ess_fraction=0.561,
            ),
            dict(
                base,
                family="omc",
                arm="c",
                quantile=0.08,
                strength=0.1,
                ess_fraction=0.52,
                converged=False,
            ),
            dict(
                base,
                family="maxent",
                arm="m1",
                quantile=np.nan,
                strength=0.01,
                ess_fraction=0.52,
            ),
            dict(
                base,
                family="maxent",
                arm="m2",
                quantile=np.nan,
                strength=0.1,
                ess_fraction=0.52,
            ),
            dict(
                base,
                family="maxent",
                arm="m3",
                quantile=np.nan,
                strength=1,
                ess_fraction=0.561,
                converged=False,
            ),
        ]
    )
    matches = experiment.ess_matches(table).set_index("omc_arm")
    assert matches.loc["a", "matched"]
    assert matches.loc["a", "nearest_maxent_arm"] == "m1"
    assert matches.loc["b", "status"] == "outside_tolerance"
    assert matches.loc["c", "status"] == "omc_unconverged"
    assert np.isnan(matches.loc["b", "mse_difference"])
    table.loc[table.family == "maxent", "converged"] = False
    assert experiment.ess_matches(table).iloc[0].status == "no_converged_maxent"


def test_small_integration_verification_resume_and_corruption(tmp_path, monkeypatch):
    source = source_fixture()
    row = dict(
        system_id="synthetic",
        role="cohort",
        rmsf_tercile="low",
        cath_class="test",
        k=2,
        cases=list(experiment.cases(source["labels"], 123)),
    )
    manifest = dict(systems=[row], expected_fits=52)
    experiment.atomic_json(tmp_path / "manifest.json", manifest)
    base = tmp_path / "systems/synthetic"
    experiment.atomic_npz(base / "source.npz", **source)
    original = experiment.fit_candidates
    calls = []

    def small_fit(*args):
        calls.append(1)
        return original(*args, checkpoints=(20, 40, 60), window=10)

    monkeypatch.setattr(experiment, "fit_candidates", small_fit)
    experiment.run(tmp_path, manifest)
    assert len(calls) == 4
    experiment.run(tmp_path, manifest)
    assert len(calls) == 4
    broken = base / row["cases"][1]["case"] / "weights.npz"
    broken.write_bytes(b"truncated")
    experiment.run(tmp_path, manifest)
    assert len(calls) == 5
    folder = base / row["cases"][1]["case"]
    identity = experiment.digest(tmp_path / "manifest.json")
    loaded = experiment.load_complete(folder, source, row["cases"][1], identity)
    assert loaded is not None and len(loaded[0]) == 13
    assert (
        experiment.load_complete(folder, source, row["cases"][1], "wrong identity")
        is None
    )
    with np.load(folder / "weights.npz") as saved:
        changed = {key: saved[key] for key in saved.files}
    changed["target"] = changed["target"] + 0.1
    with pytest.raises(AssertionError):
        experiment.verify(source, row["cases"][1], *loaded, changed)
    # Report-only does not invoke the fitter and handles unmatched/partial curves.
    pd.DataFrame([dict(k=2, silhouette=0.3, selected=True)]).to_parquet(
        base / "clustering.parquet"
    )
    experiment.report(tmp_path, manifest)
    assert len(calls) == 5
    assert (tmp_path / "index.html").exists()
    assert len(pd.read_csv(tmp_path / "results.csv")) == 52


def test_screen_replaces_ineligible_systems_before_fits(tmp_path, monkeypatch):
    rows = [
        dict(
            system_id=f"{tercile}_{i}", rmsf_tercile=tercile, cath_class=f"class{i % 2}"
        )
        for tercile in ("low", "middle", "high")
        for i in range(5)
    ]
    rows.append(
        dict(system_id=experiment.REFERENCE, rmsf_tercile="low", cath_class="reference")
    )
    monkeypatch.setattr(experiment, "load_systems", lambda: rows)
    source = source_fixture()
    ordered = experiment.stratified_order(rows, 20260826)
    excluded = ordered["low"][0]["system_id"]

    def prepare(row, config, destination, settings):
        if row["system_id"] == excluded:
            raise ValueError("no eligible structural partition")
        folder = destination / "systems" / row["system_id"]
        experiment.atomic_npz(folder / "source.npz", **source)
        pd.DataFrame([dict(k=2, selected=True)]).to_parquet(
            folder / "clustering.parquet"
        )
        return source, {}

    monkeypatch.setattr(experiment, "prepare_source", prepare)
    monkeypatch.setattr(experiment, "protocol", lambda config: dict(seed=20260826))
    manifest = experiment.screen(tmp_path)
    assert len(manifest["systems"]) == 13 and manifest["expected_fits"] == 676
    assert excluded not in {r["system_id"] for r in manifest["systems"]}
    assert pd.read_csv(tmp_path / "screening.csv").status.eq("excluded").sum() == 1
    frozen = json.loads((tmp_path / "manifest.json").read_text())
    assert experiment.screen(tmp_path) == frozen


def test_spawned_workers_lock_duplicate_candidates(tmp_path):
    source = source_fixture()
    case = list(experiment.cases(source["labels"], 123))[1]
    row = dict(
        system_id="parallel",
        role="cohort",
        rmsf_tercile="low",
        cath_class="test",
        k=2,
        cases=[case],
    )
    manifest = dict(systems=[row], expected_fits=13)
    experiment.atomic_json(tmp_path / "manifest.json", manifest)
    experiment.atomic_npz(tmp_path / "systems/parallel/source.npz", **source)
    identity = experiment.digest(tmp_path / "manifest.json")
    context = multiprocessing.get_context("spawn")
    slots = context.Queue()
    cpus = sorted(os.sched_getaffinity(0))
    slots.put([cpus[0]])
    slots.put([cpus[-1]])
    try:
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=context,
            initializer=experiment.worker_init,
            initargs=(slots,),
        ) as pool:
            futures = [
                pool.submit(experiment.run_case, tmp_path, row, case, identity)
                for _ in range(2)
            ]
            statuses = [f.result(timeout=120) for f in futures]
    finally:
        slots.close()
        slots.join_thread()
    assert sorted(statuses) == ["cached", "computed"]
    folder = tmp_path / "systems/parallel" / case["case"]
    assert experiment.load_complete(folder, source, case, identity) is not None
    completed = (folder / "complete.json").read_bytes()
    experiment.run(tmp_path, manifest, workers=2)
    assert (folder / "complete.json").read_bytes() == completed
