import numpy as np
import pytest
from jaxent.examples.ATLAS_BV.analysis import omc_coverage_control as experiment


def test_balanced_strata_partition_and_degenerate_geometry():
    for n in (17, 272, 373):
        d = abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
        for count in (1, int(n * 0.125), int(n * 0.5), n):
            strata = experiment.balanced_strata(d, count)
            assert len(strata) == count
            np.testing.assert_array_equal(np.sort(np.concatenate(strata)), np.arange(n))
            sizes = list(map(len, strata))
            assert max(sizes) - min(sizes) <= 1
    strata = experiment.balanced_strata(np.zeros((11, 11)), 4)
    assert list(map(len, strata)) == [3, 3, 3, 2]
    with pytest.raises(ValueError):
        experiment.balanced_strata(np.zeros((3, 3)), 4)


def test_paired_sampling_counts_priorities_and_reproducibility():
    labels = np.repeat([0, 1], [37, 15])
    d = abs(np.arange(52)[:, None] - np.arange(52)[None, :]).astype(float)
    seed = 20260909
    retention = 0.25
    subsets = experiment.paired_subsets(labels, d, retention, seed)
    priorities = np.random.default_rng(seed).random(37)
    strata = experiment.balanced_strata(d[:37, :37], 9)
    expected = np.sort([s[np.argmin(priorities[s])] for s in strata])
    np.testing.assert_array_equal(subsets["stratified"][:9], expected)
    np.testing.assert_array_equal(
        subsets["random"][:9], np.sort(np.argsort(priorities)[:9])
    )
    for method, keep in subsets.items():
        assert len(keep) == len(np.unique(keep)) == 24
        assert np.bincount(labels[keep]).tolist() == [9, 15]
        np.testing.assert_array_equal(
            keep, experiment.paired_subsets(labels, d, retention, seed)[method]
        )
        np.testing.assert_array_equal(keep[9:], np.arange(37, 52))


def test_kernels_use_supplied_sigmas_not_candidate_quantiles():
    work = abs(np.arange(12)[:, None] - np.arange(12)[None, :]).astype(float)
    sigmas = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    keep = np.array([1, 3, 5, 8])
    kernels, used = experiment.frozen_kernels(work, keep, sigmas)
    np.testing.assert_array_equal(used[:6], sigmas)
    for i, sigma in enumerate(sigmas):
        np.testing.assert_allclose(
            kernels[i][np.ix_(keep, keep)],
            np.exp(-np.minimum(0.5 * (work[np.ix_(keep, keep)] / sigma) ** 2, 80)),
        )
    assert not kernels[6:].any()
    assert not kernels[:, np.setdiff1d(np.arange(12), keep)].any()
    with pytest.raises(ValueError):
        experiment.frozen_kernels(work, keep, [0] * 6)


def test_candidate_resume_invalidation_and_kernel_forwarding(tmp_path, monkeypatch):
    from jaxent.examples.ATLAS_BV.analysis import omc_bandwidth_control as old

    x = np.linspace(0, 1, 24)
    source = dict(
        flat=np.stack([x, x * x, np.sin(x)]),
        indices=np.arange(24),
        labels=np.repeat([0, 1], [16, 8]),
        structural=abs(x[:, None] - x[None, :]),
        work=abs(x[:, None] - x[None, :]),
    )
    source["target"] = source["flat"].mean(axis=1)
    keep = experiment.paired_subsets(
        source["labels"], source["structural"], 0.5, 20260909
    )["stratified"]
    row = dict(
        system_id="synthetic", role="cohort", rmsf_tercile="low", cath_class="test", k=2
    )
    case = dict(
        case="paired",
        method="stratified",
        retention=0.5,
        seed=20260909,
        thinned_cluster=0,
        keep=keep.tolist(),
        sigmas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    old.atomic_npz(tmp_path / "systems/synthetic/source.npz", **source)
    original = old.fit_candidates
    calls = []

    def short_fit(values, target, mask, kernels, kinds, strengths):
        calls.append(1)
        expected, _ = experiment.frozen_kernels(source["work"], keep, case["sigmas"])
        np.testing.assert_array_equal(kernels, expected)
        return original(
            values,
            target,
            mask,
            kernels,
            kinds,
            strengths,
            checkpoints=(10, 20),
            window=5,
        )

    monkeypatch.setattr(old, "fit_candidates", short_fit)
    old.atomic_json(tmp_path / "manifest.json", case)
    identity = old.digest(tmp_path / "manifest.json")
    experiment.run_case(tmp_path, row, case, identity)
    experiment.run_case(tmp_path, row, case, identity)
    assert len(calls) == 1
    case["sigmas"][0] = 0.15
    old.atomic_json(tmp_path / "manifest.json", case)
    identity = old.digest(tmp_path / "manifest.json")
    experiment.run_case(tmp_path, row, case, identity)
    assert len(calls) == 2
    (tmp_path / "systems/synthetic/paired/weights.npz").write_bytes(b"broken")
    experiment.run_case(tmp_path, row, case, identity)
    assert len(calls) == 3
