from __future__ import annotations

import numpy as np

from profiling.eager_native_memory import classify


def _report(rss_slope: int, heap_slope: int, arrays_slope: int):
    samples = [
        {
            "phase": "checkpoint",
            "step": step,
            "rss_bytes": 1_000_000 + rss_slope * step,
            "python_heap_bytes": 100_000 + heap_slope * step,
            "python_heap_peak_bytes": 100_000 + heap_slope * step,
            "live_array_count": 100 + arrays_slope * step,
            "live_array_bytes": 1_000 + arrays_slope * step * 4,
        }
        for step in (0, 1, 10, 100, 1000)
    ]
    samples.extend(
        [
            {
                **samples[-1],
                "phase": "post_gc",
                "live_array_count": 10,
                "live_array_bytes": 100,
            },
            {
                **samples[-1],
                "phase": "post_clear_caches",
                "live_array_count": 10,
                "live_array_bytes": 100,
            },
        ]
    )
    return {"samples": samples}


def test_classify_distinguishes_retention_from_async_and_cache() -> None:
    result = classify(
        {
            "retained_checkpoint_sync": _report(10_000, 100, 5),
            "retained_per_step_sync": _report(9_500, 100, 5),
            "discarded_checkpoint_sync": _report(100, 100, 0),
        }
    )

    assert np.isclose(
        result["slopes"]["retained_checkpoint_sync"]["rss_bytes"], 10_000
    )
    assert "per-step synchronization does not remove RSS growth" in result["findings"]
    assert any("retained JAX buffers" in finding for finding in result["findings"])
    assert any("allocator caching" in finding for finding in result["findings"])


def test_classify_reports_nonmetric_live_array_retention() -> None:
    result = classify(
        {
            "retained_checkpoint_sync": _report(10_000, 100, 5),
            "retained_per_step_sync": _report(9_500, 100, 5),
            "discarded_checkpoint_sync": _report(9_000, 100, 3),
        }
    )

    assert any(
        "reference retention is the primary cause" in finding
        for finding in result["findings"]
    )
