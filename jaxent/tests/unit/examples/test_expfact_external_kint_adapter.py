from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[4]
ADAPTER = REPO / "jaxent/examples/2_CrossValidation/fitting/jaxENT/expfact_external_kint_adapter.py"
spec = importlib.util.spec_from_file_location("expfact_external_kint_adapter", ADAPTER)
assert spec and spec.loader
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def write_rates(path: Path, rates: np.ndarray) -> None:
    path.write_text("".join(f"{index} {rate}\n" for index, rate in enumerate(rates, 1)))


def test_external_kint_loader_consumes_deliberately_altered_rates(tmp_path):
    sequence = "GAPAA"
    first = np.asarray([-1.0, 2.0, -1.0, 4.0, 5.0])
    second = first.copy()
    second[1] = 200.0
    first_path, second_path = tmp_path / "first.kint", tmp_path / "second.kint"
    write_rates(first_path, first)
    write_rates(second_path, second)
    loaded_first = adapter.load_external_kints(first_path, sequence)
    loaded_second = adapter.load_external_kints(second_path, sequence)
    assert loaded_first[1] == 2.0
    assert loaded_second[1] == 200.0
    assert not np.array_equal(loaded_first, loaded_second)


@pytest.mark.parametrize(
    "rates,match",
    [
        ([-1, 2, 3, 4, 5], "sentinel placement"),
        ([-1, 2, -2, 4, 5], "only accepted negative"),
        ([-1, 0, -1, 4, 5], "finite and either positive"),
    ],
)
def test_external_kint_loader_rejects_invalid_sentinels(tmp_path, rates, match):
    path = tmp_path / "bad.kint"
    write_rates(path, np.asarray(rates, dtype=float))
    with pytest.raises(ValueError, match=match):
        adapter.load_external_kints(path, "GAPAA")
