from __future__ import annotations

import numpy as np
import pytest

from jaxent.src.models.func import uptake


def test_hdxrate_call_locks_pdla_and_disables_ph_correction(monkeypatch):
    captured = {}

    def fake(sequence, temperature, pH_read, **kwargs):
        captured.update(sequence=sequence, temperature=temperature, pH_read=pH_read, **kwargs)
        return np.asarray([0.0, 2.0, 3.0])

    monkeypatch.setattr(uptake, "k_int_from_sequence", fake)
    actual = uptake.calculate_HDXrate_from_sequence("GAA", 298.0, 4.4)

    np.testing.assert_array_equal(actual, [0.0, 2.0, 3.0])
    assert captured == {
        "sequence": "GAA",
        "temperature": 298.0,
        "pH_read": 4.4,
        "reference": "poly",
        "exchange_type": "HD",
        "d_percentage": 100.0,
        "ph_correction": False,
    }


def test_hdxrate_minute_conversion_is_explicit(monkeypatch):
    monkeypatch.setattr(
        uptake, "k_int_from_sequence", lambda *args, **kwargs: np.asarray([0.0, 2.0])
    )
    rates_s = uptake.calculate_HDXrate_from_sequence("GA", unit="s^-1")
    rates_min = uptake.calculate_HDXrate_from_sequence("GA", unit="min^-1")
    np.testing.assert_array_equal(rates_min, rates_s * 60.0)


def test_hdxrate_rejects_ambiguous_unit():
    with pytest.raises(ValueError, match="unsupported HDXrate unit"):
        uptake.calculate_HDXrate_from_sequence("GA", unit="hour^-1")  # type: ignore[arg-type]
