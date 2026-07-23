"""Light tests for the Stage C non-circular covariance linear-model runner (pure helpers)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def lm():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_covariance_linear_model")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def test_constants(lm):
    assert len(lm.AA) == 20 and len(set(lm.AA)) == 20
    assert set(lm.FEATURE_GROUPS) == {"unweighted", "elastic_network", "sequence"}


def test_fit_transfer_is_deterministic_and_uses_disjoint_ensembles(lm):
    import pandas as pd
    rng = np.random.default_rng(0)
    cols = ["a", "b"]
    Xa = pd.DataFrame(rng.normal(size=(30, 2)), columns=cols)
    ya = Xa["a"].values * 2.0 + rng.normal(scale=0.1, size=30)
    Xb = pd.DataFrame(rng.normal(size=(20, 2)), columns=cols)  # different, disjoint sample
    yb = Xb["a"].values * 2.0 + rng.normal(scale=0.1, size=20)
    r1 = lm._fit_transfer(Xa, ya, Xb, yb, cols)
    r2 = lm._fit_transfer(Xa, ya, Xb, yb, cols)
    assert r1 == r2  # deterministic (no randomness in ridge)
    assert r1 > 0.8  # a genuinely transferable linear signal is recovered


def test_ridge_pipeline_standardizes(lm):
    pipe = lm._ridge()
    names = [step[0] for step in pipe.steps]
    assert names[0] == "standardscaler"
