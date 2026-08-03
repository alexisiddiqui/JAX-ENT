from pathlib import Path

import numpy as np
import pytest

from jaxent.examples.common.loading import (
    load_hdx_timepoints_minutes,
    validate_hdx_timepoint_count,
)
from jaxent.examples.common.manifest import write_prelaunch_manifest


def test_load_moprp_protocol_timepoints() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples/2_CrossValidation/data/_MoPrP/moprp.times"
    )

    actual = load_hdx_timepoints_minutes(source)

    assert actual.shape == (15,)
    np.testing.assert_allclose(
        actual,
        [
            0.0834,
            0.3336,
            0.6666,
            1.0002,
            4.9998,
            10.0002,
            19.9998,
            30.0,
            45.0,
            60.0,
            160.0002,
            240.0,
            390.0,
            750.0,
            1440.0,
        ],
    )


def test_timepoint_loader_requires_initial_zero(tmp_path: Path) -> None:
    source = tmp_path / "times"
    source.write_text("0.1\n1.0\n")

    with pytest.raises(ValueError, match="initial zero"):
        load_hdx_timepoints_minutes(source)


def test_hdx_vector_length_must_match_protocol() -> None:
    class Datapoint:
        def extract_features(self) -> np.ndarray:
            return np.ones(5)

    with pytest.raises(ValueError, match="15-point protocol"):
        validate_hdx_timepoint_count(
            [Datapoint()], np.ones(15), label="test validation"
        )


def test_manifest_records_timepoint_values_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "times"
    source.write_text("0\n0.1\n1.0\n")

    manifest_path = write_prelaunch_manifest(
        tmp_path / "output",
        example=2,
        ensembles=["AF2_filtered"],
        losses=["MSE"],
        split_types=["spatial"],
        maxent_values=[1.0],
        learning_rate=1.0,
        lr_adjustment="off",
        frame_average_impl="tensordot",
        step_chunk_size=100,
        n_steps=5000,
        jobs=1,
        timepoints_file=source,
    )

    import json

    recorded = json.loads(manifest_path.read_text())["resolved_inputs"]["timepoints"]
    assert recorded["source"] == str(source.resolve())
    assert recorded["values_minutes"] == [6.0, 60.0]
    assert len(recorded["sha256"]) == 64
