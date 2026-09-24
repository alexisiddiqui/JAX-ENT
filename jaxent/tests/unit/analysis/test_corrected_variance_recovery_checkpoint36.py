import numpy as np
import pytest

from jaxent.examples.ATLAS_BV.analysis import (
    corrected_variance_recovery_checkpoint36 as checkpoint,
)
from jaxent.examples.ATLAS_BV.analysis.corrected_variance_recovery_checkpoint36 import (
    build_parser,
    ca_radius_of_gyration,
    coordinate_dispersion,
    work_representations,
)
from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import (
    direct_distance,
)
from jaxent.examples.common.analysis.scoring import calculate_work_metrics


def test_ca_rg_known_geometry_and_rigid_motion_invariance():
    coords = np.array(
        [[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
    )
    np.testing.assert_allclose(ca_radius_of_gyration(coords), [2.0, 3.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        ca_radius_of_gyration(coords @ rotation + [5.0, 9.0, -3.0]), [2.0, 3.0]
    )


def test_frame_representations_reproduce_all_provided_work_metrics():
    z = np.asarray(
        [
            [0.4, 1.2],
            [1.7, 0.8],
            [2.5, 3.1],
            [0.9, 2.0],
        ]
    )
    representations = work_representations(z)
    left = np.asarray([0])
    right = np.asarray([1])
    direct = {
        name: direct_distance(values, left, right, kind)[0]
        for name, (values, kind) in representations.items()
        if name.startswith("work_")
    }
    direct["work_fitting"] = direct["work_scale"] + direct["work_density"]
    direct["work_magnitude"] = direct["work_shape"] - direct["work_scale"]

    expected = calculate_work_metrics(z[:, 0], z[:, 1], T=300.0)
    rt_kj_mol = 8.314 * 300.0 / 1000.0
    for name in (
        "work_scale",
        "work_shape",
        "work_density",
        "work_fitting",
        "work_magnitude",
        "work_opt",
    ):
        assert direct[name] == pytest.approx(expected[f"{name}_kj"] / rt_kj_mol)


def test_coordinate_dispersion_is_symmetric_and_scale_invariant():
    metric = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.5, 1.0, 0.0],
        ]
    )
    neighbours = np.asarray([[1, 2], [0, 2], [3, 1], [2, 1]])
    left = np.asarray([0, 3])
    right = np.asarray([3, 0])

    feature, reference = coordinate_dispersion(
        metric, neighbours, left, right, 0.01, None
    )
    scaled, scaled_reference = coordinate_dispersion(
        7.0 * metric, neighbours, left, right, 0.01, None
    )

    np.testing.assert_allclose(feature[0], feature[1])
    np.testing.assert_allclose(feature, scaled)
    assert scaled_reference == pytest.approx(49.0 * reference)


def test_checkpoint_parser_exposes_review_gate_and_isolated_defaults():
    parser = build_parser()
    args = parser.parse_args(["--scope", "pilot", "--approve-pilot", "--workers", "2"])
    assert args.scope == "pilot"
    assert args.approve_pilot
    assert args.workers == 2


def test_energy_alignment_rejects_shifted_frames(tmp_path, monkeypatch):
    folder = tmp_path / "example"
    folder.mkdir()
    for replica in (1, 2, 3):
        np.savez(
            folder / f"example_R{replica}.energies.npz",
            frame=np.array([0, 1, 2]),
            total=np.array([-10.0, replica, replica + 0.5]),
        )
    monkeypatch.setattr(
        checkpoint, "ENERGY_SOURCES", {"energy": (tmp_path, "total", "REU")}
    )
    data = {
        "system": "example",
        "frames": np.tile([1, 2], 3),
        "replicas": np.repeat([1, 2, 3], 2),
    }
    config = {"analysis": {"frame_interval_ns": 1.0, "equilibration_ns": 0.0}}
    values, provenance = checkpoint.energy_representations(data, config)
    np.testing.assert_array_equal(values["energy"][0], [1, 1.5, 2, 2.5, 3, 3.5])
    assert len(provenance["energy"]["sources"]) == 3
    data["frames"][-1] = 3
    with pytest.raises(ValueError, match="frame alignment mismatch"):
        checkpoint.energy_representations(data, config)
