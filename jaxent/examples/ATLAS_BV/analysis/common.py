from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


HERE = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text())


def load_systems() -> list[dict[str, str]]:
    with (HERE / "data" / "systems.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def replica_paths(row: dict[str, str]) -> list[Path]:
    return [HERE / value for value in row["replica_paths"].split(";")]


def feature_dir(system: str, replica: int) -> Path:
    return HERE / "outputs" / "stage1" / system / f"R{replica}"


def post_equilibration_indices(n_frames: int, equilibration_ns: float, dt_ns: float) -> np.ndarray:
    times = np.arange(n_frames, dtype=float) * dt_ns
    return np.flatnonzero(times > equilibration_ns)


def load_contact_coordinates(system: str, replica: int, config: dict) -> dict[str, np.ndarray]:
    protocol = config["protocol"]
    settings = config["analysis"]
    with np.load(feature_dir(system, replica) / "features.npz", allow_pickle=False) as data:
        heavy = np.asarray(data["heavy_contacts"], dtype=np.float64)
        acceptor = np.asarray(data["acceptor_contacts"], dtype=np.float64)
    if heavy.shape != acceptor.shape or heavy.ndim != 2:
        raise ValueError(f"invalid feature shapes for {system} R{replica}: {heavy.shape}, {acceptor.shape}")
    keep = post_equilibration_indices(
        heavy.shape[1], settings["equilibration_ns"], settings["frame_interval_ns"]
    )
    heavy = heavy[:, keep]
    acceptor = acceptor[:, keep]
    h = heavy.sum(axis=0)
    o = acceptor.sum(axis=0)
    return {
        "frame": keep,
        "heavy": heavy,
        "acceptor": acceptor,
        "H": h,
        "O": o,
        "G": protocol["bv_bc"] * h + protocol["bv_bh"] * o,
    }


def integrated_autocorrelation_frames(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    variance = np.dot(centered, centered)
    if variance <= 0 or len(values) < 3:
        return 1
    correlation = np.correlate(centered, centered, mode="full")[len(values) - 1 :] / variance
    tau = 1.0
    for lag in range(1, len(correlation)):
        if correlation[lag] <= 0:
            break
        tau += 2.0 * correlation[lag]
    return max(1, min(len(values) // 2, int(math.ceil(tau))))


def histogram_edges(a: np.ndarray, b: np.ndarray, minimum: int, maximum: int) -> np.ndarray:
    pooled = np.concatenate([a, b])
    q25, q75 = np.quantile(pooled, [0.25, 0.75])
    width = 2.0 * (q75 - q25) / np.cbrt(len(pooled))
    if width <= 0 or not np.isfinite(width):
        bins = minimum
    else:
        bins = int(np.ceil((pooled.max() - pooled.min()) / width))
        bins = int(np.clip(bins, minimum, maximum))
    low, high = float(pooled.min()), float(pooled.max())
    if low == high:
        high = low + 1.0
    return np.linspace(low, high, bins + 1)


def distribution_distances(a: np.ndarray, b: np.ndarray, minimum: int, maximum: int) -> tuple[float, float]:
    edges = histogram_edges(a, b, minimum, maximum)
    pa = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
    pb = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
    js_bits = float(jensenshannon(pa, pb, base=2.0) ** 2)
    return float(ks_2samp(a, b, method="auto").statistic), js_bits


def moving_block_sample(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if block >= n:
        return values.copy()
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([values[start : start + block] for start in starts])[:n]


def bootstrap_distance_ci(
    a: np.ndarray,
    b: np.ndarray,
    block: int,
    samples: int,
    seed: int,
    minimum: int,
    maximum: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    stats = np.empty((samples, 2), dtype=float)
    for index in range(samples):
        stats[index] = distribution_distances(
            moving_block_sample(a, block, rng),
            moving_block_sample(b, block, rng),
            minimum,
            maximum,
        )
    quantiles = np.quantile(stats, [0.025, 0.975], axis=0)
    return (float(quantiles[0, 0]), float(quantiles[1, 0])), (
        float(quantiles[0, 1]),
        float(quantiles[1, 1]),
    )


def atomic_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(path)
