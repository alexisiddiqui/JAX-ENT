"""Physics-first EX2 forward modelling for peptide HDX-MS data.

The core observable is a peptide-average of residue-specific exchange curves,
not a single exponential attached to the peptide::

    D_p(t) = mean_i[1 - exp(-k_int_i * exp(-lnP_i) * t)]

Only exchangeable backbone amides represented by the peptide construction are
included in the mean.  This module keeps residue identifiers explicit so that
experimental peptide numbering, intrinsic rates, and trajectory features cannot
silently drift out of register.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class HDXExperimentProtocol:
    """Conditions needed to interpret an HDX fractional-uptake curve."""

    timepoints_min: np.ndarray
    temperature_k: float
    experimental_pd: float
    intrinsic_rate_ph: float
    exchange_regime: str = "EX2"
    normalization: str = "fully_protonated_and_fully_deuterated_controls"
    replicate_count: int | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.timepoints_min, dtype=float)
        if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)):
            raise ValueError("timepoints_min must be a finite one-dimensional array")
        if np.any(times <= 0) or np.any(np.diff(times) <= 0):
            raise ValueError("timepoints_min must be positive and strictly increasing")
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.exchange_regime != "EX2":
            raise ValueError("the residue protection-factor model currently supports EX2 only")
        object.__setattr__(self, "timepoints_min", times)


@dataclass(frozen=True)
class IntrinsicRateSet:
    """Intrinsic exchange rates aligned to explicit one-based residue IDs."""

    residue_ids: np.ndarray
    rates_min: np.ndarray
    provider: str
    temperature_k: float
    ph: float
    source: str

    def __post_init__(self) -> None:
        residue_ids = np.asarray(self.residue_ids, dtype=int)
        rates = np.asarray(self.rates_min, dtype=float)
        if residue_ids.ndim != 1 or rates.shape != residue_ids.shape:
            raise ValueError("residue_ids and rates_min must be aligned one-dimensional arrays")
        if len(np.unique(residue_ids)) != len(residue_ids):
            raise ValueError("residue_ids must be unique")
        if np.any(~np.isfinite(rates)):
            raise ValueError("rates_min must be finite; use a negative sentinel for exclusions")
        object.__setattr__(self, "residue_ids", residue_ids)
        object.__setattr__(self, "rates_min", rates)

    def aligned(self, residue_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        lookup = {int(residue): float(rate) for residue, rate in zip(self.residue_ids, self.rates_min)}
        missing = [int(residue) for residue in residue_ids if int(residue) not in lookup]
        if missing:
            raise ValueError(f"intrinsic rates missing residue IDs: {missing}")
        return np.asarray([lookup[int(residue)] for residue in residue_ids], dtype=float)


@dataclass(frozen=True)
class PeptideExchangeMap:
    """Row-normalized peptide-to-exchangeable-amide map."""

    matrix: np.ndarray
    residue_ids: np.ndarray
    peptide_ids: np.ndarray
    peptide_starts: np.ndarray
    peptide_ends: np.ndarray
    active_amide_counts: np.ndarray
    n_terminal_residues_dropped: int
    excluded_residue_ids: tuple[int, ...]
    convention: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        residue_ids = np.asarray(self.residue_ids, dtype=int)
        peptide_ids = np.asarray(self.peptide_ids, dtype=int)
        starts = np.asarray(self.peptide_starts, dtype=int)
        ends = np.asarray(self.peptide_ends, dtype=int)
        counts = np.asarray(self.active_amide_counts, dtype=int)
        n_peptides = matrix.shape[0] if matrix.ndim == 2 else -1
        if matrix.ndim != 2 or matrix.shape[1] != len(residue_ids):
            raise ValueError("matrix must have shape (peptides, residue_ids)")
        if any(len(values) != n_peptides for values in (peptide_ids, starts, ends, counts)):
            raise ValueError("peptide metadata must align with matrix rows")
        if np.any(counts <= 0):
            raise ValueError("every peptide must contain at least one represented exchangeable amide")
        if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("peptide map rows must sum to one")
        expected = np.count_nonzero(matrix, axis=1)
        if not np.array_equal(expected, counts):
            raise ValueError("active_amide_counts must equal the nonzero entries in each map row")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "residue_ids", residue_ids)
        object.__setattr__(self, "peptide_ids", peptide_ids)
        object.__setattr__(self, "peptide_starts", starts)
        object.__setattr__(self, "peptide_ends", ends)
        object.__setattr__(self, "active_amide_counts", counts)

    def aligned_to(self, residue_ids: Sequence[int] | np.ndarray) -> "PeptideExchangeMap":
        """Return the same physical map on a requested residue coordinate."""

        requested = np.asarray(residue_ids, dtype=int)
        index = {int(residue): column for column, residue in enumerate(self.residue_ids)}
        missing_active = sorted(
            int(self.residue_ids[column])
            for column in np.flatnonzero(np.any(self.matrix > 0, axis=0))
            if int(self.residue_ids[column]) not in set(requested.tolist())
        )
        if missing_active:
            raise ValueError(f"requested coordinate omits active amides: {missing_active}")
        columns = [index.get(int(residue)) for residue in requested]
        matrix = np.zeros((len(self.peptide_ids), len(requested)), dtype=float)
        for target_column, source_column in enumerate(columns):
            if source_column is not None:
                matrix[:, target_column] = self.matrix[:, source_column]
        return PeptideExchangeMap(
            matrix=matrix,
            residue_ids=requested,
            peptide_ids=self.peptide_ids,
            peptide_starts=self.peptide_starts,
            peptide_ends=self.peptide_ends,
            active_amide_counts=self.active_amide_counts,
            n_terminal_residues_dropped=self.n_terminal_residues_dropped,
            excluded_residue_ids=self.excluded_residue_ids,
            convention=self.convention,
        )

    @property
    def active_residue_ids(self) -> np.ndarray:
        return self.residue_ids[np.any(self.matrix > 0, axis=0)]

    def subset_peptides(self, indices: Sequence[int] | np.ndarray) -> "PeptideExchangeMap":
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("peptide indices must be one-dimensional")
        return PeptideExchangeMap(
            matrix=self.matrix[indices],
            residue_ids=self.residue_ids,
            peptide_ids=self.peptide_ids[indices],
            peptide_starts=self.peptide_starts[indices],
            peptide_ends=self.peptide_ends[indices],
            active_amide_counts=self.active_amide_counts[indices],
            n_terminal_residues_dropped=self.n_terminal_residues_dropped,
            excluded_residue_ids=self.excluded_residue_ids,
            convention=self.convention,
        )


@dataclass(frozen=True)
class EX2Fit:
    log_pf: np.ndarray
    predicted: np.ndarray
    objective: float
    rmse: float
    success: bool
    message: str
    iterations: int
    gradient_norm: float
    initialization: str


@dataclass(frozen=True)
class EX2SolutionSet:
    """All finite multistart fits; no residue-wise averaging across modes."""

    residue_ids: np.ndarray
    solutions: tuple[EX2Fit, ...]
    best_index: int

    @property
    def best(self) -> EX2Fit:
        return self.solutions[self.best_index]

    @property
    def solution_range(self) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack([solution.log_pf for solution in self.solutions])
        return values.min(axis=0), values.max(axis=0)


@dataclass(frozen=True)
class ExPfactDataset:
    sequence: str
    protocol: HDXExperimentProtocol
    peptide_map: PeptideExchangeMap
    observed_uptake: np.ndarray


@dataclass(frozen=True)
class TrajectoryHDXComparison:
    average_first_curves: np.ndarray
    frame_mixture_curves: np.ndarray
    average_first_rmse: float | None
    frame_mixture_rmse: float | None


def load_intrinsic_rate_file(
    path: str | Path,
    *,
    provider: str,
    temperature_k: float,
    ph: float,
) -> IntrinsicRateSet:
    values = np.loadtxt(path, comments="#", dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("intrinsic-rate file must contain residue_id and rate/min columns")
    return IntrinsicRateSet(
        residue_ids=values[:, 0].astype(int),
        rates_min=values[:, 1],
        provider=provider,
        temperature_k=temperature_k,
        ph=ph,
        source=str(Path(path)),
    )


def build_expfact_peptide_map(
    sequence: str,
    assignments: np.ndarray,
    *,
    residue_ids: Sequence[int] | np.ndarray | None = None,
) -> PeptideExchangeMap:
    """Reproduce exPfact's MoPrP peptide-amide construction.

    Assignment rows are ``peptide_id, start, end`` with one-based protein
    residue numbers.  The official loop uses zero-based indices ``start:end``,
    which represents one-based residue IDs ``start+1 ... end``: exactly one
    peptide N-terminal residue is dropped.
    """

    sequence = sequence.strip().upper()
    assignments = np.asarray(assignments, dtype=int)
    if assignments.ndim != 2 or assignments.shape[1] < 3:
        raise ValueError("assignments must contain peptide_id, start, and end columns")
    coordinate = np.arange(1, len(sequence) + 1, dtype=int) if residue_ids is None else np.asarray(residue_ids, dtype=int)
    column = {int(residue): index for index, residue in enumerate(coordinate)}
    excluded = {1, *(index + 1 for index, code in enumerate(sequence) if code in {"P", "B"})}
    matrix = np.zeros((len(assignments), len(coordinate)), dtype=float)
    counts = []
    for row, (_, start, end, *_) in enumerate(assignments.tolist()):
        active = [residue for residue in range(start + 1, end + 1) if residue not in excluded]
        missing = [residue for residue in active if residue not in column]
        if missing:
            raise ValueError(f"residue coordinate omits peptide {assignments[row, 0]} amides: {missing}")
        if not active:
            raise ValueError(f"peptide {assignments[row, 0]} has no exchangeable amides")
        weight = 1.0 / len(active)
        matrix[row, [column[residue] for residue in active]] = weight
        counts.append(len(active))
    return PeptideExchangeMap(
        matrix=matrix,
        residue_ids=coordinate,
        peptide_ids=assignments[:, 0],
        peptide_starts=assignments[:, 1],
        peptide_ends=assignments[:, 2],
        active_amide_counts=np.asarray(counts, dtype=int),
        n_terminal_residues_dropped=1,
        excluded_residue_ids=tuple(sorted(excluded)),
        convention="exPfact:start_plus_one_through_end;exclude_proline_and_protein_n_terminus",
    )


def load_expfact_dataset(
    directory: str | Path,
    *,
    temperature_k: float = 298.0,
    experimental_pd: float = 4.0,
    intrinsic_rate_ph: float = 4.4,
    replicate_count: int | None = 3,
) -> ExPfactDataset:
    """Load the source exPfact validation layout without rounded-time derivatives."""

    directory = Path(directory)
    sequence = (directory / "moprp.seq").read_text().strip()
    # The validation fit is run with moprp.list.  moprp.ass is a downstream
    # clustering file and differs at peptide 8's C terminus in the published
    # validation bundle, so it must not silently replace the fitted topology.
    assignment_rows = []
    for line in (directory / "moprp.list").read_text().splitlines():
        fields = line.split()
        if fields:
            assignment_rows.append([int(fields[0]), int(fields[1]), int(fields[2])])
    assignments = np.asarray(assignment_rows, dtype=int)
    source_times_h = np.loadtxt(directory / "moprp.times", dtype=float)
    dexp = np.loadtxt(directory / "moprp.dexp", dtype=float)
    if source_times_h.shape != (dexp.shape[0] + 1,) or source_times_h[0] != 0:
        raise ValueError("moprp.times must contain t=0 followed by the measured exchange times")
    if not np.allclose(source_times_h[1:], dexp[:, 0], rtol=0.0, atol=5e-6):
        raise ValueError("moprp.times and the time column in moprp.dexp disagree")
    observed = dexp[:, 1:].T
    peptide_map = build_expfact_peptide_map(sequence, assignments)
    if observed.shape != (len(peptide_map.peptide_ids), len(source_times_h) - 1):
        raise ValueError("experimental uptake dimensions do not match peptides and timepoints")
    protocol = HDXExperimentProtocol(
        timepoints_min=source_times_h[1:] * 60.0,
        temperature_k=temperature_k,
        experimental_pd=experimental_pd,
        intrinsic_rate_ph=intrinsic_rate_ph,
        replicate_count=replicate_count,
    )
    return ExPfactDataset(
        sequence=sequence,
        protocol=protocol,
        peptide_map=peptide_map,
        observed_uptake=observed,
    )


def _validate_forward_inputs(
    log_pf: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    timepoints_min: np.ndarray,
    peptide_map: PeptideExchangeMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_pf = np.asarray(log_pf, dtype=float)
    rates = np.asarray(intrinsic_rates_min, dtype=float)
    times = np.asarray(timepoints_min, dtype=float)
    if log_pf.ndim != 1 or rates.shape != log_pf.shape:
        raise ValueError("log_pf and intrinsic_rates_min must be aligned residue vectors")
    if peptide_map.matrix.shape[1] != len(log_pf):
        raise ValueError("peptide map columns must align with residue vectors")
    if times.ndim != 1 or np.any(times < 0):
        raise ValueError("timepoints_min must be a non-negative one-dimensional array")
    active_columns = np.any(peptide_map.matrix > 0, axis=0)
    if np.any(rates[active_columns] <= 0):
        bad = peptide_map.residue_ids[active_columns & (rates <= 0)].tolist()
        raise ValueError(f"active peptide amides require positive intrinsic rates: {bad}")
    return log_pf, rates, times


def predict_ex2_uptake(
    log_pf: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    timepoints_min: np.ndarray,
    peptide_map: PeptideExchangeMap,
) -> np.ndarray:
    """Predict peptide x time fractional uptake under the native EX2 model."""

    log_pf, rates, times = _validate_forward_inputs(
        log_pf, intrinsic_rates_min, timepoints_min, peptide_map
    )
    represented = (rates > 0) & np.isfinite(log_pf)
    effective_rates = np.zeros_like(rates)
    effective_rates[represented] = rates[represented] * np.exp(-log_pf[represented])
    residue_uptake = 1.0 - np.exp(-times[:, None] * effective_rates[None, :])
    return peptide_map.matrix @ residue_uptake.T


def peptide_deuteron_count_distribution(
    log_pf: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    timepoint_min: float,
    peptide_map: PeptideExchangeMap,
    peptide_index: int,
) -> np.ndarray:
    """Poisson-binomial distribution of exchanged amide counts before quench.

    This is the residue-resolved information that a centroid discards.  Natural
    isotope convolution, quench back-exchange, charge state, and instrument
    response are deliberately not included, so the result must not be compared
    directly with a raw mass spectrum.
    """

    if timepoint_min < 0:
        raise ValueError("timepoint_min must be non-negative")
    log_pf, rates, _ = _validate_forward_inputs(
        log_pf,
        intrinsic_rates_min,
        np.asarray([timepoint_min], dtype=float),
        peptide_map,
    )
    if not 0 <= peptide_index < len(peptide_map.peptide_ids):
        raise IndexError("peptide_index is out of range")
    active = np.flatnonzero(peptide_map.matrix[peptide_index] > 0)
    probabilities = 1.0 - np.exp(
        -float(timepoint_min) * rates[active] * np.exp(-log_pf[active])
    )
    distribution = np.asarray([1.0])
    for probability in probabilities:
        updated = np.zeros(len(distribution) + 1, dtype=float)
        updated[:-1] += distribution * (1.0 - probability)
        updated[1:] += distribution * probability
        distribution = updated
    return distribution / distribution.sum()


def thin_deuteron_count_distribution(
    pre_quench_distribution: np.ndarray,
    survival_probability: float,
) -> np.ndarray:
    """Apply independent effective quench survival to incorporated deuterons."""

    distribution = np.asarray(pre_quench_distribution, dtype=float)
    if distribution.ndim != 1 or np.any(distribution < 0) or distribution.sum() <= 0:
        raise ValueError("pre_quench_distribution must be a non-negative probability vector")
    if not 0.0 <= survival_probability <= 1.0:
        raise ValueError("survival_probability must lie in [0, 1]")
    distribution = distribution / distribution.sum()
    survived = np.zeros_like(distribution)
    for incorporated, mass in enumerate(distribution):
        for retained in range(incorporated + 1):
            coefficient = float(comb(incorporated, retained))
            survived[retained] += (
                mass
                * coefficient
                * survival_probability**retained
                * (1.0 - survival_probability) ** (incorporated - retained)
            )
    return survived / survived.sum()


def convolve_isotope_and_deuteron_distributions(
    protonated_isotope_distribution: np.ndarray,
    retained_deuteron_distribution: np.ndarray,
) -> np.ndarray:
    """Convolve a measured protonated isotope baseline with retained D counts."""

    isotope = np.asarray(protonated_isotope_distribution, dtype=float)
    deuteron = np.asarray(retained_deuteron_distribution, dtype=float)
    if (
        isotope.ndim != 1
        or deuteron.ndim != 1
        or np.any(isotope < 0)
        or np.any(deuteron < 0)
        or isotope.sum() <= 0
        or deuteron.sum() <= 0
    ):
        raise ValueError("both inputs must be non-negative probability vectors")
    result = np.convolve(isotope / isotope.sum(), deuteron / deuteron.sum())
    return result / result.sum()


def predict_trajectory_ex2(
    frame_log_pf: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    timepoints_min: np.ndarray,
    peptide_map: PeptideExchangeMap,
    *,
    frame_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return average-first and static-frame-mixture peptide curves.

    ``frame_log_pf`` is residues x frames.  Average-first is the primary
    Best--Vendruscolo/HDXer ensemble construction.  Frame-mixture is retained
    only as a mechanistic sensitivity for non-interconverting subpopulations.
    """

    frame_log_pf = np.asarray(frame_log_pf, dtype=float)
    if frame_log_pf.ndim != 2:
        raise ValueError("frame_log_pf must have shape (residues, frames)")
    n_frames = frame_log_pf.shape[1]
    weights = np.full(n_frames, 1.0 / n_frames) if frame_weights is None else np.asarray(frame_weights, dtype=float)
    if weights.shape != (n_frames,) or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("frame_weights must be a finite non-negative frame vector")
    if weights.sum() <= 0:
        raise ValueError("frame_weights must have positive total mass")
    weights = weights / weights.sum()
    mean_log_pf = frame_log_pf @ weights
    average_first = predict_ex2_uptake(
        mean_log_pf, intrinsic_rates_min, timepoints_min, peptide_map
    )

    rates = np.asarray(intrinsic_rates_min, dtype=float)
    active_columns = np.any(peptide_map.matrix > 0, axis=0)
    if rates.shape != (frame_log_pf.shape[0],) or np.any(rates[active_columns] <= 0):
        raise ValueError("intrinsic rates must align with frame_log_pf and be positive on active amides")
    represented = (rates > 0)[:, None] & np.isfinite(frame_log_pf)
    effective = np.zeros_like(frame_log_pf)
    effective[represented] = (
        np.broadcast_to(rates[:, None], frame_log_pf.shape)[represented]
        * np.exp(-frame_log_pf[represented])
    )
    residue_by_time = np.einsum(
        "rtf,f->tr",
        1.0 - np.exp(-np.asarray(timepoints_min)[None, :, None] * effective[:, None, :]),
        weights,
    )
    frame_mixture = peptide_map.matrix @ residue_by_time.T
    return average_first, frame_mixture


def compare_trajectory_hdx(
    frame_log_pf: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    protocol: HDXExperimentProtocol,
    peptide_map: PeptideExchangeMap,
    *,
    observed_uptake: np.ndarray | None = None,
    frame_weights: np.ndarray | None = None,
) -> TrajectoryHDXComparison:
    average_first, frame_mixture = predict_trajectory_ex2(
        frame_log_pf,
        intrinsic_rates_min,
        protocol.timepoints_min,
        peptide_map,
        frame_weights=frame_weights,
    )
    if observed_uptake is None:
        average_rmse = mixture_rmse = None
    else:
        observed = np.asarray(observed_uptake, dtype=float)
        if observed.shape != average_first.shape:
            raise ValueError("observed_uptake must have peptide x time shape")
        average_rmse = float(np.sqrt(np.mean((average_first - observed) ** 2)))
        mixture_rmse = float(np.sqrt(np.mean((frame_mixture - observed) ** 2)))
    return TrajectoryHDXComparison(
        average_first_curves=average_first,
        frame_mixture_curves=frame_mixture,
        average_first_rmse=average_rmse,
        frame_mixture_rmse=mixture_rmse,
    )


def fit_ex2_solution_set(
    observed_uptake: np.ndarray,
    intrinsic_rates_min: np.ndarray,
    timepoints_min: np.ndarray,
    peptide_map: PeptideExchangeMap,
    *,
    starts: int = 20,
    seed: int = 1729,
    log_pf_bounds: tuple[float, float] = (1e-5, 30.0),
    harmonic_strength: float = 0.0,
    random_search_steps: int = 0,
    random_search_batch_size: int = 256,
    initial_log_pf_vectors: Sequence[np.ndarray] | None = None,
    maxiter: int = 5000,
) -> EX2SolutionSet:
    """Fit all represented residue log-PFs and retain every finite start.

    The objective reproduces exPfact's unweighted ``sum(SSE)/n_peptides``
    convention.  The returned solution range is a multistart sensitivity, not
    a calibrated confidence interval.
    """

    observed = np.asarray(observed_uptake, dtype=float)
    rates = np.asarray(intrinsic_rates_min, dtype=float)
    times = np.asarray(timepoints_min, dtype=float)
    if observed.shape != (peptide_map.matrix.shape[0], len(times)):
        raise ValueError("observed_uptake must have peptide x time shape")
    represented = np.flatnonzero(np.any(peptide_map.matrix > 0, axis=0))
    if np.any(rates[represented] <= 0):
        bad = peptide_map.residue_ids[represented[rates[represented] <= 0]].tolist()
        raise ValueError(f"active residues require positive intrinsic rates: {bad}")
    fit_indices = represented.copy()
    # exPfact's source harmonic construction also fits the residue immediately
    # preceding the first represented amide.  It has zero uptake weight and is
    # constrained only through the adjacent second-difference penalty.  This
    # boundary residue must not be introduced in the unregularized fit.
    if harmonic_strength:
        first_residue = int(peptide_map.residue_ids[represented].min())
        context = np.flatnonzero(peptide_map.residue_ids == first_residue - 1)
        if len(context) == 1 and rates[context[0]] > 0:
            fit_indices = np.sort(np.concatenate((fit_indices, context)))
    active_map = jnp.asarray(peptide_map.matrix[:, fit_indices])
    active_rates = jnp.asarray(rates[fit_indices])
    times_j = jnp.asarray(times)
    observed_j = jnp.asarray(observed)
    active_ids = peptide_map.residue_ids[fit_indices]
    adjacent_triplets = np.asarray(
        [
            (index - 1, index, index + 1)
            for index in range(1, len(active_ids) - 1)
            if active_ids[index - 1] + 1 == active_ids[index]
            and active_ids[index] + 1 == active_ids[index + 1]
        ],
        dtype=int,
    )
    triplets_j = jnp.asarray(adjacent_triplets.reshape(-1, 3))

    def objective(active_log_pf: jax.Array) -> jax.Array:
        effective = active_rates * jnp.exp(-active_log_pf)
        residue_uptake = 1.0 - jnp.exp(-times_j[:, None] * effective[None, :])
        predicted = active_map @ residue_uptake.T
        score = jnp.sum((predicted - observed_j) ** 2) / observed.shape[0]
        if harmonic_strength and len(adjacent_triplets):
            values = active_log_pf[triplets_j]
            score = score + harmonic_strength * jnp.sum(
                (values[:, 0] - 2.0 * values[:, 1] + values[:, 2]) ** 2
            )
        return score

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    batched_objective = jax.jit(jax.vmap(objective))

    def scipy_objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(values))
        return float(value), np.asarray(gradient, dtype=float)

    rng = np.random.default_rng(seed)
    solutions: list[EX2Fit] = []
    bounds = [log_pf_bounds] * len(fit_indices)
    seeded_initials: list[tuple[str, np.ndarray]] = [
        ("fixed_log_pf_5_reference", np.full(len(fit_indices), 5.0, dtype=float))
    ]
    for seed_index, vector in enumerate(initial_log_pf_vectors or ()):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != rates.shape:
            raise ValueError("each initial_log_pf_vector must align with intrinsic rates")
        values = vector[fit_indices]
        if np.any(~np.isfinite(values)):
            raise ValueError("initial_log_pf_vectors must be finite on fitted residues")
        seeded_initials.append(
            (
                f"provided_seed_{seed_index}",
                np.clip(values, log_pf_bounds[0], log_pf_bounds[1]),
            )
        )
    n_runs = max(1, starts, len(seeded_initials))
    for start in range(n_runs):
        use_source_random_search = random_search_steps > 0 and start >= len(seeded_initials)
        if start < len(seeded_initials):
            initialization, initial = seeded_initials[start]
            initial = initial.copy()
        else:
            initialization = (
                f"source_random_best_of_{random_search_steps}"
                if use_source_random_search
                else "direct_random"
            )
            initial = np.full(len(fit_indices), 5.0, dtype=float)
        best_initial_value = (
            np.inf
            if use_source_random_search
            else float(objective(jnp.asarray(initial)))
        )
        remaining = max(0, random_search_steps) if use_source_random_search else 0
        while remaining:
            batch_size = min(remaining, max(1, random_search_batch_size))
            candidates = rng.uniform(
                log_pf_bounds[0],
                log_pf_bounds[1],
                size=(batch_size, len(fit_indices)),
            )
            candidate_values = np.asarray(batched_objective(jnp.asarray(candidates)))
            candidate_index = int(np.argmin(candidate_values))
            if float(candidate_values[candidate_index]) < best_initial_value:
                initial = candidates[candidate_index]
                best_initial_value = float(candidate_values[candidate_index])
            remaining -= batch_size
        if random_search_steps == 0 and start >= len(seeded_initials):
            initial = rng.uniform(log_pf_bounds[0], log_pf_bounds[1], size=len(fit_indices))
        result = minimize(
            scipy_objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not np.isfinite(result.fun) or np.any(~np.isfinite(result.x)):
            continue
        full_log_pf = np.full(len(rates), np.nan, dtype=float)
        full_log_pf[fit_indices] = result.x
        predicted = predict_ex2_uptake(full_log_pf, rates, times, peptide_map)
        solutions.append(
            EX2Fit(
                log_pf=full_log_pf,
                predicted=predicted,
                objective=float(result.fun),
                rmse=float(np.sqrt(np.mean((predicted - observed) ** 2))),
                success=bool(result.success),
                message=str(result.message),
                iterations=int(result.nit),
                gradient_norm=float(np.linalg.norm(result.jac)),
                initialization=initialization,
            )
        )
    if not solutions:
        raise RuntimeError("all EX2 optimization starts returned non-finite results")
    solutions.sort(key=lambda solution: solution.objective)
    return EX2SolutionSet(
        residue_ids=peptide_map.residue_ids,
        solutions=tuple(solutions),
        best_index=0,
    )
