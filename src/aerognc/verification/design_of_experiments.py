"""Direct reproducible sampling, screening, correlation, and bootstrap utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class Factor:
    """One finite engineering-input interval."""

    name: str
    lower: float
    upper: float
    unit: str = "1"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.unit.strip():
            raise ValueError("factor name and unit cannot be empty")
        if not np.all(np.isfinite([self.lower, self.upper])) or self.upper <= self.lower:
            raise ValueError("factor bounds must be finite with upper greater than lower")


@dataclass(frozen=True, slots=True)
class DesignMatrix:
    """Unit-hypercube and mapped physical experiment points."""

    method: str
    factors: tuple[Factor, ...]
    unit_samples: FloatArray
    samples: FloatArray
    seed: int | None

    def __post_init__(self) -> None:
        if not self.method.strip() or not self.factors:
            raise ValueError("design method and factors cannot be empty")
        unit = np.asarray(self.unit_samples, dtype=np.float64)
        physical = np.asarray(self.samples, dtype=np.float64)
        if unit.ndim != 2 or unit.shape[1] != len(self.factors) or physical.shape != unit.shape:
            raise ValueError("design sample matrices must match the factor count")
        if not np.all(np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("unit design samples must lie in [0, 1]")
        validate_samples_in_domain(physical, self.factors)
        object.__setattr__(self, "unit_samples", unit.copy())
        object.__setattr__(self, "samples", physical.copy())


@dataclass(frozen=True, slots=True)
class MorrisDesign:
    """Randomized one-at-a-time trajectories and step bookkeeping."""

    design: DesignMatrix
    trajectory_index: npt.NDArray[np.int64]
    changed_factor_index: npt.NDArray[np.int64]
    signed_step: FloatArray


@dataclass(frozen=True, slots=True)
class MorrisEffect:
    """Elementary-effect distribution summary for one factor."""

    factor_name: str
    mean: float
    mean_absolute: float
    standard_deviation: float
    effects: FloatArray


@dataclass(frozen=True, slots=True)
class FactorCorrelation:
    """Linear Pearson and average-rank Spearman coefficients."""

    factor_name: str
    linear: float
    rank: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Seeded percentile bootstrap interval for one scalar statistic."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int


def _validated_factors(factors: Sequence[Factor]) -> tuple[Factor, ...]:
    result = tuple(factors)
    if not result:
        raise ValueError("at least one design factor is required")
    names = tuple(factor.name for factor in result)
    if len(set(names)) != len(names):
        raise ValueError("design factor names must be unique")
    return result


def _map_unit_samples(unit_samples: FloatArray, factors: tuple[Factor, ...]) -> FloatArray:
    lower = np.array([factor.lower for factor in factors])
    extent = np.array([factor.upper - factor.lower for factor in factors])
    return np.asarray(lower + unit_samples * extent, dtype=np.float64)


def validate_samples_in_domain(samples: npt.ArrayLike, factors: Sequence[Factor]) -> None:
    """Reject nonfinite samples outside the closed declared factor domain."""
    factor_tuple = _validated_factors(factors)
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(factor_tuple):
        raise ValueError("sample matrix column count must equal factor count")
    if not np.all(np.isfinite(values)):
        raise ValueError("design samples must be finite")
    lower = np.array([factor.lower for factor in factor_tuple])
    upper = np.array([factor.upper for factor in factor_tuple])
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("design sample lies outside a declared factor domain")


def latin_hypercube_design(
    factors: Sequence[Factor],
    sample_count: int,
    *,
    seed: int,
) -> DesignMatrix:
    """Generate a seeded, independently permuted, jittered Latin hypercube."""
    factor_tuple = _validated_factors(factors)
    if sample_count <= 0:
        raise ValueError("Latin-hypercube sample_count must be positive")
    generator = np.random.default_rng(seed)
    unit = np.empty((sample_count, len(factor_tuple)))
    for column in range(len(factor_tuple)):
        strata = (np.arange(sample_count) + generator.random(sample_count)) / sample_count
        unit[:, column] = strata[generator.permutation(sample_count)]
    return DesignMatrix(
        "latin_hypercube",
        factor_tuple,
        unit,
        _map_unit_samples(unit, factor_tuple),
        seed,
    )


# Bratley--Fox primitive-polynomial parameters for the first 16 dimensions.
# Each entry after dimension one is (degree, coefficient bits, initial odd numerators).
_SOBOL_PARAMETERS: tuple[tuple[int, int, tuple[int, ...]], ...] = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 3, 5, 13)),
    (4, 4, (1, 1, 5, 5)),
    (5, 2, (1, 3, 3, 9, 7)),
    (5, 4, (1, 1, 3, 11, 13)),
    (5, 7, (1, 1, 5, 1, 15)),
    (5, 11, (1, 1, 7, 3, 1)),
    (5, 13, (1, 3, 7, 7, 9)),
    (5, 14, (1, 3, 5, 5, 3)),
    (6, 1, (1, 3, 1, 15, 11, 5)),
    (6, 13, (1, 1, 3, 13, 7, 35)),
    (6, 16, (1, 3, 1, 5, 19, 33)),
)


def _sobol_direction_numbers(dimension_count: int, bits: int = 32) -> npt.NDArray[np.uint32]:
    directions = np.zeros((dimension_count, bits), dtype=np.uint32)
    for bit in range(bits):
        directions[0, bit] = np.uint32(1 << (bits - bit - 1))
    for dimension in range(1, dimension_count):
        degree, coefficient, initial = _SOBOL_PARAMETERS[dimension - 1]
        for bit in range(degree):
            directions[dimension, bit] = np.uint32(initial[bit] << (bits - bit - 1))
        for bit in range(degree, bits):
            value = directions[dimension, bit - degree]
            value ^= value >> np.uint32(degree)
            for offset in range(1, degree):
                if (coefficient >> (degree - 1 - offset)) & 1:
                    value ^= directions[dimension, bit - offset]
            directions[dimension, bit] = value
    return directions


def sobol_design(
    factors: Sequence[Factor],
    sample_count: int,
    *,
    skip: int = 0,
) -> DesignMatrix:
    """Generate an unscrambled deterministic Sobol sequence in up to 16 dimensions."""
    factor_tuple = _validated_factors(factors)
    if not 1 <= len(factor_tuple) <= 16:
        raise ValueError("direct Sobol implementation supports 1 through 16 factors")
    if sample_count <= 0 or skip < 0 or sample_count + skip > 2**32:
        raise ValueError("Sobol sample_count must be positive and skip/range valid")
    directions = _sobol_direction_numbers(len(factor_tuple))
    state = np.zeros(len(factor_tuple), dtype=np.uint32)
    sequence = np.empty((sample_count + skip, len(factor_tuple)), dtype=np.float64)
    sequence[0] = 0.0
    for index in range(1, sample_count + skip):
        direction_index = (index & -index).bit_length() - 1
        state ^= directions[:, direction_index]
        sequence[index] = state.astype(np.float64) / 2.0**32
    unit = sequence[skip:].copy()
    return DesignMatrix("sobol", factor_tuple, unit, _map_unit_samples(unit, factor_tuple), None)


def morris_design(
    factors: Sequence[Factor],
    trajectory_count: int,
    *,
    levels: int = 4,
    seed: int,
) -> MorrisDesign:
    """Generate seeded Morris one-at-a-time trajectories on an even-level grid."""
    factor_tuple = _validated_factors(factors)
    if trajectory_count <= 0:
        raise ValueError("Morris trajectory_count must be positive")
    if levels < 4 or levels % 2:
        raise ValueError("Morris levels must be an even integer of at least four")
    factor_count = len(factor_tuple)
    delta = levels / (2.0 * (levels - 1.0))
    allowed = np.arange(levels // 2, dtype=np.float64) / (levels - 1.0)
    generator = np.random.default_rng(seed)
    points: list[FloatArray] = []
    trajectories: list[int] = []
    changed: list[int] = []
    steps: list[float] = []
    for trajectory in range(trajectory_count):
        base = generator.choice(allowed, size=factor_count).astype(np.float64)
        directions = generator.choice(np.array([-1.0, 1.0]), size=factor_count)
        base = np.where(directions > 0.0, base, 1.0 - base)
        order = generator.permutation(factor_count)
        current = np.asarray(base, dtype=np.float64)
        points.append(current.copy())
        trajectories.append(trajectory)
        for factor_index in order:
            signed_delta = float(directions[factor_index] * delta)
            current = current.copy()
            current[factor_index] += signed_delta
            if not -1.0e-14 <= current[factor_index] <= 1.0 + 1.0e-14:
                raise RuntimeError("Morris trajectory escaped the unit domain")
            current[factor_index] = np.clip(current[factor_index], 0.0, 1.0)
            points.append(current.copy())
            trajectories.append(trajectory)
            changed.append(int(factor_index))
            steps.append(signed_delta)
    unit = np.vstack(points)
    return MorrisDesign(
        DesignMatrix("morris", factor_tuple, unit, _map_unit_samples(unit, factor_tuple), seed),
        np.asarray(trajectories, dtype=np.int64),
        np.asarray(changed, dtype=np.int64),
        np.asarray(steps, dtype=np.float64),
    )


def morris_elementary_effects(
    design: MorrisDesign, responses: npt.ArrayLike
) -> tuple[MorrisEffect, ...]:
    """Calculate normalized elementary effects from trajectory-ordered responses."""
    values = np.asarray(responses, dtype=np.float64)
    if values.shape != (design.design.unit_samples.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError("Morris responses must be one finite value per design point")
    factor_count = len(design.design.factors)
    effects: list[list[float]] = [[] for _ in range(factor_count)]
    step_index = 0
    row_index = 0
    for _trajectory in range(int(np.max(design.trajectory_index)) + 1):
        for _ in range(factor_count):
            factor_index = int(design.changed_factor_index[step_index])
            effect = (values[row_index + 1] - values[row_index]) / design.signed_step[step_index]
            effects[factor_index].append(float(effect))
            step_index += 1
            row_index += 1
        row_index += 1
    return tuple(
        MorrisEffect(
            factor.name,
            float(np.mean(factor_effects)),
            float(np.mean(np.abs(factor_effects))),
            float(np.std(factor_effects, ddof=1)) if len(factor_effects) > 1 else 0.0,
            np.asarray(factor_effects),
        )
        for factor, factor_effects in zip(design.design.factors, effects, strict=True)
    )


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(first: FloatArray, second: FloatArray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    return 0.0 if denominator == 0.0 else float(first_centered @ second_centered / denominator)


def sensitivity_correlations(
    samples: npt.ArrayLike,
    responses: npt.ArrayLike,
    factors: Sequence[Factor],
) -> tuple[FactorCorrelation, ...]:
    """Return direct Pearson and average-rank Spearman screening coefficients."""
    factor_tuple = _validated_factors(factors)
    design = np.asarray(samples, dtype=np.float64)
    values = np.asarray(responses, dtype=np.float64)
    validate_samples_in_domain(design, factor_tuple)
    if values.shape != (design.shape[0],) or design.shape[0] < 3:
        raise ValueError("correlation responses require one value per 3+ design rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("correlation responses must be finite")
    response_ranks = _average_ranks(values)
    return tuple(
        FactorCorrelation(
            factor.name,
            _correlation(design[:, index], values),
            _correlation(_average_ranks(design[:, index]), response_ranks),
        )
        for index, factor in enumerate(factor_tuple)
    )


def bootstrap_confidence_interval(
    values: npt.ArrayLike,
    statistic: Callable[[FloatArray], float] | None = None,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int,
) -> BootstrapInterval:
    """Return a seeded nonparametric percentile confidence interval."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap values must be a finite vector with at least two samples")
    if not 0.0 < confidence < 1.0 or resamples <= 0:
        raise ValueError("bootstrap confidence must lie in (0, 1) and resamples be positive")
    function = (lambda item: float(np.mean(item))) if statistic is None else statistic
    estimate = float(function(samples))
    if not np.isfinite(estimate):
        raise ValueError("bootstrap statistic returned a nonfinite estimate")
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        resampled = samples[generator.integers(0, samples.size, size=samples.size)]
        estimates[index] = function(resampled)
    if not np.all(np.isfinite(estimates)):
        raise ValueError("bootstrap statistic returned a nonfinite resample")
    tail = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return BootstrapInterval(estimate, float(lower), float(upper), confidence, resamples, seed)
