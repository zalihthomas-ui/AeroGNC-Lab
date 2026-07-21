"""Small validated interpolation tables with explicit boundary behaviour."""

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

OutOfRange = Literal["error", "clamp", "extrapolate"]


def _validated_axis(values: npt.ArrayLike, name: str) -> FloatArray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis.copy()


def _check_policy(policy: str) -> OutOfRange:
    if policy not in {"error", "clamp", "extrapolate"}:
        raise ValueError("out_of_range must be 'error', 'clamp', or 'extrapolate'")
    return policy  # type: ignore[return-value]


def _interval(axis: FloatArray, value: float, policy: OutOfRange) -> tuple[int, float]:
    if not np.isfinite(value):
        raise ValueError("interpolation query must be finite")
    if policy == "error" and not axis[0] <= value <= axis[-1]:
        raise ValueError(f"query {value} outside table range [{axis[0]}, {axis[-1]}]")
    query = float(np.clip(value, axis[0], axis[-1])) if policy == "clamp" else value
    index = int(np.searchsorted(axis, query, side="right") - 1)
    index = min(max(index, 0), axis.size - 2)
    fraction = (query - axis[index]) / (axis[index + 1] - axis[index])
    return index, float(fraction)


@dataclass(frozen=True, slots=True)
class LinearTable1D:
    """Piecewise-linear scalar table."""

    x: FloatArray
    y: FloatArray
    out_of_range: OutOfRange = "error"

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        out_of_range: OutOfRange = "error",
    ) -> None:
        x_axis = _validated_axis(x, "x")
        y_values = np.asarray(y, dtype=np.float64)
        if y_values.shape != x_axis.shape or not np.all(np.isfinite(y_values)):
            raise ValueError("y must be finite and have the same shape as x")
        object.__setattr__(self, "x", x_axis)
        object.__setattr__(self, "y", y_values.copy())
        object.__setattr__(self, "out_of_range", _check_policy(out_of_range))

    def __call__(self, query: float) -> float:
        """Interpolate one scalar query."""
        index, fraction = _interval(self.x, float(query), self.out_of_range)
        return float(self.y[index] + fraction * (self.y[index + 1] - self.y[index]))


@dataclass(frozen=True, slots=True)
class BilinearTable2D:
    """Bilinear table where ``values[i, j]`` corresponds to ``x[i], y[j]``."""

    x: FloatArray
    y: FloatArray
    values: FloatArray
    out_of_range: OutOfRange = "error"

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        values: npt.ArrayLike,
        out_of_range: OutOfRange = "error",
    ) -> None:
        x_axis = _validated_axis(x, "x")
        y_axis = _validated_axis(y, "y")
        table = np.asarray(values, dtype=np.float64)
        if table.shape != (x_axis.size, y_axis.size) or not np.all(np.isfinite(table)):
            raise ValueError("values must be finite with shape (len(x), len(y))")
        object.__setattr__(self, "x", x_axis)
        object.__setattr__(self, "y", y_axis)
        object.__setattr__(self, "values", table.copy())
        object.__setattr__(self, "out_of_range", _check_policy(out_of_range))

    def __call__(self, x_query: float, y_query: float) -> float:
        """Interpolate one pair of scalar queries."""
        i, tx = _interval(self.x, float(x_query), self.out_of_range)
        j, ty = _interval(self.y, float(y_query), self.out_of_range)
        lower = self.values[i, j] * (1.0 - tx) + self.values[i + 1, j] * tx
        upper = self.values[i, j + 1] * (1.0 - tx) + self.values[i + 1, j + 1] * tx
        return float(lower * (1.0 - ty) + upper * ty)


@dataclass(frozen=True, slots=True)
class RegularGridTableND:
    """N-dimensional multilinear scalar table on strictly increasing axes."""

    axes: tuple[FloatArray, ...]
    values: FloatArray
    out_of_range: OutOfRange = "error"

    def __init__(
        self,
        axes: tuple[npt.ArrayLike, ...],
        values: npt.ArrayLike,
        out_of_range: OutOfRange = "error",
    ) -> None:
        if not axes:
            raise ValueError("axes must contain at least one interpolation axis")
        validated_axes = tuple(
            _validated_axis(axis, f"axes[{index}]") for index, axis in enumerate(axes)
        )
        if len(validated_axes) > 8:
            raise ValueError("at most eight interpolation axes are supported")
        table = np.asarray(values, dtype=np.float64)
        expected_shape = tuple(axis.size for axis in validated_axes)
        if table.shape != expected_shape or not np.all(np.isfinite(table)):
            raise ValueError(f"values must be finite with shape {expected_shape}")
        object.__setattr__(self, "axes", validated_axes)
        object.__setattr__(self, "values", table.copy())
        object.__setattr__(self, "out_of_range", _check_policy(out_of_range))

    def _coordinates(
        self, queries: tuple[float, ...]
    ) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
        if len(queries) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} queries, received {len(queries)}")
        indices: list[int] = []
        fractions: list[float] = []
        derivative_scales: list[float] = []
        for axis, query in zip(self.axes, queries, strict=True):
            index, fraction = _interval(axis, float(query), self.out_of_range)
            indices.append(index)
            fractions.append(fraction)
            clamped_outside = self.out_of_range == "clamp" and not axis[0] <= query <= axis[-1]
            derivative_scales.append(
                0.0 if clamped_outside else 1.0 / (axis[index + 1] - axis[index])
            )
        return tuple(indices), tuple(fractions), tuple(derivative_scales)

    def __call__(self, *queries: float) -> float:
        """Interpolate one point with tensor-product multilinear weights."""
        indices, fractions, _ = self._coordinates(tuple(float(query) for query in queries))
        result = 0.0
        for corner in product((0, 1), repeat=len(self.axes)):
            weight = 1.0
            table_index: list[int] = []
            for dimension, upper in enumerate(corner):
                fraction = fractions[dimension]
                weight *= fraction if upper else 1.0 - fraction
                table_index.append(indices[dimension] + upper)
            result += weight * float(self.values[tuple(table_index)])
        return float(result)

    def value_and_gradient(self, *queries: float) -> tuple[float, FloatArray]:
        """Return interpolated value and exact within-cell partial derivatives."""
        indices, fractions, derivative_scales = self._coordinates(
            tuple(float(query) for query in queries)
        )
        value = 0.0
        gradient = np.zeros(len(self.axes), dtype=np.float64)
        for corner in product((0, 1), repeat=len(self.axes)):
            table_index = tuple(indices[d] + corner[d] for d in range(len(self.axes)))
            corner_value = float(self.values[table_index])
            weights = tuple(
                fractions[d] if corner[d] else 1.0 - fractions[d] for d in range(len(self.axes))
            )
            value += corner_value * float(np.prod(weights))
            for derivative_dimension in range(len(self.axes)):
                derivative_weight = derivative_scales[derivative_dimension] * (
                    1.0 if corner[derivative_dimension] else -1.0
                )
                for dimension, weight in enumerate(weights):
                    if dimension != derivative_dimension:
                        derivative_weight *= weight
                gradient[derivative_dimension] += corner_value * derivative_weight
        return float(value), gradient

    def outside_axes(self, *queries: float) -> tuple[int, ...]:
        """Return indices of axes whose query lies outside the tabulated domain."""
        if len(queries) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} queries, received {len(queries)}")
        if not np.all(np.isfinite(queries)):
            raise ValueError("interpolation queries must be finite")
        return tuple(
            index
            for index, (axis, query) in enumerate(zip(self.axes, queries, strict=True))
            if query < axis[0] or query > axis[-1]
        )
