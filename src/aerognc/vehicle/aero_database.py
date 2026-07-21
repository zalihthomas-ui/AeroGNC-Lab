"""Validated regular-grid aerodynamic coefficient database and CSV import."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.interpolation import OutOfRange, RegularGridTableND
from aerognc.mathematics.vectors import FloatArray, as_vector
from aerognc.vehicle.aerodynamics import AerodynamicCoefficients

COEFFICIENT_NAMES = ("drag", "side", "normal", "roll", "pitch", "yaw")
SUPPORTED_AXIS_NAMES = (
    "mach",
    "alpha_rad",
    "beta_rad",
    "p_hat",
    "q_hat",
    "r_hat",
    "roll_control",
    "pitch_control",
    "yaw_control",
    "reynolds",
)


@dataclass(frozen=True, slots=True)
class AerodynamicCondition:
    """Complete dimensionless/radian aerodynamic database query."""

    mach: float
    alpha_rad: float = 0.0
    beta_rad: float = 0.0
    p_hat: float = 0.0
    q_hat: float = 0.0
    r_hat: float = 0.0
    roll_control: float = 0.0
    pitch_control: float = 0.0
    yaw_control: float = 0.0
    reynolds: float = 0.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.as_array())):
            raise ValueError("aerodynamic condition must contain finite values")

    def as_mapping(self) -> dict[str, float]:
        """Return canonical axis-name mapping."""
        return {
            "mach": self.mach,
            "alpha_rad": self.alpha_rad,
            "beta_rad": self.beta_rad,
            "p_hat": self.p_hat,
            "q_hat": self.q_hat,
            "r_hat": self.r_hat,
            "roll_control": self.roll_control,
            "pitch_control": self.pitch_control,
            "yaw_control": self.yaw_control,
            "reynolds": self.reynolds,
        }

    def as_array(self) -> FloatArray:
        """Return values in ``SUPPORTED_AXIS_NAMES`` order."""
        mapping = self.as_mapping()
        return np.array([mapping[name] for name in SUPPORTED_AXIS_NAMES], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AerodynamicDatabaseDiagnostics:
    """Interpolation-domain evidence for one database query."""

    outside_axes: tuple[str, ...]
    source_path: Path | None
    source_sha256: str | None

    @property
    def inside_domain(self) -> bool:
        """Return true when every coordinate is within its tabulated axis."""
        return not self.outside_axes


class TabulatedAerodynamicDatabase:
    """Six body-axis coefficient tables sharing one regular grid."""

    def __init__(
        self,
        *,
        axis_names: tuple[str, ...],
        axes: tuple[npt.ArrayLike, ...],
        coefficient_values: Mapping[str, npt.ArrayLike],
        out_of_range: OutOfRange = "error",
        source_path: Path | None = None,
        source_sha256: str | None = None,
    ) -> None:
        if not axis_names or len(axis_names) != len(axes):
            raise ValueError("axis_names and axes must have the same nonzero length")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("axis_names must be unique")
        unknown_axes = set(axis_names) - set(SUPPORTED_AXIS_NAMES)
        if unknown_axes:
            raise ValueError(f"unsupported aerodynamic axes: {', '.join(sorted(unknown_axes))}")
        if set(coefficient_values) != set(COEFFICIENT_NAMES):
            raise ValueError(f"coefficient_values must contain exactly {COEFFICIENT_NAMES}")
        self.axis_names = axis_names
        self.tables = {
            name: RegularGridTableND(axes, coefficient_values[name], out_of_range)
            for name in COEFFICIENT_NAMES
        }
        self.out_of_range = out_of_range
        self.source_path = source_path
        self.source_sha256 = source_sha256

    @property
    def axes(self) -> tuple[FloatArray, ...]:
        """Return immutable-by-convention copies of shared interpolation axes."""
        return tuple(axis.copy() for axis in self.tables["drag"].axes)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        out_of_range: OutOfRange = "error",
    ) -> TabulatedAerodynamicDatabase:
        """Load a complete tensor-product table from a long-form CSV file."""
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise ValueError(f"aerodynamic database does not exist: {source_path}")
        raw_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("aerodynamic CSV must contain a header")
            coefficient_columns = set(COEFFICIENT_NAMES)
            axis_names = tuple(name for name in fieldnames if name not in coefficient_columns)
            if set(fieldnames) != set(axis_names) | coefficient_columns:
                raise ValueError("aerodynamic CSV contains duplicate or invalid columns")
            if not axis_names:
                raise ValueError("aerodynamic CSV must contain at least one axis column")
            rows: list[dict[str, float]] = []
            for row_number, raw_row in enumerate(reader, start=2):
                parsed: dict[str, float] = {}
                for field in fieldnames:
                    try:
                        value = float(raw_row[field])
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"aerodynamic CSV row {row_number} field {field!r} is not numeric"
                        ) from error
                    if not np.isfinite(value):
                        raise ValueError(
                            f"aerodynamic CSV row {row_number} field {field!r} is not finite"
                        )
                    parsed[field] = value
                rows.append(parsed)
        if not rows:
            raise ValueError("aerodynamic CSV contains no data rows")
        axes = tuple(
            np.array(sorted({row[name] for row in rows}), dtype=np.float64) for name in axis_names
        )
        expected_count = int(np.prod([axis.size for axis in axes]))
        if len(rows) != expected_count:
            raise ValueError(
                "aerodynamic CSV must contain one row for every tensor-product grid point"
            )
        index_maps = tuple(
            {float(value): index for index, value in enumerate(axis)} for axis in axes
        )
        shape = tuple(axis.size for axis in axes)
        values = {name: np.full(shape, np.nan, dtype=np.float64) for name in COEFFICIENT_NAMES}
        seen: set[tuple[int, ...]] = set()
        for data_row in rows:
            index = tuple(
                index_map[data_row[axis_name]]
                for axis_name, index_map in zip(axis_names, index_maps, strict=True)
            )
            if index in seen:
                raise ValueError("aerodynamic CSV contains a duplicate grid point")
            seen.add(index)
            for name in COEFFICIENT_NAMES:
                values[name][index] = data_row[name]
        if any(np.isnan(table).any() for table in values.values()):
            raise ValueError("aerodynamic CSV grid is incomplete")
        return cls(
            axis_names=axis_names,
            axes=axes,
            coefficient_values=values,
            out_of_range=out_of_range,
            source_path=source_path,
            source_sha256=source_sha256,
        )

    def _queries(self, condition: AerodynamicCondition) -> tuple[float, ...]:
        mapping = condition.as_mapping()
        return tuple(mapping[name] for name in self.axis_names)

    def evaluate(self, condition: AerodynamicCondition) -> AerodynamicCoefficients:
        """Interpolate all six coefficients at one condition."""
        queries = self._queries(condition)
        values = {name: table(*queries) for name, table in self.tables.items()}
        return AerodynamicCoefficients(**values)

    def coefficients(
        self,
        mach: float,
        alpha_rad: float,
        beta_rad: float,
        nondimensional_rates: npt.ArrayLike = (0.0, 0.0, 0.0),
        control_coefficients: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> AerodynamicCoefficients:
        """Implement the common coefficient-provider interface."""
        p_hat, q_hat, r_hat = as_vector(nondimensional_rates, 3, name="nondimensional_rates")
        roll_control, pitch_control, yaw_control = as_vector(
            control_coefficients, 3, name="control_coefficients"
        )
        return self.evaluate(
            AerodynamicCondition(
                mach=mach,
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
                p_hat=p_hat,
                q_hat=q_hat,
                r_hat=r_hat,
                roll_control=roll_control,
                pitch_control=pitch_control,
                yaw_control=yaw_control,
            )
        )

    def coefficient_jacobian(
        self, condition: AerodynamicCondition
    ) -> tuple[tuple[str, ...], FloatArray]:
        """Return coefficient Jacobian with rows in ``COEFFICIENT_NAMES`` order."""
        queries = self._queries(condition)
        rows = [self.tables[name].value_and_gradient(*queries)[1] for name in COEFFICIENT_NAMES]
        return self.axis_names, np.vstack(rows)

    def diagnostics(self, condition: AerodynamicCondition) -> AerodynamicDatabaseDiagnostics:
        """Report which named axes are queried outside their database range."""
        outside_indices = self.tables["drag"].outside_axes(*self._queries(condition))
        return AerodynamicDatabaseDiagnostics(
            outside_axes=tuple(self.axis_names[index] for index in outside_indices),
            source_path=self.source_path,
            source_sha256=self.source_sha256,
        )
