"""Strict unit-bearing CSV boundaries for synthetic engineering data."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from aerognc.mathematics.interpolation import OutOfRange
from aerognc.mathematics.vectors import FloatArray
from aerognc.vehicle.aero_database import (
    COEFFICIENT_NAMES,
    SUPPORTED_AXIS_NAMES,
    TabulatedAerodynamicDatabase,
)
from aerognc.vehicle.propulsion import ThrustCurve

MassOutOfRange = Literal["error", "clamp"]

THRUST_COLUMNS = ("time_s", "thrust_n")
MASS_PROPERTY_COLUMNS = (
    "time_s",
    "mass_kg",
    "cg_from_nose_m",
    "inertia_xx_kgm2",
    "inertia_yy_kgm2",
    "inertia_zz_kgm2",
    "inertia_xy_kgm2",
    "inertia_xz_kgm2",
    "inertia_yz_kgm2",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_numeric_csv(path: str | Path, expected_columns: Sequence[str]) -> tuple[Path, FloatArray]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"engineering data file does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{source.name}: CSV header is required")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{source.name}: duplicate CSV columns are not allowed")
        if tuple(reader.fieldnames) != tuple(expected_columns):
            raise ValueError(
                f"{source.name}: expected unit-bearing columns {tuple(expected_columns)}, "
                f"received {tuple(reader.fieldnames)}"
            )
        rows: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            parsed: list[float] = []
            for name in expected_columns:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{source.name}: row {row_number} column {name!r} must be numeric"
                    ) from error
                if not np.isfinite(value):
                    raise ValueError(
                        f"{source.name}: row {row_number} column {name!r} must be finite"
                    )
                parsed.append(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{source.name}: at least one data row is required")
    return source, np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class EngineeringDataProvenance:
    """Source identity and declared interpolation boundary behavior."""

    source_path: Path
    source_sha256: str
    interpolation: str
    extrapolation: str


@dataclass(frozen=True, slots=True)
class ImportedThrustData:
    """Thrust curve plus reproducibility metadata."""

    curve: ThrustCurve
    provenance: EngineeringDataProvenance


def import_thrust_csv(
    path: str | Path,
    *,
    propellant_mass_kg: float,
    extrapolation: Literal["zero"] = "zero",
) -> ImportedThrustData:
    """Import `time_s,thrust_n` with linear interpolation and zero exterior thrust."""
    if extrapolation != "zero":
        raise ValueError("thrust extrapolation must be 'zero'")
    source, values = _read_numeric_csv(path, THRUST_COLUMNS)
    if values.shape[0] < 2 or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{source.name}: time_s must be strictly increasing with at least 2 rows")
    if np.any(values[:, 1] < 0.0):
        raise ValueError(f"{source.name}: thrust_n cannot be negative")
    curve = ThrustCurve(values[:, 0], values[:, 1], propellant_mass_kg)
    return ImportedThrustData(
        curve,
        EngineeringDataProvenance(source, _sha256(source), "piecewise-linear", extrapolation),
    )


@dataclass(frozen=True, slots=True)
class ImportedMassPropertiesSample:
    """Interpolated mass, centre of gravity, and inertia in SI units."""

    mass_kg: float
    centre_of_gravity_from_nose_m: float
    inertia_body_kgm2: FloatArray


@dataclass(frozen=True, slots=True)
class TabulatedMassPropertyData:
    """Piecewise-linear mass-property schedule imported from CSV."""

    time_s: FloatArray
    mass_kg: FloatArray
    cg_from_nose_m: FloatArray
    inertia_body_kgm2: FloatArray
    out_of_range: MassOutOfRange
    provenance: EngineeringDataProvenance

    def at_time(self, time_s: float) -> ImportedMassPropertiesSample:
        """Interpolate the table or enforce the declared exterior policy."""
        if not np.isfinite(time_s):
            raise ValueError("mass-property query time_s must be finite")
        if self.out_of_range == "error" and not self.time_s[0] <= time_s <= self.time_s[-1]:
            raise ValueError(
                f"mass-property query {time_s} s outside [{self.time_s[0]}, {self.time_s[-1]}] s"
            )
        query_s = float(np.clip(time_s, self.time_s[0], self.time_s[-1]))
        if query_s == self.time_s[-1]:
            index = self.time_s.size - 2
            fraction = 1.0
        else:
            index = int(np.searchsorted(self.time_s, query_s, side="right") - 1)
            index = max(index, 0)
            fraction = float(
                (query_s - self.time_s[index]) / (self.time_s[index + 1] - self.time_s[index])
            )
        mass_kg = (1.0 - fraction) * self.mass_kg[index] + fraction * self.mass_kg[index + 1]
        cg_m = (1.0 - fraction) * self.cg_from_nose_m[index] + fraction * self.cg_from_nose_m[
            index + 1
        ]
        inertia = (1.0 - fraction) * self.inertia_body_kgm2[
            index
        ] + fraction * self.inertia_body_kgm2[index + 1]
        return ImportedMassPropertiesSample(float(mass_kg), float(cg_m), inertia)


def import_mass_properties_csv(
    path: str | Path,
    *,
    out_of_range: MassOutOfRange = "error",
) -> TabulatedMassPropertyData:
    """Import a monotonic time history of symmetric positive-definite mass properties."""
    if out_of_range not in {"error", "clamp"}:
        raise ValueError("mass-property out_of_range must be 'error' or 'clamp'")
    source, values = _read_numeric_csv(path, MASS_PROPERTY_COLUMNS)
    if values.shape[0] < 2 or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{source.name}: time_s must be strictly increasing with at least 2 rows")
    if np.any(values[:, 1] <= 0.0):
        raise ValueError(f"{source.name}: mass_kg must be positive")
    if np.any(np.diff(values[:, 1]) > 1.0e-12):
        raise ValueError(f"{source.name}: mass_kg must be non-increasing")
    if np.any(values[:, 2] < 0.0):
        raise ValueError(f"{source.name}: cg_from_nose_m must be non-negative")
    inertia = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    for index, row in enumerate(values):
        xx, yy, zz, xy, xz, yz = row[3:]
        inertia[index] = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        if np.any(np.linalg.eigvalsh(inertia[index]) <= 0.0):
            raise ValueError(
                f"{source.name}: row {index + 2} inertia tensor is not positive definite"
            )
    return TabulatedMassPropertyData(
        time_s=values[:, 0].copy(),
        mass_kg=values[:, 1].copy(),
        cg_from_nose_m=values[:, 2].copy(),
        inertia_body_kgm2=inertia,
        out_of_range=out_of_range,
        provenance=EngineeringDataProvenance(
            source, _sha256(source), "piecewise-linear", out_of_range
        ),
    )


@dataclass(frozen=True, slots=True)
class ImportedAerodynamicData:
    """Regular-grid aerodynamic coefficients plus import policy and hash."""

    database: TabulatedAerodynamicDatabase
    provenance: EngineeringDataProvenance


def import_aerodynamic_csv(
    path: str | Path,
    *,
    out_of_range: OutOfRange = "error",
) -> ImportedAerodynamicData:
    """Import canonical dimensionless/radian aerodynamic axes and six coefficients."""
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"aerodynamic data file does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"{source.name}: CSV header is required") from error
    if len(header) != len(set(header)):
        raise ValueError(f"{source.name}: duplicate CSV columns are not allowed")
    if len(header) <= len(COEFFICIENT_NAMES):
        raise ValueError(f"{source.name}: at least one aerodynamic axis is required")
    axis_names = header[: -len(COEFFICIENT_NAMES)]
    if header[-len(COEFFICIENT_NAMES) :] != COEFFICIENT_NAMES:
        raise ValueError(
            f"{source.name}: final dimensionless coefficient columns must be {COEFFICIENT_NAMES}"
        )
    unknown = set(axis_names) - set(SUPPORTED_AXIS_NAMES)
    if unknown:
        raise ValueError(
            f"{source.name}: unsupported or unit-ambiguous aerodynamic axes {sorted(unknown)}"
        )
    database = TabulatedAerodynamicDatabase.from_csv(source, out_of_range=out_of_range)
    return ImportedAerodynamicData(
        database,
        EngineeringDataProvenance(source, _sha256(source), "multilinear", out_of_range),
    )
