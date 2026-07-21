"""Shared constant-acceleration case for optional MATLAB cross-validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from aerognc.mathematics.integrators import IntegrationResult, integrate_fixed_step
from aerognc.mathematics.vectors import FloatArray

CSV_COLUMNS = (
    "time_s",
    "north_m",
    "east_m",
    "down_m",
    "north_velocity_mps",
    "east_velocity_mps",
    "down_velocity_mps",
)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(value: object, name: str) -> FloatArray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return np.asarray([_number(item, f"{name}[{index}]") for index, item in enumerate(value)])


@dataclass(frozen=True, slots=True)
class ConstantAccelerationCase:
    """Cross-language NED translation benchmark, with all quantities in SI units."""

    case_name: str
    duration_s: float
    time_step_s: float
    mass_kg: float
    initial_position_ned_m: FloatArray
    initial_velocity_ned_mps: FloatArray
    force_ned_n: FloatArray
    gravity_ned_mps2: FloatArray
    absolute_tolerance: float

    def __post_init__(self) -> None:
        if not self.case_name:
            raise ValueError("case_name cannot be empty")
        if self.duration_s <= 0.0 or self.time_step_s <= 0.0:
            raise ValueError("duration_s and time_step_s must be positive")
        if self.mass_kg <= 0.0 or self.absolute_tolerance <= 0.0:
            raise ValueError("mass_kg and absolute_tolerance must be positive")
        step_count = self.duration_s / self.time_step_s
        if not np.isclose(step_count, round(step_count), atol=1.0e-12):
            raise ValueError("duration_s must be an integer multiple of time_step_s")

    @property
    def acceleration_ned_mps2(self) -> FloatArray:
        """Return the constant net NED acceleration."""
        return self.force_ned_n / self.mass_kg + self.gravity_ned_mps2


@dataclass(frozen=True, slots=True)
class CrossLanguageComparison:
    """Maximum component errors for Python, analytical, and optional MATLAB results."""

    python_analytic_max_abs_error: float
    matlab_analytic_max_abs_error: float | None
    python_matlab_max_abs_error: float | None
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether every available comparison is within the declared tolerance."""
        errors = [self.python_analytic_max_abs_error]
        if self.matlab_analytic_max_abs_error is not None:
            errors.append(self.matlab_analytic_max_abs_error)
        if self.python_matlab_max_abs_error is not None:
            errors.append(self.python_matlab_max_abs_error)
        return max(errors) <= self.tolerance

    def as_dict(self) -> dict[str, float | bool | None]:
        """Return a JSON-serialisable report."""
        return {
            "python_analytic_max_abs_error": self.python_analytic_max_abs_error,
            "matlab_analytic_max_abs_error": self.matlab_analytic_max_abs_error,
            "python_matlab_max_abs_error": self.python_matlab_max_abs_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def load_constant_acceleration_case(path: str | Path) -> ConstantAccelerationCase:
    """Load and strictly validate the shared JSON benchmark definition."""
    case_path = Path(path)
    raw = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cross-language case root must be a JSON object")
    payload = cast(dict[str, object], raw)
    expected = {
        "case_name",
        "duration_s",
        "time_step_s",
        "mass_kg",
        "initial_position_ned_m",
        "initial_velocity_ned_mps",
        "force_ned_N",
        "gravity_ned_mps2",
        "absolute_tolerance",
    }
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"invalid cross-language case keys; missing={missing}, unknown={unknown}")
    case_name = payload["case_name"]
    if not isinstance(case_name, str):
        raise ValueError("case_name must be a string")
    return ConstantAccelerationCase(
        case_name=case_name,
        duration_s=_number(payload["duration_s"], "duration_s"),
        time_step_s=_number(payload["time_step_s"], "time_step_s"),
        mass_kg=_number(payload["mass_kg"], "mass_kg"),
        initial_position_ned_m=_vector3(
            payload["initial_position_ned_m"], "initial_position_ned_m"
        ),
        initial_velocity_ned_mps=_vector3(
            payload["initial_velocity_ned_mps"], "initial_velocity_ned_mps"
        ),
        force_ned_n=_vector3(payload["force_ned_N"], "force_ned_N"),
        gravity_ned_mps2=_vector3(payload["gravity_ned_mps2"], "gravity_ned_mps2"),
        absolute_tolerance=_number(payload["absolute_tolerance"], "absolute_tolerance"),
    )


def simulate_constant_acceleration(case: ConstantAccelerationCase) -> IntegrationResult:
    """Integrate the shared benchmark with AeroGNC-Lab's custom RK4 solver."""
    initial_state = np.concatenate((case.initial_position_ned_m, case.initial_velocity_ned_mps))
    acceleration = case.acceleration_ned_mps2

    def derivative(_time_s: float, state: FloatArray) -> FloatArray:
        return np.concatenate((state[3:6], acceleration))

    return integrate_fixed_step(
        derivative,
        initial_state,
        (0.0, case.duration_s),
        case.time_step_s,
    )


def analytical_constant_acceleration(
    case: ConstantAccelerationCase, time_s: FloatArray
) -> FloatArray:
    """Evaluate the exact six-state trajectory at the supplied times."""
    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be a finite one-dimensional array")
    acceleration = case.acceleration_ned_mps2
    position = (
        case.initial_position_ned_m[None, :]
        + time[:, None] * case.initial_velocity_ned_mps[None, :]
        + 0.5 * time[:, None] ** 2 * acceleration[None, :]
    )
    velocity = case.initial_velocity_ned_mps[None, :] + time[:, None] * acceleration[None, :]
    return np.hstack((position, velocity))


def write_state_csv(path: str | Path, time_s: FloatArray, state: FloatArray) -> Path:
    """Write the stable cross-language CSV schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (time.size, 6):
        raise ValueError("state must have shape (len(time_s), 6)")
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for time_value, row in zip(time, values, strict=True):
            writer.writerow([f"{time_value:.17g}", *(f"{value:.17g}" for value in row)])
    return output_path


def read_state_csv(path: str | Path) -> tuple[FloatArray, FloatArray]:
    """Read and validate a Python- or MATLAB-generated benchmark CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("cross-language CSV contains no data rows")
    matrix = np.asarray(
        [[float(row[column]) for column in CSV_COLUMNS] for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(matrix)) or np.any(np.diff(matrix[:, 0]) <= 0.0):
        raise ValueError("cross-language CSV must contain finite, strictly increasing data")
    return matrix[:, 0], matrix[:, 1:]


def compare_cross_language_results(
    case: ConstantAccelerationCase,
    python_result: IntegrationResult,
    matlab_csv_path: str | Path | None = None,
) -> CrossLanguageComparison:
    """Compare numerical results with the exact trajectory and, if present, MATLAB."""
    exact = analytical_constant_acceleration(case, python_result.time_s)
    python_error = float(np.max(np.abs(python_result.state - exact)))
    matlab_exact_error: float | None = None
    python_matlab_error: float | None = None
    if matlab_csv_path is not None:
        matlab_time, matlab_state = read_state_csv(matlab_csv_path)
        if matlab_time.shape != python_result.time_s.shape or not np.allclose(
            matlab_time, python_result.time_s, atol=1.0e-13, rtol=0.0
        ):
            raise ValueError("MATLAB and Python time grids do not match")
        matlab_exact = analytical_constant_acceleration(case, matlab_time)
        matlab_exact_error = float(np.max(np.abs(matlab_state - matlab_exact)))
        python_matlab_error = float(np.max(np.abs(python_result.state - matlab_state)))
    return CrossLanguageComparison(
        python_analytic_max_abs_error=python_error,
        matlab_analytic_max_abs_error=matlab_exact_error,
        python_matlab_max_abs_error=python_matlab_error,
        tolerance=case.absolute_tolerance,
    )
