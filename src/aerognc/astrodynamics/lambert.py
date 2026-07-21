"""Zero-revolution universal-variable Lambert targeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.kepler import stumpff_c, stumpff_s
from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class LambertSolution:
    """Endpoint velocities and geometry of one zero-revolution Lambert arc."""

    departure_velocity_mps: FloatArray
    arrival_velocity_mps: FloatArray
    transfer_angle_rad: float
    time_of_flight_s: float
    iterations: int
    long_way: bool


def _transfer_angle(
    position_1: FloatArray, position_2: FloatArray, prograde: bool, long_way: bool
) -> float:
    cosine = float(
        np.clip(
            np.dot(position_1, position_2)
            / (np.linalg.norm(position_1) * np.linalg.norm(position_2)),
            -1.0,
            1.0,
        )
    )
    angle = float(np.arccos(cosine))
    cross_z = float(np.cross(position_1, position_2)[2])
    if (prograde and cross_z < 0.0) or (not prograde and cross_z >= 0.0):
        angle = 2.0 * np.pi - angle
    if long_way:
        angle = 2.0 * np.pi - angle
    return angle


def solve_lambert_universal(
    departure_position_m: npt.ArrayLike,
    arrival_position_m: npt.ArrayLike,
    time_of_flight_s: float,
    gravitational_parameter_m3_s2: float,
    *,
    prograde: bool = True,
    long_way: bool = False,
    tolerance_s: float = 1.0e-6,
    maximum_iterations: int = 120,
) -> LambertSolution:
    """Solve the zero-revolution Lambert problem with Stumpff functions and bisection."""
    position_1 = np.asarray(departure_position_m, dtype=np.float64)
    position_2 = np.asarray(arrival_position_m, dtype=np.float64)
    if position_1.shape != (3,) or position_2.shape != (3,):
        raise ValueError("Lambert endpoint positions must contain three components")
    if not np.all(np.isfinite(position_1)) or not np.all(np.isfinite(position_2)):
        raise ValueError("Lambert endpoint positions must be finite")
    if not np.isfinite(time_of_flight_s) or time_of_flight_s <= 0.0:
        raise ValueError("Lambert time of flight must be positive and finite")
    mu = gravitational_parameter_m3_s2
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational parameter must be positive and finite")
    radius_1 = float(np.linalg.norm(position_1))
    radius_2 = float(np.linalg.norm(position_2))
    if radius_1 <= 0.0 or radius_2 <= 0.0:
        raise ValueError("Lambert endpoint radii must be positive")
    angle = _transfer_angle(position_1, position_2, prograde, long_way)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    if abs(1.0 - cosine) <= 1.0e-14 or abs(sine) <= 1.0e-14:
        raise ValueError("Lambert geometry is singular for collinear endpoints")
    parameter_a = sine * np.sqrt(radius_1 * radius_2 / (1.0 - cosine))
    square_root_mu = float(np.sqrt(mu))

    def residual(z: float) -> float:
        c_value = stumpff_c(z)
        if c_value <= 0.0:
            return np.nan
        s_value = stumpff_s(z)
        y_value = radius_1 + radius_2 + parameter_a * (z * s_value - 1.0) / np.sqrt(c_value)
        if y_value < 0.0:
            return np.nan
        x_value = np.sqrt(y_value / c_value)
        calculated_time_s = (x_value**3 * s_value + parameter_a * np.sqrt(y_value)) / square_root_mu
        return float(calculated_time_s - time_of_flight_s)

    limit = 4.0 * np.pi**2 - 1.0e-7
    grid = np.linspace(-limit, limit, 801)
    bracket: tuple[float, float] | None = None
    previous_z: float | None = None
    previous_residual: float | None = None
    for current_z in grid:
        current_residual = residual(float(current_z))
        if not np.isfinite(current_residual):
            continue
        if current_residual == 0.0:
            bracket = (float(current_z), float(current_z))
            break
        if previous_residual is not None and np.signbit(current_residual) != np.signbit(
            previous_residual
        ):
            bracket = (cast_float(previous_z), float(current_z))
            break
        previous_z = float(current_z)
        previous_residual = current_residual
    if bracket is None:
        raise ValueError("no feasible zero-revolution Lambert solution for the requested time")

    lower, upper = bracket
    iterations = 0
    if lower == upper:
        z_solution = lower
    else:
        lower_value = residual(lower)
        z_solution = 0.5 * (lower + upper)
        for iteration in range(1, maximum_iterations + 1):
            iterations = iteration
            z_solution = 0.5 * (lower + upper)
            middle_value = residual(z_solution)
            if not np.isfinite(middle_value):
                lower = z_solution
                continue
            if abs(middle_value) <= tolerance_s:
                break
            if np.signbit(middle_value) == np.signbit(lower_value):
                lower = z_solution
                lower_value = middle_value
            else:
                upper = z_solution
        else:
            raise RuntimeError("Lambert bisection did not converge")

    c_value = stumpff_c(z_solution)
    s_value = stumpff_s(z_solution)
    y_value = radius_1 + radius_2 + parameter_a * (z_solution * s_value - 1.0) / np.sqrt(c_value)
    if y_value <= 0.0:
        raise FloatingPointError("Lambert solution produced a nonpositive y parameter")
    f_value = 1.0 - y_value / radius_1
    g_value = parameter_a * np.sqrt(y_value / mu)
    g_dot = 1.0 - y_value / radius_2
    if abs(g_value) <= 1.0e-14:
        raise FloatingPointError("Lambert solution produced a singular g function")
    departure_velocity = (position_2 - f_value * position_1) / g_value
    arrival_velocity = (g_dot * position_2 - position_1) / g_value
    return LambertSolution(
        departure_velocity_mps=departure_velocity,
        arrival_velocity_mps=arrival_velocity,
        transfer_angle_rad=angle,
        time_of_flight_s=time_of_flight_s,
        iterations=iterations,
        long_way=long_way,
    )


def cast_float(value: float | None) -> float:
    """Narrow an internal optional bracket value after its paired residual exists."""
    if value is None:
        raise RuntimeError("Lambert bracket state is inconsistent")
    return value
