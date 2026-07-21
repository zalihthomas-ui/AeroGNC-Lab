"""Universal-variable Kepler propagation implemented without an astrodynamics library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


def stumpff_c(z: float) -> float:
    """Evaluate the universal-variable Stumpff C function."""
    if not np.isfinite(z):
        raise ValueError("z must be finite")
    if z > 1.0e-8:
        root = np.sqrt(z)
        return float((1.0 - np.cos(root)) / z)
    if z < -1.0e-8:
        root = np.sqrt(-z)
        return float((np.cosh(root) - 1.0) / (-z))
    return float(0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40_320.0)


def stumpff_s(z: float) -> float:
    """Evaluate the universal-variable Stumpff S function."""
    if not np.isfinite(z):
        raise ValueError("z must be finite")
    if z > 1.0e-8:
        root = np.sqrt(z)
        return float((root - np.sin(root)) / root**3)
    if z < -1.0e-8:
        root = np.sqrt(-z)
        return float((np.sinh(root) - root) / root**3)
    return float(1.0 / 6.0 - z / 120.0 + z**2 / 5_040.0 - z**3 / 362_880.0)


@dataclass(frozen=True, slots=True)
class KeplerPropagation:
    """Propagated two-body state and universal-variable convergence metadata."""

    position_m: FloatArray
    velocity_mps: FloatArray
    universal_anomaly: float
    iterations: int


def propagate_universal(
    initial_position_m: npt.ArrayLike,
    initial_velocity_mps: npt.ArrayLike,
    elapsed_time_s: float,
    gravitational_parameter_m3_s2: float,
    *,
    tolerance: float = 1.0e-10,
    maximum_iterations: int = 200,
) -> KeplerPropagation:
    """Propagate an elliptic, parabolic-near, or hyperbolic two-body state.

    The solver uses universal variables and Newton iteration on the universal Kepler
    equation. No specialised orbital-mechanics package is used.
    """
    position_0 = np.asarray(initial_position_m, dtype=np.float64)
    velocity_0 = np.asarray(initial_velocity_mps, dtype=np.float64)
    if position_0.shape != (3,) or velocity_0.shape != (3,):
        raise ValueError("initial position and velocity must contain three components")
    if not np.all(np.isfinite(position_0)) or not np.all(np.isfinite(velocity_0)):
        raise ValueError("initial state must be finite")
    if not np.isfinite(elapsed_time_s):
        raise ValueError("elapsed_time_s must be finite")
    if not np.isfinite(gravitational_parameter_m3_s2) or gravitational_parameter_m3_s2 <= 0.0:
        raise ValueError("gravitational parameter must be positive and finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if elapsed_time_s == 0.0:
        return KeplerPropagation(position_0.copy(), velocity_0.copy(), 0.0, 0)

    radius_0_m = float(np.linalg.norm(position_0))
    if radius_0_m <= 0.0:
        raise ValueError("initial radius must be positive")
    speed_squared = float(np.dot(velocity_0, velocity_0))
    radial_velocity_mps = float(np.dot(position_0, velocity_0) / radius_0_m)
    mu = gravitational_parameter_m3_s2
    square_root_mu = float(np.sqrt(mu))
    reciprocal_axis = 2.0 / radius_0_m - speed_squared / mu

    if abs(reciprocal_axis) > 1.0e-10 / radius_0_m:
        anomaly = square_root_mu * abs(reciprocal_axis) * elapsed_time_s
    else:
        anomaly = square_root_mu * elapsed_time_s / radius_0_m
    if reciprocal_axis < -1.0e-12:
        sign = 1.0 if elapsed_time_s >= 0.0 else -1.0
        denominator = radial_velocity_mps + sign * np.sqrt(-mu / reciprocal_axis) * (
            1.0 - radius_0_m * reciprocal_axis
        )
        logarithm_argument = (
            -2.0 * mu * reciprocal_axis * elapsed_time_s / denominator
            if denominator != 0.0
            else -1.0
        )
        if logarithm_argument > 0.0:
            anomaly = sign * np.sqrt(-1.0 / reciprocal_axis) * np.log(logarithm_argument)

    initial_anomaly = float(anomaly)
    iterations = 0
    converged = False
    for iteration in range(1, maximum_iterations + 1):
        iterations = iteration
        z = reciprocal_axis * anomaly**2
        c_value = stumpff_c(z)
        s_value = stumpff_s(z)
        equation = (
            radius_0_m * radial_velocity_mps / square_root_mu * anomaly**2 * c_value
            + (1.0 - reciprocal_axis * radius_0_m) * anomaly**3 * s_value
            + radius_0_m * anomaly
            - square_root_mu * elapsed_time_s
        )
        derivative = (
            radius_0_m * radial_velocity_mps / square_root_mu * anomaly * (1.0 - z * s_value)
            + (1.0 - reciprocal_axis * radius_0_m) * anomaly**2 * c_value
            + radius_0_m
        )
        if derivative == 0.0 or not np.isfinite(derivative):
            raise FloatingPointError("universal Kepler iteration encountered a singular derivative")
        correction = equation / derivative
        anomaly -= correction
        if abs(correction) <= tolerance * max(1.0, abs(anomaly)):
            converged = True
            break
    if not converged:
        # Newton's method can jump across several elliptic revolutions for states
        # with high radial speed. The universal time equation is monotonic in chi,
        # so a sign-aware bracket provides a slower but dependable fallback.
        def time_equation(candidate: float) -> float:
            candidate_z = reciprocal_axis * candidate**2
            candidate_c = stumpff_c(candidate_z)
            candidate_s = stumpff_s(candidate_z)
            return float(
                radius_0_m * radial_velocity_mps / square_root_mu * candidate**2 * candidate_c
                + (1.0 - reciprocal_axis * radius_0_m) * candidate**3 * candidate_s
                + radius_0_m * candidate
                - square_root_mu * elapsed_time_s
            )

        span = max(1.0, abs(initial_anomaly))
        if elapsed_time_s > 0.0:
            lower, upper = 0.0, span
        else:
            lower, upper = -span, 0.0
        lower_value = time_equation(lower)
        upper_value = time_equation(upper)
        for _expansion in range(80):
            if (
                np.isfinite(lower_value)
                and np.isfinite(upper_value)
                and (
                    np.signbit(lower_value) != np.signbit(upper_value)
                    or lower_value == 0.0
                    or upper_value == 0.0
                )
            ):
                break
            span *= 2.0
            if elapsed_time_s > 0.0:
                upper = span
                upper_value = time_equation(upper)
            else:
                lower = -span
                lower_value = time_equation(lower)
        else:
            raise RuntimeError("universal Kepler propagation could not bracket its time equation")
        for bisection_iteration in range(1, 241):
            middle = 0.5 * (lower + upper)
            middle_value = time_equation(middle)
            iterations = maximum_iterations + bisection_iteration
            if not np.isfinite(middle_value):
                raise FloatingPointError("universal Kepler bracket produced a non-finite value")
            anomaly = middle
            if middle_value == 0.0 or abs(upper - lower) <= tolerance * max(1.0, abs(middle)):
                converged = True
                break
            if np.signbit(middle_value) == np.signbit(lower_value):
                lower = middle
                lower_value = middle_value
            else:
                upper = middle
                upper_value = middle_value
        if not converged:
            raise RuntimeError("universal Kepler propagation did not converge")

    z = reciprocal_axis * anomaly**2
    c_value = stumpff_c(z)
    s_value = stumpff_s(z)
    f_value = 1.0 - anomaly**2 / radius_0_m * c_value
    g_value = elapsed_time_s - anomaly**3 / square_root_mu * s_value
    position = f_value * position_0 + g_value * velocity_0
    radius_m = float(np.linalg.norm(position))
    if radius_m <= 0.0 or not np.isfinite(radius_m):
        raise FloatingPointError("universal propagation produced an invalid radius")
    f_dot = (
        square_root_mu
        / (radius_m * radius_0_m)
        * (reciprocal_axis * anomaly**3 * s_value - anomaly)
    )
    g_dot = 1.0 - anomaly**2 / radius_m * c_value
    velocity = f_dot * position_0 + g_dot * velocity_0
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("universal propagation produced a non-finite state")
    return KeplerPropagation(position, velocity, float(anomaly), iterations)
