"""B-plane and unpowered/powered gravity-assist feasibility calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


def _unit(vector: FloatArray, label: str) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"{label} must have nonzero finite magnitude")
    return vector / norm


@dataclass(frozen=True, slots=True)
class BPlaneTarget:
    """Planet-relative asymptote geometry and equivalent periapsis requirement."""

    incoming_excess_velocity_mps: FloatArray
    outgoing_excess_velocity_mps: FloatArray
    turn_angle_rad: float
    equivalent_periapsis_radius_m: float
    periapsis_altitude_m: float
    impact_parameter_m: float
    b_vector_m: FloatArray
    b_dot_t_m: float
    b_dot_r_m: float
    powered_flyby_delta_v_mps: float
    feasible_unpowered: bool


def target_b_plane(
    incoming_excess_velocity_mps: npt.ArrayLike,
    outgoing_excess_velocity_mps: npt.ArrayLike,
    body_mu_m3_s2: float,
    body_radius_m: float,
    *,
    minimum_altitude_m: float = 0.0,
    excess_speed_tolerance_mps: float = 10.0,
) -> BPlaneTarget:
    """Map incoming/outgoing excess velocities to an equivalent B-plane target."""
    incoming = np.asarray(incoming_excess_velocity_mps, dtype=np.float64)
    outgoing = np.asarray(outgoing_excess_velocity_mps, dtype=np.float64)
    if incoming.shape != (3,) or outgoing.shape != (3,):
        raise ValueError("incoming and outgoing excess velocities must have three components")
    if not np.all(np.isfinite(incoming)) or not np.all(np.isfinite(outgoing)):
        raise ValueError("excess velocities must be finite")
    parameters = np.array(
        [body_mu_m3_s2, body_radius_m, minimum_altitude_m, excess_speed_tolerance_mps]
    )
    if not np.all(np.isfinite(parameters)):
        raise ValueError("B-plane parameters must be finite")
    if body_mu_m3_s2 <= 0.0 or body_radius_m <= 0.0:
        raise ValueError("body gravitational parameter and radius must be positive")
    if minimum_altitude_m < 0.0 or excess_speed_tolerance_mps < 0.0:
        raise ValueError("minimum altitude and speed tolerance must be nonnegative")

    incoming_speed = float(np.linalg.norm(incoming))
    outgoing_speed = float(np.linalg.norm(outgoing))
    incoming_hat = _unit(incoming, "incoming excess velocity")
    outgoing_hat = _unit(outgoing, "outgoing excess velocity")
    turn_angle = float(np.arccos(np.clip(np.dot(incoming_hat, outgoing_hat), -1.0, 1.0)))
    representative_speed = 0.5 * (incoming_speed + outgoing_speed)
    if turn_angle <= 1.0e-12:
        eccentricity = np.inf
        periapsis_radius_m = np.inf
        impact_parameter_m = np.inf
    else:
        eccentricity = 1.0 / np.sin(0.5 * turn_angle)
        periapsis_radius_m = body_mu_m3_s2 * (eccentricity - 1.0) / representative_speed**2
        impact_parameter_m = (
            body_mu_m3_s2 / representative_speed**2 * np.sqrt(max(eccentricity**2 - 1.0, 0.0))
        )
    periapsis_altitude_m = periapsis_radius_m - body_radius_m

    flyby_normal = np.cross(incoming_hat, outgoing_hat)
    if np.linalg.norm(flyby_normal) <= 1.0e-12:
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(reference, incoming_hat)) > 0.95:
            reference = np.array([1.0, 0.0, 0.0])
        flyby_normal = np.cross(incoming_hat, reference)
    flyby_normal = _unit(flyby_normal, "flyby plane normal")
    b_hat = _unit(np.cross(flyby_normal, incoming_hat), "B-vector direction")
    b_vector = impact_parameter_m * b_hat
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, incoming_hat)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    t_hat = _unit(np.cross(reference, incoming_hat), "B-plane T axis")
    r_hat = np.cross(incoming_hat, t_hat)
    speed_mismatch = abs(outgoing_speed - incoming_speed)
    return BPlaneTarget(
        incoming_excess_velocity_mps=incoming.copy(),
        outgoing_excess_velocity_mps=outgoing.copy(),
        turn_angle_rad=turn_angle,
        equivalent_periapsis_radius_m=float(periapsis_radius_m),
        periapsis_altitude_m=float(periapsis_altitude_m),
        impact_parameter_m=float(impact_parameter_m),
        b_vector_m=b_vector,
        b_dot_t_m=float(np.dot(b_vector, t_hat)),
        b_dot_r_m=float(np.dot(b_vector, r_hat)),
        powered_flyby_delta_v_mps=speed_mismatch,
        feasible_unpowered=(
            speed_mismatch <= excess_speed_tolerance_mps
            and periapsis_altitude_m >= minimum_altitude_m
        ),
    )
