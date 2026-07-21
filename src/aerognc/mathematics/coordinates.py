"""Frame transformations and aerodynamic-angle utilities."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.quaternion import quaternion_to_dcm
from aerognc.mathematics.vectors import FloatArray, as_vector


def body_to_navigation(
    vector_b: Sequence[float] | npt.ArrayLike,
    quaternion_nb: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Transform body FRD components into navigation NED components."""
    return quaternion_to_dcm(quaternion_nb) @ as_vector(vector_b, 3, name="vector_b")


def navigation_to_body(
    vector_n: Sequence[float] | npt.ArrayLike,
    quaternion_nb: Sequence[float] | npt.ArrayLike,
) -> FloatArray:
    """Transform navigation NED components into body FRD components."""
    return quaternion_to_dcm(quaternion_nb).T @ as_vector(vector_n, 3, name="vector_n")


def aerodynamic_angles(
    air_velocity_b_mps: Sequence[float] | npt.ArrayLike,
    *,
    minimum_speed_mps: float = 1.0e-9,
) -> tuple[float, float, float]:
    """Return airspeed [m/s], angle of attack [rad], and sideslip [rad]."""
    if minimum_speed_mps < 0.0:
        raise ValueError("minimum_speed_mps must be nonnegative")
    velocity = as_vector(air_velocity_b_mps, 3, name="air_velocity_b_mps")
    speed = float(np.linalg.norm(velocity))
    if speed <= minimum_speed_mps:
        return speed, 0.0, 0.0
    alpha = float(np.arctan2(velocity[2], velocity[0]))
    beta = float(np.arcsin(np.clip(velocity[1] / speed, -1.0, 1.0)))
    return speed, alpha, beta


def launch_direction_ned(elevation_rad: float, azimuth_rad: float) -> FloatArray:
    """Return a unit launch direction for elevation above horizontal and NED azimuth.

    Azimuth is clockwise from north. Positive elevation points upward (negative down).
    """
    if not np.all(np.isfinite([elevation_rad, azimuth_rad])):
        raise ValueError("launch angles must be finite")
    horizontal = np.cos(elevation_rad)
    return np.array(
        [
            horizontal * np.cos(azimuth_rad),
            horizontal * np.sin(azimuth_rad),
            -np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )
