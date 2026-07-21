"""Restricted N-body spacecraft dynamics in a primary-centred inertial frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.astrodynamics.bodies import CircularOrbitBody, PrimaryBody
from aerognc.mathematics.vectors import FloatArray


def _gravity_acceleration(displacement_m: FloatArray, mu_m3_s2: float) -> FloatArray:
    distance_m = float(np.linalg.norm(displacement_m))
    if distance_m <= 0.0 or not np.isfinite(distance_m):
        raise FloatingPointError("gravity model encountered a zero or invalid separation")
    return mu_m3_s2 * displacement_m / distance_m**3


@dataclass(frozen=True, slots=True)
class RestrictedNBodyModel:
    """Massless spacecraft influenced by a primary and prescribed orbiting bodies.

    Planet-to-spacecraft acceleration includes the indirect term required by the
    primary-centred frame. Planets follow analytical circular ephemerides and do not
    perturb each other; this is a restricted, not a full solar-system ephemeris model.
    """

    primary: PrimaryBody
    bodies: tuple[CircularOrbitBody, ...]

    def __post_init__(self) -> None:
        if not self.bodies:
            raise ValueError("restricted N-body model requires at least one orbiting body")
        names = [body.name.casefold() for body in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("orbiting body names must be unique")
        if any(
            body.semi_major_axis_m <= self.primary.radius_m + body.radius_m for body in self.bodies
        ):
            raise ValueError("planetary orbit intersects the primary body")

    def body_state(self, name: str, time_s: float) -> tuple[FloatArray, FloatArray]:
        """Return an orbiting body's configured analytical state."""
        body = next((candidate for candidate in self.bodies if candidate.name == name), None)
        if body is None:
            raise KeyError(f"unknown orbiting body: {name}")
        return body.state_at_time(time_s, self.primary.gravitational_parameter_m3_s2)

    def acceleration(self, time_s: float, position_m: npt.ArrayLike) -> FloatArray:
        """Calculate primary-centred spacecraft acceleration in m/s²."""
        position = np.asarray(position_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_m must contain three finite components")
        acceleration = _gravity_acceleration(-position, self.primary.gravitational_parameter_m3_s2)
        for body in self.bodies:
            body_position, _body_velocity = body.state_at_time(
                time_s, self.primary.gravitational_parameter_m3_s2
            )
            direct = _gravity_acceleration(
                body_position - position, body.gravitational_parameter_m3_s2
            )
            indirect = _gravity_acceleration(body_position, body.gravitational_parameter_m3_s2)
            acceleration += direct - indirect
        return acceleration

    def derivative(self, time_s: float, state: npt.ArrayLike) -> FloatArray:
        """Return derivative of [x, y, z, vx, vy, vz] in SI units."""
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("interplanetary state must contain six finite values")
        return np.concatenate((vector[3:6], self.acceleration(time_s, vector[0:3])))

    def central_specific_energy(self, state: npt.ArrayLike) -> float:
        """Return two-body primary-relative specific mechanical energy in J/kg."""
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("interplanetary state must contain six finite values")
        radius_m = float(np.linalg.norm(vector[:3]))
        if radius_m <= 0.0:
            raise ValueError("spacecraft cannot be at the primary centre")
        return float(
            0.5 * np.dot(vector[3:6], vector[3:6])
            - self.primary.gravitational_parameter_m3_s2 / radius_m
        )

    def relative_state(
        self, body_name: str, time_s: float, state: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Return spacecraft position and velocity relative to an orbiting body."""
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("interplanetary state must contain six finite values")
        body_position, body_velocity = self.body_state(body_name, time_s)
        return vector[:3] - body_position, vector[3:6] - body_velocity
