"""Analytical Keplerian ephemerides for a fictional planetary system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from aerognc.astrodynamics.orbital_elements import (
    ClassicalOrbitalElements,
    eccentric_to_true_anomaly,
    elements_to_state,
    solve_kepler_equation,
)
from aerognc.mathematics.vectors import FloatArray

BodyRole = Literal["departure", "assist", "destination", "background"]


@dataclass(frozen=True, slots=True)
class PrimaryBody:
    """Fixed central gravitating body in the heliocentric integration frame."""

    name: str
    gravitational_parameter_m3_s2: float
    radius_m: float
    color: str = "#FDB813"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("primary body name cannot be empty")
        if not np.isfinite(self.gravitational_parameter_m3_s2) or (
            self.gravitational_parameter_m3_s2 <= 0.0
        ):
            raise ValueError("primary gravitational parameter must be positive and finite")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("primary radius must be positive and finite")


@dataclass(frozen=True, slots=True)
class CircularOrbitBody:
    """Planet with a prescribed elliptic Keplerian orbit around the primary body.

    The historical class name remains for compatibility. ``eccentricity=0`` exactly
    reproduces the original circular baseline.
    """

    name: str
    role: BodyRole
    gravitational_parameter_m3_s2: float
    radius_m: float
    semi_major_axis_m: float
    phase_at_epoch_rad: float
    inclination_rad: float = 0.0
    ascending_node_rad: float = 0.0
    color: str = "#2878B5"
    eccentricity: float = 0.0
    argument_of_periapsis_rad: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("orbiting body name cannot be empty")
        if self.role not in {"departure", "assist", "destination", "background"}:
            raise ValueError("invalid planetary role")
        positive_values = {
            "gravitational parameter": self.gravitational_parameter_m3_s2,
            "radius": self.radius_m,
            "semi-major axis": self.semi_major_axis_m,
        }
        for label, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"planetary {label} must be positive and finite")
        values = np.array(
            [
                self.phase_at_epoch_rad,
                self.inclination_rad,
                self.ascending_node_rad,
                self.eccentricity,
                self.argument_of_periapsis_rad,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("planetary elements must be finite")
        if abs(self.inclination_rad) > np.pi:
            raise ValueError("planetary inclination magnitude cannot exceed pi radians")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("prescribed planetary eccentricity must lie in [0, 1)")

    def mean_motion_rad_s(self, primary_mu_m3_s2: float) -> float:
        """Return the circular-orbit angular rate from Kepler's third law."""
        if not np.isfinite(primary_mu_m3_s2) or primary_mu_m3_s2 <= 0.0:
            raise ValueError("primary_mu_m3_s2 must be positive and finite")
        return float(np.sqrt(primary_mu_m3_s2 / self.semi_major_axis_m**3))

    def state_at_time(
        self, time_s: float, primary_mu_m3_s2: float
    ) -> tuple[FloatArray, FloatArray]:
        """Return primary-centred inertial position and velocity in SI units."""
        if not np.isfinite(time_s):
            raise ValueError("time_s must be finite")
        mean_anomaly = self.phase_at_epoch_rad + self.mean_motion_rad_s(primary_mu_m3_s2) * time_s
        eccentric_anomaly = solve_kepler_equation(mean_anomaly, self.eccentricity)
        true_anomaly = eccentric_to_true_anomaly(eccentric_anomaly, self.eccentricity)
        return elements_to_state(
            ClassicalOrbitalElements(
                semi_major_axis_m=self.semi_major_axis_m,
                eccentricity=self.eccentricity,
                inclination_rad=self.inclination_rad,
                ascending_node_rad=self.ascending_node_rad,
                argument_of_periapsis_rad=self.argument_of_periapsis_rad,
                true_anomaly_rad=true_anomaly,
            ),
            primary_mu_m3_s2,
        )

    def orbital_period_s(self, primary_mu_m3_s2: float) -> float:
        """Return the circular period in seconds."""
        return float(2.0 * np.pi / self.mean_motion_rad_s(primary_mu_m3_s2))


KeplerianOrbitBody = CircularOrbitBody
