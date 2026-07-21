"""Optional higher-fidelity astrodynamics force models in SI units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray

ASTRONOMICAL_UNIT_M = 149_597_870_700.0
SPEED_OF_LIGHT_MPS = 299_792_458.0
SOLAR_RADIATION_PRESSURE_AT_ONE_AU_PA = 4.56e-6


def _state_vector(value: npt.ArrayLike, label: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain three finite components")
    return vector


def j2_acceleration_mps2(
    position_m: npt.ArrayLike,
    gravitational_parameter_m3_s2: float,
    reference_radius_m: float,
    j2: float,
) -> FloatArray:
    """Return the oblateness acceleration in a primary-equatorial inertial frame."""
    position = _state_vector(position_m, "position")
    values = np.array([gravitational_parameter_m3_s2, reference_radius_m, j2])
    if not np.all(np.isfinite(values)) or gravitational_parameter_m3_s2 <= 0.0:
        raise ValueError("J2 parameters must be finite and gravitational parameter positive")
    if reference_radius_m <= 0.0 or j2 < 0.0:
        raise ValueError("reference radius must be positive and J2 nonnegative")
    radius = float(np.linalg.norm(position))
    if radius <= reference_radius_m:
        raise ValueError("J2 position must lie above the reference surface")
    z_ratio_squared = (position[2] / radius) ** 2
    factor = 1.5 * j2 * gravitational_parameter_m3_s2 * reference_radius_m**2 / radius**5
    return factor * np.array(
        [
            position[0] * (5.0 * z_ratio_squared - 1.0),
            position[1] * (5.0 * z_ratio_squared - 1.0),
            position[2] * (5.0 * z_ratio_squared - 3.0),
        ]
    )


def solar_radiation_pressure_acceleration_mps2(
    source_to_spacecraft_position_m: npt.ArrayLike,
    area_m2: float,
    mass_kg: float,
    reflectivity_coefficient: float,
    *,
    pressure_at_one_au_pa: float = SOLAR_RADIATION_PRESSURE_AT_ONE_AU_PA,
) -> FloatArray:
    """Return cannonball-model radiation acceleration directed away from the source."""
    position = _state_vector(source_to_spacecraft_position_m, "source-to-spacecraft position")
    values = np.array([area_m2, mass_kg, reflectivity_coefficient, pressure_at_one_au_pa])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("radiation-pressure inputs must be positive and finite")
    distance = float(np.linalg.norm(position))
    if distance <= 0.0:
        raise ValueError("radiation source distance must be positive")
    pressure = pressure_at_one_au_pa * (ASTRONOMICAL_UNIT_M / distance) ** 2
    return pressure * reflectivity_coefficient * area_m2 / mass_kg * position / distance


def schwarzschild_acceleration_mps2(
    position_m: npt.ArrayLike,
    velocity_mps: npt.ArrayLike,
    gravitational_parameter_m3_s2: float,
) -> FloatArray:
    """Return the first post-Newtonian Schwarzschild point-mass correction."""
    position = _state_vector(position_m, "position")
    velocity = _state_vector(velocity_mps, "velocity")
    if not np.isfinite(gravitational_parameter_m3_s2) or gravitational_parameter_m3_s2 <= 0.0:
        raise ValueError("gravitational parameter must be positive and finite")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("relativistic correction requires positive radius")
    speed_squared = float(np.dot(velocity, velocity))
    radial_rate_product = float(np.dot(position, velocity))
    factor = gravitational_parameter_m3_s2 / (SPEED_OF_LIGHT_MPS**2 * radius**3)
    return factor * (
        (4.0 * gravitational_parameter_m3_s2 / radius - speed_squared) * position
        + 4.0 * radial_rate_product * velocity
    )


def hill_sphere_radius_m(
    semi_major_axis_m: float, eccentricity: float, body_mass_kg: float, primary_mass_kg: float
) -> float:
    """Return the periapsis-scaled Hill-sphere approximation."""
    values = np.array([semi_major_axis_m, eccentricity, body_mass_kg, primary_mass_kg])
    if not np.all(np.isfinite(values)) or semi_major_axis_m <= 0.0:
        raise ValueError("Hill-sphere inputs must be finite with positive semi-major axis")
    if not 0.0 <= eccentricity < 1.0 or body_mass_kg <= 0.0 or primary_mass_kg <= 0.0:
        raise ValueError("Hill-sphere eccentricity/masses are nonphysical")
    return float(
        semi_major_axis_m
        * (1.0 - eccentricity)
        * (body_mass_kg / (3.0 * primary_mass_kg)) ** (1.0 / 3.0)
    )


def laplace_sphere_of_influence_radius_m(
    semi_major_axis_m: float, body_mass_kg: float, primary_mass_kg: float
) -> float:
    """Return the classical patched-conic Laplace sphere of influence."""
    values = np.array([semi_major_axis_m, body_mass_kg, primary_mass_kg])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("sphere-of-influence inputs must be positive and finite")
    return float(semi_major_axis_m * (body_mass_kg / primary_mass_kg) ** (2.0 / 5.0))


@dataclass(frozen=True, slots=True)
class PerturbationSettings:
    """Explicit switches and physical properties for optional force terms."""

    j2: float = 0.0
    reference_radius_m: float = 1.0
    radiation_area_m2: float = 0.0
    reflectivity_coefficient: float = 1.0
    include_relativity: bool = False

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.j2,
                self.reference_radius_m,
                self.radiation_area_m2,
                self.reflectivity_coefficient,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("perturbation settings must be finite")
        if self.j2 < 0.0 or self.reference_radius_m <= 0.0 or self.radiation_area_m2 < 0.0:
            raise ValueError("perturbation settings contain nonphysical values")
        if self.reflectivity_coefficient <= 0.0:
            raise ValueError("reflectivity coefficient must be positive")


def optional_perturbation_acceleration_mps2(
    position_m: npt.ArrayLike,
    velocity_mps: npt.ArrayLike,
    mass_kg: float,
    gravitational_parameter_m3_s2: float,
    settings: PerturbationSettings,
) -> FloatArray:
    """Compose enabled J2, radiation-pressure, and relativistic accelerations."""
    position = _state_vector(position_m, "position")
    velocity = _state_vector(velocity_mps, "velocity")
    acceleration = np.zeros(3)
    if settings.j2 > 0.0:
        acceleration += j2_acceleration_mps2(
            position,
            gravitational_parameter_m3_s2,
            settings.reference_radius_m,
            settings.j2,
        )
    if settings.radiation_area_m2 > 0.0:
        acceleration += solar_radiation_pressure_acceleration_mps2(
            position,
            settings.radiation_area_m2,
            mass_kg,
            settings.reflectivity_coefficient,
        )
    if settings.include_relativity:
        acceleration += schwarzschild_acceleration_mps2(
            position, velocity, gravitational_parameter_m3_s2
        )
    return acceleration
