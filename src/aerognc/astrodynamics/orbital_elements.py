"""Classical orbital elements and Cartesian-state transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class ClassicalOrbitalElements:
    """Osculating conic elements in radians and SI units."""

    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    ascending_node_rad: float
    argument_of_periapsis_rad: float
    true_anomaly_rad: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.semi_major_axis_m,
                self.eccentricity,
                self.inclination_rad,
                self.ascending_node_rad,
                self.argument_of_periapsis_rad,
                self.true_anomaly_rad,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("orbital elements must be finite")
        if self.eccentricity < 0.0 or np.isclose(self.eccentricity, 1.0, atol=1.0e-12):
            raise ValueError("eccentricity must be nonnegative and non-parabolic")
        if self.eccentricity < 1.0 and self.semi_major_axis_m <= 0.0:
            raise ValueError("elliptic semi-major axis must be positive")
        if self.eccentricity > 1.0 and self.semi_major_axis_m >= 0.0:
            raise ValueError("hyperbolic semi-major axis must be negative")
        if not 0.0 <= self.inclination_rad <= np.pi:
            raise ValueError("inclination must lie in [0, pi]")


def solve_kepler_equation(
    mean_anomaly_rad: float,
    eccentricity: float,
    *,
    tolerance: float = 1.0e-13,
    maximum_iterations: int = 80,
) -> float:
    """Solve elliptic or hyperbolic Kepler's equation with Newton iteration."""
    if not np.isfinite(mean_anomaly_rad) or not np.isfinite(eccentricity):
        raise ValueError("mean anomaly and eccentricity must be finite")
    if eccentricity < 0.0 or np.isclose(eccentricity, 1.0, atol=1.0e-12):
        raise ValueError("eccentricity must be nonnegative and non-parabolic")
    if eccentricity < 1.0:
        mean_anomaly = float(np.mod(mean_anomaly_rad + np.pi, 2.0 * np.pi) - np.pi)
        anomaly = mean_anomaly if eccentricity < 0.8 else np.copysign(np.pi, mean_anomaly)
        for _iteration in range(maximum_iterations):
            equation = anomaly - eccentricity * np.sin(anomaly) - mean_anomaly
            derivative = 1.0 - eccentricity * np.cos(anomaly)
            correction = equation / derivative
            anomaly -= correction
            if abs(correction) <= tolerance:
                return float(anomaly)
    else:
        anomaly = float(np.arcsinh(mean_anomaly_rad / eccentricity))
        for _iteration in range(maximum_iterations):
            equation = eccentricity * np.sinh(anomaly) - anomaly - mean_anomaly_rad
            derivative = eccentricity * np.cosh(anomaly) - 1.0
            correction = equation / derivative
            anomaly -= correction
            if abs(correction) <= tolerance:
                return float(anomaly)
    raise RuntimeError("Kepler equation did not converge")


def eccentric_to_true_anomaly(anomaly_rad: float, eccentricity: float) -> float:
    """Convert elliptic eccentric or hyperbolic anomaly to true anomaly."""
    if eccentricity < 1.0:
        numerator = np.sqrt(1.0 + eccentricity) * np.sin(0.5 * anomaly_rad)
        denominator = np.sqrt(1.0 - eccentricity) * np.cos(0.5 * anomaly_rad)
    elif eccentricity > 1.0:
        numerator = np.sqrt(eccentricity + 1.0) * np.sinh(0.5 * anomaly_rad)
        denominator = np.sqrt(eccentricity - 1.0) * np.cosh(0.5 * anomaly_rad)
    else:
        raise ValueError("parabolic anomaly conversion is not supported")
    return float(2.0 * np.arctan2(numerator, denominator))


def elements_to_state(
    elements: ClassicalOrbitalElements, gravitational_parameter_m3_s2: float
) -> tuple[FloatArray, FloatArray]:
    """Convert classical elements to inertial Cartesian position and velocity."""
    mu = gravitational_parameter_m3_s2
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational parameter must be positive and finite")
    eccentricity = elements.eccentricity
    semi_latus_rectum_m = elements.semi_major_axis_m * (1.0 - eccentricity**2)
    if semi_latus_rectum_m <= 0.0:
        raise ValueError("orbital elements produce a nonpositive semi-latus rectum")
    anomaly = elements.true_anomaly_rad
    denominator = 1.0 + eccentricity * np.cos(anomaly)
    if denominator <= 0.0:
        raise ValueError("true anomaly lies outside the physical conic branch")
    position_perifocal = (
        semi_latus_rectum_m / denominator * np.array([np.cos(anomaly), np.sin(anomaly), 0.0])
    )
    velocity_perifocal = np.sqrt(mu / semi_latus_rectum_m) * np.array(
        [-np.sin(anomaly), eccentricity + np.cos(anomaly), 0.0]
    )
    node = elements.ascending_node_rad
    inclination = elements.inclination_rad
    periapsis = elements.argument_of_periapsis_rad
    cosine_node, sine_node = np.cos(node), np.sin(node)
    cosine_i, sine_i = np.cos(inclination), np.sin(inclination)
    cosine_p, sine_p = np.cos(periapsis), np.sin(periapsis)
    transform = np.array(
        [
            [
                cosine_node * cosine_p - sine_node * sine_p * cosine_i,
                -cosine_node * sine_p - sine_node * cosine_p * cosine_i,
                sine_node * sine_i,
            ],
            [
                sine_node * cosine_p + cosine_node * sine_p * cosine_i,
                -sine_node * sine_p + cosine_node * cosine_p * cosine_i,
                -cosine_node * sine_i,
            ],
            [sine_p * sine_i, cosine_p * sine_i, cosine_i],
        ]
    )
    return transform @ position_perifocal, transform @ velocity_perifocal


def state_to_elements(
    position_m: npt.ArrayLike,
    velocity_mps: npt.ArrayLike,
    gravitational_parameter_m3_s2: float,
    *,
    singularity_tolerance: float = 1.0e-10,
) -> ClassicalOrbitalElements:
    """Convert an inertial Cartesian state to a nonsingular classical convention."""
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position and velocity must contain three components")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("state must be finite")
    mu = gravitational_parameter_m3_s2
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("gravitational parameter must be positive and finite")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("position radius must be positive")
    angular_momentum = np.cross(position, velocity)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm <= 0.0:
        raise ValueError("radial state has undefined classical elements")
    node_vector = np.cross([0.0, 0.0, 1.0], angular_momentum)
    node_norm = float(np.linalg.norm(node_vector))
    eccentricity_vector = np.cross(velocity, angular_momentum) / mu - position / radius
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    energy = 0.5 * float(np.dot(velocity, velocity)) - mu / radius
    if abs(energy) <= singularity_tolerance * mu / radius:
        raise ValueError("parabolic state has undefined semi-major axis")
    semi_major_axis_m = -mu / (2.0 * energy)
    inclination = float(np.arccos(np.clip(angular_momentum[2] / angular_momentum_norm, -1.0, 1.0)))

    if node_norm > singularity_tolerance:
        ascending_node = float(np.mod(np.arctan2(node_vector[1], node_vector[0]), 2.0 * np.pi))
    else:
        ascending_node = 0.0
    if eccentricity > singularity_tolerance and node_norm > singularity_tolerance:
        argument_periapsis = float(
            np.mod(
                np.arctan2(
                    np.dot(np.cross(node_vector, eccentricity_vector), angular_momentum)
                    / (node_norm * eccentricity * angular_momentum_norm),
                    np.dot(node_vector, eccentricity_vector) / (node_norm * eccentricity),
                ),
                2.0 * np.pi,
            )
        )
    elif eccentricity > singularity_tolerance:
        argument_periapsis = float(
            np.mod(np.arctan2(eccentricity_vector[1], eccentricity_vector[0]), 2.0 * np.pi)
        )
    else:
        argument_periapsis = 0.0

    if eccentricity > singularity_tolerance:
        true_anomaly = float(
            np.arctan2(
                np.dot(np.cross(eccentricity_vector, position), angular_momentum)
                / (eccentricity * radius * angular_momentum_norm),
                np.dot(eccentricity_vector, position) / (eccentricity * radius),
            )
        )
    elif node_norm > singularity_tolerance:
        true_anomaly = float(
            np.arctan2(
                np.dot(np.cross(node_vector, position), angular_momentum)
                / (node_norm * radius * angular_momentum_norm),
                np.dot(node_vector, position) / (node_norm * radius),
            )
        )
    else:
        true_anomaly = float(np.arctan2(position[1], position[0]))
    return ClassicalOrbitalElements(
        semi_major_axis_m=float(semi_major_axis_m),
        eccentricity=eccentricity,
        inclination_rad=inclination,
        ascending_node_rad=ascending_node,
        argument_of_periapsis_rad=argument_periapsis,
        true_anomaly_rad=float(np.mod(true_anomaly, 2.0 * np.pi)),
    )
