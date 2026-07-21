"""Rotating-oblate-planet strapdown inertial mechanisation in local NED."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    body_rotation_rate_ned,
    dcm_ecef_to_ned,
    geodetic_to_ecef,
    meridian_radius_m,
    prime_vertical_radius_m,
    transport_rate_ned,
)
from aerognc.mathematics.quaternion import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_dcm,
    rotation_vector_to_quaternion,
)
from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class ImuIncrement:
    """Integrated IMU output over ``[start_time_s, end_time_s]``.

    ``delta_angle_body_rad`` is the body angular increment relative to inertial,
    resolved in FRD body axes. ``delta_velocity_body_mps`` is integrated specific
    force, also body-resolved. Both include sensor errors when used by an estimator.
    """

    start_time_s: float
    end_time_s: float
    delta_angle_body_rad: FloatArray
    delta_velocity_body_mps: FloatArray

    def __init__(
        self,
        start_time_s: float,
        end_time_s: float,
        delta_angle_body_rad: npt.ArrayLike,
        delta_velocity_body_mps: npt.ArrayLike,
    ) -> None:
        if not np.all(np.isfinite([start_time_s, end_time_s])):
            raise ValueError("IMU increment times must be finite")
        if end_time_s <= start_time_s:
            raise ValueError("IMU increment end_time_s must exceed start_time_s")
        object.__setattr__(self, "start_time_s", float(start_time_s))
        object.__setattr__(self, "end_time_s", float(end_time_s))
        object.__setattr__(
            self,
            "delta_angle_body_rad",
            as_vector(delta_angle_body_rad, 3, name="delta_angle_body_rad"),
        )
        object.__setattr__(
            self,
            "delta_velocity_body_mps",
            as_vector(delta_velocity_body_mps, 3, name="delta_velocity_body_mps"),
        )

    @property
    def duration_s(self) -> float:
        """Increment duration [s]."""
        return self.end_time_s - self.start_time_s


def compensate_two_sample_imu(first: ImuIncrement, second: ImuIncrement) -> ImuIncrement:
    """Combine two contiguous raw increments with coning/sculling compensation.

    The correction is the conventional two-sample, second-order algorithm. It is
    implemented directly so its signs and multiplication order remain inspectable.
    """
    tolerance_s = 1.0e-10 * max(1.0, abs(first.end_time_s), abs(second.start_time_s))
    if abs(first.end_time_s - second.start_time_s) > tolerance_s:
        raise ValueError("two-sample IMU increments must be contiguous")
    theta_1 = first.delta_angle_body_rad
    theta_2 = second.delta_angle_body_rad
    velocity_1 = first.delta_velocity_body_mps
    velocity_2 = second.delta_velocity_body_mps
    delta_angle = theta_1 + theta_2 + (2.0 / 3.0) * np.cross(theta_1, theta_2)
    summed_angle = theta_1 + theta_2
    summed_velocity = velocity_1 + velocity_2
    delta_velocity = (
        summed_velocity
        + 0.5 * np.cross(summed_angle, summed_velocity)
        + (2.0 / 3.0) * (np.cross(theta_1, velocity_2) + np.cross(velocity_1, theta_2))
    )
    return ImuIncrement(
        first.start_time_s,
        second.end_time_s,
        delta_angle,
        delta_velocity,
    )


@dataclass(slots=True)
class RotatingNavigationState:
    """Nominal geodetic navigation state with inertial-sensor biases."""

    geodetic: GeodeticPosition
    velocity_ned_mps: FloatArray
    quaternion_nb: FloatArray
    gyro_bias_body_radps: FloatArray
    accelerometer_bias_body_mps2: FloatArray

    def __init__(
        self,
        geodetic: GeodeticPosition,
        velocity_ned_mps: npt.ArrayLike,
        quaternion_nb: npt.ArrayLike,
        gyro_bias_body_radps: npt.ArrayLike = (0.0, 0.0, 0.0),
        accelerometer_bias_body_mps2: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        self.geodetic = geodetic
        self.velocity_ned_mps = as_vector(velocity_ned_mps, 3, name="velocity_ned_mps")
        self.quaternion_nb = normalize_quaternion(quaternion_nb)
        self.gyro_bias_body_radps = as_vector(gyro_bias_body_radps, 3, name="gyro_bias_body_radps")
        self.accelerometer_bias_body_mps2 = as_vector(
            accelerometer_bias_body_mps2, 3, name="accelerometer_bias_body_mps2"
        )

    def copy(self) -> RotatingNavigationState:
        """Return a deep numerical copy suitable for delayed-state history."""
        return RotatingNavigationState(
            self.geodetic,
            self.velocity_ned_mps,
            self.quaternion_nb,
            self.gyro_bias_body_radps,
            self.accelerometer_bias_body_mps2,
        )


@dataclass(frozen=True, slots=True)
class MechanizationDiagnostics:
    """Frame rates and forces used by one mechanisation step."""

    corrected_angular_rate_body_radps: FloatArray
    corrected_specific_force_body_mps2: FloatArray
    body_rotation_rate_ned_radps: FloatArray
    transport_rate_ned_radps: FloatArray
    gravity_ned_mps2: FloatArray
    coriolis_transport_ned_mps2: FloatArray
    dcm_nb_midpoint: FloatArray


def gravity_ned_mps2(
    geodetic: GeodeticPosition,
    planet: RotatingOblatePlanet,
) -> FloatArray:
    """Return gravity plus centrifugal acceleration in local NED [m/s^2]."""
    position_ecef_m = geodetic_to_ecef(geodetic, planet.ellipsoid)
    acceleration_ecef_mps2 = planet.gravity_ecef_mps2(position_ecef_m)
    acceleration_ecef_mps2 += planet.centrifugal_ecef_mps2(position_ecef_m)
    return dcm_ecef_to_ned(geodetic.latitude_rad, geodetic.longitude_rad) @ acceleration_ecef_mps2


def displace_geodetic_ned(
    geodetic: GeodeticPosition,
    displacement_ned_m: npt.ArrayLike,
    planet: RotatingOblatePlanet,
) -> GeodeticPosition:
    """Apply a small local-NED displacement to geodetic coordinates."""
    north_m, east_m, down_m = as_vector(displacement_ned_m, 3, name="displacement_ned_m")
    meridian_m = meridian_radius_m(geodetic.latitude_rad, planet.ellipsoid) + geodetic.altitude_m
    transverse_m = (
        prime_vertical_radius_m(geodetic.latitude_rad, planet.ellipsoid) + geodetic.altitude_m
    )
    cosine_latitude = float(np.cos(geodetic.latitude_rad))
    if meridian_m <= 0.0 or transverse_m <= 0.0 or abs(cosine_latitude) <= 1.0e-10:
        raise ValueError("geodetic displacement is outside the local-NED domain")
    longitude_rad = geodetic.longitude_rad + east_m / (transverse_m * cosine_latitude)
    longitude_rad = float((longitude_rad + np.pi) % (2.0 * np.pi) - np.pi)
    return GeodeticPosition(
        latitude_rad=float(geodetic.latitude_rad + north_m / meridian_m),
        longitude_rad=longitude_rad,
        altitude_m=float(geodetic.altitude_m - down_m),
    )


def propagate_rotating_strapdown(
    state: RotatingNavigationState,
    increment: ImuIncrement,
    planet: RotatingOblatePlanet,
) -> MechanizationDiagnostics:
    """Advance one bias-corrected rotating-NED strapdown step in place.

    Gravity includes central/J2 and centrifugal terms. Coriolis and transport terms
    are applied separately as ``-(2 omega_ie + omega_en) x v``. Attitude removes
    navigation-frame rotation on the left and applies body rotation on the right.
    """
    step_s = increment.duration_s
    corrected_delta_angle = increment.delta_angle_body_rad - state.gyro_bias_body_radps * step_s
    corrected_delta_velocity = (
        increment.delta_velocity_body_mps - state.accelerometer_bias_body_mps2 * step_s
    )
    corrected_rate = corrected_delta_angle / step_s
    corrected_force = corrected_delta_velocity / step_s

    omega_ie_n = body_rotation_rate_ned(state.geodetic.latitude_rad, planet.rotation_rate_radps)
    omega_en_n = transport_rate_ned(state.geodetic, state.velocity_ned_mps, planet.ellipsoid)
    omega_in_n = omega_ie_n + omega_en_n
    half_navigation_rotation = rotation_vector_to_quaternion(-0.5 * omega_in_n * step_s)
    half_body_rotation = rotation_vector_to_quaternion(0.5 * corrected_delta_angle)
    midpoint_quaternion_nb = normalize_quaternion(
        quaternion_multiply(
            quaternion_multiply(half_navigation_rotation, state.quaternion_nb),
            half_body_rotation,
        )
    )
    dcm_nb_midpoint = quaternion_to_dcm(midpoint_quaternion_nb)
    local_gravity = gravity_ned_mps2(state.geodetic, planet)
    frame_acceleration = -np.cross(
        2.0 * omega_ie_n + omega_en_n,
        state.velocity_ned_mps,
    )
    old_velocity = state.velocity_ned_mps.copy()
    state.velocity_ned_mps += (
        dcm_nb_midpoint @ corrected_delta_velocity + (local_gravity + frame_acceleration) * step_s
    )
    average_velocity = 0.5 * (old_velocity + state.velocity_ned_mps)
    state.geodetic = displace_geodetic_ned(
        state.geodetic,
        average_velocity * step_s,
        planet,
    )
    navigation_rotation = rotation_vector_to_quaternion(-omega_in_n * step_s)
    body_rotation = rotation_vector_to_quaternion(corrected_delta_angle)
    state.quaternion_nb = normalize_quaternion(
        quaternion_multiply(
            quaternion_multiply(navigation_rotation, state.quaternion_nb),
            body_rotation,
        )
    )
    return MechanizationDiagnostics(
        corrected_angular_rate_body_radps=corrected_rate,
        corrected_specific_force_body_mps2=corrected_force,
        body_rotation_rate_ned_radps=omega_ie_n,
        transport_rate_ned_radps=omega_en_n,
        gravity_ned_mps2=local_gravity,
        coriolis_transport_ned_mps2=frame_acceleration,
        dcm_nb_midpoint=dcm_nb_midpoint,
    )
