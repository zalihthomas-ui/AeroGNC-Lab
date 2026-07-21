"""Fifteen-state NED inertial/GNSS/barometric error-state Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.quaternion import (
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_dcm,
)
from aerognc.mathematics.vectors import FloatArray, as_vector, skew_symmetric


@dataclass(frozen=True, slots=True)
class ErrorStateFilterTuning:
    """Continuous IMU noise densities and bias random walks in SI/radian units."""

    gyro_noise_std_radps_per_sqrt_hz: float
    accelerometer_noise_std_mps2_per_sqrt_hz: float
    gyro_bias_random_walk_std_radps2_per_sqrt_hz: float
    accelerometer_bias_random_walk_std_mps3_per_sqrt_hz: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.gyro_noise_std_radps_per_sqrt_hz,
                self.accelerometer_noise_std_mps2_per_sqrt_hz,
                self.gyro_bias_random_walk_std_radps2_per_sqrt_hz,
                self.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("error-state filter noise densities must be positive and finite")


@dataclass(slots=True)
class NavigationNominalState:
    """Nominal navigation state paired with a 15-component local error state."""

    position_ned_m: FloatArray
    velocity_ned_mps: FloatArray
    quaternion_nb: FloatArray
    gyro_bias_body_radps: FloatArray
    accelerometer_bias_body_mps2: FloatArray

    def __init__(
        self,
        position_ned_m: npt.ArrayLike,
        velocity_ned_mps: npt.ArrayLike,
        quaternion_nb: npt.ArrayLike,
        gyro_bias_body_radps: npt.ArrayLike = (0.0, 0.0, 0.0),
        accelerometer_bias_body_mps2: npt.ArrayLike = (0.0, 0.0, 0.0),
    ) -> None:
        self.position_ned_m = as_vector(position_ned_m, 3, name="position_ned_m")
        self.velocity_ned_mps = as_vector(velocity_ned_mps, 3, name="velocity_ned_mps")
        self.quaternion_nb = normalize_quaternion(quaternion_nb)
        self.gyro_bias_body_radps = as_vector(gyro_bias_body_radps, 3, name="gyro_bias_body_radps")
        self.accelerometer_bias_body_mps2 = as_vector(
            accelerometer_bias_body_mps2, 3, name="accelerometer_bias_body_mps2"
        )


def _rotation_vector_quaternion(rotation_vector_rad: FloatArray) -> FloatArray:
    angle = float(np.linalg.norm(rotation_vector_rad))
    if angle <= 1.0e-12:
        return normalize_quaternion(
            np.concatenate(([1.0 - angle**2 / 8.0], 0.5 * rotation_vector_rad))
        )
    half_angle = 0.5 * angle
    return np.concatenate(([np.cos(half_angle)], np.sin(half_angle) * rotation_vector_rad / angle))


class ErrorStateNavigationEKF:
    """Loosely coupled 15-state ESKF for a flat, nonrotating local NED frame.

    Error ordering is ``[δp_n, δv_n, δθ_b, δb_g, δb_a]``. The nominal quaternion
    is Hamilton scalar-first ``q_nb`` and maps FRD body vectors into NED.
    Earth rotation, curvature, coning/sculling and lever arms are intentionally out
    of scope; this filter is an auditable research-rocket navigation baseline.
    """

    def __init__(
        self,
        nominal_state: NavigationNominalState,
        initial_covariance: npt.ArrayLike,
        tuning: ErrorStateFilterTuning,
    ) -> None:
        covariance = np.asarray(initial_covariance, dtype=np.float64)
        if covariance.shape != (15, 15) or not np.all(np.isfinite(covariance)):
            raise ValueError("initial ESKF covariance must be a finite 15-by-15 matrix")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12):
            raise ValueError("initial ESKF covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("initial ESKF covariance must be positive definite")
        self.state = nominal_state
        self.covariance = covariance.copy()
        self.tuning = tuning
        self.last_innovation: FloatArray | None = None

    def predict(
        self,
        measured_angular_rate_body_radps: npt.ArrayLike,
        measured_specific_force_body_mps2: npt.ArrayLike,
        step_s: float,
        gravity_ned_mps2: npt.ArrayLike = (0.0, 0.0, 9.80665),
    ) -> None:
        """Propagate nominal strapdown kinematics and first-order error covariance."""
        angular_rate = as_vector(
            measured_angular_rate_body_radps, 3, name="measured_angular_rate_body_radps"
        )
        specific_force = as_vector(
            measured_specific_force_body_mps2, 3, name="measured_specific_force_body_mps2"
        )
        gravity = as_vector(gravity_ned_mps2, 3, name="gravity_ned_mps2")
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("ESKF step_s must be positive and finite")
        corrected_rate = angular_rate - self.state.gyro_bias_body_radps
        corrected_force = specific_force - self.state.accelerometer_bias_body_mps2
        dcm_nb = quaternion_to_dcm(self.state.quaternion_nb)
        acceleration_ned = dcm_nb @ corrected_force + gravity
        self.state.position_ned_m += (
            self.state.velocity_ned_mps * step_s + 0.5 * acceleration_ned * step_s**2
        )
        self.state.velocity_ned_mps += acceleration_ned * step_s
        delta_quaternion = _rotation_vector_quaternion(corrected_rate * step_s)
        self.state.quaternion_nb = normalize_quaternion(
            quaternion_multiply(self.state.quaternion_nb, delta_quaternion)
        )

        system = np.zeros((15, 15), dtype=np.float64)
        system[0:3, 3:6] = np.eye(3)
        system[3:6, 6:9] = -dcm_nb @ skew_symmetric(corrected_force)
        system[3:6, 12:15] = -dcm_nb
        system[6:9, 6:9] = -skew_symmetric(corrected_rate)
        system[6:9, 9:12] = -np.eye(3)
        transition = np.eye(15) + system * step_s + 0.5 * (system @ system) * step_s**2
        noise_mapping = np.zeros((15, 12), dtype=np.float64)
        noise_mapping[3:6, 0:3] = dcm_nb
        noise_mapping[6:9, 3:6] = -np.eye(3)
        noise_mapping[9:12, 6:9] = np.eye(3)
        noise_mapping[12:15, 9:12] = np.eye(3)
        noise_variances = np.array(
            [
                *([self.tuning.accelerometer_noise_std_mps2_per_sqrt_hz**2] * 3),
                *([self.tuning.gyro_noise_std_radps_per_sqrt_hz**2] * 3),
                *([self.tuning.gyro_bias_random_walk_std_radps2_per_sqrt_hz**2] * 3),
                *([self.tuning.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz**2] * 3),
            ]
        )
        process_covariance = noise_mapping @ np.diag(noise_variances) @ noise_mapping.T * step_s
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        self._stabilise_covariance()

    def update_gnss(
        self,
        position_ned_m: npt.ArrayLike,
        velocity_ned_mps: npt.ArrayLike,
        measurement_covariance: npt.ArrayLike,
    ) -> None:
        """Fuse one civilian GNSS-like NED position/velocity observation."""
        measurement = np.concatenate(
            (
                as_vector(position_ned_m, 3, name="GNSS position_ned_m"),
                as_vector(velocity_ned_mps, 3, name="GNSS velocity_ned_mps"),
            )
        )
        predicted = np.concatenate((self.state.position_ned_m, self.state.velocity_ned_mps))
        matrix = np.zeros((6, 15), dtype=np.float64)
        matrix[0:3, 0:3] = np.eye(3)
        matrix[3:6, 3:6] = np.eye(3)
        self._update(measurement - predicted, matrix, measurement_covariance)

    def update_barometric_altitude(self, altitude_m: float, measurement_variance_m2: float) -> None:
        """Fuse geometric altitude, using ``altitude=-down_position``."""
        if not np.isfinite(altitude_m) or not np.isfinite(measurement_variance_m2):
            raise ValueError("barometric altitude and variance must be finite")
        if measurement_variance_m2 <= 0.0:
            raise ValueError("barometric altitude variance must be positive")
        innovation = np.array([altitude_m - (-self.state.position_ned_m[2])])
        matrix = np.zeros((1, 15), dtype=np.float64)
        matrix[0, 2] = -1.0
        self._update(innovation, matrix, np.array([[measurement_variance_m2]]))

    def _update(
        self,
        innovation: FloatArray,
        measurement_matrix: FloatArray,
        measurement_covariance: npt.ArrayLike,
    ) -> None:
        noise = np.asarray(measurement_covariance, dtype=np.float64)
        measurement_count = innovation.size
        if noise.shape != (measurement_count, measurement_count):
            raise ValueError("measurement covariance shape does not match innovation")
        if not np.all(np.isfinite(noise)) or not np.allclose(noise, noise.T):
            raise ValueError("measurement covariance must be finite and symmetric")
        if np.any(np.linalg.eigvalsh(noise) <= 0.0):
            raise ValueError("measurement covariance must be positive definite")
        innovation_covariance = measurement_matrix @ self.covariance @ measurement_matrix.T + noise
        gain = np.linalg.solve(innovation_covariance, measurement_matrix @ self.covariance).T
        error_state = gain @ innovation
        identity = np.eye(15)
        residual = identity - gain @ measurement_matrix
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self._inject(error_state)
        self.last_innovation = innovation.copy()
        self._stabilise_covariance()

    def _inject(self, error_state: FloatArray) -> None:
        self.state.position_ned_m += error_state[0:3]
        self.state.velocity_ned_mps += error_state[3:6]
        self.state.quaternion_nb = normalize_quaternion(
            quaternion_multiply(
                self.state.quaternion_nb,
                _rotation_vector_quaternion(error_state[6:9]),
            )
        )
        self.state.gyro_bias_body_radps += error_state[9:12]
        self.state.accelerometer_bias_body_mps2 += error_state[12:15]

    def standard_deviation(self) -> FloatArray:
        """Return one-sigma bounds in documented error-state order."""
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def _stabilise_covariance(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        if eigenvalues[0] < -1.0e-10:
            raise FloatingPointError("ESKF covariance lost positive semidefiniteness")
