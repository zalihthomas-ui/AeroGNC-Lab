"""Carefully scoped vertical navigation extended Kalman filter."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray, as_vector


@dataclass(frozen=True, slots=True)
class VerticalFilterTuning:
    """Continuous/discrete noise assumptions for the vertical filter."""

    acceleration_process_std_mps2: float
    bias_random_walk_std_mps2_per_sqrt_s: float
    barometer_std_m: float
    gnss_altitude_std_m: float
    gnss_vertical_velocity_std_mps: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.acceleration_process_std_mps2,
                self.bias_random_walk_std_mps2_per_sqrt_s,
                self.barometer_std_m,
                self.gnss_altitude_std_m,
                self.gnss_vertical_velocity_std_mps,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("vertical-filter standard deviations must be positive and finite")


class VerticalNavigationEKF:
    """Three-state altitude/up-velocity/accelerometer-bias navigation filter.

    The acceleration input is assumed already rotated into local vertical and
    gravity-compensated using known attitude. The process is affine in the state but
    uses the same Jacobian/covariance machinery as the planned nonlinear extension.
    """

    def __init__(
        self,
        initial_state: npt.ArrayLike,
        initial_covariance: npt.ArrayLike,
        tuning: VerticalFilterTuning,
    ) -> None:
        self.state = as_vector(initial_state, 3, name="initial_state")
        covariance = np.asarray(initial_covariance, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("initial_covariance must be a finite 3-by-3 matrix")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12):
            raise ValueError("initial_covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("initial_covariance must be positive definite")
        self.covariance = covariance.copy()
        self.tuning = tuning

    def predict(self, measured_vertical_acceleration_mps2: float, step_s: float) -> None:
        """Propagate state/covariance using gravity-compensated acceleration input."""
        if not np.isfinite(measured_vertical_acceleration_mps2):
            raise ValueError("measured acceleration must be finite")
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("step_s must be positive and finite")
        altitude_m, velocity_mps, bias_mps2 = self.state
        corrected_acceleration = measured_vertical_acceleration_mps2 - bias_mps2
        self.state = np.array(
            [
                altitude_m + velocity_mps * step_s + 0.5 * corrected_acceleration * step_s**2,
                velocity_mps + corrected_acceleration * step_s,
                bias_mps2,
            ]
        )
        transition = np.array(
            [
                [1.0, step_s, -0.5 * step_s**2],
                [0.0, 1.0, -step_s],
                [0.0, 0.0, 1.0],
            ]
        )
        acceleration_variance = self.tuning.acceleration_process_std_mps2**2
        acceleration_mapping = np.array([0.5 * step_s**2, step_s, 0.0])
        process_covariance = acceleration_variance * np.outer(
            acceleration_mapping, acceleration_mapping
        )
        process_covariance[2, 2] += self.tuning.bias_random_walk_std_mps2_per_sqrt_s**2 * step_s
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        self._symmetrise()

    def update_barometer(self, altitude_m: float) -> None:
        """Fuse one scalar barometric altitude measurement [m]."""
        self._update(
            np.array([altitude_m]),
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[self.tuning.barometer_std_m**2]]),
        )

    def update_gnss(self, altitude_m: float, vertical_velocity_up_mps: float) -> None:
        """Fuse local GNSS-like altitude and upward velocity."""
        self._update(
            np.array([altitude_m, vertical_velocity_up_mps]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.diag(
                [
                    self.tuning.gnss_altitude_std_m**2,
                    self.tuning.gnss_vertical_velocity_std_mps**2,
                ]
            ),
        )

    def _update(self, measurement: FloatArray, matrix: FloatArray, noise: FloatArray) -> None:
        if not np.all(np.isfinite(measurement)):
            raise ValueError("navigation measurement must be finite")
        innovation = measurement - matrix @ self.state
        innovation_covariance = matrix @ self.covariance @ matrix.T + noise
        gain = np.linalg.solve(innovation_covariance, matrix @ self.covariance).T
        self.state = self.state + gain @ innovation
        identity = np.eye(3)
        residual = identity - gain @ matrix
        self.covariance = residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        self._symmetrise()

    def _symmetrise(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        if np.any(np.linalg.eigvalsh(self.covariance) < -1.0e-12):
            raise FloatingPointError("navigation covariance lost positive semidefiniteness")
