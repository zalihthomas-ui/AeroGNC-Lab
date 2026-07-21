"""Fixed-lag rotating-NED error-state Kalman filter with integrity monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.gnc.error_state_ekf import ErrorStateFilterTuning
from aerognc.gnc.strapdown_ins import (
    ImuIncrement,
    RotatingNavigationState,
    displace_geodetic_ned,
    propagate_rotating_strapdown,
)
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    ecef_position_to_ned,
    geodetic_to_ecef,
)
from aerognc.mathematics.quaternion import (
    normalize_quaternion,
    quaternion_multiply,
    rotation_vector_to_quaternion,
)
from aerognc.mathematics.vectors import FloatArray, as_vector, skew_symmetric

SensorHealth = Literal["healthy", "degraded", "failed"]


@dataclass(frozen=True, slots=True)
class InnovationGateConfiguration:
    """NIS thresholds and consecutive-rejection health-state limits."""

    gnss_nis_threshold: float
    barometer_nis_threshold: float
    degraded_after_rejections: int = 2
    failed_after_rejections: int = 5

    def __post_init__(self) -> None:
        thresholds = np.array([self.gnss_nis_threshold, self.barometer_nis_threshold])
        if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
            raise ValueError("innovation-gate thresholds must be positive and finite")
        if self.degraded_after_rejections <= 0:
            raise ValueError("degraded_after_rejections must be positive")
        if self.failed_after_rejections < self.degraded_after_rejections:
            raise ValueError("failed_after_rejections must not precede degraded state")


@dataclass(frozen=True, slots=True)
class MeasurementUpdateResult:
    """Auditable result of one current or delayed aiding update."""

    sensor_name: str
    sample_time_s: float
    processing_time_s: float
    accepted: bool
    nis: float
    threshold: float
    replayed_step_count: int
    health: SensorHealth
    reason: str


@dataclass(slots=True)
class _HealthCounter:
    accepted_count: int = 0
    rejected_count: int = 0
    consecutive_rejections: int = 0
    health: SensorHealth = "healthy"


@dataclass(frozen=True, slots=True)
class SensorIntegritySummary:
    """Cumulative measurement-gate and sensor-health evidence."""

    sensor_name: str
    accepted_count: int
    rejected_count: int
    consecutive_rejections: int
    health: SensorHealth


@dataclass(frozen=True, slots=True)
class _MeasurementRecord:
    sensor_name: str
    measurement: FloatArray
    covariance: FloatArray


@dataclass(slots=True)
class _HistoryEntry:
    time_s: float
    state: RotatingNavigationState
    covariance: FloatArray
    increment_from_previous: ImuIncrement | None
    measurements: list[_MeasurementRecord] = field(default_factory=list)


class DelayedRotatingNavigationESKF:
    """Fifteen-state geodetic ESKF with fixed-lag out-of-sequence replay.

    The local error ordering is ``[delta p_n, delta v_n, delta theta_b,
    delta b_g, delta b_a]`` with SI units and radians. The nominal state uses
    ellipsoidal latitude/longitude/altitude, NED velocity, and Hamilton scalar-first
    ``q_nb``. Accepted aiding records are retained at their acquisition epoch and
    reapplied whenever an older delayed measurement causes history replay.
    """

    def __init__(
        self,
        state: RotatingNavigationState,
        initial_covariance: npt.ArrayLike,
        tuning: ErrorStateFilterTuning,
        planet: RotatingOblatePlanet,
        position_origin: GeodeticPosition,
        *,
        fixed_lag_s: float,
        gate_configuration: InnovationGateConfiguration,
        initial_time_s: float = 0.0,
    ) -> None:
        covariance = np.asarray(initial_covariance, dtype=np.float64)
        if covariance.shape != (15, 15) or not np.all(np.isfinite(covariance)):
            raise ValueError("initial ESKF covariance must be a finite 15-by-15 matrix")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12):
            raise ValueError("initial ESKF covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("initial ESKF covariance must be positive definite")
        if not np.isfinite(fixed_lag_s) or fixed_lag_s <= 0.0:
            raise ValueError("fixed_lag_s must be positive and finite")
        if not np.isfinite(initial_time_s) or initial_time_s < 0.0:
            raise ValueError("initial_time_s must be finite and nonnegative")
        self.state = state.copy()
        self.covariance = covariance.copy()
        self.tuning = tuning
        self.planet = planet
        self.position_origin = position_origin
        self.fixed_lag_s = float(fixed_lag_s)
        self.gate_configuration = gate_configuration
        self.current_time_s = float(initial_time_s)
        self.last_transition = np.eye(15, dtype=np.float64)
        self._health: dict[str, _HealthCounter] = {
            "gnss": _HealthCounter(),
            "barometer": _HealthCounter(),
        }
        self._history = [
            _HistoryEntry(
                self.current_time_s,
                self.state.copy(),
                self.covariance.copy(),
                None,
            )
        ]

    def predict(self, increment: ImuIncrement) -> None:
        """Propagate nominal state/covariance and append a fixed-lag snapshot."""
        tolerance_s = 1.0e-9 * max(1.0, abs(self.current_time_s))
        if abs(increment.start_time_s - self.current_time_s) > tolerance_s:
            raise ValueError("IMU increment does not start at the current filter time")
        self._predict_core(increment)
        self.current_time_s = increment.end_time_s
        self._history.append(
            _HistoryEntry(
                self.current_time_s,
                self.state.copy(),
                self.covariance.copy(),
                increment,
            )
        )
        oldest_time_s = self.current_time_s - self.fixed_lag_s
        while len(self._history) > 1 and self._history[1].time_s < oldest_time_s:
            self._history.pop(0)

    def update_gnss(
        self,
        position_ned_m: npt.ArrayLike,
        velocity_ned_mps: npt.ArrayLike,
        measurement_covariance: npt.ArrayLike,
        *,
        sample_time_s: float,
    ) -> MeasurementUpdateResult:
        """Fuse a delayed civilian GNSS-like local position/velocity observation."""
        measurement = np.concatenate(
            (
                as_vector(position_ned_m, 3, name="GNSS position_ned_m"),
                as_vector(velocity_ned_mps, 3, name="GNSS velocity_ned_mps"),
            )
        )
        covariance = self._measurement_covariance(measurement_covariance, 6)
        return self._delayed_update(
            _MeasurementRecord("gnss", measurement, covariance),
            sample_time_s,
            self.gate_configuration.gnss_nis_threshold,
        )

    def update_barometric_altitude(
        self,
        altitude_m: float,
        measurement_variance_m2: float,
        *,
        sample_time_s: float,
    ) -> MeasurementUpdateResult:
        """Fuse delayed ellipsoidal altitude [m] with NIS gating."""
        if not np.isfinite(altitude_m):
            raise ValueError("barometric altitude must be finite")
        covariance = self._measurement_covariance([[measurement_variance_m2]], 1)
        return self._delayed_update(
            _MeasurementRecord("barometer", np.array([altitude_m]), covariance),
            sample_time_s,
            self.gate_configuration.barometer_nis_threshold,
        )

    def sensor_integrity(self, sensor_name: str) -> SensorIntegritySummary:
        """Return cumulative integrity counters for ``gnss`` or ``barometer``."""
        if sensor_name not in self._health:
            raise ValueError(f"unknown aiding sensor: {sensor_name}")
        counter = self._health[sensor_name]
        return SensorIntegritySummary(
            sensor_name=sensor_name,
            accepted_count=counter.accepted_count,
            rejected_count=counter.rejected_count,
            consecutive_rejections=counter.consecutive_rejections,
            health=counter.health,
        )

    def standard_deviation(self) -> FloatArray:
        """Return one-sigma values in the documented 15-state error order."""
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def history_start_time_s(self) -> float:
        """Oldest acquisition epoch currently available for delayed replay."""
        return self._history[0].time_s

    def _predict_core(self, increment: ImuIncrement) -> None:
        diagnostics = propagate_rotating_strapdown(self.state, increment, self.planet)
        step_s = increment.duration_s
        system = np.zeros((15, 15), dtype=np.float64)
        system[0:3, 3:6] = np.eye(3)
        frame_rate_ned = (
            2.0 * diagnostics.body_rotation_rate_ned_radps + diagnostics.transport_rate_ned_radps
        )
        system[3:6, 3:6] = -skew_symmetric(frame_rate_ned)
        system[3:6, 6:9] = -diagnostics.dcm_nb_midpoint @ skew_symmetric(
            diagnostics.corrected_specific_force_body_mps2
        )
        system[3:6, 12:15] = -diagnostics.dcm_nb_midpoint
        system[6:9, 6:9] = -skew_symmetric(diagnostics.corrected_angular_rate_body_radps)
        system[6:9, 9:12] = -np.eye(3)
        transition = np.eye(15) + system * step_s + 0.5 * (system @ system) * step_s**2
        noise_mapping = np.zeros((15, 12), dtype=np.float64)
        noise_mapping[3:6, 0:3] = diagnostics.dcm_nb_midpoint
        noise_mapping[6:9, 3:6] = -np.eye(3)
        noise_mapping[9:12, 6:9] = np.eye(3)
        noise_mapping[12:15, 9:12] = np.eye(3)
        variances = np.array(
            [
                *([self.tuning.accelerometer_noise_std_mps2_per_sqrt_hz**2] * 3),
                *([self.tuning.gyro_noise_std_radps_per_sqrt_hz**2] * 3),
                *([self.tuning.gyro_bias_random_walk_std_radps2_per_sqrt_hz**2] * 3),
                *([self.tuning.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz**2] * 3),
            ]
        )
        process_covariance = noise_mapping @ np.diag(variances) @ noise_mapping.T * step_s
        self.covariance = transition @ self.covariance @ transition.T + process_covariance
        self.last_transition = transition
        self._stabilise_covariance()

    def _delayed_update(
        self,
        record: _MeasurementRecord,
        sample_time_s: float,
        threshold: float,
    ) -> MeasurementUpdateResult:
        if not np.isfinite(sample_time_s) or sample_time_s < 0.0:
            raise ValueError("measurement sample_time_s must be finite and nonnegative")
        processing_time_s = self.current_time_s
        history_index = self._history_index(sample_time_s)
        if history_index is None:
            reason = (
                "outside fixed-lag history"
                if sample_time_s < self.history_start_time_s
                else "sample time is not aligned with an IMU epoch"
            )
            return MeasurementUpdateResult(
                record.sensor_name,
                sample_time_s,
                processing_time_s,
                False,
                float("nan"),
                threshold,
                0,
                self._health[record.sensor_name].health,
                reason,
            )

        current_state = self.state.copy()
        current_covariance = self.covariance.copy()
        current_transition = self.last_transition.copy()
        entry = self._history[history_index]
        self.state = entry.state.copy()
        self.covariance = entry.covariance.copy()
        innovation, measurement_matrix = self._innovation_and_matrix(record)
        nis = self._normalised_innovation_squared(
            innovation,
            measurement_matrix,
            record.covariance,
        )
        accepted = nis <= threshold
        health = self._record_health(record.sensor_name, accepted)
        if not accepted:
            self.state = current_state
            self.covariance = current_covariance
            self.last_transition = current_transition
            return MeasurementUpdateResult(
                record.sensor_name,
                sample_time_s,
                processing_time_s,
                False,
                nis,
                threshold,
                0,
                health,
                "NIS gate rejected measurement",
            )

        self._apply_linear_update(innovation, measurement_matrix, record.covariance)
        entry.measurements.append(record)
        entry.state = self.state.copy()
        entry.covariance = self.covariance.copy()
        replayed_step_count = 0
        for replay_index in range(history_index + 1, len(self._history)):
            replay_entry = self._history[replay_index]
            if replay_entry.increment_from_previous is None:
                raise RuntimeError("fixed-lag history has a missing IMU increment")
            self._predict_core(replay_entry.increment_from_previous)
            for retained_record in replay_entry.measurements:
                retained_innovation, retained_matrix = self._innovation_and_matrix(retained_record)
                self._apply_linear_update(
                    retained_innovation,
                    retained_matrix,
                    retained_record.covariance,
                )
            replay_entry.state = self.state.copy()
            replay_entry.covariance = self.covariance.copy()
            replayed_step_count += 1
        self.current_time_s = processing_time_s
        return MeasurementUpdateResult(
            record.sensor_name,
            sample_time_s,
            processing_time_s,
            True,
            nis,
            threshold,
            replayed_step_count,
            health,
            "accepted and replayed" if replayed_step_count else "accepted at current epoch",
        )

    def _history_index(self, sample_time_s: float) -> int | None:
        tolerance_s = 1.0e-8 * max(1.0, abs(sample_time_s))
        for index, entry in enumerate(self._history):
            if abs(entry.time_s - sample_time_s) <= tolerance_s:
                return index
        return None

    def _innovation_and_matrix(
        self,
        record: _MeasurementRecord,
    ) -> tuple[FloatArray, FloatArray]:
        if record.sensor_name == "gnss":
            position_ned_m = ecef_position_to_ned(
                geodetic_to_ecef(self.state.geodetic, self.planet.ellipsoid),
                self.position_origin,
                self.planet.ellipsoid,
            )
            predicted = np.concatenate((position_ned_m, self.state.velocity_ned_mps))
            matrix = np.zeros((6, 15), dtype=np.float64)
            matrix[0:3, 0:3] = np.eye(3)
            matrix[3:6, 3:6] = np.eye(3)
            return record.measurement - predicted, matrix
        if record.sensor_name == "barometer":
            matrix = np.zeros((1, 15), dtype=np.float64)
            matrix[0, 2] = -1.0
            return record.measurement - np.array([self.state.geodetic.altitude_m]), matrix
        raise RuntimeError(f"unsupported retained sensor record: {record.sensor_name}")

    @staticmethod
    def _measurement_covariance(value: npt.ArrayLike, dimension: int) -> FloatArray:
        covariance = np.asarray(value, dtype=np.float64)
        if covariance.shape != (dimension, dimension):
            raise ValueError(f"measurement covariance must have shape ({dimension}, {dimension})")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, atol=1.0e-12
        ):
            raise ValueError("measurement covariance must be finite and symmetric")
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("measurement covariance must be positive definite")
        return covariance.copy()

    def _normalised_innovation_squared(
        self,
        innovation: FloatArray,
        measurement_matrix: FloatArray,
        measurement_covariance: FloatArray,
    ) -> float:
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_covariance
        )
        return float(innovation @ np.linalg.solve(innovation_covariance, innovation))

    def _apply_linear_update(
        self,
        innovation: FloatArray,
        measurement_matrix: FloatArray,
        measurement_covariance: FloatArray,
    ) -> None:
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T + measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance,
            measurement_matrix @ self.covariance,
        ).T
        error_state = gain @ innovation
        identity = np.eye(15)
        residual = identity - gain @ measurement_matrix
        self.covariance = (
            residual @ self.covariance @ residual.T + gain @ measurement_covariance @ gain.T
        )
        self.state.geodetic = displace_geodetic_ned(
            self.state.geodetic,
            error_state[0:3],
            self.planet,
        )
        self.state.velocity_ned_mps += error_state[3:6]
        attitude_error = error_state[6:9]
        self.state.quaternion_nb = normalize_quaternion(
            quaternion_multiply(
                self.state.quaternion_nb,
                rotation_vector_to_quaternion(attitude_error),
            )
        )
        self.state.gyro_bias_body_radps += error_state[9:12]
        self.state.accelerometer_bias_body_mps2 += error_state[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] -= 0.5 * skew_symmetric(attitude_error)
        self.covariance = reset @ self.covariance @ reset.T
        self._stabilise_covariance()

    def _record_health(self, sensor_name: str, accepted: bool) -> SensorHealth:
        counter = self._health[sensor_name]
        if accepted:
            counter.accepted_count += 1
            counter.consecutive_rejections = 0
            counter.health = "healthy"
        else:
            counter.rejected_count += 1
            counter.consecutive_rejections += 1
            if counter.consecutive_rejections >= self.gate_configuration.failed_after_rejections:
                counter.health = "failed"
            elif (
                counter.consecutive_rejections >= self.gate_configuration.degraded_after_rejections
            ):
                counter.health = "degraded"
        return counter.health

    def _stabilise_covariance(self) -> None:
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(self.covariance)))
        if minimum_eigenvalue < -1.0e-9:
            raise FloatingPointError("delayed ESKF covariance lost positive semidefiniteness")
