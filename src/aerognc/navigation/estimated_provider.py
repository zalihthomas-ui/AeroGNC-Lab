"""Truth-isolated sensor/ESKF navigation provider for waypoint simulation.

Simulator truth enters this module only at the synthetic sensor boundary. Guidance,
control, safety, logging, provenance, and estimator diagnostics receive the returned
``NavigationState`` and never receive an uncorrupted truth field. Quantitative truth
errors belong in the separate verification campaign module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from aerognc.environment.rotating_planet import RotatingOblatePlanet
from aerognc.gnc.delayed_error_state_ekf import (
    DelayedRotatingNavigationESKF,
    InnovationGateConfiguration,
    MeasurementUpdateResult,
)
from aerognc.gnc.error_state_ekf import ErrorStateFilterTuning
from aerognc.gnc.strapdown_ins import (
    ImuIncrement,
    RotatingNavigationState,
    displace_geodetic_ned,
    gravity_ned_mps2,
)
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    body_rotation_rate_ned,
    ecef_position_to_ned,
    geodetic_to_ecef,
    transport_rate_ned,
)
from aerognc.mathematics.local_frame import WGS84
from aerognc.mathematics.quaternion import (
    quaternion_multiply,
    quaternion_to_dcm,
    rotation_vector_to_quaternion,
)
from aerognc.navigation.providers import NavigationProvider
from aerognc.navigation.state import NavigationState
from aerognc.vehicle.sensors import (
    AccelerometerSensor,
    AirspeedSensor,
    BarometricAltimeter,
    CivilianGnssSensor,
    GyroscopeSensor,
    SensorErrorParameters,
    SensorMeasurement,
)


def _finite_tuple(
    value: tuple[float, ...],
    length: int,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class EstimatedNavigationParameters:
    """Complete deterministic sensor, filter, initialization, and health policy."""

    step_s: float
    seed: int
    gravity_mps2: float
    gyroscope: SensorErrorParameters
    accelerometer: SensorErrorParameters
    gnss: SensorErrorParameters
    barometer: SensorErrorParameters
    airspeed: SensorErrorParameters
    initial_position_error_std_m: tuple[float, float, float]
    initial_velocity_error_std_mps: tuple[float, float, float]
    initial_attitude_error_std_rad: tuple[float, float, float]
    initial_gyro_bias_estimate_radps: tuple[float, float, float]
    initial_accelerometer_bias_estimate_mps2: tuple[float, float, float]
    initial_standard_deviation: tuple[float, ...]
    filter_tuning: ErrorStateFilterTuning
    fixed_lag_s: float
    innovation_gate: InnovationGateConfiguration
    maximum_imu_age_s: float
    maximum_gnss_age_s: float
    maximum_airspeed_age_s: float
    maximum_horizontal_position_std_m: float
    maximum_vertical_position_std_m: float

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [
                self.step_s,
                self.gravity_mps2,
                self.fixed_lag_s,
                self.maximum_imu_age_s,
                self.maximum_gnss_age_s,
                self.maximum_airspeed_age_s,
                self.maximum_horizontal_position_std_m,
                self.maximum_vertical_position_std_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars <= 0.0):
            raise ValueError("estimated-navigation timing and health limits must be positive")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32 - 6
        ):
            raise ValueError("estimated-navigation seed must lie in [0, 2^32 - 6)")

        vector_fields = (
            ("initial_position_error_std_m", 3, False, True),
            ("initial_velocity_error_std_mps", 3, False, True),
            ("initial_attitude_error_std_rad", 3, False, True),
            ("initial_gyro_bias_estimate_radps", 3, False, False),
            ("initial_accelerometer_bias_estimate_mps2", 3, False, False),
            ("initial_standard_deviation", 15, True, False),
        )
        for name, length, positive, nonnegative in vector_fields:
            object.__setattr__(
                self,
                name,
                _finite_tuple(
                    getattr(self, name),
                    length,
                    name=name,
                    positive=positive,
                    nonnegative=nonnegative,
                ),
            )

        sensors = (
            ("gyroscope", self.gyroscope, 3),
            ("accelerometer", self.accelerometer, 3),
            ("gnss", self.gnss, 6),
            ("barometer", self.barometer, 1),
            ("airspeed", self.airspeed, 1),
        )
        for name, parameters, dimension in sensors:
            if parameters.dimension != dimension:
                raise ValueError(f"{name} sensor must have dimension {dimension}")
            period_steps = 1.0 / (parameters.sample_rate_hz * self.step_s)
            if period_steps < 1.0 - 1.0e-9 or not np.isclose(
                period_steps, round(period_steps), rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(f"{name} sample period must be an integer navigation step")
            delay_steps = parameters.delay_s / self.step_s
            if not np.isclose(delay_steps, round(delay_steps), rtol=0.0, atol=1.0e-9):
                raise ValueError(f"{name} delay must be an integer navigation step")
        for name, parameters in (
            ("gyroscope", self.gyroscope),
            ("accelerometer", self.accelerometer),
        ):
            if not np.isclose(parameters.sample_rate_hz * self.step_s, 1.0, atol=1.0e-9):
                raise ValueError(f"{name} sample rate must match the navigation step")
            if parameters.delay_s != 0.0:
                raise ValueError(f"{name} delay must be zero for online mechanization")
        if self.fixed_lag_s <= max(self.gnss.delay_s, self.barometer.delay_s):
            raise ValueError("fixed_lag_s must exceed GNSS and barometer latency")


class EstimatedNavigationProvider(NavigationProvider):
    """Seeded sampled-sensor provider backed by the fixed-lag rotating-NED ESKF."""

    def __init__(self, parameters: EstimatedNavigationParameters) -> None:
        self.parameters = parameters
        radius_m = WGS84.semi_major_axis_m
        self._planet = RotatingOblatePlanet(
            "Waypoint synthetic navigation reference",
            WGS84,
            parameters.gravity_mps2 * radius_m**2,
            0.0,
            0.0,
        )
        self._origin = GeodeticPosition(0.0, 0.0, 0.0)
        self.reset()

    def reset(self) -> None:
        """Restore deterministic initialization, sensor sequences, and filter state."""
        seed = self.parameters.seed
        self._rng = np.random.default_rng(seed)
        self._gyroscope = GyroscopeSensor(self.parameters.gyroscope, seed=seed + 1)
        self._accelerometer = AccelerometerSensor(self.parameters.accelerometer, seed=seed + 2)
        self._gnss = CivilianGnssSensor(self.parameters.gnss, seed=seed + 3)
        self._barometer = BarometricAltimeter(self.parameters.barometer, seed=seed + 4)
        self._airspeed = AirspeedSensor(self.parameters.airspeed, seed=seed + 5)
        self._filter: DelayedRotatingNavigationESKF | None = None
        self._previous_truth: NavigationState | None = None
        self._step_index = 0
        self._time_s = 0.0
        self._last_gyro: SensorMeasurement | None = None
        self._last_accelerometer: SensorMeasurement | None = None
        self._last_airspeed: SensorMeasurement | None = None
        self._last_gnss_acceptance_time_s = 0.0
        self._last_updates: dict[str, MeasurementUpdateResult] = {}
        self._maximum_horizontal_position_std_m = 0.0
        self._maximum_vertical_position_std_m = 0.0
        self._maximum_velocity_std_mps = 0.0
        self._maximum_covariance_trace = 0.0
        self._maximum_imu_age_s = 0.0
        self._maximum_gnss_age_s = 0.0
        self._maximum_airspeed_age_s = 0.0
        self._imu_held_step_count = 0

    def update(self, truth: NavigationState, dt_s: float) -> NavigationState:
        """Advance sensors/filter and return only the corrupted, estimated state."""
        if not np.isfinite(dt_s) or not np.isclose(
            dt_s, self.parameters.step_s, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("estimated-navigation dt_s must match its configured step")
        if self._filter is None:
            self._initialize(truth)
            return self._estimated_state()

        previous = self._previous_truth
        if previous is None:  # pragma: no cover - initialization invariant
            raise RuntimeError("estimated-navigation provider lost its prior truth sample")
        self._step_index += 1
        self._time_s = self._step_index * self.parameters.step_s
        acceleration_ned_mps2 = (
            truth.velocity_ned_mps - previous.velocity_ned_mps
        ) / self.parameters.step_s
        self._sample_imu(truth, acceleration_ned_mps2)
        gyro = np.zeros(3) if self._last_gyro is None else self._last_gyro.value
        accelerometer = (
            np.zeros(3) if self._last_accelerometer is None else self._last_accelerometer.value
        )
        if (
            self._last_gyro is None
            or self._last_accelerometer is None
            or self._last_gyro.sample_time_s < self._time_s - 1.0e-10
            or self._last_accelerometer.sample_time_s < self._time_s - 1.0e-10
        ):
            self._imu_held_step_count += 1
        self._filter.predict(
            ImuIncrement(
                self._time_s - self.parameters.step_s,
                self._time_s,
                gyro * self.parameters.step_s,
                accelerometer * self.parameters.step_s,
            )
        )
        self._sample_aiding(truth)
        self._sample_airspeed(truth)
        self._previous_truth = self._copy_state(truth)
        self._update_maxima()
        return self._estimated_state()

    def provenance(self) -> Mapping[str, object]:
        """Return full sensor/filter identity without any simulator-truth values."""
        tuning = self.parameters.filter_tuning
        gate = self.parameters.innovation_gate
        return {
            **super().provenance(),
            "mode": "estimated",
            "filter": "fixed_lag_rotating_ned_15_state_eskf",
            "seed": self.parameters.seed,
            "step_s": self.parameters.step_s,
            "fixed_lag_s": self.parameters.fixed_lag_s,
            "process_noise": {
                "gyro_noise_std_radps_per_sqrt_hz": tuning.gyro_noise_std_radps_per_sqrt_hz,
                "accelerometer_noise_std_mps2_per_sqrt_hz": (
                    tuning.accelerometer_noise_std_mps2_per_sqrt_hz
                ),
                "gyro_bias_random_walk_std_radps2_per_sqrt_hz": (
                    tuning.gyro_bias_random_walk_std_radps2_per_sqrt_hz
                ),
                "accelerometer_bias_random_walk_std_mps3_per_sqrt_hz": (
                    tuning.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz
                ),
            },
            "innovation_gate": {
                "gnss_nis_threshold": gate.gnss_nis_threshold,
                "barometer_nis_threshold": gate.barometer_nis_threshold,
                "degraded_after_rejections": gate.degraded_after_rejections,
                "failed_after_rejections": gate.failed_after_rejections,
            },
            "sensors": {
                "gyroscope": self._sensor_record(self.parameters.gyroscope),
                "accelerometer": self._sensor_record(self.parameters.accelerometer),
                "gnss": self._sensor_record(self.parameters.gnss),
                "barometer": self._sensor_record(self.parameters.barometer),
                "airspeed": self._sensor_record(self.parameters.airspeed),
            },
        }

    def diagnostics(self) -> Mapping[str, object]:
        """Return covariance, latency, gate, and health evidence only."""
        navigation_filter = self._filter
        if navigation_filter is None:
            return {"initialized": False}
        standard_deviation = navigation_filter.standard_deviation()
        gnss = navigation_filter.sensor_integrity("gnss")
        barometer = navigation_filter.sensor_integrity("barometer")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(navigation_filter.covariance)))
        imu_age_s = self._imu_age_s()
        airspeed_age_s = self._airspeed_age_s()
        return {
            "initialized": True,
            "time_s": self._time_s,
            "position_standard_deviation_m": standard_deviation[0:3].tolist(),
            "velocity_standard_deviation_mps": standard_deviation[3:6].tolist(),
            "attitude_standard_deviation_rad": standard_deviation[6:9].tolist(),
            "covariance_trace": float(np.trace(navigation_filter.covariance)),
            "covariance_minimum_eigenvalue": minimum_eigenvalue,
            "maximum_horizontal_position_std_m": self._maximum_horizontal_position_std_m,
            "maximum_vertical_position_std_m": self._maximum_vertical_position_std_m,
            "maximum_velocity_std_mps": self._maximum_velocity_std_mps,
            "maximum_covariance_trace": self._maximum_covariance_trace,
            "maximum_imu_age_s": self._maximum_imu_age_s,
            "maximum_gnss_age_s": self._maximum_gnss_age_s,
            "maximum_airspeed_age_s": self._maximum_airspeed_age_s,
            "gnss_age_s": self._time_s - self._last_gnss_acceptance_time_s,
            "imu_age_s": imu_age_s if np.isfinite(imu_age_s) else None,
            "airspeed_age_s": airspeed_age_s if np.isfinite(airspeed_age_s) else None,
            "imu_held_step_count": self._imu_held_step_count,
            "gnss_integrity": {
                "accepted_count": gnss.accepted_count,
                "rejected_count": gnss.rejected_count,
                "consecutive_rejections": gnss.consecutive_rejections,
                "health": gnss.health,
            },
            "barometer_integrity": {
                "accepted_count": barometer.accepted_count,
                "rejected_count": barometer.rejected_count,
                "consecutive_rejections": barometer.consecutive_rejections,
                "health": barometer.health,
            },
            "last_updates": {
                name: {
                    "sample_time_s": result.sample_time_s,
                    "processing_time_s": result.processing_time_s,
                    "accepted": result.accepted,
                    "nis": result.nis if np.isfinite(result.nis) else None,
                    "threshold": result.threshold,
                    "replayed_step_count": result.replayed_step_count,
                    "health": result.health,
                    "reason": result.reason,
                }
                for name, result in self._last_updates.items()
            },
        }

    def _initialize(self, truth: NavigationState) -> None:
        parameters = self.parameters
        position_error = self._rng.normal(0.0, parameters.initial_position_error_std_m)
        velocity_error = self._rng.normal(0.0, parameters.initial_velocity_error_std_mps)
        attitude_error = self._rng.normal(0.0, parameters.initial_attitude_error_std_rad)
        estimated_position = truth.position_ned_m + position_error
        estimated_quaternion = quaternion_multiply(
            truth.quaternion_nb,
            rotation_vector_to_quaternion(attitude_error),
        )
        state = RotatingNavigationState(
            displace_geodetic_ned(self._origin, estimated_position, self._planet),
            truth.velocity_ned_mps + velocity_error,
            estimated_quaternion,
            parameters.initial_gyro_bias_estimate_radps,
            parameters.initial_accelerometer_bias_estimate_mps2,
        )
        covariance = np.diag(np.square(parameters.initial_standard_deviation))
        self._filter = DelayedRotatingNavigationESKF(
            state,
            covariance,
            parameters.filter_tuning,
            self._planet,
            self._origin,
            fixed_lag_s=parameters.fixed_lag_s,
            gate_configuration=parameters.innovation_gate,
        )
        self._sample_imu(truth, np.zeros(3))
        self._sample_aiding(truth)
        self._sample_airspeed(truth)
        self._previous_truth = self._copy_state(truth)
        self._update_maxima()

    def _sample_imu(self, truth: NavigationState, acceleration_ned_mps2: np.ndarray) -> None:
        geodetic = self._truth_geodetic(truth)
        dcm_nb = quaternion_to_dcm(truth.quaternion_nb)
        body_rate_ned = body_rotation_rate_ned(
            geodetic.latitude_rad, self._planet.rotation_rate_radps
        )
        transport_rate = transport_rate_ned(
            geodetic,
            truth.velocity_ned_mps,
            self._planet.ellipsoid,
        )
        inertial_rate_body = truth.angular_rate_body_radps + dcm_nb.T @ (
            body_rate_ned + transport_rate
        )
        frame_acceleration = -np.cross(
            2.0 * body_rate_ned + transport_rate,
            truth.velocity_ned_mps,
        )
        specific_force_body = dcm_nb.T @ (
            acceleration_ned_mps2 - gravity_ned_mps2(geodetic, self._planet) - frame_acceleration
        )
        gyroscope = self._gyroscope.measure(self._time_s, inertial_rate_body)
        accelerometer = self._accelerometer.measure(self._time_s, specific_force_body)
        if gyroscope is not None:
            self._last_gyro = gyroscope
        if accelerometer is not None:
            self._last_accelerometer = accelerometer

    def _sample_aiding(self, truth: NavigationState) -> None:
        navigation_filter = self._require_filter()
        geodetic = self._truth_geodetic(truth)
        local_position = ecef_position_to_ned(
            geodetic_to_ecef(geodetic, self._planet.ellipsoid),
            self._origin,
            self._planet.ellipsoid,
        )
        gnss = self._gnss.measure(
            self._time_s,
            np.concatenate((local_position, truth.velocity_ned_mps)),
        )
        if gnss is not None:
            covariance = self._measurement_covariance(self.parameters.gnss)
            result = navigation_filter.update_gnss(
                gnss.value[0:3],
                gnss.value[3:6],
                covariance,
                sample_time_s=gnss.sample_time_s,
            )
            self._last_updates["gnss"] = result
            if result.accepted:
                self._last_gnss_acceptance_time_s = self._time_s

        barometer = self._barometer.measure(self._time_s, [geodetic.altitude_m])
        if barometer is not None:
            covariance = self._measurement_covariance(self.parameters.barometer)
            result = navigation_filter.update_barometric_altitude(
                float(barometer.value[0]),
                float(covariance[0, 0]),
                sample_time_s=barometer.sample_time_s,
            )
            self._last_updates["barometer"] = result

    def _sample_airspeed(self, truth: NavigationState) -> None:
        measurement = self._airspeed.measure(self._time_s, [truth.airspeed_mps])
        if measurement is not None:
            self._last_airspeed = measurement

    def _estimated_state(self) -> NavigationState:
        navigation_filter = self._require_filter()
        position = ecef_position_to_ned(
            geodetic_to_ecef(navigation_filter.state.geodetic, self._planet.ellipsoid),
            self._origin,
            self._planet.ellipsoid,
        )
        dcm_nb = quaternion_to_dcm(navigation_filter.state.quaternion_nb)
        navigation_rate_ned = body_rotation_rate_ned(
            navigation_filter.state.geodetic.latitude_rad,
            self._planet.rotation_rate_radps,
        ) + transport_rate_ned(
            navigation_filter.state.geodetic,
            navigation_filter.state.velocity_ned_mps,
            self._planet.ellipsoid,
        )
        measured_rate = np.zeros(3) if self._last_gyro is None else self._last_gyro.value
        angular_rate_body = (
            measured_rate
            - navigation_filter.state.gyro_bias_body_radps
            - dcm_nb.T @ navigation_rate_ned
        )
        airspeed_mps = (
            0.0 if self._last_airspeed is None else max(0.0, float(self._last_airspeed.value[0]))
        )
        standard_deviation = navigation_filter.standard_deviation()
        gnss_health = navigation_filter.sensor_integrity("gnss").health
        valid = bool(
            self._imu_age_s() <= self.parameters.maximum_imu_age_s
            and self._time_s - self._last_gnss_acceptance_time_s
            <= self.parameters.maximum_gnss_age_s
            and self._airspeed_age_s() <= self.parameters.maximum_airspeed_age_s
            and max(standard_deviation[0], standard_deviation[1])
            <= self.parameters.maximum_horizontal_position_std_m
            and standard_deviation[2] <= self.parameters.maximum_vertical_position_std_m
            and gnss_health != "failed"
            and np.all(np.isfinite(navigation_filter.covariance))
        )
        return NavigationState(
            position_ned_m=position,
            velocity_ned_mps=navigation_filter.state.velocity_ned_mps,
            quaternion_nb=navigation_filter.state.quaternion_nb,
            angular_rate_body_radps=angular_rate_body,
            airspeed_mps=airspeed_mps,
            valid=valid,
        )

    def _update_maxima(self) -> None:
        navigation_filter = self._require_filter()
        standard_deviation = navigation_filter.standard_deviation()
        self._maximum_horizontal_position_std_m = max(
            self._maximum_horizontal_position_std_m,
            float(max(standard_deviation[0], standard_deviation[1])),
        )
        self._maximum_vertical_position_std_m = max(
            self._maximum_vertical_position_std_m,
            float(standard_deviation[2]),
        )
        self._maximum_velocity_std_mps = max(
            self._maximum_velocity_std_mps,
            float(np.max(standard_deviation[3:6])),
        )
        self._maximum_covariance_trace = max(
            self._maximum_covariance_trace,
            float(np.trace(navigation_filter.covariance)),
        )
        imu_age_s = self._imu_age_s()
        airspeed_age_s = self._airspeed_age_s()
        if np.isfinite(imu_age_s):
            self._maximum_imu_age_s = max(self._maximum_imu_age_s, imu_age_s)
        self._maximum_gnss_age_s = max(
            self._maximum_gnss_age_s,
            self._time_s - self._last_gnss_acceptance_time_s,
        )
        if np.isfinite(airspeed_age_s):
            self._maximum_airspeed_age_s = max(
                self._maximum_airspeed_age_s,
                airspeed_age_s,
            )

    def _imu_age_s(self) -> float:
        if self._last_gyro is None or self._last_accelerometer is None:
            return float("inf")
        return max(
            self._time_s - self._last_gyro.sample_time_s,
            self._time_s - self._last_accelerometer.sample_time_s,
        )

    def _airspeed_age_s(self) -> float:
        if self._last_airspeed is None:
            return float("inf")
        return self._time_s - self._last_airspeed.sample_time_s

    def _truth_geodetic(self, truth: NavigationState) -> GeodeticPosition:
        return displace_geodetic_ned(self._origin, truth.position_ned_m, self._planet)

    def _require_filter(self) -> DelayedRotatingNavigationESKF:
        if self._filter is None:  # pragma: no cover - call-order invariant
            raise RuntimeError("estimated-navigation filter is not initialized")
        return self._filter

    @staticmethod
    def _copy_state(state: NavigationState) -> NavigationState:
        return NavigationState(
            state.position_ned_m,
            state.velocity_ned_mps,
            state.quaternion_nb,
            state.angular_rate_body_radps,
            state.airspeed_mps,
            state.valid,
        )

    @staticmethod
    def _measurement_covariance(parameters: SensorErrorParameters) -> np.ndarray:
        variance = parameters.noise_std**2 + parameters.quantisation**2 / 12.0
        return np.diag(np.maximum(variance, 1.0e-12))

    @staticmethod
    def _sensor_record(parameters: SensorErrorParameters) -> dict[str, object]:
        return {
            "sample_rate_hz": parameters.sample_rate_hz,
            "noise_std": parameters.noise_std.tolist(),
            "constant_bias": parameters.constant_bias.tolist(),
            "bias_drift_std_per_sqrt_s": parameters.bias_drift_std_per_sqrt_s.tolist(),
            "quantisation": parameters.quantisation.tolist(),
            "delay_s": parameters.delay_s,
            "dropout_probability": parameters.dropout_probability,
            "dropout_intervals_s": [list(interval) for interval in parameters.dropout_intervals_s],
        }


__all__ = ["EstimatedNavigationParameters", "EstimatedNavigationProvider"]
