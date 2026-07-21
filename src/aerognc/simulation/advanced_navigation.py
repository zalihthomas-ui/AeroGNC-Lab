"""Rotating strapdown, delayed aiding, fault, and consistency simulation workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

from aerognc.configuration.advanced_navigation_loader import AdvancedNavigationConfiguration
from aerognc.gnc.delayed_error_state_ekf import (
    DelayedRotatingNavigationESKF,
    MeasurementUpdateResult,
)
from aerognc.gnc.strapdown_ins import (
    ImuIncrement,
    RotatingNavigationState,
    compensate_two_sample_imu,
    displace_geodetic_ned,
    propagate_rotating_strapdown,
)
from aerognc.mathematics.geodesy import (
    body_rotation_rate_ned,
    ecef_position_to_ned,
    geodetic_to_ecef,
    transport_rate_ned,
)
from aerognc.mathematics.quaternion import (
    euler321_to_quaternion,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_dcm,
)
from aerognc.mathematics.vectors import FloatArray
from aerognc.vehicle.sensor_faults import SensorFaultInjector
from aerognc.vehicle.sensors import BarometricAltimeter, CivilianGnssSensor, SensorMeasurement


@dataclass(frozen=True, slots=True)
class AidingUpdateLog:
    """One delayed measurement and the associated integrity decision."""

    sensor_name: str
    sample_time_s: float
    processing_time_s: float
    measurement: FloatArray
    accepted: bool
    nis: float
    threshold: float
    replayed_step_count: int
    health: str


@dataclass(frozen=True, slots=True)
class AdvancedNavigationResult:
    """Truth, estimate, covariance, integrity, and observability evidence."""

    seed: int
    time_s: FloatArray
    true_position_ned_m: FloatArray
    estimated_position_ned_m: FloatArray
    true_velocity_ned_mps: FloatArray
    estimated_velocity_ned_mps: FloatArray
    attitude_error_rad: FloatArray
    position_sigma_m: FloatArray
    velocity_sigma_mps: FloatArray
    attitude_sigma_rad: FloatArray
    true_gyro_bias_radps: FloatArray
    estimated_gyro_bias_radps: FloatArray
    true_accelerometer_bias_mps2: FloatArray
    estimated_accelerometer_bias_mps2: FloatArray
    nees_15: FloatArray
    aiding_updates: tuple[AidingUpdateLog, ...]
    observability_singular_values: FloatArray
    observability_rank: int
    uncompensated_position_error_m: float
    uncompensated_attitude_error_deg: float

    @property
    def position_error_m(self) -> FloatArray:
        """Estimated minus true local-NED position [m]."""
        return self.estimated_position_ned_m - self.true_position_ned_m

    @property
    def velocity_error_mps(self) -> FloatArray:
        """Estimated minus true NED velocity [m/s]."""
        return self.estimated_velocity_ned_mps - self.true_velocity_ned_mps

    @property
    def position_rms_m(self) -> float:
        """RMS three-dimensional position-error magnitude [m]."""
        return float(np.sqrt(np.mean(np.sum(self.position_error_m**2, axis=1))))

    @property
    def velocity_rms_mps(self) -> float:
        """RMS three-dimensional velocity-error magnitude [m/s]."""
        return float(np.sqrt(np.mean(np.sum(self.velocity_error_mps**2, axis=1))))

    @property
    def attitude_rms_deg(self) -> float:
        """RMS small-angle attitude-error magnitude [deg]."""
        return float(np.rad2deg(np.sqrt(np.mean(np.sum(self.attitude_error_rad**2, axis=1)))))

    @property
    def maximum_replayed_step_count(self) -> int:
        """Largest number of IMU steps replayed by one aiding update."""
        return max((update.replayed_step_count for update in self.aiding_updates), default=0)


@dataclass(frozen=True, slots=True)
class NavigationConsistencyResult:
    """Seeded ensemble NEES/NIS confidence evidence."""

    run_count: int
    confidence: float
    time_s: FloatArray
    mean_nees_15: FloatArray
    nees_lower: float
    nees_upper: float
    nees_inside_fraction: float
    mean_gnss_nis: float
    gnss_nis_lower: float
    gnss_nis_upper: float
    mean_barometer_nis: float
    barometer_nis_lower: float
    barometer_nis_upper: float
    seeds: tuple[int, ...]


def _local_position_ned(
    state: RotatingNavigationState,
    configuration: AdvancedNavigationConfiguration,
) -> FloatArray:
    return ecef_position_to_ned(
        geodetic_to_ecef(state.geodetic, configuration.rotating_ascent.planet.ellipsoid),
        configuration.rotating_ascent.launch_site.geodetic,
        configuration.rotating_ascent.planet.ellipsoid,
    )


def _attitude_error_vector_rad(
    estimate_quaternion_nb: FloatArray,
    truth_quaternion_nb: FloatArray,
) -> FloatArray:
    error_quaternion = normalize_quaternion(
        quaternion_multiply(quaternion_inverse(estimate_quaternion_nb), truth_quaternion_nb)
    )
    if error_quaternion[0] < 0.0:
        error_quaternion = -error_quaternion
    vector_norm = float(np.linalg.norm(error_quaternion[1:]))
    if vector_norm <= 1.0e-12:
        return 2.0 * error_quaternion[1:]
    angle_rad = float(2.0 * np.arctan2(vector_norm, error_quaternion[0]))
    return np.asarray(angle_rad * error_quaternion[1:] / vector_norm, dtype=np.float64)


def _motion_inputs(
    time_s: float,
    state: RotatingNavigationState,
    configuration: AdvancedNavigationConfiguration,
) -> tuple[FloatArray, FloatArray]:
    truth = configuration.truth
    phase_rad = 2.0 * np.pi * truth.coning_frequency_hz * time_s
    amplitudes = np.asarray(truth.coning_rate_amplitude_body_radps)
    relative_rate_body_radps = amplitudes * np.array(
        [np.sin(phase_rad), np.cos(phase_rad), np.sin(phase_rad + 0.25 * np.pi)]
    )
    planet = configuration.rotating_ascent.planet
    omega_in_n = body_rotation_rate_ned(
        state.geodetic.latitude_rad,
        planet.rotation_rate_radps,
    ) + transport_rate_ned(state.geodetic, state.velocity_ned_mps, planet.ellipsoid)
    dcm_bn = quaternion_to_dcm(state.quaternion_nb).T
    angular_rate_body_radps = relative_rate_body_radps + dcm_bn @ omega_in_n
    base_force = (
        truth.powered_specific_force_body_mps2
        if time_s < truth.powered_duration_s
        else truth.coast_specific_force_body_mps2
    )
    sculling_amplitude = np.asarray(truth.sculling_force_amplitude_body_mps2)
    sculling_force = sculling_amplitude * np.array(
        [np.sin(phase_rad), np.cos(phase_rad), np.sin(phase_rad + 0.5 * np.pi)]
    )
    return angular_rate_body_radps, np.asarray(base_force) + sculling_force


def _measurement_log(
    measurement: SensorMeasurement,
    value: FloatArray,
    update: MeasurementUpdateResult,
) -> AidingUpdateLog:
    return AidingUpdateLog(
        sensor_name=update.sensor_name,
        sample_time_s=measurement.sample_time_s,
        processing_time_s=update.processing_time_s,
        measurement=value.copy(),
        accepted=update.accepted,
        nis=update.nis,
        threshold=update.threshold,
        replayed_step_count=update.replayed_step_count,
        health=update.health,
    )


def _record_state(
    index: int,
    truth_state: RotatingNavigationState,
    navigation_filter: DelayedRotatingNavigationESKF,
    true_gyro_bias: FloatArray,
    true_accelerometer_bias: FloatArray,
    configuration: AdvancedNavigationConfiguration,
    arrays: dict[str, FloatArray],
) -> None:
    true_position = _local_position_ned(truth_state, configuration)
    estimated_position = _local_position_ned(navigation_filter.state, configuration)
    attitude_error = _attitude_error_vector_rad(
        navigation_filter.state.quaternion_nb,
        truth_state.quaternion_nb,
    )
    error_state = np.concatenate(
        (
            true_position - estimated_position,
            truth_state.velocity_ned_mps - navigation_filter.state.velocity_ned_mps,
            attitude_error,
            true_gyro_bias - navigation_filter.state.gyro_bias_body_radps,
            true_accelerometer_bias - navigation_filter.state.accelerometer_bias_body_mps2,
        )
    )
    arrays["true_position"][index] = true_position
    arrays["estimated_position"][index] = estimated_position
    arrays["true_velocity"][index] = truth_state.velocity_ned_mps
    arrays["estimated_velocity"][index] = navigation_filter.state.velocity_ned_mps
    arrays["attitude_error"][index] = attitude_error
    standard_deviation = navigation_filter.standard_deviation()
    arrays["position_sigma"][index] = standard_deviation[0:3]
    arrays["velocity_sigma"][index] = standard_deviation[3:6]
    arrays["attitude_sigma"][index] = standard_deviation[6:9]
    arrays["true_gyro_bias"][index] = true_gyro_bias
    arrays["estimated_gyro_bias"][index] = navigation_filter.state.gyro_bias_body_radps
    arrays["true_accelerometer_bias"][index] = true_accelerometer_bias
    arrays["estimated_accelerometer_bias"][index] = (
        navigation_filter.state.accelerometer_bias_body_mps2
    )
    arrays["nees"][index] = error_state @ np.linalg.solve(
        navigation_filter.covariance,
        error_state,
    )


def simulate_advanced_navigation(
    configuration: AdvancedNavigationConfiguration,
    *,
    seed: int | None = None,
    inject_faults: bool = True,
) -> AdvancedNavigationResult:
    """Run one synthetic rotating-navigation realization."""
    active_seed = configuration.random_seed if seed is None else int(seed)
    generator = np.random.default_rng(active_seed)
    planet = configuration.rotating_ascent.planet
    origin = configuration.rotating_ascent.launch_site.geodetic
    truth_quaternion = euler321_to_quaternion(*np.deg2rad(configuration.truth.initial_euler321_deg))
    truth_state = RotatingNavigationState(
        origin,
        configuration.truth.initial_velocity_ned_mps,
        truth_quaternion,
    )
    uncompensated_state = truth_state.copy()
    filter_configuration = configuration.navigation_filter
    estimated_geodetic = displace_geodetic_ned(
        origin,
        filter_configuration.initial_position_error_ned_m,
        planet,
    )
    attitude_error_quaternion = euler321_to_quaternion(
        *np.deg2rad(filter_configuration.initial_attitude_error_euler321_deg)
    )
    estimated_quaternion = normalize_quaternion(
        quaternion_multiply(truth_quaternion, attitude_error_quaternion)
    )
    estimated_state = RotatingNavigationState(
        estimated_geodetic,
        np.asarray(configuration.truth.initial_velocity_ned_mps)
        + np.asarray(filter_configuration.initial_velocity_error_ned_mps),
        estimated_quaternion,
        filter_configuration.initial_gyro_bias_estimate_radps,
        filter_configuration.initial_accelerometer_bias_estimate_mps2,
    )
    initial_std = np.asarray(filter_configuration.initial_standard_deviation)
    navigation_filter = DelayedRotatingNavigationESKF(
        estimated_state,
        np.diag(initial_std**2),
        filter_configuration.tuning,
        planet,
        origin,
        fixed_lag_s=filter_configuration.fixed_lag_s,
        gate_configuration=filter_configuration.gate,
    )
    gnss = CivilianGnssSensor(configuration.gnss, seed=active_seed + 101)
    barometer = BarometricAltimeter(configuration.barometer, seed=active_seed + 202)
    fault_injector = SensorFaultInjector(configuration.faults if inject_faults else ())
    true_gyro_bias = np.asarray(configuration.imu.initial_gyro_bias_body_radps).copy()
    true_accelerometer_bias = np.asarray(
        configuration.imu.initial_accelerometer_bias_body_mps2
    ).copy()
    gyro_noise = np.asarray(configuration.imu.gyro_noise_density_radps_per_sqrt_hz)
    accelerometer_noise = np.asarray(configuration.imu.accelerometer_noise_density_mps2_per_sqrt_hz)
    gyro_bias_walk = np.asarray(configuration.imu.gyro_bias_random_walk_radps2_per_sqrt_hz)
    accelerometer_bias_walk = np.asarray(
        configuration.imu.accelerometer_bias_random_walk_mps3_per_sqrt_hz
    )

    step_s = configuration.step_s
    step_count = round(configuration.duration_s / step_s)
    time_s = np.arange(step_count + 1, dtype=np.float64) * step_s
    vector_arrays = {
        name: np.empty((step_count + 1, 3), dtype=np.float64)
        for name in (
            "true_position",
            "estimated_position",
            "true_velocity",
            "estimated_velocity",
            "attitude_error",
            "position_sigma",
            "velocity_sigma",
            "attitude_sigma",
            "true_gyro_bias",
            "estimated_gyro_bias",
            "true_accelerometer_bias",
            "estimated_accelerometer_bias",
        )
    }
    arrays = {**vector_arrays, "nees": np.empty(step_count + 1, dtype=np.float64)}
    aiding_updates: list[AidingUpdateLog] = []
    transition_from_initial = np.eye(15)
    observability_gramian = np.zeros((15, 15), dtype=np.float64)
    gnss_matrix = np.zeros((6, 15), dtype=np.float64)
    gnss_matrix[0:3, 0:3] = np.eye(3)
    gnss_matrix[3:6, 3:6] = np.eye(3)
    barometer_matrix = np.zeros((1, 15), dtype=np.float64)
    barometer_matrix[0, 2] = -1.0
    gnss_covariance = np.diag(configuration.gnss.noise_std**2)
    barometer_covariance = np.diag(configuration.barometer.noise_std**2)

    _record_state(
        0,
        truth_state,
        navigation_filter,
        true_gyro_bias,
        true_accelerometer_bias,
        configuration,
        arrays,
    )
    initial_gnss_truth = np.concatenate(
        (_local_position_ned(truth_state, configuration), truth_state.velocity_ned_mps)
    )
    gnss.measure(0.0, initial_gnss_truth)
    barometer.measure(0.0, [truth_state.geodetic.altitude_m])

    half_step_s = 0.5 * step_s
    for step_index in range(step_count):
        start_time_s = time_s[step_index]
        midpoint_1_s = start_time_s + 0.25 * step_s
        midpoint_2_s = start_time_s + 0.75 * step_s
        end_time_s = time_s[step_index + 1]
        angular_rate_1, specific_force_1 = _motion_inputs(
            midpoint_1_s,
            truth_state,
            configuration,
        )
        angular_rate_2, specific_force_2 = _motion_inputs(
            midpoint_2_s,
            truth_state,
            configuration,
        )
        ideal_first = ImuIncrement(
            start_time_s,
            start_time_s + half_step_s,
            angular_rate_1 * half_step_s,
            specific_force_1 * half_step_s,
        )
        ideal_second = ImuIncrement(
            start_time_s + half_step_s,
            end_time_s,
            angular_rate_2 * half_step_s,
            specific_force_2 * half_step_s,
        )
        corrected_truth_increment = compensate_two_sample_imu(ideal_first, ideal_second)
        propagate_rotating_strapdown(truth_state, corrected_truth_increment, planet)
        naive_increment = ImuIncrement(
            start_time_s,
            end_time_s,
            ideal_first.delta_angle_body_rad + ideal_second.delta_angle_body_rad,
            ideal_first.delta_velocity_body_mps + ideal_second.delta_velocity_body_mps,
        )
        propagate_rotating_strapdown(uncompensated_state, naive_increment, planet)

        measured_first = ImuIncrement(
            start_time_s,
            start_time_s + half_step_s,
            ideal_first.delta_angle_body_rad
            + true_gyro_bias * half_step_s
            + gyro_noise * np.sqrt(half_step_s) * generator.standard_normal(3),
            ideal_first.delta_velocity_body_mps
            + true_accelerometer_bias * half_step_s
            + accelerometer_noise * np.sqrt(half_step_s) * generator.standard_normal(3),
        )
        measured_second = ImuIncrement(
            start_time_s + half_step_s,
            end_time_s,
            ideal_second.delta_angle_body_rad
            + true_gyro_bias * half_step_s
            + gyro_noise * np.sqrt(half_step_s) * generator.standard_normal(3),
            ideal_second.delta_velocity_body_mps
            + true_accelerometer_bias * half_step_s
            + accelerometer_noise * np.sqrt(half_step_s) * generator.standard_normal(3),
        )
        navigation_filter.predict(compensate_two_sample_imu(measured_first, measured_second))
        transition_from_initial = navigation_filter.last_transition @ transition_from_initial
        true_gyro_bias += gyro_bias_walk * np.sqrt(step_s) * generator.standard_normal(3)
        true_accelerometer_bias += (
            accelerometer_bias_walk * np.sqrt(step_s) * generator.standard_normal(3)
        )

        gnss_period_steps = round(
            configuration.imu_sample_rate_hz / configuration.gnss.sample_rate_hz
        )
        barometer_period_steps = round(
            configuration.imu_sample_rate_hz / configuration.barometer.sample_rate_hz
        )
        if (step_index + 1) % max(gnss_period_steps, 1) == 0:
            weighted = np.linalg.solve(gnss_covariance, gnss_matrix)
            observability_gramian += (
                transition_from_initial.T @ gnss_matrix.T @ weighted @ transition_from_initial
            )
        if (step_index + 1) % max(barometer_period_steps, 1) == 0:
            weighted = np.linalg.solve(barometer_covariance, barometer_matrix)
            observability_gramian += (
                transition_from_initial.T @ barometer_matrix.T @ weighted @ transition_from_initial
            )

        gnss_truth = np.concatenate(
            (_local_position_ned(truth_state, configuration), truth_state.velocity_ned_mps)
        )
        gnss_measurement = gnss.measure(end_time_s, gnss_truth)
        if gnss_measurement is not None:
            faulted = fault_injector.apply(
                "gnss",
                gnss_measurement.sample_time_s,
                gnss_measurement.value,
            )
            if faulted is not None:
                update = navigation_filter.update_gnss(
                    faulted[0:3],
                    faulted[3:6],
                    gnss_covariance,
                    sample_time_s=gnss_measurement.sample_time_s,
                )
                aiding_updates.append(_measurement_log(gnss_measurement, faulted, update))
        barometer_measurement = barometer.measure(
            end_time_s,
            [truth_state.geodetic.altitude_m],
        )
        if barometer_measurement is not None:
            faulted = fault_injector.apply(
                "barometer",
                barometer_measurement.sample_time_s,
                barometer_measurement.value,
            )
            if faulted is not None:
                update = navigation_filter.update_barometric_altitude(
                    float(faulted[0]),
                    float(barometer_covariance[0, 0]),
                    sample_time_s=barometer_measurement.sample_time_s,
                )
                aiding_updates.append(_measurement_log(barometer_measurement, faulted, update))
        _record_state(
            step_index + 1,
            truth_state,
            navigation_filter,
            true_gyro_bias,
            true_accelerometer_bias,
            configuration,
            arrays,
        )

    diagonal = np.maximum(np.diag(observability_gramian), np.finfo(np.float64).tiny)
    scaling = np.diag(1.0 / np.sqrt(diagonal))
    normalised_gramian = scaling @ observability_gramian @ scaling
    singular_values = np.linalg.svd(normalised_gramian, compute_uv=False)
    observability_rank = int(np.count_nonzero(singular_values > singular_values[0] * 1.0e-8))
    uncompensated_position_error = np.linalg.norm(
        _local_position_ned(uncompensated_state, configuration)
        - _local_position_ned(truth_state, configuration)
    )
    uncompensated_attitude_error = np.rad2deg(
        np.linalg.norm(
            _attitude_error_vector_rad(
                uncompensated_state.quaternion_nb,
                truth_state.quaternion_nb,
            )
        )
    )
    return AdvancedNavigationResult(
        seed=active_seed,
        time_s=time_s,
        true_position_ned_m=arrays["true_position"],
        estimated_position_ned_m=arrays["estimated_position"],
        true_velocity_ned_mps=arrays["true_velocity"],
        estimated_velocity_ned_mps=arrays["estimated_velocity"],
        attitude_error_rad=arrays["attitude_error"],
        position_sigma_m=arrays["position_sigma"],
        velocity_sigma_mps=arrays["velocity_sigma"],
        attitude_sigma_rad=arrays["attitude_sigma"],
        true_gyro_bias_radps=arrays["true_gyro_bias"],
        estimated_gyro_bias_radps=arrays["estimated_gyro_bias"],
        true_accelerometer_bias_mps2=arrays["true_accelerometer_bias"],
        estimated_accelerometer_bias_mps2=arrays["estimated_accelerometer_bias"],
        nees_15=arrays["nees"],
        aiding_updates=tuple(aiding_updates),
        observability_singular_values=singular_values,
        observability_rank=observability_rank,
        uncompensated_position_error_m=float(uncompensated_position_error),
        uncompensated_attitude_error_deg=float(uncompensated_attitude_error),
    )


def run_navigation_consistency(
    configuration: AdvancedNavigationConfiguration,
    *,
    run_count: int | None = None,
) -> NavigationConsistencyResult:
    """Run a deterministic fault-free ensemble and calculate NEES/NIS bounds."""
    count = configuration.consistency_runs if run_count is None else int(run_count)
    if count <= 0:
        raise ValueError("navigation consistency run_count must be positive")
    seeds = tuple(configuration.random_seed + 10_000 + index for index in range(count))
    results = tuple(
        simulate_advanced_navigation(configuration, seed=seed, inject_faults=False)
        for seed in seeds
    )
    nees = np.vstack([result.nees_15 for result in results])
    mean_nees = np.mean(nees, axis=0)
    alpha = 1.0 - configuration.consistency_confidence
    nees_lower = float(chi2.ppf(0.5 * alpha, 15 * count) / count)
    nees_upper = float(chi2.ppf(1.0 - 0.5 * alpha, 15 * count) / count)
    inside_fraction = float(np.mean((mean_nees >= nees_lower) & (mean_nees <= nees_upper)))

    gnss_nis = np.array(
        [
            update.nis
            for result in results
            for update in result.aiding_updates
            if update.sensor_name == "gnss" and np.isfinite(update.nis)
        ]
    )
    barometer_nis = np.array(
        [
            update.nis
            for result in results
            for update in result.aiding_updates
            if update.sensor_name == "barometer" and np.isfinite(update.nis)
        ]
    )

    def nis_statistics(values: FloatArray, dimension: int) -> tuple[float, float, float]:
        if values.size == 0:
            raise RuntimeError("consistency ensemble produced no aiding measurements")
        degrees_of_freedom = dimension * values.size
        return (
            float(np.mean(values)),
            float(chi2.ppf(0.5 * alpha, degrees_of_freedom) / values.size),
            float(chi2.ppf(1.0 - 0.5 * alpha, degrees_of_freedom) / values.size),
        )

    gnss_mean, gnss_lower, gnss_upper = nis_statistics(gnss_nis, 6)
    barometer_mean, barometer_lower, barometer_upper = nis_statistics(barometer_nis, 1)
    return NavigationConsistencyResult(
        run_count=count,
        confidence=configuration.consistency_confidence,
        time_s=results[0].time_s,
        mean_nees_15=mean_nees,
        nees_lower=nees_lower,
        nees_upper=nees_upper,
        nees_inside_fraction=inside_fraction,
        mean_gnss_nis=gnss_mean,
        gnss_nis_lower=gnss_lower,
        gnss_nis_upper=gnss_upper,
        mean_barometer_nis=barometer_mean,
        barometer_nis_lower=barometer_lower,
        barometer_nis_upper=barometer_upper,
        seeds=seeds,
    )
