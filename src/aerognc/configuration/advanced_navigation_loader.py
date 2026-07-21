"""Strict configuration for rotating strapdown and delayed-ESKF verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from aerognc.configuration.loader import (
    ConfigurationError,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
)
from aerognc.configuration.rotating_flight_loader import (
    RotatingAscentConfiguration,
    load_rotating_ascent_configuration,
)
from aerognc.gnc.delayed_error_state_ekf import InnovationGateConfiguration
from aerognc.gnc.error_state_ekf import ErrorStateFilterTuning
from aerognc.vehicle.sensor_faults import SensorFaultEvent, SensorFaultMode
from aerognc.vehicle.sensors import SensorErrorParameters


@dataclass(frozen=True, slots=True)
class AdvancedNavigationTruthConfiguration:
    """Synthetic motion inputs used to generate deterministic navigation truth."""

    initial_velocity_ned_mps: tuple[float, float, float]
    initial_euler321_deg: tuple[float, float, float]
    powered_duration_s: float
    powered_specific_force_body_mps2: tuple[float, float, float]
    coast_specific_force_body_mps2: tuple[float, float, float]
    coning_rate_amplitude_body_radps: tuple[float, float, float]
    coning_frequency_hz: float
    sculling_force_amplitude_body_mps2: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ImuErrorConfiguration:
    """Continuous-density noise and true bias process for the synthetic IMU."""

    gyro_noise_density_radps_per_sqrt_hz: tuple[float, float, float]
    accelerometer_noise_density_mps2_per_sqrt_hz: tuple[float, float, float]
    initial_gyro_bias_body_radps: tuple[float, float, float]
    initial_accelerometer_bias_body_mps2: tuple[float, float, float]
    gyro_bias_random_walk_radps2_per_sqrt_hz: tuple[float, float, float]
    accelerometer_bias_random_walk_mps3_per_sqrt_hz: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class AdvancedNavigationFilterConfiguration:
    """Initial errors/covariance, process tuning, lag, and integrity thresholds."""

    initial_position_error_ned_m: tuple[float, float, float]
    initial_velocity_error_ned_mps: tuple[float, float, float]
    initial_attitude_error_euler321_deg: tuple[float, float, float]
    initial_gyro_bias_estimate_radps: tuple[float, float, float]
    initial_accelerometer_bias_estimate_mps2: tuple[float, float, float]
    initial_standard_deviation: tuple[float, ...]
    tuning: ErrorStateFilterTuning
    fixed_lag_s: float
    gate: InnovationGateConfiguration


@dataclass(frozen=True, slots=True)
class AdvancedNavigationConfiguration:
    """Complete reproducible advanced-navigation scenario."""

    source_path: Path
    name: str
    safety_scope: str
    rotating_ascent: RotatingAscentConfiguration
    output_directory: Path
    duration_s: float
    imu_sample_rate_hz: float
    random_seed: int
    truth: AdvancedNavigationTruthConfiguration
    imu: ImuErrorConfiguration
    gnss: SensorErrorParameters
    barometer: SensorErrorParameters
    faults: tuple[SensorFaultEvent, ...]
    navigation_filter: AdvancedNavigationFilterConfiguration
    consistency_runs: int
    consistency_confidence: float

    @property
    def step_s(self) -> float:
        """Navigation propagation interval [s]."""
        return 1.0 / self.imu_sample_rate_hz


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    return cast(tuple[float, float, float], _number_tuple(value, context, length=3))


def _positive_vector(value: object, context: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    result = _number_tuple(value, context)
    if any(item < 0.0 if allow_zero else item <= 0.0 for item in result):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigurationError(f"{context}: all values must be {qualifier}")
    return result


def _dropout_intervals(value: object, context: str) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for index, item in enumerate(_sequence(value, context)):
        pair = _number_tuple(item, f"{context}[{index}]", length=2)
        if pair[0] < 0.0 or pair[1] <= pair[0]:
            raise ConfigurationError(f"{context}[{index}]: expected 0 <= start < end")
        result.append((pair[0], pair[1]))
    return tuple(result)


def _sensor(value: object, context: str, dimension: int) -> SensorErrorParameters:
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={
            "sample_rate_hz",
            "noise_std",
            "constant_bias",
            "bias_drift_std_per_sqrt_s",
            "quantisation",
            "delay_s",
            "dropout_probability",
            "dropout_intervals_s",
        },
    )
    return SensorErrorParameters(
        sample_rate_hz=_number(data["sample_rate_hz"], f"{context}.sample_rate_hz", positive=True),
        noise_std=_number_tuple(data["noise_std"], f"{context}.noise_std", length=dimension),
        constant_bias=_number_tuple(
            data["constant_bias"], f"{context}.constant_bias", length=dimension
        ),
        bias_drift_std_per_sqrt_s=_number_tuple(
            data["bias_drift_std_per_sqrt_s"],
            f"{context}.bias_drift_std_per_sqrt_s",
            length=dimension,
        ),
        quantisation=_number_tuple(
            data["quantisation"], f"{context}.quantisation", length=dimension
        ),
        delay_s=_number(data["delay_s"], f"{context}.delay_s", nonnegative=True),
        dropout_probability=_number(
            data["dropout_probability"],
            f"{context}.dropout_probability",
            nonnegative=True,
        ),
        dropout_intervals_s=_dropout_intervals(
            data["dropout_intervals_s"], f"{context}.dropout_intervals_s"
        ),
    )


def _initial_standard_deviation(value: object, context: str) -> tuple[float, ...]:
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={
            "position_m",
            "velocity_mps",
            "attitude_deg",
            "gyro_bias_radps",
            "accelerometer_bias_mps2",
        },
    )
    position = _positive_vector(data["position_m"], f"{context}.position_m")
    velocity = _positive_vector(data["velocity_mps"], f"{context}.velocity_mps")
    attitude_deg = _positive_vector(data["attitude_deg"], f"{context}.attitude_deg")
    gyro_bias = _positive_vector(data["gyro_bias_radps"], f"{context}.gyro_bias_radps")
    accelerometer_bias = _positive_vector(
        data["accelerometer_bias_mps2"], f"{context}.accelerometer_bias_mps2"
    )
    groups = (position, velocity, attitude_deg, gyro_bias, accelerometer_bias)
    if any(len(group) != 3 for group in groups):
        raise ConfigurationError(f"{context}: every standard-deviation group requires 3 values")
    return (*position, *velocity, *np.deg2rad(attitude_deg), *gyro_bias, *accelerometer_bias)


def load_advanced_navigation_configuration(path: str | Path) -> AdvancedNavigationConfiguration:
    """Load the public-safe rotating strapdown/delayed-ESKF demonstration."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "advanced_navigation",
        required={
            "metadata",
            "rotating_ascent_configuration",
            "simulation",
            "truth",
            "imu",
            "aiding_sensors",
            "faults",
            "filter",
            "consistency",
        },
    )
    metadata = _mapping(root["metadata"], "advanced_navigation.metadata")
    _keys(metadata, "advanced_navigation.metadata", required={"name", "safety_scope", "fictional"})
    if metadata["fictional"] is not True:
        raise ConfigurationError("advanced_navigation.metadata.fictional: must be true")
    safety_scope = _string(metadata["safety_scope"], "advanced_navigation.metadata.safety_scope")
    if "fictional" not in safety_scope.lower() or "civilian" not in safety_scope.lower():
        raise ConfigurationError(
            "advanced_navigation safety_scope must state fictional and civilian"
        )
    rotating_path = Path(
        _string(
            root["rotating_ascent_configuration"],
            "advanced_navigation.rotating_ascent_configuration",
        )
    )
    if not rotating_path.is_absolute():
        rotating_path = source_path.parent / rotating_path
    rotating_ascent = load_rotating_ascent_configuration(rotating_path)

    simulation = _mapping(root["simulation"], "advanced_navigation.simulation")
    _keys(
        simulation,
        "advanced_navigation.simulation",
        required={"output_directory", "duration_s", "imu_sample_rate_hz", "random_seed"},
    )
    duration_s = _number(
        simulation["duration_s"], "advanced_navigation.simulation.duration_s", positive=True
    )
    imu_sample_rate_hz = _number(
        simulation["imu_sample_rate_hz"],
        "advanced_navigation.simulation.imu_sample_rate_hz",
        positive=True,
    )
    if imu_sample_rate_hz < 20.0:
        raise ConfigurationError(
            "advanced_navigation.simulation.imu_sample_rate_hz: minimum is 20 Hz"
        )

    truth_data = _mapping(root["truth"], "advanced_navigation.truth")
    _keys(
        truth_data,
        "advanced_navigation.truth",
        required={
            "initial_velocity_ned_mps",
            "initial_euler321_deg",
            "powered_duration_s",
            "powered_specific_force_body_mps2",
            "coast_specific_force_body_mps2",
            "coning_rate_amplitude_body_radps",
            "coning_frequency_hz",
            "sculling_force_amplitude_body_mps2",
        },
    )
    powered_duration_s = _number(
        truth_data["powered_duration_s"],
        "advanced_navigation.truth.powered_duration_s",
        positive=True,
    )
    if powered_duration_s >= duration_s:
        raise ConfigurationError(
            "advanced_navigation.truth.powered_duration_s: must precede end time"
        )
    truth = AdvancedNavigationTruthConfiguration(
        initial_velocity_ned_mps=_vector3(
            truth_data["initial_velocity_ned_mps"],
            "advanced_navigation.truth.initial_velocity_ned_mps",
        ),
        initial_euler321_deg=_vector3(
            truth_data["initial_euler321_deg"],
            "advanced_navigation.truth.initial_euler321_deg",
        ),
        powered_duration_s=powered_duration_s,
        powered_specific_force_body_mps2=_vector3(
            truth_data["powered_specific_force_body_mps2"],
            "advanced_navigation.truth.powered_specific_force_body_mps2",
        ),
        coast_specific_force_body_mps2=_vector3(
            truth_data["coast_specific_force_body_mps2"],
            "advanced_navigation.truth.coast_specific_force_body_mps2",
        ),
        coning_rate_amplitude_body_radps=_vector3(
            truth_data["coning_rate_amplitude_body_radps"],
            "advanced_navigation.truth.coning_rate_amplitude_body_radps",
        ),
        coning_frequency_hz=_number(
            truth_data["coning_frequency_hz"],
            "advanced_navigation.truth.coning_frequency_hz",
            positive=True,
        ),
        sculling_force_amplitude_body_mps2=_vector3(
            truth_data["sculling_force_amplitude_body_mps2"],
            "advanced_navigation.truth.sculling_force_amplitude_body_mps2",
        ),
    )

    imu_data = _mapping(root["imu"], "advanced_navigation.imu")
    _keys(
        imu_data,
        "advanced_navigation.imu",
        required={
            "gyro_noise_density_radps_per_sqrt_hz",
            "accelerometer_noise_density_mps2_per_sqrt_hz",
            "initial_gyro_bias_body_radps",
            "initial_accelerometer_bias_body_mps2",
            "gyro_bias_random_walk_radps2_per_sqrt_hz",
            "accelerometer_bias_random_walk_mps3_per_sqrt_hz",
        },
    )
    imu = ImuErrorConfiguration(
        gyro_noise_density_radps_per_sqrt_hz=_vector3(
            imu_data["gyro_noise_density_radps_per_sqrt_hz"],
            "advanced_navigation.imu.gyro_noise_density_radps_per_sqrt_hz",
        ),
        accelerometer_noise_density_mps2_per_sqrt_hz=_vector3(
            imu_data["accelerometer_noise_density_mps2_per_sqrt_hz"],
            "advanced_navigation.imu.accelerometer_noise_density_mps2_per_sqrt_hz",
        ),
        initial_gyro_bias_body_radps=_vector3(
            imu_data["initial_gyro_bias_body_radps"],
            "advanced_navigation.imu.initial_gyro_bias_body_radps",
        ),
        initial_accelerometer_bias_body_mps2=_vector3(
            imu_data["initial_accelerometer_bias_body_mps2"],
            "advanced_navigation.imu.initial_accelerometer_bias_body_mps2",
        ),
        gyro_bias_random_walk_radps2_per_sqrt_hz=_vector3(
            imu_data["gyro_bias_random_walk_radps2_per_sqrt_hz"],
            "advanced_navigation.imu.gyro_bias_random_walk_radps2_per_sqrt_hz",
        ),
        accelerometer_bias_random_walk_mps3_per_sqrt_hz=_vector3(
            imu_data["accelerometer_bias_random_walk_mps3_per_sqrt_hz"],
            "advanced_navigation.imu.accelerometer_bias_random_walk_mps3_per_sqrt_hz",
        ),
    )
    nonnegative_groups = (
        imu.gyro_noise_density_radps_per_sqrt_hz,
        imu.accelerometer_noise_density_mps2_per_sqrt_hz,
        imu.gyro_bias_random_walk_radps2_per_sqrt_hz,
        imu.accelerometer_bias_random_walk_mps3_per_sqrt_hz,
    )
    if any(value < 0.0 for group in nonnegative_groups for value in group):
        raise ConfigurationError(
            "advanced_navigation.imu: noise and random walks must be nonnegative"
        )

    aiding = _mapping(root["aiding_sensors"], "advanced_navigation.aiding_sensors")
    _keys(aiding, "advanced_navigation.aiding_sensors", required={"gnss", "barometer"})
    gnss = _sensor(aiding["gnss"], "advanced_navigation.aiding_sensors.gnss", 6)
    barometer = _sensor(aiding["barometer"], "advanced_navigation.aiding_sensors.barometer", 1)

    fault_events: list[SensorFaultEvent] = []
    for index, item in enumerate(_sequence(root["faults"], "advanced_navigation.faults")):
        context = f"advanced_navigation.faults[{index}]"
        data = _mapping(item, context)
        _keys(data, context, required={"sensor", "mode", "start_time_s", "end_time_s", "value"})
        sensor_name = _string(data["sensor"], f"{context}.sensor")
        if sensor_name not in {"gnss", "barometer"}:
            raise ConfigurationError(f"{context}.sensor: expected gnss or barometer")
        mode = _string(data["mode"], f"{context}.mode")
        if mode not in {"bias_step", "spike", "stuck", "dropout"}:
            raise ConfigurationError(f"{context}.mode: unsupported fault mode")
        value = _number_tuple(data["value"], f"{context}.value")
        expected_dimension = 6 if sensor_name == "gnss" else 1
        if mode in {"bias_step", "spike"} and len(value) != expected_dimension:
            raise ConfigurationError(
                f"{context}.value: expected {expected_dimension} values for {sensor_name}"
            )
        event = SensorFaultEvent(
            sensor_name,
            cast(SensorFaultMode, mode),
            _number(data["start_time_s"], f"{context}.start_time_s", nonnegative=True),
            _number(data["end_time_s"], f"{context}.end_time_s", positive=True),
            value,
        )
        if event.end_time_s > duration_s:
            raise ConfigurationError(f"{context}.end_time_s: exceeds simulation duration")
        fault_events.append(event)

    filter_data = _mapping(root["filter"], "advanced_navigation.filter")
    _keys(
        filter_data,
        "advanced_navigation.filter",
        required={
            "initial_position_error_ned_m",
            "initial_velocity_error_ned_mps",
            "initial_attitude_error_euler321_deg",
            "initial_gyro_bias_estimate_radps",
            "initial_accelerometer_bias_estimate_mps2",
            "initial_standard_deviation",
            "process_noise",
            "fixed_lag_s",
            "innovation_gate",
        },
    )
    process = _mapping(filter_data["process_noise"], "advanced_navigation.filter.process_noise")
    _keys(
        process,
        "advanced_navigation.filter.process_noise",
        required={
            "gyro_noise_std_radps_per_sqrt_hz",
            "accelerometer_noise_std_mps2_per_sqrt_hz",
            "gyro_bias_random_walk_std_radps2_per_sqrt_hz",
            "accelerometer_bias_random_walk_std_mps3_per_sqrt_hz",
        },
    )
    gate_data = _mapping(
        filter_data["innovation_gate"],
        "advanced_navigation.filter.innovation_gate",
    )
    _keys(
        gate_data,
        "advanced_navigation.filter.innovation_gate",
        required={
            "gnss_nis_threshold",
            "barometer_nis_threshold",
            "degraded_after_rejections",
            "failed_after_rejections",
        },
    )
    fixed_lag_s = _number(
        filter_data["fixed_lag_s"], "advanced_navigation.filter.fixed_lag_s", positive=True
    )
    if fixed_lag_s <= max(gnss.delay_s, barometer.delay_s):
        raise ConfigurationError(
            "advanced_navigation.filter.fixed_lag_s: must exceed sensor delays"
        )
    navigation_filter = AdvancedNavigationFilterConfiguration(
        initial_position_error_ned_m=_vector3(
            filter_data["initial_position_error_ned_m"],
            "advanced_navigation.filter.initial_position_error_ned_m",
        ),
        initial_velocity_error_ned_mps=_vector3(
            filter_data["initial_velocity_error_ned_mps"],
            "advanced_navigation.filter.initial_velocity_error_ned_mps",
        ),
        initial_attitude_error_euler321_deg=_vector3(
            filter_data["initial_attitude_error_euler321_deg"],
            "advanced_navigation.filter.initial_attitude_error_euler321_deg",
        ),
        initial_gyro_bias_estimate_radps=_vector3(
            filter_data["initial_gyro_bias_estimate_radps"],
            "advanced_navigation.filter.initial_gyro_bias_estimate_radps",
        ),
        initial_accelerometer_bias_estimate_mps2=_vector3(
            filter_data["initial_accelerometer_bias_estimate_mps2"],
            "advanced_navigation.filter.initial_accelerometer_bias_estimate_mps2",
        ),
        initial_standard_deviation=_initial_standard_deviation(
            filter_data["initial_standard_deviation"],
            "advanced_navigation.filter.initial_standard_deviation",
        ),
        tuning=ErrorStateFilterTuning(
            _number(
                process["gyro_noise_std_radps_per_sqrt_hz"],
                "advanced_navigation.filter.process_noise.gyro_noise_std_radps_per_sqrt_hz",
                positive=True,
            ),
            _number(
                process["accelerometer_noise_std_mps2_per_sqrt_hz"],
                "advanced_navigation.filter.process_noise.accelerometer_noise_std_mps2_per_sqrt_hz",
                positive=True,
            ),
            _number(
                process["gyro_bias_random_walk_std_radps2_per_sqrt_hz"],
                "advanced_navigation.filter.process_noise.gyro_bias_random_walk_std_radps2_per_sqrt_hz",
                positive=True,
            ),
            _number(
                process["accelerometer_bias_random_walk_std_mps3_per_sqrt_hz"],
                "advanced_navigation.filter.process_noise.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz",
                positive=True,
            ),
        ),
        fixed_lag_s=fixed_lag_s,
        gate=InnovationGateConfiguration(
            gnss_nis_threshold=_number(
                gate_data["gnss_nis_threshold"],
                "advanced_navigation.filter.innovation_gate.gnss_nis_threshold",
                positive=True,
            ),
            barometer_nis_threshold=_number(
                gate_data["barometer_nis_threshold"],
                "advanced_navigation.filter.innovation_gate.barometer_nis_threshold",
                positive=True,
            ),
            degraded_after_rejections=_integer(
                gate_data["degraded_after_rejections"],
                "advanced_navigation.filter.innovation_gate.degraded_after_rejections",
                nonnegative=True,
            ),
            failed_after_rejections=_integer(
                gate_data["failed_after_rejections"],
                "advanced_navigation.filter.innovation_gate.failed_after_rejections",
                nonnegative=True,
            ),
        ),
    )

    consistency = _mapping(root["consistency"], "advanced_navigation.consistency")
    _keys(consistency, "advanced_navigation.consistency", required={"runs", "confidence"})
    confidence = _number(
        consistency["confidence"], "advanced_navigation.consistency.confidence", positive=True
    )
    if confidence >= 1.0:
        raise ConfigurationError("advanced_navigation.consistency.confidence: must be below 1")
    return AdvancedNavigationConfiguration(
        source_path=source_path,
        name=_string(metadata["name"], "advanced_navigation.metadata.name"),
        safety_scope=safety_scope,
        rotating_ascent=rotating_ascent,
        output_directory=Path(
            _string(
                simulation["output_directory"],
                "advanced_navigation.simulation.output_directory",
            )
        ),
        duration_s=duration_s,
        imu_sample_rate_hz=imu_sample_rate_hz,
        random_seed=_integer(
            simulation["random_seed"],
            "advanced_navigation.simulation.random_seed",
            nonnegative=True,
        ),
        truth=truth,
        imu=imu,
        gnss=gnss,
        barometer=barometer,
        faults=tuple(fault_events),
        navigation_filter=navigation_filter,
        consistency_runs=_integer(
            consistency["runs"], "advanced_navigation.consistency.runs", nonnegative=True
        ),
        consistency_confidence=confidence,
    )
