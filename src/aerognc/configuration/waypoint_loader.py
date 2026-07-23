"""Strict versioned configuration for the integrated waypoint GNC runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from math import radians
from pathlib import Path
from types import MappingProxyType
from typing import cast

from aerognc.configuration.aircraft_loader import (
    AircraftSandboxConfiguration,
    load_aircraft_configuration,
)
from aerognc.configuration.loader import (
    ConfigurationError,
    _boolean,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _sequence,
    _string,
)
from aerognc.gnc.delayed_error_state_ekf import InnovationGateConfiguration
from aerognc.gnc.error_state_ekf import ErrorStateFilterTuning
from aerognc.gnc.fixedwing_autopilot import AutopilotGains
from aerognc.gnc.waypoint_guidance import GuidanceGains, GuidanceMode
from aerognc.mission.safety import SafetyLimits
from aerognc.navigation.estimated_provider import (
    EstimatedNavigationParameters,
    EstimatedNavigationProvider,
)
from aerognc.navigation.providers import (
    NavigationProvider,
    NoisyStateProvider,
    PerfectStateProvider,
)
from aerognc.simulation.waypoint_backends import (
    ReducedFixedWingParams,
    VehicleBackendKind,
)
from aerognc.simulation.waypoint_mission import WaypointMissionConfig
from aerognc.vehicle.control_surfaces import SurfaceFailureMode
from aerognc.vehicle.sensors import SensorErrorParameters

WAYPOINT_CONFIGURATION_VERSION = 1


class WaypointNavigationMode(StrEnum):
    """Navigation sources supported by the simulation-only waypoint runtime."""

    PERFECT = "perfect"
    NOISY = "noisy"
    ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class WaypointNavigationConfiguration:
    """Provider selection and reproducible navigation settings."""

    mode: WaypointNavigationMode
    seed: int
    position_sigma_m: float
    velocity_sigma_mps: float
    airspeed_sigma_mps: float
    gps_dropout_window_s: tuple[float, float] | None
    estimated_parameters: EstimatedNavigationParameters | None = None

    def __post_init__(self) -> None:
        if self.mode is WaypointNavigationMode.ESTIMATED and self.estimated_parameters is None:
            raise ValueError("estimated navigation mode requires estimated parameters")
        if (
            self.mode is not WaypointNavigationMode.ESTIMATED
            and self.estimated_parameters is not None
        ):
            raise ValueError("non-estimated navigation cannot carry estimated parameters")

    def build_provider(self) -> NavigationProvider:
        """Return a fresh provider so repeated runs start from identical state."""
        if self.mode is WaypointNavigationMode.PERFECT:
            return PerfectStateProvider()
        if self.mode is WaypointNavigationMode.ESTIMATED:
            parameters = self.estimated_parameters
            if parameters is None:  # pragma: no cover - dataclass invariant
                raise RuntimeError("estimated navigation parameters are unavailable")
            return EstimatedNavigationProvider(parameters)
        return NoisyStateProvider(
            seed=self.seed,
            position_sigma_m=self.position_sigma_m,
            velocity_sigma_mps=self.velocity_sigma_mps,
            airspeed_sigma_mps=self.airspeed_sigma_mps,
            gps_dropout_window_s=self.gps_dropout_window_s,
        )


@dataclass(frozen=True, slots=True)
class WaypointRuntimeConfiguration:
    """Validated mission reference, runtime settings, and safe output policy."""

    source_path: Path
    source_sha256: str
    schema_version: int
    name: str
    safety_scope: str
    mission_path: Path
    mission_sha256: str
    output_directory: Path
    navigation: WaypointNavigationConfiguration
    mission_config: WaypointMissionConfig
    allow_real_vehicle_output: bool = False

    def build_mission_config(self) -> WaypointMissionConfig:
        """Return a run-ready configuration with a new navigation provider."""
        return replace(
            self.mission_config,
            provider=self.navigation.build_provider(),
            configuration_name=self.name,
            configuration_schema_version=self.schema_version,
            configuration_sha256=self.source_sha256,
            mission_sha256=self.mission_sha256,
        )


def _guidance_configuration(value: object) -> tuple[GuidanceMode, GuidanceGains]:
    data = _mapping(value, "waypoint.guidance")
    _keys(
        data,
        "waypoint.guidance",
        required={
            "mode",
            "vector_field_gain_per_m",
            "vector_field_max_approach_deg",
            "orbit_gain_per_m",
            "l1_distance_m",
            "altitude_error_to_climb_rate_per_s",
            "max_climb_rate_mps",
            "max_roll_feedforward_deg",
        },
    )
    mode_text = _string(data["mode"], "waypoint.guidance.mode")
    try:
        mode = GuidanceMode(mode_text)
    except ValueError as error:
        choices = ", ".join(item.value for item in GuidanceMode)
        raise ConfigurationError(f"waypoint.guidance.mode: expected one of {choices}") from error
    gains = GuidanceGains(
        vector_field_gain_per_m=_number(
            data["vector_field_gain_per_m"],
            "waypoint.guidance.vector_field_gain_per_m",
            positive=True,
        ),
        vector_field_max_approach_rad=radians(
            _number(
                data["vector_field_max_approach_deg"],
                "waypoint.guidance.vector_field_max_approach_deg",
                positive=True,
            )
        ),
        orbit_gain_per_m=_number(
            data["orbit_gain_per_m"], "waypoint.guidance.orbit_gain_per_m", positive=True
        ),
        l1_distance_m=_number(
            data["l1_distance_m"], "waypoint.guidance.l1_distance_m", positive=True
        ),
        altitude_error_to_climb_rate_per_s=_number(
            data["altitude_error_to_climb_rate_per_s"],
            "waypoint.guidance.altitude_error_to_climb_rate_per_s",
            positive=True,
        ),
        max_climb_rate_mps=_number(
            data["max_climb_rate_mps"],
            "waypoint.guidance.max_climb_rate_mps",
            positive=True,
        ),
        max_roll_feedforward_rad=radians(
            _number(
                data["max_roll_feedforward_deg"],
                "waypoint.guidance.max_roll_feedforward_deg",
                positive=True,
            )
        ),
    )
    return mode, gains


def _autopilot_configuration(value: object) -> AutopilotGains:
    data = _mapping(value, "waypoint.autopilot")
    keys = {
        "course_kp",
        "course_ki",
        "altitude_kp_rad_per_m",
        "altitude_ki",
        "airspeed_kp",
        "airspeed_ki",
        "roll_kp",
        "roll_rate_kd",
        "pitch_kp",
        "pitch_rate_kd",
        "yaw_damper_kd",
        "bank_limit_deg",
        "pitch_limit_deg",
        "integral_bank_limit_deg",
        "integral_pitch_limit_deg",
        "throttle_trim",
        "throttle_delta_limit",
        "elevator_trim",
    }
    _keys(data, "waypoint.autopilot", required=keys)

    def gain(name: str) -> float:
        return _number(data[name], f"waypoint.autopilot.{name}", nonnegative=True)

    return AutopilotGains(
        course_kp=gain("course_kp"),
        course_ki=gain("course_ki"),
        altitude_kp_rad_per_m=gain("altitude_kp_rad_per_m"),
        altitude_ki=gain("altitude_ki"),
        airspeed_kp=gain("airspeed_kp"),
        airspeed_ki=gain("airspeed_ki"),
        roll_kp=gain("roll_kp"),
        roll_rate_kd=gain("roll_rate_kd"),
        pitch_kp=gain("pitch_kp"),
        pitch_rate_kd=gain("pitch_rate_kd"),
        yaw_damper_kd=gain("yaw_damper_kd"),
        bank_limit_rad=radians(
            _number(data["bank_limit_deg"], "waypoint.autopilot.bank_limit_deg", positive=True)
        ),
        pitch_limit_rad=radians(
            _number(data["pitch_limit_deg"], "waypoint.autopilot.pitch_limit_deg", positive=True)
        ),
        integral_bank_limit_rad=radians(
            _number(
                data["integral_bank_limit_deg"],
                "waypoint.autopilot.integral_bank_limit_deg",
                positive=True,
            )
        ),
        integral_pitch_limit_rad=radians(
            _number(
                data["integral_pitch_limit_deg"],
                "waypoint.autopilot.integral_pitch_limit_deg",
                positive=True,
            )
        ),
        throttle_trim=_number(
            data["throttle_trim"], "waypoint.autopilot.throttle_trim", nonnegative=True
        ),
        throttle_delta_limit=_number(
            data["throttle_delta_limit"],
            "waypoint.autopilot.throttle_delta_limit",
            positive=True,
        ),
        elevator_trim=_number(data["elevator_trim"], "waypoint.autopilot.elevator_trim"),
    )


def _safety_configuration(value: object) -> SafetyLimits:
    data = _mapping(value, "waypoint.safety")
    _keys(
        data,
        "waypoint.safety",
        required={
            "min_airspeed_mps",
            "max_airspeed_mps",
            "max_bank_deg",
            "max_pitch_deg",
            "min_altitude_m",
            "max_altitude_m",
            "geofence_radius_m",
            "max_cross_track_m",
        },
    )
    return SafetyLimits(
        min_airspeed_mps=_number(
            data["min_airspeed_mps"], "waypoint.safety.min_airspeed_mps", positive=True
        ),
        max_airspeed_mps=_number(
            data["max_airspeed_mps"], "waypoint.safety.max_airspeed_mps", positive=True
        ),
        max_bank_rad=radians(
            _number(data["max_bank_deg"], "waypoint.safety.max_bank_deg", positive=True)
        ),
        max_pitch_rad=radians(
            _number(data["max_pitch_deg"], "waypoint.safety.max_pitch_deg", positive=True)
        ),
        min_altitude_m=_number(
            data["min_altitude_m"], "waypoint.safety.min_altitude_m", nonnegative=True
        ),
        max_altitude_m=_number(
            data["max_altitude_m"], "waypoint.safety.max_altitude_m", positive=True
        ),
        geofence_radius_m=_number(
            data["geofence_radius_m"], "waypoint.safety.geofence_radius_m", positive=True
        ),
        max_cross_track_m=_number(
            data["max_cross_track_m"], "waypoint.safety.max_cross_track_m", positive=True
        ),
    )


def _vehicle_configuration(
    value: object,
    source_directory: Path,
) -> tuple[VehicleBackendKind, ReducedFixedWingParams, AircraftSandboxConfiguration | None]:
    data = _mapping(value, "waypoint.vehicle")
    _keys(
        data,
        "waypoint.vehicle",
        required={"backend"},
        optional={"parameters", "aircraft_config"},
    )
    backend = _string(data["backend"], "waypoint.vehicle.backend")
    try:
        kind = VehicleBackendKind(backend)
    except ValueError as error:
        choices = ", ".join(item.value for item in VehicleBackendKind)
        raise ConfigurationError(f"waypoint.vehicle.backend: expected one of {choices}") from error
    if kind is VehicleBackendKind.INTERNAL_COEFFICIENT:
        _keys(data, "waypoint.vehicle", required={"backend", "aircraft_config"})
        aircraft_path = Path(_string(data["aircraft_config"], "waypoint.vehicle.aircraft_config"))
        if not aircraft_path.is_absolute():
            aircraft_path = (source_directory / aircraft_path).resolve()
        try:
            aircraft_configuration = load_aircraft_configuration(aircraft_path)
        except (OSError, ValueError) as error:
            raise ConfigurationError(f"waypoint.vehicle.aircraft_config: {error}") from error
        return kind, ReducedFixedWingParams(), aircraft_configuration

    _keys(data, "waypoint.vehicle", required={"backend", "parameters"})
    params = _mapping(data["parameters"], "waypoint.vehicle.parameters")
    _keys(
        params,
        "waypoint.vehicle.parameters",
        required={
            "roll_from_aileron",
            "roll_damping",
            "pitch_from_elevator",
            "pitch_damping",
            "yaw_from_rudder",
            "airspeed_min_mps",
            "airspeed_at_zero_throttle_mps",
            "airspeed_at_full_throttle_mps",
            "airspeed_time_constant_s",
            "climb_speed_coupling",
            "max_bank_for_turn_deg",
        },
    )

    def positive(name: str) -> float:
        return _number(params[name], f"waypoint.vehicle.parameters.{name}", positive=True)

    reduced = ReducedFixedWingParams(
        roll_from_aileron=positive("roll_from_aileron"),
        roll_damping=positive("roll_damping"),
        pitch_from_elevator=positive("pitch_from_elevator"),
        pitch_damping=positive("pitch_damping"),
        yaw_from_rudder=_number(
            params["yaw_from_rudder"],
            "waypoint.vehicle.parameters.yaw_from_rudder",
            nonnegative=True,
        ),
        airspeed_min_mps=positive("airspeed_min_mps"),
        airspeed_at_zero_throttle_mps=positive("airspeed_at_zero_throttle_mps"),
        airspeed_at_full_throttle_mps=positive("airspeed_at_full_throttle_mps"),
        airspeed_time_constant_s=positive("airspeed_time_constant_s"),
        climb_speed_coupling=_number(
            params["climb_speed_coupling"],
            "waypoint.vehicle.parameters.climb_speed_coupling",
            nonnegative=True,
        ),
        max_bank_for_turn_rad=radians(positive("max_bank_for_turn_deg")),
    )
    return kind, reduced, None


def _dropout_intervals(value: object, context: str) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for index, item in enumerate(_sequence(value, context)):
        pair = _number_tuple(item, f"{context}[{index}]", length=2)
        if pair[0] < 0.0 or pair[1] <= pair[0]:
            raise ConfigurationError(f"{context}[{index}]: expected 0 <= start < end")
        intervals.append((pair[0], pair[1]))
    return tuple(intervals)


def _estimated_sensor_configuration(
    value: object,
    context: str,
    dimension: int,
) -> SensorErrorParameters:
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
    try:
        return SensorErrorParameters(
            sample_rate_hz=_number(
                data["sample_rate_hz"], f"{context}.sample_rate_hz", positive=True
            ),
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
    except ValueError as error:
        raise ConfigurationError(f"{context}: {error}") from error


def _estimated_navigation_configuration(
    value: object,
    *,
    step_s: float,
    gravity_mps2: float,
) -> EstimatedNavigationParameters:
    context = "waypoint.navigation.estimated"
    data = _mapping(value, context)
    _keys(
        data,
        context,
        required={"seed", "initialization", "sensors", "filter", "health"},
    )
    initialization = _mapping(data["initialization"], f"{context}.initialization")
    _keys(
        initialization,
        f"{context}.initialization",
        required={
            "position_error_std_m",
            "velocity_error_std_mps",
            "attitude_error_std_deg",
            "gyro_bias_estimate_radps",
            "accelerometer_bias_estimate_mps2",
        },
    )
    sensors = _mapping(data["sensors"], f"{context}.sensors")
    _keys(
        sensors,
        f"{context}.sensors",
        required={"gyroscope", "accelerometer", "gnss", "barometer", "airspeed"},
    )
    gyroscope = _estimated_sensor_configuration(
        sensors["gyroscope"], f"{context}.sensors.gyroscope", 3
    )
    accelerometer = _estimated_sensor_configuration(
        sensors["accelerometer"], f"{context}.sensors.accelerometer", 3
    )
    gnss = _estimated_sensor_configuration(sensors["gnss"], f"{context}.sensors.gnss", 6)
    barometer = _estimated_sensor_configuration(
        sensors["barometer"], f"{context}.sensors.barometer", 1
    )
    airspeed = _estimated_sensor_configuration(
        sensors["airspeed"], f"{context}.sensors.airspeed", 1
    )

    filter_data = _mapping(data["filter"], f"{context}.filter")
    _keys(
        filter_data,
        f"{context}.filter",
        required={
            "initial_standard_deviation",
            "process_noise",
            "fixed_lag_s",
            "innovation_gate",
        },
    )
    initial_std = _mapping(
        filter_data["initial_standard_deviation"],
        f"{context}.filter.initial_standard_deviation",
    )
    _keys(
        initial_std,
        f"{context}.filter.initial_standard_deviation",
        required={
            "position_m",
            "velocity_mps",
            "attitude_deg",
            "gyro_bias_radps",
            "accelerometer_bias_mps2",
        },
    )
    process = _mapping(filter_data["process_noise"], f"{context}.filter.process_noise")
    _keys(
        process,
        f"{context}.filter.process_noise",
        required={
            "gyro_noise_std_radps_per_sqrt_hz",
            "accelerometer_noise_std_mps2_per_sqrt_hz",
            "gyro_bias_random_walk_std_radps2_per_sqrt_hz",
            "accelerometer_bias_random_walk_std_mps3_per_sqrt_hz",
        },
    )
    gate_data = _mapping(filter_data["innovation_gate"], f"{context}.filter.innovation_gate")
    _keys(
        gate_data,
        f"{context}.filter.innovation_gate",
        required={
            "gnss_nis_threshold",
            "barometer_nis_threshold",
            "degraded_after_rejections",
            "failed_after_rejections",
        },
    )
    health = _mapping(data["health"], f"{context}.health")
    _keys(
        health,
        f"{context}.health",
        required={
            "maximum_imu_age_s",
            "maximum_gnss_age_s",
            "maximum_airspeed_age_s",
            "maximum_horizontal_position_std_m",
            "maximum_vertical_position_std_m",
        },
    )

    position_std = _number_tuple(
        initial_std["position_m"],
        f"{context}.filter.initial_standard_deviation.position_m",
        length=3,
    )
    velocity_std = _number_tuple(
        initial_std["velocity_mps"],
        f"{context}.filter.initial_standard_deviation.velocity_mps",
        length=3,
    )
    attitude_std_deg = _number_tuple(
        initial_std["attitude_deg"],
        f"{context}.filter.initial_standard_deviation.attitude_deg",
        length=3,
    )
    gyro_bias_std = _number_tuple(
        initial_std["gyro_bias_radps"],
        f"{context}.filter.initial_standard_deviation.gyro_bias_radps",
        length=3,
    )
    accelerometer_bias_std = _number_tuple(
        initial_std["accelerometer_bias_mps2"],
        f"{context}.filter.initial_standard_deviation.accelerometer_bias_mps2",
        length=3,
    )
    initial_standard_deviation = (
        *position_std,
        *velocity_std,
        *(radians(item) for item in attitude_std_deg),
        *gyro_bias_std,
        *accelerometer_bias_std,
    )
    try:
        return EstimatedNavigationParameters(
            step_s=step_s,
            seed=_integer(data["seed"], f"{context}.seed", nonnegative=True),
            gravity_mps2=gravity_mps2,
            gyroscope=gyroscope,
            accelerometer=accelerometer,
            gnss=gnss,
            barometer=barometer,
            airspeed=airspeed,
            initial_position_error_std_m=cast(
                tuple[float, float, float],
                _number_tuple(
                    initialization["position_error_std_m"],
                    f"{context}.initialization.position_error_std_m",
                    length=3,
                ),
            ),
            initial_velocity_error_std_mps=cast(
                tuple[float, float, float],
                _number_tuple(
                    initialization["velocity_error_std_mps"],
                    f"{context}.initialization.velocity_error_std_mps",
                    length=3,
                ),
            ),
            initial_attitude_error_std_rad=cast(
                tuple[float, float, float],
                tuple(
                    radians(item)
                    for item in _number_tuple(
                        initialization["attitude_error_std_deg"],
                        f"{context}.initialization.attitude_error_std_deg",
                        length=3,
                    )
                ),
            ),
            initial_gyro_bias_estimate_radps=cast(
                tuple[float, float, float],
                _number_tuple(
                    initialization["gyro_bias_estimate_radps"],
                    f"{context}.initialization.gyro_bias_estimate_radps",
                    length=3,
                ),
            ),
            initial_accelerometer_bias_estimate_mps2=cast(
                tuple[float, float, float],
                _number_tuple(
                    initialization["accelerometer_bias_estimate_mps2"],
                    f"{context}.initialization.accelerometer_bias_estimate_mps2",
                    length=3,
                ),
            ),
            initial_standard_deviation=initial_standard_deviation,
            filter_tuning=ErrorStateFilterTuning(
                _number(
                    process["gyro_noise_std_radps_per_sqrt_hz"],
                    f"{context}.filter.process_noise.gyro_noise_std_radps_per_sqrt_hz",
                    positive=True,
                ),
                _number(
                    process["accelerometer_noise_std_mps2_per_sqrt_hz"],
                    f"{context}.filter.process_noise.accelerometer_noise_std_mps2_per_sqrt_hz",
                    positive=True,
                ),
                _number(
                    process["gyro_bias_random_walk_std_radps2_per_sqrt_hz"],
                    f"{context}.filter.process_noise.gyro_bias_random_walk_std_radps2_per_sqrt_hz",
                    positive=True,
                ),
                _number(
                    process["accelerometer_bias_random_walk_std_mps3_per_sqrt_hz"],
                    f"{context}.filter.process_noise.accelerometer_bias_random_walk_std_mps3_per_sqrt_hz",
                    positive=True,
                ),
            ),
            fixed_lag_s=_number(
                filter_data["fixed_lag_s"], f"{context}.filter.fixed_lag_s", positive=True
            ),
            innovation_gate=InnovationGateConfiguration(
                gnss_nis_threshold=_number(
                    gate_data["gnss_nis_threshold"],
                    f"{context}.filter.innovation_gate.gnss_nis_threshold",
                    positive=True,
                ),
                barometer_nis_threshold=_number(
                    gate_data["barometer_nis_threshold"],
                    f"{context}.filter.innovation_gate.barometer_nis_threshold",
                    positive=True,
                ),
                degraded_after_rejections=_integer(
                    gate_data["degraded_after_rejections"],
                    f"{context}.filter.innovation_gate.degraded_after_rejections",
                    nonnegative=True,
                ),
                failed_after_rejections=_integer(
                    gate_data["failed_after_rejections"],
                    f"{context}.filter.innovation_gate.failed_after_rejections",
                    nonnegative=True,
                ),
            ),
            maximum_imu_age_s=_number(
                health["maximum_imu_age_s"],
                f"{context}.health.maximum_imu_age_s",
                positive=True,
            ),
            maximum_gnss_age_s=_number(
                health["maximum_gnss_age_s"],
                f"{context}.health.maximum_gnss_age_s",
                positive=True,
            ),
            maximum_airspeed_age_s=_number(
                health["maximum_airspeed_age_s"],
                f"{context}.health.maximum_airspeed_age_s",
                positive=True,
            ),
            maximum_horizontal_position_std_m=_number(
                health["maximum_horizontal_position_std_m"],
                f"{context}.health.maximum_horizontal_position_std_m",
                positive=True,
            ),
            maximum_vertical_position_std_m=_number(
                health["maximum_vertical_position_std_m"],
                f"{context}.health.maximum_vertical_position_std_m",
                positive=True,
            ),
        )
    except ValueError as error:
        raise ConfigurationError(f"{context}: {error}") from error


def _navigation_configuration(
    value: object,
    *,
    step_s: float,
    gravity_mps2: float,
) -> WaypointNavigationConfiguration:
    data = _mapping(value, "waypoint.navigation")
    _keys(
        data,
        "waypoint.navigation",
        required={"mode"},
        optional={"noisy", "estimated"},
    )
    mode_text = _string(data["mode"], "waypoint.navigation.mode")
    try:
        mode = WaypointNavigationMode(mode_text)
    except ValueError as error:
        raise ConfigurationError(
            "waypoint.navigation.mode: expected perfect, noisy, or estimated"
        ) from error

    seed = 0
    position_sigma_m = 0.0
    velocity_sigma_mps = 0.0
    airspeed_sigma_mps = 0.0
    dropout: tuple[float, float] | None = None
    if "noisy" in data:
        noisy = _mapping(data["noisy"], "waypoint.navigation.noisy")
        _keys(
            noisy,
            "waypoint.navigation.noisy",
            required={
                "seed",
                "position_sigma_m",
                "velocity_sigma_mps",
                "airspeed_sigma_mps",
                "gps_dropout_window_s",
            },
        )
        seed = _integer(noisy["seed"], "waypoint.navigation.noisy.seed", nonnegative=True)
        if seed >= 2**32:
            raise ConfigurationError("waypoint.navigation.noisy.seed: must be below 2^32")
        dropout_value = noisy["gps_dropout_window_s"]
        if dropout_value is not None:
            parsed = _number_tuple(
                dropout_value,
                "waypoint.navigation.noisy.gps_dropout_window_s",
                length=2,
            )
            dropout = cast(tuple[float, float], parsed)
            if dropout[0] < 0.0 or dropout[1] <= dropout[0]:
                raise ConfigurationError(
                    "waypoint.navigation.noisy.gps_dropout_window_s: "
                    "expected nonnegative increasing times"
                )
        position_sigma_m = _number(
            noisy["position_sigma_m"],
            "waypoint.navigation.noisy.position_sigma_m",
            nonnegative=True,
        )
        velocity_sigma_mps = _number(
            noisy["velocity_sigma_mps"],
            "waypoint.navigation.noisy.velocity_sigma_mps",
            nonnegative=True,
        )
        airspeed_sigma_mps = _number(
            noisy["airspeed_sigma_mps"],
            "waypoint.navigation.noisy.airspeed_sigma_mps",
            nonnegative=True,
        )
    elif mode is WaypointNavigationMode.NOISY:
        raise ConfigurationError("waypoint.navigation.noisy: section is required in noisy mode")

    estimated: EstimatedNavigationParameters | None = None
    if "estimated" in data:
        parsed_estimated = _estimated_navigation_configuration(
            data["estimated"],
            step_s=step_s,
            gravity_mps2=gravity_mps2,
        )
        if mode is WaypointNavigationMode.ESTIMATED:
            estimated = parsed_estimated
    elif mode is WaypointNavigationMode.ESTIMATED:
        raise ConfigurationError(
            "waypoint.navigation.estimated: section is required in estimated mode"
        )
    return WaypointNavigationConfiguration(
        mode=mode,
        seed=seed,
        position_sigma_m=position_sigma_m,
        velocity_sigma_mps=velocity_sigma_mps,
        airspeed_sigma_mps=airspeed_sigma_mps,
        gps_dropout_window_s=dropout,
        estimated_parameters=estimated,
    )


def _actuator_configuration(
    value: object,
) -> tuple[float, float, float, float, float, float, dict[str, SurfaceFailureMode]]:
    data = _mapping(value, "waypoint.actuators")
    _keys(
        data,
        "waypoint.actuators",
        required={
            "aileron_limit_deg",
            "elevator_limit_deg",
            "rudder_limit_deg",
            "surface_time_constant_s",
            "surface_rate_limit_degps",
            "throttle_time_constant_s",
            "failures",
        },
    )
    failures_data = _mapping(data["failures"], "waypoint.actuators.failures")
    _keys(
        failures_data,
        "waypoint.actuators.failures",
        required={"aileron", "elevator", "rudder"},
    )
    failures: dict[str, SurfaceFailureMode] = {}
    for channel in ("aileron", "elevator", "rudder"):
        failure_text = _string(failures_data[channel], f"waypoint.actuators.failures.{channel}")
        try:
            failures[channel] = SurfaceFailureMode(failure_text)
        except ValueError as error:
            choices = ", ".join(item.value for item in SurfaceFailureMode)
            raise ConfigurationError(
                f"waypoint.actuators.failures.{channel}: expected one of {choices}"
            ) from error
    return (
        radians(
            _number(
                data["aileron_limit_deg"],
                "waypoint.actuators.aileron_limit_deg",
                positive=True,
            )
        ),
        radians(
            _number(
                data["elevator_limit_deg"],
                "waypoint.actuators.elevator_limit_deg",
                positive=True,
            )
        ),
        radians(
            _number(
                data["rudder_limit_deg"],
                "waypoint.actuators.rudder_limit_deg",
                positive=True,
            )
        ),
        _number(
            data["surface_time_constant_s"],
            "waypoint.actuators.surface_time_constant_s",
            positive=True,
        ),
        radians(
            _number(
                data["surface_rate_limit_degps"],
                "waypoint.actuators.surface_rate_limit_degps",
                positive=True,
            )
        ),
        _number(
            data["throttle_time_constant_s"],
            "waypoint.actuators.throttle_time_constant_s",
            positive=True,
        ),
        failures,
    )


def load_waypoint_runtime_configuration(path: str | Path) -> WaypointRuntimeConfiguration:
    """Load a complete, simulation-only waypoint mission runtime."""
    source_path = Path(path).resolve()
    root = _load_yaml(source_path)
    _keys(
        root,
        "waypoint",
        required={
            "schema_version",
            "metadata",
            "mission_file",
            "output_directory",
            "simulation",
            "environment",
            "navigation",
            "guidance",
            "autopilot",
            "safety",
            "vehicle",
            "actuators",
            "hardware",
        },
    )
    schema_version = _integer(root["schema_version"], "waypoint.schema_version")
    if schema_version != WAYPOINT_CONFIGURATION_VERSION:
        raise ConfigurationError(
            f"waypoint.schema_version: expected {WAYPOINT_CONFIGURATION_VERSION}, "
            f"got {schema_version}"
        )

    metadata = _mapping(root["metadata"], "waypoint.metadata")
    _keys(metadata, "waypoint.metadata", required={"name", "safety_scope", "fictional"})
    if not _boolean(metadata["fictional"], "waypoint.metadata.fictional"):
        raise ConfigurationError("waypoint.metadata.fictional: must be true")
    safety_scope = _string(metadata["safety_scope"], "waypoint.metadata.safety_scope")
    scope_lower = safety_scope.lower()
    if not all(term in scope_lower for term in ("fictional", "civilian", "simulation")):
        raise ConfigurationError(
            "waypoint.metadata.safety_scope: must state fictional, civilian, and simulation scope"
        )

    hardware = _mapping(root["hardware"], "waypoint.hardware")
    _keys(hardware, "waypoint.hardware", required={"allow_real_vehicle_output"})
    allow_real_vehicle_output = _boolean(
        hardware["allow_real_vehicle_output"],
        "waypoint.hardware.allow_real_vehicle_output",
    )
    if allow_real_vehicle_output:
        raise ConfigurationError(
            "waypoint.hardware.allow_real_vehicle_output: real output is unavailable in this build"
        )

    mission_path = Path(_string(root["mission_file"], "waypoint.mission_file"))
    if not mission_path.is_absolute():
        mission_path = (source_path.parent / mission_path).resolve()
    if not mission_path.is_file():
        raise ConfigurationError(f"waypoint.mission_file: file does not exist: {mission_path}")
    from aerognc.mission import load_mission

    try:
        load_mission(mission_path)
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"waypoint.mission_file: {error}") from error

    simulation = _mapping(root["simulation"], "waypoint.simulation")
    _keys(
        simulation,
        "waypoint.simulation",
        required={"dt_s", "max_time_s", "initial_altitude_m", "initial_airspeed_mps"},
    )
    environment = _mapping(root["environment"], "waypoint.environment")
    _keys(environment, "waypoint.environment", required={"wind_ned_mps", "gravity_mps2"})
    step_s = _number(simulation["dt_s"], "waypoint.simulation.dt_s", positive=True)
    gravity_mps2 = _number(
        environment["gravity_mps2"],
        "waypoint.environment.gravity_mps2",
        positive=True,
    )
    wind = cast(
        tuple[float, float, float],
        _number_tuple(environment["wind_ned_mps"], "waypoint.environment.wind_ned_mps", length=3),
    )
    guidance_mode, guidance_gains = _guidance_configuration(root["guidance"])
    navigation = _navigation_configuration(
        root["navigation"],
        step_s=step_s,
        gravity_mps2=gravity_mps2,
    )
    (
        aileron_limit_rad,
        elevator_limit_rad,
        rudder_limit_rad,
        surface_time_constant_s,
        surface_rate_limit_radps,
        throttle_time_constant_s,
        failures,
    ) = _actuator_configuration(root["actuators"])
    vehicle_backend, reduced_params, coefficient_configuration = _vehicle_configuration(
        root["vehicle"], source_path.parent
    )

    mission_config = WaypointMissionConfig(
        dt_s=step_s,
        max_time_s=_number(
            simulation["max_time_s"], "waypoint.simulation.max_time_s", positive=True
        ),
        initial_altitude_m=_number(
            simulation["initial_altitude_m"],
            "waypoint.simulation.initial_altitude_m",
            positive=True,
        ),
        initial_airspeed_mps=_number(
            simulation["initial_airspeed_mps"],
            "waypoint.simulation.initial_airspeed_mps",
            positive=True,
        ),
        guidance_mode=guidance_mode,
        guidance_gains=guidance_gains,
        autopilot_gains=_autopilot_configuration(root["autopilot"]),
        safety_limits=_safety_configuration(root["safety"]),
        vehicle_backend=vehicle_backend,
        reduced_params=reduced_params,
        coefficient_configuration=coefficient_configuration,
        wind_ned_mps=wind,
        gravity_mps2=gravity_mps2,
        aileron_limit_rad=aileron_limit_rad,
        elevator_limit_rad=elevator_limit_rad,
        rudder_limit_rad=rudder_limit_rad,
        surface_time_constant_s=surface_time_constant_s,
        surface_rate_limit_radps=surface_rate_limit_radps,
        throttle_time_constant_s=throttle_time_constant_s,
        surface_failures=MappingProxyType(failures),
    )
    return WaypointRuntimeConfiguration(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        schema_version=schema_version,
        name=_string(metadata["name"], "waypoint.metadata.name"),
        safety_scope=safety_scope,
        mission_path=mission_path,
        mission_sha256=hashlib.sha256(mission_path.read_bytes()).hexdigest(),
        output_directory=Path(_string(root["output_directory"], "waypoint.output_directory")),
        navigation=navigation,
        mission_config=mission_config,
        allow_real_vehicle_output=False,
    )


__all__ = [
    "WAYPOINT_CONFIGURATION_VERSION",
    "WaypointNavigationConfiguration",
    "WaypointNavigationMode",
    "WaypointRuntimeConfiguration",
    "load_waypoint_runtime_configuration",
]
