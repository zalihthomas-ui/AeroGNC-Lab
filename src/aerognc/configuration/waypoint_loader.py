"""Strict versioned configuration for the integrated waypoint GNC runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from math import radians
from pathlib import Path
from types import MappingProxyType
from typing import cast

from aerognc.configuration.loader import (
    ConfigurationError,
    _boolean,
    _integer,
    _keys,
    _load_yaml,
    _mapping,
    _number,
    _number_tuple,
    _string,
)
from aerognc.gnc.fixedwing_autopilot import AutopilotGains
from aerognc.gnc.waypoint_guidance import GuidanceGains, GuidanceMode
from aerognc.mission.safety import SafetyLimits
from aerognc.navigation.providers import (
    NavigationProvider,
    NoisyStateProvider,
    PerfectStateProvider,
)
from aerognc.simulation.waypoint_backends import ReducedFixedWingParams
from aerognc.simulation.waypoint_mission import WaypointMissionConfig
from aerognc.vehicle.control_surfaces import SurfaceFailureMode

WAYPOINT_CONFIGURATION_VERSION = 1


class WaypointNavigationMode(StrEnum):
    """Navigation sources supported by the simulation-only waypoint runtime."""

    PERFECT = "perfect"
    NOISY = "noisy"


@dataclass(frozen=True, slots=True)
class WaypointNavigationConfiguration:
    """Provider selection and reproducible noisy-navigation settings."""

    mode: WaypointNavigationMode
    seed: int
    position_sigma_m: float
    velocity_sigma_mps: float
    airspeed_sigma_mps: float
    gps_dropout_window_s: tuple[float, float] | None

    def build_provider(self) -> NavigationProvider:
        """Return a fresh provider so repeated runs start from identical state."""
        if self.mode is WaypointNavigationMode.PERFECT:
            return PerfectStateProvider()
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


def _vehicle_configuration(value: object) -> ReducedFixedWingParams:
    data = _mapping(value, "waypoint.vehicle")
    _keys(data, "waypoint.vehicle", required={"backend", "parameters"})
    backend = _string(data["backend"], "waypoint.vehicle.backend")
    if backend != "internal_reduced":
        raise ConfigurationError(
            "waypoint.vehicle.backend: only internal_reduced is available in this build"
        )
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

    return ReducedFixedWingParams(
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


def _navigation_configuration(value: object) -> WaypointNavigationConfiguration:
    data = _mapping(value, "waypoint.navigation")
    _keys(data, "waypoint.navigation", required={"mode", "noisy"})
    mode_text = _string(data["mode"], "waypoint.navigation.mode")
    try:
        mode = WaypointNavigationMode(mode_text)
    except ValueError as error:
        raise ConfigurationError("waypoint.navigation.mode: expected perfect or noisy") from error
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
    dropout: tuple[float, float] | None = None
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
    return WaypointNavigationConfiguration(
        mode=mode,
        seed=seed,
        position_sigma_m=_number(
            noisy["position_sigma_m"],
            "waypoint.navigation.noisy.position_sigma_m",
            nonnegative=True,
        ),
        velocity_sigma_mps=_number(
            noisy["velocity_sigma_mps"],
            "waypoint.navigation.noisy.velocity_sigma_mps",
            nonnegative=True,
        ),
        airspeed_sigma_mps=_number(
            noisy["airspeed_sigma_mps"],
            "waypoint.navigation.noisy.airspeed_sigma_mps",
            nonnegative=True,
        ),
        gps_dropout_window_s=dropout,
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
    wind = cast(
        tuple[float, float, float],
        _number_tuple(environment["wind_ned_mps"], "waypoint.environment.wind_ned_mps", length=3),
    )
    guidance_mode, guidance_gains = _guidance_configuration(root["guidance"])
    navigation = _navigation_configuration(root["navigation"])
    (
        aileron_limit_rad,
        elevator_limit_rad,
        rudder_limit_rad,
        surface_time_constant_s,
        surface_rate_limit_radps,
        throttle_time_constant_s,
        failures,
    ) = _actuator_configuration(root["actuators"])

    mission_config = WaypointMissionConfig(
        dt_s=_number(simulation["dt_s"], "waypoint.simulation.dt_s", positive=True),
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
        reduced_params=_vehicle_configuration(root["vehicle"]),
        wind_ned_mps=wind,
        gravity_mps2=_number(
            environment["gravity_mps2"],
            "waypoint.environment.gravity_mps2",
            positive=True,
        ),
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
