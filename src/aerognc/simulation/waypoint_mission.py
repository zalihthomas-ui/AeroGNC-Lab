"""Integrated waypoint fixed-wing GNC simulation loop.

Wires the full chain together:

    mission -> mission manager -> path manager -> guidance -> autopilot ->
    control surfaces -> vehicle backend (dynamics) -> navigation provider ->
    back to guidance, with the safety manager monitoring every step.

`run_waypoint_mission` air-starts the aircraft over home, arms and starts the
mission, and steps to completion, abort, emergency, or timeout, returning a
:class:`WaypointMissionResult` with a per-step log (doubling as the structured
mission log, Phase 13) and a computed summary. Real-hardware output is not
involved; this is the internal-simulation backend only.
"""

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.gnc.fixedwing_autopilot import (
    AutopilotGains,
    AutopilotOutput,
    AutopilotTrim,
    FixedWingAutopilot,
    LongitudinalControlMode,
    TotalEnergyControlGains,
)
from aerognc.gnc.path_manager import LineSegment, PathManager, PathManagerConfig
from aerognc.gnc.waypoint_envelope import (
    WaypointEnvelopeMargins,
    WaypointEnvelopeReference,
    evaluate_waypoint_envelope,
)
from aerognc.gnc.waypoint_guidance import (
    GuidanceCommand,
    GuidanceGains,
    GuidanceMode,
    PathFollowingGuidance,
    wind_corrected_heading_rad,
)
from aerognc.mission.mission import Mission
from aerognc.mission.mission_manager import MissionManager, MissionState
from aerognc.mission.safety import SafetyLimits, SafetyManager
from aerognc.navigation.providers import NavigationProvider, PerfectStateProvider
from aerognc.navigation.state import FlightEnvironment, NavigationState
from aerognc.simulation.waypoint_backends import (
    InternalCoefficientFixedWingBackend,
    InternalFixedWingBackend,
    ReducedFixedWingParams,
    VehicleBackend,
    VehicleBackendKind,
)
from aerognc.simulation.waypoint_trim import (
    TrimConvergenceError,
    TrimFailurePolicy,
    WaypointTrimOptions,
    WaypointTrimResult,
    configuration_with_resolved_trim,
    solve_coefficient_waypoint_trim,
    solve_reduced_waypoint_trim,
)
from aerognc.vehicle.control_surfaces import (
    ControlSurface,
    ControlSurfaceConfig,
    ControlSurfaceSet,
    SurfaceDeflections,
    SurfaceFailureMode,
)
from aerognc.vehicle.fixed_wing import STANDARD_GRAVITY_MPS2

_TERMINAL_STATES = frozenset(
    {
        MissionState.MISSION_COMPLETE,
        MissionState.ABORT,
        MissionState.EMERGENCY,
        MissionState.LANDED,
    }
)


@dataclass(frozen=True, slots=True)
class WaypointMissionConfig:
    """Configuration for a waypoint-mission simulation run."""

    dt_s: float = 0.05
    max_time_s: float = 900.0
    initial_altitude_m: float = 100.0
    initial_airspeed_mps: float = 20.0
    guidance_mode: GuidanceMode = GuidanceMode.VECTOR_FIELD
    guidance_gains: GuidanceGains = field(default_factory=GuidanceGains)
    path_manager_config: PathManagerConfig = field(default_factory=PathManagerConfig)
    autopilot_gains: AutopilotGains = field(default_factory=AutopilotGains)
    longitudinal_control_mode: LongitudinalControlMode = LongitudinalControlMode.ALTITUDE_AIRSPEED
    total_energy_gains: TotalEnergyControlGains = field(default_factory=TotalEnergyControlGains)
    trim_options: WaypointTrimOptions = field(default_factory=WaypointTrimOptions)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    vehicle_backend: VehicleBackendKind = VehicleBackendKind.INTERNAL_REDUCED
    reduced_params: ReducedFixedWingParams = field(default_factory=ReducedFixedWingParams)
    coefficient_configuration: AircraftSandboxConfiguration | None = None
    wind_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_mps2: float = 9.80665
    aileron_limit_rad: float = float(np.deg2rad(22.0))
    elevator_limit_rad: float = float(np.deg2rad(25.0))
    rudder_limit_rad: float = float(np.deg2rad(28.0))
    surface_time_constant_s: float = 0.12
    surface_rate_limit_radps: float = float(np.deg2rad(120.0))
    throttle_time_constant_s: float = 0.6
    surface_failures: Mapping[str, SurfaceFailureMode] = field(default_factory=dict)
    provider: NavigationProvider | None = None
    configuration_name: str | None = None
    configuration_schema_version: int | None = None
    configuration_sha256: str | None = None
    mission_sha256: str | None = None

    def __post_init__(self) -> None:
        positive = (
            self.dt_s,
            self.max_time_s,
            self.initial_altitude_m,
            self.initial_airspeed_mps,
            self.gravity_mps2,
            self.aileron_limit_rad,
            self.elevator_limit_rad,
            self.rudder_limit_rad,
            self.surface_time_constant_s,
            self.surface_rate_limit_radps,
            self.throttle_time_constant_s,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("waypoint mission scalar settings must be positive and finite")
        if len(self.wind_ned_mps) != 3 or not np.all(np.isfinite(self.wind_ned_mps)):
            raise ValueError("wind_ned_mps must contain three finite values")
        allowed = {"aileron", "elevator", "rudder"}
        unknown = set(self.surface_failures) - allowed
        if unknown:
            raise ValueError(f"unknown surface_failures channels: {sorted(unknown)}")
        if not isinstance(self.vehicle_backend, VehicleBackendKind):
            raise ValueError("vehicle_backend must be a VehicleBackendKind")
        if not isinstance(self.longitudinal_control_mode, LongitudinalControlMode):
            raise ValueError("longitudinal_control_mode must be a LongitudinalControlMode")
        if not isinstance(self.path_manager_config, PathManagerConfig):
            raise ValueError("path_manager_config must be a PathManagerConfig")
        if not isinstance(self.trim_options, WaypointTrimOptions):
            raise ValueError("trim_options must be a WaypointTrimOptions")
        if (
            self.vehicle_backend is VehicleBackendKind.INTERNAL_COEFFICIENT
            and self.coefficient_configuration is None
        ):
            raise ValueError("internal_coefficient backend requires an aircraft configuration")
        if (
            self.vehicle_backend is VehicleBackendKind.INTERNAL_REDUCED
            and self.coefficient_configuration is not None
        ):
            raise ValueError("reduced backend cannot carry a coefficient aircraft configuration")
        if self.coefficient_configuration is not None:
            if self.dt_s > 0.1:
                raise ValueError("coefficient backend step cannot exceed 0.1 s")
            if not np.isclose(
                self.gravity_mps2,
                STANDARD_GRAVITY_MPS2,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    "coefficient backend derives planet gravity and requires the standard "
                    "waypoint interface value"
                )
            geometry = self.coefficient_configuration.geometry
            configured_limits = np.array(
                [
                    self.aileron_limit_rad,
                    self.elevator_limit_rad,
                    self.rudder_limit_rad,
                ]
            )
            plant_limits = np.array(
                [
                    geometry.aileron_limit_rad,
                    geometry.elevator_limit_rad,
                    geometry.rudder_limit_rad,
                ]
            )
            if np.any(configured_limits > plant_limits + 1.0e-12):
                raise ValueError("waypoint actuator limits exceed coefficient-plant limits")
        if self.configuration_name is not None and not self.configuration_name.strip():
            raise ValueError("configuration_name must not be blank")
        if self.configuration_schema_version is not None and self.configuration_schema_version < 1:
            raise ValueError("configuration_schema_version must be positive")
        for label, digest in (
            ("configuration_sha256", self.configuration_sha256),
            ("mission_sha256", self.mission_sha256),
        ):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class MissionSample:
    """One logged step of the mission (structured log row)."""

    time_s: float
    north_m: float
    east_m: float
    altitude_m: float
    airspeed_mps: float
    groundspeed_mps: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    course_rad: float
    roll_command_rad: float
    pitch_command_rad: float
    course_command_rad: float
    altitude_command_m: float
    airspeed_command_mps: float
    climb_rate_command_mps: float
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    cross_track_error_m: float
    distance_to_waypoint_m: float
    active_waypoint_id: int
    segment_kind: str
    longitudinal_mode: str
    potential_energy_error_m2ps2: float
    kinetic_energy_error_m2ps2: float
    total_energy_error_m2ps2: float
    energy_balance_error_m2ps2: float
    stall_speed_reference_mps: float
    stall_margin_mps: float
    load_factor: float
    bank_margin_rad: float
    pitch_margin_rad: float
    minimum_surface_margin_fraction: float
    throttle_margin_fraction: float
    lower_specific_energy_margin_m2ps2: float
    upper_specific_energy_margin_m2ps2: float
    actuator_saturated: bool
    mission_state: str
    safety_response: str

    def as_row(self) -> dict[str, float | int | str]:
        """Return the sample as a flat mapping (CSV/JSON friendly)."""
        return {
            "time_s": self.time_s,
            "north_m": self.north_m,
            "east_m": self.east_m,
            "altitude_m": self.altitude_m,
            "airspeed_mps": self.airspeed_mps,
            "groundspeed_mps": self.groundspeed_mps,
            "roll_rad": self.roll_rad,
            "pitch_rad": self.pitch_rad,
            "yaw_rad": self.yaw_rad,
            "course_rad": self.course_rad,
            "roll_command_rad": self.roll_command_rad,
            "pitch_command_rad": self.pitch_command_rad,
            "course_command_rad": self.course_command_rad,
            "altitude_command_m": self.altitude_command_m,
            "airspeed_command_mps": self.airspeed_command_mps,
            "climb_rate_command_mps": self.climb_rate_command_mps,
            "aileron": self.aileron,
            "elevator": self.elevator,
            "rudder": self.rudder,
            "throttle": self.throttle,
            "cross_track_error_m": self.cross_track_error_m,
            "distance_to_waypoint_m": self.distance_to_waypoint_m,
            "active_waypoint_id": self.active_waypoint_id,
            "segment_kind": self.segment_kind,
            "longitudinal_mode": self.longitudinal_mode,
            "potential_energy_error_m2ps2": self.potential_energy_error_m2ps2,
            "kinetic_energy_error_m2ps2": self.kinetic_energy_error_m2ps2,
            "total_energy_error_m2ps2": self.total_energy_error_m2ps2,
            "energy_balance_error_m2ps2": self.energy_balance_error_m2ps2,
            "stall_speed_reference_mps": self.stall_speed_reference_mps,
            "stall_margin_mps": self.stall_margin_mps,
            "load_factor": self.load_factor,
            "bank_margin_rad": self.bank_margin_rad,
            "pitch_margin_rad": self.pitch_margin_rad,
            "minimum_surface_margin_fraction": self.minimum_surface_margin_fraction,
            "throttle_margin_fraction": self.throttle_margin_fraction,
            "lower_specific_energy_margin_m2ps2": self.lower_specific_energy_margin_m2ps2,
            "upper_specific_energy_margin_m2ps2": self.upper_specific_energy_margin_m2ps2,
            "actuator_saturated": self.actuator_saturated,
            "mission_state": self.mission_state,
            "safety_response": self.safety_response,
        }


@dataclass(frozen=True, slots=True)
class WaypointMissionResult:
    """Result of a waypoint-mission simulation."""

    outcome: str  # "complete" | "abort" | "emergency" | "timeout"
    final_state: MissionState
    duration_s: float
    samples: tuple[MissionSample, ...]
    planned_path_ned_m: np.ndarray
    metadata: Mapping[str, object]

    @property
    def completed(self) -> bool:
        return self.outcome == "complete"

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-step mission log to CSV and return the path."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [sample.as_row() for sample in self.samples]
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return file_path

    def to_json(self, path: str | Path) -> Path:
        """Write summary, metadata, and the full log to JSON; return the path."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "metadata": dict(self.metadata),
            "samples": [sample.as_row() for sample in self.samples],
        }
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return file_path

    def summary(self) -> dict[str, float | int | str | bool]:
        """Return key mission metrics for reporting."""
        if not self.samples:
            return {"outcome": self.outcome, "completed": self.completed, "samples": 0}
        cross_track = [abs(s.cross_track_error_m) for s in self.samples]
        altitude = [s.altitude_m for s in self.samples]
        airspeed = [s.airspeed_mps for s in self.samples]
        stall_margin = [s.stall_margin_mps for s in self.samples]
        load_factor = [s.load_factor for s in self.samples]
        surface_margin = [s.minimum_surface_margin_fraction for s in self.samples]
        lower_energy_margin = [s.lower_specific_energy_margin_m2ps2 for s in self.samples]
        upper_energy_margin = [s.upper_specific_energy_margin_m2ps2 for s in self.samples]
        total_energy_error = [abs(s.total_energy_error_m2ps2) for s in self.samples]
        return {
            "outcome": self.outcome,
            "completed": self.completed,
            "final_state": self.final_state.value,
            "duration_s": round(self.duration_s, 3),
            "samples": len(self.samples),
            "max_abs_cross_track_m": round(max(cross_track), 3),
            "final_cross_track_m": round(cross_track[-1], 3),
            "min_altitude_m": round(min(altitude), 3),
            "max_altitude_m": round(max(altitude), 3),
            "min_airspeed_mps": round(min(airspeed), 3),
            "max_airspeed_mps": round(max(airspeed), 3),
            "min_stall_margin_mps": round(min(stall_margin), 3),
            "max_load_factor": round(max(load_factor), 4),
            "min_surface_margin_fraction": round(min(surface_margin), 4),
            "min_lower_specific_energy_margin_m2ps2": round(min(lower_energy_margin), 3),
            "min_upper_specific_energy_margin_m2ps2": round(min(upper_energy_margin), 3),
            "max_abs_total_energy_error_m2ps2": round(max(total_energy_error), 3),
            "actuator_saturation_samples": sum(s.actuator_saturated for s in self.samples),
        }


def _build_surfaces(
    config: WaypointMissionConfig,
    trim: AutopilotTrim,
    trim_result: WaypointTrimResult | None,
) -> ControlSurfaceSet:
    def make(
        channel: str,
        limit_rad: float,
        *,
        initial_position_rad: float | None = None,
    ) -> ControlSurface:
        return ControlSurface(
            ControlSurfaceConfig(
                max_deflection_rad=limit_rad,
                time_constant_s=config.surface_time_constant_s,
                rate_limit_radps=config.surface_rate_limit_radps,
            ),
            failure=config.surface_failures.get(channel, SurfaceFailureMode.NONE),
            initial_position_rad=initial_position_rad,
        )

    coefficient_configuration = config.coefficient_configuration
    initial_throttle = (
        trim.throttle
        if trim_result is not None
        else (
            coefficient_configuration.initial_throttle
            if coefficient_configuration is not None
            else 0.0
        )
    )
    initial_elevator_rad = (
        trim.elevator_command * config.elevator_limit_rad if trim_result is not None else None
    )
    return ControlSurfaceSet(
        make("aileron", config.aileron_limit_rad),
        make(
            "elevator",
            config.elevator_limit_rad,
            initial_position_rad=initial_elevator_rad,
        ),
        make("rudder", config.rudder_limit_rad),
        throttle_time_constant_s=config.throttle_time_constant_s,
        initial_throttle=initial_throttle,
    )


def _initial_heading_rad(
    path_manager: PathManager,
    *,
    wind_ned_mps: tuple[float, float, float],
    airspeed_mps: float,
) -> float:
    first = path_manager.segments[0]
    if isinstance(first, LineSegment):
        direction = first.horizontal_direction_ne
        course_rad = float(np.arctan2(direction[1], direction[0]))
        return wind_corrected_heading_rad(
            course_rad,
            np.asarray(wind_ned_mps, dtype=np.float64),
            airspeed_mps,
        )
    return 0.0


def _build_backend(
    config: WaypointMissionConfig,
    coefficient_configuration: AircraftSandboxConfiguration | None = None,
) -> VehicleBackend:
    if config.vehicle_backend is VehicleBackendKind.INTERNAL_REDUCED:
        return InternalFixedWingBackend(config.reduced_params)
    selected_configuration = coefficient_configuration or config.coefficient_configuration
    if selected_configuration is None:  # pragma: no cover - dataclass invariant
        raise ValueError("coefficient backend has no aircraft configuration")
    return InternalCoefficientFixedWingBackend(
        selected_configuration,
        steady_wind_ned_mps=config.wind_ned_mps,
        wind_horizon_s=config.max_time_s + config.dt_s,
    )


def _resolve_trim(
    config: WaypointMissionConfig,
    *,
    heading_rad: float,
) -> tuple[
    AutopilotTrim,
    WaypointTrimResult | None,
    AircraftSandboxConfiguration | None,
]:
    configured_trim = AutopilotTrim(
        pitch_rad=0.0,
        elevator_command=config.autopilot_gains.elevator_trim,
        throttle=config.autopilot_gains.throttle_trim,
    )
    if not config.trim_options.enabled:
        return configured_trim, None, None
    if config.vehicle_backend is VehicleBackendKind.INTERNAL_REDUCED:
        result = solve_reduced_waypoint_trim(
            config.reduced_params,
            airspeed_mps=config.initial_airspeed_mps,
        )
        if not result.converged:
            if config.trim_options.failure_policy is TrimFailurePolicy.REJECT:
                raise TrimConvergenceError(
                    "reduced waypoint trim failed because requested airspeed is outside "
                    "the analytic throttle equilibrium"
                )
            speed_range = (
                config.reduced_params.airspeed_at_full_throttle_mps
                - config.reduced_params.airspeed_at_zero_throttle_mps
            )
            achieved_airspeed_mps = (
                config.reduced_params.airspeed_at_zero_throttle_mps
                + configured_trim.throttle * speed_range
            )
            residual = achieved_airspeed_mps - config.initial_airspeed_mps
            fallback = WaypointTrimResult(
                backend=result.backend,
                source="configured_reduced_fallback_after_failure",
                converged=False,
                used_fallback=True,
                angle_of_attack_rad=0.0,
                pitch_rad=configured_trim.pitch_rad,
                elevator_deflection_rad=(
                    configured_trim.elevator_command * config.elevator_limit_rad
                ),
                elevator_command=configured_trim.elevator_command,
                throttle=configured_trim.throttle,
                residual=(residual, 0.0, 0.0),
                residual_infinity_norm=abs(residual),
                iterations=result.iterations,
            )
            return configured_trim, fallback, None
        return result.autopilot_trim, result, None

    coefficient_configuration = config.coefficient_configuration
    if coefficient_configuration is None:  # pragma: no cover - dataclass invariant
        raise RuntimeError("coefficient trim requires an aircraft configuration")
    result = solve_coefficient_waypoint_trim(
        coefficient_configuration,
        altitude_m=config.initial_altitude_m,
        airspeed_mps=config.initial_airspeed_mps,
        heading_rad=heading_rad,
        steady_wind_ned_mps=config.wind_ned_mps,
        elevator_command_limit_rad=config.elevator_limit_rad,
        options=config.trim_options,
    )
    resolved_configuration = configuration_with_resolved_trim(
        coefficient_configuration,
        result,
    )
    return result.autopilot_trim, result, resolved_configuration


def _build_envelope_reference(
    config: WaypointMissionConfig,
    coefficient_configuration: AircraftSandboxConfiguration | None,
) -> WaypointEnvelopeReference:
    limits = config.safety_limits
    return WaypointEnvelopeReference(
        minimum_altitude_m=limits.min_altitude_m,
        maximum_altitude_m=limits.max_altitude_m,
        minimum_airspeed_mps=limits.min_airspeed_mps,
        maximum_airspeed_mps=limits.max_airspeed_mps,
        maximum_bank_rad=limits.max_bank_rad,
        maximum_pitch_rad=limits.max_pitch_rad,
        aileron_limit_rad=config.aileron_limit_rad,
        elevator_limit_rad=config.elevator_limit_rad,
        rudder_limit_rad=config.rudder_limit_rad,
        gravity_mps2=config.gravity_mps2,
        coefficient_configuration=coefficient_configuration,
    )


def run_waypoint_mission(
    mission: Mission, config: WaypointMissionConfig | None = None
) -> WaypointMissionResult:
    """Run an air-started waypoint mission on the internal backend to completion."""
    cfg = config or WaypointMissionConfig()
    mission.validate()
    frame = mission.local_frame()
    initial_position_ned_m = np.array([0.0, 0.0, -cfg.initial_altitude_m])
    path_manager = PathManager.from_mission(
        mission,
        frame=frame,
        config=cfg.path_manager_config,
        initial_position_ned_m=initial_position_ned_m,
    )
    manager = MissionManager(mission, path_manager)
    guidance = PathFollowingGuidance(cfg.guidance_mode, cfg.guidance_gains)
    initial_heading_rad = _initial_heading_rad(
        path_manager,
        wind_ned_mps=cfg.wind_ned_mps,
        airspeed_mps=cfg.initial_airspeed_mps,
    )
    trim, trim_result, resolved_coefficient_configuration = _resolve_trim(
        cfg,
        heading_rad=initial_heading_rad,
    )
    autopilot = FixedWingAutopilot(
        cfg.autopilot_gains,
        longitudinal_mode=cfg.longitudinal_control_mode,
        total_energy_gains=cfg.total_energy_gains,
        trim=trim,
    )
    surfaces = _build_surfaces(cfg, trim, trim_result)
    backend = _build_backend(cfg, resolved_coefficient_configuration)
    envelope_reference = _build_envelope_reference(
        cfg,
        resolved_coefficient_configuration or cfg.coefficient_configuration,
    )
    provider = cfg.provider or PerfectStateProvider()
    provider.reset()
    safety = SafetyManager(cfg.safety_limits)
    environment = FlightEnvironment(np.asarray(cfg.wind_ned_mps, dtype=float), cfg.gravity_mps2)

    backend.initialize(
        position_ned_m=initial_position_ned_m,
        heading_rad=initial_heading_rad,
        airspeed_mps=cfg.initial_airspeed_mps,
    )
    manager.arm()
    manager.start()

    samples: list[MissionSample] = []
    time_s = 0.0
    last_cross_track_m = 0.0
    outcome = "timeout"
    steps = int(cfg.max_time_s / cfg.dt_s)

    for _ in range(steps):
        truth = backend.read_state()
        nav = provider.update(truth, cfg.dt_s)
        verdict = safety.check(nav, time_s, cross_track_error_m=last_cross_track_m)
        manager_status = manager.update(nav, cfg.dt_s, verdict.response)

        if manager_status.state in _TERMINAL_STATES:
            outcome = _outcome_for(manager_status.state)
            break
        if manager_status.path_status is None:
            time_s += cfg.dt_s
            continue

        segment = manager_status.path_status.active_segment
        guidance_command = guidance.update(nav, segment, environment, cfg.dt_s)
        last_cross_track_m = guidance_command.cross_track_error_m
        autopilot_output = autopilot.update(guidance_command, nav, cfg.dt_s)
        deflections = surfaces.update(
            autopilot_output.actuator.aileron,
            autopilot_output.actuator.elevator,
            autopilot_output.actuator.rudder,
            autopilot_output.actuator.throttle,
            cfg.dt_s,
        )
        envelope_margins = evaluate_waypoint_envelope(nav, deflections, envelope_reference)
        backend.send_actuator_commands(deflections)
        try:
            backend.step(cfg.dt_s, environment)
        except FloatingPointError:
            manager.trigger_emergency("non-finite dynamics")
            outcome = "emergency"
            break

        samples.append(
            _make_sample(
                time_s,
                nav,
                guidance_command,
                autopilot_output,
                deflections,
                envelope_margins,
                manager_status.state,
                segment.waypoint_id,
                segment.kind.value,
                verdict.response.value,
            )
        )
        time_s += cfg.dt_s

    if manager.state is MissionState.MISSION_COMPLETE:
        outcome = "complete"

    metadata: dict[str, object] = {
        "guidance_mode": cfg.guidance_mode.value,
        "path_geometry": {
            "fillet_count": len(path_manager.fillet_arcs()),
            "fillet_bank_rad": cfg.path_manager_config.fillet_bank_rad,
            "fillet_max_radius_m": cfg.path_manager_config.fillet_max_radius_m,
            "fillet_leg_fraction": cfg.path_manager_config.fillet_leg_fraction,
            "tangent_orbit_transitions": cfg.path_manager_config.tangent_orbit_transitions,
            "course_command_rate_limit_radps": (cfg.guidance_gains.course_command_rate_limit_radps),
            "roll_feedforward_rate_limit_radps": (
                cfg.guidance_gains.roll_feedforward_rate_limit_radps
            ),
        },
        "autopilot": dict(autopilot.provenance()),
        "trim": (
            trim_result.summary()
            if trim_result is not None
            else {"enabled": False, "source": "configured_autopilot_feedforward"}
        ),
        "envelope_reference": {
            "stall_source": (
                "coefficient_cl_max_at_estimated_altitude_and_load"
                if envelope_reference.coefficient_configuration is not None
                else "declared_reduced_model_minimum_airspeed"
            ),
            "controller_facing_state_only": True,
        },
        "navigation_provider": type(provider).__name__,
        "navigation_provider_details": dict(provider.provenance()),
        "navigation_diagnostics": dict(provider.diagnostics()),
        "vehicle_backend": type(backend).__name__,
        "vehicle_backend_details": dict(backend.provenance()),
        "dt_s": cfg.dt_s,
        "wind_ned_mps": list(cfg.wind_ned_mps),
        "surface_failures": {k: v.value for k, v in cfg.surface_failures.items()},
        "transitions": [
            (t.time_s, t.from_state.value, t.to_state.value, t.reason) for t in manager.transitions
        ],
        "safety_events": [
            (e.time_s, e.trigger, e.value, e.threshold, e.response.value) for e in safety.events
        ],
    }
    if cfg.configuration_name is not None:
        metadata["runtime_configuration"] = {
            "name": cfg.configuration_name,
            "schema_version": cfg.configuration_schema_version,
            "sha256": cfg.configuration_sha256,
            "mission_sha256": cfg.mission_sha256,
        }

    return WaypointMissionResult(
        outcome=outcome,
        final_state=manager.state,
        duration_s=time_s,
        samples=tuple(samples),
        planned_path_ned_m=path_manager.planned_path_ned(),
        metadata=metadata,
    )


def _outcome_for(state: MissionState) -> str:
    if state is MissionState.MISSION_COMPLETE:
        return "complete"
    if state is MissionState.EMERGENCY:
        return "emergency"
    if state is MissionState.LANDED:
        return "landed"
    return "abort"


def _make_sample(
    time_s: float,
    nav: NavigationState,
    guidance: GuidanceCommand,
    autopilot: AutopilotOutput,
    deflections: SurfaceDeflections,
    envelope: WaypointEnvelopeMargins,
    state: MissionState,
    active_waypoint_id: int,
    segment_kind: str,
    safety_response: str,
) -> MissionSample:
    roll, pitch, yaw = nav.euler_rad
    return MissionSample(
        time_s=time_s,
        north_m=float(nav.position_ned_m[0]),
        east_m=float(nav.position_ned_m[1]),
        altitude_m=nav.altitude_m,
        airspeed_mps=nav.airspeed_mps,
        groundspeed_mps=nav.groundspeed_mps,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        course_rad=nav.course_rad,
        roll_command_rad=autopilot.control.roll_command_rad,
        pitch_command_rad=autopilot.control.pitch_command_rad,
        course_command_rad=guidance.course_command_rad,
        altitude_command_m=guidance.altitude_command_m,
        airspeed_command_mps=guidance.airspeed_command_mps,
        climb_rate_command_mps=guidance.climb_rate_command_mps,
        aileron=deflections.aileron_rad,
        elevator=deflections.elevator_rad,
        rudder=deflections.rudder_rad,
        throttle=deflections.throttle,
        cross_track_error_m=guidance.cross_track_error_m,
        distance_to_waypoint_m=guidance.distance_to_waypoint_m,
        active_waypoint_id=active_waypoint_id,
        segment_kind=segment_kind,
        longitudinal_mode=autopilot.longitudinal.mode,
        potential_energy_error_m2ps2=autopilot.longitudinal.potential_energy_error_m2ps2,
        kinetic_energy_error_m2ps2=autopilot.longitudinal.kinetic_energy_error_m2ps2,
        total_energy_error_m2ps2=autopilot.longitudinal.total_energy_error_m2ps2,
        energy_balance_error_m2ps2=autopilot.longitudinal.energy_balance_error_m2ps2,
        stall_speed_reference_mps=envelope.stall_speed_reference_mps,
        stall_margin_mps=envelope.stall_margin_mps,
        load_factor=envelope.load_factor,
        bank_margin_rad=envelope.bank_margin_rad,
        pitch_margin_rad=envelope.pitch_margin_rad,
        minimum_surface_margin_fraction=envelope.minimum_surface_margin_fraction,
        throttle_margin_fraction=envelope.throttle_margin_fraction,
        lower_specific_energy_margin_m2ps2=envelope.lower_specific_energy_margin_m2ps2,
        upper_specific_energy_margin_m2ps2=envelope.upper_specific_energy_margin_m2ps2,
        actuator_saturated=deflections.saturated,
        mission_state=state.value,
        safety_response=safety_response,
    )
