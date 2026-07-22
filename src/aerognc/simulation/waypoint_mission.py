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

from aerognc.gnc.fixedwing_autopilot import (
    AutopilotGains,
    FixedWingAutopilot,
)
from aerognc.gnc.path_manager import LineSegment, PathManager
from aerognc.gnc.waypoint_guidance import (
    GuidanceCommand,
    GuidanceGains,
    GuidanceMode,
    PathFollowingGuidance,
)
from aerognc.mission.mission import Mission
from aerognc.mission.mission_manager import MissionManager, MissionState
from aerognc.mission.safety import SafetyLimits, SafetyManager
from aerognc.navigation.providers import NavigationProvider, PerfectStateProvider
from aerognc.navigation.state import FlightEnvironment, NavigationState
from aerognc.simulation.waypoint_backends import InternalFixedWingBackend, ReducedFixedWingParams
from aerognc.vehicle.control_surfaces import (
    ControlSurface,
    ControlSurfaceConfig,
    ControlSurfaceSet,
    SurfaceDeflections,
    SurfaceFailureMode,
)

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
    autopilot_gains: AutopilotGains = field(default_factory=AutopilotGains)
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    reduced_params: ReducedFixedWingParams = field(default_factory=ReducedFixedWingParams)
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
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    cross_track_error_m: float
    distance_to_waypoint_m: float
    active_waypoint_id: int
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
            "aileron": self.aileron,
            "elevator": self.elevator,
            "rudder": self.rudder,
            "throttle": self.throttle,
            "cross_track_error_m": self.cross_track_error_m,
            "distance_to_waypoint_m": self.distance_to_waypoint_m,
            "active_waypoint_id": self.active_waypoint_id,
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
        }


def _build_surfaces(config: WaypointMissionConfig) -> ControlSurfaceSet:
    def make(channel: str, limit_rad: float) -> ControlSurface:
        return ControlSurface(
            ControlSurfaceConfig(
                max_deflection_rad=limit_rad,
                time_constant_s=config.surface_time_constant_s,
                rate_limit_radps=config.surface_rate_limit_radps,
            ),
            failure=config.surface_failures.get(channel, SurfaceFailureMode.NONE),
        )

    return ControlSurfaceSet(
        make("aileron", config.aileron_limit_rad),
        make("elevator", config.elevator_limit_rad),
        make("rudder", config.rudder_limit_rad),
        throttle_time_constant_s=config.throttle_time_constant_s,
    )


def _initial_heading_rad(path_manager: PathManager) -> float:
    first = path_manager.segments[0]
    if isinstance(first, LineSegment):
        direction = first.horizontal_direction_ne
        return float(np.arctan2(direction[1], direction[0]))
    return 0.0


def run_waypoint_mission(
    mission: Mission, config: WaypointMissionConfig | None = None
) -> WaypointMissionResult:
    """Run an air-started waypoint mission on the internal backend to completion."""
    cfg = config or WaypointMissionConfig()
    mission.validate()
    frame = mission.local_frame()
    path_manager = PathManager.from_mission(mission, frame=frame)
    manager = MissionManager(mission, path_manager)
    guidance = PathFollowingGuidance(cfg.guidance_mode, cfg.guidance_gains)
    autopilot = FixedWingAutopilot(cfg.autopilot_gains)
    surfaces = _build_surfaces(cfg)
    backend = InternalFixedWingBackend(cfg.reduced_params)
    provider = cfg.provider or PerfectStateProvider()
    provider.reset()
    safety = SafetyManager(cfg.safety_limits)
    environment = FlightEnvironment(np.asarray(cfg.wind_ned_mps, dtype=float), cfg.gravity_mps2)

    backend.initialize(
        position_ned_m=np.array([0.0, 0.0, -cfg.initial_altitude_m]),
        heading_rad=_initial_heading_rad(path_manager),
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
                deflections,
                manager_status.state,
                segment.waypoint_id,
                verdict.response.value,
            )
        )
        time_s += cfg.dt_s

    if manager.state is MissionState.MISSION_COMPLETE:
        outcome = "complete"

    metadata: dict[str, object] = {
        "guidance_mode": cfg.guidance_mode.value,
        "navigation_provider": type(provider).__name__,
        "vehicle_backend": type(backend).__name__,
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
    deflections: SurfaceDeflections,
    state: MissionState,
    active_waypoint_id: int,
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
        roll_command_rad=guidance.roll_feedforward_rad,
        pitch_command_rad=guidance.climb_rate_command_mps,
        course_command_rad=guidance.course_command_rad,
        altitude_command_m=guidance.altitude_command_m,
        airspeed_command_mps=guidance.airspeed_command_mps,
        aileron=deflections.aileron_rad,
        elevator=deflections.elevator_rad,
        rudder=deflections.rudder_rad,
        throttle=deflections.throttle,
        cross_track_error_m=guidance.cross_track_error_m,
        distance_to_waypoint_m=guidance.distance_to_waypoint_m,
        active_waypoint_id=active_waypoint_id,
        mission_state=state.value,
        safety_response=safety_response,
    )
