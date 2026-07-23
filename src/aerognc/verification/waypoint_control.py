"""Cross-backend acceptance campaign for trim, TECS, and path continuity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from aerognc.mathematics.local_frame import wrap_to_pi
from aerognc.mission.mission import Mission
from aerognc.navigation.providers import PerfectStateProvider
from aerognc.simulation.waypoint_backends import VehicleBackendKind
from aerognc.simulation.waypoint_mission import (
    WaypointMissionConfig,
    WaypointMissionResult,
    run_waypoint_mission,
)


@dataclass(frozen=True, slots=True)
class WaypointControlCampaignLimits:
    """Declared cross-backend control, geometry, and envelope bounds."""

    maximum_coefficient_duration_s: float = 180.0
    maximum_reduced_duration_s: float = 185.0
    maximum_coefficient_cross_track_m: float = 15.0
    maximum_reduced_cross_track_m: float = 25.0
    maximum_course_command_step_rad: float = float(np.deg2rad(3.01))
    minimum_stall_margin_mps: float = 7.5
    maximum_load_factor: float = 1.3
    minimum_surface_margin_fraction: float = 0.65
    maximum_coefficient_total_energy_error_m2ps2: float = 150.0
    maximum_reduced_total_energy_error_m2ps2: float = 75.0
    maximum_trim_residual: float = 1.0e-8
    maximum_terminal_separation_m: float = 3.0
    maximum_duration_ratio: float = 1.1
    minimum_horizontal_wind_mps: float = 0.5

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("waypoint control campaign limits must be positive and finite")
        if self.minimum_surface_margin_fraction > 1.0:
            raise ValueError("surface margin acceptance cannot exceed one")


@dataclass(frozen=True, slots=True)
class WaypointControlBackendMetrics:
    """One backend's deterministic mission/control/envelope measurements."""

    backend: str
    completed: bool
    duration_s: float
    maximum_cross_track_m: float
    maximum_course_command_step_rad: float
    minimum_stall_margin_mps: float
    maximum_load_factor: float
    minimum_surface_margin_fraction: float
    minimum_lower_specific_energy_margin_m2ps2: float
    minimum_upper_specific_energy_margin_m2ps2: float
    maximum_total_energy_error_m2ps2: float
    actuator_saturation_samples: int
    safety_event_count: int
    trim_converged: bool
    trim_used_fallback: bool
    trim_residual_infinity_norm: float
    trim_iterations: int
    fillet_count: int
    tangent_orbit_transitions: bool
    segment_kinds: tuple[str, ...]
    bumpless_initialization: bool
    terminal_north_m: float
    terminal_east_m: float
    terminal_altitude_m: float


@dataclass(frozen=True, slots=True)
class WaypointControlScenario:
    """Reproducibility-critical inputs attached to the acceptance evidence."""

    mission_name: str
    configuration_name: str | None
    configuration_sha256: str | None
    mission_sha256: str | None
    navigation_provider: str
    guidance_mode: str
    longitudinal_control_mode: str
    dt_s: float
    initial_altitude_m: float
    initial_airspeed_mps: float
    wind_ned_mps: tuple[float, float, float]
    horizontal_wind_mps: float


@dataclass(frozen=True, slots=True)
class WaypointControlCampaignResult:
    """Reduced/coefficient acceptance result with exact declared limits."""

    coefficient: WaypointControlBackendMetrics
    reduced: WaypointControlBackendMetrics
    scenario: WaypointControlScenario
    terminal_separation_m: float
    duration_ratio: float
    limits: WaypointControlCampaignLimits

    @property
    def passed(self) -> bool:
        """Return whether every trim, control, path, and envelope bound passes."""
        common = all(
            metrics.completed
            and metrics.safety_event_count == 0
            and metrics.actuator_saturation_samples == 0
            and metrics.trim_converged
            and not metrics.trim_used_fallback
            and metrics.trim_residual_infinity_norm <= self.limits.maximum_trim_residual
            and metrics.fillet_count >= 1
            and metrics.tangent_orbit_transitions
            and set(metrics.segment_kinds) == {"line", "fillet", "orbit"}
            and metrics.bumpless_initialization
            and metrics.maximum_course_command_step_rad
            <= self.limits.maximum_course_command_step_rad
            and metrics.minimum_stall_margin_mps >= self.limits.minimum_stall_margin_mps
            and metrics.maximum_load_factor <= self.limits.maximum_load_factor
            and metrics.minimum_surface_margin_fraction
            >= self.limits.minimum_surface_margin_fraction
            and metrics.minimum_lower_specific_energy_margin_m2ps2 > 0.0
            and metrics.minimum_upper_specific_energy_margin_m2ps2 > 0.0
            for metrics in (self.coefficient, self.reduced)
        )
        return bool(
            common
            and self.coefficient.duration_s <= self.limits.maximum_coefficient_duration_s
            and self.reduced.duration_s <= self.limits.maximum_reduced_duration_s
            and self.coefficient.maximum_cross_track_m
            <= self.limits.maximum_coefficient_cross_track_m
            and self.reduced.maximum_cross_track_m <= self.limits.maximum_reduced_cross_track_m
            and self.coefficient.maximum_total_energy_error_m2ps2
            <= self.limits.maximum_coefficient_total_energy_error_m2ps2
            and self.reduced.maximum_total_energy_error_m2ps2
            <= self.limits.maximum_reduced_total_energy_error_m2ps2
            and self.terminal_separation_m <= self.limits.maximum_terminal_separation_m
            and self.duration_ratio <= self.limits.maximum_duration_ratio
            and self.scenario.horizontal_wind_mps >= self.limits.minimum_horizontal_wind_mps
        )

    def summary(self) -> dict[str, object]:
        """Return deterministic portable acceptance evidence."""
        return {
            "schema_version": "1.0",
            "scope": "simulation-only trim, total-energy control, and path geometry",
            "passed": self.passed,
            "coefficient": asdict(self.coefficient),
            "reduced": asdict(self.reduced),
            "scenario": asdict(self.scenario),
            "cross_backend": {
                "terminal_separation_m": self.terminal_separation_m,
                "duration_ratio": self.duration_ratio,
            },
            "limits": asdict(self.limits),
            "interpretation": (
                "Passing demonstrates bounded behavior for two internal fictional-aircraft "
                "simulators; it is research evidence, not control or flight certification."
            ),
        }


def run_waypoint_control_campaign(
    mission: Mission,
    coefficient_configuration: WaypointMissionConfig,
    *,
    limits: WaypointControlCampaignLimits | None = None,
) -> WaypointControlCampaignResult:
    """Run the same trim/TECS/fillet mission on both internal vehicle backends."""
    if coefficient_configuration.vehicle_backend is not VehicleBackendKind.INTERNAL_COEFFICIENT:
        raise ValueError("control campaign requires a coefficient-backend base configuration")
    if coefficient_configuration.coefficient_configuration is None:
        raise ValueError("control campaign coefficient aircraft configuration is unavailable")
    if not isinstance(coefficient_configuration.provider, PerfectStateProvider):
        raise ValueError("control campaign isolates control performance with perfect navigation")
    if not coefficient_configuration.trim_options.enabled:
        raise ValueError("control campaign requires solved trim")

    reduced_configuration = replace(
        coefficient_configuration,
        vehicle_backend=VehicleBackendKind.INTERNAL_REDUCED,
        coefficient_configuration=None,
        provider=PerfectStateProvider(),
    )
    coefficient = run_waypoint_mission(mission, coefficient_configuration)
    reduced = run_waypoint_mission(mission, reduced_configuration)
    coefficient_metrics = _backend_metrics(coefficient)
    reduced_metrics = _backend_metrics(reduced)
    coefficient_terminal = np.asarray(
        [
            coefficient_metrics.terminal_north_m,
            coefficient_metrics.terminal_east_m,
            coefficient_metrics.terminal_altitude_m,
        ]
    )
    reduced_terminal = np.asarray(
        [
            reduced_metrics.terminal_north_m,
            reduced_metrics.terminal_east_m,
            reduced_metrics.terminal_altitude_m,
        ]
    )
    return WaypointControlCampaignResult(
        coefficient=coefficient_metrics,
        reduced=reduced_metrics,
        scenario=_scenario(mission, coefficient_configuration),
        terminal_separation_m=float(np.linalg.norm(coefficient_terminal - reduced_terminal)),
        duration_ratio=float(
            max(coefficient.duration_s, reduced.duration_s)
            / min(coefficient.duration_s, reduced.duration_s)
        ),
        limits=limits or WaypointControlCampaignLimits(),
    )


def _scenario(
    mission: Mission,
    configuration: WaypointMissionConfig,
) -> WaypointControlScenario:
    wind = (
        float(configuration.wind_ned_mps[0]),
        float(configuration.wind_ned_mps[1]),
        float(configuration.wind_ned_mps[2]),
    )
    return WaypointControlScenario(
        mission_name=mission.name,
        configuration_name=configuration.configuration_name,
        configuration_sha256=configuration.configuration_sha256,
        mission_sha256=configuration.mission_sha256,
        navigation_provider=type(configuration.provider).__name__,
        guidance_mode=configuration.guidance_mode.value,
        longitudinal_control_mode=configuration.longitudinal_control_mode.value,
        dt_s=configuration.dt_s,
        initial_altitude_m=configuration.initial_altitude_m,
        initial_airspeed_mps=configuration.initial_airspeed_mps,
        wind_ned_mps=wind,
        horizontal_wind_mps=float(np.linalg.norm(wind[:2])),
    )


def write_waypoint_control_campaign(
    result: WaypointControlCampaignResult,
    path: str | Path,
) -> Path:
    """Write deterministic control-campaign evidence as strict JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def _backend_metrics(result: WaypointMissionResult) -> WaypointControlBackendMetrics:
    if not result.samples:
        raise ValueError("control campaign backend produced no samples")
    trim = result.metadata.get("trim")
    path = result.metadata.get("path_geometry")
    safety_events = result.metadata.get("safety_events")
    if (
        not isinstance(trim, dict)
        or not isinstance(path, dict)
        or not isinstance(safety_events, list)
    ):
        raise ValueError("control campaign result lacks trim/path/safety metadata")
    command_steps = [
        abs(
            wrap_to_pi(
                result.samples[index].course_command_rad
                - result.samples[index - 1].course_command_rad
            )
        )
        for index in range(1, len(result.samples))
    ]
    first = result.samples[0]
    last = result.samples[-1]
    return WaypointControlBackendMetrics(
        backend=str(result.metadata["vehicle_backend"]),
        completed=result.completed,
        duration_s=result.duration_s,
        maximum_cross_track_m=max(abs(sample.cross_track_error_m) for sample in result.samples),
        maximum_course_command_step_rad=max(command_steps, default=0.0),
        minimum_stall_margin_mps=min(sample.stall_margin_mps for sample in result.samples),
        maximum_load_factor=max(sample.load_factor for sample in result.samples),
        minimum_surface_margin_fraction=min(
            sample.minimum_surface_margin_fraction for sample in result.samples
        ),
        minimum_lower_specific_energy_margin_m2ps2=min(
            sample.lower_specific_energy_margin_m2ps2 for sample in result.samples
        ),
        minimum_upper_specific_energy_margin_m2ps2=min(
            sample.upper_specific_energy_margin_m2ps2 for sample in result.samples
        ),
        maximum_total_energy_error_m2ps2=max(
            abs(sample.total_energy_error_m2ps2) for sample in result.samples
        ),
        actuator_saturation_samples=sum(sample.actuator_saturated for sample in result.samples),
        safety_event_count=len(safety_events),
        trim_converged=bool(trim["converged"]),
        trim_used_fallback=bool(trim["used_fallback"]),
        trim_residual_infinity_norm=float(trim["residual_infinity_norm"]),
        trim_iterations=int(trim["iterations"]),
        fillet_count=int(path["fillet_count"]),
        tangent_orbit_transitions=bool(path["tangent_orbit_transitions"]),
        segment_kinds=tuple(sorted({sample.segment_kind for sample in result.samples})),
        bumpless_initialization=bool(
            np.isclose(first.pitch_command_rad, first.pitch_rad, atol=1.0e-12)
            and np.isclose(first.throttle, float(trim["throttle"]), atol=1.0e-12)
        ),
        terminal_north_m=last.north_m,
        terminal_east_m=last.east_m,
        terminal_altitude_m=last.altitude_m,
    )


__all__ = [
    "WaypointControlBackendMetrics",
    "WaypointControlCampaignLimits",
    "WaypointControlCampaignResult",
    "WaypointControlScenario",
    "run_waypoint_control_campaign",
    "write_waypoint_control_campaign",
]
