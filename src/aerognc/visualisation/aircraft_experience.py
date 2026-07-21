"""Bounded timing, trails, warnings, recording, replay, and debrief services."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from aerognc.configuration.aircraft_loader import AircraftSandboxConfiguration
from aerognc.mathematics.quaternion import normalize_quaternion
from aerognc.mathematics.vectors import FloatArray, as_vector
from aerognc.simulation.aircraft_telemetry import AircraftTelemetry
from aerognc.vehicle.fixed_wing import AircraftControlCommand, project_aircraft_state

TrailMode = Literal["off", "fading", "full"]
TRAIL_MODES: tuple[TrailMode, ...] = ("off", "fading", "full")
TrailColorSource = Literal["constant", "altitude", "airspeed"]
TRAIL_COLOR_SOURCES: tuple[TrailColorSource, ...] = (
    "constant",
    "altitude",
    "airspeed",
)


@dataclass(frozen=True, slots=True)
class RealtimeClockTick:
    """One render callback's fixed-step physics allocation."""

    wall_delta_s: float
    physics_step_count: int
    simulation_duration_s: float
    interpolation_fraction: float
    dropped_simulation_s: float
    measured_fps: float
    achieved_real_time_factor: float


class RealtimeSimulationClock:
    """Convert irregular render callbacks into bounded, fixed physics steps."""

    def __init__(
        self,
        physics_step_s: float,
        real_time_factor: float = 1.0,
        *,
        maximum_catch_up_s: float = 0.5,
        time_source: Callable[[], float] = time.perf_counter,
    ) -> None:
        values = np.asarray([physics_step_s, real_time_factor, maximum_catch_up_s])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("clock step, time factor, and catch-up interval must be positive")
        if maximum_catch_up_s < physics_step_s:
            raise ValueError("clock catch-up interval cannot be shorter than one physics step")
        self.physics_step_s = float(physics_step_s)
        self.real_time_factor = float(real_time_factor)
        self.maximum_catch_up_s = float(maximum_catch_up_s)
        self._time_source = time_source
        self._last_wall_time_s: float | None = None
        self._accumulator_s = 0.0
        self._measured_fps = 0.0
        self._achieved_real_time_factor = 0.0
        self.total_dropped_simulation_s = 0.0

    def resynchronize(self, *, discard_fractional_time: bool = True) -> None:
        """Restart wall timing after pause/focus loss without a catch-up jump."""
        self._last_wall_time_s = self._time_source()
        if discard_fractional_time:
            self._accumulator_s = 0.0

    def tick(self, *, paused: bool = False) -> RealtimeClockTick:
        """Return the fixed physics work due at the current monotonic wall time."""
        now_s = float(self._time_source())
        if not np.isfinite(now_s):
            raise FloatingPointError("monotonic clock returned a non-finite value")
        if self._last_wall_time_s is None:
            self._last_wall_time_s = now_s
            return RealtimeClockTick(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wall_delta_s = max(0.0, now_s - self._last_wall_time_s)
        self._last_wall_time_s = now_s
        if wall_delta_s > 0.0:
            instantaneous_fps = 1.0 / wall_delta_s
            self._measured_fps = (
                instantaneous_fps
                if self._measured_fps == 0.0
                else 0.88 * self._measured_fps + 0.12 * instantaneous_fps
            )
        if paused:
            self._accumulator_s = 0.0
            return RealtimeClockTick(
                wall_delta_s,
                0,
                0.0,
                0.0,
                0.0,
                self._measured_fps,
                self._achieved_real_time_factor,
            )
        requested_s = self._accumulator_s + wall_delta_s * self.real_time_factor
        accepted_s = min(requested_s, self.maximum_catch_up_s)
        dropped_s = max(0.0, requested_s - accepted_s)
        self.total_dropped_simulation_s += dropped_s
        step_count = int(np.floor((accepted_s + 1.0e-12) / self.physics_step_s))
        simulation_duration_s = step_count * self.physics_step_s
        self._accumulator_s = max(0.0, accepted_s - simulation_duration_s)
        if wall_delta_s > 0.0:
            instantaneous_factor = simulation_duration_s / wall_delta_s
            self._achieved_real_time_factor = (
                instantaneous_factor
                if self._achieved_real_time_factor == 0.0
                else 0.88 * self._achieved_real_time_factor + 0.12 * instantaneous_factor
            )
        return RealtimeClockTick(
            wall_delta_s,
            step_count,
            simulation_duration_s,
            self._accumulator_s / self.physics_step_s,
            dropped_s,
            self._measured_fps,
            self._achieved_real_time_factor,
        )


@dataclass(frozen=True, slots=True)
class TrailSettings:
    """Presentation-only trail settings; none can alter propagated state."""

    mode: TrailMode = "fading"
    fading_duration_s: float = 45.0
    color_source: TrailColorSource = "constant"
    maximum_points: int = 12_000
    minimum_sample_interval_s: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in TRAIL_MODES or self.color_source not in TRAIL_COLOR_SOURCES:
            raise ValueError("unsupported aircraft trail mode or color source")
        values = np.asarray([self.fading_duration_s, self.minimum_sample_interval_s])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("trail duration and sample interval must be positive and finite")
        if isinstance(self.maximum_points, bool) or not 2 <= self.maximum_points <= 100_000:
            raise ValueError("trail maximum_points must lie in [2, 100000]")


@dataclass(frozen=True, slots=True)
class TrailView:
    """Renderable trail coordinates, scalar colors, and per-point opacity."""

    positions_display_m: FloatArray
    color_values: FloatArray
    alpha: FloatArray


class FlightTrailBuffer:
    """Decimated fixed-capacity flight trail for stable long-session rendering."""

    def __init__(self, settings: TrailSettings) -> None:
        self.settings = settings
        self._time_s: deque[float] = deque(maxlen=settings.maximum_points)
        self._position: deque[FloatArray] = deque(maxlen=settings.maximum_points)
        self._altitude_m: deque[float] = deque(maxlen=settings.maximum_points)
        self._airspeed_mps: deque[float] = deque(maxlen=settings.maximum_points)

    def __len__(self) -> int:
        return len(self._time_s)

    def append(
        self,
        time_s: float,
        position_display_m: npt.ArrayLike,
        *,
        altitude_m: float,
        airspeed_mps: float,
        force: bool = False,
    ) -> bool:
        """Append one point if the configured deterministic decimation interval elapsed."""
        values = np.asarray([time_s, altitude_m, airspeed_mps], dtype=np.float64)
        position = as_vector(position_display_m, 3, name="trail_position_display_m")
        if not np.all(np.isfinite(values)) or time_s < 0.0 or airspeed_mps < 0.0:
            raise ValueError("trail sample values must be finite and physically ordered")
        if self._time_s and time_s < self._time_s[-1] - 1.0e-12:
            raise ValueError("trail time must be monotonic")
        if (
            not force
            and self._time_s
            and time_s - self._time_s[-1] < self.settings.minimum_sample_interval_s - 1.0e-12
        ):
            return False
        self._time_s.append(float(time_s))
        self._position.append(position.copy())
        self._altitude_m.append(float(altitude_m))
        self._airspeed_mps.append(float(airspeed_mps))
        return True

    def clear(self, *, retain_latest: bool = True) -> None:
        """Clear the visible trail, optionally retaining the current point as its origin."""
        latest = None
        if retain_latest and self._time_s:
            latest = (
                self._time_s[-1],
                self._position[-1].copy(),
                self._altitude_m[-1],
                self._airspeed_mps[-1],
            )
        self._time_s.clear()
        self._position.clear()
        self._altitude_m.clear()
        self._airspeed_mps.clear()
        if latest is not None:
            self.append(
                latest[0],
                latest[1],
                altitude_m=latest[2],
                airspeed_mps=latest[3],
                force=True,
            )

    def view(self, now_s: float, *, mode: TrailMode | None = None) -> TrailView:
        """Return a bounded view for Off, Fading, or Full-session presentation."""
        selected_mode = self.settings.mode if mode is None else mode
        if selected_mode not in TRAIL_MODES:
            raise ValueError(f"trail mode must be one of {TRAIL_MODES}")
        if selected_mode == "off" or not self._time_s:
            empty_positions = np.empty((0, 3), dtype=np.float64)
            return TrailView(empty_positions, np.empty(0), np.empty(0))
        times = np.asarray(self._time_s, dtype=np.float64)
        first = 0
        if selected_mode == "fading":
            first = int(np.searchsorted(times, now_s - self.settings.fading_duration_s))
        selected_times = times[first:]
        positions = np.vstack(tuple(self._position)[first:])
        if self.settings.color_source == "altitude":
            colors = np.asarray(tuple(self._altitude_m)[first:], dtype=np.float64)
        elif self.settings.color_source == "airspeed":
            colors = np.asarray(tuple(self._airspeed_mps)[first:], dtype=np.float64)
        else:
            colors = np.zeros(selected_times.size, dtype=np.float64)
        if selected_mode == "fading":
            alpha = np.clip(
                (selected_times - (now_s - self.settings.fading_duration_s))
                / self.settings.fading_duration_s,
                0.08,
                1.0,
            )
        else:
            alpha = np.ones(selected_times.size, dtype=np.float64)
        return TrailView(positions, colors, np.asarray(alpha, dtype=np.float64))


WarningSeverity = Literal["advisory", "caution", "warning"]


@dataclass(frozen=True, slots=True)
class OperatingEnvelopeLimits:
    """Configurable synthetic cues; these are not certified structural limits."""

    maximum_mach: float = 0.92
    maximum_dynamic_pressure_pa: float = 45_000.0
    maximum_absolute_normal_load_g: float = 4.0
    angle_of_attack_warning_deg: float = 13.5
    low_fuel_fraction: float = 0.10

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.maximum_mach,
                self.maximum_dynamic_pressure_pa,
                self.maximum_absolute_normal_load_g,
                self.angle_of_attack_warning_deg,
                self.low_fuel_fraction,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("synthetic operating-envelope values must be positive and finite")
        if self.low_fuel_fraction >= 1.0:
            raise ValueError("low-fuel fraction must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class FlightWarning:
    """One unambiguous live annunciation."""

    code: str
    message: str
    severity: WarningSeverity


def evaluate_flight_warnings(
    telemetry: AircraftTelemetry,
    command: AircraftControlCommand,
    limits: OperatingEnvelopeLimits,
    *,
    numerical_lag: bool = False,
) -> tuple[FlightWarning, ...]:
    """Evaluate deterministic warning cues from shared telemetry."""
    warnings: list[FlightWarning] = []
    if telemetry.stalled:
        warnings.append(FlightWarning("stall", "STALL - synthetic separated-flow model", "warning"))
    elif telemetry.stall_margin_mps < 0.08 * telemetry.stall_speed_1g_mps:
        warnings.append(FlightWarning("stall_margin", "LOW 1-g STALL MARGIN", "caution"))
    if telemetry.mach > limits.maximum_mach:
        warnings.append(FlightWarning("overspeed", "SYNTHETIC MACH ENVELOPE EXCEEDED", "warning"))
    if telemetry.dynamic_pressure_pa > limits.maximum_dynamic_pressure_pa:
        warnings.append(FlightWarning("high_q", "SYNTHETIC DYNAMIC-PRESSURE LIMIT", "warning"))
    if abs(telemetry.normal_load_g) > limits.maximum_absolute_normal_load_g:
        warnings.append(FlightWarning("high_g", "SYNTHETIC NORMAL-LOAD LIMIT", "warning"))
    if abs(telemetry.angle_of_attack_deg) > limits.angle_of_attack_warning_deg:
        warnings.append(FlightWarning("high_alpha", "ANGLE-OF-ATTACK CAUTION", "caution"))
    if telemetry.fuel_fraction < limits.low_fuel_fraction:
        warnings.append(FlightWarning("low_fuel", "LOW FUEL", "caution"))
    if telemetry.altitude_m >= 100_000.0:
        warnings.append(FlightWarning("space_boundary", "100 km CROSSED - NOT ORBIT", "advisory"))
    if command.rocket_assist:
        warnings.append(FlightWarning("rocket", "RESEARCH ROCKET ASSIST ACTIVE", "advisory"))
    if numerical_lag:
        warnings.append(FlightWarning("lag", "PHYSICS CATCH-UP LIMITED", "caution"))
    return tuple(warnings)


@dataclass(frozen=True, slots=True)
class TouchdownAssessment:
    """Kinematic contact assessment; landing gear and ground roll are not modeled."""

    classification: Literal["soft", "firm", "hard", "unsafe_attitude"]
    vertical_speed_mps: float
    bank_deg: float
    pitch_deg: float
    cross_track_m: float
    note: str = "Kinematic surface contact only; landing gear and ground roll are omitted."


def classify_touchdown(
    telemetry: AircraftTelemetry,
    *,
    runway_cross_track_m: float = 0.0,
) -> TouchdownAssessment:
    """Classify a fictional runway contact using transparent synthetic thresholds."""
    sink_rate = max(0.0, -telemetry.vertical_speed_mps)
    unsafe_attitude = (
        abs(telemetry.roll_deg) > 8.0
        or not -3.0 <= telemetry.pitch_deg <= 15.0
        or abs(runway_cross_track_m) > 30.0
    )
    if unsafe_attitude:
        classification: Literal["soft", "firm", "hard", "unsafe_attitude"] = (
            "unsafe_attitude"
        )
    elif sink_rate <= 1.5:
        classification = "soft"
    elif sink_rate <= 3.0:
        classification = "firm"
    else:
        classification = "hard"
    return TouchdownAssessment(
        classification,
        telemetry.vertical_speed_mps,
        telemetry.roll_deg,
        telemetry.pitch_deg,
        float(runway_cross_track_m),
    )


def interpolate_ground_contact(
    previous_time_s: float,
    previous_state: npt.ArrayLike,
    next_time_s: float,
    next_state: npt.ArrayLike,
    configuration: AircraftSandboxConfiguration,
) -> tuple[float, FloatArray]:
    """Interpolate an above-to-below surface crossing and enforce exact radius."""
    before = as_vector(previous_state, 18, name="previous_aircraft_state")
    after = as_vector(next_state, 18, name="next_aircraft_state")
    values = np.asarray([previous_time_s, next_time_s])
    if not np.all(np.isfinite(values)) or not previous_time_s < next_time_s:
        raise ValueError("ground-contact times must be finite and strictly ordered")
    radius_m = configuration.planet.radius_m
    before_altitude_m = float(np.linalg.norm(before[:3])) - radius_m
    after_altitude_m = float(np.linalg.norm(after[:3])) - radius_m
    if before_altitude_m < 0.0 or after_altitude_m > 0.0:
        raise ValueError("ground interpolation requires an above-to-below crossing")
    denominator = before_altitude_m - after_altitude_m
    fraction = 0.0 if denominator <= 1.0e-15 else before_altitude_m / denominator
    contact = before + fraction * (after - before)
    contact[:3] *= radius_m / np.linalg.norm(contact[:3])
    contact = project_aircraft_state(contact, configuration)
    return float(previous_time_s + fraction * (next_time_s - previous_time_s)), contact


STATE_COLUMN_NAMES: tuple[str, ...] = (
    "x_inertial_m",
    "y_inertial_m",
    "z_inertial_m",
    "vx_inertial_mps",
    "vy_inertial_mps",
    "vz_inertial_mps",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "roll_rate_body_radps",
    "pitch_rate_body_radps",
    "yaw_rate_body_radps",
    "mass_state_kg",
    "aileron_state_rad",
    "elevator_state_rad",
    "rudder_state_rad",
    "throttle_state",
)
COMMAND_COLUMN_NAMES: tuple[str, ...] = (
    "command_roll",
    "command_pitch",
    "command_yaw",
    "command_throttle",
    "command_rocket_assist",
)
TELEMETRY_COLUMN_NAMES: tuple[str, ...] = tuple(
    field.name for field in fields(AircraftTelemetry) if field.name != "time_s"
)


@dataclass(frozen=True, slots=True)
class FlightRecord:
    """One recorder sample backed by an exact propagated state."""

    time_s: float
    state: FloatArray
    command: AircraftControlCommand
    telemetry: AircraftTelemetry
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlightRecorderArtifacts:
    """Paths written by one explicit recorder save."""

    csv_path: Path
    summary_path: Path


class FlightRecorder:
    """Bounded state recorder with CSV replay and JSON engineering debrief."""

    def __init__(self, *, maximum_samples: int = 50_000) -> None:
        if isinstance(maximum_samples, bool) or not 2 <= maximum_samples <= 200_000:
            raise ValueError("flight-recorder capacity must lie in [2, 200000]")
        self.maximum_samples = maximum_samples
        self.records: deque[FlightRecord] = deque(maxlen=maximum_samples)
        self.events: list[dict[str, float | str]] = []
        self.dropped_samples = 0

    def append(
        self,
        state: npt.ArrayLike,
        command: AircraftControlCommand,
        telemetry: AircraftTelemetry,
        warnings: Sequence[FlightWarning] = (),
    ) -> None:
        """Append an exact-state sample after monotonicity and finiteness checks."""
        state_array = as_vector(state, 18, name="recorded_aircraft_state")
        if telemetry.time_s < 0.0 or (
            self.records and telemetry.time_s <= self.records[-1].time_s
        ):
            raise ValueError("recorded flight times must be nonnegative and strictly increasing")
        if len(self.records) == self.maximum_samples:
            self.dropped_samples += 1
        self.records.append(
            FlightRecord(
                telemetry.time_s,
                state_array.copy(),
                command,
                telemetry,
                tuple(warning.code for warning in warnings),
            )
        )

    def add_event(self, name: str, time_s: float, detail: str = "") -> None:
        """Add one bounded, human-readable session event."""
        if not name.strip() or not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("flight-recorder event must be named with nonnegative finite time")
        if len(self.events) >= 1_000:
            raise ValueError("flight recorder event capacity exceeded")
        self.events.append({"name": name.strip(), "time_s": float(time_s), "detail": detail})

    def clear(self) -> None:
        """Clear samples/events for an exact simulator reset."""
        self.records.clear()
        self.events.clear()
        self.dropped_samples = 0

    def summary(self, termination_reason: str = "session saved by user") -> dict[str, object]:
        """Return transparent debrief metrics calculated only from recorded samples."""
        if not self.records:
            raise ValueError("cannot summarize an empty flight recording")
        records = tuple(self.records)
        telemetry = tuple(record.telemetry for record in records)
        time_s = np.asarray([sample.time_s for sample in telemetry])
        north_m = np.asarray([sample.north_m for sample in telemetry])
        east_m = np.asarray([sample.east_m for sample in telemetry])
        segment_distance_m = np.hypot(np.diff(north_m), np.diff(east_m))
        headings_rad = np.unwrap(np.deg2rad([sample.heading_deg for sample in telemetry]))
        turn_rate_degps = (
            np.zeros(1)
            if len(records) == 1
            else np.rad2deg(np.gradient(headings_rad, time_s))
        )
        stalled = np.asarray([sample.stalled for sample in telemetry], dtype=np.float64)
        stall_duration_s = (
            0.0 if len(records) == 1 else float(np.trapezoid(stalled, time_s))
        )
        warning_counts = Counter(
            code for record in records for code in record.warning_codes
        )
        return {
            "schema_version": "1.0",
            "scope": "fictional civilian research-aircraft simulation",
            "sample_count": len(records),
            "dropped_samples": self.dropped_samples,
            "duration_s": float(time_s[-1] - time_s[0]),
            "ground_distance_m": float(np.sum(segment_distance_m)),
            "maximum_altitude_m": max(sample.altitude_m for sample in telemetry),
            "maximum_true_airspeed_mps": max(sample.true_airspeed_mps for sample in telemetry),
            "maximum_mach": max(sample.mach for sample in telemetry),
            "maximum_dynamic_pressure_pa": max(
                sample.dynamic_pressure_pa for sample in telemetry
            ),
            "maximum_absolute_normal_load_g": max(
                abs(sample.normal_load_g) for sample in telemetry
            ),
            "maximum_absolute_turn_rate_degps": float(np.max(np.abs(turn_rate_degps))),
            "minimum_stall_margin_mps": min(sample.stall_margin_mps for sample in telemetry),
            "stall_duration_s": stall_duration_s,
            "fuel_used_kg": records[0].telemetry.mass_kg - records[-1].telemetry.mass_kg,
            "termination_reason": termination_reason,
            "warning_counts": dict(sorted(warning_counts.items())),
            "events": self.events,
            "limitations": [
                "All vehicle data and operating cues are synthetic and not certified.",
                "Touchdown is a kinematic event; landing gear and ground roll are omitted.",
                "Replay interpolates recorded state and telemetry channels without evaluating the plant.",
            ],
        }

    def write(
        self,
        output_directory: str | Path,
        *,
        termination_reason: str = "session saved by user",
    ) -> FlightRecorderArtifacts:
        """Write a bounded CSV state record and a JSON debrief."""
        if not self.records:
            raise ValueError("cannot write an empty flight recording")
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "aircraft_live_recording.csv"
        header = (
            "time_s",
            *COMMAND_COLUMN_NAMES,
            *STATE_COLUMN_NAMES,
            *TELEMETRY_COLUMN_NAMES,
            "warning_codes",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for record in self.records:
                command = record.command
                telemetry_payload = asdict(record.telemetry)
                writer.writerow(
                    [
                        f"{record.time_s:.9g}",
                        f"{command.roll:.17g}",
                        f"{command.pitch:.17g}",
                        f"{command.yaw:.17g}",
                        f"{command.throttle:.17g}",
                        int(command.rocket_assist),
                        *(f"{value:.17g}" for value in record.state),
                        *(
                            f"{float(telemetry_payload[name]):.17g}"
                            for name in TELEMETRY_COLUMN_NAMES
                        ),
                        "|".join(record.warning_codes),
                    ]
                )
        summary_path = output / "aircraft_live_debrief.json"
        summary_path.write_text(
            json.dumps(self.summary(termination_reason), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return FlightRecorderArtifacts(csv_path, summary_path)


@dataclass(frozen=True, slots=True)
class RecordedFlight:
    """Seekable state/telemetry recording; sampling never evaluates the plant."""

    time_s: FloatArray
    state: FloatArray
    command: FloatArray
    telemetry: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.time_s, dtype=np.float64)
        states = np.asarray(self.state, dtype=np.float64)
        commands = np.asarray(self.command, dtype=np.float64)
        telemetry = np.asarray(self.telemetry, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size < 1
            or states.shape != (times.size, 18)
            or commands.shape != (times.size, 5)
            or telemetry.shape != (times.size, len(TELEMETRY_COLUMN_NAMES))
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(states))
            or not np.all(np.isfinite(commands))
            or not np.all(np.isfinite(telemetry))
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("recorded flight arrays have invalid shape, values, or time order")
        object.__setattr__(self, "time_s", times.copy())
        object.__setattr__(self, "state", states.copy())
        object.__setattr__(self, "command", commands.copy())
        object.__setattr__(self, "telemetry", telemetry.copy())

    def _sample_bounds(self, time_s: float) -> tuple[float, int, int, float]:
        """Return clipped time, bracketing rows, and linear interpolation fraction."""
        if not np.isfinite(time_s):
            raise ValueError("replay sample time must be finite")
        selected_time = float(np.clip(time_s, self.time_s[0], self.time_s[-1]))
        upper = int(np.searchsorted(self.time_s, selected_time, side="right"))
        if upper == 0:
            lower = upper = 0
        elif upper >= self.time_s.size:
            lower = upper = self.time_s.size - 1
        else:
            lower = upper - 1
        fraction = (
            0.0
            if lower == upper
            else (selected_time - self.time_s[lower])
            / (self.time_s[upper] - self.time_s[lower])
        )
        return selected_time, lower, upper, float(fraction)

    def sample(self, time_s: float) -> tuple[FloatArray, AircraftControlCommand]:
        """Linearly sample recorded state/commands and renormalize only the quaternion."""
        _selected_time, lower, upper, fraction = self._sample_bounds(time_s)
        state = self.state[lower] + fraction * (self.state[upper] - self.state[lower])
        state[6:10] = normalize_quaternion(state[6:10])
        command_values = self.command[lower] + fraction * (
            self.command[upper] - self.command[lower]
        )
        command = AircraftControlCommand(
            float(command_values[0]),
            float(command_values[1]),
            float(command_values[2]),
            float(command_values[3]),
            bool(command_values[4] >= 0.5),
        )
        return np.asarray(state, dtype=np.float64), command

    def sample_telemetry(self, time_s: float) -> AircraftTelemetry:
        """Linearly sample only telemetry channels persisted in the recorder CSV."""
        selected_time, lower, upper, fraction = self._sample_bounds(time_s)
        values = self.telemetry[lower] + fraction * (
            self.telemetry[upper] - self.telemetry[lower]
        )
        return AircraftTelemetry(selected_time, *(float(value) for value in values))


def load_recorded_flight(path: str | Path) -> RecordedFlight:
    """Load exact propagated state records from a recorder CSV for state-based replay."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "time_s",
            *COMMAND_COLUMN_NAMES,
            *STATE_COLUMN_NAMES,
            *TELEMETRY_COLUMN_NAMES,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "flight replay CSV is missing required state, command, or telemetry columns"
            )
        times: list[float] = []
        states: list[list[float]] = []
        commands: list[list[float]] = []
        telemetry: list[list[float]] = []
        try:
            for row in reader:
                times.append(float(row["time_s"]))
                states.append([float(row[name]) for name in STATE_COLUMN_NAMES])
                commands.append([float(row[name]) for name in COMMAND_COLUMN_NAMES])
                telemetry.append([float(row[name]) for name in TELEMETRY_COLUMN_NAMES])
        except (TypeError, ValueError) as error:
            raise ValueError("flight replay CSV contains a non-numeric required value") from error
    if not times:
        raise ValueError("flight replay CSV contains no samples")
    return RecordedFlight(
        np.asarray(times, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
        np.asarray(commands, dtype=np.float64),
        np.asarray(telemetry, dtype=np.float64),
    )
