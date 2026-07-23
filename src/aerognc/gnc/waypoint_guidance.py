"""Fixed-wing path-following guidance laws.

Guidance converts the active :class:`~aerognc.gnc.path_manager.PathSegment` and the
current :class:`~aerognc.navigation.state.NavigationState` into a structured
:class:`GuidanceCommand`: a commanded ground course, a wind-corrected heading, a
commanded altitude/airspeed/climb-rate, and a coordinated-turn roll feedforward,
plus cross-track and range diagnostics.

Selectable backends (:class:`GuidanceMode`), following Beard & McLain,
*Small Unmanned Aircraft* (2012):

* ``direct_bearing`` / ``line_of_sight`` — steer straight at the leg terminal.
* ``l1_guidance`` — steer toward an L1 look-ahead reference point on the leg.
* ``vector_field`` — nonlinear cross-track vector field (the default; strong
  straight-line convergence).

Loiter (orbit) legs always use the orbit vector field regardless of mode, because
steering "straight at" a circle centre is not loiter following. All laws emit a
*course* command that the cascaded autopilot's course loop tracks; the roll
feedforward improves coordinated-turn tracking. Ground course and body heading are
kept strictly separate (wind correction is applied only to the heading output).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aerognc.gnc.path_manager import FilletSegment, LineSegment, OrbitSegment, PathSegment
from aerognc.mathematics.local_frame import wrap_to_pi
from aerognc.mathematics.vectors import FloatArray
from aerognc.navigation.state import FlightEnvironment, NavigationState


class GuidanceMode(StrEnum):
    """Selectable lateral guidance backend for straight legs."""

    DIRECT_BEARING = "direct_bearing"
    LINE_OF_SIGHT = "line_of_sight"
    L1_GUIDANCE = "l1_guidance"
    VECTOR_FIELD = "vector_field"


@dataclass(frozen=True, slots=True)
class GuidanceGains:
    """Tuning for the guidance laws (SI units)."""

    vector_field_gain_per_m: float = 0.05
    vector_field_max_approach_rad: float = float(np.deg2rad(80.0))
    orbit_gain_per_m: float = 0.02
    l1_distance_m: float = 80.0
    altitude_error_to_climb_rate_per_s: float = 0.4
    max_climb_rate_mps: float = 6.0
    max_roll_feedforward_rad: float = float(np.deg2rad(45.0))
    course_command_rate_limit_radps: float | None = None
    roll_feedforward_rate_limit_radps: float | None = None

    def __post_init__(self) -> None:
        positives = [
            self.vector_field_gain_per_m,
            self.vector_field_max_approach_rad,
            self.orbit_gain_per_m,
            self.l1_distance_m,
            self.altitude_error_to_climb_rate_per_s,
            self.max_climb_rate_mps,
            self.max_roll_feedforward_rad,
        ]
        if not np.all(np.isfinite(positives)) or np.any(np.asarray(positives) <= 0.0):
            raise ValueError("guidance gains must be positive and finite")
        if self.vector_field_max_approach_rad >= 0.5 * np.pi:
            raise ValueError("vector_field_max_approach_rad must be < pi/2")
        for name, value in (
            ("course_command_rate_limit_radps", self.course_command_rate_limit_radps),
            ("roll_feedforward_rate_limit_radps", self.roll_feedforward_rate_limit_radps),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when enabled")


@dataclass(frozen=True, slots=True)
class GuidanceCommand:
    """Structured guidance output consumed by the flight-control system."""

    course_command_rad: float
    heading_command_rad: float
    altitude_command_m: float
    airspeed_command_mps: float
    climb_rate_command_mps: float
    roll_feedforward_rad: float
    cross_track_error_m: float
    distance_to_waypoint_m: float
    along_track_fraction: float


def wind_corrected_heading_rad(
    course_rad: float, wind_ned_mps: FloatArray, airspeed_mps: float
) -> float:
    """Return the body heading that yields the desired ground course under wind.

    The crab angle cancels the cross-course wind component: with ``w_cross`` the
    wind projected onto the left of the course, ``heading = course - asin(w_cross /
    Va)`` (clamped). Falls back to the course when airspeed is negligible or the
    wind exceeds airspeed (unachievable crab).
    """
    if airspeed_mps <= 1.0e-3:
        return wrap_to_pi(course_rad)
    w_cross = -wind_ned_mps[0] * np.sin(course_rad) + wind_ned_mps[1] * np.cos(course_rad)
    ratio = float(np.clip(w_cross / airspeed_mps, -1.0, 1.0))
    return wrap_to_pi(course_rad - float(np.arcsin(ratio)))


class GuidanceLaw(ABC):
    """Common interface for all path-following guidance laws."""

    @abstractmethod
    def update(
        self,
        vehicle_state: NavigationState,
        path_segment: PathSegment,
        environment: FlightEnvironment,
        dt_s: float,
    ) -> GuidanceCommand:
        """Return the guidance command for the active path segment."""


class PathFollowingGuidance(GuidanceLaw):
    """Selectable fixed-wing guidance with straight-line and orbit following."""

    def __init__(
        self, mode: GuidanceMode = GuidanceMode.VECTOR_FIELD, gains: GuidanceGains | None = None
    ) -> None:
        self.mode = mode
        self.gains = gains or GuidanceGains()
        self.reset()

    def reset(self) -> None:
        """Clear command-slew history for deterministic reactivation."""
        self._previous_course_command_rad: float | None = None
        self._previous_roll_feedforward_rad: float | None = None

    def update(
        self,
        vehicle_state: NavigationState,
        path_segment: PathSegment,
        environment: FlightEnvironment,
        dt_s: float,
    ) -> GuidanceCommand:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        position = vehicle_state.position_ned_m
        airspeed = max(vehicle_state.airspeed_mps, 1.0e-3)

        if isinstance(path_segment, OrbitSegment):
            course_cmd, roll_ff = self._circular_course(
                path_segment.center_ned_m[:2],
                path_segment.radius_m,
                path_segment.direction,
                position,
                airspeed,
                environment,
            )
            fraction = 0.0
        elif isinstance(path_segment, FilletSegment):
            course_cmd, roll_ff = self._circular_course(
                path_segment.center_ne_m,
                path_segment.radius_m,
                path_segment.direction,
                position,
                airspeed,
                environment,
            )
            fraction = path_segment.along_track_fraction(position)
        elif isinstance(path_segment, LineSegment):
            course_cmd = self._line_course(path_segment, vehicle_state)
            roll_ff = 0.0
            fraction = path_segment.along_track_fraction(position)
        else:  # pragma: no cover - defensive; only two concrete segments exist
            raise TypeError(f"unsupported path segment: {type(path_segment).__name__}")

        altitude_cmd = -path_segment.commanded_down_m(position)
        altitude_error = altitude_cmd - vehicle_state.altitude_m
        climb_rate_cmd = float(
            np.clip(
                self.gains.altitude_error_to_climb_rate_per_s * altitude_error,
                -self.gains.max_climb_rate_mps,
                self.gains.max_climb_rate_mps,
            )
        )
        course_cmd = self._limited_course(wrap_to_pi(course_cmd), dt_s)
        roll_ff = self._limited_roll_feedforward(
            float(
                np.clip(
                    roll_ff,
                    -self.gains.max_roll_feedforward_rad,
                    self.gains.max_roll_feedforward_rad,
                )
            ),
            dt_s,
        )
        heading_cmd = wind_corrected_heading_rad(course_cmd, environment.wind_ned_mps, airspeed)
        return GuidanceCommand(
            course_command_rad=course_cmd,
            heading_command_rad=heading_cmd,
            altitude_command_m=altitude_cmd,
            airspeed_command_mps=path_segment.commanded_airspeed_mps(position),
            climb_rate_command_mps=climb_rate_cmd,
            roll_feedforward_rad=roll_ff,
            cross_track_error_m=path_segment.cross_track_error_m(position),
            distance_to_waypoint_m=path_segment.horizontal_distance_to_waypoint_m(position),
            along_track_fraction=fraction,
        )

    # -- lateral laws ---------------------------------------------------------
    def _line_course(self, segment: LineSegment, state: NavigationState) -> float:
        direction = segment.horizontal_direction_ne
        path_course = float(np.arctan2(direction[1], direction[0]))
        cross_track = segment.cross_track_error_m(state.position_ned_m)

        if self.mode in (GuidanceMode.DIRECT_BEARING, GuidanceMode.LINE_OF_SIGHT):
            end_ne = segment.end_ned_m[:2]
            delta = end_ne - state.position_ned_m[:2]
            if float(np.linalg.norm(delta)) < 1.0e-6:
                return path_course
            return float(np.arctan2(delta[1], delta[0]))

        if self.mode is GuidanceMode.L1_GUIDANCE:
            reference_ne = self._l1_reference_ne(segment, state.position_ned_m)
            delta = reference_ne - state.position_ned_m[:2]
            if float(np.linalg.norm(delta)) < 1.0e-6:
                return path_course
            return float(np.arctan2(delta[1], delta[0]))

        # Vector field: bend the course toward the path proportionally to XTE.
        approach = self.gains.vector_field_max_approach_rad * (2.0 / np.pi)
        correction = approach * float(np.arctan(self.gains.vector_field_gain_per_m * cross_track))
        return path_course - correction

    def _l1_reference_ne(self, segment: LineSegment, position_ned_m: FloatArray) -> FloatArray:
        """Return the L1 look-ahead point on the line ahead of the projection."""
        start_ne = segment.start_ned_m[:2]
        direction = segment.horizontal_direction_ne
        along_m = float(np.dot(position_ned_m[:2] - start_ne, direction))
        target_along_m = min(along_m + self.gains.l1_distance_m, segment.horizontal_length_m)
        return start_ne + target_along_m * direction

    def _circular_course(
        self,
        center_ne_m: FloatArray,
        radius_m: float,
        direction: int,
        position_ned_m: FloatArray,
        airspeed_mps: float,
        environment: FlightEnvironment,
    ) -> tuple[float, float]:
        delta = position_ned_m[:2] - center_ne_m
        distance_m = float(np.linalg.norm(delta))
        bearing_from_center = float(np.arctan2(delta[1], delta[0]))
        radial_fraction = (distance_m - radius_m) / radius_m
        inward = float(np.arctan(self.gains.orbit_gain_per_m * radius_m * radial_fraction))
        course_cmd = bearing_from_center + direction * (0.5 * np.pi + inward)
        # Coordinated-turn roll to hold the circle: bank into the turn.
        roll_ff = direction * float(
            np.arctan(airspeed_mps**2 / (environment.gravity_mps2 * radius_m))
        )
        return course_cmd, roll_ff

    def _limited_course(self, command_rad: float, dt_s: float) -> float:
        previous = self._previous_course_command_rad
        rate_limit = self.gains.course_command_rate_limit_radps
        if previous is None or rate_limit is None:
            limited = command_rad
        else:
            delta = float(
                np.clip(wrap_to_pi(command_rad - previous), -rate_limit * dt_s, rate_limit * dt_s)
            )
            limited = wrap_to_pi(previous + delta)
        self._previous_course_command_rad = limited
        return limited

    def _limited_roll_feedforward(self, command_rad: float, dt_s: float) -> float:
        previous = self._previous_roll_feedforward_rad
        rate_limit = self.gains.roll_feedforward_rate_limit_radps
        if previous is None or rate_limit is None:
            limited = command_rad
        else:
            limited = float(
                previous + np.clip(command_rad - previous, -rate_limit * dt_s, rate_limit * dt_s)
            )
        self._previous_roll_feedforward_rad = limited
        return limited
