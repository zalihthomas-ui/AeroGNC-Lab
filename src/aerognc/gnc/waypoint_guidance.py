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

from aerognc.gnc.path_manager import LineSegment, OrbitSegment, PathSegment
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
            course_cmd, roll_ff = self._orbit_course(path_segment, position, airspeed, environment)
            fraction = 0.0
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
        heading_cmd = wind_corrected_heading_rad(course_cmd, environment.wind_ned_mps, airspeed)
        return GuidanceCommand(
            course_command_rad=wrap_to_pi(course_cmd),
            heading_command_rad=heading_cmd,
            altitude_command_m=altitude_cmd,
            airspeed_command_mps=path_segment.airspeed_mps,
            climb_rate_command_mps=climb_rate_cmd,
            roll_feedforward_rad=float(
                np.clip(
                    roll_ff,
                    -self.gains.max_roll_feedforward_rad,
                    self.gains.max_roll_feedforward_rad,
                )
            ),
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

    def _orbit_course(
        self,
        segment: OrbitSegment,
        position_ned_m: FloatArray,
        airspeed_mps: float,
        environment: FlightEnvironment,
    ) -> tuple[float, float]:
        delta = position_ned_m[:2] - segment.center_ned_m[:2]
        distance_m = float(np.linalg.norm(delta))
        bearing_from_center = float(np.arctan2(delta[1], delta[0]))
        radial_fraction = (distance_m - segment.radius_m) / segment.radius_m
        inward = float(np.arctan(self.gains.orbit_gain_per_m * segment.radius_m * radial_fraction))
        course_cmd = bearing_from_center + segment.direction * (0.5 * np.pi + inward)
        # Coordinated-turn roll to hold the circle: bank into the turn.
        roll_ff = segment.direction * float(
            np.arctan(airspeed_mps**2 / (environment.gravity_mps2 * segment.radius_m))
        )
        return course_cmd, roll_ff
