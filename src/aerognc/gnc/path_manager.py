"""Fixed-wing path manager: segment geometry and waypoint switching.

The path manager converts an ordered :class:`~aerognc.mission.mission.Mission`
into a sequence of geometric *path segments* expressed in the home-referenced
local NED frame, tracks which segment is active, and decides when the aircraft
has reached a waypoint. Guidance (Phase 4) consumes the active segment; it does
not need to know how switching is decided.

Two segment primitives are provided, following Beard & McLain, *Small Unmanned
Aircraft* (2012):

* :class:`LineSegment` — a straight leg between two points, with linear altitude
  interpolation along the leg.
* :class:`OrbitSegment` — a loiter circle of fixed radius and turn direction.

Fixed-wing aircraft cannot change heading instantaneously, so *fly-through*
waypoints use **half-plane turn anticipation**: the switch happens when the
vehicle crosses the plane through the waypoint whose normal bisects the incoming
and outgoing legs. *Fly-over* waypoints (loiter fixes, return-home, landing)
instead require actual proximity plus altitude agreement.

Fillet-arc geometry (:func:`fillet_geometry`) and the coordinated-turn radius
(:func:`coordinated_turn_radius_m`) are provided as the tested building blocks
for fillet/Dubins arc-following, which is the next planned refinement (see
``TODO.md`` Phase 3.5); the current switcher uses half-plane anticipation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aerognc.mathematics.geodesy import GeodeticPosition
from aerognc.mathematics.local_frame import LocalTangentFrame
from aerognc.mathematics.vectors import FloatArray, as_vector
from aerognc.mission.mission import Mission
from aerognc.mission.waypoint import LoiterDirection, Waypoint, WaypointAction

GRAVITY_MPS2 = 9.80665


def coordinated_turn_radius_m(
    airspeed_mps: float,
    bank_rad: float,
    gravity_mps2: float = GRAVITY_MPS2,
) -> float:
    """Return the coordinated-turn radius ``r = v^2 / (g * tan(bank))`` [m].

    As the bank angle approaches zero the radius diverges; a near-level bank
    therefore returns ``inf`` rather than raising, which is the physically correct
    "straight line" limit. Negative/near-zero banks below ``1e-6`` rad are treated
    as level.
    """
    if not np.isfinite(airspeed_mps) or airspeed_mps <= 0.0:
        raise ValueError("airspeed_mps must be positive and finite")
    if not np.isfinite(bank_rad):
        raise ValueError("bank_rad must be finite")
    if abs(bank_rad) < 1.0e-6:
        return float("inf")
    return float(airspeed_mps**2 / (gravity_mps2 * np.tan(abs(bank_rad))))


class SegmentKind(StrEnum):
    """Discriminator for the concrete path-segment types."""

    LINE = "line"
    ORBIT = "orbit"


class PathSegment(ABC):
    """Interface every path segment exposes to the guidance layer."""

    @property
    @abstractmethod
    def kind(self) -> SegmentKind:
        """Return the segment discriminator."""

    @property
    @abstractmethod
    def waypoint_id(self) -> int:
        """Return the id of the waypoint this segment terminates at."""

    @property
    @abstractmethod
    def airspeed_mps(self) -> float:
        """Return the commanded airspeed while following this segment [m/s]."""

    @abstractmethod
    def commanded_down_m(self, position_ned_m: FloatArray) -> float:
        """Return the commanded NED Down at the vehicle's along-path location [m]."""

    @abstractmethod
    def cross_track_error_m(self, position_ned_m: FloatArray) -> float:
        """Return the signed horizontal cross-track error [m] (positive = right)."""

    @abstractmethod
    def horizontal_distance_to_waypoint_m(self, position_ned_m: FloatArray) -> float:
        """Return the horizontal distance to the segment's terminal point [m]."""


@dataclass(frozen=True, slots=True)
class LineSegment(PathSegment):
    """Straight leg from ``start_ned_m`` to ``end_ned_m`` with altitude ramp."""

    _waypoint_id: int
    start_ned_m: FloatArray
    end_ned_m: FloatArray
    _airspeed_mps: float

    def __post_init__(self) -> None:
        start = as_vector(self.start_ned_m, 3, name="start_ned_m")
        end = as_vector(self.end_ned_m, 3, name="end_ned_m")
        object.__setattr__(self, "start_ned_m", start)
        object.__setattr__(self, "end_ned_m", end)
        if self.horizontal_length_m < 1.0e-6:
            raise ValueError("line segment endpoints must differ horizontally")
        if not np.isfinite(self._airspeed_mps) or self._airspeed_mps <= 0.0:
            raise ValueError("airspeed_mps must be positive and finite")

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.LINE

    @property
    def waypoint_id(self) -> int:
        return self._waypoint_id

    @property
    def airspeed_mps(self) -> float:
        return self._airspeed_mps

    @property
    def horizontal_direction_ne(self) -> FloatArray:
        """Return the unit horizontal (North, East) direction of travel."""
        delta = self.end_ned_m[:2] - self.start_ned_m[:2]
        return delta / float(np.linalg.norm(delta))

    @property
    def horizontal_length_m(self) -> float:
        """Return the horizontal leg length [m]."""
        return float(np.linalg.norm(self.end_ned_m[:2] - self.start_ned_m[:2]))

    def along_track_fraction(self, position_ned_m: FloatArray) -> float:
        """Return the clamped along-track progress in ``[0, 1]``."""
        pos = as_vector(position_ned_m, 3, name="position_ned_m")
        rel = pos[:2] - self.start_ned_m[:2]
        projected_m = float(np.dot(rel, self.horizontal_direction_ne))
        return float(np.clip(projected_m / self.horizontal_length_m, 0.0, 1.0))

    def commanded_down_m(self, position_ned_m: FloatArray) -> float:
        fraction = self.along_track_fraction(position_ned_m)
        start_down = float(self.start_ned_m[2])
        end_down = float(self.end_ned_m[2])
        return start_down + fraction * (end_down - start_down)

    def cross_track_error_m(self, position_ned_m: FloatArray) -> float:
        pos = as_vector(position_ned_m, 3, name="position_ned_m")
        rel = pos[:2] - self.start_ned_m[:2]
        direction = self.horizontal_direction_ne
        # 2-D cross product d x rel; positive when the vehicle is right of travel.
        return float(direction[0] * rel[1] - direction[1] * rel[0])

    def horizontal_distance_to_waypoint_m(self, position_ned_m: FloatArray) -> float:
        pos = as_vector(position_ned_m, 3, name="position_ned_m")
        return float(np.linalg.norm(self.end_ned_m[:2] - pos[:2]))


@dataclass(frozen=True, slots=True)
class OrbitSegment(PathSegment):
    """Loiter circle of fixed radius and turn direction at a fixed altitude."""

    _waypoint_id: int
    center_ned_m: FloatArray
    radius_m: float
    direction: int  # +1 clockwise, -1 counter-clockwise (viewed from above)
    _airspeed_mps: float

    def __post_init__(self) -> None:
        center = as_vector(self.center_ned_m, 3, name="center_ned_m")
        object.__setattr__(self, "center_ned_m", center)
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("orbit radius_m must be positive and finite")
        if self.direction not in (-1, 1):
            raise ValueError("orbit direction must be +1 (CW) or -1 (CCW)")
        if not np.isfinite(self._airspeed_mps) or self._airspeed_mps <= 0.0:
            raise ValueError("airspeed_mps must be positive and finite")

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.ORBIT

    @property
    def waypoint_id(self) -> int:
        return self._waypoint_id

    @property
    def airspeed_mps(self) -> float:
        return self._airspeed_mps

    def radial_distance_m(self, position_ned_m: FloatArray) -> float:
        """Return the horizontal distance from the orbit centre [m]."""
        pos = as_vector(position_ned_m, 3, name="position_ned_m")
        return float(np.linalg.norm(pos[:2] - self.center_ned_m[:2]))

    def angle_rad(self, position_ned_m: FloatArray) -> float:
        """Return the bearing of the vehicle from the orbit centre [rad]."""
        pos = as_vector(position_ned_m, 3, name="position_ned_m")
        delta = pos[:2] - self.center_ned_m[:2]
        return float(np.arctan2(delta[1], delta[0]))

    def commanded_down_m(self, position_ned_m: FloatArray) -> float:
        return float(self.center_ned_m[2])

    def cross_track_error_m(self, position_ned_m: FloatArray) -> float:
        # Signed radial error: positive outside the circle, negative inside.
        return self.radial_distance_m(position_ned_m) - self.radius_m

    def horizontal_distance_to_waypoint_m(self, position_ned_m: FloatArray) -> float:
        return self.radial_distance_m(position_ned_m)


@dataclass(frozen=True, slots=True)
class FilletGeometry:
    """Circular fillet joining two legs at a corner (turn-anticipation arc)."""

    center_ne_m: FloatArray
    radius_m: float
    entry_ne_m: FloatArray
    exit_ne_m: FloatArray
    turn_angle_rad: float
    direction: int  # +1 clockwise, -1 counter-clockwise (viewed from above)


def fillet_geometry(
    previous_ne_m: FloatArray,
    corner_ne_m: FloatArray,
    next_ne_m: FloatArray,
    radius_m: float,
) -> FilletGeometry:
    """Return the fillet arc that rounds the corner with the given radius.

    Works purely in the horizontal (North, East) plane. The arc is tangent to
    both legs; ``entry``/``exit`` are the tangent points and ``turn_angle_rad`` is
    the course change. Raises if the legs are collinear (no turn to round).
    """
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive and finite")
    previous_ne = as_vector(previous_ne_m, 2, name="previous_ne_m")
    corner_ne = as_vector(corner_ne_m, 2, name="corner_ne_m")
    next_ne = as_vector(next_ne_m, 2, name="next_ne_m")

    incoming = corner_ne - previous_ne
    outgoing = next_ne - corner_ne
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm < 1.0e-9 or outgoing_norm < 1.0e-9:
        raise ValueError("fillet legs must have non-zero length")
    q_in = incoming / incoming_norm
    q_out = outgoing / outgoing_norm

    cos_turn = float(np.clip(np.dot(q_in, q_out), -1.0, 1.0))
    turn_angle_rad = float(np.arccos(cos_turn))
    if turn_angle_rad < 1.0e-6:
        raise ValueError("legs are collinear; no fillet is required")

    # Interior wedge is between the reversed incoming ray and the outgoing ray.
    bisector = -q_in + q_out
    bisector_norm = float(np.linalg.norm(bisector))
    if bisector_norm < 1.0e-9:
        raise ValueError("legs reverse exactly; fillet is undefined")
    bisector /= bisector_norm
    half_wedge_rad = 0.5 * (np.pi - turn_angle_rad)
    center_distance_m = radius_m / np.sin(half_wedge_rad)
    tangent_distance_m = radius_m / np.tan(half_wedge_rad)

    center_ne = corner_ne + center_distance_m * bisector
    entry_ne = corner_ne - tangent_distance_m * q_in
    exit_ne = corner_ne + tangent_distance_m * q_out
    # In NED (North first, East second), rotating from North toward East is a
    # clockwise/right turn viewed from above, giving a positive 2-D cross product.
    cross = float(q_in[0] * q_out[1] - q_in[1] * q_out[0])
    direction = 1 if cross > 0.0 else -1  # right turn = clockwise = +1
    return FilletGeometry(
        center_ne_m=center_ne,
        radius_m=float(radius_m),
        entry_ne_m=entry_ne,
        exit_ne_m=exit_ne,
        turn_angle_rad=turn_angle_rad,
        direction=direction,
    )


@dataclass(frozen=True, slots=True)
class PathManagerConfig:
    """Tunable switching behaviour for the path manager."""

    min_dwell_s: float = 0.0
    require_altitude_on_flyover: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_dwell_s) or self.min_dwell_s < 0.0:
            raise ValueError("min_dwell_s must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class _Leg:
    """Internal per-segment switching metadata."""

    segment: PathSegment
    acceptance_radius_m: float
    altitude_tolerance_m: float
    fly_over: bool
    switch_normal_ne: FloatArray | None = None
    loiter_duration_s: float | None = None


class MissionPhase(StrEnum):
    """Coarse path-following phase, exposed to the UI and logger."""

    NAVIGATE = "navigate"
    LOITER = "loiter"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class PathManagerStatus:
    """Immutable snapshot returned by :meth:`PathManager.update`."""

    active_index: int
    active_waypoint_id: int
    active_segment: PathSegment
    phase: MissionPhase
    distance_to_waypoint_m: float
    cross_track_error_m: float
    commanded_down_m: float
    along_track_fraction: float
    just_advanced: bool
    mission_complete: bool


class PathManager:
    """Sequences path segments and detects waypoint arrival for a mission."""

    def __init__(self, legs: list[_Leg], config: PathManagerConfig | None = None) -> None:
        if not legs:
            raise ValueError("path manager requires at least one leg")
        self._legs = legs
        self.config = config or PathManagerConfig()
        self.reset()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_mission(
        cls,
        mission: Mission,
        *,
        frame: LocalTangentFrame | None = None,
        config: PathManagerConfig | None = None,
    ) -> "PathManager":
        """Build a path manager from a validated mission.

        Home is the local NED origin. Each waypoint becomes a straight leg; loiter
        and hold waypoints add an orbit after their approach leg. ``return_home``
        targets the home horizontal position at the waypoint altitude.
        """
        mission.validate()
        active_frame = frame if frame is not None else mission.local_frame()
        legs: list[_Leg] = []
        start_ned = np.zeros(3, dtype=np.float64)  # home origin

        for waypoint in mission.waypoints:
            airspeed = mission.resolved_airspeed_mps(waypoint)
            acceptance = mission.resolved_acceptance_radius_m(waypoint)
            altitude_tol = mission.resolved_altitude_tolerance_m(waypoint)
            if waypoint.action is WaypointAction.RETURN_HOME:
                target_ned = active_frame.geodetic_to_ned(
                    _home_geodetic_at_altitude(mission, waypoint)
                )
            else:
                target_ned = mission.waypoint_ned_m(waypoint, active_frame)

            approach_is_fly_over = waypoint.action not in (
                WaypointAction.FLY_THROUGH,
                WaypointAction.TURN,
            )
            legs.append(
                _Leg(
                    segment=LineSegment(waypoint.id, start_ned, target_ned, airspeed),
                    acceptance_radius_m=acceptance,
                    altitude_tolerance_m=altitude_tol,
                    fly_over=approach_is_fly_over,
                )
            )
            if waypoint.action in (WaypointAction.LOITER, WaypointAction.HOLD):
                assert waypoint.loiter_radius_m is not None  # enforced by validation
                direction = (
                    1 if waypoint.loiter_direction is LoiterDirection.CLOCKWISE else -1
                )
                legs.append(
                    _Leg(
                        segment=OrbitSegment(
                            waypoint.id, target_ned, waypoint.loiter_radius_m, direction, airspeed
                        ),
                        acceptance_radius_m=acceptance,
                        altitude_tolerance_m=altitude_tol,
                        fly_over=True,
                        loiter_duration_s=waypoint.loiter_duration_s,
                    )
                )
            start_ned = target_ned

        _assign_switch_normals(legs)
        return cls(legs, config)

    # -- runtime --------------------------------------------------------------
    def reset(self) -> None:
        """Reset to the first leg and clear timers."""
        self._active = 0
        self._dwell_s = 0.0
        self._loiter_elapsed_s = 0.0
        self._complete = False

    @property
    def legs(self) -> tuple[_Leg, ...]:
        """Return the immutable leg tuple (segment + switching metadata)."""
        return tuple(self._legs)

    @property
    def segments(self) -> tuple[PathSegment, ...]:
        """Return the ordered path segments for the guidance layer."""
        return tuple(leg.segment for leg in self._legs)

    def update(self, position_ned_m: FloatArray, dt_s: float) -> PathManagerStatus:
        """Advance the switching logic by one step and return the status."""
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        position = as_vector(position_ned_m, 3, name="position_ned_m")
        leg = self._legs[self._active]

        just_advanced = False
        if not self._complete:
            arrived = self._arrived(leg, position, dt_s)
            if arrived:
                just_advanced = self._advance()

        # Re-read the (possibly new) active leg for the returned status.
        leg = self._legs[self._active]
        segment = leg.segment
        fraction = (
            segment.along_track_fraction(position)
            if isinstance(segment, LineSegment)
            else 0.0
        )
        phase = self._phase(leg)
        return PathManagerStatus(
            active_index=self._active,
            active_waypoint_id=segment.waypoint_id,
            active_segment=segment,
            phase=phase,
            distance_to_waypoint_m=segment.horizontal_distance_to_waypoint_m(position),
            cross_track_error_m=segment.cross_track_error_m(position),
            commanded_down_m=segment.commanded_down_m(position),
            along_track_fraction=fraction,
            just_advanced=just_advanced,
            mission_complete=self._complete,
        )

    def index_of_waypoint(self, waypoint_id: int) -> int | None:
        """Return the first leg index terminating at the given waypoint id."""
        for index, leg in enumerate(self._legs):
            if leg.segment.waypoint_id == waypoint_id:
                return index
        return None

    def force_active_index(self, index: int) -> None:
        """Jump the active leg (e.g. for return-to-home); clears timers."""
        if not 0 <= index < len(self._legs):
            raise ValueError(f"leg index {index} out of range [0, {len(self._legs)})")
        self._active = index
        self._dwell_s = 0.0
        self._loiter_elapsed_s = 0.0
        self._complete = False

    def planned_path_ned(self) -> FloatArray:
        """Return the planned polyline vertices (home + leg terminals) in NED."""
        vertices: list[FloatArray] = [np.zeros(3, dtype=np.float64)]
        for leg in self._legs:
            segment = leg.segment
            if isinstance(segment, LineSegment):
                vertices.append(segment.end_ned_m)
            elif isinstance(segment, OrbitSegment):
                vertices.append(segment.center_ned_m)
        return np.asarray(vertices, dtype=np.float64)

    def loiter_circles(self) -> list[OrbitSegment]:
        """Return the loiter/orbit segments (for map rendering)."""
        return [leg.segment for leg in self._legs if isinstance(leg.segment, OrbitSegment)]

    # -- internals ------------------------------------------------------------
    def _arrived(self, leg: _Leg, position: FloatArray, dt_s: float) -> bool:
        segment = leg.segment
        if isinstance(segment, OrbitSegment):
            return self._orbit_arrived(leg, segment, position, dt_s)
        assert isinstance(segment, LineSegment)
        return self._line_arrived(leg, segment, position, dt_s)

    def _line_arrived(
        self, leg: _Leg, segment: LineSegment, position: FloatArray, dt_s: float
    ) -> bool:
        distance_m = segment.horizontal_distance_to_waypoint_m(position)
        horizontal_ok = distance_m <= leg.acceptance_radius_m
        altitude_ok = (
            abs(float(position[2]) - float(segment.end_ned_m[2])) <= leg.altitude_tolerance_m
        )
        proximity_reached = horizontal_ok and altitude_ok

        if leg.fly_over or leg.switch_normal_ne is None:
            predicate = proximity_reached
        else:
            to_position = position[:2] - segment.end_ned_m[:2]
            crossed_plane = float(np.dot(to_position, leg.switch_normal_ne)) >= 0.0
            predicate = crossed_plane or proximity_reached

        if predicate:
            self._dwell_s += dt_s
        else:
            self._dwell_s = 0.0
        return predicate and self._dwell_s >= self.config.min_dwell_s

    def _orbit_arrived(
        self, leg: _Leg, segment: OrbitSegment, position: FloatArray, dt_s: float
    ) -> bool:
        on_circle = abs(segment.cross_track_error_m(position)) <= leg.acceptance_radius_m
        altitude_ok = (
            abs(float(position[2]) - float(segment.center_ned_m[2])) <= leg.altitude_tolerance_m
        )
        if on_circle and altitude_ok:
            self._loiter_elapsed_s += dt_s
        # An indefinite hold (no duration) never auto-advances; it awaits an
        # explicit mission command (pause/RTH/skip) handled by the mission manager.
        if leg.loiter_duration_s is None:
            return False
        return self._loiter_elapsed_s >= leg.loiter_duration_s

    def _advance(self) -> bool:
        self._dwell_s = 0.0
        self._loiter_elapsed_s = 0.0
        if self._active + 1 >= len(self._legs):
            self._complete = True
            return True
        self._active += 1
        return True

    def _phase(self, leg: _Leg) -> MissionPhase:
        if self._complete:
            return MissionPhase.COMPLETE
        if isinstance(leg.segment, OrbitSegment):
            return MissionPhase.LOITER
        return MissionPhase.NAVIGATE


def _assign_switch_normals(legs: list[_Leg]) -> None:
    """Fill in half-plane switch normals for fly-through line->line corners."""
    for index in range(len(legs) - 1):
        current = legs[index]
        following = legs[index + 1]
        if current.fly_over:
            continue
        if not isinstance(current.segment, LineSegment):
            continue
        if not isinstance(following.segment, LineSegment):
            continue
        incoming = current.segment.horizontal_direction_ne
        outgoing = following.segment.horizontal_direction_ne
        bisector = incoming + outgoing
        norm = float(np.linalg.norm(bisector))
        if norm < 1.0e-6:
            continue  # legs reverse; fall back to proximity switching
        legs[index] = _Leg(
            segment=current.segment,
            acceptance_radius_m=current.acceptance_radius_m,
            altitude_tolerance_m=current.altitude_tolerance_m,
            fly_over=current.fly_over,
            switch_normal_ne=bisector / norm,
            loiter_duration_s=current.loiter_duration_s,
        )


def _home_geodetic_at_altitude(mission: Mission, waypoint: Waypoint) -> GeodeticPosition:
    """Return the home lat/lon at a return-home waypoint's resolved altitude."""
    return GeodeticPosition(
        latitude_rad=float(np.deg2rad(mission.home.latitude_deg)),
        longitude_rad=float(np.deg2rad(mission.home.longitude_deg)),
        altitude_m=mission.absolute_altitude_m(waypoint),
    )


# Reserved for the planned fillet/Dubins arc-following refinement (TODO 3.5):
# `fillet_geometry` above already produces the arc; wiring it into the switcher
# as inserted OrbitSegments is the next step and is intentionally not stubbed.
