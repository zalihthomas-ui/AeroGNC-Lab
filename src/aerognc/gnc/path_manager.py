"""Fixed-wing path manager: segment geometry and waypoint switching.

The path manager converts an ordered :class:`~aerognc.mission.mission.Mission`
into a sequence of geometric *path segments* expressed in the home-referenced
local NED frame, tracks which segment is active, and decides when the aircraft
has reached a waypoint. Guidance (Phase 4) consumes the active segment; it does
not need to know how switching is decided.

Three segment primitives are provided, following Beard & McLain, *Small Unmanned
Aircraft* (2012):

* :class:`LineSegment` -- a straight leg between two points, with linear altitude
  interpolation along the leg.
* :class:`FilletSegment` -- a finite circular arc that rounds a fly-through corner.
* :class:`OrbitSegment` -- a loiter circle of fixed radius and turn direction.

Fixed-wing aircraft cannot change heading instantaneously, so *fly-through*
waypoints can use coordinated-turn fillets sized by airspeed, bank limit, and
available leg length. The switcher advances at tangent half-planes. Loiter
approach and departure lines are also moved to direction-consistent orbit tangent
points, and loiter exit waits for the aircraft to reach the departure region.
*Fly-over* waypoints without a rounded transition instead require actual
proximity plus altitude agreement.
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
    FILLET = "fillet"


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

    def commanded_airspeed_mps(self, position_ned_m: FloatArray) -> float:
        """Return the local airspeed reference; constant for ordinary segments."""
        as_vector(position_ned_m, 3, name="position_ned_m")
        return self.airspeed_mps


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
class FilletSegment(PathSegment):
    """Finite tangent circular arc joining two fly-through line segments."""

    _waypoint_id: int
    center_ne_m: FloatArray
    radius_m: float
    direction: int
    entry_ned_m: FloatArray
    exit_ned_m: FloatArray
    turn_angle_rad: float
    entry_airspeed_mps: float
    exit_airspeed_mps: float

    def __post_init__(self) -> None:
        center = as_vector(self.center_ne_m, 2, name="center_ne_m")
        entry = as_vector(self.entry_ned_m, 3, name="entry_ned_m")
        exit_point = as_vector(self.exit_ned_m, 3, name="exit_ned_m")
        values = np.asarray(
            [
                self.radius_m,
                self.turn_angle_rad,
                self.entry_airspeed_mps,
                self.exit_airspeed_mps,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("fillet radius, turn angle, and airspeeds must be positive")
        if self.turn_angle_rad >= np.pi:
            raise ValueError("fillet turn angle must be below pi")
        if self.direction not in (-1, 1):
            raise ValueError("fillet direction must be +1 or -1")
        radii = np.asarray(
            [
                np.linalg.norm(entry[:2] - center),
                np.linalg.norm(exit_point[:2] - center),
            ]
        )
        if not np.allclose(radii, self.radius_m, rtol=1.0e-8, atol=1.0e-8):
            raise ValueError("fillet entry and exit must lie on the declared circle")
        object.__setattr__(self, "center_ne_m", center)
        object.__setattr__(self, "entry_ned_m", entry)
        object.__setattr__(self, "exit_ned_m", exit_point)

    @property
    def kind(self) -> SegmentKind:
        return SegmentKind.FILLET

    @property
    def waypoint_id(self) -> int:
        return self._waypoint_id

    @property
    def airspeed_mps(self) -> float:
        return self.entry_airspeed_mps

    @property
    def entry_angle_rad(self) -> float:
        delta = self.entry_ned_m[:2] - self.center_ne_m
        return float(np.arctan2(delta[1], delta[0]))

    @property
    def exit_angle_rad(self) -> float:
        delta = self.exit_ned_m[:2] - self.center_ne_m
        return float(np.arctan2(delta[1], delta[0]))

    @property
    def exit_tangent_ne(self) -> FloatArray:
        angle = self.exit_angle_rad
        return self.direction * np.asarray([-np.sin(angle), np.cos(angle)])

    @property
    def entry_tangent_ne(self) -> FloatArray:
        angle = self.entry_angle_rad
        return self.direction * np.asarray([-np.sin(angle), np.cos(angle)])

    def radial_distance_m(self, position_ned_m: FloatArray) -> float:
        position = as_vector(position_ned_m, 3, name="position_ned_m")
        return float(np.linalg.norm(position[:2] - self.center_ne_m))

    def along_track_fraction(self, position_ned_m: FloatArray) -> float:
        position = as_vector(position_ned_m, 3, name="position_ned_m")
        delta = position[:2] - self.center_ne_m
        angle = float(np.arctan2(delta[1], delta[0]))
        if self.direction > 0:
            swept = (angle - self.entry_angle_rad) % (2.0 * np.pi)
        else:
            swept = (self.entry_angle_rad - angle) % (2.0 * np.pi)
        return float(np.clip(swept / self.turn_angle_rad, 0.0, 1.0))

    def commanded_down_m(self, position_ned_m: FloatArray) -> float:
        fraction = self.along_track_fraction(position_ned_m)
        return float(self.entry_ned_m[2] + fraction * (self.exit_ned_m[2] - self.entry_ned_m[2]))

    def commanded_airspeed_mps(self, position_ned_m: FloatArray) -> float:
        fraction = self.along_track_fraction(position_ned_m)
        return float(
            self.entry_airspeed_mps + fraction * (self.exit_airspeed_mps - self.entry_airspeed_mps)
        )

    def cross_track_error_m(self, position_ned_m: FloatArray) -> float:
        return self.radial_distance_m(position_ned_m) - self.radius_m

    def horizontal_distance_to_waypoint_m(self, position_ned_m: FloatArray) -> float:
        position = as_vector(position_ned_m, 3, name="position_ned_m")
        return float(np.linalg.norm(position[:2] - self.exit_ned_m[:2]))

    def sample_ned(self, count: int = 16) -> FloatArray:
        """Return deterministic arc samples for planned-path rendering."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("fillet sample count must be an integer of at least two")
        fraction = np.linspace(0.0, 1.0, count)
        angle = self.entry_angle_rad + self.direction * self.turn_angle_rad * fraction
        return np.column_stack(
            (
                self.center_ne_m[0] + self.radius_m * np.cos(angle),
                self.center_ne_m[1] + self.radius_m * np.sin(angle),
                self.entry_ned_m[2] + fraction * (self.exit_ned_m[2] - self.entry_ned_m[2]),
            )
        )


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
    fillet_bank_rad: float | None = None
    fillet_max_radius_m: float = 200.0
    fillet_leg_fraction: float = 0.45
    minimum_fillet_radius_m: float = 5.0
    tangent_orbit_transitions: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.require_altitude_on_flyover, bool) or not isinstance(
            self.tangent_orbit_transitions, bool
        ):
            raise ValueError("path-manager switching flags must be boolean")
        if not np.isfinite(self.min_dwell_s) or self.min_dwell_s < 0.0:
            raise ValueError("min_dwell_s must be nonnegative and finite")
        values = np.asarray(
            [
                self.fillet_max_radius_m,
                self.fillet_leg_fraction,
                self.minimum_fillet_radius_m,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("fillet radius/fraction settings must be positive and finite")
        if self.fillet_leg_fraction >= 0.5:
            raise ValueError("fillet_leg_fraction must be below 0.5")
        if self.minimum_fillet_radius_m > self.fillet_max_radius_m:
            raise ValueError("minimum fillet radius must not exceed its maximum")
        if self.fillet_bank_rad is not None and (
            not np.isfinite(self.fillet_bank_rad) or not 0.0 < self.fillet_bank_rad < 0.5 * np.pi
        ):
            raise ValueError("fillet_bank_rad must lie in (0, pi/2) when enabled")


@dataclass(frozen=True, slots=True)
class _Leg:
    """Internal per-segment switching metadata."""

    segment: PathSegment
    acceptance_radius_m: float
    altitude_tolerance_m: float
    fly_over: bool
    switch_normal_ne: FloatArray | None = None
    loiter_duration_s: float | None = None
    orbit_exit_ne_m: FloatArray | None = None


class MissionPhase(StrEnum):
    """Coarse path-following phase, exposed to the UI and logger."""

    NAVIGATE = "navigate"
    TURN = "turn"
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
        initial_position_ned_m: FloatArray | None = None,
    ) -> "PathManager":
        """Build a path manager from a validated mission.

        Home is the local NED origin. The first leg starts at
        ``initial_position_ned_m`` when an air-start is supplied, otherwise at home.
        Each waypoint becomes a straight leg; loiter and hold waypoints add an orbit
        after their approach leg. ``return_home`` targets the home horizontal
        position at the waypoint altitude.
        """
        mission.validate()
        active_config = config or PathManagerConfig()
        active_frame = frame if frame is not None else mission.local_frame()
        legs: list[_Leg] = []
        start_ned = (
            np.zeros(3, dtype=np.float64)
            if initial_position_ned_m is None
            else as_vector(initial_position_ned_m, 3, name="initial_position_ned_m")
        )

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
                direction = 1 if waypoint.loiter_direction is LoiterDirection.CLOCKWISE else -1
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

        if active_config.tangent_orbit_transitions:
            legs = _insert_orbit_tangencies(legs)
        if active_config.fillet_bank_rad is not None:
            legs = _insert_fillet_legs(legs, active_config)
        _assign_switch_normals(legs)
        return cls(legs, active_config)

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
            if isinstance(segment, (LineSegment, FilletSegment))
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
        """Return line vertices and sampled fillet arcs in NED."""
        first = self._legs[0].segment
        if not isinstance(first, LineSegment):  # pragma: no cover - construction invariant
            raise RuntimeError("path must begin with a line segment")
        vertices: list[FloatArray] = [first.start_ned_m.copy()]
        for leg in self._legs:
            segment = leg.segment
            if isinstance(segment, LineSegment):
                vertices.append(segment.end_ned_m)
            elif isinstance(segment, FilletSegment):
                vertices.extend(segment.sample_ned()[1:])
            elif isinstance(segment, OrbitSegment):
                vertices.append(segment.center_ned_m)
        return np.asarray(vertices, dtype=np.float64)

    def loiter_circles(self) -> list[OrbitSegment]:
        """Return the loiter/orbit segments (for map rendering)."""
        return [leg.segment for leg in self._legs if isinstance(leg.segment, OrbitSegment)]

    def fillet_arcs(self) -> list[FilletSegment]:
        """Return inserted finite turn arcs for rendering and verification."""
        return [leg.segment for leg in self._legs if isinstance(leg.segment, FilletSegment)]

    # -- internals ------------------------------------------------------------
    def _arrived(self, leg: _Leg, position: FloatArray, dt_s: float) -> bool:
        segment = leg.segment
        if isinstance(segment, FilletSegment):
            return self._fillet_arrived(leg, segment, position, dt_s)
        if isinstance(segment, OrbitSegment):
            return self._orbit_arrived(leg, segment, position, dt_s)
        assert isinstance(segment, LineSegment)
        return self._line_arrived(leg, segment, position, dt_s)

    def _fillet_arrived(
        self,
        leg: _Leg,
        segment: FilletSegment,
        position: FloatArray,
        dt_s: float,
    ) -> bool:
        normal = segment.exit_tangent_ne if leg.switch_normal_ne is None else leg.switch_normal_ne
        crossed_exit = float(np.dot(position[:2] - segment.exit_ned_m[:2], normal)) >= 0.0
        if crossed_exit:
            self._dwell_s += dt_s
        else:
            self._dwell_s = 0.0
        return crossed_exit and self._dwell_s >= self.config.min_dwell_s

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
        if self._loiter_elapsed_s < leg.loiter_duration_s:
            return False
        if leg.orbit_exit_ne_m is None:
            return True
        return (
            float(np.linalg.norm(position[:2] - leg.orbit_exit_ne_m)) <= leg.acceptance_radius_m
            and altitude_ok
        )

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
        if isinstance(leg.segment, FilletSegment):
            return MissionPhase.TURN
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
            orbit_exit_ne_m=current.orbit_exit_ne_m,
        )


def _insert_fillet_legs(legs: list[_Leg], config: PathManagerConfig) -> list[_Leg]:
    """Insert feasible tangent arcs at fly-through line-to-line corners."""
    if config.fillet_bank_rad is None:
        return legs
    output: list[_Leg] = []
    start_overrides: dict[int, FloatArray] = {}
    for index, original in enumerate(legs):
        current = original
        if index in start_overrides and isinstance(original.segment, LineSegment):
            line = original.segment
            current = _replace_leg_segment(
                original,
                LineSegment(
                    line.waypoint_id,
                    start_overrides[index],
                    line.end_ned_m,
                    line.airspeed_mps,
                ),
            )
        if index + 1 >= len(legs):
            output.append(current)
            continue
        following = legs[index + 1]
        if (
            current.fly_over
            or not isinstance(current.segment, LineSegment)
            or not isinstance(following.segment, LineSegment)
        ):
            output.append(current)
            continue
        inserted = _fillet_at_corner(current, following, config)
        if inserted is None:
            output.append(current)
            continue
        shortened, fillet, next_start = inserted
        output.extend((shortened, fillet))
        start_overrides[index + 1] = next_start
    return output


def _insert_orbit_tangencies(legs: list[_Leg]) -> list[_Leg]:
    """Move loiter approach/departure line endpoints to tangent circle points."""
    adjusted = list(legs)
    for index, leg in enumerate(tuple(adjusted)):
        orbit = leg.segment
        if not isinstance(orbit, OrbitSegment):
            continue
        entry_ne: FloatArray | None = None
        if index > 0 and isinstance(adjusted[index - 1].segment, LineSegment):
            approach_leg = adjusted[index - 1]
            approach = approach_leg.segment
            assert isinstance(approach, LineSegment)
            entry_ne = _orbit_tangent_point(
                orbit.center_ned_m[:2],
                approach.start_ned_m[:2],
                orbit.radius_m,
                orbit.direction,
                entering=True,
            )
            if entry_ne is not None:
                entry = np.asarray([*entry_ne, orbit.center_ned_m[2]], dtype=np.float64)
                adjusted[index - 1] = _replace_leg_segment(
                    approach_leg,
                    LineSegment(
                        approach.waypoint_id,
                        approach.start_ned_m,
                        entry,
                        approach.airspeed_mps,
                    ),
                )

        exit_ne: FloatArray | None = None
        if index + 1 < len(adjusted) and isinstance(adjusted[index + 1].segment, LineSegment):
            departure_leg = adjusted[index + 1]
            departure = departure_leg.segment
            assert isinstance(departure, LineSegment)
            exit_ne = _orbit_tangent_point(
                orbit.center_ned_m[:2],
                departure.end_ned_m[:2],
                orbit.radius_m,
                orbit.direction,
                entering=False,
            )
            if exit_ne is not None:
                exit_point = np.asarray([*exit_ne, orbit.center_ned_m[2]], dtype=np.float64)
                adjusted[index + 1] = _replace_leg_segment(
                    departure_leg,
                    LineSegment(
                        departure.waypoint_id,
                        exit_point,
                        departure.end_ned_m,
                        departure.airspeed_mps,
                    ),
                )
        adjusted[index] = _Leg(
            segment=orbit,
            acceptance_radius_m=leg.acceptance_radius_m,
            altitude_tolerance_m=leg.altitude_tolerance_m,
            fly_over=leg.fly_over,
            switch_normal_ne=leg.switch_normal_ne,
            loiter_duration_s=leg.loiter_duration_s,
            orbit_exit_ne_m=exit_ne,
        )
    return adjusted


def _orbit_tangent_point(
    center_ne_m: FloatArray,
    external_ne_m: FloatArray,
    radius_m: float,
    direction: int,
    *,
    entering: bool,
) -> FloatArray | None:
    delta = external_ne_m - center_ne_m
    distance_m = float(np.linalg.norm(delta))
    if distance_m <= radius_m + 1.0e-9:
        return None
    base_angle = float(np.arctan2(delta[1], delta[0]))
    offset = float(np.arccos(radius_m / distance_m))
    best_point: FloatArray | None = None
    best_alignment = -np.inf
    for angle in (base_angle - offset, base_angle + offset):
        point = center_ne_m + radius_m * np.asarray([np.cos(angle), np.sin(angle)])
        travel = point - external_ne_m if entering else external_ne_m - point
        travel /= float(np.linalg.norm(travel))
        tangent = direction * np.asarray([-np.sin(angle), np.cos(angle)])
        alignment = float(np.dot(travel, tangent))
        if alignment > best_alignment:
            best_point = point
            best_alignment = alignment
    return best_point


def _fillet_at_corner(
    current: _Leg,
    following: _Leg,
    config: PathManagerConfig,
) -> tuple[_Leg, _Leg, FloatArray] | None:
    if config.fillet_bank_rad is None:
        return None
    assert isinstance(current.segment, LineSegment)
    assert isinstance(following.segment, LineSegment)
    line = current.segment
    next_line = following.segment
    try:
        unit_geometry = fillet_geometry(
            line.start_ned_m[:2],
            line.end_ned_m[:2],
            next_line.end_ned_m[:2],
            1.0,
        )
    except ValueError:
        return None
    unit_tangent_m = float(np.linalg.norm(line.end_ned_m[:2] - unit_geometry.entry_ne_m))
    if unit_tangent_m <= 0.0:  # pragma: no cover - valid geometry invariant
        return None
    tangent_cap_m = config.fillet_leg_fraction * min(
        line.horizontal_length_m,
        next_line.horizontal_length_m,
    )
    desired_radius_m = coordinated_turn_radius_m(
        line.airspeed_mps,
        config.fillet_bank_rad,
    )
    radius_m = min(
        desired_radius_m,
        config.fillet_max_radius_m,
        tangent_cap_m / unit_tangent_m,
    )
    if radius_m < config.minimum_fillet_radius_m:
        return None
    geometry = fillet_geometry(
        line.start_ned_m[:2],
        line.end_ned_m[:2],
        next_line.end_ned_m[:2],
        radius_m,
    )
    entry_probe = np.asarray([geometry.entry_ne_m[0], geometry.entry_ne_m[1], 0.0])
    exit_probe = np.asarray([geometry.exit_ne_m[0], geometry.exit_ne_m[1], 0.0])
    entry = np.asarray(
        [*geometry.entry_ne_m, line.commanded_down_m(entry_probe)],
        dtype=np.float64,
    )
    exit_point = np.asarray(
        [*geometry.exit_ne_m, next_line.commanded_down_m(exit_probe)],
        dtype=np.float64,
    )
    shortened_line = LineSegment(
        line.waypoint_id,
        line.start_ned_m,
        entry,
        line.airspeed_mps,
    )
    fillet_segment = FilletSegment(
        line.waypoint_id,
        geometry.center_ne_m,
        geometry.radius_m,
        geometry.direction,
        entry,
        exit_point,
        geometry.turn_angle_rad,
        line.airspeed_mps,
        next_line.airspeed_mps,
    )
    shortened = _replace_leg_segment(
        current,
        shortened_line,
        switch_normal_ne=line.horizontal_direction_ne,
    )
    fillet_leg = _Leg(
        segment=fillet_segment,
        acceptance_radius_m=current.acceptance_radius_m,
        altitude_tolerance_m=current.altitude_tolerance_m,
        fly_over=False,
        switch_normal_ne=fillet_segment.exit_tangent_ne,
    )
    return shortened, fillet_leg, exit_point


def _replace_leg_segment(
    leg: _Leg,
    segment: PathSegment,
    *,
    switch_normal_ne: FloatArray | None = None,
) -> _Leg:
    return _Leg(
        segment=segment,
        acceptance_radius_m=leg.acceptance_radius_m,
        altitude_tolerance_m=leg.altitude_tolerance_m,
        fly_over=leg.fly_over,
        switch_normal_ne=switch_normal_ne,
        loiter_duration_s=leg.loiter_duration_s,
        orbit_exit_ne_m=leg.orbit_exit_ne_m,
    )


def _home_geodetic_at_altitude(mission: Mission, waypoint: Waypoint) -> GeodeticPosition:
    """Return the home lat/lon at a return-home waypoint's resolved altitude."""
    return GeodeticPosition(
        latitude_rad=float(np.deg2rad(mission.home.latitude_deg)),
        longitude_rad=float(np.deg2rad(mission.home.longitude_deg)),
        altitude_m=mission.absolute_altitude_m(waypoint),
    )


__all__ = [
    "FilletGeometry",
    "FilletSegment",
    "LineSegment",
    "MissionPhase",
    "OrbitSegment",
    "PathManager",
    "PathManagerConfig",
    "PathManagerStatus",
    "PathSegment",
    "SegmentKind",
    "coordinated_turn_radius_m",
    "fillet_geometry",
]
