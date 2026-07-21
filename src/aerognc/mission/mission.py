"""Mission container, defaults, envelope limits, and whole-mission validation.

A :class:`Mission` aggregates a home position, per-mission defaults, a flight
envelope (:class:`MissionLimits`), and an ordered tuple of
:class:`~aerognc.mission.waypoint.Waypoint`. Envelope-dependent checks that a lone
waypoint cannot perform live here: airspeed inside the safe envelope, altitude
inside configured bounds, and loiter radius above the minimum coordinated-turn
radius ``r = v^2 / (g * tan(bank))``.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from aerognc.mathematics.geodesy import GeodeticPosition, ReferenceEllipsoid
from aerognc.mathematics.local_frame import WGS84, LocalTangentFrame
from aerognc.mathematics.vectors import FloatArray
from aerognc.mission.waypoint import AltitudeReference, Waypoint, WaypointAction


class MissionValidationError(ValueError):
    """Raised when a mission fails validation; carries every issue found."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        joined = "\n  - ".join(self.issues)
        super().__init__(f"mission validation failed:\n  - {joined}")


@dataclass(frozen=True, slots=True)
class HomePosition:
    """Mission home / launch reference in degrees and metres MSL."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.latitude_deg, self.longitude_deg, self.altitude_m])):
            raise ValueError("home position values must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("home latitude_deg must be in [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("home longitude_deg must be in [-180, 180]")

    def geodetic(self) -> GeodeticPosition:
        """Return the home position as a radian :class:`GeodeticPosition`."""
        return GeodeticPosition(
            latitude_rad=float(np.deg2rad(self.latitude_deg)),
            longitude_rad=float(np.deg2rad(self.longitude_deg)),
            altitude_m=float(self.altitude_m),
        )


@dataclass(frozen=True, slots=True)
class MissionDefaults:
    """Fallback values applied to waypoints that omit them."""

    airspeed_mps: float = 20.0
    acceptance_radius_m: float = 30.0
    altitude_tolerance_m: float = 10.0

    def __post_init__(self) -> None:
        values = [self.airspeed_mps, self.acceptance_radius_m, self.altitude_tolerance_m]
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
            raise ValueError("mission defaults must be positive and finite")


@dataclass(frozen=True, slots=True)
class MissionLimits:
    """Safe flight envelope used for envelope-aware mission validation."""

    min_altitude_m: float = 0.0
    max_altitude_m: float = 3000.0
    min_airspeed_mps: float = 12.0
    max_airspeed_mps: float = 45.0
    max_bank_rad: float = float(np.deg2rad(45.0))
    gravity_mps2: float = 9.80665

    def __post_init__(self) -> None:
        finite = [
            self.min_altitude_m,
            self.max_altitude_m,
            self.min_airspeed_mps,
            self.max_airspeed_mps,
            self.max_bank_rad,
            self.gravity_mps2,
        ]
        if not np.all(np.isfinite(finite)):
            raise ValueError("mission limits must be finite")
        if self.max_altitude_m <= self.min_altitude_m:
            raise ValueError("max_altitude_m must exceed min_altitude_m")
        if not 0.0 < self.min_airspeed_mps < self.max_airspeed_mps:
            raise ValueError("require 0 < min_airspeed_mps < max_airspeed_mps")
        if not 0.0 < self.max_bank_rad < 0.5 * np.pi:
            raise ValueError("max_bank_rad must be in (0, pi/2)")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("gravity_mps2 must be positive")

    def min_turn_radius_m(self, airspeed_mps: float) -> float:
        """Minimum coordinated-turn radius at the given airspeed [m].

        Uses ``r = v^2 / (g * tan(bank_max))``. The bank is bounded away from
        ``pi/2`` in :meth:`__post_init__`, so ``tan`` is finite and positive.
        """
        if not np.isfinite(airspeed_mps) or airspeed_mps <= 0.0:
            raise ValueError("airspeed_mps must be positive and finite")
        return float(airspeed_mps**2 / (self.gravity_mps2 * np.tan(self.max_bank_rad)))


@dataclass(frozen=True, slots=True)
class Mission:
    """An ordered waypoint mission with defaults and a validated envelope."""

    name: str
    home: HomePosition
    waypoints: tuple[Waypoint, ...]
    defaults: MissionDefaults = field(default_factory=MissionDefaults)
    limits: MissionLimits = field(default_factory=MissionLimits)
    description: str = ""
    mission_version: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mission name must be non-empty")
        # Coerce to a tuple so the frozen value type is always hashable/immutable
        # even if a caller passed a list.
        object.__setattr__(self, "waypoints", tuple(self.waypoints))

    # -- resolution of per-waypoint values against defaults -------------------
    def resolved_airspeed_mps(self, waypoint: Waypoint) -> float:
        return waypoint.airspeed_mps or self.defaults.airspeed_mps

    def resolved_acceptance_radius_m(self, waypoint: Waypoint) -> float:
        return waypoint.acceptance_radius_m or self.defaults.acceptance_radius_m

    def resolved_altitude_tolerance_m(self, waypoint: Waypoint) -> float:
        return waypoint.altitude_tolerance_m or self.defaults.altitude_tolerance_m

    def local_frame(self, ellipsoid: ReferenceEllipsoid = WGS84) -> LocalTangentFrame:
        """Return the home-anchored local NED frame for this mission."""
        return LocalTangentFrame(origin=self.home.geodetic(), ellipsoid=ellipsoid)

    # -- validation -----------------------------------------------------------
    def validation_issues(self) -> list[str]:
        """Return a list of human-readable validation problems (empty if valid)."""
        issues: list[str] = []
        if not self.waypoints:
            issues.append("mission has no waypoints")
        seen_ids: set[int] = set()
        for waypoint in self.waypoints:
            if waypoint.id in seen_ids:
                issues.append(f"duplicate waypoint id {waypoint.id}")
            seen_ids.add(waypoint.id)
            issues.extend(self._waypoint_envelope_issues(waypoint))
        return issues

    def _waypoint_envelope_issues(self, waypoint: Waypoint) -> list[str]:
        issues: list[str] = []
        label = f"waypoint {waypoint.name!r}"
        absolute_altitude_m = self.absolute_altitude_m(waypoint)
        if not self.limits.min_altitude_m <= absolute_altitude_m <= self.limits.max_altitude_m:
            issues.append(
                f"{label} altitude {absolute_altitude_m:.1f} m (MSL) is outside "
                f"[{self.limits.min_altitude_m:.1f}, {self.limits.max_altitude_m:.1f}] m"
            )
        airspeed_mps = self.resolved_airspeed_mps(waypoint)
        if not self.limits.min_airspeed_mps <= airspeed_mps <= self.limits.max_airspeed_mps:
            issues.append(
                f"{label} airspeed {airspeed_mps:.1f} m/s is outside the safe envelope "
                f"[{self.limits.min_airspeed_mps:.1f}, {self.limits.max_airspeed_mps:.1f}] m/s"
            )
        if waypoint.loiter_radius_m is not None:
            min_radius_m = self.limits.min_turn_radius_m(airspeed_mps)
            if waypoint.loiter_radius_m < min_radius_m:
                issues.append(
                    f"{label} loiter_radius_m {waypoint.loiter_radius_m:.1f} m is below the "
                    f"minimum safe turn radius {min_radius_m:.1f} m at {airspeed_mps:.1f} m/s"
                )
        return issues

    def absolute_altitude_m(self, waypoint: Waypoint) -> float:
        """Return the waypoint altitude referenced to MSL.

        ``relative_home`` and ``agl`` are treated as heights above the home datum
        (this build has no terrain model, so AGL collapses to relative-home);
        ``msl`` is passed through unchanged.
        """
        if waypoint.altitude_reference in (
            AltitudeReference.RELATIVE_HOME,
            AltitudeReference.ABOVE_GROUND,
        ):
            return self.home.altitude_m + waypoint.altitude_m
        return waypoint.altitude_m

    def waypoint_ned_m(self, waypoint: Waypoint, frame: LocalTangentFrame) -> FloatArray:
        """Return the waypoint position in home-referenced local NED [m].

        The altitude reference is resolved to MSL before the geodetic-to-NED
        conversion, so the returned Down component is consistent with the
        vehicle state produced by the simulator.
        """
        geodetic = GeodeticPosition(
            latitude_rad=float(np.deg2rad(waypoint.latitude_deg)),
            longitude_rad=float(np.deg2rad(waypoint.longitude_deg)),
            altitude_m=self.absolute_altitude_m(waypoint),
        )
        return frame.geodetic_to_ned(geodetic)

    def validate(self) -> "Mission":
        """Return ``self`` if valid, else raise :class:`MissionValidationError`."""
        issues = self.validation_issues()
        if issues:
            raise MissionValidationError(issues)
        return self

    @property
    def is_valid(self) -> bool:
        """True when the mission has no validation issues."""
        return not self.validation_issues()

    # -- edit operations (pure; return a new Mission) -------------------------
    def with_waypoints(self, waypoints: tuple[Waypoint, ...]) -> "Mission":
        """Return a copy with a replaced waypoint tuple."""
        return replace(self, waypoints=tuple(waypoints))

    def terminal_actions(self) -> tuple[WaypointAction, ...]:
        """Return the action of each waypoint in order (planning convenience)."""
        return tuple(waypoint.action for waypoint in self.waypoints)
