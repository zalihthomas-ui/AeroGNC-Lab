"""Strongly typed, validated waypoint data model for fixed-wing missions.

A :class:`Waypoint` is a mission-file boundary object, so latitude and longitude
are stored in *degrees* to match the YAML schema exactly; :meth:`Waypoint.geodetic`
converts to radians for the numerical core. Envelope-dependent checks (airspeed
within the safe flight envelope, loiter radius versus minimum turn radius, altitude
within configured bounds) are intentionally left to mission-level validation in
:mod:`aerognc.mission.mission`, because a single waypoint does not know the
aircraft envelope.
"""

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any

import numpy as np

from aerognc.mathematics.geodesy import GeodeticPosition


class WaypointAction(StrEnum):
    """Supported waypoint actions (see mission-planning guide)."""

    FLY_THROUGH = "fly_through"
    TURN = "turn"
    LOITER = "loiter"
    HOLD = "hold"
    CHANGE_ALTITUDE = "change_altitude"
    CHANGE_AIRSPEED = "change_airspeed"
    TAKEOFF = "takeoff"
    LAND = "land"
    RETURN_HOME = "return_home"
    MISSION_END = "mission_end"


class AltitudeReference(StrEnum):
    """Reference datum for :attr:`Waypoint.altitude_m`."""

    RELATIVE_HOME = "relative_home"
    MEAN_SEA_LEVEL = "msl"
    ABOVE_GROUND = "agl"


class LoiterDirection(StrEnum):
    """Loiter turn sense as seen from above (north-up)."""

    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


class TurnType(StrEnum):
    """How the path manager should join legs at this waypoint."""

    SMOOTH = "smooth"
    SHARP = "sharp"


_ACTIONS_REQUIRING_LOITER = frozenset({WaypointAction.LOITER, WaypointAction.HOLD})


@dataclass(frozen=True, slots=True)
class Waypoint:
    """A single validated mission waypoint.

    Optional numeric fields left as ``None`` inherit the mission defaults during
    resolution. All angles at the numerical boundary are radians; the stored
    latitude/longitude remain in degrees for schema fidelity.
    """

    id: int
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    altitude_reference: AltitudeReference = AltitudeReference.RELATIVE_HOME
    airspeed_mps: float | None = None
    acceptance_radius_m: float | None = None
    altitude_tolerance_m: float | None = None
    action: WaypointAction = WaypointAction.FLY_THROUGH
    loiter_radius_m: float | None = None
    loiter_duration_s: float | None = None
    loiter_direction: LoiterDirection = LoiterDirection.CLOCKWISE
    turn_type: TurnType = TurnType.SMOOTH
    notes: str = ""

    def __post_init__(self) -> None:
        if int(self.id) != self.id or self.id <= 0:
            raise ValueError(f"waypoint id must be a positive integer, got {self.id!r}")
        if not self.name:
            raise ValueError("waypoint name must be non-empty")
        finite = [self.latitude_deg, self.longitude_deg, self.altitude_m]
        if not np.all(np.isfinite(finite)):
            raise ValueError(f"waypoint {self.name!r} coordinates must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(
                f"waypoint {self.name!r} latitude_deg must be in [-90, 90], "
                f"got {self.latitude_deg}"
            )
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(
                f"waypoint {self.name!r} longitude_deg must be in [-180, 180], "
                f"got {self.longitude_deg}"
            )
        self._validate_optional_positive("airspeed_mps", self.airspeed_mps)
        self._validate_optional_positive("acceptance_radius_m", self.acceptance_radius_m)
        self._validate_optional_positive("altitude_tolerance_m", self.altitude_tolerance_m)
        self._validate_optional_positive("loiter_radius_m", self.loiter_radius_m)
        if self.loiter_duration_s is not None and (
            not np.isfinite(self.loiter_duration_s) or self.loiter_duration_s < 0.0
        ):
            raise ValueError(f"waypoint {self.name!r} loiter_duration_s must be nonnegative")
        if self.action in _ACTIONS_REQUIRING_LOITER and self.loiter_radius_m is None:
            raise ValueError(
                f"waypoint {self.name!r} action {self.action.value!r} requires loiter_radius_m"
            )

    def _validate_optional_positive(self, field_name: str, value: float | None) -> None:
        if value is None:
            return
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"waypoint {self.name!r} {field_name} must be positive and finite, got {value}"
            )

    def geodetic(self) -> GeodeticPosition:
        """Return this waypoint's position as a radian :class:`GeodeticPosition`.

        The altitude is passed through unchanged; callers are responsible for
        resolving :attr:`altitude_reference` against the home/terrain datum.
        """
        return GeodeticPosition(
            latitude_rad=float(np.deg2rad(self.latitude_deg)),
            longitude_rad=float(np.deg2rad(self.longitude_deg)),
            altitude_m=float(self.altitude_m),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-friendly mapping (enums as their string values)."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m,
            "altitude_reference": self.altitude_reference.value,
            "action": self.action.value,
        }
        for key in ("airspeed_mps", "acceptance_radius_m", "altitude_tolerance_m"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.action in _ACTIONS_REQUIRING_LOITER:
            data["loiter_radius_m"] = self.loiter_radius_m
            if self.loiter_duration_s is not None:
                data["loiter_duration_s"] = self.loiter_duration_s
            data["loiter_direction"] = self.loiter_direction.value
        if self.turn_type is not TurnType.SMOOTH:
            data["turn_type"] = self.turn_type.value
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Waypoint":
        """Build a :class:`Waypoint` from a mapping, with clear error messages."""
        required = ("id", "name", "latitude_deg", "longitude_deg", "altitude_m")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"waypoint is missing required field(s): {', '.join(missing)}")
        known = {
            "id",
            "name",
            "latitude_deg",
            "longitude_deg",
            "altitude_m",
            "altitude_reference",
            "airspeed_mps",
            "acceptance_radius_m",
            "altitude_tolerance_m",
            "action",
            "loiter_radius_m",
            "loiter_duration_s",
            "loiter_direction",
            "turn_type",
            "notes",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"waypoint has unknown field(s): {', '.join(sorted(unknown))}")
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            latitude_deg=float(data["latitude_deg"]),
            longitude_deg=float(data["longitude_deg"]),
            altitude_m=float(data["altitude_m"]),
            altitude_reference=_parse_enum(
                AltitudeReference, data.get("altitude_reference"), AltitudeReference.RELATIVE_HOME
            ),
            airspeed_mps=_optional_float(data.get("airspeed_mps")),
            acceptance_radius_m=_optional_float(data.get("acceptance_radius_m")),
            altitude_tolerance_m=_optional_float(data.get("altitude_tolerance_m")),
            action=_parse_enum(WaypointAction, data.get("action"), WaypointAction.FLY_THROUGH),
            loiter_radius_m=_optional_float(data.get("loiter_radius_m")),
            loiter_duration_s=_optional_float(data.get("loiter_duration_s")),
            loiter_direction=_parse_enum(
                LoiterDirection, data.get("loiter_direction"), LoiterDirection.CLOCKWISE
            ),
            turn_type=_parse_enum(TurnType, data.get("turn_type"), TurnType.SMOOTH),
            notes=str(data.get("notes", "")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _parse_enum(enum_cls: type[Enum], value: Any, default: Enum) -> Any:
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError as error:
        allowed = ", ".join(str(member.value) for member in enum_cls)
        raise ValueError(
            f"{enum_cls.__name__} must be one of [{allowed}], got {value!r}"
        ) from error
