"""Versioned mission import/export with schema validation.

The on-disk format is YAML with an explicit top-level ``mission_version`` for
forward compatibility. Loading validates structure and version and raises
``ValueError`` with an actionable message on any problem; it does *not* silently
drop unknown fields. Dynamics-envelope validation is separate: call
:meth:`~aerognc.mission.mission.Mission.validate` after loading.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aerognc.mission.mission import (
    HomePosition,
    Mission,
    MissionDefaults,
    MissionLimits,
)
from aerognc.mission.waypoint import Waypoint

#: Mission-file schema versions this build understands.
SUPPORTED_MISSION_VERSIONS = frozenset({1})
CURRENT_MISSION_VERSION = 1


def mission_to_dict(mission: Mission) -> dict[str, Any]:
    """Serialize a :class:`Mission` to a plain, YAML-friendly mapping."""
    data: dict[str, Any] = {
        "mission_version": mission.mission_version,
        "mission": {"name": mission.name},
        "home": {
            "latitude_deg": mission.home.latitude_deg,
            "longitude_deg": mission.home.longitude_deg,
            "altitude_m": mission.home.altitude_m,
        },
        "defaults": {
            "airspeed_mps": mission.defaults.airspeed_mps,
            "acceptance_radius_m": mission.defaults.acceptance_radius_m,
            "altitude_tolerance_m": mission.defaults.altitude_tolerance_m,
        },
        "limits": {
            "min_altitude_m": mission.limits.min_altitude_m,
            "max_altitude_m": mission.limits.max_altitude_m,
            "min_airspeed_mps": mission.limits.min_airspeed_mps,
            "max_airspeed_mps": mission.limits.max_airspeed_mps,
            "max_bank_deg": float(round(_rad2deg(mission.limits.max_bank_rad), 6)),
            "gravity_mps2": mission.limits.gravity_mps2,
        },
        "waypoints": [waypoint.to_dict() for waypoint in mission.waypoints],
    }
    if mission.description:
        data["mission"]["description"] = mission.description
    return data


def mission_from_dict(data: dict[str, Any]) -> Mission:
    """Build a :class:`Mission` from a parsed mapping with clear errors."""
    if not isinstance(data, dict):
        raise ValueError("mission file must contain a top-level mapping")
    version = data.get("mission_version")
    if version is None:
        raise ValueError("mission file is missing 'mission_version'")
    if version not in SUPPORTED_MISSION_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_MISSION_VERSIONS))
        raise ValueError(
            f"unsupported mission_version {version!r}; this build supports [{supported}]"
        )

    meta = data.get("mission") or {}
    if "name" not in meta:
        raise ValueError("mission file is missing 'mission.name'")

    home_data = data.get("home")
    if not isinstance(home_data, dict):
        raise ValueError("mission file is missing a 'home' mapping")
    home = HomePosition(
        latitude_deg=float(home_data["latitude_deg"]),
        longitude_deg=float(home_data["longitude_deg"]),
        altitude_m=float(home_data.get("altitude_m", 0.0)),
    )

    defaults = _defaults_from_dict(data.get("defaults") or {})
    limits = _limits_from_dict(data.get("limits") or {})

    waypoint_data = data.get("waypoints")
    if not isinstance(waypoint_data, list):
        raise ValueError("mission file must contain a 'waypoints' list")
    waypoints = tuple(Waypoint.from_dict(item) for item in waypoint_data)

    return Mission(
        name=str(meta["name"]),
        description=str(meta.get("description", "")),
        home=home,
        waypoints=waypoints,
        defaults=defaults,
        limits=limits,
        mission_version=int(version),
    )


def load_mission(path: str | Path) -> Mission:
    """Load and parse a mission file (does not run envelope validation)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"mission file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if parsed is None:
        raise ValueError(f"mission file is empty: {file_path}")
    try:
        return mission_from_dict(parsed)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"failed to parse mission file {file_path}: {error}") from error


def save_mission(mission: Mission, path: str | Path) -> Path:
    """Write a mission to YAML and return the written path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(mission_to_dict(mission), handle, sort_keys=False, allow_unicode=True)
    return file_path


def _defaults_from_dict(data: dict[str, Any]) -> MissionDefaults:
    base = MissionDefaults()
    return MissionDefaults(
        airspeed_mps=float(data.get("airspeed_mps", base.airspeed_mps)),
        acceptance_radius_m=float(data.get("acceptance_radius_m", base.acceptance_radius_m)),
        altitude_tolerance_m=float(data.get("altitude_tolerance_m", base.altitude_tolerance_m)),
    )


def _limits_from_dict(data: dict[str, Any]) -> MissionLimits:
    base = MissionLimits()
    max_bank_rad = base.max_bank_rad
    if "max_bank_deg" in data:
        max_bank_rad = _deg2rad(float(data["max_bank_deg"]))
    elif "max_bank_rad" in data:
        max_bank_rad = float(data["max_bank_rad"])
    return MissionLimits(
        min_altitude_m=float(data.get("min_altitude_m", base.min_altitude_m)),
        max_altitude_m=float(data.get("max_altitude_m", base.max_altitude_m)),
        min_airspeed_mps=float(data.get("min_airspeed_mps", base.min_airspeed_mps)),
        max_airspeed_mps=float(data.get("max_airspeed_mps", base.max_airspeed_mps)),
        max_bank_rad=max_bank_rad,
        gravity_mps2=float(data.get("gravity_mps2", base.gravity_mps2)),
    )


def _deg2rad(value: float) -> float:
    return float(np.deg2rad(value))


def _rad2deg(value: float) -> float:
    return float(np.rad2deg(value))
