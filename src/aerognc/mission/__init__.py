"""Waypoint mission domain models, validation, and versioned I/O.

This package is the *Mission* layer of the waypoint fixed-wing GNC workflow. It
holds only data models and pure logic (no simulation, UI, or hardware), so it is
fully unit-testable and reusable across the internal simulator, SITL backends, and
the map-based planner. See ``docs/waypoint_gnc/`` and the repository ``TODO.md``.
"""

from aerognc.mission.mission import (
    HomePosition,
    Mission,
    MissionDefaults,
    MissionLimits,
    MissionValidationError,
)
from aerognc.mission.mission_io import (
    CURRENT_MISSION_VERSION,
    SUPPORTED_MISSION_VERSIONS,
    load_mission,
    mission_from_dict,
    mission_to_dict,
    save_mission,
)
from aerognc.mission.mission_manager import (
    MissionManager,
    MissionManagerStatus,
    MissionState,
    SafetyResponse,
    StateTransition,
)
from aerognc.mission.safety import SafetyEvent, SafetyLimits, SafetyManager, SafetyVerdict
from aerognc.mission.waypoint import (
    AltitudeReference,
    LoiterDirection,
    TurnType,
    Waypoint,
    WaypointAction,
)

__all__ = [
    "CURRENT_MISSION_VERSION",
    "SUPPORTED_MISSION_VERSIONS",
    "AltitudeReference",
    "HomePosition",
    "LoiterDirection",
    "Mission",
    "MissionDefaults",
    "MissionLimits",
    "MissionManager",
    "MissionManagerStatus",
    "MissionState",
    "MissionValidationError",
    "SafetyEvent",
    "SafetyLimits",
    "SafetyManager",
    "SafetyResponse",
    "SafetyVerdict",
    "StateTransition",
    "TurnType",
    "Waypoint",
    "WaypointAction",
    "load_mission",
    "mission_from_dict",
    "mission_to_dict",
    "save_mission",
]
