"""Tests for the high-level aerognc.fly_mission API façade."""

from pathlib import Path

import pytest

import aerognc
from aerognc.api import fly_configured_mission, fly_mission
from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.mission import HomePosition, Mission, Waypoint, WaypointAction

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "missions" / "waypoint_demo.mission.yaml"
RUNTIME = REPO_ROOT / "configs" / "waypoint_gnc.yaml"


def _mission() -> Mission:
    return Mission(
        name="api",
        home=HomePosition(0.0, 0.0, 0.0),
        waypoints=(
            Waypoint(id=1, name="A", latitude_deg=0.006, longitude_deg=0.0, altitude_m=120.0),
            Waypoint(
                id=2,
                name="B",
                latitude_deg=0.0,
                longitude_deg=0.0,
                altitude_m=100.0,
                action=WaypointAction.RETURN_HOME,
            ),
        ),
    )


def test_top_level_fly_mission_is_lazily_exposed() -> None:
    assert aerognc.fly_mission is fly_mission
    assert aerognc.fly_configured_mission is fly_configured_mission


def test_fly_mission_accepts_a_mission_object() -> None:
    result = fly_mission(_mission())
    assert result.completed


def test_fly_mission_accepts_a_path() -> None:
    result = fly_mission(DEMO)
    assert result.completed


def test_fly_mission_guidance_and_wind_kwargs() -> None:
    result = fly_mission(_mission(), guidance="l1_guidance", wind_east_mps=3.0)
    assert result.completed
    assert result.metadata["guidance_mode"] == GuidanceMode.L1_GUIDANCE.value
    assert result.metadata["wind_ned_mps"] == [0.0, 3.0, 0.0]


def test_fly_configured_mission_loads_the_complete_runtime() -> None:
    result = fly_configured_mission(RUNTIME)

    assert result.completed
    assert result.metadata["navigation_provider"] == "PerfectStateProvider"
    assert result.metadata["vehicle_backend"] == "InternalFixedWingBackend"
    provenance = result.metadata["runtime_configuration"]
    assert isinstance(provenance, dict)
    assert provenance["name"] == "waypoint_demo_internal"
    assert len(provenance["sha256"]) == 64


def test_unknown_top_level_attribute_raises() -> None:
    with pytest.raises(AttributeError):
        _ = aerognc.does_not_exist
