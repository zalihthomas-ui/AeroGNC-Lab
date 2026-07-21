"""Unit tests for mission I/O, versioning, and envelope-aware validation."""

from pathlib import Path

import numpy as np
import pytest

from aerognc.mission import (
    HomePosition,
    Mission,
    MissionDefaults,
    MissionLimits,
    MissionValidationError,
    Waypoint,
    WaypointAction,
    load_mission,
    mission_from_dict,
    mission_to_dict,
    save_mission,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "missions" / "waypoint_demo.mission.yaml"


def _mission(**overrides: object) -> Mission:
    base: dict[str, object] = {
        "name": "unit_mission",
        "home": HomePosition(39.925, 32.8369, 0.0),
        "waypoints": (
            Waypoint(id=1, name="WP1", latitude_deg=39.927, longitude_deg=32.840, altitude_m=120.0),
            Waypoint(
                id=2,
                name="WP2",
                latitude_deg=39.930,
                longitude_deg=32.847,
                altitude_m=180.0,
                airspeed_mps=22.0,
                action=WaypointAction.LOITER,
                loiter_radius_m=100.0,
            ),
        ),
    }
    base.update(overrides)
    return Mission(**base)  # type: ignore[arg-type]


def test_example_mission_loads_and_validates() -> None:
    mission = load_mission(EXAMPLE_MISSION)
    assert mission.name == "waypoint_demo"
    assert len(mission.waypoints) == 3
    mission.validate()  # must not raise
    assert mission.is_valid


def test_dict_round_trip() -> None:
    mission = _mission()
    restored = mission_from_dict(mission_to_dict(mission))
    assert restored.name == mission.name
    assert restored.waypoints == mission.waypoints
    assert restored.home == mission.home


def test_file_round_trip(tmp_path: Path) -> None:
    mission = _mission()
    out = save_mission(mission, tmp_path / "m.mission.yaml")
    reloaded = load_mission(out)
    assert reloaded.waypoints == mission.waypoints
    assert reloaded.limits.max_bank_rad == pytest.approx(mission.limits.max_bank_rad, abs=1e-6)


def test_unsupported_version_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported mission_version"):
        mission_from_dict(
            {"mission_version": 99, "mission": {"name": "x"}, "home": {}, "waypoints": []}
        )


def test_missing_version_rejected() -> None:
    with pytest.raises(ValueError, match="mission_version"):
        mission_from_dict({"mission": {"name": "x"}, "home": {}, "waypoints": []})


def test_missing_waypoints_rejected() -> None:
    with pytest.raises(ValueError, match="waypoints"):
        mission_from_dict(
            {
                "mission_version": 1,
                "mission": {"name": "x"},
                "home": {"latitude_deg": 0, "longitude_deg": 0},
            }
        )


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_mission(tmp_path / "does_not_exist.yaml")


def test_airspeed_outside_envelope_flagged() -> None:
    mission = _mission(
        waypoints=(
            Waypoint(
                id=1,
                name="TOO_FAST",
                latitude_deg=39.927,
                longitude_deg=32.840,
                altitude_m=120.0,
                airspeed_mps=999.0,
            ),
        )
    )
    issues = mission.validation_issues()
    assert any("safe envelope" in issue for issue in issues)
    with pytest.raises(MissionValidationError):
        mission.validate()


def test_altitude_outside_bounds_flagged() -> None:
    mission = _mission(
        waypoints=(
            Waypoint(id=1, name="HIGH", latitude_deg=39.9, longitude_deg=32.8, altitude_m=99999.0),
        )
    )
    assert any("outside" in issue for issue in mission.validation_issues())


def test_loiter_radius_below_min_turn_radius_flagged() -> None:
    # At 22 m/s and 45 deg bank the minimum turn radius is ~49 m; ask for 5 m.
    mission = _mission(
        waypoints=(
            Waypoint(
                id=1,
                name="TIGHT",
                latitude_deg=39.9,
                longitude_deg=32.8,
                altitude_m=150.0,
                airspeed_mps=22.0,
                action=WaypointAction.LOITER,
                loiter_radius_m=5.0,
            ),
        )
    )
    assert any("minimum safe turn radius" in issue for issue in mission.validation_issues())


def test_duplicate_ids_flagged() -> None:
    mission = _mission(
        waypoints=(
            Waypoint(id=1, name="A", latitude_deg=39.9, longitude_deg=32.8, altitude_m=120.0),
            Waypoint(id=1, name="B", latitude_deg=39.91, longitude_deg=32.81, altitude_m=130.0),
        )
    )
    assert any("duplicate waypoint id" in issue for issue in mission.validation_issues())


def test_min_turn_radius_formula() -> None:
    limits = MissionLimits(max_bank_rad=float(np.deg2rad(45.0)), gravity_mps2=9.80665)
    # r = v^2 / (g tan45) = 22^2 / 9.80665 ~ 49.35 m
    assert limits.min_turn_radius_m(22.0) == pytest.approx(484.0 / 9.80665, rel=1e-6)


def test_defaults_resolution() -> None:
    mission = _mission(defaults=MissionDefaults(airspeed_mps=18.0, acceptance_radius_m=25.0))
    wp_no_override = mission.waypoints[0]
    wp_override = mission.waypoints[1]
    assert mission.resolved_airspeed_mps(wp_no_override) == pytest.approx(18.0)
    assert mission.resolved_airspeed_mps(wp_override) == pytest.approx(22.0)
    assert mission.resolved_acceptance_radius_m(wp_no_override) == pytest.approx(25.0)
