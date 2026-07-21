"""Integration and scenario tests for the end-to-end waypoint GNC loop."""

from pathlib import Path

import numpy as np
import pytest

from aerognc.gnc.waypoint_guidance import GuidanceMode
from aerognc.mission import HomePosition, Mission, MissionDefaults, Waypoint, WaypointAction
from aerognc.mission.mission_manager import MissionState
from aerognc.navigation.providers import NoisyStateProvider
from aerognc.simulation.waypoint_mission import (
    WaypointMissionConfig,
    run_waypoint_mission,
)
from aerognc.vehicle.control_surfaces import SurfaceFailureMode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _straight_mission(action_last: WaypointAction = WaypointAction.FLY_THROUGH) -> Mission:
    return Mission(
        name="scenario",
        home=HomePosition(0.0, 0.0, 0.0),
        defaults=MissionDefaults(airspeed_mps=20.0, acceptance_radius_m=40.0),
        waypoints=(
            Waypoint(id=1, name="A", latitude_deg=0.006, longitude_deg=0.0, altitude_m=120.0),
            Waypoint(id=2, name="B", latitude_deg=0.012, longitude_deg=0.004, altitude_m=160.0),
            Waypoint(
                id=3, name="C", latitude_deg=0.012, longitude_deg=0.010, altitude_m=160.0,
                action=action_last,
            ),
        ),
    )


def _no_nans(result) -> None:
    for sample in result.samples:
        assert np.isfinite(sample.north_m)
        assert np.isfinite(sample.altitude_m)
        assert np.isfinite(sample.airspeed_mps)


# --- nominal ----------------------------------------------------------------


def test_nominal_multi_waypoint_completes() -> None:
    result = run_waypoint_mission(_straight_mission())
    assert result.completed
    assert result.final_state is MissionState.MISSION_COMPLETE
    _no_nans(result)


def test_bundled_demo_mission_completes() -> None:
    from aerognc.mission import load_mission

    mission = load_mission(REPO_ROOT / "missions" / "waypoint_demo.mission.yaml")
    result = run_waypoint_mission(mission)
    assert result.completed
    summary = result.summary()
    assert summary["min_airspeed_mps"] > 10.0  # never stalled to a standstill
    assert summary["max_abs_cross_track_m"] < 400.0


@pytest.mark.parametrize("mode", list(GuidanceMode))
def test_all_guidance_modes_complete(mode: GuidanceMode) -> None:
    result = run_waypoint_mission(
        _straight_mission(), WaypointMissionConfig(guidance_mode=mode)
    )
    assert result.completed


# --- environment / disturbances ---------------------------------------------


def test_crosswind_mission_completes_and_tracks() -> None:
    config = WaypointMissionConfig(wind_ned_mps=(0.0, 6.0, 0.0))  # 6 m/s from the west
    result = run_waypoint_mission(_straight_mission(), config)
    assert result.completed
    # After settling, cross-track stays bounded despite the steady crosswind.
    settled = [abs(s.cross_track_error_m) for s in result.samples[len(result.samples) // 2 :]]
    assert max(settled) < 120.0


def test_gps_dropout_mission_still_completes() -> None:
    config = WaypointMissionConfig(
        provider=NoisyStateProvider(seed=3, gps_dropout_window_s=(20.0, 30.0))
    )
    result = run_waypoint_mission(_straight_mission(), config)
    # It should either complete or safely return home / abort - never crash or NaN.
    assert result.outcome in {"complete", "abort"}
    _no_nans(result)


# --- return home & safety ---------------------------------------------------


def test_return_home_waypoint_completes_at_home() -> None:
    result = run_waypoint_mission(_straight_mission(WaypointAction.RETURN_HOME))
    assert result.completed
    last = result.samples[-1]
    assert np.hypot(last.north_m, last.east_m) < 100.0  # ended near home horizontally


def test_geofence_breach_triggers_return_or_abort() -> None:
    from aerognc.mission.safety import SafetyLimits

    # A tight geofence the mission legs exceed -> safety should intervene.
    config = WaypointMissionConfig(safety_limits=SafetyLimits(geofence_radius_m=300.0))
    result = run_waypoint_mission(_straight_mission(), config)
    triggered = [e for e in result.metadata["safety_events"]]  # type: ignore[index]
    assert any(event[1] == "geofence" for event in triggered)


# --- actuator failure -------------------------------------------------------


def test_elevator_failure_does_not_crash_the_simulation() -> None:
    config = WaypointMissionConfig(
        surface_failures={"elevator": SurfaceFailureMode.STUCK}, max_time_s=300.0
    )
    result = run_waypoint_mission(_straight_mission(), config)
    # With a stuck elevator the mission likely will not complete, but the loop
    # must terminate cleanly with finite state and a definite outcome.
    assert result.outcome in {"complete", "abort", "emergency", "timeout"}
    _no_nans(result)


# --- logging / export -------------------------------------------------------


def test_csv_and_json_export(tmp_path: Path) -> None:
    result = run_waypoint_mission(_straight_mission())
    csv_path = result.to_csv(tmp_path / "log.csv")
    json_path = result.to_json(tmp_path / "log.json")
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert json_path.is_file() and json_path.stat().st_size > 0
