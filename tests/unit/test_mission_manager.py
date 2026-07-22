"""Unit tests for the mission state machine."""

import numpy as np
import pytest

from aerognc.gnc.path_manager import PathManager
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mission import HomePosition, Mission, MissionDefaults, Waypoint, WaypointAction
from aerognc.mission.mission_manager import MissionManager, MissionState, SafetyResponse
from aerognc.navigation.state import NavigationState


def _mission() -> Mission:
    return Mission(
        name="mm",
        home=HomePosition(0.0, 0.0, 0.0),
        defaults=MissionDefaults(acceptance_radius_m=30.0),
        waypoints=(
            Waypoint(id=1, name="WP1", latitude_deg=0.01, longitude_deg=0.0, altitude_m=100.0),
            Waypoint(id=2, name="WP2", latitude_deg=0.02, longitude_deg=0.0, altitude_m=100.0),
            Waypoint(
                id=3,
                name="RTH",
                latitude_deg=0.0,
                longitude_deg=0.0,
                altitude_m=100.0,
                action=WaypointAction.RETURN_HOME,
            ),
        ),
    )


def _manager() -> MissionManager:
    mission = _mission()
    return MissionManager(mission, PathManager.from_mission(mission))


def _state(ned) -> NavigationState:
    return NavigationState(
        position_ned_m=np.asarray(ned, dtype=float),
        velocity_ned_mps=np.array([20.0, 0.0, 0.0]),
        quaternion_nb=euler321_to_quaternion(0.0, 0.0, 0.0),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=20.0,
    )


def test_arm_and_start_transitions() -> None:
    manager = _manager()
    assert manager.state is MissionState.DISARMED
    manager.arm()
    assert manager.state is MissionState.READY
    manager.start()
    assert manager.state is MissionState.NAVIGATE


def test_pause_blocks_waypoint_advance() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    manager.pause()
    wp1_ned = manager.path_manager.segments[0].end_ned_m  # type: ignore[attr-defined]
    status = manager.update(_state(wp1_ned), 0.1)
    assert status.state is MissionState.PAUSED
    assert status.path_status is None  # path not advanced while paused
    manager.resume()
    assert manager.state is MissionState.NAVIGATE


def test_return_home_jumps_to_rth_leg() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    assert manager.request_return_home()
    assert manager.state is MissionState.RETURN_HOME
    # Active waypoint should now be the return-home waypoint (id 3).
    status = manager.update(_state([500.0, 0.0, -100.0]), 0.1)
    assert status.active_waypoint_id == 3


def test_mission_completes_when_flown() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    vertices = manager.path_manager.planned_path_ned()
    complete = False
    for target in vertices[1:]:
        for _ in range(30):
            status = manager.update(_state(target), 0.2)
            if status.mission_complete:
                complete = True
                break
        if complete:
            break
    assert complete
    assert manager.state is MissionState.MISSION_COMPLETE


def test_safety_abort_forces_abort_state() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    status = manager.update(_state([10.0, 0.0, -100.0]), 0.1, SafetyResponse.ABORT)
    assert status.state is MissionState.ABORT


def test_safety_terminate_triggers_emergency() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    status = manager.update(_state([10.0, 0.0, -100.0]), 0.1, SafetyResponse.TERMINATE)
    assert status.state is MissionState.EMERGENCY


def test_transitions_are_logged() -> None:
    manager = _manager()
    manager.arm()
    manager.start()
    manager.pause()
    reasons = [t.to_state for t in manager.transitions]
    assert MissionState.READY in reasons
    assert MissionState.NAVIGATE in reasons
    assert MissionState.PAUSED in reasons


def test_arm_rejects_invalid_mission() -> None:
    bad = Mission(
        name="bad",
        home=HomePosition(0.0, 0.0, 0.0),
        waypoints=(
            Waypoint(
                id=1,
                name="F",
                latitude_deg=0.01,
                longitude_deg=0.0,
                altitude_m=100.0,
                airspeed_mps=999.0,
            ),
        ),
    )
    manager = MissionManager(bad, PathManager.from_mission(_mission()))
    with pytest.raises(ValueError):
        manager.arm()
