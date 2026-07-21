"""Unit tests for the fixed-wing path manager: geometry and switching."""

from itertools import pairwise

import numpy as np
import pytest

from aerognc.gnc.path_manager import (
    LineSegment,
    OrbitSegment,
    PathManager,
    PathManagerConfig,
    coordinated_turn_radius_m,
    fillet_geometry,
)
from aerognc.mission import (
    HomePosition,
    Mission,
    MissionDefaults,
    Waypoint,
    WaypointAction,
)

# --- turn geometry -----------------------------------------------------------


def test_coordinated_turn_radius_value() -> None:
    # 20 m/s at 30 deg bank: r = 400 / (9.80665 * tan30) ~ 70.6 m
    radius = coordinated_turn_radius_m(20.0, np.deg2rad(30.0))
    assert radius == pytest.approx(400.0 / (9.80665 * np.tan(np.deg2rad(30.0))), rel=1e-9)


def test_coordinated_turn_radius_level_is_infinite() -> None:
    assert coordinated_turn_radius_m(20.0, 0.0) == float("inf")


def test_coordinated_turn_radius_rejects_bad_speed() -> None:
    with pytest.raises(ValueError):
        coordinated_turn_radius_m(-1.0, 0.3)


# --- line segment ------------------------------------------------------------


def _line() -> LineSegment:
    return LineSegment(1, np.zeros(3), np.array([100.0, 0.0, -50.0]), 20.0)


def test_line_cross_track_sign_positive_to_the_right() -> None:
    segment = _line()  # heading due North
    # A point to the East is to the right of a North-bound path -> positive.
    assert segment.cross_track_error_m(np.array([50.0, 10.0, 0.0])) == pytest.approx(10.0)
    assert segment.cross_track_error_m(np.array([50.0, -10.0, 0.0])) == pytest.approx(-10.0)


def test_line_along_track_fraction_and_altitude_ramp() -> None:
    segment = _line()
    assert segment.along_track_fraction(np.array([50.0, 0.0, 0.0])) == pytest.approx(0.5)
    # Down ramps from 0 to -50 over the leg; halfway -> -25.
    assert segment.commanded_down_m(np.array([50.0, 0.0, 0.0])) == pytest.approx(-25.0)
    # Clamped beyond the ends.
    assert segment.along_track_fraction(np.array([200.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_line_distance_to_waypoint() -> None:
    segment = _line()
    assert segment.horizontal_distance_to_waypoint_m(np.array([70.0, 0.0, 0.0])) == pytest.approx(
        30.0
    )


def test_line_rejects_zero_horizontal_length() -> None:
    with pytest.raises(ValueError):
        LineSegment(1, np.zeros(3), np.array([0.0, 0.0, -50.0]), 20.0)


# --- orbit segment -----------------------------------------------------------


def test_orbit_radial_error_and_distance() -> None:
    orbit = OrbitSegment(2, np.array([0.0, 0.0, -100.0]), 60.0, 1, 22.0)
    outside = np.array([80.0, 0.0, -100.0])
    assert orbit.radial_distance_m(outside) == pytest.approx(80.0)
    assert orbit.cross_track_error_m(outside) == pytest.approx(20.0)  # 80 - 60
    inside = np.array([40.0, 0.0, -100.0])
    assert orbit.cross_track_error_m(inside) == pytest.approx(-20.0)


def test_orbit_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        OrbitSegment(2, np.zeros(3), 60.0, 0, 22.0)


# --- fillet geometry ---------------------------------------------------------


def test_fillet_ninety_degree_corner() -> None:
    # Incoming due North, outgoing due East, radius 10 -> center at corner+(-10,10).
    previous = np.array([-100.0, 0.0])
    corner = np.array([0.0, 0.0])
    following = np.array([0.0, 100.0])
    fillet = fillet_geometry(previous, corner, following, 10.0)
    assert fillet.turn_angle_rad == pytest.approx(0.5 * np.pi, abs=1e-9)
    assert np.allclose(fillet.center_ne_m, np.array([-10.0, 10.0]), atol=1e-9)
    assert fillet.direction == 1  # right turn = clockwise
    # Tangent points sit one radius back/forward along each leg.
    assert np.allclose(fillet.entry_ne_m, np.array([-10.0, 0.0]), atol=1e-9)
    assert np.allclose(fillet.exit_ne_m, np.array([0.0, 10.0]), atol=1e-9)


def test_fillet_rejects_collinear_legs() -> None:
    with pytest.raises(ValueError, match="collinear"):
        fillet_geometry(np.array([-1.0, 0.0]), np.zeros(2), np.array([1.0, 0.0]), 10.0)


# --- path manager construction & switching -----------------------------------


def _demo_mission() -> Mission:
    return Mission(
        name="pm_demo",
        home=HomePosition(0.0, 0.0, 0.0),
        defaults=MissionDefaults(airspeed_mps=20.0, acceptance_radius_m=5.0),
        waypoints=(
            Waypoint(id=1, name="WP1", latitude_deg=0.01, longitude_deg=0.0, altitude_m=100.0),
            Waypoint(
                id=2, name="WP2", latitude_deg=0.01, longitude_deg=0.01, altitude_m=100.0
            ),
        ),
    )


def test_from_mission_builds_expected_segments() -> None:
    manager = PathManager.from_mission(_demo_mission())
    assert len(manager.segments) == 2
    assert manager.planned_path_ned().shape == (3, 3)  # home + 2 leg ends


def test_loiter_mission_segment_layout() -> None:
    mission = Mission(
        name="loiter_demo",
        home=HomePosition(0.0, 0.0, 0.0),
        waypoints=(
            Waypoint(
                id=1,
                name="LOIT",
                latitude_deg=0.01,
                longitude_deg=0.0,
                altitude_m=150.0,
                airspeed_mps=20.0,
                action=WaypointAction.LOITER,
                loiter_radius_m=80.0,
                loiter_duration_s=5.0,
            ),
        ),
    )
    manager = PathManager.from_mission(mission)
    assert [s.kind.value for s in manager.segments] == ["line", "orbit"]
    assert len(manager.loiter_circles()) == 1


def test_turn_anticipation_switches_before_reaching_waypoint() -> None:
    manager = PathManager.from_mission(_demo_mission())
    wp1_ned = manager.segments[0].end_ned_m  # type: ignore[attr-defined]
    # Position past the bisector half-plane but > acceptance radius (5 m) from WP1.
    position = wp1_ned + np.array([10.0, -5.0, 0.0])
    status = manager.update(position, dt_s=1.0)
    assert status.just_advanced
    assert status.active_index == 1
    assert status.active_waypoint_id == 2


def test_fly_over_requires_proximity_not_just_plane() -> None:
    mission = _demo_mission()
    # Make WP1 a fly-over (change_altitude) so anticipation does not apply.
    mission = mission.with_waypoints(
        (
            Waypoint(
                id=1,
                name="WP1",
                latitude_deg=0.01,
                longitude_deg=0.0,
                altitude_m=100.0,
                action=WaypointAction.CHANGE_ALTITUDE,
            ),
            mission.waypoints[1],
        )
    )
    manager = PathManager.from_mission(mission)
    wp1_ned = manager.segments[0].end_ned_m  # type: ignore[attr-defined]
    far = wp1_ned + np.array([10.0, -5.0, 0.0])  # outside 5 m radius
    assert not manager.update(far, dt_s=1.0).just_advanced
    near = wp1_ned + np.array([2.0, 0.0, 0.0])  # within 5 m radius and altitude
    assert manager.update(near, dt_s=1.0).just_advanced


def test_march_completes_mission_without_chatter() -> None:
    manager = PathManager.from_mission(_demo_mission())
    vertices = manager.planned_path_ned()
    indices: list[int] = []
    rng = np.random.default_rng(2024)
    # Walk the planned polyline in small steps with bounded position noise.
    for start, end in pairwise(vertices):
        for alpha in np.linspace(0.0, 1.0, 40):
            noise = rng.uniform(-1.0, 1.0, size=3)
            noise[2] = 0.0
            position = start + alpha * (end - start) + noise
            status = manager.update(position, dt_s=0.1)
            indices.append(status.active_index)
    assert indices == sorted(indices)  # monotonic: no backward switching
    assert manager.update(vertices[-1], dt_s=0.1).mission_complete


def test_loiter_dwell_holds_then_completes() -> None:
    mission = Mission(
        name="loiter_dwell",
        home=HomePosition(0.0, 0.0, 0.0),
        waypoints=(
            Waypoint(
                id=1,
                name="LOIT",
                latitude_deg=0.01,
                longitude_deg=0.0,
                altitude_m=150.0,
                airspeed_mps=20.0,
                action=WaypointAction.LOITER,
                loiter_radius_m=80.0,
                loiter_duration_s=5.0,
            ),
        ),
    )
    manager = PathManager.from_mission(mission, config=PathManagerConfig(min_dwell_s=0.0))
    center = manager.loiter_circles()[0].center_ned_m
    # Reach the loiter fix -> advance the approach line to the orbit.
    manager.update(center, dt_s=1.0)
    on_circle = center + np.array([80.0, 0.0, 0.0])
    # Orbit for less than the dwell time: not complete yet.
    for _ in range(4):
        assert not manager.update(on_circle, dt_s=1.0).mission_complete
    # Crossing the 5 s dwell threshold completes the (final) loiter leg.
    assert manager.update(on_circle, dt_s=1.0).mission_complete


def test_from_mission_rejects_invalid_mission() -> None:
    bad = Mission(
        name="bad",
        home=HomePosition(0.0, 0.0, 0.0),
        waypoints=(
            Waypoint(
                id=1,
                name="FAST",
                latitude_deg=0.01,
                longitude_deg=0.0,
                altitude_m=100.0,
                airspeed_mps=999.0,
            ),
        ),
    )
    with pytest.raises(ValueError):
        PathManager.from_mission(bad)
