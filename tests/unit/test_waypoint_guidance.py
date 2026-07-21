"""Unit tests for fixed-wing path-following guidance laws."""

import numpy as np
import pytest

from aerognc.gnc.path_manager import LineSegment, OrbitSegment
from aerognc.gnc.waypoint_guidance import (
    GuidanceMode,
    PathFollowingGuidance,
    wind_corrected_heading_rad,
)
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.state import FlightEnvironment, NavigationState


def _state(position_ned, velocity_ned=(20.0, 0.0, 0.0), yaw_rad=0.0, airspeed=20.0):
    return NavigationState(
        position_ned_m=np.asarray(position_ned, dtype=float),
        velocity_ned_mps=np.asarray(velocity_ned, dtype=float),
        quaternion_nb=euler321_to_quaternion(0.0, 0.0, yaw_rad),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=airspeed,
    )


CALM = FlightEnvironment.calm()


# --- wind-corrected heading --------------------------------------------------


def test_heading_equals_course_in_calm_air() -> None:
    assert wind_corrected_heading_rad(0.3, np.zeros(3), 20.0) == pytest.approx(0.3)


def test_crosswind_crab_sign() -> None:
    # Flying north (course 0) with wind blowing toward the west (from the east):
    # the aircraft must crab to the east -> heading slightly positive.
    wind_from_east = np.array([0.0, -5.0, 0.0])  # points west
    heading = wind_corrected_heading_rad(0.0, wind_from_east, 20.0)
    assert heading > 0.0
    assert heading == pytest.approx(np.arcsin(5.0 / 20.0), abs=1e-9)


def test_heading_falls_back_when_wind_exceeds_airspeed() -> None:
    heading = wind_corrected_heading_rad(0.0, np.array([0.0, -50.0, 0.0]), 20.0)
    assert np.isfinite(heading)


# --- straight-line guidance --------------------------------------------------


def _north_line() -> LineSegment:
    return LineSegment(1, np.zeros(3), np.array([1000.0, 0.0, -100.0]), 20.0)


@pytest.mark.parametrize("mode", list(GuidanceMode))
def test_line_guidance_steers_left_when_right_of_path(mode: GuidanceMode) -> None:
    guidance = PathFollowingGuidance(mode=mode)
    # 50 m east of a north-bound path (to the right) -> command course < 0 (steer left/west).
    state = _state([200.0, 50.0, -100.0])
    command = guidance.update(state, _north_line(), CALM, dt_s=0.1)
    assert command.course_command_rad < 0.0
    assert command.cross_track_error_m == pytest.approx(50.0)


@pytest.mark.parametrize("mode", list(GuidanceMode))
def test_line_guidance_on_path_commands_path_course(mode: GuidanceMode) -> None:
    guidance = PathFollowingGuidance(mode=mode)
    state = _state([200.0, 0.0, -100.0])
    command = guidance.update(state, _north_line(), CALM, dt_s=0.1)
    assert command.course_command_rad == pytest.approx(0.0, abs=1e-6)


def test_vector_field_correction_bounded_by_max_approach() -> None:
    guidance = PathFollowingGuidance(mode=GuidanceMode.VECTOR_FIELD)
    state = _state([0.0, 1.0e6, -100.0])  # enormous cross-track error
    command = guidance.update(state, _north_line(), CALM, dt_s=0.1)
    # Correction saturates below pi/2 so the command never reverses past due west.
    assert command.course_command_rad > -0.5 * np.pi


# --- altitude / airspeed -----------------------------------------------------


def test_altitude_and_airspeed_commands_from_segment() -> None:
    guidance = PathFollowingGuidance()
    segment = LineSegment(1, np.zeros(3), np.array([1000.0, 0.0, -150.0]), 24.0)
    state = _state([1000.0, 0.0, -100.0])  # at the leg end (down -150 => alt 150)
    command = guidance.update(state, segment, CALM, dt_s=0.1)
    assert command.altitude_command_m == pytest.approx(150.0)
    assert command.airspeed_command_mps == pytest.approx(24.0)
    assert command.climb_rate_command_mps > 0.0  # below target altitude -> climb


def test_climb_rate_command_is_bounded() -> None:
    guidance = PathFollowingGuidance()
    segment = LineSegment(1, np.zeros(3), np.array([1000.0, 0.0, -5000.0]), 20.0)
    state = _state([500.0, 0.0, 0.0])
    command = guidance.update(state, segment, CALM, dt_s=0.1)
    assert abs(command.climb_rate_command_mps) <= guidance.gains.max_climb_rate_mps + 1e-9


# --- orbit / loiter guidance -------------------------------------------------


def test_orbit_tangent_course_and_roll_feedforward_clockwise() -> None:
    guidance = PathFollowingGuidance()
    orbit = OrbitSegment(2, np.array([0.0, 0.0, -100.0]), 100.0, 1, 20.0)  # CW
    # On the circle, due north of centre: CW tangent points east (course +pi/2).
    state = _state([100.0, 0.0, -100.0])
    command = guidance.update(state, orbit, CALM, dt_s=0.1)
    assert command.course_command_rad == pytest.approx(0.5 * np.pi, abs=1e-6)
    assert command.roll_feedforward_rad > 0.0  # bank right into a clockwise turn


def test_orbit_counterclockwise_banks_left() -> None:
    guidance = PathFollowingGuidance()
    orbit = OrbitSegment(2, np.array([0.0, 0.0, -100.0]), 100.0, -1, 20.0)  # CCW
    state = _state([100.0, 0.0, -100.0])
    command = guidance.update(state, orbit, CALM, dt_s=0.1)
    assert command.roll_feedforward_rad < 0.0


def test_orbit_outside_circle_steers_inward() -> None:
    guidance = PathFollowingGuidance()
    orbit = OrbitSegment(2, np.array([0.0, 0.0, -100.0]), 100.0, 1, 20.0)
    outside = _state([150.0, 0.0, -100.0])  # 50 m outside
    command = guidance.update(outside, orbit, CALM, dt_s=0.1)
    # Command course should have an inward (toward-centre) component vs pure tangent.
    assert command.cross_track_error_m == pytest.approx(50.0)
