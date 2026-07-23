"""Unit tests for the cascaded fixed-wing autopilot."""

import numpy as np
import pytest

from aerognc.gnc.fixedwing_autopilot import (
    AutopilotGains,
    AutopilotTrim,
    FixedWingAutopilot,
    LongitudinalControlMode,
)
from aerognc.gnc.waypoint_guidance import GuidanceCommand
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.state import NavigationState


def _state(roll=0.0, pitch=0.0, yaw=0.0, rates=(0.0, 0.0, 0.0), airspeed=20.0, down=-100.0):
    return NavigationState(
        position_ned_m=np.array([0.0, 0.0, down]),
        velocity_ned_mps=np.array([airspeed * np.cos(yaw), airspeed * np.sin(yaw), 0.0]),
        quaternion_nb=euler321_to_quaternion(roll, pitch, yaw),
        angular_rate_body_radps=np.asarray(rates, dtype=float),
        airspeed_mps=airspeed,
    )


def _guidance(course=0.0, altitude=100.0, airspeed=20.0, roll_ff=0.0):
    return GuidanceCommand(
        course_command_rad=course,
        heading_command_rad=course,
        altitude_command_m=altitude,
        airspeed_command_mps=airspeed,
        climb_rate_command_mps=0.0,
        roll_feedforward_rad=roll_ff,
        cross_track_error_m=0.0,
        distance_to_waypoint_m=100.0,
        along_track_fraction=0.5,
    )


def test_course_error_commands_roll_toward_target() -> None:
    autopilot = FixedWingAutopilot()
    # Flying north, commanded to turn east (course +) -> should bank right (roll > 0).
    out = autopilot.update(_guidance(course=0.5), _state(yaw=0.0), dt_s=0.1)
    assert out.control.roll_command_rad > 0.0
    assert out.actuator.aileron > 0.0


def test_roll_command_is_bank_limited() -> None:
    gains = AutopilotGains(bank_limit_rad=np.deg2rad(30.0))
    autopilot = FixedWingAutopilot(gains)
    out = autopilot.update(_guidance(course=np.pi), _state(yaw=0.0), dt_s=0.1)
    assert abs(out.control.roll_command_rad) <= np.deg2rad(30.0) + 1e-9


def test_roll_rate_damping_opposes_aileron() -> None:
    autopilot = FixedWingAutopilot()
    still = autopilot.update(_guidance(course=0.3), _state(yaw=0.0), dt_s=0.1)
    autopilot.reset()
    rolling = autopilot.update(
        _guidance(course=0.3), _state(yaw=0.0, rates=(2.0, 0.0, 0.0)), dt_s=0.1
    )
    assert rolling.actuator.aileron < still.actuator.aileron


def test_low_altitude_commands_nose_up_and_up_elevator() -> None:
    autopilot = FixedWingAutopilot()
    out = autopilot.update(_guidance(altitude=300.0), _state(down=-100.0), dt_s=0.1)
    assert out.altitude_error_m == pytest.approx(200.0)
    assert out.control.pitch_command_rad > 0.0
    assert out.actuator.elevator > 0.0


def test_pitch_command_is_limited() -> None:
    gains = AutopilotGains(pitch_limit_rad=np.deg2rad(15.0))
    autopilot = FixedWingAutopilot(gains)
    out = autopilot.update(_guidance(altitude=100000.0), _state(down=0.0), dt_s=0.1)
    assert abs(out.control.pitch_command_rad) <= np.deg2rad(15.0) + 1e-9


def test_slow_airspeed_increases_throttle_within_bounds() -> None:
    autopilot = FixedWingAutopilot()
    out = autopilot.update(_guidance(airspeed=30.0), _state(airspeed=20.0), dt_s=0.1)
    assert out.control.throttle_command > autopilot.gains.throttle_trim
    assert 0.0 <= out.control.throttle_command <= 1.0


def test_yaw_damper_opposes_yaw_rate() -> None:
    autopilot = FixedWingAutopilot()
    out = autopilot.update(_guidance(), _state(rates=(0.0, 0.0, 0.5)), dt_s=0.1)
    assert out.actuator.rudder < 0.0


def test_outputs_are_finite_and_clipped() -> None:
    autopilot = FixedWingAutopilot()
    guidance = _guidance(course=3.0, altitude=9999.0, airspeed=99.0)
    out = autopilot.update(guidance, _state(), dt_s=0.1)
    for value in (out.actuator.aileron, out.actuator.elevator, out.actuator.rudder):
        assert -1.0 <= value <= 1.0
    assert 0.0 <= out.actuator.throttle <= 1.0


def test_reset_clears_integrators() -> None:
    autopilot = FixedWingAutopilot()
    for _ in range(20):
        autopilot.update(_guidance(course=0.4), _state(), dt_s=0.1)
    autopilot.reset()
    out = autopilot.update(_guidance(course=0.0), _state(), dt_s=0.1)
    # With zero error immediately after reset the roll command is only feedforward (0).
    assert out.control.roll_command_rad == pytest.approx(0.0, abs=1e-9)


def test_total_energy_mode_starts_at_trim_then_coordinates_climb() -> None:
    trim = AutopilotTrim(pitch_rad=0.04, elevator_command=0.12, throttle=0.30)
    autopilot = FixedWingAutopilot(
        longitudinal_mode=LongitudinalControlMode.TOTAL_ENERGY,
        trim=trim,
    )
    guidance = _guidance(altitude=200.0)
    first = autopilot.update(guidance, _state(pitch=0.04), dt_s=0.1)
    second = autopilot.update(guidance, _state(pitch=0.04), dt_s=0.1)

    assert first.control.pitch_command_rad == pytest.approx(trim.pitch_rad)
    assert first.control.throttle_command == pytest.approx(trim.throttle)
    assert first.longitudinal.mode == "total_energy"
    assert second.longitudinal.total_energy_error_m2ps2 > 0.0
    assert second.control.pitch_command_rad > trim.pitch_rad
    assert second.control.throttle_command > trim.throttle
    assert first.actuator.elevator == pytest.approx(trim.elevator_command)


def test_longitudinal_mode_switch_is_bumpless() -> None:
    autopilot = FixedWingAutopilot()
    before = autopilot.update(_guidance(altitude=120.0, airspeed=22.0), _state(), 0.1)
    autopilot.set_longitudinal_mode(LongitudinalControlMode.TOTAL_ENERGY)
    after = autopilot.update(_guidance(altitude=120.0, airspeed=22.0), _state(), 0.1)

    assert after.control.pitch_command_rad == pytest.approx(before.control.pitch_command_rad)
    assert after.control.throttle_command == pytest.approx(before.control.throttle_command)
    assert autopilot.provenance()["longitudinal_mode"] == "total_energy"


def test_autopilot_rejects_invalid_trim_and_mode() -> None:
    with pytest.raises(ValueError, match="elevator trim"):
        AutopilotTrim(elevator_command=2.0)
    with pytest.raises(ValueError, match="pitch trim exceeds"):
        FixedWingAutopilot(
            AutopilotGains(pitch_limit_rad=0.1),
            trim=AutopilotTrim(pitch_rad=0.2),
        )
    with pytest.raises(ValueError, match="LongitudinalControlMode"):
        FixedWingAutopilot(longitudinal_mode="total_energy")  # type: ignore[arg-type]
