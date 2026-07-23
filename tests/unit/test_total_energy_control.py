"""Unit tests for bumpless fixed-wing total-energy control."""

import numpy as np
import pytest

import aerognc.gnc as gnc
from aerognc.gnc.total_energy_control import (
    TotalEnergyControlGains,
    TotalEnergyController,
)
from aerognc.gnc.waypoint_guidance import GuidanceCommand
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.state import NavigationState


def _state(*, altitude_m: float = 100.0, airspeed_mps: float = 20.0) -> NavigationState:
    return NavigationState(
        position_ned_m=np.array([0.0, 0.0, -altitude_m]),
        velocity_ned_mps=np.array([airspeed_mps, 0.0, 0.0]),
        quaternion_nb=euler321_to_quaternion(0.0, 0.0, 0.0),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=airspeed_mps,
    )


def _guidance(
    *, altitude_m: float = 100.0, airspeed_mps: float = 20.0, climb_rate_mps: float = 0.0
) -> GuidanceCommand:
    return GuidanceCommand(
        course_command_rad=0.0,
        heading_command_rad=0.0,
        altitude_command_m=altitude_m,
        airspeed_command_mps=airspeed_mps,
        climb_rate_command_mps=climb_rate_mps,
        roll_feedforward_rad=0.0,
        cross_track_error_m=0.0,
        distance_to_waypoint_m=100.0,
        along_track_fraction=0.5,
    )


def test_total_energy_activation_is_bumpless_and_resettable() -> None:
    controller = TotalEnergyController(pitch_trim_rad=0.04, throttle_trim=0.3)
    activated = controller.activate(
        _guidance(altitude_m=200.0, climb_rate_mps=4.0),
        _state(),
        pitch_command_rad=0.08,
        throttle_command=0.45,
    )
    assert activated.pitch_command_rad == pytest.approx(0.08)
    assert activated.throttle_command == pytest.approx(0.45)
    assert activated.total_energy_error_m2ps2 == 0.0

    controller.reset()
    first = controller.update(_guidance(altitude_m=200.0), _state(), 0.05)
    assert first.pitch_command_rad == pytest.approx(0.04)
    assert first.throttle_command == pytest.approx(0.3)


def test_total_energy_channels_coordinate_altitude_and_airspeed() -> None:
    climb = TotalEnergyController(pitch_trim_rad=0.04, throttle_trim=0.3)
    climb.update(_guidance(altitude_m=150.0), _state(), 0.1)
    climb_output = climb.update(_guidance(altitude_m=150.0), _state(), 0.1)
    assert climb_output.total_energy_error_m2ps2 > 0.0
    assert climb_output.energy_balance_error_m2ps2 > 0.0
    assert climb_output.pitch_command_rad > 0.04
    assert climb_output.throttle_command > 0.3

    accelerate = TotalEnergyController(pitch_trim_rad=0.04, throttle_trim=0.3)
    accelerate.update(_guidance(airspeed_mps=25.0), _state(), 0.1)
    speed_output = accelerate.update(_guidance(airspeed_mps=25.0), _state(), 0.1)
    assert speed_output.kinetic_energy_error_m2ps2 > 0.0
    assert speed_output.total_energy_error_m2ps2 > 0.0
    assert speed_output.energy_balance_error_m2ps2 < 0.0
    assert speed_output.pitch_command_rad < 0.04
    assert speed_output.throttle_command > 0.3


def test_total_energy_limits_and_anti_windup_remain_bounded() -> None:
    controller = TotalEnergyController(
        TotalEnergyControlGains(
            altitude_reference_rate_limit_mps=100.0,
            airspeed_reference_rate_limit_mps2=100.0,
        ),
        pitch_trim_rad=0.0,
        throttle_trim=0.5,
        pitch_limit_rad=0.2,
        throttle_delta_limit=0.3,
    )
    guidance = _guidance(altitude_m=10_000.0, airspeed_mps=100.0, climb_rate_mps=50.0)
    output = controller.update(guidance, _state(), 0.1)
    for _ in range(500):
        output = controller.update(guidance, _state(), 0.1)
    assert -0.2 <= output.pitch_command_rad <= 0.2
    assert 0.0 <= output.throttle_command <= 1.0
    assert output.pitch_saturated
    assert output.throttle_saturated


def test_total_energy_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        TotalEnergyControlGains(total_energy_kp=-1.0)
    with pytest.raises(ValueError, match="pitch trim"):
        TotalEnergyController(pitch_trim_rad=1.0, pitch_limit_rad=0.2)
    with pytest.raises(ValueError, match="positive"):
        TotalEnergyController().update(_guidance(), _state(), 0.0)


def test_total_energy_and_path_types_are_available_from_public_gnc_surface() -> None:
    assert gnc.TotalEnergyController is TotalEnergyController
    assert gnc.TotalEnergyControlGains is TotalEnergyControlGains
    assert gnc.LongitudinalControlMode.TOTAL_ENERGY.value == "total_energy"
    assert gnc.FilletSegment.__name__ == "FilletSegment"
    assert gnc.WaypointEnvelopeReference.__name__ == "WaypointEnvelopeReference"
