"""Unit tests for controller-facing waypoint envelope margins."""

from pathlib import Path

import numpy as np
import pytest

from aerognc.configuration.aircraft_loader import load_aircraft_configuration
from aerognc.gnc.waypoint_envelope import (
    WaypointEnvelopeReference,
    evaluate_waypoint_envelope,
)
from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.navigation.state import NavigationState
from aerognc.vehicle.control_surfaces import SurfaceDeflections

REPO_ROOT = Path(__file__).resolve().parents[2]


def _state(*, roll_rad: float = 0.0, airspeed_mps: float = 20.0) -> NavigationState:
    return NavigationState(
        position_ned_m=np.array([0.0, 0.0, -100.0]),
        velocity_ned_mps=np.array([airspeed_mps, 0.0, 0.0]),
        quaternion_nb=euler321_to_quaternion(roll_rad, 0.0, 0.0),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=airspeed_mps,
    )


def _reference(*, coefficient: bool = False) -> WaypointEnvelopeReference:
    configuration = (
        load_aircraft_configuration(REPO_ROOT / "configs" / "aircraft_waypoint_uav.yaml")
        if coefficient
        else None
    )
    return WaypointEnvelopeReference(
        minimum_altitude_m=5.0,
        maximum_altitude_m=3000.0,
        minimum_airspeed_mps=12.0,
        maximum_airspeed_mps=45.0,
        maximum_bank_rad=np.deg2rad(60.0),
        maximum_pitch_rad=np.deg2rad(45.0),
        aileron_limit_rad=np.deg2rad(22.0),
        elevator_limit_rad=np.deg2rad(25.0),
        rudder_limit_rad=np.deg2rad(28.0),
        coefficient_configuration=configuration,
    )


def test_reduced_envelope_reports_declared_stall_and_actuator_margins() -> None:
    margins = evaluate_waypoint_envelope(
        _state(),
        SurfaceDeflections(0.0, 0.0, 0.0, 0.3, saturated=False),
        _reference(),
    )
    assert margins.stall_speed_reference_mps == pytest.approx(12.0)
    assert margins.stall_margin_mps == pytest.approx(8.0)
    assert margins.load_factor == pytest.approx(1.0)
    assert margins.minimum_surface_margin_fraction == pytest.approx(1.0)
    assert margins.throttle_margin_fraction == pytest.approx(0.3)
    assert margins.lower_specific_energy_margin_m2ps2 > 0.0
    assert margins.upper_specific_energy_margin_m2ps2 > 0.0


def test_coefficient_stall_reference_increases_with_bank_load() -> None:
    deflections = SurfaceDeflections(0.1, -0.1, 0.0, 0.5, saturated=False)
    level = evaluate_waypoint_envelope(_state(), deflections, _reference(coefficient=True))
    banked = evaluate_waypoint_envelope(
        _state(roll_rad=np.deg2rad(45.0)),
        deflections,
        _reference(coefficient=True),
    )
    assert level.stall_reference_source.startswith("coefficient")
    assert banked.load_factor == pytest.approx(np.sqrt(2.0))
    assert banked.stall_speed_reference_mps > level.stall_speed_reference_mps
    assert banked.bank_margin_rad < level.bank_margin_rad


def test_envelope_reference_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValueError, match="airspeed envelope"):
        WaypointEnvelopeReference(
            minimum_altitude_m=0.0,
            maximum_altitude_m=100.0,
            minimum_airspeed_mps=20.0,
            maximum_airspeed_mps=10.0,
            maximum_bank_rad=0.5,
            maximum_pitch_rad=0.5,
            aileron_limit_rad=0.3,
            elevator_limit_rad=0.3,
            rudder_limit_rad=0.3,
        )
