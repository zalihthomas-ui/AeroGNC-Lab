"""Unit tests for the safety manager."""

import numpy as np

from aerognc.mathematics.quaternion import euler321_to_quaternion
from aerognc.mission.mission_manager import SafetyResponse
from aerognc.mission.safety import SafetyLimits, SafetyManager
from aerognc.navigation.state import NavigationState


def _state(ned=(0.0, 0.0, -100.0), roll=0.0, pitch=0.0, airspeed=20.0, valid=True):
    return NavigationState(
        position_ned_m=np.asarray(ned, dtype=float),
        velocity_ned_mps=np.array([airspeed, 0.0, 0.0]),
        quaternion_nb=euler321_to_quaternion(roll, pitch, 0.0),
        angular_rate_body_radps=np.zeros(3),
        airspeed_mps=airspeed,
        valid=valid,
    )


def test_nominal_state_is_safe() -> None:
    verdict = SafetyManager().check(_state(), 1.0)
    assert verdict.response is SafetyResponse.NONE
    assert not verdict.triggered


def test_geofence_breach_recommends_return_home() -> None:
    manager = SafetyManager(SafetyLimits(geofence_radius_m=1000.0))
    verdict = manager.check(_state(ned=(2000.0, 0.0, -100.0)), 2.0)
    assert verdict.response is SafetyResponse.RETURN_HOME
    assert any(e.trigger == "geofence" for e in verdict.events)


def test_min_altitude_breach_aborts() -> None:
    manager = SafetyManager(SafetyLimits(min_altitude_m=20.0))
    verdict = manager.check(_state(ned=(0.0, 0.0, -5.0)), 3.0)  # altitude 5 m < 20 m
    assert verdict.response is SafetyResponse.ABORT


def test_navigation_invalid_returns_home() -> None:
    verdict = SafetyManager().check(_state(valid=False), 5.0)
    assert verdict.response is SafetyResponse.RETURN_HOME


def test_bank_exceedance_is_advisory_limit() -> None:
    manager = SafetyManager(SafetyLimits(max_bank_rad=np.deg2rad(30.0)))
    verdict = manager.check(_state(roll=np.deg2rad(50.0)), 6.0)
    assert verdict.response is SafetyResponse.LIMIT
    assert any(e.trigger == "bank" for e in verdict.events)


def test_cross_track_breach_returns_home() -> None:
    manager = SafetyManager(SafetyLimits(max_cross_track_m=100.0))
    verdict = manager.check(_state(), 7.0, cross_track_error_m=250.0)
    assert verdict.response is SafetyResponse.RETURN_HOME


def test_most_severe_response_wins() -> None:
    manager = SafetyManager(SafetyLimits(min_altitude_m=20.0, max_bank_rad=np.deg2rad(30.0)))
    # Both a bank LIMIT and a min-altitude ABORT trigger; ABORT is more severe.
    verdict = manager.check(_state(ned=(0.0, 0.0, -5.0), roll=np.deg2rad(50.0)), 8.0)
    assert verdict.response is SafetyResponse.ABORT
    assert len(verdict.events) >= 2


def test_events_accumulate() -> None:
    manager = SafetyManager(SafetyLimits(geofence_radius_m=100.0))
    manager.check(_state(ned=(500.0, 0.0, -100.0)), 1.0)
    manager.check(_state(ned=(600.0, 0.0, -100.0)), 2.0)
    assert len(manager.events) == 2
