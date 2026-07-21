"""Unit tests for rendezvous / proximity-operations relative dynamics."""

import numpy as np
import pytest

from aerognc.astrodynamics.relative_motion import (
    EARTH_MU_M3_S2,
    ClohessyWiltshireModel,
    cw_state_transition,
    mean_motion_radps,
    orbit_change_from_impulse,
    simulate_rendezvous,
)

# ~500 km circular orbit.
RADIUS_M = 6_878_137.0
MODEL = ClohessyWiltshireModel.from_orbit(RADIUS_M)


def test_mean_motion_matches_known_period() -> None:
    n = mean_motion_radps(EARTH_MU_M3_S2, RADIUS_M)
    period_s = 2.0 * np.pi / n
    assert 5400.0 < period_s < 5800.0  # ~94 min LEO period


def test_stm_at_zero_is_identity() -> None:
    assert np.allclose(cw_state_transition(MODEL.mean_motion_n, 0.0), np.eye(6))


def test_stationary_offset_stays_bounded() -> None:
    # A pure along-track offset with zero relative velocity stays bounded (it
    # drifts but does not blow up over one orbit).
    state = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    period_s = 2.0 * np.pi / MODEL.mean_motion_n
    propagated = MODEL.propagate(state, period_s)
    assert np.linalg.norm(propagated[:3]) < 1000.0


def test_two_impulse_rendezvous_reaches_hold_point() -> None:
    initial = np.array([200.0, -500.0, 50.0, 0.0, 0.0, 0.0])
    hold_point = np.array([0.0, -100.0, 0.0])  # 100 m behind on the V-bar
    leg = MODEL.two_impulse_rendezvous(initial, hold_point, transfer_time_s=600.0)

    # Apply the departure burn and coast; we must arrive at the hold point.
    state_after_burn = initial.copy()
    state_after_burn[3:] += leg.departure_burn.delta_v_lvlh_mps
    arrived = MODEL.propagate(state_after_burn, 600.0)
    assert np.allclose(arrived[:3], hold_point, atol=1e-6)
    # Arrival burn nulls the relative velocity (station-keep).
    final_velocity = arrived[3:] + leg.arrival_burn.delta_v_lvlh_mps
    assert np.allclose(final_velocity, np.zeros(3), atol=1e-6)
    assert leg.total_delta_v_mps > 0.0


def test_closest_approach_reports_minimum() -> None:
    state = np.array([0.0, 500.0, 0.0, 0.0, -0.1, 0.0])  # closing along-track
    min_range, min_time = MODEL.closest_approach(state, horizon_s=3000.0)
    assert min_range < 500.0
    assert 0.0 <= min_time <= 3000.0


def test_simulate_rendezvous_approaches_target() -> None:
    initial = np.array([300.0, -800.0, 0.0, 0.0, 0.0, 0.0])
    # A safe stepped V-bar approach: 300 m -> 100 m -> 30 m behind the target.
    hold_points = [
        np.array([0.0, -300.0, 0.0]),
        np.array([0.0, -100.0, 0.0]),
        np.array([0.0, -30.0, 0.0]),
    ]
    trajectory = simulate_rendezvous(MODEL, initial, hold_points, leg_time_s=500.0)
    assert trajectory.states_lvlh.shape[1] == 6
    assert trajectory.total_delta_v_mps > 0.0
    # Ends within a few metres of the final hold point (30 m behind).
    final_position = trajectory.states_lvlh[-1, :3]
    assert np.linalg.norm(final_position - np.array([0.0, -30.0, 0.0])) < 5.0
    # Never passes through the target closer than the final hold distance minus margin.
    assert trajectory.closest_approach_m > 5.0


def test_prograde_burn_raises_apoapsis() -> None:
    # Circular orbit in the equatorial plane; +y is the velocity direction.
    speed = float(np.sqrt(EARTH_MU_M3_S2 / RADIUS_M))
    position = np.array([RADIUS_M, 0.0, 0.0])
    velocity = np.array([0.0, speed, 0.0])
    change = orbit_change_from_impulse(position, velocity, np.array([0.0, 20.0, 0.0]))
    assert change.apoapsis_altitude_after_m > change.apoapsis_altitude_before_m
    assert change.semi_major_axis_after_m > change.semi_major_axis_before_m


def test_retrograde_burn_lowers_orbit() -> None:
    speed = float(np.sqrt(EARTH_MU_M3_S2 / RADIUS_M))
    position = np.array([RADIUS_M, 0.0, 0.0])
    velocity = np.array([0.0, speed, 0.0])
    change = orbit_change_from_impulse(position, velocity, np.array([0.0, -20.0, 0.0]))
    assert change.periapsis_altitude_after_m < change.periapsis_altitude_before_m


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        mean_motion_radps(EARTH_MU_M3_S2, -1.0)
    with pytest.raises(ValueError):
        MODEL.two_impulse_rendezvous(np.zeros(6), np.zeros(3), transfer_time_s=0.0)
