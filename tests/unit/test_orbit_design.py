import numpy as np
import pytest

from aerognc.astrodynamics import (
    CircularOrbitBody,
    ClassicalOrbitalElements,
    compute_porkchop_grid,
    elements_to_state,
    propagate_universal,
    solve_lambert_universal,
    state_to_elements,
    stumpff_c,
    stumpff_s,
    target_b_plane,
)

MU_EARTH_M3_S2 = 3.986004418e14


def _planet(name: str, radius_m: float, phase_rad: float) -> CircularOrbitBody:
    return CircularOrbitBody(
        name=name,
        role="departure" if name == "Inner" else "destination",
        gravitational_parameter_m3_s2=3.986e14,
        radius_m=6.4e6,
        semi_major_axis_m=radius_m,
        phase_at_epoch_rad=phase_rad,
    )


def test_stumpff_origin_and_universal_quarter_orbit() -> None:
    radius_m = 7.0e6
    speed_mps = np.sqrt(MU_EARTH_M3_S2 / radius_m)
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)

    assert stumpff_c(0.0) == pytest.approx(0.5)
    assert stumpff_s(0.0) == pytest.approx(1.0 / 6.0)
    propagated = propagate_universal(
        [radius_m, 0.0, 0.0],
        [0.0, speed_mps, 0.0],
        period_s / 4.0,
        MU_EARTH_M3_S2,
    )

    np.testing.assert_allclose(propagated.position_m, [0.0, radius_m, 0.0], atol=2.0e-5)
    np.testing.assert_allclose(propagated.velocity_mps, [-speed_mps, 0.0, 0.0], atol=2.0e-8)


def test_orbital_element_round_trip_for_inclined_ellipse() -> None:
    expected = ClassicalOrbitalElements(
        semi_major_axis_m=12.0e6,
        eccentricity=0.23,
        inclination_rad=np.deg2rad(38.0),
        ascending_node_rad=np.deg2rad(74.0),
        argument_of_periapsis_rad=np.deg2rad(27.0),
        true_anomaly_rad=np.deg2rad(113.0),
    )
    position, velocity = elements_to_state(expected, MU_EARTH_M3_S2)
    recovered = state_to_elements(position, velocity, MU_EARTH_M3_S2)
    reconstructed_position, reconstructed_velocity = elements_to_state(recovered, MU_EARTH_M3_S2)

    np.testing.assert_allclose(reconstructed_position, position, rtol=1.0e-12, atol=1.0e-6)
    np.testing.assert_allclose(reconstructed_velocity, velocity, rtol=1.0e-12, atol=1.0e-9)


def test_lambert_endpoint_matches_independent_universal_propagation() -> None:
    radius_m = 7.0e6
    quarter_period_s = 0.5 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    departure = np.array([radius_m, 0.0, 0.0])
    arrival = np.array([0.0, radius_m, 0.0])

    solution = solve_lambert_universal(departure, arrival, quarter_period_s, MU_EARTH_M3_S2)
    propagated = propagate_universal(
        departure,
        solution.departure_velocity_mps,
        quarter_period_s,
        MU_EARTH_M3_S2,
    )

    np.testing.assert_allclose(propagated.position_m, arrival, atol=2.0e-3)
    np.testing.assert_allclose(propagated.velocity_mps, solution.arrival_velocity_mps, atol=2.0e-6)


def test_porkchop_grid_is_repeatable_and_reports_best_feasible_cell() -> None:
    primary_mu = 1.32712440018e20
    departure = _planet("Inner", 1.2e11, 0.0)
    destination = _planet("Outer", 1.8e11, 1.1)
    days = 86_400.0
    departure_times = np.array([0.0, 50.0 * days, 100.0 * days])
    arrival_times = np.array([200.0 * days, 300.0 * days, 400.0 * days])

    first = compute_porkchop_grid(
        departure,
        destination,
        primary_mu,
        departure_times,
        arrival_times,
    )
    second = compute_porkchop_grid(
        departure,
        destination,
        primary_mu,
        departure_times,
        arrival_times,
    )

    np.testing.assert_array_equal(first.objective_mps, second.objective_mps)
    departure_index, arrival_index = first.best_indices()
    assert first.feasible[departure_index, arrival_index]
    assert np.isfinite(first.objective_mps[departure_index, arrival_index])


def test_b_plane_geometry_distinguishes_unpowered_and_powered_flybys() -> None:
    unpowered = target_b_plane(
        [8_000.0, 0.0, 0.0],
        [0.0, 8_000.0, 0.0],
        body_mu_m3_s2=1.26686534e17,
        body_radius_m=6.9911e7,
        minimum_altitude_m=100_000.0,
    )
    powered = target_b_plane(
        [8_000.0, 0.0, 0.0],
        [0.0, 9_000.0, 0.0],
        body_mu_m3_s2=1.26686534e17,
        body_radius_m=6.9911e7,
        minimum_altitude_m=100_000.0,
    )

    assert unpowered.feasible_unpowered
    assert unpowered.powered_flyby_delta_v_mps == pytest.approx(0.0)
    assert np.linalg.norm(unpowered.b_vector_m) == pytest.approx(unpowered.impact_parameter_m)
    assert not powered.feasible_unpowered
    assert powered.powered_flyby_delta_v_mps == pytest.approx(1_000.0)
