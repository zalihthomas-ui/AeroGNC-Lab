import numpy as np
import pytest

from aerognc.astrodynamics.multirevolution import (
    search_lambert_transfers,
    solve_lambert_revolutions,
)

MU_EARTH_M3_S2 = 3.986004418e14


def test_one_revolution_solver_recovers_circular_orbit_and_verifies_endpoint() -> None:
    radius_m = 7.0e6
    circular_speed_mps = np.sqrt(MU_EARTH_M3_S2 / radius_m)
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    departure = np.array([radius_m, 0.0, 0.0])
    arrival = np.array([0.0, radius_m, 0.0])
    solutions = solve_lambert_revolutions(
        departure,
        arrival,
        1.25 * period_s,
        MU_EARTH_M3_S2,
        revolutions=1,
        direction="prograde",
        endpoint_tolerance_m=0.1,
    )

    assert len(solutions) == 2
    circular = min(
        solutions,
        key=lambda item: np.linalg.norm(
            item.departure_velocity_mps - [0.0, circular_speed_mps, 0.0]
        ),
    )
    np.testing.assert_allclose(
        circular.departure_velocity_mps, [0.0, circular_speed_mps, 0.0], atol=2.0e-5
    )
    assert circular.endpoint_position_error_m < 0.1
    assert circular.endpoint_velocity_error_mps < 1.0e-6


def test_search_evaluates_both_directions_and_revolutions_with_stable_ranking() -> None:
    radius_m = 7.0e6
    speed_mps = np.sqrt(MU_EARTH_M3_S2 / radius_m)
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / MU_EARTH_M3_S2)
    first = search_lambert_transfers(
        [radius_m, 0.0, 0.0],
        [0.0, speed_mps, 0.0],
        [0.0, radius_m, 0.0],
        [-speed_mps, 0.0, 0.0],
        1.25 * period_s,
        MU_EARTH_M3_S2,
        revolutions=(0, 1),
    )
    second = search_lambert_transfers(
        [radius_m, 0.0, 0.0],
        [0.0, speed_mps, 0.0],
        [0.0, radius_m, 0.0],
        [-speed_mps, 0.0, 0.0],
        1.25 * period_s,
        MU_EARTH_M3_S2,
        revolutions=(0, 1),
    )

    assert first.attempted_geometry_count == 4
    assert {(item.solution.direction, item.solution.revolutions) for item in first.candidates} >= {
        ("prograde", 0),
        ("retrograde", 0),
        ("prograde", 1),
    }
    assert first.best.solution.revolutions == 1
    assert first.best.solution.direction == "prograde"
    assert [item.objective_mps for item in first.candidates] == [
        item.objective_mps for item in second.candidates
    ]
    assert [item.rank for item in first.candidates] == list(range(1, len(first.candidates) + 1))


def test_multirevolution_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError, match="revolutions"):
        solve_lambert_revolutions(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            1.0,
            revolutions=-1,
            direction="prograde",
        )
    with pytest.raises(ValueError, match="unique"):
        search_lambert_transfers(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            10.0,
            1.0,
            revolutions=(0, 0),
        )
