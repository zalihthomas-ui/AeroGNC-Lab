from pathlib import Path

import pytest

from aerognc.astrodynamics.launch_window import optimize_launch_window
from aerognc.configuration.planetary_catalog import load_planetary_catalog


def test_launch_window_refinement_is_feasible_deterministic_and_nonworsening() -> None:
    catalog = load_planetary_catalog(Path("configs/fictional_planetary_system.yaml"))
    arguments = dict(
        departure_bounds_s=(0.0, 120.0 * 86_400.0),
        arrival_bounds_s=(200.0 * 86_400.0, 420.0 * 86_400.0),
        departure_grid_count=6,
        arrival_grid_count=7,
        departure_parking_altitude_m=300_000.0,
        arrival_parking_altitude_m=300_000.0,
        maximum_c3_m2_s2=40.0e6,
        maximum_arrival_excess_speed_mps=8_000.0,
        maximum_refinement_iterations=30,
        epoch_tolerance_s=1_200.0,
    )
    first = optimize_launch_window(
        catalog.body("Asteria"),
        catalog.body("Neria"),
        catalog.primary.gravitational_parameter_m3_s2,
        **arguments,
    )
    second = optimize_launch_window(
        catalog.body("Asteria"),
        catalog.body("Neria"),
        catalog.primary.gravitational_parameter_m3_s2,
        **arguments,
    )

    assert first.converged and first.optimum.feasible
    assert first.optimum.total_delta_v_mps < 7_500.0
    assert first.optimum.departure_time_s == pytest.approx(second.optimum.departure_time_s)
    assert first.optimum.arrival_time_s == pytest.approx(second.optimum.arrival_time_s)
    assert first.optimum.total_delta_v_mps == pytest.approx(second.optimum.total_delta_v_mps)
    assert first.optimum.total_delta_v_mps <= min(item.total_delta_v_mps for item in first.history)
