from dataclasses import replace

import numpy as np
import pytest

from aerognc.configuration import load_orbit_sandbox_configuration
from aerognc.simulation.orbit_sandbox import (
    orbit_sandbox_payload,
    simulate_orbit_sandbox,
)


def _short_configuration(model: str):  # type: ignore[no-untyped-def]
    configuration = load_orbit_sandbox_configuration("configs/orbit_sandbox.yaml")
    return replace(
        configuration,
        model=model,
        duration_s=600.0,
        integration_step_s=2.0,
        output_step_s=20.0,
    )


def test_free_motion_matches_the_analytical_solution() -> None:
    configuration = _short_configuration("free")
    simulation = simulate_orbit_sandbox(configuration)
    columns = simulation.result.columns
    initial_position = np.array([columns[name][0] for name in ("x_m", "y_m", "z_m")])
    initial_velocity = np.array([columns[name][0] for name in ("vx_mps", "vy_mps", "vz_mps")])
    final_position = np.array([columns[name][-1] for name in ("x_m", "y_m", "z_m")])

    np.testing.assert_allclose(
        final_position,
        initial_position + configuration.duration_s * initial_velocity,
        rtol=1.0e-12,
        atol=1.0e-6,
    )
    assert "not an orbit" in orbit_sandbox_payload(simulation)["model_description"]


def test_two_body_circular_orbit_returns_after_one_period() -> None:
    base = _short_configuration("two_body")
    radius_m = base.primary.radius_m + base.initial.altitude_m
    period_s = 2.0 * np.pi * np.sqrt(radius_m**3 / base.primary.gravitational_parameter_m3_s2)
    configuration = replace(
        base,
        duration_s=float(period_s),
        integration_step_s=5.0,
        output_step_s=25.0,
    )
    simulation = simulate_orbit_sandbox(configuration)
    columns = simulation.result.columns
    initial = np.array([columns[name][0] for name in ("x_m", "y_m", "z_m")])
    final = np.array([columns[name][-1] for name in ("x_m", "y_m", "z_m")])

    assert np.linalg.norm(final - initial) < 100.0
    assert abs(columns["altitude_m"][-1] - columns["altitude_m"][0]) < 1.0
    assert abs(columns["revolutions_completed"][-1] - 1.0) < 1.0e-4


def test_restricted_and_full_n_body_modes_produce_finite_states() -> None:
    for model in ("restricted_three_body", "full_n_body"):
        simulation = simulate_orbit_sandbox(_short_configuration(model))
        assert np.all(np.isfinite(simulation.result.columns["radius_m"]))
        assert simulation.result.columns["radius_m"].size == 31
        assert simulation.result.time_s[-1] == 600.0


def test_drag_model_changes_energy_and_reports_only_finite_horizon_survival() -> None:
    base = _short_configuration("perturbed_decay")
    vacuum = simulate_orbit_sandbox(replace(base, atmosphere_density_scale=0.0))
    drag = simulate_orbit_sandbox(replace(base, atmosphere_density_scale=1.0e4))
    mu = base.primary.gravitational_parameter_m3_s2

    def specific_energy(simulation) -> float:  # type: ignore[no-untyped-def]
        columns = simulation.result.columns
        return float(0.5 * columns["speed_mps"][-1] ** 2 - mu / columns["radius_m"][-1])

    assert specific_energy(drag) < specific_energy(vacuum)
    assert np.max(drag.result.columns["drag_acceleration_mps2"]) > 0.0
    assert "not proven infinite" in vacuum.survival_statement


def test_default_decay_reports_the_exact_reentry_threshold() -> None:
    configuration = load_orbit_sandbox_configuration("configs/orbit_sandbox.yaml")
    simulation = simulate_orbit_sandbox(configuration)

    assert simulation.reentered
    event = next(
        item for item in simulation.result.event_summary if item["name"] == "reentry_threshold"
    )
    assert event["altitude_m"] == pytest.approx(configuration.reentry_altitude_m, abs=1.0e-8)
    assert simulation.result.columns["altitude_m"][-1] == pytest.approx(
        configuration.reentry_altitude_m, abs=1.0e-8
    )
