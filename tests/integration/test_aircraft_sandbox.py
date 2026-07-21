from dataclasses import replace

import numpy as np

from aerognc.configuration import load_aircraft_configuration
from aerognc.simulation.aircraft_sandbox import simulate_aircraft
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    FixedWingFlightModel,
    longitudinal_trim_command,
)
from aerognc.visualisation.aircraft_live import research_ascent_assist_command


def _short_configuration():  # type: ignore[no-untyped-def]
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    return replace(configuration, duration_s=8.0, output_step_s=0.1)


def test_trimmed_aircraft_case_remains_finite_and_airborne() -> None:
    simulation = simulate_aircraft(_short_configuration())
    columns = simulation.result.columns

    assert not simulation.impacted_ground
    assert np.all(np.isfinite(columns["altitude_m"]))
    assert np.max(np.abs(columns["roll_deg"])) < 1.0
    assert np.max(np.abs(columns["true_airspeed_mps"] - 82.0)) < 2.0
    assert simulation.stalled_duration_s == 0.0


def test_short_aileron_pulse_changes_bank_heading_and_path() -> None:
    configuration = replace(_short_configuration(), duration_s=12.0)
    trim = longitudinal_trim_command(configuration)

    def command(time_s, _state):  # type: ignore[no-untyped-def]
        return AircraftControlCommand(
            roll=0.35 if time_s < 1.0 else 0.0,
            pitch=trim.pitch,
            throttle=trim.throttle,
        )

    simulation = simulate_aircraft(configuration, command)
    columns = simulation.result.columns

    assert np.max(columns["roll_deg"]) > 20.0
    assert np.max(np.abs(columns["turn_rate_degps"])) > 1.0
    assert abs(columns["north_m"][-1]) > 5.0


def test_aerodynamic_coefficient_change_alters_propagated_path() -> None:
    configuration = _short_configuration()
    baseline = simulate_aircraft(configuration)
    reduced_lift_configuration = replace(
        configuration,
        aerodynamics=replace(
            configuration.aerodynamics,
            cl_alpha_per_rad=0.7 * configuration.aerodynamics.cl_alpha_per_rad,
        ),
    )
    altered = simulate_aircraft(reduced_lift_configuration)

    difference = np.linalg.norm(altered.final_state[:3] - baseline.final_state[:3])
    assert difference > 5.0


def test_fictional_research_ascent_assist_can_cross_100_km_boundary() -> None:
    configuration = replace(
        _short_configuration(),
        duration_s=125.0,
        integration_step_s=0.05,
        output_step_s=0.1,
    )
    model = FixedWingFlightModel(configuration)

    def command(time_s, state):  # type: ignore[no-untyped-def]
        return research_ascent_assist_command(model, state, time_s)

    simulation = simulate_aircraft(configuration, command)

    assert simulation.reached_space
    assert np.max(simulation.result.columns["altitude_m"]) > 100_000.0
    assert "karman_line_ascent" in {event["name"] for event in simulation.result.event_summary}
