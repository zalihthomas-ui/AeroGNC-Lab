from dataclasses import replace

from aerognc.configuration import load_aircraft_configuration
from aerognc.simulation.aircraft_telemetry import aircraft_telemetry
from aerognc.simulation.aircraft_training import (
    AIRCRAFT_PRESETS,
    aircraft_preflight,
    apply_aircraft_preset,
    evaluate_training_task,
    scripted_demo_command,
)
from aerognc.vehicle.fixed_wing import (
    AircraftState,
    FixedWingFlightModel,
    aircraft_initial_state,
    longitudinal_trim_command,
)


def test_aircraft_presets_and_preflight_are_explicit_and_plausible() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    assert len(AIRCRAFT_PRESETS) == 5
    crosswind = apply_aircraft_preset(configuration, "crosswind_response")
    research = apply_aircraft_preset(configuration, "high_altitude_research")
    assert crosswind.gust_amplitude_ned_mps[1] == 8.0
    assert crosswind.turbulence_std_ned_mps[1] > 0.0
    assert research.initial.altitude_m == 13_000.0
    preflight = aircraft_preflight(configuration)
    assert preflight.wing_loading_kgpm2 > 100.0
    assert preflight.stall_speed_1g_mps > 0.0
    assert preflight.estimated_fuel_endurance_s > 1_000.0
    assert "Synthetic" in preflight.warning


def test_scripted_demo_uses_normal_bounded_command_interface() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    state = AircraftState.from_array(aircraft_initial_state(configuration))
    before = scripted_demo_command(2.0, state, configuration)
    turn = scripted_demo_command(7.0, state, configuration)
    climb = scripted_demo_command(22.0, state, configuration)
    assert before == longitudinal_trim_command(configuration)
    assert turn.roll > 0.0 and turn.yaw > 0.0
    assert climb.pitch > before.pitch and climb.throttle > before.throttle


def test_civilian_training_scores_use_recorded_telemetry() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    state = aircraft_initial_state(configuration)
    command = longitudinal_trim_command(configuration)
    sample = aircraft_telemetry(
        model, 0.0, state, command, initial_position_inertial_m=state[:3]
    )
    steady = [
        sample,
        replace(sample, time_s=1.0, altitude_m=sample.altitude_m + 5.0),
        replace(sample, time_s=2.0, altitude_m=sample.altitude_m - 5.0),
    ]
    hold = evaluate_training_task("altitude_speed_hold", steady, [command] * 3)
    assert hold.passed
    assert hold.score_percent > 90.0

    research = evaluate_training_task(
        "research_altitude_crossing",
        [sample, replace(sample, time_s=10.0, altitude_m=100_100.0)],
        [command, command],
    )
    assert research.passed
    assert "does not demonstrate orbit" in research.interpretation

    stall_samples = [
        sample,
        replace(sample, time_s=1.0, stall_fraction=0.5),
        replace(sample, time_s=3.0, stall_fraction=0.0, altitude_m=sample.altitude_m - 50.0),
    ]
    recovery = evaluate_training_task("stall_recovery", stall_samples, [command] * 3)
    assert recovery.passed
