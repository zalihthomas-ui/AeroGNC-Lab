import numpy as np
import pytest

from aerognc.configuration import load_aircraft_configuration
from aerognc.simulation.aircraft_telemetry import aircraft_telemetry
from aerognc.vehicle.fixed_wing import (
    AircraftControlCommand,
    FixedWingFlightModel,
    aircraft_initial_state,
    longitudinal_trim_command,
)
from aerognc.visualisation.aircraft_experience import (
    FlightRecorder,
    FlightTrailBuffer,
    OperatingEnvelopeLimits,
    RealtimeSimulationClock,
    TrailSettings,
    classify_touchdown,
    evaluate_flight_warnings,
    interpolate_ground_contact,
    load_recorded_flight,
)
from aerognc.visualisation.aircraft_live import advance_live_aircraft


class FakeClock:
    def __init__(self) -> None:
        self.time_s = 0.0

    def __call__(self) -> float:
        return self.time_s


def test_realtime_clock_uses_fixed_steps_and_is_render_rate_invariant() -> None:
    def accumulated(callback_step_s: float, count: int) -> float:
        source = FakeClock()
        clock = RealtimeSimulationClock(0.02, time_source=source)
        assert clock.tick().physics_step_count == 0
        total = 0.0
        for _ in range(count):
            source.time_s += callback_step_s
            tick = clock.tick()
            assert tick.simulation_duration_s == pytest.approx(tick.physics_step_count * 0.02)
            total += tick.simulation_duration_s
        return total

    assert accumulated(0.1, 10) == pytest.approx(1.0)
    assert accumulated(0.05, 20) == pytest.approx(1.0)


def test_realtime_clock_bounds_catchup_and_discards_paused_time() -> None:
    source = FakeClock()
    clock = RealtimeSimulationClock(0.02, maximum_catch_up_s=0.5, time_source=source)
    clock.tick()
    source.time_s = 2.0
    catchup = clock.tick()
    assert catchup.simulation_duration_s == pytest.approx(0.5)
    assert catchup.dropped_simulation_s == pytest.approx(1.5)
    source.time_s = 12.0
    assert clock.tick(paused=True).simulation_duration_s == 0.0
    source.time_s = 12.02
    assert clock.tick().simulation_duration_s == pytest.approx(0.02)


def test_trail_modes_decimation_clear_and_capacity() -> None:
    settings = TrailSettings(
        mode="fading", fading_duration_s=0.25, maximum_points=4, minimum_sample_interval_s=0.1
    )
    trail = FlightTrailBuffer(settings)
    assert trail.append(0.0, [0.0, 0.0, 0.0], altitude_m=0.0, airspeed_mps=10.0)
    assert not trail.append(0.05, [1.0, 0.0, 0.0], altitude_m=1.0, airspeed_mps=11.0)
    for index in range(1, 7):
        trail.append(
            0.1 * index,
            [float(index), 0.0, 0.0],
            altitude_m=float(index),
            airspeed_mps=10.0 + index,
        )
    assert len(trail) == 4
    assert trail.view(0.6, mode="full").positions_display_m.shape == (4, 3)
    fading = trail.view(0.6)
    assert fading.positions_display_m.shape[0] == 3
    assert np.all(np.diff(fading.alpha) >= 0.0)
    assert trail.view(0.6, mode="off").positions_display_m.shape == (0, 3)
    trail.clear()
    assert len(trail) == 1


def test_ground_contact_interpolation_ends_exactly_on_surface() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    before = aircraft_initial_state(configuration)
    unit_radius = before[:3] / np.linalg.norm(before[:3])
    before[:3] = unit_radius * (configuration.planet.radius_m + 10.0)
    after = before.copy()
    after[:3] = unit_radius * (configuration.planet.radius_m - 10.0)
    time_s, state = interpolate_ground_contact(2.0, before, 2.2, after, configuration)
    assert time_s == pytest.approx(2.1)
    assert np.linalg.norm(state[:3]) == pytest.approx(configuration.planet.radius_m)
    assert np.linalg.norm(state[6:10]) == pytest.approx(1.0)


def test_recorder_round_trip_replays_recorded_state_and_debriefs(tmp_path) -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    command = longitudinal_trim_command(configuration)
    initial = aircraft_initial_state(configuration)
    following = advance_live_aircraft(model, initial, 0.0, 0.02, command)
    recorder = FlightRecorder(maximum_samples=10)
    recorder.append(
        initial,
        command,
        aircraft_telemetry(model, 0.0, initial, command, initial_position_inertial_m=initial[:3]),
    )
    recorder.append(
        following,
        command,
        aircraft_telemetry(
            model, 0.02, following, command, initial_position_inertial_m=initial[:3]
        ),
    )
    recorder.add_event("test", 0.01, "deterministic unit event")
    artifacts = recorder.write(tmp_path, termination_reason="unit test")
    replay = load_recorded_flight(artifacts.csv_path)
    replay_state, replay_command = replay.sample(0.02)
    np.testing.assert_allclose(replay_state, following, rtol=1.0e-14, atol=1.0e-12)
    assert replay_command == command
    summary = recorder.summary("unit test")
    assert summary["sample_count"] == 2
    assert summary["termination_reason"] == "unit test"
    assert artifacts.summary_path.is_file()


def test_warning_and_touchdown_services_use_named_telemetry() -> None:
    configuration = load_aircraft_configuration("configs/aircraft_sandbox.yaml")
    model = FixedWingFlightModel(configuration)
    state = aircraft_initial_state(configuration)
    command = AircraftControlCommand(throttle=configuration.initial_throttle, rocket_assist=True)
    telemetry = aircraft_telemetry(
        model, 0.0, state, command, initial_position_inertial_m=state[:3]
    )
    warning_codes = {
        warning.code
        for warning in evaluate_flight_warnings(
            telemetry,
            command,
            OperatingEnvelopeLimits(maximum_mach=0.01),
            numerical_lag=True,
        )
    }
    assert {"overspeed", "rocket", "lag"}.issubset(warning_codes)
    assessment = classify_touchdown(telemetry, runway_cross_track_m=100.0)
    assert assessment.classification == "unsafe_attitude"
    assert "ground roll" in assessment.note
